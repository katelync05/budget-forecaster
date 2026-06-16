"""
scenario.py – Scenario analysis and Monte Carlo cash-flow simulation.

Provides:
  - ScenarioEngine : apply slider-driven income/expense adjustments.
  - monte_carlo    : 10,000-run simulation with distributional outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Scenario Engine ───────────────────────────────────────────────────────────

@dataclass
class Scenario:
    """User-defined parameter overrides for a what-if scenario."""
    income_change_pct: float  = 0.0     # e.g. +10  → 10% raise
    rent_change_pct: float    = 0.0     # e.g. +5   → 5% rent hike
    food_change_pct: float    = 0.0     # groceries + dining
    entertainment_change_pct: float = 0.0
    inflation_rate: float     = 0.0     # annual %, applied monthly
    unexpected_expense: float = 0.0     # one-off extra cost per month


class ScenarioEngine:
    """
    Apply scenario parameters to historical averages and produce
    a 12-month forward projection.

    Parameters
    ----------
    base_monthly_income : float
    category_averages   : dict[str, float]
        Mapping of category → average monthly spend (positive value).
    current_balance     : float
    """

    def __init__(
        self,
        base_monthly_income: float,
        category_averages: dict[str, float],
        current_balance: float = 0.0,
        emergency_fund: float  = 0.0,
    ) -> None:
        self.base_income        = base_monthly_income
        self.category_averages  = {k.lower(): v for k, v in category_averages.items()}
        self.current_balance    = current_balance
        self.emergency_fund     = emergency_fund

    # ── helpers ───────────────────────────────────────────────────────────────

    def _adjusted_income(self, s: Scenario) -> float:
        return self.base_income * (1 + s.income_change_pct / 100)

    def _adjusted_expenses(self, s: Scenario, month_idx: int = 0) -> float:
        monthly_inflation = (1 + s.inflation_rate / 100) ** (1 / 12)
        inflation_factor  = monthly_inflation ** month_idx

        total = 0.0
        for cat, base_amt in self.category_averages.items():
            adj = base_amt
            if "rent" in cat or "housing" in cat:
                adj *= (1 + s.rent_change_pct / 100)
            elif any(k in cat for k in ("groceries", "dining", "food")):
                adj *= (1 + s.food_change_pct / 100)
            elif "entertainment" in cat:
                adj *= (1 + s.entertainment_change_pct / 100)
            total += adj * inflation_factor

        total += s.unexpected_expense
        return total

    # ── Public API ─────────────────────────────────────────────────────────────

    def project(self, scenario: Scenario, horizon: int = 12) -> pd.DataFrame:
        """
        Return a month-by-month projection under the given scenario.

        Returns
        -------
        pd.DataFrame with columns:
            Month, Income, Expenses, Net, Balance, EmergencyFundMonths
        """
        rows = []
        balance = self.current_balance

        for m in range(horizon):
            income   = self._adjusted_income(scenario)
            expenses = self._adjusted_expenses(scenario, month_idx=m)
            net      = income - expenses
            balance += net
            ef_months = (
                self.emergency_fund / (expenses / 1) if expenses > 0 else float("inf")
            )
            rows.append({
                "Month":               m + 1,
                "Income":              round(income, 2),
                "Expenses":            round(expenses, 2),
                "Net":                 round(net, 2),
                "Balance":             round(balance, 2),
                "EmergencyFundMonths": round(ef_months, 1),
            })

        return pd.DataFrame(rows)

    def compare_scenarios(
        self, scenarios: dict[str, Scenario], horizon: int = 12
    ) -> dict[str, pd.DataFrame]:
        """Run multiple named scenarios and return a dict of projection tables."""
        return {name: self.project(s, horizon) for name, s in scenarios.items()}


# ── Monte Carlo Simulation ────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    """Output container for a Monte Carlo run."""
    n_simulations:         int
    horizon:               int
    prob_positive:         float    # P(ending balance > 0)
    prob_below_emergency:  float    # P(balance < emergency threshold)
    prob_goal_achieved:    float    # P(balance ≥ savings goal)
    final_balance_p10:     float
    final_balance_p50:     float
    final_balance_p90:     float
    balance_matrix:        np.ndarray   # shape (n_simulations, horizon)
    monthly_income_mean:   float
    monthly_expense_mean:  float


def monte_carlo(
    current_balance:       float,
    monthly_income_mean:   float,
    monthly_income_std:    float,
    monthly_expense_mean:  float,
    monthly_expense_std:   float,
    horizon:               int   = 12,
    n_simulations:         int   = 10_000,
    emergency_threshold:   float = 1_000.0,
    savings_goal:          float = 10_000.0,
    unexpected_expense_prob: float = 0.05,
    unexpected_expense_mean: float = 500.0,
    seed:                  Optional[int] = 42,
) -> MonteCarloResult:
    """
    Run a Monte Carlo cash-flow simulation.

    Each simulation randomly draws income and expenses from normal
    distributions and optionally adds a surprise expense event.

    Parameters
    ----------
    current_balance        : Starting cash balance.
    monthly_income_mean    : Average monthly income.
    monthly_income_std     : Standard deviation of monthly income.
    monthly_expense_mean   : Average monthly expenses.
    monthly_expense_std    : Standard deviation of monthly expenses.
    horizon                : Months to simulate.
    n_simulations          : Number of Monte Carlo paths.
    emergency_threshold    : Balance below this is flagged.
    savings_goal           : Target final balance.
    unexpected_expense_prob: Probability of a surprise expense each month.
    unexpected_expense_mean: Mean size of surprise expense.
    seed                   : Random seed for reproducibility.

    Returns
    -------
    MonteCarloResult
    """
    rng = np.random.default_rng(seed)

    # Draw income and expense matrices: shape (n_simulations, horizon)
    income_draws  = rng.normal(monthly_income_mean,  max(monthly_income_std,  1),
                               size=(n_simulations, horizon))
    expense_draws = rng.normal(monthly_expense_mean, max(monthly_expense_std, 1),
                               size=(n_simulations, horizon))

    # Clip negatives (income and expenses can't be negative)
    income_draws  = np.clip(income_draws,  0, None)
    expense_draws = np.clip(expense_draws, 0, None)

    # Surprise expense: Bernoulli event each month
    surprise_mask  = rng.random(size=(n_simulations, horizon)) < unexpected_expense_prob
    surprise_costs = rng.exponential(unexpected_expense_mean,  size=(n_simulations, horizon))
    expense_draws += surprise_mask * surprise_costs

    # Accumulate balances
    net          = income_draws - expense_draws                       # (n, horizon)
    balance_mat  = np.cumsum(net, axis=1) + current_balance           # (n, horizon)

    final_balances = balance_mat[:, -1]

    return MonteCarloResult(
        n_simulations         = n_simulations,
        horizon               = horizon,
        prob_positive         = float((final_balances > 0).mean() * 100),
        prob_below_emergency  = float((balance_mat.min(axis=1) < emergency_threshold).mean() * 100),
        prob_goal_achieved    = float((final_balances >= savings_goal).mean() * 100),
        final_balance_p10     = float(np.percentile(final_balances, 10)),
        final_balance_p50     = float(np.percentile(final_balances, 50)),
        final_balance_p90     = float(np.percentile(final_balances, 90)),
        balance_matrix        = balance_mat,
        monthly_income_mean   = monthly_income_mean,
        monthly_expense_mean  = monthly_expense_mean,
    )
