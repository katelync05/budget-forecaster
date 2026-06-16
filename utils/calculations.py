"""
calculations.py – Core financial metrics and KPI computations.

All functions accept a cleaned, categorised transaction DataFrame
and return plain Python / Pandas values ready for display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Basic KPIs ────────────────────────────────────────────────────────────────

def current_balance(df: pd.DataFrame) -> float:
    """Sum of all transaction amounts (net position)."""
    return float(df["Amount"].sum())


def monthly_income(df: pd.DataFrame, year: int | None = None, month: int | None = None) -> float:
    """Total income for a given month (or the latest month if not specified)."""
    mask = df["Amount"] > 0
    if year and month:
        mask &= (df["Year"] == year) & (df["Month"] == month)
    else:
        latest = df["YearMonth"].max()
        mask &= df["YearMonth"] == latest
    return float(df.loc[mask, "Amount"].sum())


def monthly_expenses(df: pd.DataFrame, year: int | None = None, month: int | None = None) -> float:
    """Total expenses (positive value) for a given month."""
    mask = df["Amount"] < 0
    if year and month:
        mask &= (df["Year"] == year) & (df["Month"] == month)
    else:
        latest = df["YearMonth"].max()
        mask &= df["YearMonth"] == latest
    return float(df.loc[mask, "Amount"].abs().sum())


def monthly_savings(df: pd.DataFrame, year: int | None = None, month: int | None = None) -> float:
    """Net savings = income − expenses for a given month."""
    return monthly_income(df, year, month) - monthly_expenses(df, year, month)


def savings_rate(df: pd.DataFrame, year: int | None = None, month: int | None = None) -> float:
    """Savings rate as a percentage of income."""
    income = monthly_income(df, year, month)
    if income == 0:
        return 0.0
    return monthly_savings(df, year, month) / income * 100


# ── Trend Analytics ───────────────────────────────────────────────────────────

def monthly_expense_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a time-series of monthly expenses (abs value)."""
    expenses = df[df["Amount"] < 0].copy()
    monthly = (
        expenses.groupby("YearMonth")["Amount"]
        .sum()
        .abs()
        .reset_index()
        .rename(columns={"Amount": "Expenses"})
    )
    monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()
    return monthly.sort_values("YearMonth")


def monthly_income_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a time-series of monthly income."""
    income = df[df["Amount"] > 0].copy()
    monthly = (
        income.groupby("YearMonth")["Amount"]
        .sum()
        .reset_index()
        .rename(columns={"Amount": "Income"})
    )
    monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()
    return monthly.sort_values("YearMonth")


def rolling_avg_expenses(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Monthly expenses with a rolling mean column."""
    series = monthly_expense_series(df)
    series[f"Rolling_{window}M"] = series["Expenses"].rolling(window, min_periods=1).mean()
    return series


def spending_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """Average daily spending grouped by day of the week."""
    expenses = df[df["Amount"] < 0].copy()
    expenses["AbsAmount"] = expenses["Amount"].abs()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    result = (
        expenses.groupby("DayOfWeek")["AbsAmount"]
        .mean()
        .reindex(order)
        .reset_index()
        .rename(columns={"DayOfWeek": "Day", "AbsAmount": "AvgSpend"})
    )
    return result


def spending_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot of total spending: rows = Month name, columns = Day of week.
    Suitable for a Plotly/seaborn heatmap.
    """
    expenses = df[df["Amount"] < 0].copy()
    expenses["AbsAmount"] = expenses["Amount"].abs()
    expenses["MonthName"] = expenses["Date"].dt.strftime("%b")
    expenses["MonthNum"]  = expenses["Date"].dt.month
    order_dow   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    expenses["DayAbbr"] = expenses["Date"].dt.strftime("%a")

    pivot = (
        expenses.groupby(["MonthName", "MonthNum", "DayAbbr"])["AbsAmount"]
        .sum()
        .reset_index()
        .pivot_table(index=["MonthName", "MonthNum"], columns="DayAbbr", values="AbsAmount", aggfunc="sum")
        .reindex(columns=order_dow)
    )
    pivot = pivot.sort_values("MonthNum").drop(columns="MonthNum", errors="ignore")
    pivot.index = pivot.index.droplevel(1)
    return pivot.fillna(0)


# ── Advanced Metrics ──────────────────────────────────────────────────────────

def expense_volatility(df: pd.DataFrame) -> float:
    """Standard deviation of monthly total expenses."""
    return float(monthly_expense_series(df)["Expenses"].std())


def avg_transaction_value(df: pd.DataFrame) -> float:
    """Mean absolute value of expense transactions."""
    return float(df.loc[df["Amount"] < 0, "Amount"].abs().mean())


def largest_merchants(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-n merchants by total absolute spending."""
    expenses = df[df["Amount"] < 0].copy()
    expenses["AbsAmount"] = expenses["Amount"].abs()
    return (
        expenses.groupby("Merchant")["AbsAmount"]
        .agg(Total="sum", Transactions="count")
        .sort_values("Total", ascending=False)
        .head(n)
        .reset_index()
    )


def recurring_expenses(df: pd.DataFrame, min_months: int = 3) -> pd.DataFrame:
    """
    Detect likely recurring charges – merchants that appear in ≥ min_months.
    """
    expenses = df[df["Amount"] < 0].copy()
    expenses["AbsAmount"] = expenses["Amount"].abs()
    merchant_months = (
        expenses.groupby(["Merchant", "YearMonth"])["AbsAmount"]
        .sum()
        .reset_index()
    )
    counts = merchant_months.groupby("Merchant").agg(
        MonthCount=("YearMonth", "count"),
        AvgAmount=("AbsAmount", "mean"),
    )
    recurring = counts[counts["MonthCount"] >= min_months].sort_values(
        "MonthCount", ascending=False
    )
    return recurring.reset_index()


def financial_health_score(df: pd.DataFrame, emergency_fund_months: float = 0.0) -> dict:
    """
    Compute a 0–100 financial health score with component breakdown.

    Components
    ----------
    savings_rate_score (0–30) : Based on % of income saved.
    emergency_fund_score (0–25): Months of expenses covered.
    expense_consistency_score (0–20): Low CoV → stable spending.
    cash_flow_score (0–25): Positive net cash flow months.

    Returns
    -------
    dict with 'score', 'grade', 'components', and 'recommendations'.
    """
    exp_series = monthly_expense_series(df)
    inc_series = monthly_income_series(df)

    # Savings rate component
    total_income  = inc_series["Income"].sum()
    total_expense = exp_series["Expenses"].sum()
    overall_sr = max(0.0, (total_income - total_expense) / total_income * 100) if total_income else 0
    sr_score = min(30, overall_sr / 20 * 30)  # 20% savings → full marks

    # Emergency fund component
    monthly_exp_avg = exp_series["Expenses"].mean() if len(exp_series) else 1
    ef_months = emergency_fund_months
    ef_score = min(25, ef_months / 6 * 25)  # 6 months → full marks

    # Expense consistency (low CV → good)
    cov = exp_series["Expenses"].std() / exp_series["Expenses"].mean() if len(exp_series) > 1 else 1
    consistency_score = max(0, 20 - cov * 20)

    # Cash-flow positivity
    merged = pd.merge(inc_series, exp_series, on="YearMonth", how="outer").fillna(0)
    positive_months = (merged["Income"] > merged["Expenses"]).sum()
    total_months = len(merged)
    cf_score = (positive_months / total_months * 25) if total_months else 0

    total = sr_score + ef_score + consistency_score + cf_score

    grade_map = [(90, "Excellent"), (75, "Good"), (60, "Fair"), (0, "Needs Improvement")]
    grade = next(g for threshold, g in grade_map if total >= threshold)

    # Recommendations
    recs: list[str] = []
    if sr_score < 15:
        recs.append(f"Increase savings rate; currently at {overall_sr:.1f}% (target ≥ 20%).")
    if ef_score < 12:
        recs.append(f"Build emergency fund toward 6 months of expenses (~${monthly_exp_avg*6:,.0f}).")
    if consistency_score < 10:
        recs.append("Your spending is volatile month-to-month — consider a monthly budget cap.")
    if cf_score < 15:
        recs.append("Some months show negative cash flow; review discretionary spending.")

    return {
        "score": round(total, 1),
        "grade": grade,
        "components": {
            "Savings Rate":        round(sr_score, 1),
            "Emergency Fund":      round(ef_score, 1),
            "Expense Consistency": round(consistency_score, 1),
            "Cash Flow":           round(cf_score, 1),
        },
        "recommendations": recs or ["Great job — keep up the good financial habits!"],
    }
