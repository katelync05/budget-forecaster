"""
charts.py – Plotly figure factory for Budget Forecaster.

Each function returns a go.Figure / px Figure ready to pass to
st.plotly_chart().  All figures use the shared THEME for consistent
finance-inspired styling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.forecast import ForecastResult
from models.scenario import MonteCarloResult

# ── Shared Theme ──────────────────────────────────────────────────────────────

NAVY   = "#0A2342"
TEAL   = "#1B6CA8"
GREEN  = "#27AE60"
RED    = "#E74C3C"
GOLD   = "#F39C12"
LGRAY  = "#F4F6F8"
DGRAY  = "#7F8C8D"

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, sans-serif", size=13, color="#2C3E50"),
    paper_bgcolor="white",
    plot_bgcolor=LGRAY,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=NAVY)), **LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#DDE1E7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#DDE1E7", zeroline=False)
    return fig


# ── Income / Expense Trends ───────────────────────────────────────────────────

def monthly_cashflow_chart(monthly: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: Income vs Expenses per month, with Net line overlay.
    `monthly` must have columns: YearMonth, Income, Expenses, Net.
    """
    fig = go.Figure()
    fig.add_bar(x=monthly["YearMonth"], y=monthly["Income"],
                name="Income", marker_color=GREEN, opacity=0.85)
    fig.add_bar(x=monthly["YearMonth"], y=monthly["Expenses"],
                name="Expenses", marker_color=RED, opacity=0.85)
    fig.add_scatter(x=monthly["YearMonth"], y=monthly["Net"],
                    mode="lines+markers", name="Net",
                    line=dict(color=NAVY, width=2.5),
                    marker=dict(size=6))
    fig.update_layout(barmode="group")
    return _apply_theme(fig, "Monthly Cash Flow – Income vs Expenses")


def spending_trend_chart(exp_series: pd.DataFrame, rolling_col: str | None = None) -> go.Figure:
    """Line chart of monthly expenses with optional rolling average."""
    fig = go.Figure()
    fig.add_scatter(x=exp_series["YearMonth"], y=exp_series["Expenses"],
                    mode="lines+markers", name="Monthly Expenses",
                    line=dict(color=RED, width=2), marker=dict(size=5))
    if rolling_col and rolling_col in exp_series.columns:
        fig.add_scatter(x=exp_series["YearMonth"], y=exp_series[rolling_col],
                        mode="lines", name="Rolling Avg",
                        line=dict(color=TEAL, width=2.5, dash="dash"))
    return _apply_theme(fig, "Monthly Expense Trend")


def savings_trend_chart(monthly: pd.DataFrame) -> go.Figure:
    """Area chart of cumulative savings over time."""
    monthly = monthly.copy()
    monthly["CumSavings"] = monthly["Net"].cumsum()
    fig = go.Figure()
    fig.add_scatter(x=monthly["YearMonth"], y=monthly["CumSavings"],
                    fill="tozeroy",
                    line=dict(color=GREEN, width=2.5),
                    fillcolor="rgba(39,174,96,0.15)",
                    name="Cumulative Savings")
    return _apply_theme(fig, "Cumulative Savings Over Time")


# ── Category Charts ───────────────────────────────────────────────────────────

def category_pie_chart(category_summary: pd.DataFrame) -> go.Figure:
    """Donut chart of expense categories."""
    fig = go.Figure(go.Pie(
        labels=category_summary["Category"],
        values=category_summary["Total"],
        hole=0.45,
        textposition="inside",
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Bold),
    ))
    return _apply_theme(fig, "Expense Breakdown by Category")


def category_bar_chart(category_summary: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top spending categories."""
    df = category_summary.head(top_n).sort_values("Total")
    fig = go.Figure(go.Bar(
        x=df["Total"], y=df["Category"],
        orientation="h",
        marker_color=TEAL,
        text=df["Total"].apply(lambda x: f"${x:,.0f}"),
        textposition="outside",
    ))
    return _apply_theme(fig, f"Top {top_n} Spending Categories")


# ── Forecasting Charts ────────────────────────────────────────────────────────

def forecast_chart(result: ForecastResult, historical: pd.DataFrame | None = None) -> go.Figure:
    """
    Plot historical expenses + forecast with confidence band.
    `historical` should have YearMonth, Expenses columns.
    """
    fig = go.Figure()

    if historical is not None and not historical.empty:
        fig.add_scatter(
            x=historical["YearMonth"], y=historical["Expenses"],
            mode="lines+markers", name="Historical",
            line=dict(color=NAVY, width=2), marker=dict(size=5),
        )

    fc = result.forecast
    fig.add_scatter(
        x=fc["Date"], y=fc["Predicted"],
        mode="lines+markers", name=f"Forecast ({result.model_name})",
        line=dict(color=TEAL, width=2.5, dash="dash"), marker=dict(size=6),
    )
    fig.add_scatter(
        x=pd.concat([fc["Date"], fc["Date"][::-1]]),
        y=pd.concat([fc["Upper"], fc["Lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(27,108,168,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
    )
    return _apply_theme(fig, f"{result.model_name} – 12-Month Expense Forecast")


def cash_flow_projection_chart(cf_df: pd.DataFrame) -> go.Figure:
    """Waterfall-style chart of projected ending balance."""
    fig = go.Figure()
    colors = [GREEN if v >= 0 else RED for v in cf_df["Ending Balance"]]
    fig.add_bar(
        x=cf_df["Month"], y=cf_df["Ending Balance"],
        marker_color=colors, name="Projected Balance",
        text=cf_df["Ending Balance"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
    )
    return _apply_theme(fig, "Projected Cash Balance")


# ── Heatmap ───────────────────────────────────────────────────────────────────

def spending_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """Heatmap of spending by month (rows) × weekday (columns)."""
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Blues",
        text=pivot.values.round(0).astype(int),
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="$"),
    ))
    return _apply_theme(fig, "Spending Heatmap – Month × Weekday")


# ── Monte Carlo Chart ─────────────────────────────────────────────────────────

def monte_carlo_histogram(mc: MonteCarloResult) -> go.Figure:
    """Histogram of final-balance distribution across all simulations."""
    final = mc.balance_matrix[:, -1]
    fig = go.Figure()
    fig.add_histogram(
        x=final, nbinsx=80,
        marker_color=TEAL, opacity=0.75, name="Simulated Balance",
    )
    for pct, label, color in [
        (mc.final_balance_p10, "P10", RED),
        (mc.final_balance_p50, "P50 (Median)", NAVY),
        (mc.final_balance_p90, "P90", GREEN),
    ]:
        fig.add_vline(x=pct, line_color=color, line_width=2,
                      annotation_text=label, annotation_position="top")
    return _apply_theme(fig, "Monte Carlo – Final Balance Distribution")


def monte_carlo_fan_chart(mc: MonteCarloResult, n_paths: int = 200) -> go.Figure:
    """Fan chart showing random simulation paths + percentile bands."""
    months = list(range(1, mc.horizon + 1))
    fig = go.Figure()

    # Thin sample paths
    idx = np.random.choice(mc.n_simulations, min(n_paths, mc.n_simulations), replace=False)
    for i in idx:
        fig.add_scatter(
            x=months, y=mc.balance_matrix[i],
            mode="lines", line=dict(color=TEAL, width=0.5),
            opacity=0.08, showlegend=False,
        )

    # Percentile bands
    for pct, name, color in [
        (10, "P10", RED), (50, "Median", NAVY), (90, "P90", GREEN)
    ]:
        fig.add_scatter(
            x=months, y=np.percentile(mc.balance_matrix, pct, axis=0),
            mode="lines", name=name,
            line=dict(color=color, width=2.5),
        )
    return _apply_theme(fig, f"Monte Carlo Fan Chart ({mc.n_simulations:,} Simulations)")


# ── Health Score Gauge ────────────────────────────────────────────────────────

def health_score_gauge(score: float, grade: str) -> go.Figure:
    """Gauge chart for the financial health score."""
    color = GREEN if score >= 75 else (GOLD if score >= 60 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title=dict(text=f"Financial Health – <b>{grade}</b>", font=dict(size=16, color=NAVY)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=color),
            steps=[
                dict(range=[0, 60],  color="#FADBD8"),
                dict(range=[60, 75], color="#FDEBD0"),
                dict(range=[75, 90], color="#D5F5E3"),
                dict(range=[90, 100],color="#A9DFBF"),
            ],
            threshold=dict(line=dict(color=NAVY, width=4), value=score),
        ),
    ))
    fig.update_layout(height=280, paper_bgcolor="white",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig
