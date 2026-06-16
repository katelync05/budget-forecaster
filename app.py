"""
app.py – Budget Forecaster | Main Streamlit Dashboard

Run with:  streamlit run app.py
"""

from __future__ import annotations

# ── Path bootstrap – must come before ANY local imports ──────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from utils.loader       import load_transactions, load_categories, get_monthly_summary
from utils.calculations import (
    current_balance, monthly_income, monthly_expenses, monthly_savings,
    savings_rate, monthly_expense_series, monthly_income_series,
    rolling_avg_expenses, spending_by_weekday, spending_heatmap_data,
    expense_volatility, avg_transaction_value, largest_merchants,
    recurring_expenses, financial_health_score,
)
from utils.charts import (
    monthly_cashflow_chart, spending_trend_chart, savings_trend_chart,
    category_pie_chart, category_bar_chart, forecast_chart,
    cash_flow_projection_chart, spending_heatmap, monte_carlo_histogram,
    monte_carlo_fan_chart, health_score_gauge,
)
from models.categorizer import TransactionCategorizer
from models.forecast    import run_forecast, cash_flow_projection
from models.scenario    import Scenario, ScenarioEngine, monte_carlo

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Budget Forecaster",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.kpi-card {
    background: #ffffff;
    border: 1px solid #e0e6ef;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(10,35,66,.06);
    transition: transform .15s ease;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: #7f8c8d;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0A2342;
    line-height: 1.1;
}
.kpi-value.positive { color: #27AE60; }
.kpi-value.negative { color: #E74C3C; }

.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0A2342;
    border-left: 4px solid #1B6CA8;
    padding-left: 10px;
    margin: 28px 0 14px;
}

section[data-testid="stSidebar"] { background: #0A2342; }
section[data-testid="stSidebar"] * { color: #ECF0F1 !important; }
section[data-testid="stSidebar"] hr { border-color: #1B6CA8; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, positive=None) -> str:
    cls = "positive" if positive is True else ("negative" if positive is False else "")
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {cls}">{value}</div>
    </div>"""


def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


@st.cache_data(show_spinner=False)
def _load_data(file_bytes, use_sample: bool) -> pd.DataFrame:
    if use_sample or file_bytes is None:
        path = Path(__file__).parent / "data" / "sample_transactions.csv"
        return load_transactions(path)
    return load_transactions(pd.read_csv(io.BytesIO(file_bytes), dtype=str))


@st.cache_data(show_spinner=False)
def _load_cats() -> dict:
    return load_categories()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 💰 Budget Forecaster")
    st.markdown("*Personal FP&A Dashboard*")
    st.markdown("---")

    st.markdown("### 📂 Data Source")
    use_sample = st.checkbox("Use sample data", value=True)
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader(
            "Upload CSV (Date, Merchant, Amount)",
            type=["csv"],
            help="Columns: Date, Merchant, Amount, Account (opt), Type (opt)",
        )

    st.markdown("---")
    st.markdown("### 📈 Forecast Settings")
    model_choice = st.selectbox(
        "Forecast Model",
        options=["exponential_smoothing", "arima", "linear_regression", "moving_average"],
        format_func=lambda x: {
            "exponential_smoothing": "Exponential Smoothing",
            "arima":                 "ARIMA",
            "linear_regression":     "Linear Regression",
            "moving_average":        "Moving Average",
        }[x],
    )
    horizon = st.slider("Forecast Horizon (months)", 3, 24, 12)

    st.markdown("---")
    st.markdown("### 🏦 Assumptions")
    manual_balance  = st.number_input("Current Balance ($)",          value=8_500.0, step=100.0)
    emergency_fund  = st.number_input("Emergency Fund ($)",           value=3_000.0, step=100.0)
    expected_income = st.number_input("Expected Monthly Income ($)",  value=3_200.0, step=50.0)

    st.markdown("---")
    st.markdown("### 🎲 Monte Carlo")
    mc_simulations   = st.select_slider("Simulations", [1000, 5000, 10000], value=10000)
    mc_savings_goal  = st.number_input("Savings Goal ($)",            value=15_000.0, step=500.0)
    mc_emergency_thr = st.number_input("Emergency Threshold ($)",     value=1_000.0,  step=100.0)

    st.markdown("---")
    st.markdown("### ⚙️ Scenario Analysis")
    sc_income    = st.slider("Income Change (%)",           -30,  50,   0)
    sc_rent      = st.slider("Rent / Housing Change (%)",   -20,  50,   0)
    sc_food      = st.slider("Food Budget Change (%)",       -30,  30,   0)
    sc_entertain = st.slider("Entertainment Change (%)",     -50,  50,   0)
    sc_inflation = st.slider("Annual Inflation Rate (%)",    0.0, 10.0, 3.0, step=0.5)
    sc_unexp     = st.slider("Unexpected Monthly Expense ($)", 0, 1000,  0,  step=50)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CATEGORISATION
# ═══════════════════════════════════════════════════════════════════════════════

file_bytes = uploaded.read() if uploaded else None

with st.spinner("Loading transactions…"):
    df_raw = _load_data(file_bytes, use_sample)
    cats   = _load_cats()

categorizer = TransactionCategorizer(cats)
df = categorizer.categorise_dataframe(df_raw)

monthly_df = get_monthly_summary(df)
exp_series = monthly_expense_series(df)
inc_series = monthly_income_series(df)
roll_exp   = rolling_avg_expenses(df, window=3)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 💰 Budget Forecaster")
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.caption(
        f"Data range: **{df['Date'].min().strftime('%b %d, %Y')}** "
        f"→ **{df['Date'].max().strftime('%b %d, %Y')}** "
        f"· {len(df):,} transactions"
    )
with col_h2:
    st.download_button(
        "⬇ Export CSV",
        data=df.to_csv(index=False),
        file_name="budget_forecaster_export.csv",
        mime="text/csv",
    )

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════

section("📊 Key Performance Indicators")

latest_ym = df["YearMonth"].max()
ly, lm    = latest_ym.year, latest_ym.month

bal    = manual_balance
m_inc  = monthly_income(df, ly, lm)
m_exp  = monthly_expenses(df, ly, lm)
m_sav  = monthly_savings(df, ly, lm)
s_rate = savings_rate(df, ly, lm)
net_cf = m_inc - m_exp

k1, k2, k3, k4, k5, k6 = st.columns(6)
for col, label, value, pos in [
    (k1, "Current Balance",   fmt_dollar(bal),    None),
    (k2, "Monthly Income",    fmt_dollar(m_inc),   True),
    (k3, "Monthly Expenses",  fmt_dollar(m_exp),   False),
    (k4, "Monthly Savings",   fmt_dollar(m_sav),   m_sav >= 0),
    (k5, "Savings Rate",      fmt_pct(s_rate),     s_rate >= 20),
    (k6, "Net Cash Flow",     fmt_dollar(net_cf),  net_cf >= 0),
]:
    with col:
        st.markdown(kpi_card(label, value, pos), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CASH FLOW CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

section("📈 Cash Flow Overview")

tab_cf1, tab_cf2, tab_cf3 = st.tabs(["Income vs Expenses", "Expense Trend", "Savings Growth"])

with tab_cf1:
    st.plotly_chart(monthly_cashflow_chart(monthly_df), use_container_width=True)
with tab_cf2:
    st.plotly_chart(spending_trend_chart(roll_exp, rolling_col="Rolling_3M"), use_container_width=True)
with tab_cf3:
    st.plotly_chart(savings_trend_chart(monthly_df), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

section("🏷️ Expense Categories")

cat_summary = categorizer.get_category_summary(df)

col_pie, col_bar = st.columns(2)
with col_pie:
    st.plotly_chart(category_pie_chart(cat_summary), use_container_width=True)
with col_bar:
    st.plotly_chart(category_bar_chart(cat_summary), use_container_width=True)

with st.expander("📋 Category Details"):
    cat_display = cat_summary.copy()
    cat_display["Total"] = cat_display["Total"].apply(fmt_dollar)
    cat_display["Avg"]   = cat_display["Avg"].apply(fmt_dollar)
    cat_display["Pct"]   = cat_display["Pct"].apply(fmt_pct)
    st.dataframe(cat_display, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SPENDING PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

section("📅 Spending Patterns")

col_heat, col_dow = st.columns([3, 2])

with col_heat:
    pivot = spending_heatmap_data(df)
    st.plotly_chart(spending_heatmap(pivot), use_container_width=True)

with col_dow:
    dow = spending_by_weekday(df)
    fig_dow = px.bar(
        dow, x="Day", y="AvgSpend",
        color="AvgSpend", color_continuous_scale="Blues",
        labels={"AvgSpend": "Avg Daily Spend ($)"},
        title="Average Spending by Day of Week",
    )
    fig_dow.update_layout(showlegend=False, plot_bgcolor="#F4F6F8", paper_bgcolor="white")
    st.plotly_chart(fig_dow, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BUDGET ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

section("🔍 Budget Analytics")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Top Merchants**")
    top_merch = largest_merchants(df, 8)
    top_merch["Total"] = top_merch["Total"].apply(fmt_dollar)
    st.dataframe(top_merch, use_container_width=True, hide_index=True)

with col_b:
    st.markdown("**Recurring Expenses**")
    rec = recurring_expenses(df)
    if not rec.empty:
        rec["AvgAmount"] = rec["AvgAmount"].apply(fmt_dollar)
        st.dataframe(rec, use_container_width=True, hide_index=True)
    else:
        st.info("No recurring charges detected (need ≥ 3 months).")

with col_c:
    st.markdown("**Quick Stats**")
    stats = {
        "Avg Monthly Expenses":    fmt_dollar(exp_series["Expenses"].mean()),
        "Median Monthly Expenses": fmt_dollar(exp_series["Expenses"].median()),
        "Expense Volatility (σ)":  fmt_dollar(expense_volatility(df)),
        "Avg Transaction Value":   fmt_dollar(avg_transaction_value(df)),
        "Total Transactions":      f"{len(df):,}",
        "Months of Data":          str(len(exp_series)),
    }
    for k, v in stats.items():
        st.metric(k, v)


# ═══════════════════════════════════════════════════════════════════════════════
# FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

section("🔭 Expense Forecasting")

with st.spinner(f"Running {model_choice.replace('_', ' ').title()} forecast…"):
    fc_result = run_forecast(exp_series, model=model_choice, horizon=horizon)

col_fc1, col_fc2 = st.columns([3, 1])

with col_fc1:
    st.plotly_chart(forecast_chart(fc_result, historical=exp_series), use_container_width=True)

with col_fc2:
    st.markdown("**Model Accuracy**")
    for k, v in fc_result.metrics.items():
        st.metric(k, f"{v:,.1f}")
    st.markdown("**Forecast Summary**")
    fc_df = fc_result.forecast
    st.metric("Avg Predicted / Month", fmt_dollar(fc_df["Predicted"].mean()))
    st.metric("Total 12-Month Spend",  fmt_dollar(fc_df["Predicted"].sum()))


# ── Cash Flow Projection ──────────────────────────────────────────────────────

section("💵 Cash Flow Projection")

cf_proj = cash_flow_projection(bal, expected_income, fc_result)

col_cft, col_cfp = st.columns([2, 3])

with col_cft:
    cf_display = cf_proj.copy()
    for col in ["Beginning Balance", "Income", "Expenses", "Ending Balance"]:
        cf_display[col] = cf_display[col].apply(fmt_dollar)
    st.dataframe(cf_display, use_container_width=True, hide_index=True)

with col_cfp:
    st.plotly_chart(cash_flow_projection_chart(cf_proj), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

section("⚙️ Scenario Analysis")

cat_avg = (
    df[df["Amount"] < 0]
    .groupby("Category")["Amount"]
    .mean()
    .abs()
    .to_dict()
)

engine = ScenarioEngine(
    base_monthly_income=expected_income,
    category_averages=cat_avg,
    current_balance=bal,
    emergency_fund=emergency_fund,
)

user_scenario = Scenario(
    income_change_pct=sc_income,
    rent_change_pct=sc_rent,
    food_change_pct=sc_food,
    entertainment_change_pct=sc_entertain,
    inflation_rate=sc_inflation,
    unexpected_expense=sc_unexp,
)

projections = engine.compare_scenarios({
    "Baseline":                   Scenario(),
    "Your Scenario":              user_scenario,
    "Optimistic (+10% income)":   Scenario(income_change_pct=10, food_change_pct=-10),
    "Conservative (−15% income)": Scenario(income_change_pct=-15, rent_change_pct=5),
}, horizon=12)

_sc_colors = {
    "Baseline":                   "#0A2342",
    "Your Scenario":              "#1B6CA8",
    "Optimistic (+10% income)":   "#27AE60",
    "Conservative (−15% income)": "#E74C3C",
}
sc_fig = go.Figure()
for name, proj in projections.items():
    sc_fig.add_scatter(
        x=proj["Month"], y=proj["Balance"],
        mode="lines+markers", name=name,
        line=dict(width=2.5, color=_sc_colors.get(name, "#555")),
    )
sc_fig.update_layout(
    title="12-Month Balance Under Different Scenarios",
    paper_bgcolor="white", plot_bgcolor="#F4F6F8",
    font=dict(family="Inter", color="#2C3E50"),
    legend=dict(orientation="h", y=1.08),
)
st.plotly_chart(sc_fig, use_container_width=True)

with st.expander("📋 Your Scenario Detail"):
    sc_disp = projections["Your Scenario"].copy()
    for col in ["Income", "Expenses", "Net", "Balance"]:
        sc_disp[col] = sc_disp[col].apply(fmt_dollar)
    st.dataframe(sc_disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

section("🎲 Monte Carlo Simulation")

mc_income_std  = inc_series["Income"].std()  if len(inc_series) > 1  else expected_income * 0.1
mc_expense_std = exp_series["Expenses"].std() if len(exp_series) > 1 else exp_series["Expenses"].mean() * 0.15

with st.spinner(f"Running {mc_simulations:,} simulations…"):
    mc_result = monte_carlo(
        current_balance      = bal,
        monthly_income_mean  = expected_income,
        monthly_income_std   = mc_income_std,
        monthly_expense_mean = exp_series["Expenses"].mean(),
        monthly_expense_std  = mc_expense_std,
        horizon              = 12,
        n_simulations        = mc_simulations,
        emergency_threshold  = mc_emergency_thr,
        savings_goal         = mc_savings_goal,
    )

mc_k1, mc_k2, mc_k3 = st.columns(3)
with mc_k1:
    st.markdown(kpi_card("Prob. Balance Stays Positive",
                         fmt_pct(mc_result.prob_positive),
                         mc_result.prob_positive >= 80), unsafe_allow_html=True)
with mc_k2:
    st.markdown(kpi_card("Prob. Below Emergency Threshold",
                         fmt_pct(mc_result.prob_below_emergency),
                         mc_result.prob_below_emergency <= 20), unsafe_allow_html=True)
with mc_k3:
    st.markdown(kpi_card("Prob. Savings Goal Achieved",
                         fmt_pct(mc_result.prob_goal_achieved),
                         mc_result.prob_goal_achieved >= 60), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

mc_c1, mc_c2 = st.columns(2)
with mc_c1:
    st.plotly_chart(monte_carlo_histogram(mc_result), use_container_width=True)
with mc_c2:
    st.plotly_chart(monte_carlo_fan_chart(mc_result), use_container_width=True)

st.info(
    f"📊 Median 12-month ending balance: **{fmt_dollar(mc_result.final_balance_p50)}** "
    f"(P10: {fmt_dollar(mc_result.final_balance_p10)}, "
    f"P90: {fmt_dollar(mc_result.final_balance_p90)})"
)


# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL HEALTH SCORE
# ═══════════════════════════════════════════════════════════════════════════════

section("🩺 Financial Health Score")

health = financial_health_score(
    df,
    emergency_fund_months=emergency_fund / max(exp_series["Expenses"].mean(), 1)
)

col_gauge, col_health = st.columns([2, 3])

with col_gauge:
    st.plotly_chart(health_score_gauge(health["score"], health["grade"]), use_container_width=True)

with col_health:
    st.markdown("**Component Scores**")
    comp_df = pd.DataFrame(list(health["components"].items()), columns=["Component", "Score"])
    comp_df["Max"] = [30, 25, 20, 25]
    for _, row in comp_df.iterrows():
        st.markdown(f"**{row['Component']}** &nbsp; `{row['Score']:.1f} / {row['Max']}`")
        st.progress(min(row["Score"] / row["Max"], 1.0))

    st.markdown("**💡 Recommendations**")
    for rec in health["recommendations"]:
        st.markdown(f"- {rec}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION REGISTER
# ═══════════════════════════════════════════════════════════════════════════════

section("📄 Transaction Register")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    sel_cat = st.selectbox("Filter by Category", ["All"] + sorted(df["Category"].unique().tolist()))
with col_f2:
    sel_acc = st.selectbox("Filter by Account",  ["All"] + sorted(df["Account"].unique().tolist()))
with col_f3:
    search = st.text_input("Search Merchant", "")

tx_display = df.copy()
if sel_cat != "All":
    tx_display = tx_display[tx_display["Category"] == sel_cat]
if sel_acc != "All":
    tx_display = tx_display[tx_display["Account"] == sel_acc]
if search:
    tx_display = tx_display[tx_display["Merchant"].str.contains(search, case=False, na=False)]

tx_display = tx_display[["Date", "Merchant", "Amount", "Category", "Account", "Type"]].copy()
tx_display["Date"] = tx_display["Date"].dt.strftime("%Y-%m-%d")

st.dataframe(
    tx_display.style.format({"Amount": "${:,.2f}"}),
    use_container_width=True,
    height=400,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Budget Forecaster · Built with Streamlit, Plotly, Statsmodels, and Scikit-learn")
