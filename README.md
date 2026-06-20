# 💰 Budget Forecaster

A professional personal financial planning and cash flow forecasting tool
built with Python — modelled after the type of financial dashboards used by
FP&A teams. Powered by Streamlit, Plotly, Statsmodels, and Scikit-learn.

---

## ✨ Features

| Module | What it does |
|---|---|
| **Transaction Import** | Upload any bank CSV; auto-cleans dates, amounts, duplicates |
| **Auto-Categoriser** | Keyword + fuzzy matching categorises every transaction |
| **KPI Dashboard** | Balance, income, expenses, savings rate, net cash flow |
| **Trend Charts** | Monthly cash flow, rolling averages, savings growth |
| **Heatmap** | Spending intensity by month × weekday |
| **Forecasting** | Moving Average, Linear Regression, Exp. Smoothing, ARIMA |
| **Cash Flow Projection** | Month-by-month forward balance table |
| **Scenario Analysis** | Income/rent/food/inflation sliders with instant re-projection |
| **Monte Carlo** | 10,000-path simulation with probability outputs |
| **Health Score** | 0–100 financial health score with component breakdown |
| **Transaction Register** | Searchable, filterable transaction table |

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- pip

### 2. Install
```bash
cd budget_forecaster
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`.

---

## 📂 Project Structure

```
budget_forecaster/
├── app.py                      ← Streamlit dashboard (entry point)
├── requirements.txt
├── README.md
│
├── data/
│   ├── sample_transactions.csv ← 12 months of realistic demo data
│   └── categories.csv          ← Keyword → category mapping
│
├── models/
│   ├── categorizer.py          ← Keyword + fuzzy transaction categoriser
│   ├── forecast.py             ← MA / LR / ES / ARIMA forecasting engine
│   └── scenario.py             ← Scenario engine + Monte Carlo simulation
│
├── utils/
│   ├── loader.py               ← CSV ingestion and data cleaning
│   ├── calculations.py         ← KPIs, metrics, and aggregations
│   └── charts.py               ← Plotly figure factory
│
└── exports/                    ← (future) PDF / Excel reports
```

---

## 📤 Uploading Your Own Data

Export a CSV from your bank with these columns:

| Column | Required | Notes |
|---|---|---|
| Date | ✅ | Any common format (MM/DD/YY, YYYY-MM-DD, …) |
| Merchant | ✅ | Payee / description |
| Amount | ✅ | Negative = expense, positive = income |
| Account | ☐ | Defaults to "Checking" |
| Type | ☐ | Auto-derived from Amount sign |

---

## 🔭 Forecasting Models

| Model | Best for | Notes |
|---|---|---|
| **Exponential Smoothing** | Most data (default) | Holt-Winters, handles trend |
| **ARIMA** | Autocorrelated series | Auto-selects fallback order |
| **Linear Regression** | Clear trends | Simple, interpretable |
| **Moving Average** | Stable spending | Naïve baseline |

---

## 🎲 Monte Carlo Simulation

Runs N simulations (default 10,000) drawing monthly income and expenses
from normal distributions and adding random surprise expenses.

Outputs:
- Probability balance stays positive
- Probability balance falls below emergency threshold
- Probability savings goal is achieved
- P10 / P50 / P90 final balance percentiles
- Fan chart of simulation paths

---

## 🩺 Financial Health Score

| Component | Weight | Target |
|---|---|---|
| Savings Rate | 30 pts | ≥ 20% |
| Emergency Fund | 25 pts | ≥ 6 months |
| Expense Consistency | 20 pts | Low month-to-month variance |
| Cash Flow Positivity | 25 pts | Income > expenses every month |

Grades: **Excellent** (90–100) · **Good** (75–89) · **Fair** (60–74) · **Needs Improvement** (< 60)

---