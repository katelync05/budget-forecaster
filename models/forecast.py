"""
forecast.py – Multi-model expense forecasting engine.

Supports:
  - Moving Average
  - Linear Regression
  - Exponential Smoothing (Holt-Winters)
  - ARIMA

Each model returns a standardised ForecastResult dataclass.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

ModelName = Literal["moving_average", "linear_regression", "exponential_smoothing", "arima"]


@dataclass
class ForecastResult:
    """Container for a forecast run."""
    model_name: str
    forecast: pd.DataFrame               # columns: Date, Predicted, Lower, Upper
    metrics: dict[str, float] = field(default_factory=dict)
    in_sample: pd.DataFrame = field(default_factory=pd.DataFrame)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prepare_series(monthly_expenses: pd.DataFrame) -> pd.Series:
    """Return a clean numeric Series indexed by month timestamp."""
    s = monthly_expenses.set_index("YearMonth")["Expenses"].astype(float)
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def _future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, MAPE."""
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE":  float(mean_absolute_error(actual, predicted)),
        "MAPE": float(mape),
    }


def _make_forecast_df(dates: pd.DatetimeIndex, predicted: np.ndarray,
                      lower: np.ndarray, upper: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "Date":      dates,
        "Predicted": np.clip(predicted, 0, None),
        "Lower":     np.clip(lower, 0, None),
        "Upper":     np.clip(upper, 0, None),
    })


# ── Model Implementations ─────────────────────────────────────────────────────

def moving_average_forecast(series: pd.Series, horizon: int = 12, window: int = 3) -> ForecastResult:
    """
    Simple rolling moving-average forecast with naïve confidence bands.
    CI width = 1.96 × rolling std.
    """
    values = series.values
    ma     = pd.Series(values).rolling(window, min_periods=1).mean().values
    std    = pd.Series(values).rolling(window, min_periods=1).std().fillna(0).values

    # In-sample fitted values
    in_sample = pd.DataFrame({"Date": series.index, "Actual": values, "Fitted": ma})

    # Forecast: repeat the last MA value with expanding CI
    last_ma  = float(ma[-1])
    last_std = float(std[-1]) if std[-1] > 0 else float(np.std(values))
    future_dates = _future_dates(series.index[-1], horizon)

    predicted = np.full(horizon, last_ma)
    t_factors  = np.arange(1, horizon + 1) ** 0.5    # widen CI over time
    lower = predicted - 1.96 * last_std * t_factors
    upper = predicted + 1.96 * last_std * t_factors

    metrics = _compute_metrics(values, ma)
    return ForecastResult(
        model_name="Moving Average",
        forecast=_make_forecast_df(future_dates, predicted, lower, upper),
        metrics=metrics,
        in_sample=in_sample,
    )


def linear_regression_forecast(series: pd.Series, horizon: int = 12) -> ForecastResult:
    """
    OLS linear regression on a numeric time index.
    CI based on prediction interval (±1.96 × residual std).
    """
    n = len(series)
    X = np.arange(n).reshape(-1, 1)
    y = series.values

    model = LinearRegression()
    model.fit(X, y)
    fitted = model.predict(X)

    residual_std = float(np.std(y - fitted))

    future_X = np.arange(n, n + horizon).reshape(-1, 1)
    predicted = model.predict(future_X)

    lower = predicted - 1.96 * residual_std
    upper = predicted + 1.96 * residual_std
    future_dates = _future_dates(series.index[-1], horizon)

    in_sample = pd.DataFrame({"Date": series.index, "Actual": y, "Fitted": fitted})
    metrics    = _compute_metrics(y, fitted)

    return ForecastResult(
        model_name="Linear Regression",
        forecast=_make_forecast_df(future_dates, predicted, lower, upper),
        metrics=metrics,
        in_sample=in_sample,
    )


def exponential_smoothing_forecast(series: pd.Series, horizon: int = 12) -> ForecastResult:
    """
    Holt-Winters Triple Exponential Smoothing (additive trend, no seasonality
    unless ≥ 24 observations).
    """
    n = len(series)
    seasonal = "add" if n >= 24 else None
    seasonal_periods = 12 if seasonal else None

    es_model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fit = es_model.fit(optimized=True)
    fitted = fit.fittedvalues

    forecast_obj = fit.forecast(horizon)
    future_dates = _future_dates(series.index[-1], horizon)
    predicted    = forecast_obj.values

    # Bootstrap CI from residuals
    residual_std = float(np.std(series.values - fitted.values))
    t_factors    = np.arange(1, horizon + 1) ** 0.5
    lower = predicted - 1.96 * residual_std * t_factors
    upper = predicted + 1.96 * residual_std * t_factors

    in_sample = pd.DataFrame({
        "Date": series.index, "Actual": series.values, "Fitted": fitted.values
    })
    metrics = _compute_metrics(series.values, fitted.values)

    return ForecastResult(
        model_name="Exponential Smoothing",
        forecast=_make_forecast_df(future_dates, predicted, lower, upper),
        metrics=metrics,
        in_sample=in_sample,
    )


def arima_forecast(series: pd.Series, horizon: int = 12,
                   order: tuple[int, int, int] = (1, 1, 1)) -> ForecastResult:
    """
    ARIMA(p, d, q) forecast.  Defaults to (1,1,1); auto-selects order if
    the default fails.
    """
    orders_to_try = [order, (2, 1, 2), (1, 0, 1), (0, 1, 1)]
    fit = None
    for o in orders_to_try:
        try:
            fit = ARIMA(series, order=o).fit()
            break
        except Exception:
            continue

    if fit is None:
        # Final fallback: simple mean model
        return moving_average_forecast(series, horizon)

    fitted = fit.fittedvalues
    fc     = fit.get_forecast(steps=horizon)
    future_dates = _future_dates(series.index[-1], horizon)

    ci = fc.conf_int(alpha=0.05)
    predicted = fc.predicted_mean.values
    lower     = ci.iloc[:, 0].values
    upper     = ci.iloc[:, 1].values

    in_sample = pd.DataFrame({
        "Date": series.index, "Actual": series.values, "Fitted": fitted.values
    })
    metrics = _compute_metrics(series.values[1:], fitted.values[1:])  # skip first NaN

    return ForecastResult(
        model_name="ARIMA",
        forecast=_make_forecast_df(future_dates, predicted, lower, upper),
        metrics=metrics,
        in_sample=in_sample,
    )


# ── Public Entry Point ────────────────────────────────────────────────────────

def run_forecast(
    monthly_expenses: pd.DataFrame,
    model: ModelName = "exponential_smoothing",
    horizon: int = 12,
) -> ForecastResult:
    """
    Run a forecast using the specified model.

    Parameters
    ----------
    monthly_expenses : pd.DataFrame
        Output of `calculations.monthly_expense_series()`.
        Must have columns 'YearMonth' (timestamp) and 'Expenses'.
    model : ModelName
        One of 'moving_average', 'linear_regression',
        'exponential_smoothing', 'arima'.
    horizon : int
        Number of months to forecast ahead.

    Returns
    -------
    ForecastResult
    """
    series = _prepare_series(monthly_expenses)

    dispatch = {
        "moving_average":        moving_average_forecast,
        "linear_regression":     linear_regression_forecast,
        "exponential_smoothing": exponential_smoothing_forecast,
        "arima":                 arima_forecast,
    }

    fn = dispatch.get(model)
    if fn is None:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(dispatch)}")

    return fn(series, horizon=horizon)


def cash_flow_projection(
    current_balance: float,
    monthly_income: float,
    forecast_result: ForecastResult,
) -> pd.DataFrame:
    """
    Build a month-by-month cash-flow table.

    Returns
    -------
    pd.DataFrame
        Columns: Month, Beginning Balance, Income, Expenses, Ending Balance.
    """
    rows = []
    balance = current_balance
    for _, row in forecast_result.forecast.iterrows():
        beginning = balance
        income    = monthly_income
        expenses  = float(row["Predicted"])
        ending    = beginning + income - expenses
        rows.append({
            "Month":             row["Date"].strftime("%b %Y"),
            "Beginning Balance": round(beginning, 2),
            "Income":            round(income, 2),
            "Expenses":          round(expenses, 2),
            "Ending Balance":    round(ending, 2),
        })
        balance = ending
    return pd.DataFrame(rows)
