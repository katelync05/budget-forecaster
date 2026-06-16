"""
loader.py – Transaction data ingestion and cleaning pipeline.

Handles CSV imports, data validation, deduplication, date parsing,
and merchant name standardization.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── Column name aliases accepted from external CSVs ──────────────────────────
_DATE_ALIASES    = {"date", "transaction date", "trans date", "posted date"}
_MERCHANT_ALIASES = {"merchant", "description", "payee", "name", "vendor"}
_AMOUNT_ALIASES  = {"amount", "debit/credit", "value", "transaction amount"}
_ACCOUNT_ALIASES = {"account", "account name", "bank account"}
_TYPE_ALIASES    = {"type", "transaction type", "category type"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names (Date, Merchant, Amount, Account, Type)."""
    rename: dict[str, str] = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in _DATE_ALIASES:
            rename[col] = "Date"
        elif lower in _MERCHANT_ALIASES:
            rename[col] = "Merchant"
        elif lower in _AMOUNT_ALIASES:
            rename[col] = "Amount"
        elif lower in _ACCOUNT_ALIASES:
            rename[col] = "Account"
        elif lower in _TYPE_ALIASES:
            rename[col] = "Type"
    return df.rename(columns=rename)


def _clean_amount(series: pd.Series) -> pd.Series:
    """Strip currency symbols / commas and cast to float."""
    return (
        series.astype(str)
        .str.replace(r"[$,\s]", "", regex=True)
        .str.replace(r"\((.+)\)", r"-\1", regex=True)  # (100) → -100
        .pipe(pd.to_numeric, errors="coerce")
    )


def _standardise_merchant(series: pd.Series) -> pd.Series:
    """Lowercase, strip extra whitespace, remove reference numbers."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        # remove trailing transaction IDs like "#12345" or "REF 9987"
        .str.replace(r"\s*(#|ref|txn|id)\s*\w+$", "", flags=re.IGNORECASE, regex=True)
        .str.title()
    )


def load_transactions(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """
    Load, clean, and validate transaction data.

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        Path to a CSV file or an already-loaded DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        Date (datetime64), Merchant (str), Amount (float),
        Account (str), Type (str).
    """
    # ── 1. Ingest ─────────────────────────────────────────────────────────────
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source, dtype=str)

    df = _normalise_columns(df)

    # ── 2. Require mandatory columns ──────────────────────────────────────────
    required = {"Date", "Merchant", "Amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ── 3. Parse dates ────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ── 4. Clean amounts ──────────────────────────────────────────────────────
    df["Amount"] = _clean_amount(df["Amount"])

    # ── 5. Standardise merchant names ─────────────────────────────────────────
    df["Merchant"] = _standardise_merchant(df["Merchant"])

    # ── 6. Fill optional columns ──────────────────────────────────────────────
    if "Account" not in df.columns:
        df["Account"] = "Checking"
    else:
        df["Account"] = df["Account"].fillna("Unknown").str.strip().str.title()

    derived_type = pd.Series(
        np.where(df["Amount"] >= 0, "Income", "Expense"), index=df.index
    )
    if "Type" not in df.columns:
        df["Type"] = derived_type
    else:
        df["Type"] = df["Type"].fillna(derived_type)

    # ── 7. Drop rows with critical nulls ──────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["Date", "Amount"])
    dropped = before - len(df)
    if dropped:
        print(f"[loader] ⚠  Dropped {dropped} rows with unparseable Date or Amount.")

    # ── 8. Remove exact duplicates ────────────────────────────────────────────
    dupes = df.duplicated()
    df = df[~dupes].reset_index(drop=True)
    if dupes.sum():
        print(f"[loader] ℹ  Removed {dupes.sum()} duplicate rows.")

    # ── 9. Sort chronologically ───────────────────────────────────────────────
    df = df.sort_values("Date").reset_index(drop=True)

    # ── 10. Derived helper columns ────────────────────────────────────────────
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["YearMonth"]  = df["Date"].dt.to_period("M")
    df["DayOfWeek"]  = df["Date"].dt.day_name()
    df["AbsAmount"]  = df["Amount"].abs()

    return df


def load_categories(path: str | Path | None = None) -> dict[str, str]:
    """
    Load keyword → category mapping from CSV.

    The CSV must have columns: keyword, category.
    Returns a lowercase-keyed dict for fast substring matching.
    """
    if path is None:
        path = Path(__file__).parent.parent / "data" / "categories.csv"

    cat_df = pd.read_csv(path, dtype=str)
    cat_df.columns = cat_df.columns.str.strip().str.lower()

    if not {"keyword", "category"}.issubset(cat_df.columns):
        raise ValueError("categories.csv must have 'keyword' and 'category' columns.")

    return dict(zip(cat_df["keyword"].str.lower().str.strip(),
                    cat_df["category"].str.strip()))


def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions into a monthly summary.

    Returns a DataFrame indexed by YearMonth with columns:
    Income, Expenses, Net, Transactions.
    """
    monthly = df.groupby("YearMonth").apply(
        lambda g: pd.Series({
            "Income":       g.loc[g["Amount"] > 0, "Amount"].sum(),
            "Expenses":     abs(g.loc[g["Amount"] < 0, "Amount"].sum()),
            "Net":          g["Amount"].sum(),
            "Transactions": len(g),
        })
    ).reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()
    return monthly.sort_values("YearMonth").reset_index(drop=True)
