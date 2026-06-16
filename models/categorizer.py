"""
categorizer.py – Automatic and manual transaction categorisation.

Uses keyword matching (fast) followed by fuzzy matching (fallback).
Users can supply manual overrides that take highest priority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import process, fuzz


# Default fallback category
_UNCATEGORISED = "Other"


class TransactionCategorizer:
    """
    Categorise bank transactions by merchant name.

    Priority order (highest → lowest):
    1. Manual overrides supplied by the user.
    2. Exact keyword/substring match against the category dictionary.
    3. Fuzzy string match (RapidFuzz) when no exact match is found.
    4. 'Other' as the final fallback.

    Parameters
    ----------
    category_dict : dict[str, str]
        Mapping of lowercase keywords → category names.
    manual_overrides : dict[str, str], optional
        Mapping of merchant name (any case) → category name.
        These are applied first and override all automatic rules.
    fuzzy_threshold : int
        Minimum fuzzy-match score (0–100) to accept a match. Default 75.
    """

    def __init__(
        self,
        category_dict: dict[str, str],
        manual_overrides: Optional[dict[str, str]] = None,
        fuzzy_threshold: int = 75,
    ) -> None:
        self.category_dict: dict[str, str] = {
            k.lower(): v for k, v in category_dict.items()
        }
        self.manual_overrides: dict[str, str] = {
            k.lower(): v for k, v in (manual_overrides or {}).items()
        }
        self.fuzzy_threshold = fuzzy_threshold
        # Pre-build keyword list for fuzzy matching
        self._keywords = list(self.category_dict.keys())

    # ── Public API ─────────────────────────────────────────────────────────────

    def categorise(self, merchant: str) -> str:
        """Return the category for a single merchant name."""
        if not isinstance(merchant, str) or not merchant.strip():
            return _UNCATEGORISED

        merchant_lower = merchant.lower().strip()

        # 1. Manual overrides
        if merchant_lower in self.manual_overrides:
            return self.manual_overrides[merchant_lower]

        # 2. Keyword substring match
        for keyword, category in self.category_dict.items():
            if keyword in merchant_lower:
                return category

        # 3. Fuzzy fallback
        if self._keywords:
            result = process.extractOne(
                merchant_lower,
                self._keywords,
                scorer=fuzz.partial_ratio,
                score_cutoff=self.fuzzy_threshold,
            )
            if result:
                matched_keyword, _score, _idx = result
                return self.category_dict[matched_keyword]

        return _UNCATEGORISED

    def categorise_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add or update a 'Category' column on the DataFrame in-place.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'Merchant' column.

        Returns
        -------
        pd.DataFrame
            The same DataFrame with a populated 'Category' column.
        """
        df = df.copy()
        df["Category"] = df["Merchant"].apply(self.categorise)
        return df

    def add_manual_override(self, merchant: str, category: str) -> None:
        """Add or update a single manual override at runtime."""
        self.manual_overrides[merchant.lower().strip()] = category

    def bulk_override(self, overrides: dict[str, str]) -> None:
        """Apply a batch of manual overrides."""
        for merchant, category in overrides.items():
            self.add_manual_override(merchant, category)

    def get_category_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarise spending by category.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'Category' and 'Amount' columns.

        Returns
        -------
        pd.DataFrame
            Columns: Category, Total, Count, Avg, Pct.
        """
        expenses = df[df["Amount"] < 0].copy()
        expenses["AbsAmount"] = expenses["Amount"].abs()

        summary = (
            expenses.groupby("Category")["AbsAmount"]
            .agg(Total="sum", Count="count", Avg="mean")
            .reset_index()
            .sort_values("Total", ascending=False)
        )
        total_spend = summary["Total"].sum()
        summary["Pct"] = (summary["Total"] / total_spend * 100).round(1)
        return summary.reset_index(drop=True)

    def get_all_categories(self) -> list[str]:
        """Return sorted list of all unique category values in the dict."""
        return sorted(set(self.category_dict.values()))
