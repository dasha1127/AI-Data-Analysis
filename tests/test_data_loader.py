"""
Tests for data_loader — CSV normalization and schema consistency.
Run with: pytest tests/
"""

import pandas as pd
import pytest

from data_loader import load_combined, most_improved, UNIFIED_COLS


EXPECTED_COLS = set(UNIFIED_COLS)


class TestLoadCombined:
    @pytest.fixture(scope="class")
    def df(self):
        return load_combined()

    def test_has_all_unified_columns(self, df):
        assert EXPECTED_COLS == set(df.columns)

    def test_covers_all_8_years(self, df):
        assert set(df["year"].unique()) == {2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022}

    def test_no_duplicate_country_year_pairs(self, df):
        dupes = df.duplicated(subset=["country", "year"]).sum()
        assert dupes == 0

    def test_happiness_score_in_valid_range(self, df):
        scores = df["happiness_score"].dropna()
        assert scores.between(0, 10).all(), "Scores must be between 0 and 10"

    def test_year_column_is_integer(self, df):
        assert df["year"].dtype in ("int64", "int32")

    def test_country_column_has_no_leading_spaces(self, df):
        has_spaces = df["country"].str.startswith(" ").any()
        assert not has_spaces

    def test_row_count_reasonable(self, df):
        assert len(df) >= 1000, "Expected at least 1000 rows across 8 years"


class TestMostImproved:
    @pytest.fixture(scope="class")
    def df(self):
        return load_combined()

    def test_returns_correct_number_of_rows(self, df):
        result = most_improved(df, top_n=5)
        assert len(result) == 5

    def test_change_column_is_positive(self, df):
        result = most_improved(df, top_n=10)
        assert (result["change"] > 0).all()

    def test_sorted_descending(self, df):
        result = most_improved(df, top_n=10)
        assert result["change"].is_monotonic_decreasing
