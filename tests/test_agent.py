"""
Tests for DataAgent — intent detection, rule-based analysis, safe execution.
Run with: pytest tests/
"""

import pandas as pd
import pytest

from agent import DataAgent, detect_intent, safe_exec, extract_column


# ── Fixtures — reusable sample dataframes ─────────────────────────────────────

@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    return pd.DataFrame({
        "score":  [7.8, 7.6, 7.5, 6.9, 6.4],
        "gdp":    [1.4, 1.3, 1.3, 1.1, 0.9],
        "year":   [2021, 2021, 2021, 2021, 2021],
    })


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "country": ["Finland", "Denmark", "Iceland", "Norway", "Sweden"],
        "region":  ["Western Europe"] * 5,
        "score":   [7.8, 7.6, 7.5, 6.9, 6.4],
        "gdp":     [1.4, 1.3, 1.3, 1.1, 0.9],
        "year":    [2021, 2021, 2021, 2021, 2021],
    })


@pytest.fixture()
def trend_df() -> pd.DataFrame:
    return pd.DataFrame({
        "country": ["Finland"] * 3 + ["Denmark"] * 3,
        "year":    [2019, 2020, 2021] * 2,
        "score":   [7.7, 7.8, 7.9, 7.5, 7.6, 7.7],
    })


@pytest.fixture()
def missing_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "country": ["Finland", "Denmark", None, "Norway"],
        "score":   [7.8, None, 7.5, 6.9],
    })
    return df


# ── Intent detection ──────────────────────────────────────────────────────────

class TestDetectIntent:
    def test_correlation(self):
        assert detect_intent("show the correlation heatmap") == "correlation"

    def test_correlation_vs(self):
        assert detect_intent("gdp vs happiness score") == "correlation"

    def test_distribution(self):
        assert detect_intent("distribution of happiness scores") == "distribution"

    def test_trend(self):
        assert detect_intent("how has happiness changed over time?") == "trend"

    def test_top(self):
        assert detect_intent("top 10 happiest countries") == "top"

    def test_bottom(self):
        assert detect_intent("bottom 5 worst performing countries") == "bottom"

    def test_missing(self):
        assert detect_intent("are there any missing values?") == "missing"

    def test_outlier(self):
        assert detect_intent("find outliers in the data") == "outlier"

    def test_average(self):
        assert detect_intent("what is the average score?") == "average"

    def test_unknown_defaults_to_summary(self):
        assert detect_intent("xyzzy blorp") == "summary"

    def test_case_insensitive(self):
        assert detect_intent("SHOW THE TREND OVER TIME") == "trend"


# ── Column extraction ─────────────────────────────────────────────────────────

class TestExtractColumn:
    def test_finds_column_in_question(self, mixed_df):
        assert extract_column("distribution of score", mixed_df) == "score"

    def test_returns_none_when_no_match(self, mixed_df):
        assert extract_column("show me something interesting", mixed_df) is None

    def test_case_insensitive(self, mixed_df):
        assert extract_column("what is the SCORE distribution", mixed_df) == "score"


# ── Safe code execution ───────────────────────────────────────────────────────

class TestSafeExec:
    def test_assigns_result(self, numeric_df):
        code = "result = df['score'].mean()"
        out = safe_exec(code, numeric_df)
        assert "text" in out
        assert float(out["text"]) == pytest.approx(numeric_df["score"].mean())

    def test_assigns_fig(self, numeric_df):
        code = "import plotly.express as px\nfig = px.histogram(df, x='score')"
        out = safe_exec(code, numeric_df)
        assert "fig" in out

    def test_does_not_mutate_original(self, numeric_df):
        original_mean = numeric_df["score"].mean()
        code = "df['score'] = 0"
        safe_exec(code, numeric_df)
        assert numeric_df["score"].mean() == original_mean

    def test_returns_empty_dict_when_nothing_assigned(self, numeric_df):
        out = safe_exec("x = 1 + 1", numeric_df)
        assert out == {}


# ── DataAgent rule-based analysis ─────────────────────────────────────────────

class TestDataAgent:
    def test_correlation_returns_fig(self, numeric_df):
        result = DataAgent(numeric_df).analyze("show correlation")
        assert "fig" in result

    def test_correlation_returns_text(self, numeric_df):
        result = DataAgent(numeric_df).analyze("show correlation")
        assert "text" in result
        assert "↔" in result["text"]

    def test_distribution_numeric(self, mixed_df):
        result = DataAgent(mixed_df).analyze("distribution of score")
        assert "fig" in result
        assert "Mean" in result.get("text", "")

    def test_distribution_categorical(self, mixed_df):
        result = DataAgent(mixed_df).analyze("distribution of country")
        assert "fig" in result

    def test_top_ranking(self, mixed_df):
        result = DataAgent(mixed_df).analyze("top countries by score")
        assert "fig" in result

    def test_bottom_ranking(self, mixed_df):
        result = DataAgent(mixed_df).analyze("bottom countries by score")
        assert "fig" in result

    def test_missing_values_detected(self, missing_df):
        result = DataAgent(missing_df).analyze("any missing values?")
        assert "fig" in result or "missing" in result.get("text", "").lower()

    def test_no_missing_values_message(self, numeric_df):
        result = DataAgent(numeric_df).analyze("any missing values?")
        assert "No missing" in result.get("text", "")

    def test_outlier_detection(self, numeric_df):
        result = DataAgent(numeric_df).analyze("outliers in score")
        assert "fig" in result
        assert "outlier" in result.get("text", "").lower()

    def test_trend_with_year_column(self, trend_df):
        result = DataAgent(trend_df).analyze("trend over time")
        assert "fig" in result

    def test_compare(self, mixed_df):
        result = DataAgent(mixed_df).analyze("compare score by country")
        assert "fig" in result

    def test_summary_fallback(self, mixed_df):
        result = DataAgent(mixed_df).analyze("xyzzy blorp nonsense")
        assert "text" in result or "fig" in result

    def test_auto_insights_returns_list(self, mixed_df):
        insights = DataAgent(mixed_df).auto_insights()
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all("fig" in i for i in insights)

    def test_no_groq_key_uses_rule_based(self, mixed_df):
        agent = DataAgent(mixed_df, groq_key=None)
        result = agent.analyze("show correlation")
        assert "fig" in result
