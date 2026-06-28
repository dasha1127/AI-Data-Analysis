"""
DataAgent — routes natural language questions to chart + text answers.

Two modes:
  LLM mode  : Groq API generates pandas/plotly code, agent executes it safely.
  Rule-based: Intent classification + hardcoded chart templates (no API needed).
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import CHART, INTENT_PATTERNS, MODEL

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

AnalysisResult = dict[str, Any]  # keys: "fig", "text", "code" (all optional)


# ── Exceptions ────────────────────────────────────────────────────────────────

class AnalysisError(Exception):
    """Raised when analysis fails and cannot recover."""


class CodeExecutionError(AnalysisError):
    """Raised when LLM-generated code fails to execute."""


# ── Intent detection ─────────────────────────────────────────────────────────

def detect_intent(question: str) -> str:
    """Return the analysis intent for a plain-English question."""
    lowered = question.lower()
    for intent, keywords in INTENT_PATTERNS.items():
        if any(kw in lowered for kw in keywords):
            return intent
    return "summary"


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def extract_column(question: str, df: pd.DataFrame) -> str | None:
    """Return the first column name mentioned in the question, or None."""
    lowered = question.lower()
    return next((col for col in df.columns if col.lower() in lowered), None)


# ── Safe code execution ───────────────────────────────────────────────────────

def safe_exec(code: str, df: pd.DataFrame) -> AnalysisResult:
    """
    Execute LLM-generated code in an isolated namespace.

    The code may assign to `fig` (plotly figure) or `result` (text answer).
    A copy of df is provided so the original is never mutated.
    """
    namespace: dict[str, Any] = {
        "df": df.copy(),
        "pd": pd,
        "px": px,
        "go": go,
        "np": np,
        "fig": None,
        "result": None,
    }
    try:
        exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
    except Exception:
        # Retry with builtins if the restricted sandbox blocks a needed call
        exec(code, namespace)  # noqa: S102

    output: AnalysisResult = {}
    if namespace["fig"] is not None:
        output["fig"] = namespace["fig"]
    if namespace["result"] is not None:
        output["text"] = str(namespace["result"])
    return output


# ── DataAgent ─────────────────────────────────────────────────────────────────

class DataAgent:
    """
    Wraps a DataFrame and answers natural-language questions about it.

    Usage:
        agent = DataAgent(df, groq_key="gsk_...")
        result = agent.analyze("Which country improved the most?")
        # result = {"fig": <plotly figure>, "text": "..."}
    """

    def __init__(self, df: pd.DataFrame, groq_key: str | None = None) -> None:
        self.df = df
        self.groq_key = groq_key
        self._num_cols = numeric_cols(df)
        self._cat_cols = categorical_cols(df)

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(self, question: str) -> AnalysisResult:
        """Route a question to LLM or rule-based analysis and return a result."""
        if self.groq_key:
            try:
                return self._llm_analyze(question)
            except Exception as exc:
                logger.warning("LLM analysis failed (%s), falling back to rules.", exc)
        return self._rule_based_analyze(question)

    def auto_insights(self) -> list[AnalysisResult]:
        """Return a list of automatic insight charts for the loaded dataset."""
        insights: list[AnalysisResult] = []

        if len(self._num_cols) >= 2:
            corr = self.df[self._num_cols].corr()
            insights.append({
                "title": "Correlations",
                "fig": px.imshow(
                    corr, text_auto=".2f", title="Correlation Heatmap",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    height=CHART.insight_height,
                ),
            })

        if self._num_cols:
            col = self._num_cols[0]
            insights.append({
                "title": f"Distribution of {col}",
                "fig": px.histogram(
                    self.df, x=col, nbins=CHART.histogram_bins,
                    title=f"Distribution — {col}", marginal="box",
                    height=350,
                ),
            })

        if self._cat_cols and self._num_cols:
            cat, num = self._cat_cols[0], self._num_cols[0]
            top = self.df[cat].value_counts().head(CHART.top_n_categories).index
            filtered = self.df[self.df[cat].isin(top)]
            insights.append({
                "title": f"{num} by {cat}",
                "fig": px.box(filtered, x=cat, y=num, title=f"{num} by {cat}", height=350),
            })

        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            miss_df = (missing[missing > 0] / len(self.df) * 100).round(1).reset_index()
            miss_df.columns = ["Column", "Missing %"]
            insights.append({
                "title": "Missing Values",
                "fig": px.bar(miss_df, x="Column", y="Missing %",
                              title="Missing Values (%)", height=300),
            })

        return insights

    # ── Private: LLM mode ─────────────────────────────────────────────────────

    def _build_schema(self) -> str:
        lines = [
            f"  - {col} ({self.df[col].dtype}): e.g. {self.df[col].dropna().head(3).tolist()}"
            for col in self.df.columns
        ]
        return "\n".join(lines)

    def _llm_analyze(self, question: str) -> AnalysisResult:
        from groq import Groq  # lazy import — only needed in LLM mode

        client = Groq(api_key=self.groq_key)
        schema = self._build_schema()
        shape = f"{self.df.shape[0]} rows × {self.df.shape[1]} cols"

        system_prompt = (
            "You are a data analysis expert. Generate Python code to answer questions "
            "about a pandas DataFrame.\n\n"
            "Rules:\n"
            "- The dataframe is already loaded as `df`\n"
            "- Available: pd (pandas), px (plotly.express), go (plotly.graph_objects), np (numpy)\n"
            "- Assign charts to `fig`, text answers to `result`\n"
            "- Return executable Python only — no markdown fences, no explanation\n"
        )

        user_msg = f"Dataset — Shape: {shape}\nColumns:\n{schema}\n\nQuestion: {question}\n\nCode:"

        response = client.chat.completions.create(
            model=MODEL.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=MODEL.temperature,
            max_tokens=MODEL.max_tokens,
        )

        code = response.choices[0].message.content.strip()
        code = re.sub(r"^```(?:python)?\n?", "", code)
        code = re.sub(r"\n?```$", "", code)

        try:
            result = safe_exec(code, self.df)
            result["code"] = code
            return result
        except Exception as exc:
            raise CodeExecutionError(f"Generated code failed: {exc}\n\n{code}") from exc

    # ── Private: rule-based mode ──────────────────────────────────────────────

    def _rule_based_analyze(self, question: str) -> AnalysisResult:
        intent = detect_intent(question)
        col = extract_column(question, self.df)
        num = self._num_cols
        cat = self._cat_cols

        try:
            return self._dispatch(intent, col, num, cat)
        except Exception as exc:
            logger.exception("Rule-based analysis failed for intent '%s'.", intent)
            return {"text": f"Could not analyze: {exc}. Try rephrasing with a column name."}

    def _dispatch(
        self,
        intent: str,
        col: str | None,
        num: list[str],
        cat: list[str],
    ) -> AnalysisResult:
        if intent == "correlation" and len(num) >= 2:
            return self._chart_correlation(num)
        if intent == "distribution":
            return self._chart_distribution(col, num, cat)
        if intent in ("top", "bottom"):
            return self._chart_ranking(intent, col, num, cat)
        if intent == "compare" and cat and num:
            return self._chart_compare(col, num, cat)
        if intent == "missing":
            return self._chart_missing()
        if intent == "outlier" and num:
            return self._chart_outlier(col, num)
        if intent == "trend" and num:
            return self._chart_trend(col, num, cat)
        if intent == "average" and num:
            return self._chart_average(col, num, cat)
        return self._chart_summary(num, cat)

    def _chart_correlation(self, num: list[str]) -> AnalysisResult:
        corr = self.df[num].corr()
        fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        unstacked = corr.unstack()
        mask = unstacked.index.get_level_values(0) != unstacked.index.get_level_values(1)
        top = unstacked[mask].sort_values(ascending=False).drop_duplicates()
        text = f"Strongest: {top.index[0][0]} ↔ {top.index[0][1]} ({top.iloc[0]:.2f})"
        return {"fig": fig, "text": text}

    def _chart_distribution(
        self, col: str | None, num: list[str], cat: list[str]
    ) -> AnalysisResult:
        target = col or (num[0] if num else None)
        if target and target in num:
            stats = self.df[target].describe()
            return {
                "fig": px.histogram(self.df, x=target, nbins=CHART.histogram_bins,
                                    title=f"Distribution of {target}", marginal="box"),
                "text": (f"Mean: {stats['mean']:.2f} | "
                         f"Median: {self.df[target].median():.2f} | "
                         f"Std: {stats['std']:.2f}"),
            }
        target = col or (cat[0] if cat else self.df.columns[0])
        vc = self.df[target].value_counts().head(20)
        return {
            "fig": px.bar(x=vc.index, y=vc.values, title=f"Distribution of {target}",
                          labels={"x": target, "y": "Count"}),
            "text": f"Most common: {vc.index[0]} ({vc.iloc[0]} times)",
        }

    def _chart_ranking(
        self, intent: str, col: str | None, num: list[str], cat: list[str]
    ) -> AnalysisResult:
        target = col or (num[0] if num else None)
        if not target:
            return {"text": "Please mention a column name in your question."}
        ascending = intent == "bottom"
        label = "Top" if intent == "top" else "Bottom"
        if cat and cat[0] != target:
            grp = (self.df.groupby(cat[0])[target].mean()
                   .reset_index()
                   .sort_values(target, ascending=ascending)
                   .head(CHART.top_n_categories))
            return {"fig": px.bar(grp, x=cat[0], y=target, title=f"{label} 10 by {target}")}
        sorted_df = (self.df[[target]].dropna()
                     .sort_values(target, ascending=ascending)
                     .head(CHART.top_n_categories))
        return {"fig": px.bar(sorted_df, y=target, title=f"{label} 10 values of {target}")}

    def _chart_compare(
        self, col: str | None, num: list[str], cat: list[str]
    ) -> AnalysisResult:
        cat_col = col if col in cat else cat[0]
        num_col = num[0]
        top = self.df[cat_col].value_counts().head(CHART.top_n_categories).index
        filtered = self.df[self.df[cat_col].isin(top)]
        means = filtered.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
        return {
            "fig": px.box(filtered, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}"),
            "text": f"Highest avg {num_col}: {means.index[0]} ({means.iloc[0]:.2f})",
        }

    def _chart_missing(self) -> AnalysisResult:
        missing = self.df.isnull().sum()
        miss_df = pd.DataFrame({
            "Missing Count": missing,
            "Missing %": (missing / len(self.df) * 100).round(2),
        })
        miss_df = miss_df[miss_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
        if miss_df.empty:
            return {"text": "No missing values found in this dataset."}
        return {
            "fig": px.bar(miss_df, y=miss_df.index, x="Missing %",
                          orientation="h", title="Missing Values by Column"),
            "text": (f"Total missing: {missing.sum()} | "
                     f"Columns affected: {(missing > 0).sum()}"),
        }

    def _chart_outlier(self, col: str | None, num: list[str]) -> AnalysisResult:
        target = col or num[0]
        q1, q3 = self.df[target].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_outliers = int(((self.df[target] < q1 - 1.5 * iqr) |
                          (self.df[target] > q3 + 1.5 * iqr)).sum())
        return {
            "fig": px.box(self.df, y=target, title=f"Outlier Detection — {target}",
                          points="outliers"),
            "text": f"Found {n_outliers} outliers ({n_outliers / len(self.df) * 100:.1f}%) in {target}.",
        }

    def _chart_trend(
        self, col: str | None, num: list[str], cat: list[str]
    ) -> AnalysisResult:
        year_col = next((c for c in self.df.columns if c.lower() == "year"), None)
        non_year = [c for c in num if c.lower() != "year"]
        target = (col if col and col.lower() != "year" else None) or (non_year[0] if non_year else num[0])

        if year_col:
            group_col = next((c for c in cat if c != year_col), None)
            if group_col:
                top = self.df[group_col].value_counts().head(CHART.top_n_trend_groups).index
                trend = (self.df[self.df[group_col].isin(top)]
                         .groupby([year_col, group_col])[target].mean()
                         .reset_index())
                return {"fig": px.line(trend, x=year_col, y=target, color=group_col,
                                       title=f"{target} by {group_col} over years", markers=True)}
            trend = self.df.groupby(year_col)[target].mean().reset_index()
            return {"fig": px.line(trend, x=year_col, y=target,
                                   title=f"Average {target} over years", markers=True)}

        date_cols = [c for c in self.df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_cols:
            x_col = date_cols[0]
            tmp = self.df.copy()
            tmp[x_col] = pd.to_datetime(tmp[x_col], errors="coerce")
            trend = (tmp.dropna(subset=[x_col])
                     .sort_values(x_col)
                     .set_index(x_col)[target]
                     .resample("ME").mean()
                     .reset_index())
            return {"fig": px.line(trend, x=x_col, y=target, title=f"{target} over Time")}

        return {"fig": px.line(self.df[target].dropna().reset_index(),
                               x="index", y=target, title=f"{target} trend")}

    def _chart_average(
        self, col: str | None, num: list[str], cat: list[str]
    ) -> AnalysisResult:
        target = col or num[0]
        if cat:
            top = self.df[cat[0]].value_counts().head(15).index
            avg = (self.df[self.df[cat[0]].isin(top)]
                   .groupby(cat[0])[target].mean()
                   .sort_values(ascending=False)
                   .reset_index())
            return {
                "fig": px.bar(avg, x=cat[0], y=target, title=f"Average {target} by {cat[0]}"),
                "text": f"Overall average {target}: {self.df[target].mean():.2f}",
            }
        stats = self.df[num].mean().reset_index()
        stats.columns = ["Column", "Mean"]
        return {"fig": px.bar(stats, x="Column", y="Mean", title="Mean of all numeric columns")}

    def _chart_summary(self, num: list[str], cat: list[str]) -> AnalysisResult:
        text = "\n".join([
            f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns",
            f"Numeric columns: {', '.join(num) or 'none'}",
            f"Categorical columns: {', '.join(cat) or 'none'}",
            f"Missing values: {self.df.isnull().sum().sum()}",
        ])
        if len(num) >= 2:
            fig = px.imshow(self.df[num].corr(), text_auto=".2f",
                            title="Correlation Overview", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1)
            return {"fig": fig, "text": text}
        return {"text": text}


# ── Module-level convenience wrappers (keep app.py unchanged) ─────────────────

def analyze(question: str, df: pd.DataFrame, groq_key: str | None = None) -> AnalysisResult:
    return DataAgent(df, groq_key).analyze(question)


def auto_insights(df: pd.DataFrame) -> list[AnalysisResult]:
    return DataAgent(df).auto_insights()
