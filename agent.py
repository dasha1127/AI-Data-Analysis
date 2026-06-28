import re
import traceback
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


INTENT_PATTERNS = {
    "correlation": ["correlat", "relationship", "relate", "vs", "versus", "impact", "affect"],
    "distribution": ["distribut", "spread", "histogram", "range", "frequency"],
    "trend": ["trend", "over time", "timeline", "change", "growth", "decline"],
    "top": ["top", "highest", "most", "best", "largest", "biggest", "rank"],
    "bottom": ["bottom", "lowest", "least", "worst", "smallest"],
    "compare": ["compar", "differ", "between", "group", "categor", "by"],
    "summary": ["summar", "overview", "describe", "statistics", "stat", "info"],
    "missing": ["missing", "null", "nan", "empty", "incomplete"],
    "outlier": ["outlier", "anomal", "unusual", "extreme", "weird"],
    "average": ["average", "mean", "median", "typical"],
    "count": ["count", "how many", "number of", "total"],
}


def detect_intent(question: str) -> str:
    q = question.lower()
    for intent, keywords in INTENT_PATTERNS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "summary"


def _extract_column(question: str, df: pd.DataFrame) -> str | None:
    q = question.lower()
    for col in df.columns:
        if col.lower() in q:
            return col
    return None


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def rule_based_analysis(question: str, df: pd.DataFrame) -> dict:
    """Fallback analysis when no LLM key is available."""
    intent = detect_intent(question)
    col = _extract_column(question, df)
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    try:
        if intent == "correlation" and len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f", title="Correlation Heatmap",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            )
            unstacked = corr.unstack()
            # Remove self-correlations and duplicate pairs
            mask = unstacked.index.get_level_values(0) != unstacked.index.get_level_values(1)
            top = unstacked[mask].sort_values(ascending=False).drop_duplicates()
            text = f"Strongest correlation: {top.index[0][0]} ↔ {top.index[0][1]} ({top.iloc[0]:.2f})"
            return {"fig": fig, "text": text}

        elif intent == "distribution":
            target = col or (num_cols[0] if num_cols else None)
            if target and target in num_cols:
                fig = px.histogram(df, x=target, nbins=30, title=f"Distribution of {target}",
                                   marginal="box")
                stats = df[target].describe()
                text = f"Mean: {stats['mean']:.2f} | Median: {df[target].median():.2f} | Std: {stats['std']:.2f}"
            else:
                target = col or (cat_cols[0] if cat_cols else df.columns[0])
                vc = df[target].value_counts().head(20)
                fig = px.bar(x=vc.index, y=vc.values, title=f"Distribution of {target}",
                             labels={"x": target, "y": "Count"})
                text = f"Most common: {vc.index[0]} ({vc.iloc[0]} times)"
            return {"fig": fig, "text": text}

        elif intent in ("top", "bottom"):
            target = col or (num_cols[0] if num_cols else None)
            if not target:
                return {"text": "Please specify a column name in your question."}
            groupby_col = cat_cols[0] if cat_cols else None
            if groupby_col and groupby_col != target:
                grp = df.groupby(groupby_col)[target].mean().reset_index()
                grp = grp.sort_values(target, ascending=(intent == "bottom")).head(10)
                fig = px.bar(grp, x=groupby_col, y=target,
                             title=f"{'Top' if intent == 'top' else 'Bottom'} 10 by {target}")
            else:
                sorted_df = df[[target]].dropna().sort_values(
                    target, ascending=(intent == "bottom")
                ).head(10)
                fig = px.bar(sorted_df, y=target, title=f"{'Top' if intent == 'top' else 'Bottom'} 10 values of {target}")
            return {"fig": fig}

        elif intent == "compare" and cat_cols and num_cols:
            cat = col if col in cat_cols else cat_cols[0]
            num = num_cols[0]
            top_cats = df[cat].value_counts().head(10).index
            filtered = df[df[cat].isin(top_cats)]
            fig = px.box(filtered, x=cat, y=num, title=f"{num} by {cat}")
            means = filtered.groupby(cat)[num].mean().sort_values(ascending=False)
            text = f"Highest avg {num}: {means.index[0]} ({means.iloc[0]:.2f})"
            return {"fig": fig, "text": text}

        elif intent == "missing":
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            miss_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
            miss_df = miss_df[miss_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
            if miss_df.empty:
                return {"text": "Great news! No missing values found in this dataset."}
            fig = px.bar(miss_df, y=miss_df.index, x="Missing %", orientation="h",
                         title="Missing Values by Column")
            text = f"Total missing: {missing.sum()} | Columns affected: {(missing > 0).sum()}"
            return {"fig": fig, "text": text}

        elif intent == "outlier" and num_cols:
            target = col or num_cols[0]
            q1, q3 = df[target].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = df[(df[target] < q1 - 1.5 * iqr) | (df[target] > q3 + 1.5 * iqr)]
            fig = px.box(df, y=target, title=f"Outlier Detection — {target}",
                         points="outliers")
            text = f"Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%) in {target}."
            return {"fig": fig, "text": text}

        elif intent == "trend" and num_cols:
            # Prefer integer year column for groupby (e.g. World Happiness)
            year_col = next((c for c in df.columns if c.lower() == "year"), None)
            # Don't use "year" itself as the metric target
            non_year_nums = [c for c in num_cols if c.lower() != "year"]
            target = (col if col and col.lower() != "year" else None) or (non_year_nums[0] if non_year_nums else num_cols[0])
            date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

            if year_col:
                group_col = next((c for c in cat_cols if c not in ("year",)), None)
                if group_col:
                    top_cats = df[group_col].value_counts().head(8).index
                    trend = df[df[group_col].isin(top_cats)].groupby([year_col, group_col])[target].mean().reset_index()
                    fig = px.line(trend, x=year_col, y=target, color=group_col,
                                  title=f"{target} by {group_col} over years", markers=True)
                else:
                    trend = df.groupby(year_col)[target].mean().reset_index()
                    fig = px.line(trend, x=year_col, y=target,
                                  title=f"Average {target} over years", markers=True)
                return {"fig": fig}
            elif date_cols:
                x_col = date_cols[0]
                try:
                    df = df.copy()
                    df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
                    trend = df.dropna(subset=[x_col]).sort_values(x_col).set_index(x_col)[target].resample("ME").mean().reset_index()
                    fig = px.line(trend, x=x_col, y=target, title=f"{target} over Time")
                    return {"fig": fig}
                except Exception:
                    pass
            fig = px.line(df[target].dropna().reset_index(), x="index", y=target, title=f"{target} trend")
            return {"fig": fig}

        elif intent == "average" and num_cols:
            target = col or num_cols[0]
            if cat_cols:
                cat = cat_cols[0]
                top_cats = df[cat].value_counts().head(15).index
                avg = df[df[cat].isin(top_cats)].groupby(cat)[target].mean().sort_values(ascending=False).reset_index()
                fig = px.bar(avg, x=cat, y=target, title=f"Average {target} by {cat}")
                text = f"Overall average {target}: {df[target].mean():.2f}"
                return {"fig": fig, "text": text}
            else:
                stats = df[num_cols].mean().reset_index()
                stats.columns = ["Column", "Mean"]
                fig = px.bar(stats, x="Column", y="Mean", title="Mean of all numeric columns")
                return {"fig": fig}

        else:  # summary
            buf = io.StringIO()
            df.describe(include="all").to_csv(buf)
            shape_text = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"
            num_text = f"Numeric columns: {', '.join(num_cols) or 'none'}"
            cat_text = f"Categorical columns: {', '.join(cat_cols) or 'none'}"
            miss_text = f"Missing values: {df.isnull().sum().sum()}"
            if num_cols:
                fig = px.imshow(df[num_cols].corr(), text_auto=".2f",
                                title="Correlation Overview", color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1)
                return {"fig": fig, "text": "\n".join([shape_text, num_text, cat_text, miss_text])}
            return {"text": "\n".join([shape_text, num_text, cat_text, miss_text])}

    except Exception as e:
        return {"text": f"Could not analyze: {str(e)}. Try rephrasing with a column name."}


def _safe_exec(code: str, df: pd.DataFrame) -> dict:
    local_ns = {
        "df": df.copy(),
        "pd": pd,
        "px": px,
        "go": go,
        "np": np,
        "fig": None,
        "result": None,
    }
    try:
        exec(code, {"__builtins__": {}}, local_ns)  # noqa: S102
    except Exception:
        exec(code, local_ns)  # retry with builtins if restricted exec fails  # noqa: S102

    out = {}
    if local_ns.get("fig") is not None:
        out["fig"] = local_ns["fig"]
    if local_ns.get("result") is not None:
        out["text"] = str(local_ns["result"])
    return out


def llm_analysis(question: str, df: pd.DataFrame, groq_key: str) -> dict:
    """LLM-powered analysis via Groq (free tier)."""
    from groq import Groq

    client = Groq(api_key=groq_key)

    schema_lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(3).tolist()
        schema_lines.append(f"  - {col} ({dtype}): e.g. {sample}")

    schema = "\n".join(schema_lines)
    shape = f"{df.shape[0]} rows × {df.shape[1]} cols"

    system_prompt = """You are a data analysis expert. Generate Python code to answer questions about a pandas DataFrame.

Rules:
- The dataframe is already loaded as `df`
- Available imports: pd (pandas), px (plotly.express), go (plotly.graph_objects), np (numpy)
- For visualizations: create a plotly figure and assign it to `fig`
- For text answers: assign the answer string to `result`
- You can set both `fig` and `result`
- Write clean, executable Python code only — no markdown fences, no explanation
- Keep the code concise and correct
"""

    user_msg = f"""Dataset info:
Shape: {shape}
Columns:
{schema}

Question: {question}

Write Python code to answer this:"""

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    code = response.choices[0].message.content.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    try:
        result = _safe_exec(code, df)
        result["code"] = code
        return result
    except Exception as e:
        return {
            "text": f"Code execution error: {e}\n\nGenerated code:\n{code}",
            "code": code,
        }


def analyze(question: str, df: pd.DataFrame, groq_key: str | None = None) -> dict:
    if groq_key:
        try:
            return llm_analysis(question, df, groq_key)
        except Exception as e:
            return rule_based_analysis(question, df)
    return rule_based_analysis(question, df)


def auto_insights(df: pd.DataFrame) -> list[dict]:
    """Generate automatic insights shown on dataset upload."""
    insights = []
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    if num_cols and len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        height=400)
        insights.append({"title": "Correlations", "fig": fig})

    if num_cols:
        sample_col = num_cols[0]
        fig = px.histogram(df, x=sample_col, nbins=30,
                           title=f"Distribution — {sample_col}", marginal="box", height=350)
        insights.append({"title": f"Distribution of {sample_col}", "fig": fig})

    if cat_cols and num_cols:
        cat, num = cat_cols[0], num_cols[0]
        top_cats = df[cat].value_counts().head(10).index
        filtered = df[df[cat].isin(top_cats)]
        fig = px.box(filtered, x=cat, y=num, title=f"{num} by {cat}", height=350)
        insights.append({"title": f"{num} by {cat}", "fig": fig})

    missing = df.isnull().sum()
    if missing.sum() > 0:
        miss_df = (missing[missing > 0] / len(df) * 100).round(1).reset_index()
        miss_df.columns = ["Column", "Missing %"]
        fig = px.bar(miss_df, x="Column", y="Missing %",
                     title="Missing Values (%)", height=300)
        insights.append({"title": "Missing Values", "fig": fig})

    return insights
