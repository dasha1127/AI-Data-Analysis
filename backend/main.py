"""
FastAPI backend — serves the data analysis API.

Endpoints:
  GET  /datasets          → list available datasets
  POST /upload            → upload a CSV, returns dataset_id
  POST /analyze           → answer a question about a dataset
  GET  /insights/{dataset}→ auto-generated insight charts
  GET  /happiness/trends  → happiness-specific trend charts
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
_ENV_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
import plotly.express as px
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Make sure parent directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import DataAgent
from data_loader import load_combined, most_improved

app = FastAPI(title="DataMind AI", version="1.0.0")

# Allow the browser (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend folder as static files
FRONTEND = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


# ── In-memory dataset store ───────────────────────────────────────────────────

_store: dict[str, pd.DataFrame] = {}


def _get_df(dataset: str) -> pd.DataFrame:
    if dataset in _store:
        return _store[dataset]
    if dataset == "happiness":
        df = load_combined()
        _store["happiness"] = df
        return df
    if dataset == "tips":
        return px.data.tips()
    if dataset == "iris":
        return px.data.iris()
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")


def _fig_to_dict(fig: Any) -> dict:
    return json.loads(fig.to_json())


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str
    dataset: str
    groq_key: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(FRONTEND / "index.html"))


@app.get("/datasets")
def list_datasets():
    built_in = ["happiness", "tips", "iris"]
    uploaded = [k for k in _store if k not in built_in]
    return {"datasets": built_in + uploaded}


@app.post("/upload")
async def upload_csv(file: UploadFile):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")
    dataset_id = Path(file.filename or "upload").stem
    _store[dataset_id] = df
    return {
        "dataset_id": dataset_id,
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": df.columns.tolist(),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    df = _get_df(req.dataset)
    key = req.groq_key or _ENV_GROQ_KEY or None
    agent = DataAgent(df, key)
    result = agent.analyze(req.question)
    return {
        "fig":  _fig_to_dict(result["fig"]) if result.get("fig") else None,
        "text": result.get("text"),
        "code": result.get("code"),
    }


@app.get("/insights/{dataset}")
def insights(dataset: str):
    df = _get_df(dataset)
    raw = DataAgent(df).auto_insights()
    return [{"title": i["title"], "fig": _fig_to_dict(i["fig"])} for i in raw]


@app.get("/happiness/trends")
def happiness_trends():
    df = _get_df("happiness")
    latest = df["year"].max()

    avg = df.groupby("year")["happiness_score"].mean().reset_index()
    top10 = df[df["year"] == latest].nlargest(10, "happiness_score")
    region = (df[df["region"].notna()]
              .groupby(["year", "region"])["happiness_score"].mean().reset_index())
    gdp = df[df["year"] == latest].dropna(subset=["gdp_per_capita", "happiness_score"])
    improved = most_improved(df, top_n=10)

    return {
        "global_trend":  _fig_to_dict(px.line(avg, x="year", y="happiness_score",
                                               title="Global Average Happiness (2015–2022)",
                                               markers=True, template="plotly_dark")),
        "top10":         _fig_to_dict(px.bar(top10, x="happiness_score", y="country",
                                              orientation="h",
                                              title=f"Top 10 Happiest Countries ({latest})",
                                              color="happiness_score",
                                              color_continuous_scale="Blues",
                                              template="plotly_dark")),
        "by_region":     _fig_to_dict(px.line(region, x="year", y="happiness_score",
                                               color="region", markers=True,
                                               title="Happiness by Region Over Time",
                                               template="plotly_dark")),
        "gdp_scatter":   _fig_to_dict(px.scatter(gdp, x="gdp_per_capita", y="happiness_score",
                                                   color="region", hover_name="country",
                                                   trendline="ols",
                                                   title=f"GDP vs Happiness ({latest})",
                                                   template="plotly_dark")),
        "most_improved": _fig_to_dict(px.bar(improved, x="change", y="country",
                                              orientation="h",
                                              title="Most Improved Countries",
                                              color="change",
                                              color_continuous_scale="Greens",
                                              template="plotly_dark")),
    }


@app.get("/schema/{dataset}")
def schema(dataset: str):
    df = _get_df(dataset)
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": [
            {"name": c, "dtype": str(df[c].dtype)}
            for c in df.columns
        ],
    }
