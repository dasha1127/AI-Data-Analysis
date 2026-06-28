from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass(frozen=True)
class ChartConfig:
    top_n_categories: int = 10
    top_n_trend_groups: int = 8
    histogram_bins: int = 30
    insight_height: int = 400


MODEL = ModelConfig()
CHART = ChartConfig()

INTENT_PATTERNS: dict[str, list[str]] = {
    "correlation":  ["correlat", "relationship", "relate", "vs", "versus", "impact", "affect"],
    "distribution": ["distribut", "spread", "histogram", "range", "frequency"],
    "trend":        ["trend", "over time", "timeline", "change", "growth", "decline"],
    "top":          ["top", "highest", "most", "best", "largest", "biggest", "rank"],
    "bottom":       ["bottom", "lowest", "least", "worst", "smallest"],
    "compare":      ["compar", "differ", "between", "group", "categor", "by"],
    "summary":      ["summar", "overview", "describe", "statistics", "stat", "info"],
    "missing":      ["missing", "null", "nan", "empty", "incomplete"],
    "outlier":      ["outlier", "anomal", "unusual", "extreme", "weird"],
    "average":      ["average", "mean", "median", "typical"],
    "count":        ["count", "how many", "number of", "total"],
}
