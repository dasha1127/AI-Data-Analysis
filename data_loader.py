"""
Loads and normalizes all 8 years of World Happiness Report CSVs
into a single unified DataFrame.

Unified schema:
  year, country, region, happiness_score, happiness_rank,
  gdp_per_capita, social_support, life_expectancy,
  freedom, generosity, corruption
"""

import glob
import pandas as pd

# Maps each year's raw column names → unified names
COLUMN_MAPS = {
    2015: {
        "Country":                       "country",
        "Region":                        "region",
        "Happiness Rank":                "happiness_rank",
        "Happiness Score":               "happiness_score",
        "Economy (GDP per Capita)":      "gdp_per_capita",
        "Family":                        "social_support",
        "Health (Life Expectancy)":      "life_expectancy",
        "Freedom":                       "freedom",
        "Generosity":                    "generosity",
        "Trust (Government Corruption)": "corruption",
    },
    2016: {
        "Country":                       "country",
        "Region":                        "region",
        "Happiness Rank":                "happiness_rank",
        "Happiness Score":               "happiness_score",
        "Economy (GDP per Capita)":      "gdp_per_capita",
        "Family":                        "social_support",
        "Health (Life Expectancy)":      "life_expectancy",
        "Freedom":                       "freedom",
        "Generosity":                    "generosity",
        "Trust (Government Corruption)": "corruption",
    },
    2017: {
        "Country":                        "country",
        "Happiness.Rank":                 "happiness_rank",
        "Happiness.Score":                "happiness_score",
        "Economy..GDP.per.Capita.":       "gdp_per_capita",
        "Family":                         "social_support",
        "Health..Life.Expectancy.":       "life_expectancy",
        "Freedom":                        "freedom",
        "Generosity":                     "generosity",
        "Trust..Government.Corruption.":  "corruption",
    },
    2018: {
        "Country or region":             "country",
        "Overall rank":                  "happiness_rank",
        "Score":                         "happiness_score",
        "GDP per capita":                "gdp_per_capita",
        "Social support":                "social_support",
        "Healthy life expectancy":       "life_expectancy",
        "Freedom to make life choices":  "freedom",
        "Generosity":                    "generosity",
        "Perceptions of corruption":     "corruption",
    },
    2019: {
        "Country or region":             "country",
        "Overall rank":                  "happiness_rank",
        "Score":                         "happiness_score",
        "GDP per capita":                "gdp_per_capita",
        "Social support":                "social_support",
        "Healthy life expectancy":       "life_expectancy",
        "Freedom to make life choices":  "freedom",
        "Generosity":                    "generosity",
        "Perceptions of corruption":     "corruption",
    },
    2020: {
        "Country name":                  "country",
        "Regional indicator":            "region",
        "Ladder score":                  "happiness_score",
        "Logged GDP per capita":         "gdp_per_capita",
        "Social support":                "social_support",
        "Healthy life expectancy":       "life_expectancy",
        "Freedom to make life choices":  "freedom",
        "Generosity":                    "generosity",
        "Perceptions of corruption":     "corruption",
    },
    2021: {
        "Country name":                  "country",
        "Regional indicator":            "region",
        "Ladder score":                  "happiness_score",
        "Logged GDP per capita":         "gdp_per_capita",
        "Social support":                "social_support",
        "Healthy life expectancy":       "life_expectancy",
        "Freedom to make life choices":  "freedom",
        "Generosity":                    "generosity",
        "Perceptions of corruption":     "corruption",
    },
    2022: {
        "Country":                                   "country",
        "RANK":                                      "happiness_rank",
        "Happiness score":                           "happiness_score",
        "Explained by: GDP per capita":              "gdp_per_capita",
        "Explained by: Social support":              "social_support",
        "Explained by: Healthy life expectancy":     "life_expectancy",
        "Explained by: Freedom to make life choices":"freedom",
        "Explained by: Generosity":                  "generosity",
        "Explained by: Perceptions of corruption":   "corruption",
    },
}

UNIFIED_COLS = [
    "year", "country", "region", "happiness_score", "happiness_rank",
    "gdp_per_capita", "social_support", "life_expectancy",
    "freedom", "generosity", "corruption",
]


def _load_year(path: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_map = COLUMN_MAPS[year]
    df = df.rename(columns=col_map)
    df = df[[c for c in col_map.values() if c in df.columns]]
    df["year"] = year
    # Add missing columns as NaN
    for col in UNIFIED_COLS:
        if col not in df.columns:
            df[col] = None
    return df[UNIFIED_COLS]


def load_combined(data_dir: str = "data") -> pd.DataFrame:
    """Load and merge all years into one clean DataFrame."""
    frames = []
    for year, _ in COLUMN_MAPS.items():
        path = f"{data_dir}/{year}.csv"
        try:
            frames.append(_load_year(path, year))
        except FileNotFoundError:
            pass

    df = pd.concat(frames, ignore_index=True)

    # Clean up types
    df["year"] = df["year"].astype(int)
    df["happiness_score"] = pd.to_numeric(df["happiness_score"], errors="coerce")
    df["happiness_rank"] = pd.to_numeric(df["happiness_rank"], errors="coerce")
    df["gdp_per_capita"] = pd.to_numeric(df["gdp_per_capita"], errors="coerce")
    df["social_support"] = pd.to_numeric(df["social_support"], errors="coerce")
    df["life_expectancy"] = pd.to_numeric(df["life_expectancy"], errors="coerce")
    df["freedom"] = pd.to_numeric(df["freedom"], errors="coerce")
    df["generosity"] = pd.to_numeric(df["generosity"], errors="coerce")
    df["corruption"] = pd.to_numeric(df["corruption"], errors="coerce")
    df["country"] = df["country"].str.strip()

    # Forward-fill region for countries that appear in 2015/16/20/21
    region_map = (
        df[df["region"].notna()]
        .groupby("country")["region"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    df["region"] = df["country"].map(region_map)

    return df.sort_values(["year", "happiness_rank"]).reset_index(drop=True)


def most_improved(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Countries with biggest happiness score improvement from first to last year."""
    first = df.groupby("country")["year"].min().reset_index()
    last = df.groupby("country")["year"].max().reset_index()

    first_scores = df.merge(first, on=["country", "year"])[["country", "happiness_score"]].rename(
        columns={"happiness_score": "first_score"}
    )
    last_scores = df.merge(last, on=["country", "year"])[["country", "happiness_score"]].rename(
        columns={"happiness_score": "last_score"}
    )

    merged = first_scores.merge(last_scores, on="country")
    merged["change"] = merged["last_score"] - merged["first_score"]
    return merged.sort_values("change", ascending=False).head(top_n)
