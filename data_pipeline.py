"""Data preparation pipeline for TravelMind.

This script demonstrates the required data work on external files:
1. Load several raw/source CSV files from the data folder.
2. Clean missing values and normalize numeric feature columns.
3. Merge destination-level route statistics and Google-review signals.
4. Build combined text for TF-IDF/search.
5. Create simple K-Means style clusters and export cluster profiles.

The Flask app uses the generated files:
- data/unified_cleaned_destinations.csv
- data/cluster_profiles.csv
"""

from __future__ import annotations

import argparse

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "culture", "adventure", "nature", "beaches", "nightlife",
    "cuisine", "wellness", "urban", "seclusion",
]

FEATURE_LABELS = {
    "culture": "culture",
    "adventure": "adventure",
    "nature": "nature",
    "beaches": "beaches",
    "nightlife": "nightlife",
    "cuisine": "cuisine",
    "wellness": "wellness",
    "urban": "urban",
    "seclusion": "seclusion",
}

DATA_DIR = Path(__file__).parent / "data"


def read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required external data file: {path}")
    return pd.read_csv(path)


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.max() == values.min():
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - values.min()) / (values.max() - values.min())


def kmeans_labels(values: np.ndarray, clusters: int, iterations: int = 40) -> np.ndarray:
    """Small deterministic K-Means implementation, without extra dependencies."""
    rng = np.random.default_rng(42)
    if len(values) == 0:
        return np.array([], dtype=int)
    clusters = min(max(2, clusters), len(values))
    centers = values[rng.choice(len(values), clusters, replace=False)].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(iterations):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(clusters):
            members = values[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def clean_destinations(cities: pd.DataFrame) -> pd.DataFrame:
    df = cities.copy()
    df = df.drop_duplicates(subset=["city", "country"], keep="first")
    df["city"] = df["city"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["region"] = df["region"].fillna("unknown").astype(str).str.strip().str.lower()
    df["short_description"] = df["short_description"].fillna("").astype(str)
    df["budget_level"] = df["budget_level"].fillna("Mid-range")
    df["ideal_durations"] = df["ideal_durations"].fillna('["Short trip"]')

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df.get(feature, 0), errors="coerce").fillna(0).clip(0, 5)

    df["has_coordinates"] = df["latitude"].notna() & df["longitude"].notna()
    df["source_dataset"] = "worldwide_travel_cities"
    return df


def route_statistics_by_destination(routes: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    if not stats.empty:
        stats["DestinationID"] = pd.to_numeric(stats["DestinationID"], errors="coerce")
    # The provided route statistics are already aggregated by DestinationID.
    wanted = [
        "DestinationID", "route_count", "avg_route_cost", "avg_route_duration",
        "avg_satisfaction", "common_weather", "common_companions",
        "common_route_budget", "common_theme", "common_transport",
    ]
    return stats[[column for column in wanted if column in stats.columns]]


def google_review_features(reviews: pd.DataFrame) -> dict:
    numeric = reviews.drop(columns=["User"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    category_means = numeric.mean(numeric_only=True).fillna(0)
    return {
        "avg_google_interest": float(category_means.mean()),
        "google_category_signal": float(category_means.max() if len(category_means) else 0),
    }


def build_combined_text(row: pd.Series) -> str:
    feature_words = []
    for feature in FEATURES:
        rating = int(pd.to_numeric(row.get(feature, 0), errors="coerce") or 0)
        feature_words.extend([FEATURE_LABELS[feature]] * max(0, rating))
    return " ".join([
        str(row.get("city", "")),
        str(row.get("country", "")),
        str(row.get("region", "")),
        str(row.get("short_description", "")),
        str(row.get("common_theme", "")),
        " ".join(feature_words),
    ]).strip()


def add_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = df[FEATURES].to_numpy(dtype=float)
    std = values.std(axis=0)
    scaled = (values - values.mean(axis=0)) / np.where(std == 0, 1, std)
    df = df.copy()
    df["cluster"] = kmeans_labels(scaled, clusters=8)

    profiles = df.groupby("cluster").agg(
        destination_count=("city", "count"),
        avg_popularity=("popularity_score", "mean"),
        avg_google_interest=("avg_google_interest", "mean"),
        avg_satisfaction=("avg_satisfaction", "mean"),
        **{f"avg_{feature}": (feature, "mean") for feature in FEATURES},
    ).reset_index()

    labels = []
    for _, row in profiles.iterrows():
        top = sorted(FEATURES, key=lambda feature: row[f"avg_{feature}"], reverse=True)[:2]
        labels.append(" + ".join(top))
    profiles["cluster_label"] = labels
    return df, profiles


def build_unified_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    cities = clean_destinations(read_csv("worldwide_travel_cities.csv"))
    routes = read_csv("tourism_routes.csv") if (DATA_DIR / "tourism_routes.csv").exists() else pd.DataFrame()
    route_stats = route_statistics_by_destination(routes, read_csv("route_destination_statistics.csv"))
    review_signal = google_review_features(read_csv("google_review_ratings.csv"))

    df = cities.reset_index(drop=True)
    df["DestinationID"] = np.arange(1, len(df) + 1)
    df = df.merge(route_stats, on="DestinationID", how="left")

    df["route_count"] = pd.to_numeric(df.get("route_count", 0), errors="coerce").fillna(0).astype(int)
    df["avg_route_cost"] = pd.to_numeric(df.get("avg_route_cost", np.nan), errors="coerce")
    df["avg_route_duration"] = pd.to_numeric(df.get("avg_route_duration", np.nan), errors="coerce")
    df["avg_satisfaction"] = pd.to_numeric(df.get("avg_satisfaction", np.nan), errors="coerce").fillna(3.0)
    df["common_theme"] = df.get("common_theme", "").fillna("").astype(str)
    df["common_transport"] = df.get("common_transport", "").fillna("").astype(str)
    df["popularity_score"] = df[FEATURES].sum(axis=1) / len(FEATURES)
    df["avg_google_interest"] = review_signal["avg_google_interest"]
    df["google_category_signal"] = review_signal["google_category_signal"]
    df["combined_text"] = df.apply(build_combined_text, axis=1)

    df, cluster_profiles = add_clusters(df)
    return df, cluster_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and merge TravelMind external data files.")
    parser.add_argument(
        "--overwrite-app-data",
        action="store_true",
        help="Write directly to unified_cleaned_destinations.csv and cluster_profiles.csv.",
    )
    args = parser.parse_args()

    unified, cluster_profiles = build_unified_dataset()
    unified_name = "unified_cleaned_destinations.csv" if args.overwrite_app_data else "pipeline_unified_cleaned_destinations.csv"
    clusters_name = "cluster_profiles.csv" if args.overwrite_app_data else "pipeline_cluster_profiles.csv"

    unified.to_csv(DATA_DIR / unified_name, index=False)
    cluster_profiles.to_csv(DATA_DIR / clusters_name, index=False)

    summary = {
        "rows": int(len(unified)),
        "clusters": int(cluster_profiles["cluster"].nunique()),
        "external_files_used": [
            "worldwide_travel_cities.csv",
            "tourism_routes.csv",
            "route_destination_statistics.csv",
            "google_review_ratings.csv",
        ],
        "outputs": [unified_name, clusters_name],
    }
    (DATA_DIR / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


