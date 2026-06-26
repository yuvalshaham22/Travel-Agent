import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


FEATURES = [
    "culture", "adventure", "nature", "beaches", "nightlife",
    "cuisine", "wellness", "urban", "seclusion",
]

FEATURE_TERMS = {
    "culture": ["culture", "cultural", "cultured", "history", "museum", "תרבות", "היסטוריה", "מוזיאון"],
    "adventure": ["adventure", "extreme", "hiking", "הרפתקה", "אקסטרים", "טיולים"],
    "nature": ["nature", "mountain", "green", "view", "views", "scenery", "landscape", "טבע", "הרים", "ירוק", "נוף", "נופים"],
    "beaches": ["beach", "beaches", "sea", "coast", "חוף", "חופים", "ליד הים"],
    "nightlife": ["nightlife", "clubs", "party", "parties", "חיי לילה", "מסיבות", "מועדונים"],
    "cuisine": ["food", "cuisine", "restaurant", "אוכל", "קולינריה", "מסעדות"],
    "wellness": ["wellness", "spa", "relax", "רוגע", "ספא", "להירגע"],
    "urban": ["urban", "city", "shopping", "עירוני", "עיר", "קניות"],
    "seclusion": ["quiet", "secluded", "remote", "שקט", "מבודד", "פרטיות"],
}

BUDGET_TERMS = {
    "Budget": ["cheap", "low cost", "זול", "חסכוני", "תקציב נמוך"],
    "Mid-range": ["mid-range", "medium budget", "תקציב בינוני", "בינוני"],
    "Luxury": ["luxury", "premium", "expensive", "יוקרתי", "פאר", "תקציב גבוה"],
}

MONTH_TERMS = {
    1: ["january", "jan", "ינואר"], 2: ["february", "feb", "פברואר"],
    3: ["march", "mar", "מרץ"], 4: ["april", "apr", "אפריל"],
    5: ["may", "מאי"], 6: ["june", "jun", "יוני"],
    7: ["july", "jul", "יולי"], 8: ["august", "aug", "אוגוסט"],
    9: ["september", "sep", "ספטמבר"], 10: ["october", "oct", "אוקטובר"],
    11: ["november", "nov", "נובמבר"], 12: ["december", "dec", "דצמבר"],
}

HEBREW_FEATURES = {
    "culture": "תרבות", "adventure": "הרפתקאות", "nature": "טבע",
    "beaches": "חופים", "nightlife": "חיי לילה", "cuisine": "אוכל",
    "wellness": "רוגע", "urban": "עירוניות", "seclusion": "שקט",
}
ENGLISH_FEATURES = {
    "culture": "culture", "adventure": "adventure", "nature": "nature",
    "beaches": "beaches", "nightlife": "nightlife", "cuisine": "food",
    "wellness": "wellness", "urban": "city life", "seclusion": "quiet",
}

BUDGET_ORDER = {"Budget": 1, "Mid-range": 2, "Luxury": 3}
DEFAULT_BUDGET_AMOUNTS = {
    "ILS": {"Budget": 3000, "Mid-range": 7000, "Luxury": 14000},
    "USD": {"Budget": 900, "Mid-range": 2200, "Luxury": 4500},
    "EUR": {"Budget": 800, "Mid-range": 2000, "Luxury": 4000},
}
DAILY_BUDGET_THRESHOLDS = {
    "ILS": (800, 1800),
    "USD": (220, 500),
    "EUR": (200, 450),
}
HEBREW_THOUSANDS = {
    "\u05d0\u05dc\u05e3": 1000, "\u05d0\u05dc\u05e4\u05d9\u05d9\u05dd": 2000,
    "\u05e9\u05dc\u05d5\u05e9\u05ea \u05d0\u05dc\u05e4\u05d9\u05dd": 3000,
    "\u05e9\u05dc\u05d5\u05e9\u05d4 \u05d0\u05dc\u05e4\u05d9\u05dd": 3000,
    "\u05d0\u05e8\u05d1\u05e2\u05ea \u05d0\u05dc\u05e4\u05d9\u05dd": 4000,
    "\u05d0\u05e8\u05d1\u05e2\u05d4 \u05d0\u05dc\u05e4\u05d9\u05dd": 4000,
    "\u05d7\u05de\u05e9\u05ea \u05d0\u05dc\u05e4\u05d9\u05dd": 5000,
    "\u05d7\u05de\u05d9\u05e9\u05d4 \u05d0\u05dc\u05e4\u05d9\u05dd": 5000,
    "\u05e9\u05e9\u05ea \u05d0\u05dc\u05e4\u05d9\u05dd": 6000,
    "\u05e9\u05d9\u05e9\u05d4 \u05d0\u05dc\u05e4\u05d9\u05dd": 6000,
    "\u05e9\u05d1\u05e2\u05ea \u05d0\u05dc\u05e4\u05d9\u05dd": 7000,
    "\u05e9\u05d1\u05e2\u05d4 \u05d0\u05dc\u05e4\u05d9\u05dd": 7000,
    "\u05e9\u05de\u05d5\u05e0\u05ea \u05d0\u05dc\u05e4\u05d9\u05dd": 8000,
    "\u05e9\u05de\u05d5\u05e0\u05d4 \u05d0\u05dc\u05e4\u05d9\u05dd": 8000,
    "\u05ea\u05e9\u05e2\u05ea \u05d0\u05dc\u05e4\u05d9\u05dd": 9000,
    "\u05ea\u05e9\u05e2\u05d4 \u05d0\u05dc\u05e4\u05d9\u05dd": 9000,
    "\u05e2\u05e9\u05e8\u05ea \u05d0\u05dc\u05e4\u05d9\u05dd": 10000,
    "\u05e2\u05e9\u05e8\u05d4 \u05d0\u05dc\u05e4\u05d9\u05dd": 10000,
}
ENGLISH_THOUSANDS = {
    word: value * 1000 for value, word in enumerate(
        ["", "one thousand", "two thousand", "three thousand", "four thousand",
         "five thousand", "six thousand", "seven thousand", "eight thousand",
         "nine thousand", "ten thousand"]
    ) if word
}
CITY_ALIASES = {"סרנדה": "Saranda", "טירנה": "Tirana", "sarandë": "Saranda"}
COUNTRY_ALIASES = {
    "יוון": "Greece", "איטליה": "Italy", "ספרד": "Spain", "צרפת": "France",
    "פורטוגל": "Portugal", "אלבניה": "Albania", "ישראל": "Israel",
    "יפן": "Japan", "תאילנד": "Thailand", "הודו": "India",
    "ארצות הברית": "United States", "ארהב": "United States",
}
SKIP_TERMS = ["לא משנה", "פתוח להצעות", "תפתיע", "any", "doesn't matter", "surprise me"]


def kmeans_labels(values: np.ndarray, clusters: int, iterations: int = 40) -> np.ndarray:
    """Small deterministic K-Means implementation to keep deployment lightweight."""
    rng = np.random.default_rng(42)
    centers = values[rng.choice(len(values), clusters, replace=False)].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(iterations):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(clusters):
            members = values[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z\u0590-\u05FF]{2,}", str(text).lower())


def parse_budget_amount(text: str, language: Optional[str] = None, allow_plain: bool = False):
    normalized = str(text or "").lower().replace(",", "").strip()
    currency = None
    if re.search(r"\u20aa|\u05e9\u05e7\u05dc|\u05e9\u05f4\u05d7|\u05e9\u05d7|ils", normalized):
        currency = "ILS"
    elif re.search(r"\$|usd|dollars?", normalized):
        currency = "USD"
    elif re.search(r"\u20ac|eur|euros?", normalized):
        currency = "EUR"

    for phrase, amount in {**HEBREW_THOUSANDS, **ENGLISH_THOUSANDS}.items():
        if phrase in normalized:
            inferred_currency = None if allow_plain and not currency else currency or ("ILS" if language == "he" else "USD")
            return amount, inferred_currency

    matches = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*([km])?\b", normalized)
    if not matches:
        return None, None
    amounts = []
    for value, suffix in matches:
        amount = float(value)
        if suffix == "k":
            amount *= 1000
        elif suffix == "m":
            amount *= 1000000
        amounts.append(amount)
    amount = max(amounts)
    if amount < 100:
        return None, None
    inferred_currency = None if allow_plain and not currency else currency or ("ILS" if language == "he" else "USD")
    return int(round(amount)), inferred_currency


def budget_level_from_amount(amount: Optional[int], currency: Optional[str], days: Optional[int]):
    if not amount:
        return None
    currency = currency if currency in DAILY_BUDGET_THRESHOLDS else "ILS"
    daily_amount = amount / max(1, days or 5)
    low, high = DAILY_BUDGET_THRESHOLDS[currency]
    if daily_amount <= low:
        return "Budget"
    if daily_amount <= high:
        return "Mid-range"
    return "Luxury"


def format_budget(amount: Optional[int], currency: Optional[str], language: str = "he"):
    if not amount:
        return "\u05dc\u05d0 \u05e6\u05d5\u05d9\u05df" if language == "he" else "not specified"
    symbols = {"ILS": "\u20aa", "USD": "$", "EUR": "\u20ac"}
    symbol = symbols.get(currency, currency or "")
    if language == "en" and currency in {"USD", "EUR"}:
        return f"{symbol}{amount:,.0f}"
    return f"{amount:,.0f} {symbol}".strip()


def build_tfidf(documents: List[str], max_features: int = 700):
    tokenized = [tokenize(document) for document in documents]
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    vocabulary = [
        term for term, _ in document_frequency.most_common(max_features)
        if document_frequency[term] >= 2
    ]
    lookup = {term: index for index, term in enumerate(vocabulary)}
    idf = np.array([
        math.log((1 + len(documents)) / (1 + document_frequency[term])) + 1
        for term in vocabulary
    ], dtype=np.float32)
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float32)
    for row_index, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        total = max(1, len(tokens))
        for term, count in counts.items():
            if term in lookup:
                matrix[row_index, lookup[term]] = (count / total) * idf[lookup[term]]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms), lookup, idf


class TravelPlanner:
    def __init__(self, data_path: Optional[str] = None):
        default_path = Path(__file__).parent / "data" / "unified_cleaned_destinations.csv"
        path = Path(data_path or default_path)
        if not path.exists():
            path = Path(__file__).parent / "data" / "worldwide_travel_cities.csv"
        self.df = pd.read_csv(path)
        self.external_cluster_profiles = self._load_external_cluster_profiles()
        self._add_curated_destinations()
        for feature in FEATURES:
            self.df[feature] = pd.to_numeric(self.df[feature], errors="coerce").fillna(0)
        self.df["budget_rank"] = self.df["budget_level"].map(BUDGET_ORDER).fillna(2)
        self.df["climate"] = self.df["avg_temp_monthly"].apply(self._parse_json)
        self.df["durations"] = self.df["ideal_durations"].apply(self._parse_json)
        self.df["combined_text"] = self.df.get("combined_text", self.df["short_description"]).fillna("").astype(str)
        self.tfidf_matrix, self.tfidf_lookup, self.tfidf_idf = build_tfidf(self.df["combined_text"].tolist())

        if "cluster" in self.df.columns:
            self.df["traveler_cluster"] = pd.to_numeric(self.df["cluster"], errors="coerce").fillna(0).astype(int)
        else:
            values = self.df[FEATURES].to_numpy(dtype=float)
            std = values.std(axis=0)
            scaled = (values - values.mean(axis=0)) / np.where(std == 0, 1, std)
            cluster_count = min(7, max(2, len(self.df) // 80))
            self.df["traveler_cluster"] = kmeans_labels(scaled, cluster_count)
        self.cluster_profiles = self.df.groupby("traveler_cluster")[FEATURES].mean()

    def _load_external_cluster_profiles(self) -> Dict[int, Dict[str, str]]:
        """Load a second external data file with prepared cluster-level profiles."""
        profiles_path = Path(__file__).parent / "data" / "cluster_profiles.csv"
        if not profiles_path.exists():
            return {}
        profiles_df = pd.read_csv(profiles_path).fillna("")
        profiles = {}
        for _, row in profiles_df.iterrows():
            try:
                cluster_id = int(row.get("cluster"))
            except (TypeError, ValueError):
                continue
            profiles[cluster_id] = {
                "cluster_label": str(row.get("cluster_label", "")).strip(),
                "destination_count": str(row.get("destination_count", "")).strip(),
                "avg_popularity": str(row.get("avg_popularity", "")).strip(),
                "avg_satisfaction": str(row.get("avg_satisfaction", "")).strip(),
            }
        return profiles

    def _external_cluster_profile_for(self, cluster_id) -> Dict[str, str]:
        try:
            return self.external_cluster_profiles.get(int(cluster_id), {})
        except (TypeError, ValueError):
            return {}

    def _add_curated_destinations(self):
        """Add important demo destinations missing from the source dataset."""
        if "Saranda" in set(self.df["city"]):
            return
        climate = {
            str(month): {"avg": avg, "min": avg - 4, "max": avg + 5}
            for month, avg in enumerate([10, 11, 13, 16, 20, 25, 28, 28, 24, 19, 15, 11], 1)
        }
        row = {
            "id": "curated-saranda-albania", "city": "Saranda", "country": "Albania",
            "region": "europe",
            "short_description": "A compact Albanian Riviera town known for beaches, seaside restaurants, and lively summer evenings.",
            "latitude": 39.8756, "longitude": 20.0053,
            "avg_temp_monthly": json.dumps(climate), "ideal_durations": '["Weekend","Short trip"]',
            "budget_level": "Budget", "culture": 3, "adventure": 3, "nature": 4,
            "beaches": 5, "nightlife": 4, "cuisine": 4, "wellness": 4, "urban": 2, "seclusion": 2,
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    @staticmethod
    def _parse_json(value):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _contains(text: str, terms: List[str]) -> bool:
        return any(term in text for term in terms)

    def parse_request(self, text: str) -> Dict:
        normalized = str(text or "").lower().strip()
        language = "he" if re.search(r"[\u0590-\u05FF]", normalized) else ("en" if re.search(r"[a-zA-Z]", normalized) else None)
        preferences = {
            feature: 5 if self._contains(normalized, terms) else 0
            for feature, terms in FEATURE_TERMS.items()
        }
        budget = next(
            (level for level, terms in BUDGET_TERMS.items() if self._contains(normalized, terms)),
            None,
        )
        budget_amount, budget_currency = parse_budget_amount(normalized, language)
        explicit_month = next(
            (number for number, terms in MONTH_TERMS.items() if self._contains(normalized, terms)), None
        )
        days_match = re.search(r"(\d+)\s*(?:days?|ימים?|לילות?)", normalized)
        days = min(30, max(1, int(days_match.group(1)))) if days_match else None
        if budget and not budget_amount:
            budget_currency = "ILS" if language == "he" else "USD"
            budget_amount = DEFAULT_BUDGET_AMOUNTS[budget_currency][budget]
        cities = [
            city for city in self.df["city"].astype(str).tolist()
            if re.search(rf"(?<!\w){re.escape(city.lower())}(?!\w)", normalized)
        ]
        cities.extend(canonical for alias, canonical in CITY_ALIASES.items() if alias in normalized)
        countries = [
            country for country in self.df["country"].dropna().astype(str).unique().tolist()
            if re.search(rf"(?<!\w){re.escape(country.lower())}(?!\w)", normalized)
        ]
        countries.extend(canonical for alias, canonical in COUNTRY_ALIASES.items() if alias in normalized)
        return {
            "text": normalized,
            "preferences": preferences,
            "budget": budget or budget_level_from_amount(budget_amount, budget_currency, days),
            "budget_amount": budget_amount,
            "budget_currency": budget_currency,
            "month": explicit_month,
            "days": days,
            "cities": list(dict.fromkeys(cities)),
            "countries": list(dict.fromkeys(countries)),
            "language": language,
            "skip": any(term in normalized for term in SKIP_TERMS),
        }

    def _score(self, request: Dict) -> pd.DataFrame:
        scored = self.df.copy()
        active = {k: v for k, v in request["preferences"].items() if v}
        if active:
            weights = np.array([active.get(feature, 0) for feature in FEATURES], dtype=float)
            weights /= weights.sum()
            scored["preference_score"] = (scored[FEATURES].values / 5.0).dot(weights) * 55
        else:
            scored["preference_score"] = scored[FEATURES].mean(axis=1) / 5.0 * 45

        desired_budget = BUDGET_ORDER.get(request["budget"], 2)
        difference = (scored["budget_rank"] - desired_budget).abs()
        scored["budget_score"] = np.where(difference == 0, 15, np.where(difference == 1, 7, 0))

        month = str(request.get("month") or datetime.now().month)
        scored["month_temp"] = scored["climate"].apply(
            lambda climate: float(climate.get(month, {}).get("avg", np.nan))
        )
        wants_beach = bool(active.get("beaches"))
        target_temp = 27 if wants_beach else 21
        scored["weather_score"] = (10 - (scored["month_temp"] - target_temp).abs() * 0.65).clip(0, 10)
        scored["weather_score"] = scored["weather_score"].fillna(5)
        query_vector = self._query_tfidf(request["text"])
        scored["text_similarity"] = (self.tfidf_matrix @ query_vector).clip(0, 1)
        scored["text_score"] = scored["text_similarity"] * 15
        satisfaction = pd.to_numeric(scored.get("avg_satisfaction", 3), errors="coerce").fillna(3)
        google_interest = pd.to_numeric(scored.get("google_interest_score", 2.5), errors="coerce").fillna(2.5)
        scored["evidence_score"] = ((satisfaction / 5) * 3 + (google_interest / 5) * 2).clip(0, 5)
        scored["match_score"] = (
            scored["preference_score"] + scored["budget_score"] + scored["weather_score"]
            + scored["text_score"] + scored["evidence_score"]
        ).round(1)
        return scored.sort_values("match_score", ascending=False)

    def _query_tfidf(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self.tfidf_lookup), dtype=np.float32)
        tokens = tokenize(text)
        counts = Counter(tokens)
        total = max(1, len(tokens))
        for term, count in counts.items():
            index = self.tfidf_lookup.get(term)
            if index is not None:
                vector[index] = (count / total) * self.tfidf_idf[index]
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def recommend(self, request: Dict, limit: int = 3) -> List[Dict]:
        scored = self._score(request)
        if request["cities"]:
            chosen = scored[scored["city"].isin(request["cities"])].copy()
            if len(chosen):
                scored = chosen
        elif request.get("countries"):
            chosen = scored[scored["country"].isin(request["countries"])].copy()
            if len(chosen):
                scored = chosen
        return [self._destination_record(row, request) for _, row in scored.head(limit).iterrows()]

    def analyze_patterns(self, request: Dict, destination: Dict) -> Dict:
        """Create simple data-mining insights for trends and anomalies.

        This supports the assignment requirement: AI/data tooling for identifying
        patterns, trends, added value and anomalies in the destination data. The
        algorithm compares the top matching destinations against the whole dataset
        and flags features where the selected destination is unusually strong.
        """
        language = request.get("language") or "he"
        labels = HEBREW_FEATURES if language == "he" else ENGLISH_FEATURES
        scored = self._score(request).head(50)
        global_means = self.df[FEATURES].mean(numeric_only=True)
        top_means = scored[FEATURES].mean(numeric_only=True)
        lifts = (top_means - global_means).sort_values(ascending=False)

        trends = []
        for feature, lift in lifts.head(3).items():
            if lift > 0.25:
                if language == "he":
                    trends.append(f"בקרב היעדים הדומים לבקשה שלך בולטת מגמה של {labels[feature]} גבוה מהממוצע.")
                else:
                    trends.append(f"Among the destinations similar to your request, {labels[feature]} is trending above the dataset average.")
        if not trends:
            trends.append("הבקשה שלך מתאימה לפרופיל מאוזן יחסית, בלי מגמה אחת חריגה במיוחד." if language == "he" else "Your request fits a relatively balanced profile without one dominant trend.")

        row = self.df[
            (self.df["city"].astype(str) == str(destination.get("city")))
            & (self.df["country"].astype(str) == str(destination.get("country")))
        ]
        anomalies = []
        if len(row):
            row = row.iloc[0]
            for feature in FEATURES:
                std = float(self.df[feature].std() or 0)
                if std <= 0:
                    continue
                z_score = (float(row[feature]) - float(global_means[feature])) / std
                if z_score >= 1.15 and float(row[feature]) >= 4:
                    if language == "he":
                        anomalies.append(f"{labels[feature]} ביעד גבוה משמעותית מהממוצע בדאטה ({int(row[feature])}/5).")
                    else:
                        anomalies.append(f"{labels[feature]} is unusually high compared with the dataset average ({int(row[feature])}/5).")
                if len(anomalies) >= 3:
                    break
        if not anomalies:
            anomalies.append("לא זוהתה חריגה שלילית מרכזית ביעד שנבחר ביחס להעדפות שהוזנו." if language == "he" else "No major negative anomaly was detected for the selected destination based on your preferences.")

        added_value = []
        if destination.get("text_similarity") is not None:
            if language == "he":
                added_value.append(f"התאמה טקסטואלית לבקשה: {destination.get('text_similarity')} — מבוסס TF-IDF על תיאור היעדים.")
            else:
                added_value.append(f"Text similarity to your request: {destination.get('text_similarity')} — based on TF-IDF over destination descriptions.")
        if destination.get("avg_satisfaction"):
            if language == "he":
                added_value.append(f"שביעות רצון היסטורית בנתונים: {destination.get('avg_satisfaction')}/5.")
            else:
                added_value.append(f"Historical satisfaction in the data: {destination.get('avg_satisfaction')}/5.")

        return {
            "title": "תובנות AI מהדאטה" if language == "he" else "AI data insights",
            "trends_title": "מגמות" if language == "he" else "Trends",
            "anomalies_title": "אנומליות" if language == "he" else "Anomalies",
            "added_value_title": "ערך נוסף" if language == "he" else "Added value",
            "trends": trends[:3],
            "anomalies": anomalies[:3],
            "added_value": added_value[:3],
        }

    def _destination_record(self, row: pd.Series, request: Dict) -> Dict:
        def number(name, default=0.0):
            value = pd.to_numeric(row.get(name, default), errors="coerce")
            return default if pd.isna(value) else float(value)

        def text(name):
            value = row.get(name, "")
            value = "" if pd.isna(value) else str(value)
            return "" if value == "Not applicable" else value

        active = [feature for feature, value in request["preferences"].items() if value]
        strengths = sorted(
            FEATURES,
            key=lambda feature: (feature in active, row[feature]),
            reverse=True,
        )[:3]
        climate = row["climate"].get(str(request.get("month") or datetime.now().month), {})
        labels = HEBREW_FEATURES if (request.get("language") or "he") == "he" else ENGLISH_FEATURES
        active_reasons = sorted(active, key=lambda feature: row[feature], reverse=True)[:3]
        reason_parts = [f"{labels[feature]} {int(row[feature])}/5" for feature in active_reasons]
        if climate.get("avg") is not None:
            reason_parts.append(f"{climate['avg']}°C")
        data_evidence = [
            f"{labels[feature]}: {int(row[feature])}/5" for feature in active_reasons
        ]
        if climate.get("avg") is not None:
            data_evidence.append(
                f"{'אקלים חודשי בדאטה' if request.get('language') == 'he' else 'Monthly climate in data'}: {climate['avg']}°C"
            )
        data_evidence.append(
            f"{'רמת תקציב בדאטה' if request.get('language') == 'he' else 'Budget level in data'}: {row['budget_level']}"
        )
        if number("route_count"):
            data_evidence.append(
                f"{'מסלולים היסטוריים' if request.get('language') == 'he' else 'Historical routes'}: "
                f"{int(number('route_count'))}"
            )
        supplemental_info = (
            "המסלול המוצע נבנה על בסיס החוזקות וההתאמות שבדאטה. שמות הפעילויות הם הצעה כללית של הסוכן ויש לאמת שעות, מחירים וזמינות."
            if request.get("language") == "he" else
            "The itinerary is generated from the data-backed strengths and matches. Activity wording is general guidance and times, prices, and availability should be verified."
        )
        return {
            "city": row["city"],
            "country": row["country"],
            "region": row["region"],
            "description": row["short_description"],
            "source_dataset": text("source_dataset"),
            "budget": row["budget_level"],
            "score": float(row["match_score"]),
            "temperature": climate,
            "strengths": [{"name": key, "label": labels[key], "rating": int(row[key])} for key in strengths],
            "match_reason": ", ".join(reason_parts),
            "data_evidence": data_evidence,
            "supplemental_info": supplemental_info,
            "cluster": int(row["traveler_cluster"]),
            "cluster_label": text("cluster_label"),
            "external_cluster_profile": self._external_cluster_profile_for(row["traveler_cluster"]),
            "text_similarity": round(number("text_similarity"), 3),
            "route_count": int(number("route_count")),
            "avg_route_cost": round(number("avg_route_cost"), 1),
            "avg_satisfaction": round(number("avg_satisfaction"), 2),
            "common_theme": text("common_theme"),
            "common_transport": text("common_transport"),
            "similar_destinations": [
                text(f"similar_destination_{rank}")
                for rank in range(1, 4) if text(f"similar_destination_{rank}")
            ],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "has_coordinates": bool(row.get("has_coordinates", True)),
        }

    def build_itinerary(
        self,
        destination: Dict,
        days: int,
        language: str = "he",
        route_style: Optional[str] = None,
        pace: Optional[str] = None,
    ) -> List[Dict]:
        labels = [item["label"] for item in destination["strengths"]]
        activity_note = {
            "relaxed": "with generous free time" if language == "en" else "עם זמן חופשי ומעט עומס",
            "balanced": "with a balanced pace" if language == "en" else "בקצב מאוזן",
            "intensive": "with a fuller schedule" if language == "en" else "בקצב אינטנסיבי יותר",
        }.get(pace or "balanced")
        if language == "en":
            if route_style == "cultural":
                plans = [
                    ("Cultural introduction", f"Museums, historic streets, and {labels[0]} {activity_note}"),
                    ("Local heritage", f"A guided-style route through landmarks, markets, and food stops"),
                    ("Neighborhood day", f"Explore a characterful district using {destination['common_transport'] or 'local transport'}"),
                    ("Flexible cultural day", "Choose a day trip, performance, or deeper museum visit"),
                ]
            elif route_style == "scenic":
                plans = [
                    ("Scenic arrival", f"Viewpoints, waterfronts, or green areas focused on {labels[0]} {activity_note}"),
                    ("Nature route", f"A day route around {labels[1]} with photo stops and rest time"),
                    ("Local landscape", f"Combine {labels[2]} with an easy outdoor experience"),
                    ("Flexible scenic day", "Choose between a nature day trip, beach time, or a slower local route"),
                ]
            else:
                plans = [
                    ("Meet the destination", f"A central tour combining {labels[0]} and an introduction to the area {activity_note}"),
                    ("Your travel style", f"A {labels[1]}-focused experience. Suggested transport: {destination['common_transport'] or 'based on availability'}"),
                    ("Local experience", f"Combine {labels[2]} with a recommended local place"),
                    ("Flexible day", "Choose between a day trip, rest, or returning to a favorite place"),
                ]
        else:
            if route_style == "cultural":
                plans = [
                    ("היכרות תרבותית", f"מוזיאונים, רחובות היסטוריים ו{labels[0]} {activity_note}"),
                    ("מורשת מקומית", "מסלול בין אתרים מרכזיים, שווקים ונקודות אוכל מקומיות"),
                    ("יום שכונות", f"העמקה באזור עם אופי מקומי בעזרת {destination['common_transport'] or 'תחבורה זמינה'}"),
                    ("יום תרבות גמיש", "בחירה בין טיול יום, מופע, סיור מודרך או מוזיאון נוסף"),
                ]
            elif route_style == "scenic":
                plans = [
                    ("היכרות נופית", f"תצפיות, טיילות או אזורים ירוקים סביב {labels[0]} {activity_note}"),
                    ("מסלול טבע", f"יום סביב {labels[1]} עם עצירות צילום וזמן מנוחה"),
                    ("נוף מקומי", f"שילוב {labels[2]} עם חוויה חיצונית קלה"),
                    ("יום נופי גמיש", "בחירה בין טיול יום בטבע, חוף, או מסלול מקומי רגוע"),
                ]
            else:
                plans = [
                    ("היכרות עם היעד", f"סיור מרכזי שמשלב {labels[0]} והיכרות עם האזור {activity_note}"),
                    ("יום מותאם לאופי המטייל", f"חוויה ממוקדת {labels[1]}. תחבורה מומלצת: {destination['common_transport'] or 'לפי זמינות'}"),
                    ("החוויה המקומית", f"שילוב {labels[2]} ומקום מקומי מומלץ"),
                    ("יום גמיש", "בחירה בין טיול יום, מנוחה או העמקה במקום שאהבתם"),
                ]
        return [
            {
                "day": day,
                "title": plans[(day - 1) % len(plans)][0],
                "activity": plans[(day - 1) % len(plans)][1],
                "basis": destination["strengths"][(day - 1) % len(destination["strengths"])]["label"],
            }
            for day in range(1, days + 1)
        ]

    @staticmethod
    def live_weather(destination: Dict) -> Optional[Dict]:
        if not destination.get("has_coordinates"):
            return None
        params = urlencode({
            "latitude": destination["latitude"],
            "longitude": destination["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        })
        try:
            with urlopen(f"https://api.open-meteo.com/v1/forecast?{params}", timeout=2) as response:
                current = json.loads(response.read().decode("utf-8")).get("current", {})
            return {
                "temperature": current.get("temperature_2m"),
                "wind": current.get("wind_speed_10m"),
                "source": "Open-Meteo",
            }
        except Exception:
            return None

    def answer(self, text: str, fetch_live_weather: bool = False) -> Dict:
        request = self.parse_request(text)
        request["days"] = request["days"] or 3
        request["month"] = request["month"] or datetime.now().month
        return self.answer_request(request, fetch_live_weather=fetch_live_weather)

    def answer_request(self, request: Dict, fetch_live_weather: bool = False) -> Dict:
        destinations = self.recommend(request, limit=max(3, len(request["cities"])))
        best = destinations[0]
        live = self.live_weather(best) if fetch_live_weather else None
        comparison = ""
        language = request.get("language") or "he"
        if len(request["cities"]) >= 2 and len(destinations) >= 2:
            comparison = (
                f"לפי ההעדפות שזיהיתי, {destinations[0]['city']} עדיפה על "
                f"{destinations[1]['city']} ({destinations[0]['score']} מול "
                f"{destinations[1]['score']} נקודות התאמה)."
            ) if language == "he" else (
                f"Based on your preferences, {destinations[0]['city']} is a better match than "
                f"{destinations[1]['city']} ({destinations[0]['score']} vs. {destinations[1]['score']})."
            )
        active_labels = [
            (HEBREW_FEATURES if language == "he" else ENGLISH_FEATURES)[key]
            for key, value in request["preferences"].items() if value
        ]
        summary = comparison or (
            f"היעד המתאים ביותר הוא {best['city']}, {best['country']} עם "
            f"{best['score']} נקודות התאמה. הסיבה המרכזית: {best['match_reason']}."
        ) if language == "he" else comparison or (
            f"The best-matching destination is {best['city']}, {best['country']} "
            f"with a match score of {best['score']}. Main data signals: {best['match_reason']}."
        )
        return {
            "type": "recommendation",
            "language": language,
            "reply": summary,
            "understood": {
                "preferences": active_labels or (["פתוח להצעות"] if language == "he" else ["open to suggestions"]),
                "budget": format_budget(request.get("budget_amount"), request.get("budget_currency"), language),
                "budget_level": request["budget"] or ("לא סווג" if language == "he" else "not classified"),
                "landscape": request.get("landscape") or ("לא צוין" if language == "he" else "not specified"),
                "route_style": request.get("route_style") or ("מאוזן" if language == "he" else "balanced"),
                "pace": request.get("pace") or ("מאוזן" if language == "he" else "balanced"),
                "days": request["days"],
                "month": request["month"],
                "countries": request.get("countries", []),
            },
            "destinations": destinations,
            "data_insights": self.analyze_patterns(request, best),
            "itinerary": self.build_itinerary(
                best, request["days"], language, request.get("route_style"), request.get("pace")
            ),
            "live_weather": live,
            "information_policy": (
                "ההמלצות והדירוגים חושבו קודם מהדאטה המאוחד. מידע כללי שנוצר על ידי הסוכן מוצג בנפרד ואינו מחליף בדיקת מידע עדכני."
                if language == "he" else
                "Recommendations and rankings are calculated from the unified data first. General agent-generated guidance is shown separately and does not replace current-information checks."
            ),
            "follow_up_suggestions": (
                ["בנה לי מסלול תרבותי", "בנה לי מסלול נופי", "שנה לקצב רגוע"]
                if language == "he" else
                ["Build a cultural route", "Build a scenic route", "Change to a relaxed pace"]
            ),
            "limitations": (
                "המחירים והאקלים הם אינדיקציה. יש לבדוק טיסות, זמינות ואזהרות מסע לפני הזמנה."
                if language == "he" else
                "Prices and climate are indicative. Check flights, availability, and travel advisories before booking."
            ),
        }






