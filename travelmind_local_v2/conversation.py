from datetime import datetime
import re
from difflib import get_close_matches
from typing import Dict
from urllib.parse import quote_plus

from planner import FEATURES, TravelPlanner, budget_level_from_amount, parse_budget_amount
from gemini_service import analyze_free_travel_chat, build_gemini_recommendations, build_search_url


# Fixed questionnaire removed: Gemini now manages the open conversation.

GREETINGS = {
    "he": ["היי", "הייי", "שלום", "אהלן", "הי"],
    "en": ["hi", "hello", "hey"],
}


# These are the core data points required by the assignment before the system chooses a destination.
# Gemini can ask them naturally, but the deterministic recommender should not run until they are known.
REQUIRED_FREE_FIELDS = ["experience", "travel_party", "budget", "days", "month", "pace"]

TYPO_LEXICON = [
    "טיול", "חופשה", "יעד", "חופים", "חוף", "רוגע", "נוף", "נופים", "טבע",
    "תרבות", "היסטוריה", "מוזיאונים", "אטרקציות", "מסעדות", "אוכל", "קולינריה",
    "חיי", "לילה", "עירוניות", "בילויים", "תקציב", "שקלים", "ימים", "זוג",
    "חברים", "משפחה", "לבד", "רגוע", "מאוזן", "אינטנסיבי", "ינואר", "פברואר",
    "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר",
    "נובמבר", "דצמבר", "יוון", "איטליה", "ספרד", "צרפת", "פורטוגל", "אלבניה",
    "תאילנד", "יפן", "travel", "vacation", "trip", "destination", "beach", "beaches",
    "culture", "history", "nature", "budget", "days", "couple", "friends", "family",
    "relaxed", "balanced", "intensive", "nightlife",
]

COMMON_TYPO_REPLACEMENTS = {
    "חופימ": "חופים", "חופעם": "חופים", "חופיים": "חופים", "חופין": "חופים",
    "תרבוט": "תרבות", "תרבוץ": "תרבות", "תרבותי": "תרבות",
    "תצקיב": "תקציב", "תקציבב": "תקציב", "שקלחם": "שקלים", "שח": "₪",
    "ימימ": "ימים", "יומימ": "ימים", "זוגי": "זוג", "איננטנסיבי": "אינטנסיבי",
    "רגועע": "רגוע", "מאוזנ": "מאוזן", "מסעדוצ": "מסעדות", "מוזאונים": "מוזיאונים",
    "אטרקציוץ": "אטרקציות", "חיי לילהה": "חיי לילה", "באלי": "בא לי",
    # Common month typos / keyboard slips. These are important because the agent
    # often asks for the month as a one-word answer, so Gemini or the local
    # parser must not get stuck on a small typo such as "רפריל".
    "רפריל": "אפריל", "אפרל": "אפריל", "אפרייל": "אפריל", "אפרילל": "אפריל",
    "ינוארר": "ינואר", "פברוארר": "פברואר", "פבואר": "פברואר", "מרצ": "מרץ",
    "יוניי": "יוני", "יוליי": "יולי", "אוגוסטט": "אוגוסט", "ספטממבר": "ספטמבר",
    "ספטמברר": "ספטמבר", "אוקטוברר": "אוקטובר", "נובמברר": "נובמבר", "דצמברר": "דצמבר",
}


def _guess_language(text: str, default: str = "he") -> str:
    if re.search(r"[\u0590-\u05FF]", str(text or "")):
        return "he"
    if re.search(r"[a-zA-Z]", str(text or "")):
        return "en"
    return default


def _correct_common_typos(text: str) -> str:
    """Lightweight typo normalization for local fallback when Gemini is unavailable."""
    corrected = str(text or "")
    lowered = corrected.lower()
    for wrong, right in COMMON_TYPO_REPLACEMENTS.items():
        if wrong in lowered:
            corrected = re.sub(re.escape(wrong), right, corrected, flags=re.IGNORECASE)
            lowered = corrected.lower()

    def replace_token(match):
        token = match.group(0)
        if len(token) < 4:
            return token
        options = get_close_matches(token.lower(), TYPO_LEXICON, n=1, cutoff=0.78)
        if options and options[0] != token.lower():
            return options[0]
        return token

    return re.sub(r"[a-zA-Z\u0590-\u05FF]{4,}", replace_token, corrected)


def _append_history(state: Dict, role: str, text: str) -> None:
    history = state.get("history") or []
    clean_text = str(text or "").strip()
    if clean_text:
        history.append({"role": role, "text": clean_text})
    state["history"] = history[-12:]


def _merge_list_unique(existing, incoming):
    if isinstance(existing, str):
        existing = [existing]
    if isinstance(incoming, str):
        incoming = [incoming]
    if not isinstance(existing, (list, tuple, set)):
        existing = []
    if not isinstance(incoming, (list, tuple, set)):
        incoming = []
    merged = []
    for item in (existing or []) + (incoming or []):
        value = str(item or "").strip()
        if value and value not in merged:
            merged.append(value)
    return merged


def _safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _merge_ai_extraction(profile: Dict, ai_result: Dict, original_message: str) -> None:
    if not ai_result or not ai_result.get("ok"):
        return

    language = ai_result.get("language")
    if language in {"he", "en"}:
        profile["language"] = language

    corrected = str(ai_result.get("corrected_user_text") or "").strip()
    if corrected and corrected.lower() != str(original_message or "").strip().lower():
        profile["corrected_last_message"] = corrected
        if corrected not in profile.get("text_parts", []):
            profile["text_parts"].append(corrected)

    extracted = ai_result.get("extracted") or {}
    if not isinstance(extracted, dict):
        return

    profile["countries"] = _merge_list_unique(profile.get("countries"), extracted.get("countries"))
    profile["cities"] = _merge_list_unique(profile.get("cities"), extracted.get("cities"))

    for field in [
        "experience_type", "travel_party", "pace", "avoid", "travel_goal",
        "landscape", "route_style", "budget_currency", "budget_level",
    ]:
        value = extracted.get(field)
        if value not in (None, "", [], {}):
            if field == "budget_level":
                profile["budget"] = value
            else:
                profile[field] = value

    month = _safe_int(extracted.get("month"))
    if month and 1 <= month <= 12:
        profile["month"] = month

    days = _safe_int(extracted.get("days"))
    if days:
        profile["days"] = min(14, max(1, days))

    budget_amount = _safe_int(extracted.get("budget_amount"))
    if budget_amount:
        profile["budget_amount"] = budget_amount
        if extracted.get("budget_currency"):
            profile["budget_currency"] = extracted.get("budget_currency")

    preferences = extracted.get("preferences") or {}
    if isinstance(preferences, dict):
        for feature in FEATURES:
            value = _safe_int(preferences.get(feature))
            if value:
                profile["preferences"][feature] = max(profile["preferences"].get(feature, 0), min(5, value))

    if profile.get("budget_amount"):
        profile["budget"] = budget_level_from_amount(
            profile.get("budget_amount"), profile.get("budget_currency"), profile.get("days")
        ) or profile.get("budget")


def _merge_free_message(planner: TravelPlanner, state: Dict, message: str) -> Dict:
    """Merge a natural free-text turn.

    Gemini is responsible for understanding which question was asked and what the
    user meant. The local parser only adds lightweight backup signals from the
    text; it does not drive a fixed questionnaire or decide the next question.
    """
    corrected = _correct_common_typos(message)
    state.pop("pending_field", None)
    state.pop("pending_free_field", None)
    state = merge_message(planner, state, corrected)
    profile = state["profile"]

    # In free chat, a plain number like "5000" can be a budget. Smaller one-word
    # numbers such as "12" are left for Gemini to interpret from conversation context
    # (days/month/etc.) instead of being forced into a local field.
    amount, currency = parse_budget_amount(corrected, profile.get("language", "he"), allow_plain=True)
    if amount:
        profile["budget_amount"] = amount
        profile["budget_currency"] = currency
        profile["budget"] = budget_level_from_amount(amount, currency, profile.get("days"))

    month = _parse_month_from_free_text(corrected)
    if month:
        profile["month"] = month

    if re.search(r"\b(רגוע|מאוזן|אינטנסיבי|relaxed|balanced|intensive)\b", corrected.lower()):
        _apply_pace(profile, corrected)
    _apply_travel_party(profile, corrected)
    _apply_avoid(profile, corrected)
    return state


def _free_missing_fields(profile: Dict) -> list:
    """Return the core fields that still need to be collected before recommending."""
    missing = []
    if not any(profile.get("preferences", {}).values()) and not profile.get("experience_type") and not profile.get("travel_goal"):
        missing.append("experience")
    if not profile.get("travel_party"):
        missing.append("travel_party")
    if not profile.get("budget_amount") and not profile.get("budget"):
        missing.append("budget")
    if not profile.get("days"):
        missing.append("days")
    if not profile.get("month"):
        missing.append("month")
    if not profile.get("pace"):
        missing.append("pace")
    return [field for field in REQUIRED_FREE_FIELDS if field in missing]

def _filter_missing_by_profile(missing_fields, profile: Dict) -> list:
    """Keep Gemini's missing-fields list aligned with what was already extracted.

    This is only a safety gate before running the deterministic recommender; it is
    not used to choose a scripted question. The next question still comes from Gemini.
    """
    local_missing = set(_free_missing_fields(profile))
    ordered = [str(field) for field in (missing_fields or []) if str(field) in local_missing]
    for field in _free_missing_fields(profile):
        if field not in ordered:
            ordered.append(field)
    return ordered


def _gemini_unavailable_reply(language: str = "he", gemini_result: Dict | None = None) -> str:
    gemini_result = gemini_result or {}
    missing_key = gemini_result.get("error") == "missing_api_key" or not gemini_result.get("had_api_key", False)
    tried_models = gemini_result.get("tried_models") or []
    model_hint = f" ניסיתי את המודלים: {', '.join(tried_models)}." if tried_models else ""

    if language == "en":
        if missing_key:
            return (
                "Gemini is not connected because GEMINI_API_KEY is missing. "
                "Set GEMINI_API_KEY in the same PowerShell window and run the app again."
            )
        return (
            "Gemini API key exists, but the request failed. This is usually caused by an old or unavailable GEMINI_MODEL. "
            "Use GEMINI_MODEL=gemini-2.5-flash-lite or remove GEMINI_MODEL and run again." + model_hint
        )

    if missing_key:
        return (
            "כרגע Gemini לא מחובר כי חסר GEMINI_API_KEY. "
            "צריך להגדיר את המפתח באותו חלון PowerShell שמריץ את app.py."
        )
    return (
        "מצאתי GEMINI_API_KEY, אבל הקריאה ל־Gemini נכשלה. לרוב זה קורה בגלל מודל ישן או לא זמין. "
        "מחקי את GEMINI_MODEL או הגדירי אותו ל־gemini-2.5-flash-lite ואז תריצי מחדש." + model_hint
    )


def _soft_fallback_followup(profile: Dict, missing: list) -> str:
    """A non-questionnaire fallback used only when Gemini is unavailable or returns an empty reply."""
    language = profile.get("language", "he")
    if language == "en":
        return "Tell me a little more about the trip you imagine, especially anything missing such as timing, budget, pace or who is traveling."
    return "ספרי לי עוד קצת על הטיול שאת מדמיינת — למשל מתי, עם מי, תקציב, קצב או סגנון — ואני אמשיך לדייק מתוך השיחה."


def _apply_recommendation_defaults(profile: Dict) -> None:
    if not any(profile.get("preferences", {}).values()):
        profile["experience_type"] = profile.get("experience_type") or "balanced_mix"
        profile["travel_goal"] = profile.get("travel_goal") or "mixed"
        profile["route_style"] = profile.get("route_style") or "balanced"
        profile["landscape"] = profile.get("landscape") or "varied"
        for feature in ["culture", "nature", "cuisine", "urban"]:
            profile["preferences"][feature] = max(profile["preferences"].get(feature, 0), 3)
    profile["days"] = profile.get("days") or 5
    profile["month"] = profile.get("month") or datetime.now().month
    profile["pace"] = profile.get("pace") or "balanced"
    profile["travel_party"] = profile.get("travel_party") or "flexible"
    profile["route_style"] = profile.get("route_style") or "balanced"
    profile["landscape"] = profile.get("landscape") or "varied"
    profile["budget"] = profile.get("budget") or budget_level_from_amount(
        profile.get("budget_amount"), profile.get("budget_currency"), profile.get("days")
    ) or "Mid-range"


def _profile_summary_for_reply(profile: Dict) -> str:
    language = profile.get("language", "he")
    if language == "en":
        pieces = []
        if profile.get("days"):
            pieces.append(f"{profile['days']} days")
        if profile.get("budget_amount"):
            pieces.append(f"budget {profile['budget_amount']}")
        if profile.get("month"):
            pieces.append(f"month {profile['month']}")
        return ", ".join(pieces)
    month_names_he = ["", "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]
    pieces = []
    if profile.get("days"):
        pieces.append(f"{profile['days']} ימים")
    if profile.get("budget_amount"):
        pieces.append(f"תקציב {profile['budget_amount']:,} {profile.get('budget_currency') or ''}".strip())
    if profile.get("month"):
        pieces.append(month_names_he[int(profile["month"])])
    return ", ".join(pieces)


def empty_profile(language="he"):
    return {
        "language": language,
        "countries": [],
        "cities": [],
        "days": None,
        "month": None,
        "budget": None,
        "budget_amount": None,
        "budget_currency": None,
        "experience_type": None,
        "travel_party": None,
        "avoid": None,
        "travel_goal": None,
        "landscape": None,
        "route_style": None,
        "pace": None,
        "preferences": {feature: 0 for feature in FEATURES},
        "text_parts": [],
        "open_location": True,
        "open_month": False,
    }


def greeting_language(message: str):
    normalized = re.sub(r"[!?.\s,]+", "", message.lower())
    loose = message.lower().strip()
    for language, greetings in GREETINGS.items():
        for greeting in greetings:
            if normalized == greeting or loose.startswith(greeting + " "):
                return language
    return None


def _looks_like_new_trip_request(message: str) -> bool:
    normalized = message.lower().strip()
    new_trip_terms = [
        "תמליץ לי", "המלצה לטיול", "טיול שאוכל", "אני רוצה טיול", "בא לי טיול",
        "תבחר לי יעד", "לאן כדאי", "תן לי המלצות לטיול", "תכנן לי טיול",
        "recommend", "suggest a trip", "trip where i can", "plan a trip"
    ]
    return any(term in normalized for term in new_trip_terms)


def _is_out_of_scope(message: str) -> bool:
    """Return True for questions that are clearly not related to travel planning."""
    normalized = message.lower().strip()

    travel_terms = [
        "טיול", "לטייל", "נסיעה", "חופשה", "יעד", "מדינה", "עיר", "מסלול",
        "מלון", "לינה", "שדה תעופה", "טיסה", "מוזיאון", "מסעד", "חוף", "חופים",
        "נוף", "טבע", "תרבות", "היסטוריה", "גלישה", "גלים", "אטרקציות",
        "travel", "trip", "vacation", "destination", "itinerary", "hotel",
        "airport", "flight", "museum", "restaurant", "beach", "nature",
        "culture", "history", "surf", "attractions"
    ]

    blocked_terms = [
        "מתכון", "עוגה", "אוכל להכין", "לבשל", "אפייה", "לאפות", "בא לי לישון", "עייפה", "עייף", "אני עייפה", "אני עייף",
        "שיעורי בית", "תרגיל", "sql", "קוד", "פייתון", "java", "erp",
        "קורות חיים", "מייל", "רפואה", "תרופה", "כואב", "בדיחה",
        "recipe", "cake", "cook", "bake", "sleep", "sleepy", "tired", "homework", "code", "resume",
        "email", "medicine", "joke"
    ]

    has_travel = any(term in normalized for term in travel_terms)
    has_blocked = any(term in normalized for term in blocked_terms)

    return has_blocked and not has_travel


def _is_casual_non_travel(message: str) -> bool:
    normalized = message.lower().strip()
    casual_phrases = [
        "בא לי לישון", "אני עייפה", "אני עייף", "עייפה", "עייף",
        "משעמם לי", "אין לי כוח", "בא לי לאכול", "i want to sleep", "i am tired", "i'm tired"
    ]
    return any(phrase in normalized for phrase in casual_phrases)


def _out_of_scope_response(language: str = "he"):
    if language == "en":
        return {
            "type": "question",
            "language": "en",
            "reply": (
                "I’m TravelMind, a travel-planning and destination recommendation agent. "
                "This question is outside my area of responsibility. "
                "I can help you choose a destination, plan a trip, match a budget, or find museums, restaurants and attractions."
            ),
            "suggestions": ["Choose a destination", "Plan a cultural trip", "Find restaurants in my destination"],
            "destinations": [],
            "itinerary": [],
        }

    return {
        "type": "question",
        "language": "he",
        "reply": (
            "אני TravelMind, סוכן לתכנון טיולים והמלצות יעד. "
            "השאלה הזו לא חלק מתחום האחריות שלי. "
            "אשמח לעזור לך לבחור יעד, להתאים טיול לתקציב, או למצוא מוזיאונים, מסעדות ואטרקציות ביעד."
        ),
        "suggestions": ["תבחר לי יעד", "תכנן לי טיול תרבותי", "מצא לי מסעדות ביעד"],
        "destinations": [],
        "itinerary": [],
    }


MONTH_NAMES = {
    "ינואר": 1, "פברואר": 2, "מרץ": 3, "אפריל": 4, "מאי": 5, "יוני": 6,
    "יולי": 7, "אוגוסט": 8, "ספטמבר": 9, "אוקטובר": 10, "נובמבר": 11, "דצמבר": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

SEASON_TO_MONTH = {
    "אביב": 4, "קיץ": 7, "סתיו": 10, "חורף": 1,
    "spring": 4, "summer": 7, "autumn": 10, "fall": 10, "winter": 1,
}


def _word_tokens(text: str):
    return re.findall(r"[a-zA-Z\u0590-\u05FF]+", str(text or "").lower())


def _fuzzy_month_from_text(text: str, allow_fuzzy: bool = True):
    """Return a month even when the user has a small typo, e.g. רפריל -> אפריל.

    The fuzzy check is intentionally limited to month/season words, so it will not
    turn every unknown word into a travel preference. This is mostly used after the
    agent asks a direct month question, but it also helps in free text like
    "אני רוצה לטוס ברפריל".
    """
    normalized = _correct_common_typos(text).lower().strip()

    for name, value in MONTH_NAMES.items():
        if name in normalized:
            return value

    for name, value in SEASON_TO_MONTH.items():
        if name in normalized:
            return value

    if not allow_fuzzy:
        return None

    candidates = list(MONTH_NAMES.keys()) + list(SEASON_TO_MONTH.keys())
    for token in _word_tokens(normalized):
        # Three-letter words are more collision-prone, so require a stricter score.
        cutoff = 0.84 if len(token) <= 3 else 0.74
        matches = get_close_matches(token, candidates, n=1, cutoff=cutoff)
        if matches:
            match = matches[0]
            return MONTH_NAMES.get(match) or SEASON_TO_MONTH.get(match)
    return None


def _parse_month_answer(text: str):
    normalized = _correct_common_typos(text).lower().strip()
    number = re.search(r"\b(1[0-2]|[1-9])\b", normalized)
    if number:
        return int(number.group(1))
    return _fuzzy_month_from_text(normalized, allow_fuzzy=True)


def _parse_month_from_free_text(text: str):
    normalized = _correct_common_typos(str(text or "")).lower().strip()
    month = _fuzzy_month_from_text(normalized, allow_fuzzy=True)
    if month:
        return month
    explicit = re.search(r"(?:חודש|month|בחודש|מספר חודש)\s*(1[0-2]|[1-9])", normalized)
    if explicit:
        return int(explicit.group(1))
    return None


def _apply_experience(profile: Dict, text: str) -> None:
    normalized = text.lower().strip()
    if any(term in normalized for term in ["תרבות", "היסטוריה", "אווירה מקומית", "culture", "history", "local atmosphere"]):
        profile["experience_type"] = "culture_local"
        profile["travel_goal"] = "culture"
        profile["route_style"] = "cultural"
        profile["landscape"] = "urban_culture"
        profile["preferences"]["culture"] = 5
        profile["preferences"]["urban"] = max(profile["preferences"].get("urban", 0), 4)
        profile["preferences"]["cuisine"] = max(profile["preferences"].get("cuisine", 0), 3)

    elif any(term in normalized for term in ["חופים", "חוף", "רוגע", "נוף", "beaches", "relaxation", "views"]):
        profile["experience_type"] = "beach_relax"
        profile["travel_goal"] = "beach"
        profile["route_style"] = "beaches"
        profile["landscape"] = "tropical"
        profile["preferences"]["beaches"] = 5
        profile["preferences"]["wellness"] = 5
        profile["preferences"]["nature"] = max(profile["preferences"].get("nature", 0), 4)

    elif any(term in normalized for term in ["טבע", "נופים", "הרפתקאות", "nature", "scenery", "adventure"]):
        profile["experience_type"] = "nature_adventure"
        profile["travel_goal"] = "nature"
        profile["route_style"] = "scenic"
        profile["landscape"] = "varied"
        profile["preferences"]["nature"] = 5
        profile["preferences"]["adventure"] = 5

    elif any(term in normalized for term in ["חיי לילה", "בילויים", "עירוניות", "nightlife", "entertainment", "city"]):
        profile["experience_type"] = "nightlife_urban"
        profile["travel_goal"] = "nightlife"
        profile["route_style"] = "urban"
        profile["landscape"] = "urban_culture"
        profile["preferences"]["nightlife"] = 5
        profile["preferences"]["urban"] = 5
        profile["preferences"]["cuisine"] = max(profile["preferences"].get("cuisine", 0), 4)

    elif any(term in normalized for term in ["שילוב", "מאוזן", "balanced", "mix"]):
        profile["experience_type"] = "balanced_mix"
        profile["travel_goal"] = "mixed"
        profile["route_style"] = "balanced"
        profile["landscape"] = "varied"
        for feature in ["culture", "nature", "cuisine", "urban"]:
            profile["preferences"][feature] = max(profile["preferences"].get(feature, 0), 4)


def _apply_travel_party(profile: Dict, text: str) -> None:
    normalized = text.lower().strip()
    if any(term in normalized for term in ["לבד", "solo"]):
        profile["travel_party"] = "solo"
        profile["preferences"]["urban"] = max(profile["preferences"].get("urban", 0), 3)
    elif any(term in normalized for term in ["זוג", "couple"]):
        profile["travel_party"] = "couple"
        profile["preferences"]["cuisine"] = max(profile["preferences"].get("cuisine", 0), 4)
        profile["preferences"]["wellness"] = max(profile["preferences"].get("wellness", 0), 3)
    elif any(term in normalized for term in ["חברות", "חברים", "חברים/ות", "friends"]):
        profile["travel_party"] = "friends"
        profile["preferences"]["cuisine"] = max(profile["preferences"].get("cuisine", 0), 4)
        profile["preferences"]["nightlife"] = max(profile["preferences"].get("nightlife", 0), 3)
        profile["preferences"]["urban"] = max(profile["preferences"].get("urban", 0), 3)
    elif any(term in normalized for term in ["משפחה", "family"]):
        profile["travel_party"] = "family"
        profile["preferences"]["culture"] = max(profile["preferences"].get("culture", 0), 3)
        profile["preferences"]["nature"] = max(profile["preferences"].get("nature", 0), 3)
    elif any(term in normalized for term in ["לא משנה", "פתוח", "flexible", "does not matter"]):
        profile["travel_party"] = "flexible"


def _apply_pace(profile: Dict, text: str) -> None:
    normalized = text.lower().strip()
    if any(term in normalized for term in ["רגוע", "מעט", "זמן חופשי", "relaxed", "fewer"]):
        profile["pace"] = "relaxed"
    elif any(term in normalized for term in ["אינטנסיבי", "להספיק", "intensive", "see as much"]):
        profile["pace"] = "intensive"
    elif any(term in normalized for term in ["מאוזן", "בלי עומס", "balanced"]):
        profile["pace"] = "balanced"


def _apply_avoid(profile: Dict, text: str) -> None:
    normalized = text.lower().strip()
    if any(term in normalized for term in ["יקרים", "יקר", "expensive"]):
        profile["avoid"] = "expensive"
        # Soften luxury-oriented features a bit.
        profile["preferences"]["wellness"] = min(profile["preferences"].get("wellness", 0), 3)
    elif any(term in normalized for term in ["הליכות", "walking"]):
        profile["avoid"] = "too_much_walking"
    elif any(term in normalized for term in ["מוזיאונים", "museums"]):
        profile["avoid"] = "too_many_museums"
        profile["preferences"]["culture"] = min(profile["preferences"].get("culture", 0), 3)
    elif any(term in normalized for term in ["חיי לילה", "רעש", "nightlife", "noise"]):
        profile["avoid"] = "nightlife_noise"
        profile["preferences"]["nightlife"] = 0
        profile["preferences"]["seclusion"] = max(profile["preferences"].get("seclusion", 0), 3)
    elif any(term in normalized for term in ["אין", "nothing", "no special"]):
        profile["avoid"] = "none"


def _enrich_special_travel_intent(profile: Dict, message: str) -> None:
    normalized = message.lower()
    if any(term in normalized for term in ["גלישה", "לגלוש", "גלים", "surf", "surfing", "waves"]):
        profile["travel_goal"] = "surfing"
        profile["experience_type"] = profile.get("experience_type") or "beach_adventure"
        profile["route_style"] = profile.get("route_style") or "beaches"
        profile["landscape"] = profile.get("landscape") or "tropical"
        profile["preferences"]["beaches"] = 5
        profile["preferences"]["adventure"] = max(profile["preferences"].get("adventure", 0), 5)
        profile["preferences"]["nature"] = max(profile["preferences"].get("nature", 0), 4)


def merge_message(planner: TravelPlanner, state: Dict, message: str) -> Dict:
    parsed = planner.parse_request(message)
    profile = state.get("profile") or empty_profile(parsed["language"] or "he")
    pending = state.get("pending_field")

    if parsed["language"]:
        profile["language"] = parsed["language"]

    profile["text_parts"].append(message)

    if parsed["countries"]:
        profile["countries"] = parsed["countries"]
    if parsed["cities"]:
        profile["cities"] = parsed["cities"]
    if parsed["month"]:
        profile["month"] = parsed["month"]

    # Use parsed days/budget from any natural user message.
    if parsed["days"]:
        profile["days"] = parsed["days"]
    if parsed["budget"]:
        profile["budget"] = parsed["budget"]
    if parsed.get("budget_amount"):
        profile["budget_amount"] = parsed["budget_amount"]
        profile["budget_currency"] = parsed["budget_currency"]

    # First message can still add useful preference signals.
    for feature, value in parsed["preferences"].items():
        if value:
            profile["preferences"][feature] = max(profile["preferences"].get(feature, 0), value)

    # Optional field-specific interpretation remains for legacy sessions; Gemini is the main conversation manager.
    if pending == "experience":
        _apply_experience(profile, message)
    elif pending == "travel_party":
        _apply_travel_party(profile, message)
    elif pending == "budget":
        amount, currency = parse_budget_amount(message, profile["language"], allow_plain=True)
        if amount:
            profile["budget_amount"] = amount
            profile["budget_currency"] = currency
    elif pending == "days":
        number = re.search(r"\d+", message)
        if number:
            profile["days"] = min(14, max(1, int(number.group())))
    elif pending == "month":
        month = _parse_month_answer(message)
        if month:
            profile["month"] = month
    elif pending == "pace":
        _apply_pace(profile, message)
    elif pending == "avoid":
        _apply_avoid(profile, message)

    # Also detect important intent from free text.
    _apply_experience(profile, message)
    _enrich_special_travel_intent(profile, message)

    if profile.get("budget_amount"):
        profile["budget"] = budget_level_from_amount(
            profile["budget_amount"], profile["budget_currency"], profile["days"]
        )

    state["profile"] = profile
    return state


def _search_url(query: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(query)}"


def _extract_places_from_itinerary(itinerary: list):
    places = []
    seen = set()
    for day in itinerary or []:
        text = str(day.get("activity") or "")
        matches = re.findall(r"(?:בוקר|צהריים|אחר הצהריים|ערב):\s*([^\.]+)", text)
        for match in matches:
            place = match.strip()
            key = place.lower()
            if place and key not in seen:
                seen.add(key)
                places.append(place)
    return places


def _fallback_pack(destination: dict, profile: dict, itinerary: list) -> dict:
    city = destination.get("city")
    country = destination.get("country")
    places = _extract_places_from_itinerary(itinerary)
    if not places:
        places = [city]

    def make_item(name, reason, kind):
        return {
            "name": name,
            "reason": reason,
            "kind": kind,
            "url": _search_url(f"{name} {city} {country}"),
            "source_query": f"{name} {city} {country}",
        }

    culture = [make_item(p, "אתר תרבות/מורשת מרכזי ביעד", "culture") for p in places[:5]]
    food_candidates = [
        p for p in places
        if any(word in p.lower() for word in ["restaurant", "café", "cafe", "market", "halles", "food", "מסעד", "קפה", "שוק"])
    ]
    if not food_candidates:
        food_candidates = [f"מסעדות מומלצות ב{city}", f"שוק אוכל מרכזי ב{city}", f"בתי קפה מקומיים ב{city}"]
    food = [make_item(p, "מקום אוכל מומלץ ביעד", "food") for p in food_candidates[:5]]

    return {
        "culture": culture,
        "food": food,
    }


def _clean_match_reason_for_ui(reason: str) -> str:
    """Remove weather from the match reason so climate is shown only once in the card."""
    reason = str(reason or "").strip()
    reason = re.sub(r",?\s*\d+(?:\.\d+)?°C\.?", "", reason).strip()
    return reason.strip(" ,")


def _clean_user_reply(reply: str, language: str = "he") -> str:
    if language == "he":
        # Convert: "היעד המתאים ביותר הוא Moorea, French Polynesia עם 79.4 נקודות התאמה. הסיבה המרכזית: ..."
        match = re.search(r"היעד המתאים ביותר הוא (.+?) עם [\\d\\.]+ נקודות התאמה\\.", reply)
        if match:
            return f"היעד שהכי מתאים למה שסיפרת לי הוא {match.group(1)}."
    else:
        match = re.search(r"The best-matching destination is (.+?) with a match score of [\\d\\.]+\\.", reply)
        if match:
            return f"The destination that best fits what you shared is {match.group(1)}."
    return reply


def _personalized_reason_he(profile: Dict) -> str:
    experience_labels = {
        "culture_local": "חוויה תרבותית, היסטורית ואווירה מקומית",
        "beach_relax": "חופים, רוגע ונוף",
        "nature_adventure": "טבע, נופים והרפתקאות",
        "nightlife_urban": "חיי לילה, בילויים ועירוניות",
        "balanced_mix": "שילוב מאוזן של כמה דברים",
        "beach_adventure": "חופים ופעילות",
    }
    party_labels = {
        "solo": "לבד",
        "couple": "זוג",
        "friends": "חברים/ות",
        "family": "משפחה",
        "flexible": "פתוח",
    }
    pace_labels = {
        "relaxed": "קצב רגוע",
        "balanced": "קצב מאוזן",
        "intensive": "קצב אינטנסיבי",
    }
    avoid_labels = {
        "expensive": "להימנע ממקומות יקרים מדי",
        "too_much_walking": "להימנע מיותר מדי הליכות",
        "too_many_museums": "להימנע מיותר מדי מוזיאונים",
        "nightlife_noise": "להימנע מחיי לילה ורעש",
        "none": "אין מגבלה מיוחדת",
    }

    parts = []
    if profile.get("experience_type"):
        parts.append(experience_labels.get(profile["experience_type"], profile["experience_type"]))
    if profile.get("travel_party"):
        parts.append(f"נוסעת עם {party_labels.get(profile['travel_party'], profile['travel_party'])}")
    if profile.get("pace"):
        parts.append(pace_labels.get(profile["pace"], profile["pace"]))
    if profile.get("month"):
        month_names_he = ["", "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]
        parts.append(f"חודש נסיעה: {month_names_he[int(profile['month'])]}")
    if not parts:
        return ""
    return " לפי מה שסיפרת לי — " + ", ".join(parts) + "."


def _personalized_reason_en(profile: Dict) -> str:
    parts = []
    if profile.get("experience_type"):
        parts.append(profile["experience_type"].replace("_", " "))
    if profile.get("travel_party"):
        parts.append(f"traveling with {profile['travel_party']}")
    if profile.get("pace"):
        parts.append(f"{profile['pace']} pace")
    if profile.get("avoid"):
        parts.append(f"avoid: {profile['avoid']}")
    if not parts:
        return ""
    return " Based on your personal preferences — " + ", ".join(parts) + "."


def _build_recommendation_response(planner: TravelPlanner, profile: Dict, fetch_live_weather=True):
    _apply_recommendation_defaults(profile)
    request = {
        "text": " ".join(profile.get("text_parts", [])),
        "preferences": profile["preferences"],
        "budget": profile["budget"],
        "budget_amount": profile.get("budget_amount"),
        "budget_currency": profile.get("budget_currency"),
        "travel_goal": profile.get("travel_goal"),
        "landscape": profile.get("landscape"),
        "route_style": profile.get("route_style") or "balanced",
        "pace": profile.get("pace"),
        "month": profile.get("month") or datetime.now().month,
        "days": profile.get("days") or 5,
        "cities": profile.get("cities", []),
        "countries": profile.get("countries", []),
        "language": profile.get("language", "he"),
    }
    result = planner.answer_request(request, fetch_live_weather=fetch_live_weather)

    if result.get("destinations"):
        result["destinations"] = result["destinations"][:1]
        destination = result["destinations"][0]
        destination["source_url"] = build_search_url(f"{destination.get('city')} {destination.get('country')} tourism")
        destination["source_query"] = f"{destination.get('city')} {destination.get('country')} tourism"
        month_names_en = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        selected_month_name = month_names_en[int(profile.get("month") or 1)]
        destination["weather_query"] = f"{destination.get('city')} {destination.get('country')} weather {selected_month_name}"
        destination["weather_url"] = build_search_url(destination["weather_query"])
        destination["display_match_reason"] = _clean_match_reason_for_ui(destination.get("match_reason", ""))
        destination["read_more_label"] = "למידע נוסף" if profile["language"] == "he" else "Read more"

    if result.get("destinations"):
        selected = result["destinations"][0]
        if profile["language"] == "he":
            result["reply"] = f"היעד שהכי מתאים למה שסיפרת לי הוא {selected.get('city')}, {selected.get('country')}."
        else:
            result["reply"] = f"The destination that best fits what you shared is {selected.get('city')}, {selected.get('country')}."
    else:
        result["reply"] = _clean_user_reply(result.get("reply", ""), profile["language"])

    destination = result.get("destinations", [{}])[0]
    if destination:
        result["match_explanation"] = {
            "title": "למה זה מתאים?" if profile["language"] == "he" else "Why it fits",
            "items": [_clean_match_reason_for_ui(destination.get("match_reason", ""))]
        }
    local_pack = _fallback_pack(destination, profile, result.get("itinerary", [])) if destination else None
    gemini = build_gemini_recommendations(destination, profile, result.get("itinerary", [])) if destination else {"ok": False}
    result["recommendation_pack"] = gemini.get("recommendation_pack") if gemini.get("ok") else local_pack

    # Hide noisy meta text from the UI.
    result["gemini_note"] = None
    result["information_policy"] = ""
    result["follow_up_suggestions"] = []
    result["limitations"] = ""
    result["profile"] = profile
    return result


def conversational_reply(planner: TravelPlanner, message: str, state: Dict, fetch_live_weather=True):
    state = state or {}
    greeting = greeting_language(message)

    if state.get("completed") and _looks_like_new_trip_request(message):
        previous_language = (state.get("profile") or {}).get("language", _guess_language(message))
        state = {"profile": empty_profile(previous_language), "history": [], "mode": "free_chat"}

    if not state.get("profile"):
        state["profile"] = empty_profile(greeting or _guess_language(message))
        state["history"] = state.get("history") or []
        state["mode"] = "free_chat"

    current_language = (state.get("profile") or {}).get("language", _guess_language(message))
    # Gemini is deliberately the first semantic layer on every turn, including the
    # first greeting. Local keyword rules are used only if the API is unavailable.
    profile_before_ai = state.get("profile") or empty_profile(current_language)
    gemini_result = analyze_free_travel_chat(message, profile_before_ai, state.get("history", []))

    if gemini_result.get("ok") and gemini_result.get("is_travel_related") is False:
        response = _out_of_scope_response(gemini_result.get("language") or current_language)
        _append_history(state, "user", message)
        _append_history(state, "assistant", response["reply"])
        return response, state

    if not gemini_result.get("ok") and (_is_out_of_scope(message) or _is_casual_non_travel(message)):
        response = _out_of_scope_response(current_language)
        _append_history(state, "user", message)
        _append_history(state, "assistant", response["reply"])
        return response, state

    corrected_message = (
        str(gemini_result.get("corrected_user_text") or "").strip()
        if gemini_result.get("ok") else ""
    ) or _correct_common_typos(message)

    _append_history(state, "user", message)
    state = _merge_free_message(planner, state, corrected_message)
    profile = state["profile"]
    _merge_ai_extraction(profile, gemini_result, message)

    # Gemini leads the conversation. The local code only checks whether it is safe
    # to run the deterministic recommender; it no longer selects a scripted next question.
    if gemini_result.get("ok"):
        gemini_missing = gemini_result.get("missing_fields") or []
        missing = _filter_missing_by_profile(gemini_missing, profile)
        ready_to_recommend = bool(gemini_result.get("ready_to_recommend")) and not missing
    else:
        missing = _free_missing_fields(profile)
        ready_to_recommend = False

    if not ready_to_recommend:
        if gemini_result.get("ok"):
            reply = str(gemini_result.get("assistant_reply") or "").strip()
            lower_reply = reply.lower()
            sounds_ready = any(term in lower_reply for term in [
                "יש לי מספיק", "יש מספיק", "בונה", "אבנה", "recommendation", "ready", "i have enough"
            ])
            if not reply or (missing and sounds_ready):
                reply = _soft_fallback_followup(profile, missing)
        else:
            reply = _gemini_unavailable_reply(profile.get("language", "he"), gemini_result)

        _append_history(state, "assistant", reply)
        return {
            "type": "question",
            "language": profile.get("language", "he"),
            "reply": reply,
            "suggestions": [],
            "profile": profile,
            "missing_fields": missing,
            "corrected_text": profile.get("corrected_last_message"),
            "destinations": [],
            "itinerary": [],
            "gemini_ok": bool(gemini_result.get("ok")),
        }, state

    # Gemini decided the conversation has enough information; now the data model chooses the destination.
    result = _build_recommendation_response(planner, profile, fetch_live_weather=fetch_live_weather)
    result["conversation_summary"] = _profile_summary_for_reply(profile)
    result["corrected_text"] = profile.get("corrected_last_message")

    state["completed"] = True
    _append_history(state, "assistant", result.get("reply", ""))
    return result, state
