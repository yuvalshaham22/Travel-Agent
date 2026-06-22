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
WIZARD_FIELDS = ["experience", "travel_party", "budget", "days", "month", "pace"]

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
    state = merge_message(planner, state, corrected)
    state.pop("pending_field", None)
    state.pop("pending_free_field", None)
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
    elif profile.get("budget_amount") and not profile.get("budget_currency"):
        missing.append("budget_currency")
    if not profile.get("days"):
        missing.append("days")
    if not profile.get("month"):
        missing.append("month")
    if not profile.get("pace"):
        missing.append("pace")
    return [field for field in REQUIRED_FREE_FIELDS if field in missing]


def _guided_question(field: str, language: str = "he") -> tuple[str, list]:
    """Return one clear question and clickable answers for the next missing field."""
    if language == "en":
        questions = {
            "experience": ("What kind of experience would you most enjoy?", ["Culture and food", "Beaches and relaxation", "Nature and adventure", "City and nightlife", "A balanced mix"]),
            "travel_party": ("Who is travelling?", ["Solo", "Couple", "Friends", "Family"]),
            "budget": ("What is your approximate total budget? You can also type any amount.", ["3,000 ILS", "5,000 ILS", "10,000 ILS"]),
            "budget_currency": ("Which currency is the budget in?", ["ILS", "USD", "EUR"]),
            "days": ("How many days should the trip be?", ["3 days", "5 days", "7 days", "10 days"]),
            "month": ("When would you like to travel?", ["January", "April", "July", "October"]),
            "pace": ("What pace would you prefer?", ["Relaxed", "Balanced", "Intensive"]),
        }
    else:
        questions = {
            "experience": (
                "היי 😊 אני אעזור לך לתכנן טיול בכמה שאלות קצרות — ואז אבחר יעד אחד ואבנה לך המלצות מסודרות. איזו חוויה את מחפשת בטיול?",
                ["תרבות, היסטוריה ואווירה מקומית", "חופים, רוגע ונוף", "טבע, נופים והרפתקאות", "חיי לילה, בילויים ועירוניות", "שילוב מאוזן של כמה דברים"],
            ),
            "travel_party": ("עם מי את נוסעת?", ["לבד", "זוג", "חברים/ות", "משפחה", "לא משנה / פתוח"]),
            "budget": ("מה התקציב הכולל לטיול? אפשר לכתוב סכום, למשל 5,000 ₪ או 10,000 ₪.", ["5,000 ₪", "10,000 ₪", "15,000 ₪", "אחר"]),
            "days": ("לכמה ימים מתוכנן הטיול?", ["4 ימים", "5 ימים", "7 ימים", "10 ימים"]),
            "month": ("באיזה חודש תרצי לטוס? זה יעזור לי להתאים את מזג האוויר ביעד.", ["ינואר", "אפריל", "יולי", "אוקטובר"]),
            "pace": ("איזה קצב טיול מתאים לך?", ["רגוע — מעט פעילויות והרבה זמן חופשי", "מאוזן — כמה דברים ביום בלי עומס", "אינטנסיבי — להספיק כמה שיותר"]),
        }
    return questions.get(field, ("ספרו לי עוד פרט קטן על הטיול.", []))


def _wizard_field_complete(field: str, profile: Dict) -> bool:
    checks = {
        "experience": bool(any(profile.get("preferences", {}).values()) or profile.get("experience_type") or profile.get("travel_goal")),
        "travel_party": bool(profile.get("travel_party")),
        "budget": bool(profile.get("budget_amount") or profile.get("budget")),
        "days": bool(profile.get("days")),
        "month": bool(profile.get("month")),
        "pace": bool(profile.get("pace")),
    }
    return checks.get(field, False)


def _turn_answers_wizard_field(field: str, message: str, ai_result: Dict) -> bool:
    """Require an answer to the question currently on screen before advancing."""
    normalized = str(message or "").lower().strip()
    extracted = ai_result.get("extracted") or {} if ai_result.get("ok") else {}
    if field == "experience":
        terms = ["תרבות", "היסטוריה", "חופים", "רוגע", "טבע", "נופים", "הרפתקאות", "חיי לילה", "בילויים", "עירוניות", "שילוב", "culture", "beach", "nature", "nightlife", "balanced"]
        return any(term in normalized for term in terms) or bool(extracted.get("experience_type") or extracted.get("travel_goal") or any((extracted.get("preferences") or {}).values()))
    if field == "travel_party":
        terms = ["לבד", "זוג", "חבר", "משפחה", "לא משנה", "פתוח", "solo", "couple", "friends", "family", "flexible"]
        return any(term in normalized for term in terms) or bool(extracted.get("travel_party"))
    if field == "budget":
        amount, _ = parse_budget_amount(message, allow_plain=True)
        return bool(amount or extracted.get("budget_amount") or extracted.get("budget_level"))
    if field == "days":
        return bool(re.search(r"\b\d{1,2}\b", normalized) or extracted.get("days"))
    if field == "month":
        return bool(_parse_month_answer(message) or extracted.get("month"))
    if field == "pace":
        terms = ["רגוע", "מאוזן", "אינטנסיבי", "relaxed", "balanced", "intensive"]
        return any(term in normalized for term in terms) or bool(extracted.get("pace"))
    return False

def _filter_missing_by_profile(missing_fields, profile: Dict) -> list:
    """Keep Gemini's missing-fields list aligned with what was already extracted.

    This is only a safety gate before running the deterministic recommender; it is
    not used to choose a scripted question. The next question still comes from Gemini.
    """
    # The profile is the source of truth and REQUIRED_FREE_FIELDS supplies a stable,
    # predictable order for the guided cards. Gemini's list is advisory only.
    return _free_missing_fields(profile)


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

HEBREW_MONTH_LABELS = ["", "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]
ENGLISH_MONTH_LABELS = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _month_label(month: int, language: str = "he") -> str:
    labels = ENGLISH_MONTH_LABELS if language == "en" else HEBREW_MONTH_LABELS
    return labels[month] if 1 <= int(month or 0) <= 12 else ""


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
    # Fuzzy matching across a full sentence can turn an unrelated Hebrew word
    # into a month. Use it only for a one-word answer; exact month names and the
    # explicit typo dictionary still work inside longer sentences.
    month = _fuzzy_month_from_text(normalized, allow_fuzzy=len(_word_tokens(normalized)) == 1)
    if month:
        return month
    explicit = re.search(r"(?:חודש|month|בחודש|מספר חודש)\s*(1[0-2]|[1-9])", normalized)
    if explicit:
        return int(explicit.group(1))
    return None


def _apply_experience(profile: Dict, text: str) -> None:
    normalized = text.lower().strip()
    if any(term in normalized for term in ["רומנטי", "רומנטית", "ירח דבש", "romantic", "honeymoon"]):
        profile["experience_type"] = "romantic_relax"
        profile["travel_goal"] = "romance"
        profile["route_style"] = "balanced"
        profile["landscape"] = "scenic"
        profile["travel_party"] = profile.get("travel_party") or "couple"
        profile["preferences"]["cuisine"] = max(profile["preferences"].get("cuisine", 0), 4)
        profile["preferences"]["wellness"] = max(profile["preferences"].get("wellness", 0), 4)
        profile["preferences"]["seclusion"] = max(profile["preferences"].get("seclusion", 0), 3)

    elif any(term in normalized for term in ["תרבות", "היסטוריה", "אווירה מקומית", "culture", "history", "local atmosphere"]):
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


def _apply_budget_currency(profile: Dict, text: str) -> None:
    normalized = str(text or "").lower()
    if any(term in normalized for term in ["שקל", "שח", "ils", "₪"]):
        profile["budget_currency"] = "ILS"
    elif any(term in normalized for term in ["דולר", "usd", "$"]):
        profile["budget_currency"] = "USD"
    elif any(term in normalized for term in ["אירו", "יורו", "eur", "€"]):
        profile["budget_currency"] = "EUR"


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
    elif pending == "budget_currency":
        _apply_budget_currency(profile, message)
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


def _build_packing_list(destination: dict, profile: dict) -> dict:
    """Build a practical list from destination climate, trip style and party."""
    language = profile.get("language", "he")
    city = str(destination.get("city") or "").strip()
    country = str(destination.get("country") or "").strip()
    try:
        days = max(1, int(profile.get("days") or 1))
    except (TypeError, ValueError):
        days = 1
    temperature = destination.get("temperature") or {}
    try:
        average = float(temperature.get("avg")) if temperature.get("avg") is not None else None
    except (TypeError, ValueError):
        average = None

    items = []

    def add(*values):
        for value in values:
            if value and value not in items:
                items.append(value)

    if language == "en":
        add("Passport, travel documents and offline copies", "Medicines and a small first-aid kit", "Universal power adapter and portable charger", "Comfortable walking shoes", "Reusable water bottle")
        if average is not None and average <= 10:
            add("Warm coat", "Thermal layers", "Warm hat, scarf and gloves", "Compact umbrella")
            weather = f"Cold weather, about {average:g}°C on average"
        elif average is not None and average <= 20:
            add("Light jacket", "Layered clothing", "Closed shoes", "Compact umbrella")
            weather = f"Mild/cool weather, about {average:g}°C on average"
        elif average is not None and average >= 28:
            add("Light breathable clothing", "Sun hat", "High-SPF sunscreen", "Sunglasses")
            weather = f"Hot weather, about {average:g}°C on average"
        else:
            add("Light clothing", "Thin evening layer", "Sunscreen", "Sunglasses")
            weather = f"Warm weather, about {average:g}°C on average" if average is not None else "Check the forecast 48 hours before departure"
    else:
        add("דרכון, מסמכי נסיעה ועותקים זמינים גם ללא אינטרנט", "תרופות אישיות וערכת עזרה ראשונה קטנה", "מתאם חשמל אוניברסלי ומטען נייד", "נעלי הליכה נוחות", "בקבוק מים רב־פעמי")
        if average is not None and average <= 10:
            add("מעיל חם", "ביגוד תרמי ושכבות", "כובע חם, צעיף וכפפות", "מטרייה קומפקטית")
            weather = f"מזג אוויר קר — כ־{average:g}°C בממוצע"
        elif average is not None and average <= 20:
            add("ז׳קט קל", "ביגוד בשכבות", "נעליים סגורות", "מטרייה קומפקטית")
            weather = f"מזג אוויר מתון עד קריר — כ־{average:g}°C בממוצע"
        elif average is not None and average >= 28:
            add("בגדים קלים ונושמים", "כובע שמש", "קרם הגנה SPF גבוה", "משקפי שמש")
            weather = f"מזג אוויר חם — כ־{average:g}°C בממוצע"
        else:
            add("בגדים קלים", "שכבה דקה לערב", "קרם הגנה", "משקפי שמש")
            weather = f"מזג אוויר חמים — כ־{average:g}°C בממוצע" if average is not None else "מומלץ לבדוק תחזית 48 שעות לפני היציאה"

    preferences = profile.get("preferences") or {}
    if preferences.get("beaches", 0) >= 3:
        add("בגד ים ובגד להחלפה" if language == "he" else "Swimwear and a change of clothes", "כפכפים ותיק חוף" if language == "he" else "Sandals and a beach bag")
    if preferences.get("nature", 0) >= 4 or preferences.get("adventure", 0) >= 4:
        add("נעלי שטח וגרביים מתאימות" if language == "he" else "Trail shoes and suitable socks", "תיק יום קטן" if language == "he" else "Small daypack")
    if preferences.get("culture", 0) >= 4 or preferences.get("urban", 0) >= 4:
        add("תיק יום קל ולבוש מכבד לאתרי תרבות" if language == "he" else "Light day bag and respectful clothing for cultural sites")

    party = profile.get("travel_party")
    party_items = {
        "solo": "עותקי מסמכים נפרדים ופרטי קשר לשעת חירום",
        "couple": "מטען משותף, מפצל קטן ותיק מסמכים זוגי",
        "friends": "מטען רב־יציאות וערכת ציוד משותפת לקבוצה",
        "family": "חטיפים לדרך, תרופות בסיסיות ועותקי מסמכים לכל בני המשפחה",
    }
    party_items_en = {
        "solo": "Separate document copies and emergency contacts",
        "couple": "Shared charger, small power splitter and joint document pouch",
        "friends": "Multi-port charger and a shared group essentials kit",
        "family": "Travel snacks, basic medicines and document copies for everyone",
    }
    add((party_items_en if language == "en" else party_items).get(party))

    if language == "en":
        if days <= 5:
            add(f"{days} changes of underwear and socks")
        else:
            add("Up to 7 changes of underwear and socks", "Small laundry bag and travel-size detergent")
    else:
        if days <= 5:
            add(f"{days} סטים של לבנים וגרביים")
        else:
            add("עד 7 סטים של לבנים וגרביים", "שק כביסה קטן וחומר כביסה בגודל נסיעות")

    destination_name = ", ".join(part for part in [city, country] if part)
    party_labels = {"solo": "לבד", "couple": "זוג", "friends": "חברים/ות", "family": "משפחה", "flexible": "הרכב גמיש"}
    party_labels_en = {"solo": "solo", "couple": "couple", "friends": "friends", "family": "family", "flexible": "flexible party"}
    party_label = (party_labels_en if language == "en" else party_labels).get(party)
    duration_context = f"{days} days" if language == "en" else f"{days} ימים"
    context = " · ".join(value for value in [weather, duration_context, party_label] if value)
    return {
        "title": (f"Packing list for {destination_name}" if language == "en" else f"רשימת ציוד מותאמת ל־{destination_name}"),
        "subtitle": context,
        "items": items,
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
    result["packing_list"] = _build_packing_list(destination, profile) if destination else None
    # Keep pattern/anomaly calculations internal for the assignment, but do not
    # expose them to the UI. This also prevents an older cached template from
    # rendering the former "AI data insights" section.
    result.pop("data_insights", None)

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

    if greeting:
        state = {"profile": empty_profile(greeting), "history": [], "mode": "guided", "wizard_completed": []}
    elif state.get("completed") and _looks_like_new_trip_request(message):
        previous_language = (state.get("profile") or {}).get("language", _guess_language(message))
        state = {"profile": empty_profile(previous_language), "history": [], "mode": "guided", "wizard_completed": []}

    if not state.get("profile"):
        state["profile"] = empty_profile(greeting or _guess_language(message))
        state["history"] = state.get("history") or []
        state["mode"] = "guided"
        state["wizard_completed"] = []

    current_language = (state.get("profile") or {}).get("language", _guess_language(message))
    state["profile"].pop("corrected_last_message", None)
    pending_before_ai = state.get("pending_field")
    local_context_month = _parse_month_answer(message) if pending_before_ai == "month" else None
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

    if local_context_month:
        corrected_message = _month_label(local_context_month, current_language)
        if gemini_result.get("ok"):
            gemini_result = dict(gemini_result)
            gemini_result["corrected_user_text"] = corrected_message
            gemini_result["extracted"] = dict(gemini_result.get("extracted") or {})
            gemini_result["extracted"]["month"] = local_context_month
    else:
        corrected_message = (
            str(gemini_result.get("corrected_user_text") or "").strip()
            if gemini_result.get("ok") else ""
        ) or _correct_common_typos(message)

    _append_history(state, "user", message)
    state = _merge_free_message(planner, state, corrected_message)
    profile = state["profile"]
    _merge_ai_extraction(profile, gemini_result, message)
    if local_context_month:
        profile["month"] = local_context_month
        if corrected_message.lower() != message.strip().lower():
            profile["corrected_last_message"] = corrected_message

    completed_fields = list(state.get("wizard_completed") or [])
    if (
        pending_before_ai in WIZARD_FIELDS
        and _turn_answers_wizard_field(pending_before_ai, message, gemini_result)
        and _wizard_field_complete(pending_before_ai, profile)
        and pending_before_ai not in completed_fields
    ):
        completed_fields.append(pending_before_ai)
    state["wizard_completed"] = completed_fields

    # In the Hebrew guided flow, a bare amount is treated as ILS, matching the
    # shekel examples shown in the budget question.
    if pending_before_ai == "budget" and profile.get("budget_amount") and not profile.get("budget_currency"):
        profile["budget_currency"] = "ILS" if profile.get("language") == "he" else "USD"

    next_field = next((field for field in WIZARD_FIELDS if field not in completed_fields), None)
    missing = [field for field in WIZARD_FIELDS if field not in completed_fields]
    ready_to_recommend = next_field is None

    if not ready_to_recommend:
        state["pending_field"] = next_field
        reply, suggestions = _guided_question(next_field, profile.get("language", "he"))
        if next_field == "budget" and message.strip().lower() in {"אחר", "other"}:
            reply = "כתבי את התקציב הכולל והמטבע, למשל 8,000 ₪ או 2,000 €."
            suggestions = []

        _append_history(state, "assistant", reply)
        return {
            "type": "question",
            "language": profile.get("language", "he"),
            "reply": reply,
            "suggestions": suggestions,
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
