import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]


def _candidate_models():
    """Return the requested model first, then configurable fallbacks.

    This protects the demo from deprecated model names such as gemini-2.0-flash.
    GEMINI_FALLBACK_MODELS may contain a comma-separated priority list.
    """
    requested = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    configured = os.environ.get("GEMINI_FALLBACK_MODELS", "").strip()
    fallbacks = (
        [model.strip() for model in configured.split(",") if model.strip()]
        if configured else DEFAULT_GEMINI_FALLBACK_MODELS
    )
    models = [requested] + fallbacks
    unique = []
    for model in models:
        if model and model not in unique:
            unique.append(model)
    return unique


def build_search_url(query: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(query)}"


def _extract_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join(texts).strip()


def _json_from_text(text: str):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_items(items, city: str, country: str, default_reason: str, kind: str):
    normalized = []
    seen = set()
    for item in items or []:
        if isinstance(item, str):
            name = item.strip()
            reason = default_reason
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            reason = str(item.get("reason") or default_reason).strip()
        else:
            continue
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "name": name,
            "reason": reason,
            "kind": kind,
            "url": build_search_url(f"{name} {city} {country}"),
        })
    return normalized[:5]



def _read_http_error(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:1200]
    except Exception:
        return str(exc)


def _safe_failure(error: str, had_api_key: bool, tried_models=None, attempt_errors=None) -> dict:
    """Return a stable shape so callers can always fall back to local data."""
    return {
        "ok": False,
        "error": error,
        "had_api_key": had_api_key,
        "tried_models": tried_models or [],
        "attempt_errors": attempt_errors or [],
        "museums": [],
        "restaurants": [],
        "culture": [],
        "food": [],
        "recommendation_pack": {"culture": [], "food": []},
    }


def _should_try_next_model(status_code: int, error_text: str) -> bool:
    """Try another model for quota, availability, or invalid-model errors.

    Authentication failures intentionally stop immediately because changing the
    model cannot repair an expired or invalid API key.
    """
    lowered = (error_text or "").lower()
    quota_error = status_code == 429 or (
        status_code in {403, 429}
        and ("resource_exhausted" in lowered or "quota" in lowered or "rate limit" in lowered)
    )
    temporary_model_error = status_code in {500, 502, 503, 504}
    invalid_model_error = status_code in {400, 404} and (
            "not found" in lowered
            or "not supported" in lowered
            or "deprecated" in lowered
            or "is not found" in lowered
    )
    return quota_error or temporary_model_error or invalid_model_error


def _call_gemini_json(prompt: str, temperature: float = 0.2, max_tokens: int = 1800) -> dict:
    """Call Gemini and parse a JSON object.

    If the user configured an old/deprecated model, try newer fallback models
    before giving up. This prevents the UI from saying "Gemini is disconnected"
    when the real problem is only a stale GEMINI_MODEL value.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _safe_failure("missing_api_key", had_api_key=False)

    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }).encode("utf-8")

    tried = []
    attempt_errors = []
    last_error = ""
    for model in _candidate_models():
        tried.append(model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?{urlencode({'key': api_key})}"
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_text = _read_http_error(exc)
            last_error = f"HTTP {exc.code} with {model}: {error_text}"
            attempt_errors.append({"model": model, "status": exc.code})
            print(f"[Gemini] {last_error}")
            if _should_try_next_model(exc.code, error_text):
                print(f"[Gemini] Trying the next configured model after {model}.")
                continue
            return _safe_failure(last_error, True, tried, attempt_errors)
        except json.JSONDecodeError:
            last_error = f"invalid_json_response with {model}"
            attempt_errors.append({"model": model, "status": "invalid_json_response"})
            print(f"[Gemini] {last_error}")
            continue
        except (URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__} with {model}: {exc}"
            print(f"[Gemini] {last_error}")
            return _safe_failure(last_error, True, tried, attempt_errors)

        try:
            parsed = _json_from_text(_extract_text(payload))
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            parsed = None
            print(f"[Gemini] Could not parse response from {model}: {type(exc).__name__}")
        if not isinstance(parsed, dict):
            last_error = f"invalid_json with {model}"
            attempt_errors.append({"model": model, "status": "invalid_json"})
            print(f"[Gemini] {last_error}")
            continue
        parsed["ok"] = True
        parsed["model_used"] = model
        if model != os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip():
            parsed["model_fallback_used"] = True
        parsed["tried_models"] = tried
        return parsed

    final_error = "invalid_json" if last_error.startswith("invalid_json") else (last_error or "all_models_failed")
    return _safe_failure(final_error, True, tried, attempt_errors)


def analyze_free_travel_chat(message: str, profile: dict, history: list) -> dict:
    """Use Gemini as the open conversational layer.

    Gemini manages the interview: it keeps context, fixes typos, notices unclear or
    inconsistent answers, extracts structured trip signals, and decides whether to
    ask another natural follow-up or move to the destination model.
    """
    safe_history = (history or [])[-10:]
    safe_profile = profile or {}
    prompt = f"""
You are TravelMind, a Hebrew/English travel-planning AI agent.
You are responsible for the OPEN conversation with the user before the data-based destination model runs.
Do NOT behave like a fixed form or questionnaire. Do not show multiple-choice buttons. Do not ask the same fields in a fixed order just because a template says so.

Your job:
1. Read the full chat context and current profile.
2. Correct obvious spelling mistakes and keyboard mistakes without changing meaning.
3. Understand the user's intent even when the answer is short, partial, informal, or misspelled.
4. Detect answers that do not fit the current context. If the user gives an irrelevant/illogical answer, gently clarify and redirect.
5. Extract structured signals for the local recommendation model.
6. Ask ONE natural follow-up question only if more information is truly needed for a reliable destination recommendation.
7. When enough information exists, set ready_to_recommend=true.

Return ONLY valid JSON with this exact structure:
{{
  "language": "he" or "en",
  "is_travel_related": true,
  "corrected_user_text": "the latest user message after fixing obvious typos, without changing meaning",
  "assistant_reply": "short natural reply. If more information is needed, ask one conversational follow-up. If enough information exists, say briefly that you have enough to build the recommendation.",
  "ready_to_recommend": false,
  "confidence": 0.0,
  "missing_fields": [],
  "context_issue": null,
  "extracted": {{
    "countries": [],
    "cities": [],
    "days": null,
    "month": null,
    "budget_amount": null,
    "budget_currency": null,
    "budget_level": null,
    "experience_type": null,
    "travel_party": null,
    "pace": null,
    "avoid": null,
    "travel_goal": null,
    "landscape": null,
    "route_style": null,
    "preferences": {{
      "culture": 0,
      "adventure": 0,
      "nature": 0,
      "beaches": 0,
      "nightlife": 0,
      "cuisine": 0,
      "wellness": 0,
      "urban": 0,
      "seclusion": 0
    }}
  }}
}}

Guidelines for enough information:
- A strong recommendation usually needs: desired experience/style, trip length, approximate budget or budget level, travel month/season, pace, and who is traveling.
- You may decide that some details are optional if the user explicitly says they are flexible, e.g. "לא משנה", "פתוח", "אין העדפה", "surprise me", "flexible".
- If a detail is missing, ask about it naturally based on the conversation, not as a fixed numbered questionnaire.
- If the user asks "תמליץ כבר" but important context is missing, do not recommend yet. Ask a natural clarification.
- If the user contradicts earlier context, ask a clarifying question instead of silently overwriting important details.
- If the latest answer is unrelated to travel or does not answer the question, set context_issue to a short explanation and redirect politely.

Spelling and context policy:
- Use your language understanding, the current question, the full chat history and the profile to infer spelling corrections. There is no predefined typo dictionary.
- Correct only when the intended meaning is reasonably clear from context. When two meanings are plausible, keep the original text and set context_issue so the interface can clarify.
- For short answers, interpret the answer primarily as a response to the latest assistant question. For example, a misspelled month after a month question should be interpreted as the most contextually and linguistically likely month.
- corrected_user_text must contain the corrected latest message, while extracted must contain the structured meaning of that corrected message.

Field normalization:
- travel_party: solo, couple, friends, family, flexible, or null.
- pace: relaxed, balanced, intensive, or null.
- experience_type examples: culture_local, beach_relax, nature_adventure, nightlife_urban, balanced_mix, beach_adventure.
- route_style examples: cultural, beaches, scenic, urban, balanced.
- landscape examples: urban_culture, tropical, varied, quiet, scenic.
- preferences values must be integers from 0 to 5.
- Budget currency: ILS for ₪/שקל/שח, USD for $, EUR for €.
- Never infer budget currency only from the conversation language. A plain number such as "5000" must keep budget_currency=null so the guided interface can ask for the currency.
- Month must be 1-12 when a month or season can be inferred; otherwise null.
- missing_fields should contain only the fields that are still genuinely needed, for example: experience, travel_party, budget, days, month, pace. Do not include fields already known from profile/history.

Scope rules:
- Travel planning includes destinations, routes, accommodation, transport, attractions, restaurants while travelling, weather for a trip, budgets and travel preferences.
- A greeting, thanks, confirmation, short answer, or clarification is travel-related when it continues this conversation. Never reject it merely because it has no travel keyword.
- General cooking, programming, homework, medical, CV, entertainment or unrelated requests are outside scope unless they are clearly connected to planning a trip.
- If the latest message is outside scope, set is_travel_related=false. In assistant_reply, briefly explain that TravelMind focuses on travel and invite the user back with one concrete travel-planning example.
- Use the chat history to resolve pronouns and short replies. Do not discard previously known facts.

Current profile:
{json.dumps(safe_profile, ensure_ascii=False)}

Chat history:
{json.dumps(safe_history, ensure_ascii=False)}

Latest user message:
{message}
""".strip()
    return _call_gemini_json(prompt, temperature=0.25, max_tokens=1800)

def build_gemini_recommendations(destination: dict, profile: dict, local_itinerary: list) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "ok": False, "error": "missing_api_key", "note": None,
            "museums": [], "restaurants": [],
            "recommendation_pack": {"culture": [], "food": []},
        }

    language = profile.get("language", "he")
    days = int(profile.get("days") or len(local_itinerary) or 3)
    city = destination.get("city")
    country = destination.get("country")
    prompt = f"""
You are a travel-planning assistant.
The destination was already selected by a local deterministic data model.
Do not replace the destination.

Return ONLY valid JSON in this exact structure:
{{
  "note": "short sentence in Hebrew",
  "must_see": [{{"name": "...", "reason": "..."}}],
  "culture": [{{"name": "...", "reason": "..."}}],
  "food": [{{"name": "...", "reason": "..."}}],
  "trip_structure": [{{"title": "...", "summary": "..."}}]
}}

Language: {language}
Destination: {city}, {country}
Trip length: {days} days
User profile: {json.dumps(profile, ensure_ascii=False)}
Local data evidence: {json.dumps(destination.get('data_evidence', []), ensure_ascii=False)}
Local fallback itinerary: {json.dumps(local_itinerary, ensure_ascii=False)}

Rules:
- culture: 3-5 REAL museums / heritage / culture places.
- food: 3-5 REAL restaurants, markets, food halls, or known food areas.
- Use ONLY specific place names, never generic text like "main museum" or "old town cafes".
- Do not repeat the same place too many times across sections.
- No "maybe", "perhaps", or option lists.
- Keep each reason very short.
- Do not invent prices or opening hours.
""".strip()

    parsed = _call_gemini_json(prompt, temperature=0.3, max_tokens=1800)
    if not parsed.get("ok"):
        return {
            "ok": False,
            "error": parsed.get("error") or "gemini_unavailable",
            "note": None,
            "museums": [],
            "restaurants": [],
            "recommendation_pack": {"culture": [], "food": []},
        }

    recommendation_pack = {
        "culture": _normalize_items(parsed.get("culture"), city, country, "אתר תרבות/מורשת מרכזי ביעד", "culture"),
        "food": _normalize_items(parsed.get("food"), city, country, "מקום אוכל מומלץ ביעד", "food"),
    }
    if not recommendation_pack.get("culture") and not recommendation_pack.get("food"):
        return {
            "ok": False, "error": "empty_recommendations", "note": None,
            "museums": [], "restaurants": [],
            "recommendation_pack": {"culture": [], "food": []},
        }

    return {
        "ok": True,
        "note": str(parsed.get("note") or "").strip() or None,
        "recommendation_pack": recommendation_pack,
    }
