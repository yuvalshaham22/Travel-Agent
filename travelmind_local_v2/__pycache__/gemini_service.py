import json
import os
from typing import Dict, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON even when Gemini wraps it in markdown."""
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


def gemini_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def enrich_trip_with_gemini(
    destination: Dict,
    request_data: Dict,
    local_itinerary: List[Dict],
    timeout: int = 20,
) -> Optional[Dict]:
    """
    Uses Gemini only as a fallback/enrichment layer.
    The destination is already selected by the local dataset and scoring model.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    language = request_data.get("language", "he")
    days = int(request_data.get("days") or len(local_itinerary) or 5)

    prompt = f"""
You are TravelMind's enrichment layer, not the ranking engine.
The local deterministic dataset already selected exactly one destination.
Do NOT change the destination.
Use Gemini only to enrich missing itinerary details, attractions, flow, and practical notes.

Return ONLY valid JSON. No markdown.

Language: {'Hebrew' if language == 'he' else 'English'}
Selected destination from local data: {destination.get('city')}, {destination.get('country')}
Trip days: {days}
Budget amount: {request_data.get('budget_amount')} {request_data.get('budget_currency')}
Budget level from local data: {request_data.get('budget')}
Travel goal: {request_data.get('travel_goal')}
Landscape: {request_data.get('landscape')}
Route style: {request_data.get('route_style')}
Pace: {request_data.get('pace')}
Month: {request_data.get('month')}
Local data strengths: {destination.get('strengths')}
Local data description: {destination.get('description')}
Local itinerary draft: {local_itinerary}

Required JSON structure:
{{
  "gemini_used": true,
  "why_this_destination": "2-3 sentences explaining why the selected destination fits the user's profile, while saying the choice was made by the local data first.",
  "itinerary": [
    {{"day": 1, "title": "short title", "activity": "specific day plan with ideas for places/areas/activities", "basis": "data + Gemini enrichment"}}
  ],
  "extra_route_ideas": ["3-5 additional route ideas the user can choose from"],
  "practical_notes": ["3 short notes about budget/pace/booking/weather verification"]
}}

Rules:
- itinerary must contain exactly {days} days.
- Be useful and specific, but do not invent exact prices or opening hours.
- If something may change, say it should be verified before booking.
- Keep it suitable for a student project demo.
""".strip()

    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "response_mime_type": "application/json",
        },
    }).encode("utf-8")

    url = GEMINI_ENDPOINT.format(model=quote(model, safe=""), key=quote(api_key, safe=""))
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts)
        parsed = _extract_json(text)
        if not parsed:
            return None
        if not isinstance(parsed.get("itinerary"), list) or len(parsed["itinerary"]) != days:
            return None
        parsed["source"] = "Gemini API"
        return parsed
    except Exception as exc:
        return {
            "gemini_used": False,
            "source": "Gemini API",
            "error": str(exc)[:220],
        }
