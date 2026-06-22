import os
from gemini_service import _call_gemini_json

print("GEMINI_API_KEY exists:", bool(os.environ.get("GEMINI_API_KEY", "").strip()))
print("GEMINI_MODEL:", os.environ.get("GEMINI_MODEL") or "default")

result = _call_gemini_json('''Return only this JSON: {"status":"ok", "message":"gemini works"}''', temperature=0, max_tokens=80)
print("Result:", result)

if result.get("ok"):
    print("OK - Gemini is working. Model used:", result.get("model_used"))
    print("Models tried:", ", ".join(result.get("tried_models") or []))
    print("Fallback used:", bool(result.get("model_fallback_used")))
else:
    print("ERROR - Gemini failed. Check the error above.")
    print("Models tried:", ", ".join(result.get("tried_models") or []))
    raise SystemExit(1)
