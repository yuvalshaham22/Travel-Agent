import os
from flask import Flask, jsonify, render_template, request, session

from conversation import conversational_reply
from planner import TravelPlanner

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "travelmind-local-dev")
planner = TravelPlanner()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({
            "type": "question",
            "language": "he",
            "reply": "כתבי לי חופשי מה בא לך בטיול 😊",
            "suggestions": [],
            "destinations": [],
            "itinerary": [],
        })
    if len(message) > 2000:
        return jsonify({
            "type": "question",
            "language": "he",
            "reply": "ההודעה ארוכה מדי. נסי לקצר אותה לעד 2,000 תווים.",
            "suggestions": [],
            "destinations": [],
            "itinerary": [],
        }), 400
    state = session.get("state", {})
    response, new_state = conversational_reply(planner, message, state, fetch_live_weather=True)
    session["state"] = new_state
    return jsonify(response)


@app.post("/reset")
def reset():
    session.pop("state", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=port, debug=debug)
