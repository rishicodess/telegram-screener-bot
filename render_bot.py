import os
from flask import Flask, request
import requests

# 🔥 IMPORT FROM swing.py (NOT swing_screener.py)
from swing import main_scan

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = int(os.environ.get("TELEGRAM_CHAT_ID"))

app = Flask(__name__)

def send(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg},
            timeout=10
        )
    except Exception as e:
        print("Telegram send error:", e)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True)
    message = data.get("message", {})
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")

    # 🔒 Allow only your chat
    if chat_id != CHAT_ID:
        return "Unauthorized", 403

    if text == "/scan":
        send("☁️ Cloud scan started… this may take 1–2 minutes")
        main_scan()   # 🔥 YOUR REAL ANALYSIS
        send("✅ Cloud scan completed")

    return "OK", 200

@app.route("/")
def home():
    return "Cloud Swing Bot is alive 🚀"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
