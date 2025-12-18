import os
from flask import Flask, request
import requests

# ✅ CORRECT: read ENV VARIABLE NAMES
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = int(os.environ.get("TELEGRAM_CHAT_ID"))

app = Flask(__name__)

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg
    }
    requests.post(url, json=payload)

def run_scan():
    send_telegram("📊 Scan started...")
    # main_scan()  # your screener logic
    send_telegram("✅ Scan completed.")

# ✅ SAFE, STATIC WEBHOOK PATH
@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    data = request.get_json(force=True)

    message = data.get("message", {})
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")

    # 🔒 Security: allow only your chat
    if chat_id != CHAT_ID:
        return "Unauthorized", 403

    if text == "/scan":
        run_scan()

    return "OK", 200

@app.route("/")
def home():
    return "Bot is alive 🚀"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)