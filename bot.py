import os
from flask import Flask, request
import requests

TELEGRAM_TOKEN = os.environ.get("8254088115:AAEVVnMW4TE07EiVLhccoecPIee5GXVOV8M")
CHAT_ID = os.environ.get("1463068783")

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
    
    # ==============================
    # 🔥 PUT YOUR EXISTING SCREENER LOGIC HERE
    # main_scan()
    # ==============================

    send_telegram("✅ Scan completed.")

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    data = request.json

    if "message" in data:
        text = data["message"].get("text", "")
        if text == "/scan":
            run_scan()

    return "OK"

@app.route("/")
def home():
    return "Bot is alive 🚀"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
