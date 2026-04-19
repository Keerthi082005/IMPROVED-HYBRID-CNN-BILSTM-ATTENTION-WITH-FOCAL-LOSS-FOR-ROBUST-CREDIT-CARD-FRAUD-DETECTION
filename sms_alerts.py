# ============================================================
#  sms_alerts.py — Twilio SMS Integration (Global)
# # 🔐 Twilio Configuration

#Before running this project, you need to set your Twilio credentials.

#1. Create a `.env` file in the project root.
#2. Add your Twilio details:

#TWILIO_ACCOUNT_SID=your_account_sid
#TWILIO_AUTH_TOKEN=your_auth_token
#TWILIO_PHONE=your_twilio_phone_number

#3. Get these values from your Twilio dashboard.

#⚠️ Note: Never share or upload your actual credentials to GitHub.
# ============================================================

from twilio.rest import Client
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
# User's Twilio credentials (from task)

# Optional override from env
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")

if not ACCOUNT_SID or not AUTH_TOKEN or not TWILIO_PHONE:
    raise ValueError("❌ Please set TWILIO credentials in .env file")

def send_fraud_sms_twilio(
    phone_number: str,
    transaction_id: str,
    fraud_probability: float,
    customer_name: str = "Customer",
    account_sid: str = None,
    auth_token: str = None,
    twilio_phone: str = None
) -> dict:
    """
    Send fraud alert via Twilio SMS.
    Mirrors Fast2SMS func signature for drop-in replacement.
    Phone: +91xxxxxxxxxx or 10-digit (auto +91 prefix).
    """
    sid = account_sid or ACCOUNT_SID
    token = auth_token or AUTH_TOKEN
    from_ph = twilio_phone or TWILIO_PHONE

    if not all([sid, token, from_ph]):
        return {
            "success": False,
            "message": "❌ Missing Twilio credentials (SID/Token/Phone)",
            "response": None
        }

    # Clean/format phone for Twilio (+91xxxxxxxxxx)
    phone = str(phone_number).strip()
    phone = phone.replace("+91", "").replace(" ", "").replace("-", "")
    if len(phone) == 10:
        phone = f"+91{phone}"
    elif not phone.startswith('+'):
        phone = f"+91{phone}"
    
    if len(phone) < 12 or not phone.startswith('+91'):
        return {
            "success": False,
            "message": f"❌ Invalid number: {phone_number} (expect +91xxxxxxxxxx or 10-digit)",
            "response": None
        }

    now = datetime.now().strftime("%d-%b-%Y %I:%M %p")
    message = (
        f"FRAUD ALERT! Dear {customer_name}, "
        f"Txn {transaction_id} flagged FRAUDULENT "
        f"({fraud_probability*100:.1f}% risk) at {now}. "
        f"Not you? Call bank NOW! -FraudShield"
    )

    try:
        client = Client(sid, token)
        msg = client.messages.create(
            body=message,
            from_=from_ph,
            to=phone
        )
        return {
            "success": True,
            "message": f"✅ SMS sent to {phone} (SID: {msg.sid[:8]}...)",
            "response": {"sid": msg.sid, "status": msg.status}
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"❌ Twilio error: {str(e)}",
            "response": None
        }

def send_fraud_sms(
    phone_number: str,
    transaction_id: str,
    fraud_probability: float,
    customer_name: str = "Customer",
    api_key: str = None  # Ignored for Twilio
) -> dict:
    """
    Twilio SMS wrapper (drop-in for Fast2SMS).
    """
    return send_fraud_sms_twilio(phone_number, transaction_id, fraud_probability, customer_name)

