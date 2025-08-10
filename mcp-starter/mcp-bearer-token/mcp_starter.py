import base64
import asyncio
import os
import sys
import random
import re
import json
from typing import Annotated, List
from datetime import datetime
from urllib.parse import urlparse, unquote

import httpx
import asyncpg
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import McpError, ErrorData
from mcp.server.auth.provider import AccessToken
from mcp.types import INVALID_PARAMS
from pydantic import Field
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from scipy.ndimage import gaussian_filter
from numpy import array
import io
import traceback
from twilio.rest import Client

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
TO_NUMBER = os.environ.get("TO_NUMBER")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
POSTGRES_URL = os.environ.get("POSTGRES_URL")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER")


assert TOKEN, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER, "Please set MY_NUMBER in your .env file"
assert OPENROUTER_API_KEY, "Please set OPENROUTER_API_KEY in your .env file"
assert POSTGRES_URL, "Please set POSTGRES_URL in your .env file"
assert TWILIO_ACCOUNT_SID, "Please set TWILIO_ACCOUNT_SID in your .env file"
assert TWILIO_AUTH_TOKEN, "Please set TWILIO_AUTH_TOKEN in your .env file"
assert TWILIO_WHATSAPP_NUMBER, "Please set TWILIO_WHATSAPP_NUMBER in your .env file"

# Configure Tesseract path for Windows
if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Twilio Client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        return None

# --- MCP Server Setup ---
mcp = FastMCP(
    "Health Assistant MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# ============================
#  POSTGRESQL POOL & DB FUNCTIONS
# ============================
pg_pool = None

async def init_pg_pool():
    global pg_pool
    url = urlparse(POSTGRES_URL)
    user = unquote(url.username) if url.username else None
    password = unquote(url.password) if url.password else None
    database = url.path.lstrip('/') if url.path else None
    host = url.hostname or "localhost"
    port = url.port or 5432

    pg_pool = await asyncpg.create_pool(
        user=user,
        password=password,
        database=database,
        host=host,
        port=port,
        min_size=1,
        max_size=10,
    )
    # Create tables if not exists
    async with pg_pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id SERIAL PRIMARY KEY,
            medicine_name VARCHAR(100) NOT NULL,
            quantity VARCHAR(50) NOT NULL,
            time VARCHAR(50) NOT NULL,
            is_medicine_taken BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS blood_reports (
            id SERIAL PRIMARY KEY,
            test_date DATE NOT NULL,
            hemoglobin FLOAT,
            total_cholesterol FLOAT,
            bilirubin FLOAT,
            wbc_count FLOAT,
            blood_glucose FLOAT,
            serum_creatinine FLOAT,
            tsh FLOAT,
            platelet_count FLOAT,
            alt FLOAT,
            hbA1c FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """)

async def save_prescription_to_db(data: dict):
    async with pg_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO prescriptions (medicine_name, quantity, time)
            VALUES ($1, $2, $3)
        """, data["medicine_name"], data["quantity"], data["time"])

async def fetch_prescriptions_from_db():
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch("SELECT medicine_name, quantity, time FROM prescriptions")
        return [dict(row) for row in rows]

async def save_blood_report_to_db(data: dict):
    async with pg_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO blood_reports
            (test_date, hemoglobin, total_cholesterol, bilirubin, wbc_count, blood_glucose, serum_creatinine,
             tsh, platelet_count, alt, hbA1c)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        """,
        data.get("test_date"),
        data.get("hemoglobin"),
        data.get("total_cholesterol"),
        data.get("bilirubin"),
        data.get("wbc_count"),
        data.get("blood_glucose"),
        data.get("serum_creatinine"),
        data.get("tsh"),
        data.get("platelet_count"),
        data.get("alt"),
        data.get("hbA1c"),
        )

async def fetch_all_blood_reports():
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT test_date, hemoglobin, total_cholesterol, bilirubin, wbc_count, blood_glucose, serum_creatinine,
                    tsh, platelet_count, alt, hbA1c
            FROM blood_reports
            ORDER BY test_date ASC
        """)
        return [dict(row) for row in rows]

# ============================
#  SYMPTOM CHECKER TOOL
# ============================
SYMPTOM_CHECKER_DESCRIPTION = """
Analyzes user-provided symptoms and suggests possible conditions, home remedies, and specialist recommendations.
This tool does not provide a diagnosis and users should always consult a medical professional.
"""

async def query_openrouter(symptoms: str) -> str:
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "X-Title": "Symptom Checker MCP Server",
                },
                json={
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "messages": [
                        {"role": "system", "content": (
                            "You are a health information assistant. Provide possible conditions, "
                            "home remedies, and specialist recommendations for the given symptoms. "
                            "Do not provide a diagnosis. Always include a disclaimer."
                        )},
                        {"role": "user", "content": f"Analyze these symptoms: {symptoms}"},
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"<error>Failed to query AI model: {str(e)}</error>"

@mcp.tool(description=SYMPTOM_CHECKER_DESCRIPTION)
async def symptom_checker(symptoms: Annotated[str, Field(description="A description of the user's symptoms.")]) -> str:
    if not symptoms or len(symptoms.strip()) < 5:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide a detailed description of symptoms."))

    disclaimer = (
        "⚠ Disclaimer: This is for educational purposes only and not medical advice. "
        "Always consult a doctor.\n\n"
    )

    ai_response = await query_openrouter(symptoms)
    if "<error>" in ai_response:
        return f"{disclaimer}Symptoms: {symptoms}\nError: {ai_response}"

    return f"{disclaimer}Symptoms: {symptoms}\nAI Analysis:\n{ai_response}"

# ============================
#  PRESCRIPTION & MEDICINE ORDERING TOOLS
# ============================
def parse_medicines_from_text(text: str):
    patterns = [
        r'\b([A-Z][a-zA-Z0-9]+)\s+(Tab|Cap|Syr|Inj|Oint|Cr|Lot|Susp|Drops)\b',
        r'\b([A-Z][a-zA-Z0-9]+)\s+([0-9.]+)(mg|g|ml|mcg|units)\b'
    ]
    
    medicines = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            medicines.add(match[0].strip())
    
    return list(medicines)

@mcp.tool(description="Saves a medical prescription.")
async def save_prescription(prescription_text: Annotated[str, Field(description="Raw prescription text, e.g., 'Paracetamol 500mg morning'")]) -> str:
    data = parse_prescription(prescription_text)
    try:
        await save_prescription_to_db(data)
        return f"✅ Prescription saved: {data['medicine_name']} ({data['quantity']}) for the {data['time']}."
    except Exception as e:
        return f"Error saving prescription: {str(e)}"

def parse_prescription(text: str) -> dict:
    """Extract medicine name, quantity, and time from prescription text using regex."""
    med_match = re.search(r"([A-Za-z]+)", text)
    qty_match = re.search(r"(\d+)\s*(mg|ml|pills|tablet[s]?)", text, re.IGNORECASE)
    time_match = re.search(r"(morning|afternoon|evening|night|\d+\s*(?:am|pm))", text, re.IGNORECASE)

    return {
        "medicine_name": med_match.group(1) if med_match else "Unknown",
        "quantity": qty_match.group(0) if qty_match else "Unknown",
        "time": time_match.group(0) if time_match else "Unknown",
    }
    
@mcp.tool(description="Marks a medicine as taken.")
async def mark_medicine_taken(medicine_name: Annotated[str, Field(description="Name of the medicine to mark as taken.")]) -> str:
    async with pg_pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE prescriptions SET is_medicine_taken = TRUE WHERE medicine_name ILIKE $1 AND is_medicine_taken = FALSE
        """, medicine_name)
        if "UPDATE 1" in result:
            return f"✅ Marked {medicine_name} as taken."
        else:
            return f"❌ Could not find {medicine_name} to mark as taken, or it was already marked as taken."

@mcp.tool(description="Checks and returns any prescription reminders for the current time of day.")
async def get_prescription_reminders() -> List[str]:
    reminders = []
    try:
        hour = datetime.now().hour
        now_word = "morning" if 5 <= hour < 12 else \
                   "afternoon" if 12 <= hour < 17 else \
                   "evening" if 17 <= hour < 21 else \
                   "night"

        async with pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT medicine_name, quantity
                FROM prescriptions
                WHERE LOWER(time) = $1 AND is_medicine_taken = FALSE
            """, now_word)
        
        if not rows:
            reminders.append(f"👍 No prescriptions scheduled for the {now_word}.")
        else:
            for p in rows:
                reminders.append(f"💊 Reminder: It's time to take {p['medicine_name']} ({p['quantity']}).")

    except Exception as e:
        reminders.append(f"[Error fetching reminders] {e}")

    return reminders

@mcp.tool(description="Extract medicine names from a prescription image (Base64-encoded).")
async def extract_medicines_from_prescription(
    puch_image_data: Annotated[str, Field(description="Base64-encoded prescription image data (jpg/png)")]
) -> str:
    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        gray_image = ImageOps.grayscale(image)
        width, height = gray_image.size
        new_width = int(width * (300 / 72)) if width < 1000 else width
        new_height = int(height * (300 / 72)) if height < 1000 else height
        resized_image = gray_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        denoised_image = resized_image.filter(ImageFilter.MedianFilter(3))

        img_array = array(denoised_image)
        blurred_img = gaussian_filter(img_array, sigma=1)
        
        binarized_img_array = ((blurred_img > 128) * 255).astype('uint8')
        binarized_image = Image.fromarray(binarized_img_array).convert('1')

        extracted_text = pytesseract.image_to_string(binarized_image, config='--psm 6')

        if not extracted_text.strip():
             extracted_text = pytesseract.image_to_string(denoised_image, config='--psm 3')

        medicines = parse_medicines_from_text(extracted_text)
        
        if not medicines:
            return "No medicines detected in the prescription image."

        return f"Extracted medicines: {', '.join(medicines)}"

    except Exception as e:
        traceback.print_exc()
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"OCR processing failed: {str(e)}"))

pending_orders = {}

def generate_order_id():
    return "ORDER" + os.urandom(4).hex().upper()

def generate_otp():
    return f"{random.randint(100000, 999999)}"

@mcp.tool(description="Order medicines (with OTP confirmation).")
async def order_medicines(medicine_list: Annotated[str, Field(description="Comma-separated list of medicines")]) -> str:
    medicines = [m.strip() for m in medicine_list.split(",") if m.strip()]
    if not medicines:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="No medicines specified."))

    order_id = generate_order_id()
    otp = generate_otp()
    pending_orders[order_id] = {"medicines": medicines, "otp": otp, "confirmed": False}
    return f"Order ID: {order_id}\nMedicines: {', '.join(medicines)}\nOTP (for demo): {otp}"

@mcp.tool(description="Confirm medicine order with OTP.")
async def confirm_order(order_id: str, otp: str) -> str:
    order = pending_orders.get(order_id)
    if not order:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid order ID."))
    if otp != order["otp"]:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid OTP."))
    order["confirmed"] = True
    return f"Order {order_id} confirmed for: {', '.join(order['medicines'])}"

# ============================
#  BLOOD REPORTS TOOL
# ============================
BLOOD_REPORT_PATTERNS = {
    "hemoglobin": r"Hemoglobin\s*[:=]?\s*([\d.]+)",
    "total_cholesterol": r"Total Cholesterol\s*[:=]?\s*([\d.]+)",
    "bilirubin": r"Bilirubin\s*[:=]?\s*([\d.]+)",
    "wbc_count": r"White Blood Cell(?: Count)?\s*[:=]?\s*([\d.]+)",
    "blood_glucose": r"Blood Glucose\s*[:=]?\s*([\d.]+)",
    "serum_creatinine": r"Serum Creatinine\s*[:=]?\s*([\d.]+)",
    "tsh": r"TSH\s*[:=]?\s*([\d.]+)",
    "platelet_count": r"Platelet Count\s*[:=]?\s*([\d.]+)",
    "alt": r"ALT\s*[:=]?\s*([\d.]+)",
    "hbA1c": r"HbA1c\s*[:=]?\s*([\d.]+)",
    "test_date": r"Date\s*[:=]?\s*([\d/\-]+)",
}

def parse_blood_report(text: str) -> dict:
    data = {}
    for key, pattern in BLOOD_REPORT_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1)
            if key == "test_date":
                for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                    try:
                        data[key] = datetime.strptime(val, fmt).date()
                        break
                    except ValueError:
                        continue
                if key not in data:
                    data[key] = None
            else:
                try:
                    data[key] = float(val)
                except ValueError:
                    data[key] = None
        else:
            data[key] = None
    return data

@mcp.tool(description="Save blood report data by parsing the input text.")
async def save_blood_report(report_text: Annotated[str, Field(description="Raw blood report text with a date.")]) -> str:
    data = parse_blood_report(report_text)
    if not data.get("test_date"):
        return "Error: Test date not found or invalid in the report. Please include date in DD/MM/YYYY or YYYY-MM-DD format."

    try:
        await save_blood_report_to_db(data)
        response_data = {k: v for k, v in data.items() if v is not None}
        return f"✅ Blood report saved for date {data['test_date']}. Data captured: {json.dumps(response_data, default=str)}"
    except Exception as e:
        return f"Error saving blood report: {str(e)}"

BLOOD_REPORT_SUMMARY_SYSTEM_PROMPT = """
You are a medical data analyst assistant.
Given a person's blood test history including parameters like Hemoglobin, Total Cholesterol, Bilirubin, White Blood Cell count, Blood Glucose, Serum Creatinine, TSH, Platelet Count, ALT, and HbA1c over multiple dates, provide a clear and concise progress summary.
Explain trends, whether values are increasing, decreasing, or stable.
Indicate if any values are out of normal healthy ranges and suggest if urgent medical consultation is recommended.
The summary should be easy to understand for a non-medical person but medically accurate.
Always include a disclaimer that this is not medical advice and they should consult a healthcare professional for diagnosis.
"""

async def query_blood_report_summary_ai(reports_json: str) -> str:
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "X-Title": "Blood Report Summary MCP Server",
                },
                json={
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "messages": [
                        {"role": "system", "content": BLOOD_REPORT_SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Here is the blood test history data:\n{reports_json}\nPlease provide a progress summary."}
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"<error>Failed to query AI model: {str(e)}</error>"

@mcp.tool(description="Get an AI-generated progress summary based on your blood test history.")
async def blood_report_progress_summary() -> str:
    try:
        reports = await fetch_all_blood_reports()
        if not reports:
            return "No blood reports found to analyze."

        reports_json = json.dumps(reports, indent=2, default=str)
        ai_summary = await query_blood_report_summary_ai(reports_json)
        
        if "<error>" in ai_summary:
            return f"AI summary generation failed: {ai_summary}"
        
        disclaimer = (
            "⚠ Disclaimer: This is for educational purposes only and not medical advice. "
            "Please consult a healthcare professional for any diagnosis or treatment."
        )
        return f"{disclaimer}\n\n{ai_summary}"
    except Exception as e:
        return f"Error generating blood report summary: {str(e)}"

def send_whatsapp_message(to_number: str, message_body: str):
    """Sends a WhatsApp message using the Twilio API."""
    try:
        twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=f'whatsapp:{to_number}',
            body=message_body
        )
        return True
    except Exception as e:
        print(f"Failed to send WhatsApp message: {e}")
        return False

# --- Conceptual background task for proactive reminders ---
async def reminder_sender_loop():
    while True:
        try:
            # Check for pending reminders for the current time
            hour = datetime.now().hour
            now_word = "morning" if 5 <= hour < 12 else \
                       "afternoon" if 12 <= hour < 17 else \
                       "evening" if 17 <= hour < 21 else \
                       "night"
            
            async with pg_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT medicine_name, quantity
                    FROM prescriptions
                    WHERE LOWER(time) = $1 AND is_medicine_taken = FALSE
                """, now_word)

            if rows:
                for p in rows:
                    reminder_message = f"💊 Reminder: It's time to take {p['medicine_name']} ({p['quantity']})."
                    
                    if send_whatsapp_message(TO_NUMBER, reminder_message):
                        print(f"PROACTIVE REMINDER SENT to {TO_NUMBER}: {reminder_message}")
            
            # Reset is_medicine_taken for the next day's reminders
            if now_word == "night":
                async with pg_pool.acquire() as conn:
                    await conn.execute("UPDATE prescriptions SET is_medicine_taken = FALSE")
        
        except Exception as e:
            print(f"Error in reminder sender loop: {e}")

        # Sleep for a period before checking again (e.g., every 30 minutes)
        await asyncio.sleep(1800)  # 1800 seconds = 30 minutes

# ============================
#  SERVER START
# ============================
async def main():
    print("🚀 Starting MCP server with PostgreSQL and Twilio integration...")
    await init_pg_pool()
    # Schedule the reminder sender loop as a background task
    asyncio.create_task(reminder_sender_loop())
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())