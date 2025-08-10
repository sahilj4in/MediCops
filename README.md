# Health Assistant MCP Server for Puch AI

This is a starter template for creating your own Model Context Protocol (MCP) server that works with Puch AI. This project provides a comprehensive health assistant server with ready-to-use tools for managing prescriptions, analyzing blood reports, and more.

## What is MCP?

MCP (Model Context Protocol) allows AI assistants like Puch to connect to external tools and data sources safely. Think of it like giving your AI extra superpowers without compromising security.

## What's Included in This Starter?

### 🎯 Health Assistant Tools
- **Symptom Checker** - Analyzes user-provided symptoms and suggests possible conditions, remedies, and when to see a doctor.
- **OCR Prescription Extractor** - Processes a prescription image to extract medicine names.
- **Prescription Manager** - Saves and retrieves prescription details from a PostgreSQL database.
- **Proactive Reminders (Twilio Integration)** - Sends automated WhatsApp reminders for medicines that haven't been marked as taken.
- **Blood Report Analysis** - Parses raw text from blood reports and provides an AI-generated progress summary.
- **Medicine Ordering** - Simulates an order process with OTP confirmation.

### 🔐 Built-in Authentication
- Bearer token authentication (required by Puch AI)
- Validation tool that returns your phone number

## Quick Setup Guide

### Step 1: Install Dependencies

First, make sure you have Python 3.10 or higher installed. Then:

1.  **Create a virtual environment.**
    ```bash
    python -m venv venv
    ```

2.  **Activate the environment.**
    * **On Windows:**
        ```bash
        venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install the required Python libraries.**
    Create a `requirements.txt` file with the following contents:

    ```
    fastmcp
    asyncpg
    python-dotenv
    Pillow
    pytesseract
    scipy
    numpy
    httpx
    twilio
    ```

    Then, install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Tesseract OCR Installation

You must have Tesseract OCR installed on your system.

* **Windows:** Download the installer from the [Tesseract-OCR GitHub page](https://github.com/UB-Mannheim/tesseract/wiki).
* **macOS/Linux:** Use a package manager (`brew install tesseract` or `sudo apt-get install tesseract-ocr`).
* **Update Tesseract path:** The code includes a fix for Windows, but if Tesseract is not in your system's PATH, you may need to update the `pytesseract.pytesseract.tesseract_cmd` path in `mcp_starter.py`.

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
Then edit .env and add your details:
```

```bash
AUTH_TOKEN=your_secret_token_here
MY_NUMBER=your_phone_number_with_country_code
OPENROUTER_API_KEY=your_openrouter_api_key

POSTGRES_URL="postgresql://user:password@host:port/database"

TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

```

## Important Notes:

- AUTH_TOKEN: Your secret token for authentication.

- MY_NUMBER: Your WhatsApp number in format {country_code}{number} (e.g., 919876543210).

- POSTGRES_URL: Your PostgreSQL database connection string.

- TWILIO_*: Your Twilio credentials for WhatsApp integration.

## Step 4: Database Setup
Your code will automatically create the necessary tables (prescriptions and blood_reports) on the first run. However, if you've run the server before adding the new reminder features, you must manually add the is_medicine_taken column.

1. Connect to your PostgreSQL database.

2. Run the following SQL command:

```bash 

Step 4: Database Setup
Your code will automatically create the necessary tables (prescriptions and blood_reports) on the first run. However, if you've run the server before adding the new reminder features, you must manually add the is_medicine_taken column.

Connect to your PostgreSQL database.

Run the following SQL command:

```

```bash
cd mcp-bearer-token
python mcp_starter.py
```

Markdown

# Health Assistant MCP Server for Puch AI

This is a starter template for creating your own Model Context Protocol (MCP) server that works with Puch AI. This project provides a comprehensive health assistant server with ready-to-use tools for managing prescriptions, analyzing blood reports, and more.

## What is MCP?

MCP (Model Context Protocol) allows AI assistants like Puch to connect to external tools and data sources safely. Think of it like giving your AI extra superpowers without compromising security.

## What's Included in This Starter?

### 🎯 Health Assistant Tools
- **Symptom Checker** - Analyzes user-provided symptoms and suggests possible conditions, remedies, and when to see a doctor.
- **OCR Prescription Extractor** - Processes a prescription image to extract medicine names.
- **Prescription Manager** - Saves and retrieves prescription details from a PostgreSQL database.
- **Proactive Reminders (Twilio Integration)** - Sends automated WhatsApp reminders for medicines that haven't been marked as taken.
- **Blood Report Analysis** - Parses raw text from blood reports and provides an AI-generated progress summary.
- **Medicine Ordering** - Simulates an order process with OTP confirmation.

### 🔐 Built-in Authentication
- Bearer token authentication (required by Puch AI)
- Validation tool that returns your phone number

## Quick Setup Guide

### Step 1: Install Dependencies

First, make sure you have Python 3.10 or higher installed. Then:

1.  **Create a virtual environment.**
    ```bash
    python -m venv venv
    ```

2.  **Activate the environment.**
    * **On Windows:**
        ```bash
        venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install the required Python libraries.**
    Create a `requirements.txt` file with the following contents:

    ```
    fastmcp
    asyncpg
    python-dotenv
    Pillow
    pytesseract
    scipy
    numpy
    httpx
    twilio
    ```

    Then, install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Tesseract OCR Installation

You must have Tesseract OCR installed on your system.

* **Windows:** Download the installer from the [Tesseract-OCR GitHub page](https://github.com/UB-Mannheim/tesseract/wiki).
* **macOS/Linux:** Use a package manager (`brew install tesseract` or `sudo apt-get install tesseract-ocr`).
* **Update Tesseract path:** The code includes a fix for Windows, but if Tesseract is not in your system's PATH, you may need to update the `pytesseract.pytesseract.tesseract_cmd` path in `mcp_starter.py`.

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```
Then edit .env and add your details:

Code snippet
```bash
AUTH_TOKEN=your_secret_token_here
MY_NUMBER=your_phone_number_with_country_code
OPENROUTER_API_KEY=your_openrouter_api_key

POSTGRES_URL="postgresql://user:password@host:port/database"

TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
``` 
## Important Notes:

- AUTH_TOKEN: Your secret token for authentication.

- MY_NUMBER: Your WhatsApp number in format {country_code}{number} (e.g., 919876543210).

- POSTGRES_URL: Your PostgreSQL database connection string.

- TWILIO_*: Your Twilio credentials for WhatsApp integration.

### Step 4: Database Setup
Your code will automatically create the necessary tables (prescriptions and blood_reports) on the first run. However, if you've run the server before adding the new reminder features, you must manually add the is_medicine_taken column.

Connect to your PostgreSQL database.

Run the following SQL command:
```bash

ALTER TABLE prescriptions
ADD COLUMN is_medicine_taken BOOLEAN DEFAULT FALSE;

```
### Step 5: Run the Server

```bash
cd mcp-bearer-token
python mcp_starter.py
```
You'll see: 🚀 Starting MCP server with PostgreSQL and Twilio integration...

### Step 6: Make It Public (Required by Puch)
Since Puch needs to access your server over HTTPS, you need to expose your local server:

Option A: Using ngrok (Recommended)
1. Install ngrok:
Download from https://ngrok.com/download

2. Get your authtoken:

Go to https://dashboard.ngrok.com/get-started/your-authtoken

Copy your authtoken
```bash
Run: ngrok config add-authtoken YOUR_AUTHTOKEN
```
3. Start the tunnel:

```bash
ngrok http 8086
```

## Option B: Deploy to Cloud
You can also deploy this to services like:

Railway

Render

Heroku

DigitalOcean App Platform
