# Use Python slim image based on Debian Bullseye
FROM python:3.10-slim-bullseye

# Install Tesseract and dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

# Verify Tesseract installation
RUN tesseract --version || { echo "Tesseract installation failed"; exit 1; }

# Check if binary exists
RUN ls /usr/bin | grep tesseract || { echo "Tesseract binary not found"; exit 1; }

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable
ENV TESSERACT_CMD=/usr/bin/tesseract

# Run the application
CMD ["python", "mcp_starter.py"]