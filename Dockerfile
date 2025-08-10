# Use a Debian base image for better control
FROM debian:bullseye-slim

# Install Python, Tesseract, and dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip tesseract-ocr libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

# Verify Tesseract installation
RUN tesseract --version || { echo "Tesseract installation failed"; exit 1; }

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY mcp-starter/mcp-bearer-token/requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY mcp-starter/mcp-bearer-token .

# Set environment variable for Tesseract
ENV TESSERACT_CMD=/usr/bin/tesseract

# Set the entry point
CMD ["python3", "mcp_starter.py"]