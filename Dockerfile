# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies, including Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# Set the working directory to the directory containing your application code
WORKDIR /app/mcp-starter/mcp-bearer-token

# Copy the requirements file to the working directory
COPY mcp-starter/mcp-bearer-token/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY mcp-starter/mcp-bearer-token .

# Set the entry point to run your application
CMD ["python", "mcp_starter.py"]