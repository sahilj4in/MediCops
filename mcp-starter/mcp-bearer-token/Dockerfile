# Use a standard Python runtime as the parent image
FROM python:3.10-slim

# Install system dependencies, including Tesseract OCR and its data files
# The executable will be installed to /usr/bin/tesseract on this base image.
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for the container to the root of the app
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install your Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the TESSERACT_CMD environment variable to the correct path
ENV TESSERACT_CMD=/usr/bin/tesseract

# Set the entry point to run your application
CMD ["python", "mcp_starter.py"]