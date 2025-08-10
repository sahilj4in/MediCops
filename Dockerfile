# Use a base image that comes with Tesseract pre-installed.
# This eliminates all apt-get and pathing issues.
FROM tesseract-ocr/tesseract:latest

# The 'tesseract' image is based on a slim Debian distribution,
# so we can use apt-get to install Python.
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to the directory containing your application code
WORKDIR /app

# Copy the requirements file into the working directory
COPY mcp-starter/mcp-bearer-token/requirements.txt .

# Install your Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY mcp-starter/mcp-bearer-token .

# Set the entry point to run your application
CMD ["python3", "mcp_starter.py"]