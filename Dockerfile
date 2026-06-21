# Use a slim Python base image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Start the FastAPI server
# Railway provides the PORT env variable; default to 8000 locally
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}