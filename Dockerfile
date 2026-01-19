# Use official Python 3.11 slim as base image (adjust Python version if needed)
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for PyMuPDF and others
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# (Optional) Download spaCy transformer model in Docker build for faster container startup
RUN python -m spacy download en_core_web_trf

# Expose port (must match the one your server listens on)
EXPOSE 8000

# Command to run the FastAPI server with uvicorn
CMD ["uvicorn", "app2:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
