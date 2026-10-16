FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV & PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (code and model weights)
COPY . .

EXPOSE 7860

# Run with a single Gunicorn worker (models are large and loaded at import time),
# bound to Railway's injected PORT.
CMD ["sh", "-c", "gunicorn flask_app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 120"]
