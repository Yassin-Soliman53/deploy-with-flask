FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV & PyTorch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (code and model weights)
COPY . .

# Expose Hugging Face default port
EXPOSE 7860

# Run the app using Gunicorn or Flask built-in server bound to port 7860
CMD ["python", "flask_app.py"]