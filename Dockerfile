FROM python:3.9-slim

# Install minimal system dependencies (no Tesseract needed - using AI vision)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
