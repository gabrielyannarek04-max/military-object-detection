# Build from the repository root:
#   docker build -f deployment/Dockerfile -t military-detection-api .
# Run with the weights mounted:
#   docker run -p 8000:8000 -v "$(pwd)/models:/app/models" military-detection-api
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/main.py .

ENV MODEL_PATH=/app/models/best.pt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
