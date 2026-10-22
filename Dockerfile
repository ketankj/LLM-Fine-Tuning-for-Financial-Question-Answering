# Serving image. For GPU inference use an nvidia/cuda base image and run with --gpus all.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/

# Mount or copy your trained adapter here, e.g. -v ./outputs/.../adapter:/models/adapter
ENV FINQA_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    FINQA_ADAPTER=/models/adapter \
    FINQA_LOAD_4BIT=0

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
