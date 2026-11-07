
FROM python:3.11-slim

# Configuration Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONPATH=/app/src

WORKDIR /app

# Dépendances système nécessaires à OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier le projet
COPY . .

# Installer uv
RUN pip install --no-cache-dir uv

# Installer les dépendances du projet sans installer torch/torchvision
RUN uv sync --no-dev \
    --no-install-package torch \
    --no-install-package torchvision

# Installer PyTorch et Torchvision en version CPU
RUN uv pip install \
    --python /opt/venv/bin/python \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["/opt/venv/bin/uvicorn", "lightshift.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
