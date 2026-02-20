FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOLLAMA_HOST=0.0.0.0:11435

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY model-registry ./model-registry

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

EXPOSE 11435

CMD ["tollamad"]
