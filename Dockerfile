FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY model-registry ./model-registry

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    TOLLAMA_HOST=0.0.0.0:11435

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY model-registry ./model-registry

RUN useradd --create-home --shell /bin/bash tollama \
    && chown -R tollama:tollama /app

USER tollama

EXPOSE 11435

CMD ["tollamad"]
