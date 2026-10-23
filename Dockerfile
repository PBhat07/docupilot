
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/

ARG INSTALL_EXTRAS=dev

RUN pip install --no-cache-dir --default-timeout=300 --retries=5 -e ".[${INSTALL_EXTRAS}]"

EXPOSE 8000

CMD ["uvicorn", "docupilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
