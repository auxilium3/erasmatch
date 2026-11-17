FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Apply migrations, then serve.
CMD ["sh", "-c", "erasmatch migrate && uvicorn --factory erasmatch.api:create_app --host 0.0.0.0 --port 8000"]
