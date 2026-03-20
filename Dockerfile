FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies
COPY pyproject.toml ./
RUN uv pip install --system .

# Copy source and data
COPY src ./src
COPY data ./data

# Create index directory
RUN mkdir -p data/index

ENV PYTHONUNBUFFERED=1
ENV OPENAI_TIMEOUT_SECONDS=8

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
