FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for native extensions (Kuzu/ChromaDB might need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging files and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir .[all]

# Copy application source
COPY . .

# Ensure data directory exists
RUN mkdir -p /app/memai_data

EXPOSE 8000

# Default environment variables
ENV MEMAI_DATA_DIR=/app/memai_data
ENV MEMAI_HOST=0.0.0.0
ENV MEMAI_PORT=8000

# Start the FastAPI server using the CLI
CMD ["memai", "serve"]
