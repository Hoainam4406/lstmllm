# Build optimized Docker image for LSTM + LLM Ablation Study
# Supports both local and Kaggle deployment

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=0.0.0.0:11434

# Install system dependencies (minimal - headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenmpi-dev \
    libgomp1 \
    wget \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies (CPU-only PyTorch to save space)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy all project files
COPY *.py ./
COPY *.md ./
COPY *.txt ./

# Create results directory
RUN mkdir -p /app/results/ablation_study/plots

# Download Ollama binary
RUN wget -q https://ollama.ai/download/ollama-linux-x86_64 -O /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama && \
    mkdir -p /root/.ollama

# Health check for Ollama service
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

# Default command: start Ollama and run pipeline
CMD ["bash", "-c", "ollama serve &>/dev/null & sleep 5 && ollama pull gemma:4b && python ablation_pipeline.py --variants all"]

# Expose Ollama port
EXPOSE 11434
