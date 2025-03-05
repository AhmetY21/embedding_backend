# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working direc
WORKDIR /app

# Set environment variables
ENV MODEL_CACHE_DIR=/app/model_cache
ENV PYTHONPATH=/app
ENV PORT=8000

# Install git (needed for huggingface model download)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app app/

# Create model cache directory
RUN mkdir -p ${MODEL_CACHE_DIR}

# Download model during build (optional)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='${MODEL_CACHE_DIR}')"

# Expose the port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]