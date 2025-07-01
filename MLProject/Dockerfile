FROM python:3.12-slim

# 1) Install system dependencies & clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2) Set working directory
WORKDIR /app

# 3) Install only required serving dependencies
RUN pip install --no-cache-dir \
    mlflow==2.17.0 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==2.1.2 \
    xgboost==2.1.1 \
    fastapi==0.111.0 \
    uvicorn==0.29.0

# 4) Copy only necessary model files (optimized in CI)
COPY model /app/model

# 5) Expose ports
EXPOSE 8080

# 6) Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 7) Entrypoint with fallback
ENTRYPOINT ["sh", "-c"]
CMD ["mlflow models serve --model-uri /app/model --no-conda --host 0.0.0.0 --port 8080"]
