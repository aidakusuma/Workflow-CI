FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY conda.yaml .
COPY modelling.py .
COPY weatherdata_preprocessing.csv* ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2

# Set environment variable for CI
ENV CI=true

# Run the model
CMD ["python", "modelling.py"]
