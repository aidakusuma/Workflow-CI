# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment file
COPY conda.yaml .

# Install conda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Create conda environment
RUN conda env create -f conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "weather_prediction_env", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Set environment variables
ENV MLFLOW_TRACKING_URI=file:./mlruns
ENV PYTHONPATH=/app

# Expose MLflow UI port
EXPOSE 5000

# Default command
CMD ["conda", "run", "-n", "weather_prediction_env", "python", "modelling.py"]
