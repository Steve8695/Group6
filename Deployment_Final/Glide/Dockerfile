# Base image with PyTorch + CUDA 11.8 support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first (for Docker layer caching)
COPY requirements.txt .

# Install system packages needed for image processing and Git
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files (app.py, inference.py, templates/, static/, etc.)
COPY . .

# Optional: Improve GPU performance
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Expose Flask port
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
