FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the exact same "Sunday-safe" versions we used for training
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    "torchvision>=0.15.0" \
    "segmentation-models-pytorch" \
    "albumentations" \
    "opencv-python-headless<4.10" \
    "wandb==0.12.21"

# Copy your code and the TRAINED model weights
COPY predict.py /app/predict.py
COPY model.py /app/model.py
COPY model.pt /app/model.pt

ENTRYPOINT ["python", "/app/predict.py"]