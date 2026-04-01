FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the exact same versions we used for training
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    "torchvision>=0.16.0" \
    "segmentation-models-pytorch" \
    "albumentations" \
    "opencv-python-headless<4.10" \
    "transformers==4.30.2" \
    "tqdm"

# --- THE "AIR-GAP" FIX ---
# Copy the local config folder you created with save_model.py
COPY segformer_b3_config /app/segformer_b3_config

# Copy your code and the TRAINED model weights
COPY predict.py /app/predict.py
COPY model.py /app/model.py
COPY model.pt /app/model.pt

# Ensure the output directory exists inside the container
RUN mkdir -p /output

ENTRYPOINT ["python", "/app/predict.py"]