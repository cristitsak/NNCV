#!/bin/bash

# 1. Setup Python paths
export PYTHONPATH=$PYTHONPATH:/home/scur2428/.local/lib/python3.10/site-packages
export WANDB_DIR="./wandb_logs"

# 2. Smart Install: Only install if 'transformers' is missing
# This saves time and avoids "Externally Managed Environment" errors on some nodes
if ! python3 -c "import transformers" &> /dev/null; then
    echo "Installing missing libraries..."
    pip install --user --no-cache-dir \
        "numpy<2.0.0" \
        "wandb==0.12.21" \
        "segmentation-models-pytorch" \
        "albumentations" \
        "opencv-python-headless<4.10" \
        "transformers"
fi

# 3. Run the training
# Make sure your train.py is set up to save 'model.pt' at the end
python3 train.py \
    --data-dir ./data \
    --batch-size 8 \
    --epochs 60 \
    --lr 6e-5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer-b3-training"