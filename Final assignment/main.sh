#!/bin/bash
# 1. Point to your local library folder
export PYTHONPATH=$PYTHONPATH:/home/scur2428/.local/lib/python3.10/site-packages

# 2. Tell W&B exactly where to write its files
export WANDB_DIR="./wandb_logs"

# 3. Force install libraries
# Update line 9 in main.sh to include transformers
pip install --user --no-cache-dir "numpy<2.0.0" "wandb==0.12.21" "segmentation-models-pytorch" "albumentations" "opencv-python-headless<4.10" "transformers"

# 4. Run the training
python3 train.py \
    --data-dir ./data \
    --batch-size 16 \
    --epochs 3 \
    --lr 0.00006 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer-b0-training"