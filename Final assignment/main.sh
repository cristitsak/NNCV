#!/bin/bash

# Run the training
python3 train.py \
    --data-dir ./data \
    --batch-size 64 \
    --epochs 3 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-training"