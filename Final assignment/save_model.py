import os
from transformers import SegformerForSemanticSegmentation

# 1. Define the model name exactly as it is in your model.py
model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"

# 2. Define where you want to save it
save_directory = "./segformer_b3_config"

print(f"Downloading and saving {model_name} to {save_directory}...")

# 3. Download the model and config from Hugging Face
# We set num_labels=19 to match the Cityscapes classes
model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=19,
    ignore_mismatched_sizes=True
)

# 4. Save the files to the folder
model.save_pretrained(save_directory)

print("Done! You should now see a 'segformer_b3_config' folder in your directory.")