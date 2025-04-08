import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# Paths
base_dir = "dataset/cancer"  # Replace with your image folder
output_dir = "dataset/staged_dataset"  # Replace with your output folder
metadata_file = "dataset/generated_metadata.csv"  # Metadata output file

# Create output directories for stages
stages = [1, 2, 3, 4]
for stage in stages:
    os.makedirs(os.path.join(output_dir, f"stage_{stage}"), exist_ok=True)

# Load pre-trained Mask R-CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

model.eval().to(device)
# Image preprocessing
'''transform = transforms.Compose([
    transforms.ToTensor(),
])'''
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512
    transforms.ToTensor(),
])


# Function to calculate tumor size
def calculate_tumor_size(mask):
    """Calculate the tumor size based on the segmented mask area."""
    return np.sum(mask)

# Function to determine cancer stage based on tumor size
def determine_stage(size):
    """Determine cancer stage based on tumor size thresholds."""
    if size < 5000:       # Small tumor
        return 1
    elif size < 20000:    # Medium tumor
        return 2
    elif size < 50000:    # Large tumor
        return 3
    else:                 # Very large tumor
        return 4

# Initialize metadata list
metadata = []

# Process images in the dataset
for image_name in os.listdir(base_dir):
    image_path = os.path.join(base_dir, image_name)
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device).unsqueeze(0)
    
    # Perform tumor segmentation
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Extract the largest mask (assumed to correspond to the tumor)
    masks = outputs[0]['masks'].cpu().numpy() if 'masks' in outputs[0] else []
    if len(masks) > 0:
        largest_mask = masks[0, 0]  # Use the first detected mask
        tumor_size = calculate_tumor_size(largest_mask > 0.5)  # Binarize mask
    else:
        tumor_size = 0  # No tumor detected
    
    # Determine cancer stage based on tumor size
    stage = determine_stage(tumor_size)
    
    # Add metadata entry
    metadata.append({
        "image_name": image_name,
        "tumor_size": tumor_size,
        "stage": stage
    })
    
    # Save the image into the corresponding stage folder
    dest_path = os.path.join(output_dir, f"stage_{stage}", image_name)
    shutil.copy(image_path, dest_path)

# Save metadata to a CSV file
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(metadata_file, index=False)

print(f"Processing complete! Metadata saved to {metadata_file}.")
print(f"Processing: {image_name}")

