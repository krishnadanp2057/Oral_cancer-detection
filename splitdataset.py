import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_dataset_dir = "/Users/krishnandanpandit/Desktop/oral_cancer/augmented_images"  # Path to the original dataset
train_dir = "train_dir"  # Directory for training data
val_dir = "val_dir"  # Directory for validation data

# Create train and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Loop through each class folder (stage_1, stage_2, stage_3, stage_4, non-cancer)
for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    
    if os.path.isdir(class_path):  # Ensure it's a directory
        # Create class folders in train and validation directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Get list of all images in the class
        all_images = os.listdir(class_path)
        
        # Split the dataset
        train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)  # 80-20 split
        
        # Move training images
        for image_name in train_images:
            src = os.path.join(class_path, image_name)
            dst = os.path.join(train_dir, class_name, image_name)
            shutil.copy(src, dst)
        
        # Move validation images
        for image_name in val_images:
            src = os.path.join(class_path, image_name)
            dst = os.path.join(val_dir, class_name, image_name)
            shutil.copy(src, dst)

print("Dataset has been split into training and validation sets.")
print(f"Training data is stored in: {train_dir}")
print(f"Validation data is stored in: {val_dir}")
