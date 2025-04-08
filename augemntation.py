'''import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from tqdm import tqdm  # For progress tracking

# Define the augmentation pipeline with updated parameters
augmentation_pipeline = A.Compose([
    # Geometric Transformations
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),  # Rotate within ±15°
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
    A.ElasticTransform(alpha=0.3, sigma=20, p=0.3),  # Removed invalid 'alpha_affine' parameter

    # Color and Noise Augmentations
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
    ], p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),

    # Sharpening for image clarity
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.3),  # Fixed parameter ranges

    # Convert to PyTorch Tensor
    ToTensorV2()
])

# Function to apply augmentations
def augment_image(image_path, output_dir, num_augmentations=3):
    """
    Augments a given image multiple times and saves them to the output directory.

    Parameters:
    - image_path: Path to the input image.
    - output_dir: Directory to save augmented images.
    - num_augmentations: Number of augmented images to generate per input image.
    """
    # Check if the file exists and is an image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    valid_image_extensions = ['.jpg', '.jpeg', '.png']
    if os.path.splitext(image_path)[1].lower() not in valid_image_extensions:
        print(f"Skipping non-image file: {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    file_name = os.path.splitext(os.path.basename(image_path))[0]
    for i in range(num_augmentations):
        try:
            # Apply augmentations
            augmented = augmentation_pipeline(image=image)['image']
            
            # Convert Tensor to NumPy
            augmented_image_np = augmented.permute(1, 2, 0).numpy()
            
            # Clip pixel values to a valid range [0, 255] to avoid black spots and overflow
            augmented_image_np = np.clip(augmented_image_np, 0, 255).astype(np.uint8)
            
            # Save the augmented image
            output_path = os.path.join(output_dir, f"{file_name}_aug_{i}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(augmented_image_np, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")

# Main script to apply augmentations to the entire dataset
def main():
    input_dir = "/Users/krishnandanpandit/Desktop/oral_cancer/train_dir"  # Path to your dataset (contains subfolders for each stage)
    output_dir = "augmented_train_dr"  # Path to save augmented dataset
    
    num_augmentations = 3  # Number of augmentations per image

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each stage (subfolder)
    for stage in os.listdir(input_dir):
        stage_input_dir = os.path.join(input_dir, stage)
        stage_output_dir = os.path.join(output_dir, stage)
        
        if not os.path.exists(stage_output_dir):
            os.makedirs(stage_output_dir)
        
        print(f"Processing stage: {stage}")
        for img_file in tqdm(os.listdir(stage_input_dir), desc=f"Augmenting {stage}"):
            img_path = os.path.join(stage_input_dir, img_file)
            augment_image(img_path, stage_output_dir, num_augmentations)

    print(f"Augmentation complete! Augmented images saved in: {output_dir}")

if __name__ == "__main__":
    main()'''
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# Define the directories
input_dir = '/Users/krishnandanpandit/Desktop/oral_cancer/train_dir'  # Directory with subdirectories for each stage
output_dir = '/Users/krishnandanpandit/Desktop/oral_cancer/augmented_images'  # Directory to save augmented images

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define data augmentation techniques with minimal adjustments
datagen = ImageDataGenerator(
    horizontal_flip=True,  # Randomly flip images horizontally (small change)
    brightness_range=[0.98, 1.02],  # Very slight brightness adjustments
    fill_mode='nearest',  # How to fill in pixels that are transformed
    rescale=1./255  # Rescale to normalize pixel values
)

# Function to augment images in the directory and save them
def augment_images(input_dir, output_dir, datagen):
    # Iterate through each subdirectory (representing a cancer stage)
    for stage_dir in os.listdir(input_dir):
        stage_path = os.path.join(input_dir, stage_dir)

        # Only process directories (subdirectories represent stages)
        if os.path.isdir(stage_path):
            print(f"Processing stage: {stage_dir}")  # Debugging line

            # Create corresponding output subdirectory for augmented images
            augmented_stage_dir = os.path.join(output_dir, stage_dir)
            if not os.path.exists(augmented_stage_dir):
                os.makedirs(augmented_stage_dir)

            # Get all image files in the subdirectory
            image_files = [f for f in os.listdir(stage_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(image_files) == 0:
                print(f"No images found in {stage_dir}. Skipping...")
                continue

            # Iterate through each image in the subdirectory
            for image_file in image_files:
                print(f"Processing image: {image_file}")  # Debugging line
                # Load the image
                img_path = os.path.join(stage_path, image_file)
                img = load_img(img_path)
                img_array = img_to_array(img)  # Convert image to array
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Apply augmentation
                augmented_images = datagen.flow(img_array, batch_size=1, save_to_dir=augmented_stage_dir,
                                                save_prefix='aug_', save_format='jpeg')

                # Generate and save augmented images (example: generate 5 augmented images per original image)
                for i in range(5):  # Generate 5 augmented images per original image (to avoid too many noisy images)
                    next(augmented_images)  # This will save the augmented images to the specified directory
                    print(f"Augmented image {i+1} saved for {image_file} in {stage_dir}")  # Debugging line

# Call the function to augment and save images
augment_images(input_dir, output_dir, datagen)

print(f"Augmented images have been saved to {output_dir}")
