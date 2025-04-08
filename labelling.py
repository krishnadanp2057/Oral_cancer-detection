'''from tkinter import Tk, Canvas, Button  # Fix the import to include Button
from PIL import Image, ImageTk
import os

# Initialize the main application window
root = Tk()
root.title("Oral Cancer Image Labelling")
root.geometry("800x600")  # Window size

file_path = "train_dir/stage_4/pre_cancer.498.jpg"  # Path to your image

def save_image_with_labels():
    """Save the labelled image."""
    print("Saving labelled image...")
    canvas.postscript(file="labelled_image.eps")
    # Convert postscript to PNG
    img = Image.open("labelled_image.eps")
    img.save("labelled_image.png", "PNG")
    print("Image saved as labelled_image.png")

def add_text_label(event):
    """Add text to the image on click."""
    x, y = event.x, event.y
    text = "Stage 4"  # Example label, can be changed based on user input
    canvas.create_text(x, y, text=text, fill="red", font=('Helvetica', 12, 'bold'))

def add_rectangle_label(event):
    """Add rectangle annotation to the image on click."""
    x, y = event.x, event.y
    size = 50  # Size of the rectangle (can adjust)
    canvas.create_rectangle(x, y, x + size, y + size, outline="yellow", width=2)

try:
    # Check if the image exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    # Open the image using PIL
    image = Image.open(file_path)
    
    # Resize the image
    image = image.resize((800, 600), Image.Resampling.LANCZOS)
    
    # Convert image to Tkinter-compatible format
    photo = ImageTk.PhotoImage(image)

    # Create a Canvas to display the image
    canvas = Canvas(root, width=800, height=600)
    canvas.pack()

    # Display the image on the canvas
    canvas.create_image(0, 0, anchor="nw", image=photo)

    # Set up event listeners to add text or rectangles on click
    canvas.bind("<Button-1>", add_text_label)  # Left click for text labels
    canvas.bind("<Button-3>", add_rectangle_label)  # Right click for rectangle labels

    # Create a Button to save the labelled image
    save_button = Button(root, text="Save Labelled Image", command=save_image_with_labels)
    save_button.pack()

    # Start the Tkinter event loop
    root.mainloop()

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")'''

import cv2
import numpy as np

# Load image
image = cv2.imread('i/Users/krishnandanpandit/Desktop/oral_cancer/augmented_train_dr/stage_1/024_aug_0.jpg')

# Apply Gaussian blur (common method to reduce noise)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Denoising using Non-Local Means Denoising (a good method for removing noise)
denoised_image = cv2.fastNlMeansDenoisingColored(blurred_image, None, 10, 10, 7, 21)

# Save the denoised image
cv2.imwrite('denoised_image.jpg', denoised_image)

