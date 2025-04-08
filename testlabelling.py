'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the classification model
model = load_model('/Users/krishnandanpandit/Desktop/oral_cancer/best_model_resnet50_optimizedfinal.keras')

# Object detection model (Using YOLO as an example, but you can use any model)
# You can use pre-trained YOLO weights, here it's assumed you have YOLO or another detection model loaded.
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_net.getLayers()]

# Load and preprocess the input image for classification
def preprocess_image_for_classification(image_path):
    img = image.load_img(image_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array

# Load and preprocess the input image for detection (Resize for YOLO)
def preprocess_image_for_detection(image_path):
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    return img, blob, height, width

# Function to classify whether the image shows cancer or not
def classify_image(image_path):
    img_array = preprocess_image_for_classification(image_path)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)
    if class_idx != 0:  # If it's not 'healthy', consider it as cancer
        return True, class_idx
    return False, class_idx

# Function to detect cancerous areas (if detected) and circle them
def detect_and_circle(image_path):
    cancer_detected, class_idx = classify_image(image_path)
    
    if cancer_detected:
        # If cancer is detected, perform object detection to find affected area
        img, blob, height, width = preprocess_image_for_detection(image_path)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(output_layers)

        # Post-process the YOLO outputs to find bounding boxes
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes around detected regions
        img = cv2.imread(image_path)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with the bounding boxes (highlighting the affected areas)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Cancer Area - Class {class_idx[0]}")
        plt.axis('off')
        plt.show()
    else:
        print("No cancer detected in the image.")

# Example usage
image_path = '/Users/krishnandanpandit/Desktop/oral_cancer/train_dir/stage_4/009_aug_2.jpg'
detect_and_circle(image_path)'''

from ultralytics import YOLO
import cv2
model=YOLO('../Yolo-Weights/yolov8n.pt')
result=model("/Users/krishnandanpandit/Downloads/CekLidah/valid/images/tongue.jpg",show=True)
cv2.waitKey(0)
