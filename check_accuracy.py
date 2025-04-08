'''import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
data_dir = "dataset/staged_dataset"  # Replace with your dataset path
model_path = "oral_cancer_model_efficientnetb0.h5"  # Replace with your trained model path

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 300,300
BATCH_SIZE = 32

# Load the trained model (try loading without compile first)
try:
    model = tf.keras.models.load_model(model_path, compile=False)  # Disable compilation initially
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Manually compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing
datagen = ImageDataGenerator(rescale=1.0 / 255)  # Normalize images

# Load test dataset
test_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Ensure predictions align with file order
)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_gen)

# Print the results
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")'''

'''import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

# Load the model
try:
    model_path = "best_model_resnet50_optimized.keras"
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load test dataset
try:
    # Replace the following with your dataset loading logic
    # For example:
    # X_test = np.load("X_test.npy")  # Test images
    # y_test = np.load("y_test.npy")  # True labels (one-hot encoded or categorical)
    
    # Example placeholders:
    X_test = np.random.rand(100,300, 300, 3)  # Replace with your test images
    y_test = np.random.randint(0, 2, size=(100,))  # Replace with true labels for binary classification

    print(f"Test dataset loaded. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
except Exception as e:
    print(f"Error loading test dataset: {e}")
    exit()

# Preprocessing (if necessary)
# Ensure the test images are normalized as the model expects (e.g., pixel values between 0 and 1)
X_test = X_test / 255.0

# Make predictions
try:
    print("Generating predictions...")
    y_pred_prob = model.predict(X_test, batch_size=16)  # Use batch_size to avoid memory issues
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels
    print("Predictions completed.")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# Convert true labels if they are one-hot encoded
if len(y_test.shape) > 1 and y_test.shape[1] > 1:  # Check if one-hot encoded
    y_true = np.argmax(y_test, axis=1)
else:
    y_true = y_test

# Calculate metrics
try:
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # F1-Score
    f1 = f1_score(y_true, y_pred, average="weighted")  # Use "weighted" for multiclass, "binary" for binary
    print(f"F1-Score: {f1:.2f}")

    # Classification report
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
except Exception as e:
    print(f"Error calculating metrics: {e}")'''
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

# Constants
IMAGE_SIZE = (300, 300)  # The input image size (must match the one used during training)
BATCH_SIZE = 32
VAL_DIR = '/Users/krishnandanpandit/Desktop/oral_cancer/val_dir'  # Path to the validation dataset directory

# Custom Cast layer (updated to avoid dtype serialization issue)
class Cast(Layer):
    def __init__(self, dtype=tf.float32, **kwargs):
        super(Cast, self).__init__(**kwargs)
        self._dtype = dtype  # Store dtype in an internal attribute

    def call(self, inputs):
        # Cast the input tensor to the specified dtype
        return tf.cast(inputs, self._dtype)
    
    def get_config(self):
        # Save the configuration including dtype as a string, not as a direct attribute
        config = super(Cast, self).get_config()
        config.update({"dtype": str(self._dtype)})  # Save dtype as a string
        return config

    @classmethod
    def from_config(cls, config):
        # Ensure that the dtype is loaded properly as a string
        dtype = tf.dtypes.as_dtype(config['dtype']).as_numpy_dtype
        return cls(dtype=dtype)

# 1. Load the model
model_path = '/Users/krishnandanpandit/Desktop/oral_cancer/augmented_images/optimized_resnet50_modelfinal.h5'
model = load_model(model_path, custom_objects={'Cast': Cast})

# 2. Data Preprocessing and Validation Data Generator
# Prepare the validation data pipeline
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# Create the validation data generator
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Important: No shuffle for validation to keep order consistent for metrics
)

# 3. Evaluate the model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# 4. Make Predictions
# Collect all predictions from the validation generator
y_true = val_generator.classes  # True labels
y_pred = np.argmax(model.predict(val_generator, verbose=1), axis=-1)  # Predicted labels

# 5. Compute the accuracy score
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

# 6. Compute the F1 Score (Weighted)
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")

# 7. Compute the Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# 8. Plot Confusion Matrix
class_names = list(val_generator.class_indices.keys())  # Class labels from generator
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# 9. Classification Report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", report)
