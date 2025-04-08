from google.colab import drive
import zipfile
import os

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Verify File Path
# Check if the file exists in the provided path
zip_file_path = '/content/drive/My Drive/Archive.zip'  # Update with the correct path

if not os.path.exists(zip_file_path):
    raise FileNotFoundError(f"The file {zip_file_path} does not exist. Please check the path.")

# Step 3: Extract the zip file
destination_path = '/content/dataset'  # Directory where the files will be extracted

# Create the destination directory if it doesn't exist
os.makedirs(destination_path, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(destination_path)

print(f"Files extracted to: {destination_path}")

# Optional: List the files in the extracted directory
for root, dirs, files in os.walk(destination_path):
    for file in files:
        print(os.path.join(root, file))


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Constants
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 64
NUM_CLASSES = 5
EPOCHS_INITIAL = 30
EPOCHS_FINE = 30
TRAIN_SPLIT = 0.8

# Enable mixed precision training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Dataset directory
DATASET_DIR = '//content/dataset/train_dir'

# Helper functions
def preprocess_image(file_path, label):
    """Reads and preprocesses an image."""
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

def augment_image(image, label):
    """Applies data augmentation."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

def create_dataset(image_paths, labels, batch_size, augment=False, shuffle=True):
    """Creates a tf.data dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Load dataset
image_paths = glob(f"{DATASET_DIR}/*/*.[jp]*[ge]*[g]*")  # This will match jpg, jpeg, png, etc.

labels = [path.split('/')[-2] for path in image_paths]
label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
labels = [label_to_index[label] for label in labels]

# One-hot encode the labels
labels_one_hot = to_categorical(labels, num_classes=NUM_CLASSES)

# Train-validation split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels_one_hot, test_size=(1 - TRAIN_SPLIT), stratify=labels, random_state=42
)

# Create datasets
train_dataset = create_dataset(train_paths, train_labels, BATCH_SIZE, augment=True)
val_dataset = create_dataset(val_paths, val_labels, BATCH_SIZE, augment=False, shuffle=False)

# Define the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
base_model.trainable = False  # Freeze base model initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=AdamW(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
checkpoint = ModelCheckpoint('/content/best_model_resnet50_optimizedfinal.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# Initial training
history_initial = model.fit(
    train_dataset,
    epochs=EPOCHS_INITIAL,
    validation_data=val_dataset,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1
)

# Gradual unfreezing
for layer in base_model.layers[-100:]:
    layer.trainable = True

# Recompile for fine-tuning
model.compile(optimizer=AdamW(learning_rate=1e-5),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(
    train_dataset,
    epochs=EPOCHS_FINE,
    validation_data=val_dataset,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Predictions and evaluation
y_true = np.concatenate([y.numpy() for _, y in val_dataset])
y_pred = np.argmax(model.predict(val_dataset), axis=-1)

accuracy = accuracy_score(np.argmax(y_true, axis=1), y_pred)
f1 = f1_score(np.argmax(y_true, axis=1), y_pred, average='weighted')
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
class_labels = sorted(label_to_index.keys())
conf_matrix = confusion_matrix(np.argmax(y_true, axis=1), y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification Report
report = classification_report(np.argmax(y_true, axis=1), y_pred, target_names=class_labels)
print("Classification Report:\n", report)

# Save the model in .h5 format
model.save('/content/optimized_resnet50_modelfinal.h5')
print("Model saved successfully in .h5 format.")
