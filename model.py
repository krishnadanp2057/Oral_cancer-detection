'''import os #the code with 72% accuracy
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers

# Paths
data_dir = "dataset/staged_dataset"
model_save_path = "cancer_stage_model.h5"

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into training and validation
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load pre-trained EfficientNetB0 with transfer learning
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze base layers

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save the model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")'''
'''import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


import tensorflow as tf

# Check if a GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("GPU not available.")
else:
    print(f"Available GPUs: {physical_devices}")

import tensorflow as tf

# Enable device placement logging
tf.debugging.set_log_device_placement(True)


# Create a small tensor to see the logs
x = tf.random.normal([1000, 1000])
y = tf.matmul(x, x)


# Constants
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS_INITIAL = 1  # Initial training (frozen base)
EPOCHS_FINE = 1     # Fine-tuning (unfrozen base)
NUM_CLASSES = 5
TRAIN_DIR = 'train_dir'  # Replace with your training directory
VAL_DIR = 'val_dir'      # Replace with your validation directory

# Validate images
def validate_images(directory):
    print(f"Validating images in {directory}...")
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify image integrity
            except (IOError, SyntaxError) as e:
                print(f"Corrupted image detected: {file_path}. Error: {e}")
                os.remove(file_path)

validate_images(TRAIN_DIR)
validate_images(VAL_DIR)

# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # EfficientNet preprocessing
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Safe data generator to avoid crashes
def safe_data_generator(generator):
    while True:
        try:
            yield next(generator)
        except Exception as e:
            print(f"Data generation error: {e}")
            continue

train_generator_safe = safe_data_generator(train_generator)
val_generator_safe = safe_data_generator(val_generator)

# Compute Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

# Weighted loss function
def weighted_categorical_crossentropy(class_weights):
    def loss_fn(y_true, y_pred):
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        unweighted_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return unweighted_loss * weights
    return loss_fn

# Define Model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
base_model.trainable = False  # Freeze base model initially

# Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile Model
loss_function = weighted_categorical_crossentropy(class_weights_tensor)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=loss_function,
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model_efficientnetb3.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train with Frozen Base
history_initial = model.fit(
    train_generator_safe,
    epochs=EPOCHS_INITIAL,
    validation_data=val_generator_safe,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Unfreeze Base Model for Fine-Tuning
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss=loss_function,
              metrics=['accuracy'])

# Fine-Tune Model
history_fine = model.fit(
    train_generator_safe,
    epochs=EPOCHS_FINE,
    validation_data=val_generator_safe,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Save Final Model
model.save('oral_cancer_model_efficientnetb3.h5')

# Evaluate Model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Generate Confusion Matrix
y_true = val_generator.classes
y_pred = np.argmax(model.predict(val_generator), axis=-1)
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_names = list(train_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", report)'''
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# Constants
IMAGE_SIZE = (300, 300)  # Image dimensions (height, width)
BATCH_SIZE = 32
EPOCHS_INITIAL = 15  # Increased epochs for better convergence
EPOCHS_FINE = 15   # Increased fine-tuning epochs
NUM_CLASSES = 5
TRAIN_DIR = 'train_dir'  # Replace with your training directory
VAL_DIR = 'val_dir'      # Replace with your validation directory

# Check if a GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("GPU not available.")
else:
    print(f"Available GPUs: {physical_devices}")

# Enable device placement logging
# tf.debugging.set_log_device_placement(True)

# Validate images
def validate_images(directory):
    print(f"Validating images in {directory}...")
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify image integrity
            except (IOError, SyntaxError) as e:
                print(f"Corrupted image detected: {file_path}. Error: {e}")
                os.remove(file_path)

validate_images(TRAIN_DIR)
validate_images(VAL_DIR)

# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # EfficientNet preprocessing
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Safe data generator to avoid crashes
def safe_data_generator(generator):
    while True:
        try:
            yield next(generator)
        except Exception as e:
            print(f"Data generation error: {e}")
            continue

train_generator_safe = safe_data_generator(train_generator)
val_generator_safe = safe_data_generator(val_generator)

# Compute Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

# Weighted loss function
def weighted_categorical_crossentropy(class_weights):
    def loss_fn(y_true, y_pred):
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        unweighted_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return unweighted_loss * weights
    return loss_fn

# Define Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
base_model.trainable = False  # Freeze base model initially

# Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile Model
loss_function = weighted_categorical_crossentropy(class_weights_tensor)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=loss_function,
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model_efficientnetb0.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# Train with Frozen Base
history_initial = model.fit(
    train_generator_safe,
    epochs=EPOCHS_INITIAL,
    validation_data=val_generator_safe,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
    layer.trainable = True

# Recompile model after unfreezing layers
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss=loss_function,
              metrics=['accuracy'])

# Fine-tune Model
history_fine = model.fit(
    train_generator_safe,
    epochs=EPOCHS_FINE,
    validation_data=val_generator_safe,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Save Final Model
model.save('oral_cancer_model_efficientnetb0.h5')

# Evaluate Model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Generate Confusion Matrix
y_true = val_generator.classes
y_pred = np.argmax(model.predict(val_generator), axis=-1)
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_names = list(train_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", report)

