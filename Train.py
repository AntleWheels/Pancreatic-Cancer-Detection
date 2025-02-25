import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras import applications
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
# Set Image Dimensions
IMG_SIZE = 256
BATCH_SIZE = 32



# Paths to Image Datasets
TUMOR_DIR = "pancreatic_tumor"   # Path to tumor images
NORMAL_DIR = "normal"            # Path to normal images

# Function to Load Images and Labels
def load_images_from_directory(directory, label):
    images, labels = [], []
    for file in os.listdir(directory):
        img_path = os.path.join(directory, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to uniform size
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load Tumor Images
X_tumor, y_tumor = load_images_from_directory(TUMOR_DIR, label=1)

# Load Normal Images
X_normal, y_normal = load_images_from_directory(NORMAL_DIR, label=0)

# Data Augmentation for Normal & Tumor Images
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_images = []
augmented_labels = []

# Augment Normal Images (7 copies per image)
for img in X_normal:
    img = img.reshape((1,) + img.shape)  # Reshape for augmentation
    for _ in range(7):  
        aug_img = next(datagen.flow(img, batch_size=1))[0]
        augmented_images.append(aug_img)
        augmented_labels.append(0)  # Normal label

# Augment Tumor Images (5 copies per image)
for img in X_tumor:
    img = img.reshape((1,) + img.shape)  
    for _ in range(5):  
        aug_img = next(datagen.flow(img, batch_size=1))[0]
        augmented_images.append(aug_img)
        augmented_labels.append(1)  # Tumor label
# Print Initial Counts
print(f"Original Tumor Images: {len(X_tumor)}")
print(f"Original Normal Images: {len(X_normal)}")

# Print Augmentation Counts
print(f"Augmented Normal Images: {len(augmented_labels) - np.sum(augmented_labels)}")
print(f"Augmented Tumor Images: {np.sum(augmented_labels)}")

# Convert Augmented Data to NumPy Arrays
X_augmented = np.array(augmented_images)
y_augmented = np.array(augmented_labels)

# Merge All Data
X = np.concatenate((X_tumor, X_normal, X_augmented), axis=0)
y = np.concatenate((y_tumor, y_normal, y_augmented), axis=0)

# Normalize Pixel Values
X = X / 255.0  # Scale pixel values to [0,1]

# Split Dataset into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute Class Weights
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print("Computed Class Weights:", class_weights_dict)

# Define Focal Loss Function
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = K.mean(K.binary_crossentropy(y_true, y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = alpha * (1 - p_t) ** gamma * bce
        return loss
    return loss

# Load Pretrained ResNet50 Model
base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze Pretrained Layers

# Build Final CNN Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Prevent Overfitting
    layers.Dense(1, activation='sigmoid')  # Binary Classification Output
])

# Compile Model with Focal Loss
model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# Train Model
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    class_weight=class_weights_dict
)

# Evaluate Model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Save Model
model.save("pancreatic_cancer_detection_model.h5")
