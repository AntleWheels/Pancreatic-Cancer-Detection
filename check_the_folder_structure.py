import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set dataset paths (update these if needed)
X_train_path = "pancreatic_cancer_dataset/Dataset/X_train"
Y_train_cancer_path = "pancreatic_cancer_dataset/Dataset/Mask/Y_train_Cancer"
Y_train_organ_path = "pancreatic_cancer_dataset/Dataset/Mask/Y_train_Organ"

# Get list of images (assuming they are sorted in the same order)
x_train_images = sorted(os.listdir(X_train_path))
y_train_cancer_images = sorted(os.listdir(Y_train_cancer_path))
y_train_organ_images = sorted(os.listdir(Y_train_organ_path))

# Select a random index for visualization
index = np.random.randint(0, len(x_train_images))

# Load MRI image
x_img = cv2.imread(os.path.join(X_train_path, x_train_images[index]), cv2.IMREAD_GRAYSCALE)

# Load corresponding mask images
y_cancer_mask = cv2.imread(os.path.join(Y_train_cancer_path, y_train_cancer_images[index]), cv2.IMREAD_GRAYSCALE)
y_organ_mask = cv2.imread(os.path.join(Y_train_organ_path, y_train_organ_images[index]), cv2.IMREAD_GRAYSCALE)

# Resize images if needed (ensure all images have the same dimensions)
target_size = (256, 256)  # Change according to your dataset
x_img = cv2.resize(x_img, target_size)
y_cancer_mask = cv2.resize(y_cancer_mask, target_size)
y_organ_mask = cv2.resize(y_organ_mask, target_size)

# Plot MRI scan and corresponding masks
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(x_img, cmap="gray")
axes[0].set_title("MRI Scan (X_train)")

axes[1].imshow(y_cancer_mask, cmap="gray")
axes[1].set_title("Cancer Mask (Y_train_Cancer)")

axes[2].imshow(y_organ_mask, cmap="gray")
axes[2].set_title("Organ Mask (Y_train_Organ)")

for ax in axes:
    ax.axis("off")

plt.show()
