import os
# Define the directories
normal_dir = 'pancreatic_tumor'#path for the normal images
#tumor_dir = 'pancreatic_tumor' #path for the tumor images

# Function to rename images in a directory
def rename_images(image_dir, label):
    images = os.listdir(image_dir)# List all the images in the directory 
    # Iterate over the images and rename them
    for i, image_name in enumerate(images, 1):
        file_extension = image_name.split('.')[-1]
        # Create a new name using the label and an index
        new_name = f"{label}_{i:03d}.{file_extension}"  # e.g., normal_001.jpg
        # Get the full paths
        old_path = os.path.join(image_dir, image_name)
        new_path = os.path.join(image_dir, new_name)
        # Rename the file
        os.rename(old_path, new_path)
    print(f"Renamed {len(images)} images in {image_dir}")
# Rename images in both directories
rename_images(normal_dir, 'tumor')
