import os
import shutil
import random

def copy_images_from_subfolders(source_folder, destination_folder, num_images_per_subfolder=50):
    current_dir = os.getcwd()
    
    # Construct absolute paths for source and destination folders
    source_folder = os.path.normpath(os.path.join(current_dir, source_folder))
    destination_folder = os.path.normpath(os.path.join(current_dir, destination_folder))
    print(source_folder)
    print(destination_folder)

    # Iterate through each subfolder in the source folder
    for root, dirs, files in os.walk(source_folder):
        # Check if there are any image files in the current subfolder
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        if len(image_files) > 0:
            # Determine how many images to copy from this subfolder
            num_to_copy = min(num_images_per_subfolder, len(image_files))
            
            # Select num_to_copy random images if needed
            images_to_copy = random.sample(image_files, num_to_copy)
            
            os.makedirs(destination_folder, exist_ok=True)
            
            # Copy each selected image to the destination subfolder
            for image in images_to_copy:
                source_path = os.path.join(root, image)
                destination_path = os.path.join(destination_folder, image)
                shutil.copy(source_path, destination_path)
                print(f"Copied {image} to {destination_path}")

# Example usage:
source_folder = '../wikiart'
destination_folder = 'lost_paintings_training'
print('hi')
copy_images_from_subfolders(source_folder, destination_folder, num_images_per_subfolder=50)
