import os
import shutil

import torch

def search_and_copy_images(src_folder, dest_folder, keyword):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Walk through the source folder and its subfolders
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            # Check if the keyword is in the file name
            if file.startswith(keyword):
                # Construct the full path of the source file
                src_file_path = os.path.join(root, file)

                #the style is the parent folder name. add that to the filename for DB insert info
                parent_folder_name = os.path.basename(root).replace("_", "-")

                # Construct the full path of the destination file
                dest_file_path = os.path.join(dest_folder, f"{parent_folder_name}_{file}")

                # Copy the file to the destination folder
                shutil.copy(src_file_path, dest_file_path)

                print(f"Copied {file} to {dest_folder}")

# adjust names of destination folders and search keywords
source_folder = "../../wikiart"     
artist_keyword_list = ["sandro-botticelli", "giovanni-bellini", "salvador-dali", "leonardo-da-vinci", "roy-lichtenstein", "pablo-picasso", "jackson-pollock", "andy-warhol"]
destination_folder_base = "../artist_training_data/"

for keyword in artist_keyword_list:

    destination_folder = destination_folder_base + keyword
    search_and_copy_images(source_folder, destination_folder, keyword)
