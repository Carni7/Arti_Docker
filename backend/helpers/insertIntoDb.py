import pymongo
import os

def extract_painting_info(filename):

    # Remove any file extensions at the end
    # Remove any "(X)" at the end of the filename
    filename = filename.split(".")[0].split("(")[0]
    
    # Split filename by "_" to separate artist and painting
    parts = filename.split("_")
    
    style = parts[0]
    artist_name = parts[1]
    painting_name = parts[2]

    
    # Replace "-" with spaces and start each word with upper case
    style = style.replace("-", " ").title()
    artist_name = artist_name.replace("-", " ").title()
    painting_name = painting_name.replace("-", " ").title()
    
    return style, artist_name, painting_name


client = pymongo.MongoClient("mongodb://mongo:27017/")
db = client["arti"]
collection = db["paintings"]

# Specify the directory containing images
artist_dirs = ["sandro-botticelli", "giovanni-bellini", "salvador-dali", "leonardo-da-vinci", "roy-lichtenstein", "pablo-picasso", "jackson-pollock", "andy-warhol"]
image_dir_base = "../artist_training_data/"

for artist in artist_dirs:
    image_dir = image_dir_base + artist
    # Iterate through each image file in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image file
            with open(os.path.join(image_dir, filename), "rb") as f:
                image_data = f.read()

            style, artist_name, painting_name = extract_painting_info(filename)
            # Create a document to insert into the collection
            image_document = {
                "filename": filename,
                "data": image_data,
                "paintingName": painting_name,  # Provide the painting name
                "artistName": artist_name,    # Provide the artist's name
                "style": style,         # Provide the painting style
                "vectorRepresentation": "",  # Empty initial entry
                "isOriginal": True,
                "isLost": False,
                "cnnUsage": None,
                "siameseUsage": None,
                "dataOrigin": "dataset"
            }

            # Insert the document into the collection
            collection.insert_one(image_document)
            print(f"Inserted {filename} into MongoDB")

print("All images inserted into MongoDB successfully.")
