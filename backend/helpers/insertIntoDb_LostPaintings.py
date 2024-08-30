import pymongo
import os

def extract_painting_info(filename):

    # Remove any file extensions at the end
    # Remove any "(X)" at the end of the filename
    filename = filename.rsplit(".", 1)[0]
    
    # Split filename by "_" to separate artist and painting
    parts = filename.split("_")

    painting_name = parts[0]
    artist_name = parts[1]

    return artist_name, painting_name


client = pymongo.MongoClient("mongodb://mongo:27017/")
db = client["arti"]
collection = db["lost_paintings"]

image_dir = "../lost_paintings"

# Iterate through each image file in the lost paintings directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".gif"):
        # Read the image file
        with open(os.path.join(image_dir, filename), "rb") as f:
            image_data = f.read()

        artist_name, painting_name = extract_painting_info(filename)
        # Create a document to insert into the collection
        image_document = {
            "filename": filename,
            "data": image_data,
            "paintingName": painting_name,  # Provide the painting name
            "artistName": artist_name,    # Provide the artist's name
            "vectorRepresentation": "",  # Empty initial entry
            "isOriginal": True,
            "isLost": True,
            "cnnUsage": None,
            "siameseUsage": None,
            "dataOrigin": "dataset"
        }

        # Insert the document into the collection
        collection.insert_one(image_document)
        print(f"Inserted {filename} into MongoDB")

print("All images inserted into MongoDB successfully.")
