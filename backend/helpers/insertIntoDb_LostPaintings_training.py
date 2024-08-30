import pymongo
import os

client = pymongo.MongoClient("mongodb://mongo:27017/")
db = client["arti"]
collection = db["lost_paintings_training"]

image_dir = "../lost_paintings_training"

# Iterate through each image file in the lost paintings directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".gif"):
        # Read the image file
        with open(os.path.join(image_dir, filename), "rb") as f:
            image_data = f.read()

        # Create a document to insert into the collection
        image_document = {
            "filename": filename,
            "data": image_data,
            "vectorRepresentation": "",  # Empty initial entry
            "siameseUsage": None,
            "dataOrigin": "dataset"
        }

        # Insert the document into the collection
        collection.insert_one(image_document)
        print(f"Inserted {filename} into MongoDB")

print("All images inserted into MongoDB successfully.")
