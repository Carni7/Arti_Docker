from enum import Enum
import random
from matplotlib import transforms
import pymongo
import io
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

class Purpose(Enum):
    TRAINING = 'training'
    TESTING = 'testing'

class MongoDBSiameseImageDataset(Dataset):
    def __init__(self, mongo_uri, db_name, collection_name, artist_name, num_anchor_images=40, num_pairs_per_image=10, transform=None, purpose=Purpose.TRAINING):
        
        if not isinstance(purpose, Purpose):
            raise ValueError("Invalid value for 'purpose'. Must be one of 'Purpose.TRAINING' or 'Purpose.TESTING'.")

        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        # number of positive samples, as well as number of negative ones. total = 2 * num_pairs_per_image
        self.num_pairs_per_image = num_pairs_per_image
        # number of anchor images used for creating pairs. each anchor image gets num_pairs_per_image pairs as anchor
        self.num_anchor_images = num_anchor_images
        self.transform = transform
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]


        # Dictionary to hold image records per artist
        self.artist_image_records = {}

        # Training: Reset all usage fields to None and randomly pick some that are then set to "training" so they are not chosen for testing
        # Setting to testing also enables checking which ones were used in the database
        if purpose is Purpose.TRAINING:
            self.collection.update_many(
                {"artistName": artist_name, "siameseUsage": {"$in": ["training", "testing"]}},
                {"$set": {"siameseUsage": None}}
            )

            artist_records = list(self.collection.find({"artistName": artist_name, "siameseUsage": None}))
            random.shuffle(artist_records)
            artist_records = artist_records[:num_anchor_images]

            self.collection.update_many(
                {"_id": {"$in": [record["_id"] for record in artist_records]}},
                {"$set": {"siameseUsage": "training"}}
            )

        # Testing: Same as for training, but do not reset beforehand so images that were used for training are not chosen here
        else:
            artist_records = list(self.collection.find({"artistName": artist_name, "siameseUsage": None}))
            random.shuffle(artist_records)
            artist_records = artist_records[:num_anchor_images]

            self.collection.update_many(
                {"_id": {"$in": [record["_id"] for record in artist_records]}},
                {"$set": {"siameseUsage": "testing"}}
            )


        # Initialize lists to store training and testing pairs
        self.siamese_pairs = []

        for anchor_record in artist_records:
            similar_records = random.sample(artist_records, num_pairs_per_image)
            dissimilar_records = self.collection.aggregate([
                { "$match": { "artistName": { "$ne": artist_name } } },  # Exclude documents with the current artistName to crate negative pairs
                { "$sample": { "size": num_pairs_per_image } }  # Sample num_pairs_per_image random documents
            ])

            # Create pairs with labels: (record1, record2, label)
            for similar_record in similar_records:
                pair = (anchor_record, similar_record, 1)               
                self.siamese_pairs.append(pair)
            
            for dissimilar_record in dissimilar_records:
                pair = (anchor_record, dissimilar_record, 0)                
                self.siamese_pairs.append(pair)

        # Shuffle the pairs
        random.shuffle(self.siamese_pairs)        

    
    def __len__(self):
        return len(self.siamese_pairs)
    

    def __getitem__(self, idx):

        anchor_record, pair_record, label = self.siamese_pairs[idx]
            
        if self.transform:
            anchor_image_data = anchor_record['data']
            anchor_image_bytes = io.BytesIO(anchor_image_data)
            anchor_image = Image.open(anchor_image_bytes).convert('RGB')
            anchor_image = self.transform(anchor_image)

            pair_image_data = pair_record['data']
            pair_image_bytes = io.BytesIO(pair_image_data)
            pair_image = Image.open(pair_image_bytes).convert('RGB')
            pair_image = self.transform(pair_image)


        
        return anchor_image, pair_image, label

def getTrainingAndTestDataloaders(artist_name):
    #TODO: Adjust transforms, maybe add crops, paddings etc.

    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize the images to fit resnet34
        transforms.RandomRotation(degrees=15),  # Random rotation up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random brightness, contrast, saturation, and hue adjustments
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),         # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics - adapt this potentially or remove
    ])

    # MongoDB connection details
    mongo_uri = "mongodb://mongo:27017/"
    db_name = "arti"
    collection_name = "paintings"
    #artist_name = "Sandro Botticelli"  # Example artist name
    num_anchor_images_training = 40
    num_pairs_per_image_training = 10
    num_anchor_images_testing = 10
    num_pairs_per_image_testing = 5
    # Create dataset instance


    train_dataset = MongoDBSiameseImageDataset(mongo_uri, db_name, collection_name, artist_name=artist_name, num_anchor_images=num_anchor_images_training, num_pairs_per_image=num_pairs_per_image_training, transform=transform, purpose=Purpose.TRAINING)
    test_dataset = MongoDBSiameseImageDataset(mongo_uri, db_name, collection_name, artist_name=artist_name, num_anchor_images=num_anchor_images_testing, num_pairs_per_image=num_pairs_per_image_testing, transform=transform, purpose=Purpose.TESTING)

    # Define batch size
    batch_size = 32

    # Create DataLoader instances
    if torch.cuda.is_available():
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,  pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    '''
    # Iterate over DataLoader to print batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        for image, label in zip(images, labels):
            print(f"Image: {image.shape}, Label: {label}")
    '''
    return train_loader, test_loader