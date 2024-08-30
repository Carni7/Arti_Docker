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


class MongoDBSiameseLPImageDataset(Dataset):
    def __init__(self, mongo_uri, db_name, collection_name, anchor_records, num_neg_pairs_per_image=10, transform=None):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        # number of negative samples per painting.
        self.num_neg_pairs_per_image = num_neg_pairs_per_image
        # number of anchor images used for creating training set
        self.anchor_records = anchor_records
        self.transform = transform
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

        # Initialize lists to store training and testing pairs
        self.siamese_pairs = []

        for anchor_record in anchor_records:
            dissimilar_records = self.collection.aggregate([
                { "$match": { "_id": { "$ne": anchor_record['_id'] } } },  # Exclude documents with the current id to crate negative pairs
                { "$sample": { "size": num_neg_pairs_per_image } }  # Sample num_neg_pairs_per_image random documents
            ])
            
            for dissimilar_record in dissimilar_records:
                pair = (anchor_record, dissimilar_record, 0)                
                self.siamese_pairs.append(pair)

            similar_record = (anchor_record, anchor_record, 1)
            self.siamese_pairs.append(similar_record)

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

def getTrainingAndTestDataloaders():

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
    collection_name = "lost_paintings_training"
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    #train
    num_anchor_images_training = 40
    num_neg_pairs_per_image_training = 10
    #val
    num_anchor_images_validation = 10
    num_neg_pairs_per_image_validation = 5
    #test
    num_anchor_images_testing = 10
    num_neg_pairs_per_image_testing = 5

    
    lost_records = list(collection.find())
    random.shuffle(lost_records)

    train_records = lost_records[:num_anchor_images_training]
    val_records = lost_records[num_anchor_images_training:num_anchor_images_training + num_anchor_images_validation]
    test_records = lost_records[num_anchor_images_training + num_anchor_images_validation:num_anchor_images_training + num_anchor_images_validation + num_anchor_images_testing]

    train_dataset = MongoDBSiameseLPImageDataset(mongo_uri, db_name, collection_name, anchor_records=train_records, num_neg_pairs_per_image=num_neg_pairs_per_image_training, transform=transform)
    val_dataset = MongoDBSiameseLPImageDataset(mongo_uri, db_name, collection_name, anchor_records=val_records, num_neg_pairs_per_image=num_neg_pairs_per_image_validation, transform=transform)
    test_dataset = MongoDBSiameseLPImageDataset(mongo_uri, db_name, collection_name, anchor_records=test_records, num_neg_pairs_per_image=num_neg_pairs_per_image_testing, transform=transform)

    # Define batch size
    batch_size = 32

    # Create DataLoader instances
    if torch.cuda.is_available():
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,  pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader