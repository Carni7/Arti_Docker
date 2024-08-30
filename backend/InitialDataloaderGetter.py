import random
from matplotlib import transforms
import pymongo
import io
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

class DynamicCrop:
    def __call__(self, img, min_height, min_width):
        width, height = img.size
        
        left_crop = random.randint(0, 50)
        top_crop = random.randint(0, 50)
        right_crop = random.randint(0, 50)
        bottom_crop = random.randint(0, 50)
        
        # Calculate new width and height after cropping
        new_width = width - left_crop - right_crop
        new_height = height - top_crop - bottom_crop

        if new_height < min_height or  new_width < min_width:
            return img

        # Apply the crop
        img = img.crop((left_crop, top_crop, width - right_crop, height - bottom_crop))
        
        return img

class MongoDBImageDataset(Dataset):
    def __init__(self, mongo_uri, db_name, collection_name, num_per_artist=80, transform=None):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.num_per_artist = num_per_artist
        self.transform = transform
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.artist_names = self.collection.distinct("artistName")


        # Dictionary to hold image records per artist
        self.artist_image_records = {}

        # Fetch image records per artist
        for artist_name in self.artist_names:
            records = list(self.collection.find({"artistName": artist_name}))
            random.shuffle(records)  # Shuffle the records
            self.artist_image_records[artist_name] = records[:num_per_artist]  # Select first num_per_artist records

        # Flatten the list of image records
        self.image_records = [record for records in self.artist_image_records.values() for record in records]

    
    def __len__(self):
        return len(self.image_records)
    
    def __getitem__(self, idx):
        image_record = self.image_records[idx]
        image_data = image_record['data']
        image_bytes = io.BytesIO(image_data)
        image = Image.open(image_bytes).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get the original label string
        original_label = image_record['artistName']

        # Get the index of the artist name in the list of artist names
        label_index = self.artist_names.index(original_label)

        # One-hot encode the label
        #one_hot_label = torch.nn.functional.one_hot(torch.tensor(label_index), num_classes=len(self.artist_names))
        label = label_index

        # Return the image and the one-hot encoded label
        return image, label


def get_artists_from_collection():
    mongo_uri = "mongodb://mongo:27017/"
    db_name = "arti"
    collection_name = "paintings"

    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    artist_names = collection.distinct("artistName")

    return artist_names

def get_original_label(label_index, artist_names):

    return artist_names[label_index]

def getDataloaders():
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

    # Create dataset instance
    dataset = MongoDBImageDataset(mongo_uri, db_name, collection_name, transform=transform)

    # Define split ratios
    train_ratio = 0.75
    val_ratio = 0.125
    #test_ratio = 0.125

    # Calculate sizes of each set
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split dataset into training, validation and testing sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Define batch size
    batch_size = 32

    # Create DataLoader instances
    if torch.cuda.is_available():
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,  pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,  pin_memory=True)

    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    '''
    # Iterate over DataLoader to print batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        for image, label in zip(images, labels):
            print(f"Image: {image.shape}, Label: {label}")
    '''
    return train_loader, val_loader, test_loader

def getNumberOfClassesInDB():
    # MongoDB connection details
    mongo_uri = "mongodb://mongo:27017/"
    db_name = "arti"
    collection_name = "paintings"

    # Connect to the MongoDB server
    client = pymongo.MongoClient(mongo_uri)

    # Select the database and collection
    db = client[db_name]
    collection = db[collection_name]

    # Perform aggregation to get the distinct count of artist names
    pipeline = [
        {"$group": {"_id": "$artistName"}},
        {"$group": {"_id": None, "count": {"$sum": 1}}}
    ]

    result = list(collection.aggregate(pipeline))

    return result[0]["count"] if result[0]["count"] else 0