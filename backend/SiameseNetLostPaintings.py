import base64
import random
import matplotlib.pyplot as plt # for plotting

import numpy as np
import torch
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torch.utils.data import random_split
import torchvision.models as models
import os
import glob
import torchvision.transforms as transforms

import datetime
import SiameseLostPaintingsDataloaderGetter as SLPDataloaderGetter
import pymongo
from PIL import Image
import io


# https://github.com/ultimateabhi719/face-recognition/tree/main/src

#######################################################
# custom loss (contrastiveLoss) for the siamese net
#######################################################

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, y):
        diff = input1 - input2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / input1.size()[0]
        return loss


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained_resnet, fc_dim, out_dim):
        super(SiameseNetwork, self).__init__()
        self.pretrained_resnet = pretrained_resnet
        # define them 1 by 1 so if the resnet changes for example to 101 the layers can be adjusted if they are different
        self.siamese_cnn = nn.Sequential(
            pretrained_resnet.conv1,
            pretrained_resnet.bn1,
            pretrained_resnet.relu,
            pretrained_resnet.maxpool,
            pretrained_resnet.layer1,
            pretrained_resnet.layer2,
            pretrained_resnet.layer3,
            pretrained_resnet.layer4
        )

        #here the 25088 stands for 3x224x224, ie, the 3 channels rbg and 224x224 image input size
        # comes from transforms.Resize((224, 224)) from SiameseDataloaderGetter.py
        self.fcl = nn.Sequential(
            nn.Linear(51200, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, out_dim)
        )
  
    def forward_once(self, x):
        #pass through cnn, fix shape and input into the fcl
        output = self.siamese_cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fcl(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def extract_code_layer(self, input):
        features = self.siamese_cnn(input)
        features = features.view(features.size(0), -1)
        return features

def __get_latest_path_file():

    models_directory = os.path.join(os.path.dirname(__file__), 'models')

    # search pattern - here siamesemodel + the converted artist name
    pattern = os.path.join(models_directory, f'*siamesemodellostpaintings*.pth')

    matching_files = glob.glob(pattern)

    # Get the latest file
    matching_files.sort(key=os.path.getmtime)
    latest_file = matching_files[-1] if matching_files else None
    if latest_file:
        print(f"Latest .pth file found: {latest_file}")
        return os.path.relpath(latest_file, os.getcwd())
    else:
        return None
    

def __init_base(learning_rate):
        
    # cpu or cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #create resnet to pass to siamese net construction
    # Load a pre-trained ResNet-34 model
    basemodel = models.resnet34(weights="DEFAULT").to(device)

    # Freeze all layers except the last couple
    for param in basemodel.parameters():
        param.requires_grad = False

    # fine tune only last couple of layers
    layers_to_finetune = [basemodel.layer3, basemodel.layer4, basemodel.fc]
    for layer in layers_to_finetune:
        for param in layer.parameters():
            param.requires_grad = True

    model = SiameseNetwork(pretrained_resnet=basemodel, fc_dim=256, out_dim=2)
    criterion = ContrastiveLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return device, model, criterion, optimizer

def __training_process(model, device, num_epochs, train_loader, val_loader, optimizer, criterion):
    print("Start training siamese net")
    model = model.to(device)
    best_val_loss = np.inf
    early_stopping_patience = 3
    num_no_improvements = 0    
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for anchor_image, pair_image, label in train_loader:
            anchor_image = anchor_image.to(device)
            pair_image = pair_image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output_anchor, output_pair = model(anchor_image, pair_image)
            loss = criterion(output_anchor, output_pair, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        # validation for early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor_image, pair_image, label in val_loader:
                anchor_image = anchor_image.to(device)
                pair_image = pair_image.to(device)
                label = label.to(device)

                output_anchor, output_pair = model(anchor_image, pair_image)
                loss = criterion(output_anchor, output_pair, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, val-Loss: {avg_val_loss} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            num_no_improvements = 0
        else:
            num_no_improvements += 1
            if num_no_improvements >= early_stopping_patience:
                print(f"No improvement for {early_stopping_patience} epochs. Early stopping...")
                model.load_state_dict(best_model_state)
                break
            else:
                print(f"No improvement for {num_no_improvements} epochs")


    print("Training complete")

    # Save the entire model to a file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'models/siamesemodellostpaintings_{timestamp}.pth'
    torch.save(model.state_dict(), filename)

    print("Model saved as " + filename)


def __testing_process(model, device, test_loader, criterion):
    print("Start testing siamese net")
    model = model.to(device)
    model.eval()
    with torch.no_grad():  # Disable gradient tracking for evaluation
        running_loss = 0.0
        for anchor_image, pair_image, label in test_loader:
            anchor_image = anchor_image.to(device)
            pair_image = pair_image.to(device)
            label = label.to(device)

            output_anchor, output_pair = model(anchor_image, pair_image)
            loss = criterion(output_anchor, output_pair, label)
            running_loss += loss.item()

            distance = torch.norm(output_anchor - output_pair, dim=1)

            print('Distance for label' + str(label) + ': ' + str(distance))

    test_loss = running_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")


def train(learning_rate=0.001, num_epochs=3):

    print("init data")
    device, model, criterion, optimizer = __init_base(learning_rate)
    train_loader, val_loader, _ = SLPDataloaderGetter.getTrainingAndTestDataloaders()

    __training_process(model, device, num_epochs, train_loader, val_loader, optimizer, criterion)
    return "Finished training"

def test():

    print("init data and load model")
    device, model, criterion, _ = __init_base(learning_rate=0.001)    #lr doesnt matter since optimizer isnt used here
    _, _, test_loader = SLPDataloaderGetter.getTrainingAndTestDataloaders()

    saved_model_path = __get_latest_path_file()
    if saved_model_path is None:
        return "Failed to get model"
    
    model.load_state_dict(torch.load(saved_model_path, map_location=device))

    __testing_process(model, device, test_loader, criterion)
    return "Finished training"

def train_and_test(learning_rate=0.001, num_epochs=3):

    print("init data")
    device, model, criterion, optimizer = __init_base(learning_rate)
    train_loader, val_loader, test_loader = SLPDataloaderGetter.getTrainingAndTestDataloaders()
    
    __training_process(model, device, num_epochs, train_loader, val_loader, optimizer, criterion)

    __testing_process(model, device, test_loader, criterion)
    
    return "Finished training and testing"


def infer(image):
    # infer a single image and return results

    device, model, _, _ = __init_base(learning_rate=0.001)    #lr doesnt matter since optimizer isnt used here

    saved_model_path = __get_latest_path_file()
    if saved_model_path is None:
        return "Failed to get model"
    
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.to(device)
    model.eval()

    # get code layer of given image for distance calculation
    transform = transforms.Compose([
        transforms.Resize((299, 299)),          # Resize the images to fit resnet34
        transforms.ToTensor(),                  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
    ])

    transformed_image = transform(image)

    transformed_image = transformed_image.to(device)  # Move input tensor to the same device as the model

    with torch.no_grad():  # Disable gradient tracking for inference
        calculated_vector_representation = model.extract_code_layer(transformed_image.unsqueeze(0))

    calculated_vector_representation = calculated_vector_representation.to(device)

    client = pymongo.MongoClient("mongodb://mongo:27017/")
    db = client["arti"]
    collection = db["lost_paintings"]

    closest_image_record = None
    min_distance = float('inf')
    distance_threshold = 500
    close_records = []
    total_distance = 0
    num_records = 0

    for image_record in collection.find():
        db_vector_representation = image_record["vectorRepresentation"]
        db_vector_representation = torch.tensor(db_vector_representation)
        db_vector_representation = db_vector_representation.to(device)
        distance = torch.norm(calculated_vector_representation - db_vector_representation)
        # for most similar image
        if distance < min_distance:
            min_distance = distance
            closest_image_record = image_record

        #for images in a certain range
        if distance < distance_threshold:
            close_records.append({
                'image_record': image_record,
                'distance': distance
        })
        total_distance = total_distance + distance
        num_records = num_records + 1


    if closest_image_record is None:
        return 'Found no image record'

    #pre-process image data of closest image to infer similarity
    closest_image_data = closest_image_record["data"] 

    closest_image_bytes = io.BytesIO(closest_image_data)
    closest_image = Image.open(closest_image_bytes).convert("RGB")  # Convert bytes to PIL image

    transformed_closest_image = transform(closest_image)
    transformed_closest_image = transformed_closest_image.to(device)

    with torch.no_grad():
        output1, output2 = model(transformed_image.unsqueeze(0), transformed_closest_image.unsqueeze(0))
 
    similarity = torch.norm(output1 - output2)

    #add at most 10 images that are close in order excluding the closest one
    close_paintings = []
    sorted_records = sorted(close_records, key=lambda x: x['distance'])
    closest_records = sorted_records[:10]

    for painting in closest_records:
        if painting['image_record']['_id'] != closest_image_record["_id"]:
            binary_data = painting['image_record']['data']
            base64_data = base64.b64encode(binary_data).decode('utf-8')
            close_paintings.append({
                'paintingName': painting['image_record']['paintingName'],
                'artistName': painting['image_record']['artistName'],
                'base64data': base64_data,
                'distance': float(painting['distance'])
            })


    return similarity, closest_image_record, close_paintings, distance_threshold

    

def create_vectors():

    print("init data and load model")
    device, model, _, _ = __init_base(learning_rate=0.001)    #lr doesnt matter since optimizer isnt used here

    saved_model_path = __get_latest_path_file()

    if saved_model_path is None:
        return "Failed to get model"
    
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.to(device)
    model.eval()

    #transforms for image preprocessing same as when training (size and normalize)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),          # Resize the images to fit resnet34
        transforms.ToTensor(),                  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
    ])

    client = pymongo.MongoClient("mongodb://mongo:27017/")
    db = client["arti"]
    collection = db["lost_paintings"]
    
    update_requests = []
    # Iterate over documents/images in the collection
    for document in collection.find():

        image_data = document["data"] 

        image_bytes = io.BytesIO(image_data)
        image = Image.open(image_bytes).convert("RGB")  # Convert bytes to PIL image

        # Apply the transformation pipeline to preprocess the image
        transformed_image = transform(image)

        transformed_image = transformed_image.to(device)  # Move input tensor to the same device as the model

        with torch.no_grad():  # Disable gradient tracking for inference
            vector_representation = model.extract_code_layer(transformed_image.unsqueeze(0))

        update_requests.append(
            pymongo.UpdateOne({"_id": document["_id"]}, {"$set": {"vectorRepresentation": vector_representation.tolist()}})
        )

    result = collection.bulk_write(update_requests)
    print("Number of documents updated:", result.modified_count)

