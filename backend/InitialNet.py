import base64
import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import pymongo
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
import InitialDataloaderGetter

def __get_latest_path_file():
    models_directory = os.path.join(os.path.dirname(__file__), 'models')

    # search pattern - here just cnnmodel
    pattern = os.path.join(models_directory, '*cnnmodel*.pth')

    matching_files = glob.glob(pattern)

    # Get the latest file
    matching_files.sort(key=os.path.getmtime)
    latest_file = matching_files[-1] if matching_files else None
    if latest_file:
        print(f"Latest .pth file found: {latest_file}")
        return os.path.relpath(latest_file, os.getcwd())
    else:
        return None
    


def __init_base(learning_rate, momentum):
    
    # cpu or cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet152(weights="DEFAULT").to(device)

    # Get the names of layers in the model - just for knowing what to finetung
    
    #for name, child in model.named_children():
    #    print(name)
    

    # Freeze all layers except the last couple
    for param in model.parameters():
        param.requires_grad = False

    # fine tune only last couple of layers
    layers_to_finetune = [model.layer3, model.layer4, model.fc]
    for layer in layers_to_finetune:
        for param in layer.parameters():
            param.requires_grad = True

    # set classification head to match my classes
    num_classes = InitialDataloaderGetter.getNumberOfClassesInDB()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    # test Adam and potentially change
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    return device, model, criterion, optimizer

def __training_process(model, device, num_epochs, train_loader, val_loader, optimizer, criterion):
    print("Start training CNN")
    model = model.to(device)
    best_val_loss = np.inf
    early_stopping_patience = 3
    num_no_improvements = 0    
    best_model_state = None


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, train-Loss: {train_loss / len(train_loader)}")


        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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
    filename = f'models/cnnmodel_{timestamp}.pth'
    torch.save(model.state_dict(), filename)

    print("Model saved as " + filename)


def __testing_process(model, device, test_loader, criterion):
    print("Start testing CNN")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    artist_names = InitialDataloaderGetter.get_artists_from_collection()
    with torch.no_grad():  # Disable gradient tracking for evaluation
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(predicted)):
                print(f"Predicted: {InitialDataloaderGetter.get_original_label(predicted[i], artist_names)}, Actual: {InitialDataloaderGetter.get_original_label(labels[i], artist_names)}")

    test_accuracy = 100 * correct / total
    test_loss = running_loss / len(test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

def train_and_test(learning_rate=0.001, momentum=0.9, num_epochs=10):

    print("init data")

    device, model, criterion, optimizer = __init_base(learning_rate, momentum)
    train_loader, val_loader, test_loader = InitialDataloaderGetter.getDataloaders()

    __training_process(model, device, num_epochs, train_loader, val_loader, optimizer, criterion)

    __testing_process(model, device, test_loader, criterion)

    return "Finished training and testing"

# NO GUARANTEE THE TEST SET DOES NOT HAVE OVERLAP WITH THE TRAINING DATA, NEEDS TO BE ADJUSTED
# OR SIMPLY USE TRAIN_AND_TEST AND DO NOT USE THIS
def test():
    print("init data")

    device, model, criterion, _ = __init_base(0.001, 0.9) # values do not matter since optimizer is not necessary for testing
    _, _, test_loader = InitialDataloaderGetter.getDataloaders()

    saved_model_path = __get_latest_path_file()
    if saved_model_path is None:
        return "Failed to get model"
    
    model.load_state_dict(torch.load(saved_model_path, map_location=device))

    __testing_process(model, device, test_loader, criterion)

    return "Finished testing"

def infer(image):
    device, model, _, _ = __init_base(learning_rate=0.001, momentum=0.9)    #lr doesnt matter since optimizer isnt used here

    saved_model_path = __get_latest_path_file()
    if saved_model_path is None:
        return "Failed to get model"
    
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.to(device)
    model.eval()

    artist_names = InitialDataloaderGetter.get_artists_from_collection()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize the images to fit resnet34
        transforms.ToTensor(),         # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics - adapt this potentially or remove
    ])

    transformed_image = transform(image)
    transformed_image = transformed_image.to(device)  # Move input tensor to the same device as the model

    output = model(transformed_image.unsqueeze(0))

    # Convert output probabilities to predicted class
    _, predicted = torch.max(output, 1)

    # Get the predicted label and actual label
    predicted_label = InitialDataloaderGetter.get_original_label(predicted.item(), artist_names)

    random_paintings_result = __get_random_paintings(predicted_label)

    random_paintings = []
    for painting in random_paintings_result:
        binary_data = painting['data']
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        random_paintings.append({
            'paintingName': painting['paintingName'],
            'base64data': base64_data
        })

    
    return predicted_label, random_paintings


def __get_random_paintings(artist_name):
    client = pymongo.MongoClient('mongodb://mongo:27017/')
    db = client['arti']
    collection = db['paintings']

    pipeline = [
        {"$match": {"artistName": artist_name}},
        {"$sample": {"size": 7}},
    ]
    
    results = list(collection.aggregate(pipeline))
    
    return results