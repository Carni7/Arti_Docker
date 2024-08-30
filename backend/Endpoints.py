from bson import ObjectId
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from torchvision.models import resnet18
#from SiameseNet import train, test, infer, createVectorsForArtist
import SiameseNet
import InitialNet
import SiameseNetLostPaintings as SiameseNetLP
from torchvision.transforms import transforms
from PIL import Image
import io
import numpy as np

import pymongo
import base64
import io

# Create a Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

#######################################################
# initial CNN
#######################################################

@app.route('/trainAndTestCNN', methods=['POST'])
def train_and_test_CNN():

    data = request.json

    num_epochs = data.get('num_epochs', 10)
    learning_rate = data.get('learning_rate', 0.001) 
    momentum = data.get('momentum', 0.9) 

    InitialNet.train_and_test(float(learning_rate), float(momentum), num_epochs)

    return jsonify({'message': 'Finished training and testing CNN successfully'}), 200

@app.route('/inferImageCNN', methods=['POST'])
def infer_image_CNN():
    image_file = request.files['image']
   

    if image_file.filename == '':
        return jsonify({'error': 'Image given as File required'}), 400
    
    image = Image.open(image_file)
    image = image.convert("RGB")

    predicted_label, random_paintings = InitialNet.infer(image)

    response = {
        'artist_name': predicted_label,
        'random_paintings': random_paintings
    }
    return jsonify(response), 200



#######################################################
# siamese network
#######################################################

@app.route('/trainSiamese', methods=['POST'])
def train_siamese():

    data = request.json

    num_epochs = data.get('num_epochs', 10)
    learning_rate = data.get('learning_rate', 0.001) 
    artist_name = data.get('artist_name') 

    if artist_name is None or not isinstance(artist_name, str):
        return jsonify({'error': 'Artist name is not of type String'}), 400


    SiameseNet.train(artist_name, learning_rate, num_epochs)

    return jsonify({'message': 'Finished Training successfully'}), 200

@app.route('/testSiamese', methods=['POST'])
def test_siamese():

    data = request.json

    artist_name = data.get('artist_name') 

    if artist_name is None or not isinstance(artist_name, str):
        return jsonify({'error': 'Artist name is not of type String'}), 400


    SiameseNet.test(artist_name)

    return jsonify({'message': 'Finished Testing successfully'}), 200

@app.route('/createVectorsForArtist', methods=['POST'])
def create_vectors_for_artist():

    data = request.json

    artist_name = data.get('artist_name') 

    if artist_name is None or not isinstance(artist_name, str):
        return jsonify({'error': 'Artist name is not of type String'}), 400


    SiameseNet.create_vectors_for_artist(artist_name)

    return jsonify({'message': 'Finished vector creation successfully'}), 200


@app.route('/inferImageSiamese', methods=['POST'])
def infer_image_siamese():

    image_file = request.files['image']
   
    artist_name = artist_name = request.form.get('artist_name') 

    if image_file.filename == '' or artist_name is None or not isinstance(artist_name, str):
        return jsonify({'error': 'Artist name needs to be given as String and Image as File'}), 400
    
    image = Image.open(image_file)
    image = image.convert("RGB")

    similarity, closest_image_record, close_paintings, closeness_threshold, average_distance = SiameseNet.infer(image, artist_name)

    binary_data = closest_image_record['data']
    closest_image_data = base64.b64encode(binary_data).decode('utf-8')

    response = {
        "similarity": float(similarity), 
        "closest_image_id": str(closest_image_record['_id']), 
        "painting_name": str(closest_image_record['paintingName']), 
        "file_name": str(closest_image_record['filename']), 
        "closest_image_data": str(closest_image_data),
        "close_paintings": close_paintings,
        "closeness_threshold": closeness_threshold,
        "average_distance": average_distance
    }

    return jsonify(response), 200


@app.route('/testEuclidean', methods=['GET'])
def testEuclidean():

    fixed_vector = np.random.rand(25088)

    random_vectors = [np.random.rand(25088) for _ in range(200)]

    # Calculate Euclidean distance between the fixed vector and each random vector
    distances = [np.linalg.norm(fixed_vector - random_vector) for random_vector in random_vectors]

    # Print the distances for the first 10 random vectors
    print("Euclidean distances between the fixed vector and the first 10 random vectors:")
    for idx, distance in enumerate(distances[:10], start=1):
        print(f"Random vector {idx}: {distance}")

    return jsonify({'message': 'Finished euclidean test'}), 200





#######################################################
# Angular endpoints
#######################################################

client = pymongo.MongoClient('mongodb://mongo:27017/')
db = client['arti']
collection = db['paintings']

@app.route('/image/<id>', methods=['GET'])
def get_image(id):
    try:
        # Retrieve the document by its ID
        document = collection.find_one({"_id": ObjectId(id)})
        
        if not document:
            abort(404)
        
        # Get the binary data from the document
        binary_data = document['data']
        
        # Decode the base64 string to binary data
        image_data = base64.b64decode(binary_data)
        
        # Use an in-memory bytes buffer to serve the image
        image_stream = io.BytesIO(image_data)
        
        # Send the image as a response
        return send_file(image_stream, mimetype='image/jpeg', as_attachment=False, attachment_filename=document['filename'])
    
    except Exception as e:
        print(e)
        abort(404)





#######################################################
# siamese network for lost paintings
#######################################################

@app.route('/trainSiameseLostPaintings', methods=['POST'])
def train_siamese_lost_paintings():

    data = request.json

    num_epochs = data.get('num_epochs', 10)
    learning_rate = data.get('learning_rate', 0.001) 

    SiameseNetLP.train(learning_rate, num_epochs)

    return jsonify({'message': 'Finished Training successfully'}), 200

@app.route('/testSiameseLostPaintings', methods=['GET'])
def test_siamese_lost_paintings():

    SiameseNetLP.test()

    return jsonify({'message': 'Finished Testing successfully'}), 200

@app.route('/createVectorsLostPaintings', methods=['GET'])
def create_vectors_lost_paintings():

    SiameseNetLP.create_vectors()

    return jsonify({'message': 'Finished vector creation successfully'}), 200


@app.route('/inferImageSiameseLostPaintings', methods=['POST'])
def infer_image_siamese_lost_paintings():

    image_file = request.files['image']
   
    if image_file.filename == '':
        return jsonify({'error': 'Image must be given as File'}), 400
    
    image = Image.open(image_file)
    image = image.convert("RGB")

    similarity, closest_image_record, close_paintings, closeness_threshold = SiameseNetLP.infer(image)

    binary_data = closest_image_record['data']
    closest_image_data = base64.b64encode(binary_data).decode('utf-8')

    response = {
        "similarity": float(similarity),
        "painting_name": str(closest_image_record['paintingName']), 
        "artist_name": str(closest_image_record['artistName']), 
        "closest_image_id": str(closest_image_record['_id']), 
        "file_name": str(closest_image_record['filename']), 
        "closest_image_data": str(closest_image_data),
        "close_paintings": close_paintings,
        "closeness_threshold": closeness_threshold,
    }

    return jsonify(response), 200




# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)