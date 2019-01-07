import base64
import io

import keras
from keras.models import load_model

import cv2
from PIL import Image
import numpy as np

from flask import request
from flask import Flask
from flask import jsonify
from flask_cors import CORS

# initialize our Flask application and the Keras model

app = Flask(__name__)
CORS(app)
model = load_model('cloud-fail.model')
CATEGORIES = ["CNV", "DMV", "DRUSEN"]


def prepare(image):
    IMG_SIZE = 150
    img_array = cv2.imread(image)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    image1 = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    return image1

@app.route('/deep', methods=["POST"])
def predict():
    message = request.get_json(force= True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    prediction = model.predict([prepare(image)])
    response = (CATEGORIES[(np.argmax(prediction[0]))])
    return jsonify(response)
