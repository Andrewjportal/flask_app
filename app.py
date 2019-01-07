import base64
import io
from PIL import Image
import keras
import cv2
from PIL import Image
import numpy as np
from flask import request
from flask import Flask
from flask import jsonify
import cv2
import io
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = 'cloud-fail.model'
print('Model loaded')

def get_file_path(request):
    # Get the file from post request
    file_path = request.files['file']

    return file_path


@app.route('/', methods=['GET'])
def index():
    # Main page
    return ('app.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        IMG_SIZE=299
        file_path = get_file_path(request)
        img_array = cv2.imread(file_path)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_data=new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


        preds = model.predict(img_data)

        # decode the results into a list of tuples (class, description, probability)
        pred_class = ["CNV", "DMV", "DRUSEN"]
        result = str(pred_class[(np.argmax(preds[0]))])


        return result
    return None