from flask import Flask, request, jsonify, render_template
import os
import io
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__, template_folder='/home/maddy/Desktop/Bioinformatics/templates')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

model = load_model('models/model_vgg16.h5')  # Replace with the actual path

# Set a confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.5

# Define a route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    if uploaded_file:
        # Read the image data from the file storage object
        image_data = uploaded_file.read()
        
        # Use io.BytesIO to create a bytes-like object
        image_stream = io.BytesIO(image_data)
        
        # Load the image using load_img
        img = load_img(image_stream, target_size=(224, 224))
        
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        img_data = preprocess_input(x)
        predictions = model.predict(img_data)
        
        # Extract the predicted class as an integer
        predicted_class = int(np.argmax(predictions))
        
        # Get the confidence score for the prediction
        confidence = float(predictions[0][predicted_class])

        # Map predicted class to labels
        if confidence < CONFIDENCE_THRESHOLD:
            label = "Can't predict"
        elif predicted_class == 0:
            label = "Negative"
        elif predicted_class == 1:
            label = "Positive"
        else:
            label = "Not Clear"

        return jsonify({'predicted_class': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
