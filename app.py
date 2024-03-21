from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('tomato_leaf_model.pkl')  # Replace with your actual model file path

# Define class labels
class_labels = {
    0: 'Bacterial_spot',
    1: 'Early_blight',
    2: 'Late_blight',
    3: 'Leaf_Mold',
    4: 'Septoria_leaf_spot',
    5: 'Spider_mites Two-spotted_spider_mite',
    6: 'Target_Spot',
    7: 'Tomato_Yellow_Leaf_Curl_Virus',
    8: 'Tomato_mosaic_virus',
    9: 'Healthy'
}

def preprocess_image(image_data):
    # Convert base64 encoded image data to numpy array
    image = Image.open(BytesIO(image_data))
    # Perform any required preprocessing (e.g., resizing, normalization)
    image = image.resize((224, 224))  # Example resizing to match model input shape
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming image data is sent as base64 encoded string in JSON
        data = request.json
        image_data = data['image']

        # Preprocess image
        image_np = preprocess_image(image_data)

        # Make prediction
        prediction = model.predict(np.expand_dims(image_np, axis=0))[0]

        # Convert prediction to a human-readable label
        predicted_class = np.argmax(prediction)
        label = class_labels[predicted_class]

        # Return prediction
        return jsonify({'prediction': label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
