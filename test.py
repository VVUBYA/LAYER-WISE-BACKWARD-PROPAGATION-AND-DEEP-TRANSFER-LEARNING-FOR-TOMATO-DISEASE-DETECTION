import requests
import base64

# Replace with the actual URL where your Flask app is running
url = 'http://127.0.0.1:5000/predict'

# Load the image
with open('path/to/your/image.jpg', 'rb') as file:
    image_data = base64.b64encode(file.read()).decode('utf-8')

# Sample JSON data with the base64 encoded image
data = {
    'image': image_data
}

# Send POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
