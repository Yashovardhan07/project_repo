from flask import Flask, request, jsonify
import numpy as np
import os
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Define the path to the TFLite model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'potato_classifier.tflite')

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ['Early Blight', 'Healthy', 'Late Blight']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load and preprocess the image
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))  # Make sure this matches your model input
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Set the tensor to the input image
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get prediction results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        class_idx = np.argmax(output_data[0])
        class_name = class_labels[class_idx]
        confidence = float(output_data[0][class_idx])

        return jsonify({'prediction': class_name, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)