import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64

# Initialize Flask app
app = Flask(__name__)

# Mapping of class indices to disease names
mapping = {
    0: 'Mulberry Healthy',
    1: 'Mulberry Healthy',
    2: 'Jute Healthy',
    3: 'Arjun Healthy',
    4: 'Bael Diseased',
    5: 'Basil Healthy',
    6: 'Chinar Diseased',
    7: 'Chinar Healthy',
    8: 'Guava Diseased',
    9: 'Guava Healthy',
    10: 'Jamun Diseased',
    11: 'Jamun Healthy',
    12: 'Jatropha Diseased',
    13: 'Jatropha Healthy',
    14: 'Jute Diseased',
    15: 'Jute Healthy',
    16: 'Lemon Diseased',
    17: 'Lemon Healthy',
    18: 'Mango Diseased',
    19: 'Mango Healthy',
    20: 'Mulberry Diseased',
    21: 'Mulberry Healthy',
    22: 'Pomegranate Diseased',
    23: 'Pomegranate Healthy',
    24: 'Pongamia Pinnata Diseased',
    25: 'Pongamia Pinnata Healthy'
}


# Load the pre-trained model
def load_keras_model():
    # Focus on loading the realmnv2.h5 model as requested
    print("Loading realmnv2.h5 model...")
    
    # Define custom objects to handle potential issues
    class CustomInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs.pop('batch_shape')
            super().__init__(**kwargs)
    
    # Comprehensive custom objects dictionary
    custom_objects = {
        'InputLayer': CustomInputLayer,
        'DTypePolicy': tf.keras.mixed_precision.Policy
    }
    
    # Try loading with custom objects
    try:
        model = load_model('MHAMNV2.h5', custom_objects=custom_objects)
        print("Successfully loaded realmnv2.h5 model!")
        return model
    except Exception as e:
        print(f"Failed to load realmnv2.h5 with custom objects: {e}")
        
        # Try without custom objects as fallback
        try:
            print("Attempting to load realmnv2.h5 without custom objects...")
            model = load_model('realmnv2.h5')
            print("Successfully loaded realmnv2.h5 model without custom objects!")
            return model
        except Exception as e2:
            print(f"Failed to load realmnv2.h5 without custom objects: {e2}")
            
            print("All attempts to load realmnv2.h5 model failed.")
            return None

# Global model variable
MODEL = load_keras_model()

# Check if model loaded successfully
if MODEL is None:
    print("ERROR: Failed to load the model. The application will not function correctly.")
    # We're not using a dummy model anymore as requested

# Preprocess the image for prediction
def preprocess_image(img_path):
    # Read the image file
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert image to numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array = img_array / 255.0
    
    return img_array

# Route for the main page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if model is loaded
    if MODEL is None:
        return jsonify({'error': 'Failed to load realmnv2.h5 model. Please check server logs.'}), 500
    
    # Save the uploaded file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(file_path)
        
        # Make prediction
        prediction = MODEL.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class] * 100
        
        # Remove the temporary file
        os.remove(file_path)
        
        # Return prediction results
        return jsonify({
            'prediction': mapping[predicted_class],
            'confidence': f'{confidence:.2f}%'
        })
    
    except Exception as e:
        # Remove the temporary file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({'error': str(e)}), 500

# Route for camera capture classification
@app.route('/classify_camera', methods=['POST'])
def classify_camera():
    # Get base64 encoded image
    image_data = request.json.get('image')
    
    if not image_data:
        return jsonify({'error': 'No image data'}), 400
    
    # Check if model is loaded
    if MODEL is None:
        return jsonify({'error': 'Failed to load realmnv2.h5 model. Please check server logs.'}), 500
    
    # Remove the data URL prefix
    image_data = image_data.split(',')[1]
    
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Save the image
    file_path = os.path.join('uploads', 'camera_capture.png')
    
    try:
        # Decode and save the image
        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
        
        # Preprocess the image
        processed_image = preprocess_image(file_path)
        
        # Make prediction
        prediction = MODEL.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class] * 100
        
        # Remove the temporary file
        os.remove(file_path)
        
        # Return prediction results
        return jsonify({
            'prediction': mapping[predicted_class],
            'confidence': f'{confidence:.2f}%'
        })
    
    except Exception as e:
        # Remove the temporary file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create uploads directory
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True)