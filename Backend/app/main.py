import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import tempfile

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": [
            "http://localhost:3000",
            "https://brainee-classifier-web.vercel.app/"
        ]
    }
})

# Configuration for Vercel
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'Brain_Tumor_Classifier_Enhanced.h5')

# Load pre-trained model (consider lazy loading or alternative model hosting)
model = load_model(MODEL_PATH)

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path, target_size=(512, 512)):
    """Preprocess image for model prediction"""
    img = load_img(file_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def get_class_labels():
    """Return dictionary of class labels"""
    return {
        0: 'No Tumor',
        1: 'Tumor', 
    }

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Validate file type and size
    if file and allowed_file(file.filename):
        try:
            # Use tempfile for serverless environment
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                temp_filepath = temp_file.name
            
            # Preprocess image
            processed_img = preprocess_image(temp_filepath)
            
            # Make prediction
            prediction = model.predict(processed_img)
            class_labels = get_class_labels()
            
            # Get the predicted class and confidence
            predicted_class_index = np.argmax(prediction)
            confidence = prediction[0][predicted_class_index]
            predicted_class = class_labels[predicted_class_index]
            
            # Remove temporary file
            os.unlink(temp_filepath)
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': float(confidence)
            })
        
        except Exception as e:
            # Log the error for debugging
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Error processing image'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

# Vercel serverless function entry point
def handler(event, context):
    return app(event, context)

# For local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
