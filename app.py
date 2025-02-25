from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from keras import models, preprocessing
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Set up configurations
UPLOAD_FOLDER = 'static/uploads'  # Store images in a static folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained AI model
MODEL_PATH = 'model/pancreatic_cancer_detection_model.h5'
model = models.load_model(MODEL_PATH, compile=False)

# Function to check valid file format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

# File Upload & Prediction Route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part in the request')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # Save the uploaded image
        
        # Preprocess the image for the model
        img = preprocessing.image.load_img(file_path, target_size=(256, 256))  
        img_array = preprocessing.image.img_to_array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        result_message = "Cancer Detected" if prediction >= 0.5 else "No Cancer Detected"
        confidence = round(prediction * 100, 2) if result_message == "Cancer Detected" else round((1 - prediction) * 100, 2)
        
        # Pass image path to result page
        image_url = f"/{file_path}"  

        # Send results to the frontend
        return render_template('result.html', result_message=result_message, confidence=confidence, image_url=image_url)
    
    flash('Invalid file format. Please upload a .png, .jpg, or .jpeg file.')
    return redirect(url_for('index'))

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
