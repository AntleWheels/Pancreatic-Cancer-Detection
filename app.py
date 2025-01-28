from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from keras import models
from keras import preprocessing
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Set up app configuration
UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed file formats
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained AI model
# MODEL_PATH = 'model/pancreatic_cancer_model.h5'
# model = models.load_model(MODEL_PATH)

# Function to check if the file format is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route: Renders the main page
@app.route('/')
def index():
    return render_template('index.html')

# File upload route: Handles predictions
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
        file.save(file_path)
        
        # Preprocess the image for the model
        img = preprocessing.image.load_img(file_path, target_size=(224, 224))  # Adjust size to match the model input
        img_array = preprocessing.image.img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        # prediction = model.predict(img_array)[0][0]
        # result = "Cancer Detected" if prediction >= 0.5 else "No Cancer Detected"
        # confidence = round(prediction * 100, 2) if result == "Cancer Detected" else round((1 - prediction) * 100, 2)
        
        # Remove the uploaded image after prediction
        os.remove(file_path)
        
        # Display the results
        # return render_template('result.html', result=result, confidence=confidence)
    
    flash('Invalid file format. Please upload a .png, .jpg, or .jpeg file.')
    return redirect(url_for('index'))

# Run the app
if __name__ == '__main__':
    # Ensure the uploads folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
