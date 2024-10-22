import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Labels for prediction
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Ensure the uploads folder exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to process the image and make predictions
def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

# Route for the home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for the prediction
@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400

    f = request.files['file']
    if f.filename == '':
        return "No selected file", 400

    # Save the file to the uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(file_path)

    # Get the prediction
    try:
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    except Exception as e:
        return f"Error in prediction: {str(e)}", 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
