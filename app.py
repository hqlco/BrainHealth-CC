from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('model.keras')

# Define class mappings
class_mappings = {0: 'Glioma', 1: 'Meninigioma', 2: 'Notumor', 3: 'Pituitary'}
# inv_class_mappings = {v: k for k, v in class_mappings.items()}

def load_and_preprocess_image(image_path, image_shape=(168, 168)):
    img = image.load_img(image_path, target_size=image_shape, color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add the batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join('static', file.filename)
        file.save(filepath)
        img_array = load_and_preprocess_image(filepath)
        prediction = model.predict(img_array)
        predicted_label = class_mappings[np.argmax(prediction)]
        return render_template('result.html', prediction=predicted_label, image_path=filepath)

@app.route('/test')
def test():
    # Assuming 'file.jpg' is in the same directory as app.py
    test_image_path = 'G_28_RO_.jpg'
    if os.path.exists(test_image_path):
        img_array = load_and_preprocess_image(test_image_path)
        prediction = model.predict(img_array)
        predicted_label = class_mappings[np.argmax(prediction)]
        return render_template('result.html', prediction=predicted_label, image_path=test_image_path)
    else:
        return 'Test image not found'

if __name__ == '__main__':
    app.run(debug=True)
