from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
# Import your image processing functions
from image_representation import process_image, apply_grayscale, apply_blur, apply_edge_detection

app = Flask(__name__)

# Configurations for file uploads
UPLOAD_FOLDER = 'static/uploads'
DEFAULT_IMAGE_PATH = 'static/default/default_image.jpeg'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process_image_path', methods=['POST'])
def process_image_path():
    image_path = request.form['imagePath']
    if not os.path.isfile(image_path):
        # Use default image path if the provided path is invalid
        image_path = DEFAULT_IMAGE_PATH
    
    filename = os.path.basename(image_path)
    # Process the image here if needed; else, just render
    return render_template('index.html', filename=filename)

@app.route('/grayscale/<filename>')
def grayscale_image(filename):
    processed_image = apply_grayscale(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return send_from_directory(app.config['UPLOAD_FOLDER'], processed_image)

@app.route('/blur/<filename>/<int:blur_value>')
def blur_image(filename, blur_value):
    processed_image = apply_blur(os.path.join(app.config['UPLOAD_FOLDER'], filename), blur_value)
    return send_from_directory(app.config['UPLOAD_FOLDER'], processed_image)

@app.route('/edge_detection/<filename>')
def edge_detection(filename):
    processed_image = apply_edge_detection(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return send_from_directory(app.config['UPLOAD_FOLDER'], processed_image)

if __name__ == "__main__":
    app.run(debug=True)
