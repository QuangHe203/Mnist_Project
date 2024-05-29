from flask import Flask, render_template, request, send_from_directory
from Image_processing import extract_digits
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/submit', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['image']
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        prediction = extract_digits(img_path)

        return render_template('index.html', prediction=prediction, img_path=img.filename)

if __name__ == '__main__':
    app.run(debug=True)
