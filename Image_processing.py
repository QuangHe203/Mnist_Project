import os
import cv2
from flask import current_app
from werkzeug.utils import secure_filename

def save_uploaded_file(file):
    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath, filename

def convert_to_grayscale(filepath, filename):
    # Đọc ảnh và chuyển sang đen trắng
    image = cv2.imread(filepath)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_filename = 'gray_' + filename
    gray_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], gray_filename)
    cv2.imwrite(gray_filepath, gray_image)
    return gray_filename
