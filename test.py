from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Tạo thư mục lưu trữ hình ảnh nếu chưa tồn tại
UPLOAD_FOLDER = 'My_image'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load mô hình đã được huấn luyện
model = load_model('mnist_model.h5')

# Định nghĩa các nhãn cho các chữ số
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Hàm dự đoán chữ số từ ảnh
def predict_digit(img_path):
    # Đọc ảnh và chuyển thành ảnh xám
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img = image.img_to_array(img)

    # Chuẩn hóa ảnh
    img_array = img / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Dự đoán xác suất cho mỗi lớp
    predictions = model.predict(img_array)

    # Lấy nhãn của lớp có xác suất cao nhất
    predicted_label = np.argmax(predictions)

    return labels[predicted_label]

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

        prediction = predict_digit(img_path)

        return render_template('index.html', prediction=prediction, img_path=img.filename)

if __name__ == '__main__':
    app.run(debug=True)
