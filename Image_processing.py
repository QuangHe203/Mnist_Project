import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('mnist_model.h5')

def extract_digits(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, im_th = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    predictions = []

    for x, y, w, h in rects:
        y = max(y - 3, 0)
        x = max(x - 3, 0)
        w += 3
        h += 3

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        sliced = im_th[y:y + h, x:x + w]
        sliced = img_to_array(sliced, dtype='float32')
        sliced = cv2.resize(sliced, (28, 28)).reshape(28, 28, 1) / 255.0
        sliced = np.expand_dims(sliced, axis=0)

        prediction = np.argmax(model.predict(sliced), axis=-1)
        predictions.append(prediction[0])

    return ''.join(map(str, predictions))