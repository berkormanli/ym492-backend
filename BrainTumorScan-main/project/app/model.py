import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps

base = 'C:/Users/msi/Desktop/BrainTumorScan-main/BrainTumorScan-main/project/ML_Model'
model = load_model(f'{base}/model.h5')

def image_pre(path):
    size = (150, 150)
    image = Image.open(path)
    image = ImageOps.grayscale(image)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0
    data = image_array.reshape((1, 150, 150, 1))
    return data

def predict(data):
    prediction = model.predict(data)
    return float(prediction[0][0])

def generate_heatmap(image_path, save_path='C:/Users/msi/Desktop/BrainTumorScan-main/BrainTumorScan-main/project/app/static/output_marked.jpg'):
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150), color_mode='grayscale')
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("conv2d_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = 1 - predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :]
    heatmap = tf.reduce_sum(heatmap, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (150, 150))
    heatmap_uint8 = np.uint8(255 * heatmap)

    _, thresh = cv2.threshold(heatmap_uint8, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (150, 150))

    # Kutu boşluk margin değeri
    margin = 10 

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        shrink_ratio = 0.8
        new_w = int(w * shrink_ratio)
        new_h = int(h * shrink_ratio)
        center_x = x + w // 2
        center_y = y + h // 2
        new_x = max(center_x - new_w // 2, 0)
        new_y = max(center_y - new_h // 2, 0)

        # Yeşil kutu
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, orig.shape[1])
        y2 = min(y + h + margin, orig.shape[0])

        # Kırmızı kutu 
        cv2.rectangle(orig, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 2)
        
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 200, 0), 2)

    cv2.imwrite(save_path, orig)
