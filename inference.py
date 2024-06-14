import numpy as np
import cv2
import tensorflow as tf
from model import ssd_vgg16
from utils import preprocess_image, decode_predictions

# Load the model
model = tf.keras.models.load_model('ssd_model.h5', custom_objects={'ssd_loss': ssd_loss})

# Load and preprocess image
image_path = 'path_to_image.jpg'  # Replace with actual image path
image = preprocess_image(image_path)

# Predict with the model
predictions = model.predict(image)

# Decode predictions
boxes, classes, scores = decode_predictions(predictions[0])

# Display results
for box, class_id, score in zip(boxes, classes, scores):
    print(f'Class: {class_id}, Score: {score}, Box: {box}')
