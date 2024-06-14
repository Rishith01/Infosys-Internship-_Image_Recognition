import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

def decode_predictions(predictions, confidence_threshold=0.5):
    # Decode predictions to get bounding boxes and class labels
    boxes, classes, scores = [], [], []
    for pred in predictions:
        score = pred[4:]  # Confidence scores for all classes
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > confidence_threshold:
            box = pred[:4]  # Bounding box coordinates
            boxes.append(box)
            classes.append(class_id)
            scores.append(confidence)
    return boxes, classes, scores
