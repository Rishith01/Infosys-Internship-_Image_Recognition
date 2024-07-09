import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Path to the trained model and the image
model_path = "YOLO_dataset/best.pt"
image_path = "path/to/your/test/image.jpg"

# Load the model
model = YOLO(model_path)

# Perform inference
results = model(image_path)


# Display results
def display_results(image_path, results):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', lw=2))

    plt.axis('off')
    plt.show()


# Call the display function
display_results(image_path, results)
