import os
from ultralytics import YOLO

# Paths
dataset_dir = "Autodistill_dataset"
output_dir = "YOLO_dataset"

# Create YOLOv8 config file
data_yaml = """
path: {output_dir}
train: .
val: .

nc: 10
names: [Apple, Bicycle, Book, Bottle, Cat, Chair, Clock, Dog, Laptop, Television]
""".format(output_dir=output_dir)

config_path = os.path.join(output_dir, "data.yaml")
with open(config_path, "w") as f:
    f.write(data_yaml)

# Initialize YOLO model (assuming you're using a pre-trained model for fine-tuning)
model = YOLO('yolov8n.pt')  # You can use 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', or 'yolov8x.pt' depending on your needs

# Train the model
model.train(data=config_path, epochs=100, imgsz=640, batch=16)

# Save the model
model_path = os.path.join(output_dir, "best.pt")
model.save(model_path)
