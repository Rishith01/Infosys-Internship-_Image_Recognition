from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')

model.train(data='custom_data.yaml', epochs=100)
