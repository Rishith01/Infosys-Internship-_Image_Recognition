import os
from PIL import Image

# Define class mapping
class_mapping = {
    "Apple": 0,
    "Bicycle": 1,
    "Book": 2,
    "Bottle": 3,
    "Cat": 4,
    "Chair": 5,
    "Clock": 6,
    "Dog": 7,
    "Laptop": 8,
    "Television": 9
}

def convert_to_yolo_format(label, width, height):
    class_name, x_min, y_min, x_max, y_max = label.split()
    class_id = class_mapping[class_name]
    x_center = (float(x_min) + float(x_max)) / 2.0 / width
    y_center = (float(y_min) + float(y_max)) / 2.0 / height
    bbox_width = (float(x_max) - float(x_min)) / width
    bbox_height = (float(y_max) - float(y_min)) / height
    return f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}"

def process_labels(dataset_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir) and class_name in class_mapping:
            print(f"Processing class directory: {class_dir}")
            label_dir = os.path.join(class_dir, "Label")
            if not os.path.exists(label_dir):
                print(f"Label directory not found: {label_dir}")
                continue

            output_class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                print(f"Created output class directory: {output_class_dir}")

            for image_file in os.listdir(class_dir):
                if image_file.endswith(".jpg") or image_file.endswith(".png"):
                    image_path = os.path.join(class_dir, image_file)
                    label_file = os.path.splitext(image_file)[0] + ".txt"
                    label_path = os.path.join(label_dir, label_file)

                    if os.path.exists(label_path):
                        print(f"Processing label file: {label_path}")
                        with open(label_path, "r") as f:
                            label = f.readline().strip()
                            print(f"Read label: {label}")

                        with Image.open(image_path) as img:
                            width, height = img.size
                            print(f"Image dimensions: width={width}, height={height}")

                        yolo_label = convert_to_yolo_format(label, width, height)
                        print(f"Converted label: {yolo_label}")

                        output_label_path = os.path.join(output_class_dir, label_file)
                        with open(output_label_path, "w") as f:
                            f.write(yolo_label + "\n")
                            print(f"Saved YOLO label to: {output_label_path}")

# Usage
dataset_dir = "Autodistill_dataset"
output_dir = "YOLO_dataset"
process_labels(dataset_dir, output_dir)
