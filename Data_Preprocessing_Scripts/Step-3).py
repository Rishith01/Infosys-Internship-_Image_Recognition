import os
import json
import cv2
import random
import re


def get_image_metadata(image_path):
    # Open the image to get its dimensions
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Bounding box calculations
    x = random.randint(10, 50)
    y = random.randint(10, 50)
    h = random.randint(height - 50, height)
    w = random.randint(width - 50, width)
    bbox = [x, y, h, w]
    return width, height, bbox


# update this category according to our classes
categories = [
    {"id": 1, "name": "scooty"},
    {"id": 2, "name": "bike"}
]


# Match filename for category name to get category ID
def get_category_id(filename):
    category_id = None
    match = re.match(r"([a-zA-Z]+)-\d+", filename)
    if match:
        name = match.group(1)

        # Find category ID based on name
        for category in categories:
            if category["name"] == name:
                category_id = category["id"]
                break
    return category_id


def create_coco_annotations(images_dir, annotations_data):
    coco_format = {"images": [], "annotations": [], "categories": categories}

    # Dummy categories

    image_id = 1
    annotation_id = 1

    for filename in os.listdir(images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, filename)
            width, height, bbox = get_image_metadata(image_path)

            # Add image info
            coco_format["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": filename
            })

            # Create dummy annotation data (replace with actual data retrieval)
            # Example bounding box
            category_id = get_category_id(filename)  # Example category id

            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3]  # width * height
            })

            image_id += 1
            annotation_id += 1

    return coco_format


def main(images_dir, output_json):
    coco_annotations = create_coco_annotations(images_dir)

    # Write to JSON file
    with open(output_json, 'w') as json_file:
        json.dump(coco_annotations, json_file, indent=4)


if __name__ == "__main__":
    # Input images directory
    images_dir = 'dataset/test'

    # Output annotation file directory
    output_json = 'dataset/test/annotations.json'

    main(images_dir, output_json)
