import re
import os
import random
import shutil
import yaml
from PIL import Image
from rembg import remove
from django.conf import settings

def rename_images(image_dir, base_name):
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if re.match(r'.*\.(jpg|png|jpeg)', f, re.IGNORECASE)]

    # Sort the image files for consistent renaming
    image_files.sort()

    # Rename each file sequentially
    for index, file_name in enumerate(image_files, start=1):
        # Get the file extension
        file_ext = file_name.split('.')[-1]

        # Create the new file name
        new_file_name = f"{base_name}-{index:03d}.{file_ext}"

        # Define the full old and new file paths
        old_file_path = os.path.join(image_dir, file_name)
        new_file_path = os.path.join(image_dir, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {file_name} to {new_file_name}")

    print("Renaming completed.")

def extract_class_index(file_name, class_mapping):
    match = re.match(r'([A-Za-z]+)-\d+', file_name)
    if match:
        class_name = match.group(1).lower()  # Use lowercase to match class_mapping
        if class_name in class_mapping:
            return class_mapping[class_name]
    return None

def convert_to_yolo_format(x_center, y_center, width, height, image_width, image_height):
    x = x_center / image_width
    y = y_center / image_height
    w = width / image_width
    h = height / image_height
    return x, y, w, h

def place_images_on_canvas(image_paths, background_images, save_image_dir, annotation_dir, class_mapping, min_size_ratio=0.2,
                           max_size_ratio=0.7):
    canvas_sizes = [(640, 640), (640, 480), (480, 640)]
    canvas_size = random.choice(canvas_sizes)

    print('Canvassing started....')

    annotations = []
    placed_positions = []
    output_path = 'output.png'
    flip_output_path = 'flip.png'

    background_path = random.choice(background_images)
    canvas = Image.open(background_path).convert('RGBA').resize(canvas_size)

    # Decide placement strategy
    horizontal_placement = random.choice([True, False])

    for image_path in image_paths:
        input_image = Image.open(image_path).convert('RGBA')
        if not (input_image.mode in ('RGBA', 'LA') or (input_image.mode == 'P' and 'transparency' in input_image.info)):
            output_image = remove(input_image)
            output_image.save(output_path)

            img = Image.open(output_path).convert('RGBA')
        else:
            img = input_image

        min_size = int(min(canvas_size) * min_size_ratio)
        max_size = int(min(canvas_size) * max_size_ratio)
        new_size = random.randint(min_size, max_size)

        if random.choice([True, False]):
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            img.save(flip_output_path)
            img = Image.open(flip_output_path).convert('RGBA')

        img.thumbnail((new_size, new_size), Image.BICUBIC)

        if horizontal_placement:
            if not placed_positions:
                top_left_x = random.randint(0, canvas_size[0] // 10)  # Starting x position
            else:
                last_position = placed_positions[-1]
                top_left_x = last_position[2] + random.randint(0, 10)  # Place next to the last image with a small gap
            top_left_y = (canvas_size[1] - img.size[1]) // 2 + random.randint(-10, 10)  # Centered with slight vertical variation
            if top_left_x + img.size[0] > canvas_size[0]:  # If the image goes beyond canvas width, skip placement
                continue
        else:
            max_x = canvas_size[0] - img.size[0]
            max_y = canvas_size[1] - img.size[1]
            overlap = True
            attempts = 0

            while overlap and attempts < 100:
                overlap = False
                top_left_x = random.randint(0, max_x)
                top_left_y = random.randint(0, max_y)
                bounding_box = (top_left_x, top_left_y, top_left_x + img.size[0], top_left_y + img.size[1])

                for pos in placed_positions:
                    if (bounding_box[0] < pos[2] and bounding_box[2] > pos[0] and
                            bounding_box[1] < pos[3] and bounding_box[3] > pos[1]):
                        overlap = True
                        break
                attempts += 1

            if overlap:
                continue

        placed_positions.append((top_left_x, top_left_y, top_left_x + img.size[0], top_left_y + img.size[1]))
        canvas.paste(img, (top_left_x, top_left_y), img)

        x_min = top_left_x + 2
        y_min = top_left_y + 2
        x_max = top_left_x + img.size[0] - 2
        y_max = top_left_y + img.size[1] - 2

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        image_width, image_height = canvas_size
        x, y, w, h = convert_to_yolo_format(x_center, y_center, width, height, image_width, image_height)

        file_name = os.path.basename(image_path)
        class_index = extract_class_index(file_name, class_mapping)
        
        if class_index is not None:
            annotations.append(f"{class_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    if annotations:
        prefix = os.path.splitext(os.path.basename(image_paths[0]))[0].split('-')[0].lower()
        index = 1
        while os.path.exists(os.path.join(save_image_dir, f"{prefix}-comb-{index:03}.jpg")):
            index += 1

        annotation_file = os.path.join(annotation_dir, f"{prefix}-comb-{index:03}.txt")
        with open(annotation_file, 'w') as f:
            for annotation in annotations:
                f.write(f"{annotation}\n")

        print(f"Annotation saved: {annotation_file}")

        save_image_path = os.path.join(save_image_dir, f"{prefix}-comb-{index:03}.jpg")
        canvas.convert('RGB').save(save_image_path)  # Convert to RGB before saving
        print(f"Image saved: {save_image_path}")


def split_data(session_dir, image_dir, annotation_dir, train_ratio=0.8):
    # Create directories for train and val sets
    train_images_dir = os.path.join(session_dir, 'data', 'train', 'images')
    train_annotations_dir = os.path.join(session_dir, 'data', 'train', 'labels')
    val_images_dir = os.path.join(session_dir, 'data', 'val', 'images')
    val_annotations_dir = os.path.join(session_dir, 'data', 'val', 'labels')
   
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_annotations_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_annotations_dir, exist_ok=True)

  
    # Get list of images and annotations
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]

    
    # Shuffle files to randomize the split
    random.shuffle(image_files)
    random.shuffle(annotation_files)

    # Calculate number of files for train set
    num_train = int(len(image_files) * train_ratio)

    # Split images
    train_images = image_files[:num_train]
    val_images = image_files[num_train:]

    # Split annotations based on corresponding images
    train_annotations = [f.replace('.jpg', '.txt') for f in train_images]
    val_annotations = [f.replace('.jpg', '.txt') for f in val_images]

    # Move train images and annotations
    for img in train_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_images_dir, img))
    for ann in train_annotations:
        shutil.copy(os.path.join(annotation_dir, ann), os.path.join(train_annotations_dir, ann))

    # Move val images and annotations
    for img in val_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(val_images_dir, img))
    for ann in val_annotations:
        shutil.copy(os.path.join(annotation_dir, ann), os.path.join(val_annotations_dir, ann))

    print(f"Split complete. {len(train_images)} images for training and {len(val_images)} images for validation.")

def create_data_yaml(data_dir, train_dir, val_dir, class_mapping):
    reversed_mapping = {index: class_name for class_name, index in class_mapping.items()}
    data_yaml = {
        'path': data_dir,
        'train': train_dir,
        'val': val_dir,
        'names': reversed_mapping
    }
    print(f'data.yaml = {data_yaml}')
    yaml_file = os.path.join(data_dir, 'data.yaml')
    with open(yaml_file, 'w') as file:
        yaml.dump(data_yaml, file, default_flow_style=False)

    print(f"Created data.yaml file at {yaml_file}")


def main_processing(session_dir):
    output_dir = os.path.join(session_dir, 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    input_image_dir = os.path.join(output_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'processed_images')
    background_dir = os.path.join(settings.MEDIA_ROOT, 'backgrounds')

    # Get subfolders (class names) sorted alphabetically
    subfolders = sorted(os.listdir(input_image_dir))
    class_mapping = {class_name.lower(): index for index, class_name in enumerate(subfolders)}
    print(f'Class Mapping: {class_mapping}')
    
    for folder in subfolders:
        input_folder = os.path.join(input_image_dir, folder)
        images_dir = os.path.join(output_image_dir, folder, 'images')
        labels_dir = os.path.join(output_image_dir, folder, 'labels')
        num_combinations = min(100, (int)(len(os.listdir(input_folder)) * 1.5))

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        rename_images(input_folder, folder)
        images = [os.path.join(input_folder, img) for img in os.listdir(input_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
        backgrounds = [os.path.join(background_dir, img) for img in os.listdir(background_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

        for _ in range(num_combinations):
            num_images = random.randint(1, 5)
            selected_images = random.sample(images, num_images)
            place_images_on_canvas(selected_images, backgrounds, images_dir, labels_dir, class_mapping)

        split_data(session_dir, images_dir, labels_dir)

    # Create data.yaml file
    data_directory = os.path.join(session_dir, 'data')
    train_images_directory = os.path.join(session_dir, 'data/train/images')
    val_images_directory = os.path.join(session_dir, 'data/val/images')
    create_data_yaml(data_directory, train_images_directory, val_images_directory, class_mapping)

    # Clean up temporary directories
    shutil.rmtree(input_image_dir)
    shutil.rmtree(output_image_dir)