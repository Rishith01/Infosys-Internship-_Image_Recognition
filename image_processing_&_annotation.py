import os
import re
import random
from PIL import Image
from rembg import remove
import shutil

# Define your class mapping with class index
class_mapping = {
    'apple': 0,
    'banana': 1,
    'bicycle': 2,
    'bird': 3,
    'cat': 4,
    'clock': 5,
    'cup': 6,
    'dog': 7,
    'helicopter': 8,
    'laptop': 9,
    'orange': 10,
    'sandal': 11,
    'shoe': 12,
    'table': 13,
}


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


def extract_class_index(file_name):
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


def place_images_on_canvas(image_paths, background_images, save_image_dir, annotation_dir, min_size_ratio=0.2,
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

        placed_positions.append(bounding_box)
        canvas.paste(img, (top_left_x, top_left_y), img)

        x_min = top_left_x + 5
        y_min = top_left_y + 5
        x_max = top_left_x + img.size[0] - 5
        y_max = top_left_y + img.size[1] - 5

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        image_width, image_height = canvas_size
        x, y, w, h = convert_to_yolo_format(x_center, y_center, width, height, image_width, image_height)

        file_name = os.path.basename(image_path)
        class_index = extract_class_index(file_name)
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


def split_data(image_dir, annotation_dir, train_ratio=0.8):
    # Create directories for train and val sets
    train_images_dir = 'data/train/images'
    train_annotations_dir = 'data/train/labels'
    val_images_dir = 'data/val/images'
    val_annotations_dir = 'data/val/labels'

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

if __name__ == '__main__':
    '''
       ### Folder Structure ###
        
        Main Directory
            |---images
            |      |---apple
            |      |     |---Image_1.png
            |      |     |---Image_2.png
            |      |     |---Image_3.png
            |      |            .
            |      |            .
            |      |            . 
            |      |            .
            |      |---banana
            |      |---cat
            |      .
            |      .
            |      .
            |          
            |
            |---backgrounds
            |       |---bg-1.jpg
            |       |---bg-2.jpeg
            |       |---bg3.jpg
            |             .
            |             .
            |             .

     # Other directories will be automatically created in Main Directory
     
       data folder will have final dataset which will be used for training.
    '''


    input_image_dir = 'images'
    output_image_dir = 'processed_images'
    background_dir = 'backgrounds'
    subfolders = os.listdir(input_image_dir);
    print(subfolders)
    for folder in subfolders:
        input_folder = f'{input_image_dir}/{folder}'
        images_dir = f'{output_image_dir}/{folder}/images'  # Replace with your save image directory
        labels_dir = f'{output_image_dir}/{folder}/labels'  # Replace with your annotation directory

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        rename_images(input_folder, folder)
        images = [os.path.join(input_folder, img) for img in os.listdir(input_folder) if
                  img.endswith(('.jpg', '.jpeg', '.png'))]
        print(images)
        backgrounds = [os.path.join(background_dir, img) for img in os.listdir(background_dir) if
                       img.endswith(('.jpg', '.jpeg', '.png'))]

        # Rename images in order

        num_combinations = 100
        for _ in range(num_combinations):
            num_images = random.randint(1, 5)
            selected_images = random.sample(images, num_images)
            place_images_on_canvas(selected_images, backgrounds, images_dir, labels_dir)

        split_data(images_dir, labels_dir)

    shutil.rmtree(input_image_dir)
    shutil.rmtree(output_image_dir)

