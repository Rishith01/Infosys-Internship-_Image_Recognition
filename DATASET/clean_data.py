import os
from PIL import Image
import cv2
import random
import shutil

def rename_and_convert_images(directory, prefix):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    # List all files in the directory
    files = os.listdir(directory)

    # Filter out files that are not images (you can add more extensions if needed)
    image_extensions = ['.jpg', '.jpeg', '.png']
    images = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]

    # Sort images to ensure consistent numbering
    images.sort()

    # Rename and convert each image to JPEG with the prefix and a number
    for index, image in enumerate(images, start=1):
        # Create the new file name with .jpg extension
        new_name = f"{prefix}-{index:03d}.jpg"

        # Construct full file paths
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)

        # Check if the new file already exists to avoid reprocessing
        if os.path.exists(new_path):
            print(f"Skipping already processed image: '{new_name}'")
            continue

        # Open the image and convert to JPEG
        with Image.open(old_path) as img:
            # Convert image to RGB if it is not
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(new_path, "JPEG")

        # Remove the old image file
        os.remove(old_path)
        print(f"Converted and renamed '{image}' to '{new_name}'")

    print("Conversion and renaming completed.")

def augment_image(image):
    chooser = random.randint(0, 1)

    if chooser == 0:  # Vertical flip + brightness/contrast change
        image = cv2.flip(image, 0)
        beta = random.randint(-70, 70)  # Simple brightness control
        image = cv2.convertScaleAbs(image, beta=beta)
    elif chooser == 1:  # Horizontal flip + brightness/contrast change
        image = cv2.flip(image, 1)
        alpha = random.uniform(0.3, 2.0)  # Simple contrast control
        image = cv2.convertScaleAbs(image, alpha=alpha)

    return image

def process_images(input_dir):
    print('Starting image processing...')

    image_files = os.listdir(input_dir)

    for filename in image_files:
        # Skip images that have already been augmented
        if filename.endswith('-aug.jpg'):
            print(f"Skipping already augmented image: '{filename}'")
            continue

        print(f'Processing image: {filename}')
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        name = filename.split('.')[0]

        augmented_image = augment_image(image)
        output_path = os.path.join(input_dir, f'{name}-aug.jpg')
        cv2.imwrite(output_path, augmented_image)

        print(f'Saved augmented image: {output_path}')

def split_images(source_dir, train_dir, val_dir, split_ratio=0.8):
    # Create target directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List all files in the source directory
    files = os.listdir(source_dir)
    # Shuffle the list of files
    random.shuffle(files)

    # Calculate split index based on split_ratio
    split_index = int(len(files) * split_ratio)

    # Split files into train and val sets
    train_files = files[:split_index]
    val_files = files[split_index:]

    # Move files to train directory
    for file in train_files:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(train_dir, file)
        shutil.copyfile(source_path, target_path)

    # Move files to val directory
    for file in val_files:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(val_dir, file)
        shutil.copyfile(source_path, target_path)


# Example usage
directory_path = "images/Television"
prefix = "TV"
train_directory = "Dataset/train/images"
val_directory = "Dataset/validation/images"

rename_and_convert_images(directory_path, prefix)
process_images(directory_path)
split_images(directory_path, train_directory, val_directory)
