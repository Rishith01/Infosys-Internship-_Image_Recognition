from rembg import remove
from PIL import Image, ImageFilter
import os
import random
from sklearn.model_selection import train_test_split

# Directories
input_dir = 'dataset/train'
output_dir = 'dataset/final'
train_dir = f'{output_dir}/train'
test_dir = f'{output_dir}/test'
val_dir = f'{output_dir}/val'

# Create output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get list of images
image_list = os.listdir(input_dir)

# Split the images into train, test, and validation sets
train_files, test_files = train_test_split(image_list, test_size=0.2, random_state=42)
test_files, val_files = train_test_split(test_files, test_size=0.5, random_state=42)

# Function to process images
def process_images(files, target_dir, class_name='Scooty'):
    index = 1
    for image in files:
        # Store paths
        input_path = os.path.join(input_dir, image)
        output_path = 'output.png'
        sick_image_path = os.path.join(target_dir, f'{class_name}-{index}.png')
        index += 1

        # Open the input image
        input_image = Image.open(input_path)

        # Removing the background from the input image
        output_image = remove(input_image)

        # Save the extracted foreground
        output_image.save(output_path)

        # Blur the original image
        blurred_image = input_image.filter(ImageFilter.GaussianBlur(3))

        # Overlay the extracted foreground onto the blurred image
        blurred_image.paste(output_image, (0, 0), output_image)

        # Save the resulting image
        blurred_image.save(sick_image_path)
        print(f'{sick_image_path} Done!')

# Process images for each set
print("Processing training images...")
process_images(train_files, train_dir)
print("Processing test images...")
process_images(test_files, test_dir)
print("Processing validation images...")
process_images(val_files, val_dir)

print("All images processed and saved into train, test, and val directories.")
