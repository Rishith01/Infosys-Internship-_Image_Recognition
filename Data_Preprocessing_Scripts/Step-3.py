import os
import cv2
import random


def resize(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    new_image = cv2.resize(image, (224, 224))
    return new_image


def augmenting_images(image_path):
    image = cv2.imread(image_path)
    chooser = random.randint(0, 1)
    # operations in order are vertical_rotation, horizontal_rotation, zoom, brightness, contrast
    if chooser:
        chooser2 = random.randint(0, 4)
        center = (image.shape[0] // 2, image.shape[1] // 2)
        if chooser2 == 0:
            rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            return rotated_image
        elif chooser2 == 1:
            angle = 90 * random.choice([-1, 1])
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            return rotated_image
        elif chooser2 == 2:
            # Calculate ROI for zooming
            zoom_factor = random.uniform(0.5, 2.0)
            height, width = image.shape[:2]
            zoomed_height = int(height * zoom_factor)
            zoomed_width = int(width * zoom_factor)
            y_start = (height - zoomed_height) // 2
            y_end = y_start + zoomed_height
            x_start = (width - zoomed_width) // 2
            x_end = x_start + zoomed_width

            # Crop to zoomed region
            zoomed_image = image[y_start:y_end, x_start:x_end]
            return zoomed_image
        elif chooser2 == 3:
            alpha = random.uniform(0.5, 2.0)
            beta = random.randint(-50, 50)
            brightness_varied_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return brightness_varied_image
        else:
            alpha = random.uniform(0.5, 2.0)
            contrast_image = cv2.convertScaleAbs(image, alpha=alpha)
            return contrast_image
    else:
        return image  # Return original image if randomizer is 0


def main():
    # Example usage: iterate through a directory of images
    parent_directory = 'Pretraining_Dataset_Images/images'
    output_dir = 'Preprocessed_Dataset_Images'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each image in the directory
    for class_name in os.listdir(parent_directory):
        class_dir = os.path.join(parent_directory, class_name)

        # Skip files in the parent directory
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                image_path = os.path.join(class_dir, filename)
                image = cv2.imread(image_path)
                # Apply augmentation
                augmented_image = augmenting_images(image)

                # Save augmented image
                output_path = os.path.join(output_dir, f'{class_name}_{filename}')
                cv2.imwrite(output_path, augmented_image)

                print(f'Saved augmented image: {output_path}')


if __name__ == "__main__":
    main()
