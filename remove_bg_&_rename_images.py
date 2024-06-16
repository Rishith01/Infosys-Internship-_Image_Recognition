from rembg import remove
from PIL import Image, ImageFilter
import os

input_dir = 'dataset/train'
image_list = os.listdir(input_dir)

output_dir = 'dataset/Blurred'
index = 1

for image in image_list:
    # Store paths
    input_path = f'{input_dir}/{image}'
    output_path = 'output.png'
    class_name = 'Scooty'

    sick_image_path = f'{output_dir}/{class_name}-{index}.png'
    index = index + 1
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