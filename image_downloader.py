# pip install bing_image_downloader
from bing_image_downloader import downloader

# add your class name
class_name = ('Purse in white background')

# no of photos to download
no_of_photos = 30

# Images download directory
output_directory = 'Purse in white background'

# Run the program
downloader.download(query=class_name, limit=no_of_photos,  output_dir=output_directory)