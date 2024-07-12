from ultralytics import YOLO
import os
import shutil
from django.conf import settings
import cv2
from moviepy.editor import VideoFileClip
from moviepy.video.fx.speedx import speedx


# Initialize YOLO model
# Function to get the latest prediction directory
# def get_latest_prediction_dir(base_dir='runs/detect'):
#     subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
#     if not subdirs:
#         raise FileNotFoundError(f"No subdirectories found in {base_dir}.")
#     latest_subdir = max(subdirs, key=os.path.getmtime)
#     return latest_subdir

# Function to perform object detection and return annotated image path and detection results
def detect_images(image_path):
    model = YOLO('/static/yolov8n.pt')
    # Define the output directory for annotated images
    output_dir = os.path.join(settings.STATICFILES_DIRS[0], 'yolo_predictions')

    
    # Remove the directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists('runs'):
        shutil.rmtree('runs')

    os.makedirs(output_dir, exist_ok=True)

    # Predict objects in the image and save the annotated image
    results = model.predict(source=image_path, save=True)
    
    # Extract results
    if isinstance(results, list):
        results = results[0]

    # Refine results
    refined_results = {
        'scores': [],
        'class_names': [],
    }

    boxes = results.boxes
    for box in boxes:
        refined_results['scores'].append(float(box.conf))  # Extract confidence scores
        refined_results['class_names'].append(results.names[int(box.cls)])  # Extract class names

    # Get the latest prediction directory
    latest_prediction_dir = 'runs/detect/predict'

    # Locate the saved annotated image
    annotated_image_path = None
    for file in os.listdir(latest_prediction_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            annotated_image_path = os.path.join(latest_prediction_dir, file)
            break

    if not annotated_image_path:
        raise FileNotFoundError("Annotated image not found in the latest prediction directory.")
    
    # Move the annotated image to the output directory
    new_annotated_image_path = os.path.join(output_dir, os.path.basename(annotated_image_path))
    filename = os.path.basename(annotated_image_path)
    shutil.move(annotated_image_path, new_annotated_image_path)

    # Remove the prediction directory after moving the image
    shutil.rmtree(latest_prediction_dir)

    return filename, refined_results



def detect_video(video_path):
    model = YOLO('/static/yolov8n.pt')
    # Define the output directory for annotated images
    output_dir = os.path.join(settings.STATICFILES_DIRS[0], 'yolo_predictions')
    

    # Remove the directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    if os.path.exists('runs'):
        shutil.rmtree('runs')

    os.makedirs(output_dir, exist_ok=True)

    unique_classes = set()

    results = model.predict(video_path, save=True, vid_stride=5)
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            class_name = r.names[class_id]
            unique_classes.add(class_name)

    # Convert the set to a list
    classes = list(unique_classes)

    latest_prediction_dir = 'runs/detect/predict'

    # Locate the saved annotated image
    annotated_video_path = None
    for file in os.listdir(latest_prediction_dir):
        if file.lower().endswith(('.avi', '.mp4')):
            annotated_video_path = os.path.join(latest_prediction_dir, file)
            break

    if not annotated_video_path:
        raise FileNotFoundError("Annotated image not found in the latest prediction directory.")
    
    # Move the annotated image to the output directory
    new_annotated_video_path = os.path.join(output_dir, os.path.basename(annotated_video_path))
    shutil.move(annotated_video_path, new_annotated_video_path)

    # Remove the prediction directory after moving the image
    shutil.rmtree(latest_prediction_dir)
    
    clip = VideoFileClip(new_annotated_video_path)

    slowed_clip = speedx(clip, 0.5) 
        # Slow down the video by 0.5x
    slow_video_output_path = os.path.join(output_dir, 'output_video.mp4')
    slowed_clip.write_videofile(slow_video_output_path, codec='libx264')
        
    return classes, 'output_video.mp4'
