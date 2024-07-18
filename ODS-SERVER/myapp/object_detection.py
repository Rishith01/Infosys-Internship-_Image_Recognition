from ultralytics import YOLO
import os
import shutil
from django.conf import settings
from moviepy.editor import VideoFileClip
from moviepy.video.fx.speedx import speedx


def detect_images(image_path, model_name, output_dir):
    model_path = os.path.join(settings.MEDIA_ROOT, f'models/{model_name}.pt')

    model = YOLO(model_path)

    file_name = image_path.split('\\')[-1]
    print('File name : ', file_name)


    prediction_dir = os.path.join(output_dir, 'predict')

    if os.path.exists(prediction_dir):
        shutil.rmtree(prediction_dir)

    # Predict objects in the image and save the annotated image
    results = model.predict(source=image_path, save=True, project=output_dir)

    # Extract results
    results = results[0]

    # Refine results
    refined_results = {
        'scores': [],
        'class_names': [],
    }

    boxes = results.boxes
    for box in boxes:
        refined_results['scores'].append(float(box.conf))
        refined_results['class_names'].append(results.names[int(box.cls)])

    session_id = output_dir.split('\\')[-1]

    annotated_image_path = os.path.join(prediction_dir, file_name)
    
    new_path_dir = os.path.join(settings.STATICFILES_DIRS[0], 'yolo_detections', session_id)

    if os.path.exists(new_path_dir):
        shutil.rmtree(new_path_dir)
    os.makedirs(new_path_dir, exist_ok=True)

    new_path = os.path.join(new_path_dir, file_name)

    shutil.move(annotated_image_path, new_path)

    if os.path.exists(prediction_dir):
        shutil.rmtree(prediction_dir)

    relative_path = os.path.join('static', 'yolo_detections', session_id, file_name)
    relative_path = relative_path.replace('\\', '/')

    return relative_path, refined_results



def detect_video(video_path, model_name, output_dir):
    model_path = os.path.join(settings.MEDIA_ROOT, f'models/{model_name}.pt')
    model = YOLO(model_path)

    prediction_dir = os.path.join(output_dir, 'predict')

    print('Prediction Dir : ', prediction_dir)

    # Remove the directory if it exists
    if os.path.exists(prediction_dir):
        shutil.rmtree(prediction_dir)
    
    unique_classes = set()

    results = model.predict(video_path, save=True, vid_stride=2, project=output_dir)
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            class_name = r.names[class_id]
            unique_classes.add(class_name)

    # Convert the set to a list
    classes = list(unique_classes)

    session_id = output_dir.split('\\')[-1]

    file_name = video_path.split('\\')[-1]
    base_name, _ = os.path.splitext(file_name)
    print('File Name : ', file_name)
    print('Base Name : ', base_name)

    # Locate the saved annotated image
    annotated_video_path = os.path.join(prediction_dir, f'{base_name}.avi')

    print('Annotated video path :' ,annotated_video_path)
    
    # Move the annotated image to the output directory
    new_path_dir = os.path.join(settings.STATICFILES_DIRS[0], 'yolo_detections', session_id)

    print(f'New path dir : {new_path_dir}')

    if os.path.exists(new_path_dir):
        shutil.rmtree(new_path_dir)

    os.makedirs(new_path_dir, exist_ok=True)
    
    clip = VideoFileClip(annotated_video_path)
    slowed_clip = speedx(clip, 0.5)
    slow_video_output_path = os.path.join(new_path_dir, os.path.splitext(file_name)[0] + '.mp4')

    print('Slow video path : ', slow_video_output_path)

    slowed_clip.write_videofile(slow_video_output_path, codec='libx264')

    print('Video Done!!!!!!!!!!')
    if os.path.exists(prediction_dir):
        shutil.rmtree(prediction_dir)

    print('File Removed......')

    relative_path = os.path.join('static', 'yolo_detections', session_id, os.path.basename(slow_video_output_path))

    relative_path = relative_path.replace('\\', '/')
    print('relative path', relative_path)

    return relative_path, classes
