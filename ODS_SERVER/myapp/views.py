from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from django.conf import settings
from .object_detection import detect_images, detect_video 
import cv2
import base64
import numpy as np
from ultralytics import YOLO

def landing_page(request):
    return render(request, 'landing.html')

def detection_page(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_name = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
        file_path = os.path.join(default_storage.location, file_name)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.gif', '.mkv'}

        _, file_extension = os.path.splitext(file_name)
        file_extension = file_extension.lower() 

        if file_extension in image_extensions:
            filename, detection_results = detect_images(file_path)
            response_data = {
                'annotated_image': filename,
                'detection_results': detection_results,
            }
            return JsonResponse(response_data)
        elif file_extension in video_extensions:
            filename, results = detect_video(file_path)  
            response_data = {
                'annotated_video': results,
                'detection_results': filename,
            }
            print(f'Response = {response_data}')
            return JsonResponse(response_data)
        
        default_storage.delete(file_path)
        
    
    return render(request, 'detection.html')

def detect_camera(request):
    return render(request, 'live.html')

@csrf_exempt
def live(request):
    if request.method == 'POST':
        try:
            data = request.POST['image']
            image_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            model = YOLO('/static/yolov8n.pt')
            # Run YOLOv8 inference on the frame
            results = model(image)

            # Annotate the frame with detected objects
            annotated_frame = results[0].plot()

            # Convert annotated frame to base64 string
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            # Return the annotated frame as a JSON response
            return JsonResponse({'image': 'data:image/jpeg;base64,' + encoded_image})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)


def get_frames(request):
    try:
        frames_path = os.path.join(settings.BASE_DIR, 'static', 'video_frames')
        if os.path.exists(frames_path):
            frames = sorted(os.listdir(frames_path))
            return JsonResponse({'frames': frames})
        else:
            return JsonResponse({'frames': []})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)