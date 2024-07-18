from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from django.conf import settings
from .object_detection import detect_images, detect_video 
import cv2
import base64
import json
import numpy as np
from ultralytics import YOLO
import shutil
from .process_image import main_processing
import yaml
import uuid


def generate_session_id(request):
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4())
    return request.session['session_id']

def get_session_dir(request):
    session_id = generate_session_id(request)
    session_dir = os.path.join(settings.MEDIA_ROOT, 'sessions', session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def landing_page(request):
    return render(request, 'landing.html')
@csrf_exempt
def detection_page(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        model_name = request.POST.get('model')
        model_names = []

        with open(os.path.join(settings.STATICFILES_DIRS[0], 'models.json'), 'r') as file:
            data = json.load(file)
            if data:
                model_names = data.get('model_names', [])

        if not model_name or model_name not in model_names:
            model_name = 'yolov8n'

        file_name = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
        file_path = os.path.join(default_storage.location, file_name)

        session_dir = get_session_dir(request)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.gif', '.mkv'}

        _, file_extension = os.path.splitext(file_name)
        file_extension = file_extension.lower() 

        try:
            if file_extension in image_extensions:
                filepath, detection_results = detect_images(file_path, model_name, session_dir)
                response_data = {
                    'annotated_image': filepath,
                    'detection_results': detection_results,
                }
            elif file_extension in video_extensions:
                filepath, detection_results = detect_video(file_path, model_name, session_dir)
                response_data = {
                    'annotated_video': filepath,
                    'detection_results': detection_results,
                }
            else:
                return JsonResponse({'error': 'Unsupported file type'}, status=400)

            default_storage.delete(file_path)
            return JsonResponse(response_data)
        except Exception as e:
            default_storage.delete(file_path)
            return JsonResponse({'error': f'An error occurred! Please try again. {str(e)}'}, status=500)

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


def training_page(request):
    return render(request, 'training.html')

@csrf_exempt
def upload_class_and_images(request):
    if request.method == 'POST':
        class_name = request.POST.get('class-name', '').strip()
        if not class_name:
            return JsonResponse({'error': 'Class name cannot be empty'}, status=400)

        files = request.FILES.getlist('class-image')
        if not files:
            return JsonResponse({'error': 'No images uploaded'}, status=400)

        session_dir = get_session_dir(request)
        output_dir = os.path.join(session_dir, 'dataset')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create images subfolder within Dataset folder
        images_folder = os.path.join(output_dir, 'images')
        os.makedirs(images_folder, exist_ok=True)

        # Create class folder within images folder
        class_folder = os.path.join(images_folder, class_name.lower())
        os.makedirs(class_folder, exist_ok=True)

        # Save images to the class folder
        for file in files:
            file_path = os.path.join(class_folder, file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

        return JsonResponse({'message': 'Class and images uploaded successfully'})
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
@require_POST
def process_images(request):
    if request.method == 'POST':
        session_dir = get_session_dir(request)
        output_dir = os.path.join(session_dir, 'dataset')
        try:
            main_processing(session_dir)
            return JsonResponse({'message': 'Images processed successfully'}, status=200)
        except Exception as e:
            if(os.path.exists(output_dir)):
                shutil.rmtree(output_dir)
            if(os.path.exists(os.path.join(session_dir, 'data'))):
                shutil.rmtree(os.path.join(session_dir, 'data'))
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
@require_POST
def train_AI(request):
    if request.method == 'POST':
        session_dir = get_session_dir(request)
        try:
            print('Training Started.................')
            
            data_file = os.path.join(session_dir, 'data', 'data.yaml')
            pretrained_model_path = os.path.join(settings.MEDIA_ROOT, 'models/yolov8s.pt')

            train_path = os.path.join(session_dir, 'train')

            if os.path.exists(train_path):
                shutil.rmtree(train_path)
        
            num_epochs = 0
            if(os.path.exists(data_file)):
                print('data.yaml exists')
                with open(data_file, 'r') as file:
                    data = yaml.safe_load(file)
                    num_classes = len(data['names'])
                    num_epochs = min(100, num_classes * 10)

                
                model = YOLO(pretrained_model_path)
                try:
                    model.train(data=data_file, epochs=num_epochs, project=session_dir)
                    print('Training Done...........')
                except Exception as e:
                    if os.path.exists(os.path.join(session_dir, 'train')):
                        shutil.rmtree(os.path.join(session_dir, 'train'))
                    return JsonResponse({'error': 'An error occured'}, status=500)
                
                session_id = session_dir.split('\\')[-1]
                
                result_path = os.path.join(train_path, 'results.png')

                result_save_path = os.path.join(settings.STATICFILES_DIRS[0], 'yolo_train', session_id)
                
                if os.path.exists(result_save_path):
                    shutil.rmtree(result_save_path)

                os.makedirs(result_save_path)

                result_img_path = os.path.join(result_save_path, 'results.png')

                shutil.move(result_path, result_img_path)
                
                new_model_path = os.path.join(train_path, 'weights/best.pt')

                temp_model_path = os.path.join(session_dir, 'temp_model')

                if os.path.exists(temp_model_path):
                    shutil.rmtree(temp_model_path)
                os.makedirs(temp_model_path, exist_ok=True)

                shutil.move(new_model_path, os.path.join(temp_model_path, 'temp.pt'))

                if os.path.exists(os.path.join(session_dir, 'train')):
                    shutil.rmtree(os.path.join(session_dir, 'train'))

                if os.path.exists(os.path.join(session_dir, 'dataset')):
                    shutil.rmtree(os.path.join(session_dir, 'dataset'))

                relative_path = os.path.join('static', 'yolo_train', session_id, 'results.png')
                relative_path = relative_path.replace('\\', '/')

            return JsonResponse({'message': 'Training successfull', 'img': relative_path}, status=200)
        
        except Exception as e:
            if(os.path.exists(os.path.join(session_dir, 'data'))):
                shutil.rmtree(os.path.join(session_dir, 'data'))
            if os.path.exists(os.path.join(session_dir, 'train')):
                shutil.rmtree(os.path.join(session_dir, 'train'))

            return JsonResponse({'error': 'Training failed'}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
@require_POST
def test_model(request):
    if request.method == 'POST':
        session_dir = get_session_dir(request)
        try:
            file = request.FILES['file']
            model_path = os.path.join(session_dir, 'temp_model', 'temp.pt')
            print('Model Path : ', model_path)
            print('File Name : ', file.name)
            file_name = default_storage.save(file.name, ContentFile(file.read()))
            file_path = os.path.join(default_storage.location, file_name)
            print('File Path : ', file_path)
            model = YOLO(model_path)
            print('Test File : ', file_name)
            runs_dir = os.path.join(session_dir, 'test')
            if os.path.exists(runs_dir):
                shutil.rmtree(runs_dir)

            model.predict(file_path, save=True, project=runs_dir)

            print('Predicted........')

            image_path = os.path.join(runs_dir, f'predict/{file.name}')
            print('Image Path', image_path)

            session_id = session_dir.split('\\')[-1]

            save_path = os.path.join(settings.STATICFILES_DIRS[0],'yolo_test', session_id)

            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            
            os.makedirs(save_path, exist_ok=True)

            default_storage.delete(file_path)

            print('Save Path', save_path)

            shutil.move(image_path, os.path.join(save_path, file.name))

            print('Test Image Moved')

            if os.path.exists(runs_dir):
                shutil.rmtree(runs_dir)

            relative_path = os.path.join('static', 'yolo_test', session_id, file.name)

            relative_path = relative_path.replace('\\', '/')

            print('Relative Path : ', relative_path)

            return JsonResponse({'img': relative_path})

        except Exception as e:
            return JsonResponse({'error': 'An error occured!'})

@csrf_exempt       
@require_POST
def upload_model(request):
    if request.method == 'POST':
        session_dir = get_session_dir(request)
        # session_id = session_dir.split('\\')[-1]
        try:
            model_name = request.POST.get('model-name')
            model_desc = request.POST.get('model-desc')

            print('Model Name : ', model_name)
            print('Model Desc : ', model_desc)

            temp_model = os.path.join(session_dir, 'temp_model/temp.pt')
            print('Temp Model : ', temp_model)
            new_model = os.path.join(settings.MEDIA_ROOT, f'models/{model_name}.pt')
            print('New Model : ', new_model)
            shutil.copy(temp_model, new_model)
            print('Model Moved!')
            data = None
            with open(os.path.join(settings.STATICFILES_DIRS[0], 'models.json'), 'r') as file :
                data = json.load(file)
            
            if data:
                data["model_names"].append(model_name)
                data["model_description"].append(model_desc)

                with open(os.path.join(settings.STATICFILES_DIRS[0], 'models.json'), 'w') as file:
                    json.dump(data, file, indent=4)
                
                print('JSON Updated')

                if os.path.exists(os.path.join(session_dir, 'temp_model')):
                    shutil.rmtree(os.path.join(session_dir, 'temp_model'))

                return JsonResponse({'message': 'Model Uploaded successfully!'}, status=200)
            
            else:
                return JsonResponse({'message': 'Model Uploading failed!'}, status=500)
            
        except Exception as e:
            return JsonResponse({'message': 'Model Uploading failed!'}, status=500)