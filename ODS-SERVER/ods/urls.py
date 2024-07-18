"""
URL configuration for ods project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp.views import *

urlpatterns = [
    path('', landing_page, name='landing_page'),

    path('detect/', detection_page, name='detection_page'),
    path('live-camera-feed/', detect_camera, name='index'),
    path('live-camera-feed/live/', live, name='live'),

    path('train/', training_page, name='training_page'),
    path('train/submit-class/', upload_class_and_images, name='submit_class_image'),
    path('train/process-images/', process_images, name='process_image'),
    path('train/train-ai/', train_AI, name='train_ai'),

    path('train/test-ai/', test_model, name='test_ai'),
    path('train/upload-model/', upload_model, name='upoload_model'),
    path('admin/', admin.site.urls),
]
