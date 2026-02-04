from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),  # ✅ FIXED: Changed from upload_file to upload
    path('predict/', views.predict, name='predict'),
    path('history/', views.history, name='history'),
]