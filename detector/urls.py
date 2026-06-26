from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    path('history/', views.history, name='history'),

    # Authentication
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # API endpoints
    path('upload/', views.upload, name='upload'),
    path('predict/', views.predict, name='predict'),
    path('check-quality/', views.check_quality, name='check_quality'),
    path('dashboard/', views.dashboard, name='dashboard'),
]