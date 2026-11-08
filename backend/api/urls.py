"""
URL Configuration for API endpoints
"""

from django.urls import path
from api import views

urlpatterns = [
    path('detect/', views.detect_manufacturer, name='detect_manufacturer'),
    
    path('detect-stream/', views.detect_manufacturer_stream, name='detect_manufacturer_stream'),
    
    path('history/', views.get_inspection_history, name='inspection_history'),
    path('statistics/', views.get_statistics, name='statistics'),
    
    path('health/', views.health_check, name='health_check'),
]
