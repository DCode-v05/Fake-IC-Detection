"""
URL Configuration for API endpoints
"""

from django.urls import path
from api import views

urlpatterns = [
    # Main detection endpoint
    path('detect/', views.detect_manufacturer, name='detect_manufacturer'),
    
    # History and statistics
    path('history/', views.get_inspection_history, name='inspection_history'),
    path('statistics/', views.get_statistics, name='statistics'),
    
    # Health check
    path('health/', views.health_check, name='health_check'),
]
