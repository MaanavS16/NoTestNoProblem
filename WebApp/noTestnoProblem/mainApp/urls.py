from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='app-home'),
    path('diagnose', views.diagnose, name='app-diagnose')
]
