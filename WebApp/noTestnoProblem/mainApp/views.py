from django.http import HttpResponse
from django.shortcuts import render
#from .forms import

def home(request):
    return render(request, 'mainApp/home.html')
