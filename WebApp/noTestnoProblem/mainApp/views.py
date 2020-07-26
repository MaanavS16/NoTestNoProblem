from django.http import HttpResponse
from django.shortcuts import render
from .forms import patientForm, uploadForm
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
from .predict import imagePredictor
import base64
#from .forms import

predictor = imagePredictor()

def home(request):
    context = {
        'pageName' : 'Home'
    }
    return render(request, 'mainApp/home.html', context)

def diagnose(request):
    form = patientForm()
    imageForm = uploadForm()
    context = {}
    if request.method == 'POST':
        form = patientForm(request.POST)
        imageForm = uploadForm(request.POST)

        imageData = request.FILES.get('image')
        img = Image.open(imageData)

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        #print(img_str)

        img = img.resize((256, 256)).convert('RGB')
        imgArr = np.array(img)
        preds = predictor.predictModel(imgArr)

        if preds[1] >= .5:
            covidPositve = False
            conf = preds[1]
        else:
            covidPositve = True
            conf = 1 - preds[1]

        context.update({
                'c1' : round((1-preds[0][0])*100, 3),
                'c2' : round((1-preds[0][1])*100, 3),
                'c3' : round((1-preds[0][2])*100, 3),
                'c4' : round((1-preds[0][3])*100, 3),
                'c5' : round((1-preds[0][4])*100, 3),
                'c6' : round((1-preds[0][5])*100, 3),
                'c7' : round((1-preds[0][6])*100, 3),
                'c8' : round((1-preds[0][7])*100, 3),
                'covidPositive' : covidPositve,
                'confidence' : round(conf*100, 3),
                'resultsLoaded': True,
                'bs64' : img_str.decode("utf-8")
                })

    context.update({
        'pageName' : 'Form',
        'form' : form,
        'imageForm': imageForm,
    })
    return render(request, 'mainApp/form.html', context)
