from django import forms


class patientForm(forms.Form):
    patientFirstName = forms.CharField(label='Patient First Name')
    patientLastName = forms.CharField(label='Patient Last Name')
    patientDOB = forms.DateField(label='Patient Date of Birth', widget=forms.SelectDateWidget)
    birth_date= forms.DateField(label='Entry Date', widget=forms.SelectDateWidget)


class uploadForm(forms.Form):
    image = forms.ImageField(label='Patient CT Scan')
