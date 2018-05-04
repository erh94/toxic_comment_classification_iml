from django import forms

from .models import ModelInput

class ModelInputForm(forms.ModelForm):

    class Meta:
        model = ModelInput
        fields = ('comment',)
