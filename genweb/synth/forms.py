# forms.py
from django import forms

class InferForm(forms.Form):
    ok_image = forms.ImageField(label="OK 이미지")
    cond_image = forms.ImageField(label="결함 마스크 (ControlNet 조건)")
    mask_image = forms.ImageField(label="Inpaint 마스크", required=False)
    prompt = forms.CharField(label="텍스트 프롬프트", widget=forms.Textarea)
