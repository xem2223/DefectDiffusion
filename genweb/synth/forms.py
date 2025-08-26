from django import forms

class UploadForm(forms.Form):
    ok_image   = forms.ImageField(label="OK 이미지 (원본)")
    mask_image = forms.ImageField(label="결함 마스크 (흑/백)")
    prompt     = forms.CharField(label="결함 프롬프트", max_length=200)

    # 선택 파라미터
    steps    = forms.IntegerField(initial=30, min_value=5, max_value=100)
    guidance = forms.FloatField(initial=7.5, min_value=0.0, max_value=20.0)
    strength = forms.FloatField(initial=0.35, min_value=0.05, max_value=1.0)
    seed     = forms.IntegerField(initial=123)
