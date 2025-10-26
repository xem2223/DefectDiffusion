# generator/views.py

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from PIL import Image
import os

from .models_loader import run_infer, GLOBAL_PIPE, calculate_metrics

def home_view(request):
    """홈페이지 (home.html)를 렌더링합니다."""
    return render(request, 'home.html') 

def project_view(request):
    return render(request, "project.html")

def generate_view(request):
    """이미지 입력 폼을 표시하고, POST 요청이 오면 처리합니다."""
    # HTML 템플릿의 generated_image_url 변수에 전달될 최종 URL
    generated_image_url = None
    error_message = None
    metrics = None

    # 🌟 이미지 생성 버튼을 누를 때 (POST 요청)
    if request.method == 'POST':
        # 1. 파일 및 프롬프트 데이터 가져오기 (HTML의 name 속성으로 직접 접근)
        image1 = request.FILES.get('image1')  # Cond Image (결함 마스크)
        image2 = request.FILES.get('image2')  # OK Image (정상 이미지)
        prompt = request.POST.get('prompt')   # 결함 프롬프트

        if image1 and image2 and prompt:
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            
            # 2. 파일 저장 (추론 전에 PIL로 로드하기 위해)
            # 원본 파일명 유지 및 접두사 사용
            filename1 = fs.save("cond_" + image1.name, image1)
            filename2 = fs.save("ok_" + image2.name, image2)
            
            # 3. 파일 경로에서 PIL Image 로드
            try:
                # Cond/Mask 이미지를 image1로, OK/Init 이미지를 image2로 사용합니다.
                cond_img = Image.open(fs.path(filename1)).convert("RGB")
                ok_img = Image.open(fs.path(filename2)).convert("RGB")
                
            except Exception as e:
                error_message = f"이미지 파일 로드 실패: {e}"
                
            if not error_message:
                # 4. 모델 추론 실행 (결과: 로컬 절대 경로)
                try:
                    # 🌟 run_infer 호출: models_loader의 run_infer API와 일치시킵니다.
                    result_path_abs = run_infer(
                        ok_img=ok_img,
                        cond_img=cond_img,
                        prompt=prompt,
                        # models_loader.py에 정의된 상대 경로를 전달 (로딩 내부에서 사용됨)
                        unet_lora_dir="./checkpoints/unet_lora", 
                        ctrl_lora_dir="./checkpoints/ctrl_lora", 
                        # 하드코딩된 파라미터 (원하는 값으로 수정 가능)
                        steps=40, 
                        scale=7.5, 
                        strength=0.8, 
                        cscale=4.0, 
                        mask_feather=6, 
                        debug=False,
                        primary_seed=8 # 시드 고정
                    )
                    
                    # 5. 결과 파일 절대 경로를 URL로 변환
                    result_filename_rel = os.path.relpath(result_path_abs, settings.MEDIA_ROOT)
                    generated_image_url = settings.MEDIA_URL + result_filename_rel
                    
                    print(f"[SUCCESS] 최종 생성된 이미지 URL: {generated_image_url}")

                    generated_img = Image.open(result_path_abs).convert("RGB")
                    metrics = calculate_metrics(ok_img, generated_img) 
                    print(f"[METRICS] LPIPS: {metrics['lpips']}") # 🌟 출력 지표 변경

                except Exception as e:
                    # 추론 중 발생하는 모든 오류를 처리
                    error_message = f"AI 추론 중 오류 발생: {e}"
                    print(f"[ERROR] 추론 중 오류: {e}")
        else:
             error_message = "모든 필수 필드를 채워주세요."
        
    context = {
        'generated_image_url': generated_image_url,
        'error': error_message, # 오류 메시지를 템플릿에 전달하여 표시
        'metrics': metrics,
    }
    
    # templates/index.html 파일을 렌더링
    return render(request, 'index.html', context)
