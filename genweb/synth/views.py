# synth/views.py
from django.shortcuts import render
from django.http import HttpResponseBadRequest
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from PIL import Image
import os, time

from .forms import InferForm
from .infer import run_infer

def home(request):
    # 팀 소개 랜딩 페이지
    return render(request, "home.html")

def project(request):
    return render(request, "project.html")

def generator(request):
    # 기존 index 로직을 그대로 가져와서 사용 (POST→저장→추론→결과 비교뷰)
    if request.method == "GET":
        return render(request, "generate.html", {"form": InferForm()})

    form = InferForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "generate.html", {"form": form})

    uploads_root = os.path.join(settings.MEDIA_ROOT, "uploads")
    fs = FileSystemStorage(location=uploads_root, base_url=settings.MEDIA_URL + "uploads/")

    ts = int(time.time())
    ok_file   = request.FILES["ok_image"]
    cond_file = request.FILES["cond_image"]
    ok_name   = fs.save(f"ok_{ts}_{ok_file.name}", ok_file)
    cond_name = fs.save(f"cond_{ts}_{cond_file.name}", cond_file)
    ok_url    = fs.url(ok_name)
    cond_url  = fs.url(cond_name)

    mask_file = request.FILES.get("mask_image")
    mask_url  = None
    mask_path = None
    if mask_file:
        mask_name = fs.save(f"mask_{ts}_{mask_file.name}", mask_file)
        mask_url  = fs.url(mask_name)
        mask_path = fs.path(mask_name)

    ok_img_path   = fs.path(ok_name)
    cond_img_path = fs.path(cond_name)
    ok_img   = Image.open(ok_img_path)
    cond_img = Image.open(cond_img_path)
    mask_img = Image.open(mask_path) if mask_path else None

    prompt = form.cleaned_data["prompt"]

    result_url = run_infer(
        ok_img=ok_img,
        mask_img=mask_img,
        cond_img=cond_img,
        prompt=prompt,
        unet_lora_dir="./checkpoints/unet_lora",
        ctrl_lora_dir="./checkpoints/ctrl_lora",
        seed=42,
    )

    cache_bust = f"?v={ts}"

    ctx = {
        "form": InferForm(),
        "result_path": result_url + cache_bust,
        "ok_url": ok_url + cache_bust,
        "cond_url": cond_url + cache_bust,
        "mask_url": (mask_url + cache_bust) if mask_url else None,
    }
    return render(request, "generate.html", ctx)
