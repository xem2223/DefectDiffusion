from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import io, base64, os, tempfile, shutil
from PIL import Image
import torch

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.models.attention_processor import AttnProcessor2_0
from peft import PeftModel

# ────────────────────────────────────────────────────────────────────
# 체크포인트 경로 
# ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DEFAULT_CKPT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "checkpoints"))

CTRL_LORA_DIR = os.getenv("CTRL_LORA_DIR", os.path.join(DEFAULT_CKPT_DIR, "ctrl_lora"))
UNET_LORA_DIR = os.getenv("UNET_LORA_DIR", os.path.join(DEFAULT_CKPT_DIR, "unet_lora"))

SD_ID = "runwayml/stable-diffusion-v1-5"
CTRL_BASE_ID = "lllyasviel/sd-controlnet-scribble"

# ────────────────────────────────────────────────────────────────────
# 모델 로드
# ────────────────────────────────────────────────────────────────────
pipe = None
try:
    print("────────────────────────────────────────────────────────")
    print(f"[Init] ControlNet base model: {CTRL_BASE_ID}")
    controlnet = ControlNetModel.from_pretrained(
        CTRL_BASE_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    print(f"[Init] SD base model: {SD_ID}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    # 안정적인 SDPA 어텐션
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.controlnet.set_attn_processor(AttnProcessor2_0())

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device_str)

    # LoRA 디렉터리 확인 후 로드
    print(f"[Init] Load LoRA adapters:")
    print(f"       UNet LoRA  -> {UNET_LORA_DIR}")
    print(f"       Ctrl LoRA  -> {CTRL_LORA_DIR}")
    if not (os.path.isdir(UNET_LORA_DIR) and os.path.isdir(CTRL_LORA_DIR)):
        raise FileNotFoundError("LoRA 디렉터리(UNET/CTRL)가 존재하지 않습니다.")

    # UNet / ControlNet LoRA 결합
    pipe.unet = PeftModel.from_pretrained(pipe.unet, UNET_LORA_DIR)
    pipe.controlnet = PeftModel.from_pretrained(pipe.controlnet, CTRL_LORA_DIR)

    # 추론 성능을 위해 병합
    pipe.unet = pipe.unet.merge_and_unload()
    pipe.controlnet = pipe.controlnet.merge_and_unload()

    print("[Init] Model ready. (SDPA on, LoRA merged)")
    print("────────────────────────────────────────────────────────")
except Exception as e:
    print("────────────────────────────────────────────────────────")
    print(f"[Init][Error] 모델 로딩 실패: {e}")
    print("────────────────────────────────────────────────────────")
    pipe = None


def index(request):
    return render(request, "pybo/index.html")


def _preprocess_image(fp, size=(512, 512), mode="RGB"):
    """간단 전처리: 리사이즈 + 색공간 변환"""
    img = Image.open(fp)
    if mode:
        img = img.convert(mode)
    if size:
        img = img.resize(size, Image.BICUBIC)
    return img


@csrf_exempt
def generate_image(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST 요청만 허용됩니다."}, status=405)
    if pipe is None:
        return JsonResponse({"error": "모델이 아직 초기화되지 않았습니다."}, status=500)

    temp_dir = None
    try:
        manufacturing_image_file = request.FILES.get("manufacturing-image")
        mask_image_file = request.FILES.get("mask-image")
        prompt = request.POST.get("prompt", "industrial product defect")
        negative_prompt = request.POST.get("negative_prompt", "blurry, unrealistic, artifacts, watermark")

        if not manufacturing_image_file or not mask_image_file:
            return JsonResponse({"error": "OK 이미지와 마스크 이미지를 모두 업로드해야 합니다."}, status=400)

        # 임시 저장
        temp_dir = tempfile.mkdtemp()
        manuf_path = os.path.join(temp_dir, manufacturing_image_file.name)
        mask_path = os.path.join(temp_dir, mask_image_file.name)
        with open(manuf_path, "wb+") as f:
            for chunk in manufacturing_image_file.chunks():
                f.write(chunk)
        with open(mask_path, "wb+") as f:
            for chunk in mask_image_file.chunks():
                f.write(chunk)

        # 전처리 (PIL)
        ok_img   = _preprocess_image(manuf_path, size=(512, 512), mode="RGB")
        mask_img = _preprocess_image(mask_path,  size=(512, 512), mode="RGB")  # L 또는 RGB 모두 가능

        # 하이퍼파라미터
        num_inference_steps = int(request.POST.get("steps", 30))
        guidance_scale = float(request.POST.get("cfg", 7.0))
        strength = float(request.POST.get("strength", 0.45))
        ctrl_scale = float(request.POST.get("ctrl_scale", 1.0))
        ctrl_start = float(request.POST.get("ctrl_start", 0.0))
        ctrl_end   = float(request.POST.get("ctrl_end",   0.6))
        seed = request.POST.get("seed", None)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device_str).manual_seed(int(seed)) if seed is not None else None

        # 추론
        with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=(device_str == "cuda")):
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=ok_img,                                  # img2img init
                controlnet_conditioning_image=mask_img,        # ControlNet 조건
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=ctrl_scale,
                control_guidance_start=ctrl_start,
                control_guidance_end=ctrl_end,
                generator=generator,
            )

        gen = out.images[0]

        # 응답용 base64
        buf_gen = io.BytesIO(); gen.save(buf_gen, format="PNG")
        img_str_generated = base64.b64encode(buf_gen.getvalue()).decode()

        buf_org = io.BytesIO(); ok_img.save(buf_org, format="PNG")
        img_str_original = base64.b64encode(buf_org.getvalue()).decode()

        return JsonResponse({
            "before_image": f"data:image/png;base64,{img_str_original}",
            "after_image":  f"data:image/png;base64,{img_str_generated}",
            "used_checkpoint": {
                "unet_lora": UNET_LORA_DIR,
                "ctrl_lora": CTRL_LORA_DIR,
            },
            "params": {
                "steps": num_inference_steps,
                "cfg": guidance_scale,
                "strength": strength,
                "ctrl_scale": ctrl_scale,
                "ctrl_window": [ctrl_start, ctrl_end],
                "seed": (None if generator is None else int(seed)),
            }
        })

    except Exception as e:
        return JsonResponse({"error": f"이미지 생성 중 오류: {e}"}, status=500)
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
