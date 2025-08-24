#app.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import io
import base64
import os
import tempfile
import shutil
from PIL import Image
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from peft import PeftModel

"""
체크리스트
./models/checkpoints/unet_lora와 ./models/checkpoints/ctrl_lora 폴더 안에 
adapter_model.safetensors(또는 .bin)와 **adapter_config.json**이 있어야 함

((결과가 너무 강하면 strength를 0.30~0.45 범위로, 약하면 0.5 이상으로 올려보기))
"""

# --- 실제 모델 로딩 및 추론 코드 ---
pipe = None
try:
    print("------------------------------------------------------------------")
    print("1/5. ControlNet 모델 파일 로딩 (자동 다운로드)...")
    print("------------------------------------------------------------------")
    controlnet_model = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    print("2/5. Img2Img 파이프라인 로딩 (SD 1.5 + ControlNet)...")
    print("------------------------------------------------------------------")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device_str)

    # 2-1) SDPA로 고정 (xFormers 미사용) — dtype mismatch 방지
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.controlnet.set_attn_processor(AttnProcessor2_0())

    print("3/5. LoRA 어댑터 모델 로딩 및 병합...")
    # 폴더 구조: ./models/checkpoints/{unet_lora, ctrl_lora}
    base_dir = os.path.dirname(__file__)
    ckpt_root = os.path.abspath(os.path.join(base_dir, "..", "models", "checkpoints"))
    unet_lora_dir = os.path.join(ckpt_root, "unet_lora")
    ctrl_lora_dir = os.path.join(ckpt_root, "ctrl_lora")

    if not (os.path.isdir(unet_lora_dir) and os.path.isdir(ctrl_lora_dir)):
        raise FileNotFoundError(f"LoRA 폴더가 없습니다: {unet_lora_dir} / {ctrl_lora_dir}")

    # UNet / ControlNet 각각 LoRA 로드 → 병합
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_dir)
    pipe.controlnet = PeftModel.from_pretrained(pipe.controlnet, ctrl_lora_dir)

    pipe.unet = pipe.unet.merge_and_unload()
    pipe.controlnet = pipe.controlnet.merge_and_unload()

    # (선택) VRAM이 빡빡할 때만 켜세요. 속도는 조금 느려집니다.
    # if torch.cuda.is_available():
    #     pipe.enable_attention_slicing()
    #     pipe.enable_model_cpu_offload()

    print("5/5. 서버 준비 완료. 모델이 성공적으로 로딩되었습니다.")
    print("------------------------------------------------------------------")

except Exception as e:
    print("------------------------------------------------------------------")
    print(f"모델 로딩 중 치명적인 오류 발생: {e}")
    print("서버 시작이 지연되거나 실패할 수 있습니다.")
    print("------------------------------------------------------------------")
    pipe = None


def index(request):
    return render(request, "pybo/index.html")


@csrf_exempt
def generate_image(request):
    print(f"디버그: 'generate_image' 함수가 받은 요청 메서드: {request.method}")
    if request.method == "POST" and pipe is not None:
        temp_dir = None
        try:
            manufacturing_image_file = request.FILES.get("manufacturing-image")  # OK 이미지
            mask_image_file = request.FILES.get("mask-image")                    # 결함 마스크
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

            # --- inference.py와 동일하게: PIL 그대로 사용 ---
            ok_img = Image.open(manuf_path).convert("RGB").resize((512, 512), Image.BICUBIC)
            # 마스크는 L 또는 RGB 모두 가능. 경계 보존에는 L+NEAREST가 유리
            mask_img = Image.open(mask_path).convert("L").resize((512, 512), Image.NEAREST)

            # 하이퍼파라미터 (필요시 폼에서 전달 가능)
            num_inference_steps = int(request.POST.get("steps", 30))
            guidance_scale = float(request.POST.get("cfg", 7.5))
            strength = float(request.POST.get("strength", 0.35))
            ctrl_scale = float(request.POST.get("ctrl_scale", 1.0))
            ctrl_start = float(request.POST.get("ctrl_start", 0.0))
            ctrl_end   = float(request.POST.get("ctrl_end",   0.6))
            seed = request.POST.get("seed", None)

            generator = None
            if seed is not None:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device_str).manual_seed(int(seed))

            # 추론 (img2img + ControlNet)
            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"),
                                dtype=torch.float16, enabled=torch.cuda.is_available()):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=ok_img,                                   # ← init (OK)
                    controlnet_conditioning_image=mask_img,         # ← ControlNet cond(마스크)
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=ctrl_scale,
                    control_guidance_start=ctrl_start,
                    control_guidance_end=ctrl_end,
                    generator=generator,
                )

            generated_image = result.images[0]

            # 응답용 base64
            buf_gen = io.BytesIO()
            generated_image.save(buf_gen, format="PNG")
            img_str_generated = base64.b64encode(buf_gen.getvalue()).decode()

            buf_org = io.BytesIO()
            ok_img.save(buf_org, format="PNG")
            img_str_original = base64.b64encode(buf_org.getvalue()).decode()

            return JsonResponse({
                "before_image": f"data:image/png;base64,{img_str_original}",
                "after_image": f"data:image/png;base64,{img_str_generated}",
                "params": {
                    "steps": num_inference_steps,
                    "cfg": guidance_scale,
                    "strength": strength,
                    "ctrl_scale": ctrl_scale,
                    "ctrl_window": [ctrl_start, ctrl_end],
                    "seed": (None if generator is None else int(seed)),
                },
            })

        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {e}")
            return JsonResponse({"error": str(e)}, status=500)
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    print(f"디버그: 요청 메서드가 POST가 아니거나 파이프가 초기화되지 않았습니다. 현재 메서드: {request.method}")
    return JsonResponse({"error": "POST 요청만 허용됩니다."}, status=405)
