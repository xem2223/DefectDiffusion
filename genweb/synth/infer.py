# infer.py (최종 버전)
import os, re, torch
from PIL import Image, ImageFilter, ImageOps
from typing import Optional
from django.conf import settings  # ✅ 추가 (Django settings import)
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, AutoencoderKL
from diffusers.models import UNet2DConditionModel
from peft import PeftModel, PeftConfig

def sanitize_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text[:90]

def resize(img: Image.Image, size=512, mode="LANCZOS") -> Image.Image:
    return img.resize((size, size), Image.LANCZOS if mode == "LANCZOS" else Image.NEAREST)

def align_images(init: Image.Image, cond: Image.Image, size=512):
    return resize(init, size), resize(cond, size, "NEAREST")

def auto_mask(cond: Image.Image, size=512, thresh=128, invert=False, feather=6):
    m = cond.convert("L").resize((size, size), Image.NEAREST)
    m = m.point(lambda v: 255 if v >= thresh else 0)
    if invert:
        m = ImageOps.invert(m)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return m

def merge_lora(module, lora_dir: str, name: str):
    try:
        peft_model = PeftModel.from_pretrained(module, lora_dir, is_trainable=False)
        return peft_model.merge_and_unload()
    except Exception as e:
        print(f"[warn] {name} LoRA merge failed: {e}")
        return module  # fallback

def load_pipeline(unet_lora, ctrl_lora, fp16=True):
    dtype = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
    unet = merge_lora(
        UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=dtype),
        unet_lora, "UNet"
    )
    controlnet = merge_lora(
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=dtype),
        ctrl_lora, "ControlNet"
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=unet,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        vae=AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=dtype)
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def run_infer(
    ok_img: Image.Image,
    mask_img: Optional[Image.Image],
    cond_img: Image.Image,
    prompt: str,
    unet_lora_dir: str,
    ctrl_lora_dir: str,
    seed: int = 42,
) -> str:
    # ✅ MEDIA_ROOT 하위에 저장
    save_dir = os.path.join(settings.MEDIA_ROOT, "results")
    os.makedirs(save_dir, exist_ok=True)

    # 전처리
    init, cond = align_images(ok_img, cond_img)
    mask = resize(mask_img.convert("L"), 512, "NEAREST") if mask_img else auto_mask(cond)

    # 파이프라인
    pipe = load_pipeline(unet_lora=unet_lora_dir, ctrl_lora=ctrl_lora_dir)
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    # 추론
    result = pipe(
        prompt=prompt,
        image=init,
        mask_image=mask,
        control_image=cond,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.35,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=0.6,
        generator=generator
    )

    # 저장 경로
    base = sanitize_filename(prompt)
    filename = f"{base}_seed{seed}.png"
    abs_path = os.path.join(save_dir, filename)       # 실제 저장 경로
    rel_url = settings.MEDIA_URL + "results/" + filename  # 브라우저 접근 URL

    result.images[0].save(abs_path)
    return rel_url
