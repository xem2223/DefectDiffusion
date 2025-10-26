# inference.py
# ------------------------------------------------------------
# - UNet/ControlNet LoRA 풀 병합 → 순수 모듈로 파이프라인 구성
# - Inpaint 모드 지원 (init + cond + mask)
# - mask 생략 시 cond에서 자동 생성(그레이 변환 → 이진화 → feather)
# - ControlNet 세기/구간 튜닝(cscale/cstart/cend)
# ------------------------------------------------------------

import os, re, argparse
from typing import Optional, Tuple
import torch
from PIL import Image, ImageFilter, ImageOps

from diffusers import (
    ControlNetModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
)
from diffusers.models import UNet2DConditionModel

from peft import PeftModel, PeftConfig
try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


# ---------------------- Utils ----------------------

def sanitize_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text[:90]

def load_image_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def resize_square(img: Image.Image, size=512, mode="LANCZOS") -> Image.Image:
    resample = Image.LANCZOS if mode == "LANCZOS" else Image.NEAREST
    return img.resize((size, size), resample)

def align_pair_for_img2img(init_img: Image.Image, cond_img: Image.Image, target: int = 512):
    if cond_img.size != init_img.size:
        cond_img = cond_img.resize(init_img.size, Image.NEAREST)
    init_img = resize_square(init_img, target, "LANCZOS")
    cond_img = resize_square(cond_img, target, "NEAREST")
    return init_img, cond_img

def align_for_text2img(cond_img: Image.Image, target: int = 512):
    if cond_img.size != (target, target):
        cond_img = resize_square(cond_img, target, "NEAREST")
    return cond_img

def auto_mask_from_cond(cond_rgb: Image.Image, size: int = 512,
                        thresh: int = 128, invert: bool = False, feather_px: int = 6) -> Image.Image:
    """
    cond RGB → Gray → Threshold → (optional invert) → Feather
    반환: 'L' (0~255), 흰색=수정, 검정=보존
    """
    m = cond_rgb.convert("L")
    m = resize_square(m, size, "NEAREST")
    # 이진화
    m = m.point(lambda v: 255 if v >= thresh else 0)
    if invert:
        m = ImageOps.invert(m)
    if feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather_px))
    return m


# ---------------------- LoRA 병합 ----------------------

def _load_adapter_state(adapter_dir: str):
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.isfile(st_path) and safe_load_file:
        return safe_load_file(st_path, device="cpu")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"adapter weights not found under {adapter_dir}")

def _find_base_key_like(state_keys, candidate_suffix: str):
    for k in state_keys:
        if k.endswith(candidate_suffix):
            return k
    return None

def _merge_weight(base_w, A, B, alpha, r):
    scale = float(alpha) / float(r) if r else 1.0
    if base_w.ndim == 2:   # Linear
        delta = (B @ A) * scale
        return base_w + delta.to(base_w.dtype)
    elif base_w.ndim == 4: # Conv2d
        out_c, in_c, kH, kW = base_w.shape
        A_flat = A.view(A.size(0), -1)
        B_flat = B.view(B.size(0), -1)
        delta = (B_flat @ A_flat).view(out_c, in_c, kH, kW) * scale
        return base_w + delta.to(base_w.dtype)
    return base_w

def _manual_merge(module, lora_dir: str, name: str):
    cfg = PeftConfig.from_pretrained(lora_dir)
    adapter_state = _load_adapter_state(lora_dir)
    base_sd = module.state_dict()
    merged = 0
    for k in list(adapter_state.keys()):
        if not k.endswith("lora_A.weight"): continue
        prefix = k[:-len("lora_A.weight")]
        kA, kB = f"{prefix}lora_A.weight", f"{prefix}lora_B.weight"
        if kB not in adapter_state: continue
        A, B = adapter_state[kA], adapter_state[kB]
        suffix = k.replace(".lora_A.weight", ".weight")
        # 접두사 차이 흡수
        candidates = [
            suffix.replace("base_model.model.", "").replace("base_model.", "").replace("model.", ""),
            suffix
        ]
        base_key = None
        for cand in candidates:
            hit = _find_base_key_like(base_sd.keys(), cand)
            if hit is not None:
                base_key = hit
                break
        if base_key is None: continue
        try:
            r = int(getattr(cfg, "r", A.shape[0]))
            alpha = int(getattr(cfg, "lora_alpha", r))
            base_sd[base_key] = _merge_weight(base_sd[base_key], A, B, alpha, r)
            merged += 1
        except Exception as e:
            print(f"[warn] {name}: merge failed for {base_key}: {e}")
    module.load_state_dict(base_sd, strict=False)
    print(f"[info] {name}: manually merged {merged} LoRA layers")
    return module

def merge_module(base_module, lora_dir: Optional[str], name: str):
    if not (lora_dir and os.path.isdir(lora_dir)):
        print(f"[warn] {name} LoRA not found: {lora_dir}")
        return base_module
    try:
        _ = PeftConfig.from_pretrained(lora_dir)
        peft_wrapped = PeftModel.from_pretrained(base_module, lora_dir, is_trainable=False)
        if hasattr(peft_wrapped, "merge_and_unload"):
            merged = peft_wrapped.merge_and_unload()
            print(f"[info] {name}: merged via PEFT")
            return merged
    except Exception as e:
        print(f"[warn] {name}: PEFT merge failed ({e}); fallback manual merge")
    return _manual_merge(base_module, lora_dir, name)


# ---------------------- Pipeline Loader ----------------------

def load_pipe_full(sd_id, controlnet_id, unet_lora, ctrl_lora, vae_id, fp16, use_xformers, mode):
    torch_dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32

    base_unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=torch_dtype)
    base_controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch_dtype)

    unet = merge_module(base_unet, unet_lora, "UNet")
    controlnet = merge_module(base_controlnet, ctrl_lora, "ControlNet")

    if mode == "inpaint":
        PipeCls = StableDiffusionControlNetInpaintPipeline
    elif mode == "img2img":
        PipeCls = StableDiffusionControlNetImg2ImgPipeline
    else:
        PipeCls = StableDiffusionControlNetPipeline

    kwargs = dict(unet=unet, controlnet=controlnet, torch_dtype=torch_dtype, safety_checker=None)
    if vae_id:
        vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=torch_dtype)
        pipe = PipeCls.from_pretrained(sd_id, vae=vae, **kwargs)
    else:
        pipe = PipeCls.from_pretrained(sd_id, **kwargs)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if use_xformers:
        try: pipe.enable_xformers_memory_efficient_attention()
        except: pass
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.controlnet.to(memory_format=torch.channels_last)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe


# ---------------------- Main ----------------------

def main():
    p = argparse.ArgumentParser(description="Full-merge + Inpaint, auto mask from cond when mask not provided")
    p.add_argument("--sd_id", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--controlnet_id", default="lllyasviel/sd-controlnet-scribble")
    p.add_argument("--vae_id", default=None)

    p.add_argument("--unet_lora_dir", required=True)
    p.add_argument("--ctrl_lora_dir", required=True)

    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="")

    p.add_argument("--cond", required=True, help="ControlNet condition image (also used to auto-build mask if --mask omitted)")
    p.add_argument("--init", default=None, help="OK image (if set: img2img/inpaint)")

    # mask 생략 시 cond에서 자동 생성
    p.add_argument("--mask", default=None, help="Inpaint mask (white=edit, black=preserve). If omitted, auto from cond.")
    p.add_argument("--mask_thresh", type=int, default=128, help="Threshold for auto mask (0~255)")
    p.add_argument("--mask_invert", action="store_true", help="Invert auto mask after threshold")
    p.add_argument("--feather", type=int, default=6, help="Feather(px) for auto/provided mask")

    # ControlNet 영향/구간
    p.add_argument("--cscale", type=float, default=1.0, help="ControlNet conditioning scale")
    p.add_argument("--cstart", type=float, default=0.0, help="Control guidance start (0~1)")
    p.add_argument("--cend",   type=float, default=0.6, help="Control guidance end (0~1)")

    p.add_argument("--out", default="./samples_infer")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--scale", type=float, default=7.5)
    p.add_argument("--strength", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no_xformers", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 모드 결정
    use_inpaint = bool(args.init)  # init 있으면 inpaint/img2img 중 택
    mode = "inpaint" if use_inpaint else "txt2img"

    pipe = load_pipe_full(args.sd_id, args.controlnet_id, args.unet_lora_dir,
                          args.ctrl_lora_dir, args.vae_id, args.fp16, not args.no_xformers, mode)

    # 이미지 로드
    cond_rgb = load_image_rgb(args.cond)
    if use_inpaint:
        init_rgb = load_image_rgb(args.init)
        init_rgb, cond_rgb = align_pair_for_img2img(init_rgb, cond_rgb, target=512)
    else:
        cond_rgb = align_for_text2img(cond_rgb, target=512)
        init_rgb = None

    # mask 준비: 제공되면 feather, 없으면 cond에서 자동 생성
    if use_inpaint:
        if args.mask:
            mask = Image.open(args.mask).convert("L")
            mask = resize_square(mask, 512, "NEAREST")
            if args.feather > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(args.feather))
        else:
            mask = auto_mask_from_cond(cond_rgb, 512, args.mask_thresh, args.mask_invert, args.feather)
            mask.save(os.path.join(args.out, "__debug_mask_from_cond.png"))
    else:
        mask = None

    # 시드 & 실행
    gen = torch.Generator(device=pipe.device.type).manual_seed(args.seed)
    base_name = sanitize_filename(args.prompt)
    neg = args.negative or None

    for i in range(args.n):
        g = gen.manual_seed(args.seed + i)

        if use_inpaint:
            # Inpaint + ControlNet
            result = StableDiffusionControlNetInpaintPipeline.__call__(
                pipe,
                prompt=args.prompt,
                negative_prompt=neg,
                image=init_rgb,
                mask_image=mask,
                control_image=cond_rgb,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                strength=float(min(max(args.strength, 0.0), 1.0)),
                controlnet_conditioning_scale=args.cscale,
                control_guidance_start=args.cstart,
                control_guidance_end=args.cend,
                generator=g,
            )
        else:
            # text2img + ControlNet
            result = StableDiffusionControlNetPipeline.__call__(
                pipe,
                prompt=args.prompt,
                negative_prompt=neg,
                image=cond_rgb,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                controlnet_conditioning_scale=args.cscale,
                control_guidance_start=args.cstart,
                control_guidance_end=args.cend,
                generator=g,
            )

        out = result.images[0]
        fname = f"{i:02d}_{base_name}_seed{args.seed+i}.png"
        out.save(os.path.join(args.out, fname))
        print("[saved]", fname)


if __name__ == "__main__":
    main()
