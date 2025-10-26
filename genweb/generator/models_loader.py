import os, re, argparse
from typing import Optional, Tuple, List
import torch
import lpips
import torchvision.transforms as transforms 
from PIL import Image, ImageFilter, ImageOps
from django.conf import settings # settings.MEDIA_ROOT 사용을 위해 추가

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

# load_image_rgb 함수는 views.py가 Image 객체를 직접 전달하므로 필요 없습니다.

def resize_square(img: Image.Image, size=512, mode="LANCZOS") -> Image.Image:
    resample = Image.LANCZOS if mode == "LANCZOS" else Image.NEAREST
    return img.resize((size, size), resample)

def align_pair_for_img2img(init_img: Image.Image, cond_img: Image.Image, target: int = 512):
    if cond_img.size != init_img.size:
        cond_img = cond_img.resize(init_img.size, Image.NEAREST)
    init_img = resize_square(init_img, target, "LANCZOS")
    cond_img = resize_square(cond_img, target, "NEAREST")
    return init_img, cond_img

# align_for_text2img 함수는 views.py가 Inpaint 모드만 사용하므로 필요 없습니다.

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

# ---------------------LPIPS Metric---------------------
LPIPS_MODEL = None
LPIPS_DEVICE = None

def get_lpips_model():
    """LPIPS 모델을 로드하고 캐시합니다."""
    global LPIPS_MODEL, LPIPS_DEVICE
    if LPIPS_MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LPIPS_DEVICE = device
        print(f"[INFO] LPIPS Model Loading onto {LPIPS_DEVICE}...")
        LPIPS_MODEL = lpips.LPIPS(net='alex').to(device)
        LPIPS_MODEL.eval() 
    return LPIPS_MODEL

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL 이미지를 LPIPS 모델 입력([-1, 1] 범위)에 적합한 텐서로 변환합니다."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [0, 1] -> [-1, 1]
    ])
    return transform(image).unsqueeze(0)


# ---------------------- Metric Calculation (LPIPS 전용) ----------------------
def calculate_metrics(img1: Image.Image, img2: Image.Image) -> dict:
    """
    두 PIL Image를 비교하여 LPIPS만 계산합니다.
    """
    target_size = 512
    
    # 1. 전처리: 크기 통일 및 RGB 변환
    img1 = img1.resize((target_size, target_size), Image.LANCZOS).convert("RGB")
    img2 = img2.resize((target_size, target_size), Image.LANCZOS).convert("RGB")
    
    lpips_value = 0.0
    try:
        lpips_fn = get_lpips_model()
        device = LPIPS_DEVICE 
        
        # 텐서로 변환 후 디바이스로 이동
        img1_tensor = pil_to_tensor(img1).to(device)
        img2_tensor = pil_to_tensor(img2).to(device)
        
        # LPIPS 거리 계산 (0에 가까울수록 유사)
        with torch.no_grad():
            lpips_score = lpips_fn(img1_tensor, img2_tensor)
            lpips_value = lpips_score.item()
            
    except Exception as e:
        print(f"LPIPS 계산 중 오류 발생: {e}")
        lpips_value = "ERROR"
    
    return {
        "lpips": round(lpips_value, 4) if isinstance(lpips_value, float) else lpips_value
    }
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
    if base_w.ndim == 2:    # Linear
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

# ---------------------- GLOBAL MODEL LOAD (서버 시작 시 한 번만 로드) ----------------------
try:
    _SD_ID = "runwayml/stable-diffusion-v1-5"
    _CTRL_ID = "lllyasviel/sd-controlnet-scribble"
    # NOTE: LORA 경로는 views.py에서 전달되지만, 여기서는 모델 로딩에 필요하므로 기본 경로를 지정합니다.
    _UNET_LORA_DIR = "./checkpoints/unet_lora" 
    _CTRL_LORA_DIR = "./checkpoints/ctrl_lora"
    _VAE_ID = None
    _FP16 = True
    _USE_XFORMERS = True
    _MODE = "inpaint" 

    print("[INFO] Starting Global Model Load...")
    # NOTE: Gunicorn 실행 시 이 코드가 실행됩니다.
    GLOBAL_PIPE = load_pipe_full(
        _SD_ID, _CTRL_ID, _UNET_LORA_DIR, _CTRL_LORA_DIR, 
        _VAE_ID, _FP16, _USE_XFORMERS, _MODE
    )
    print("[INFO] Global Model Load Complete.")
except Exception as e:
    # GPU 미설치 또는 파일 오류 시 fatal 에러 발생
    print(f"[FATAL] Failed to load GLOBAL_PIPE: {e}") 
    GLOBAL_PIPE = None


# ---------------------- API 래퍼 함수 (views.py와 인자 일치) ----------------------

def run_infer(
    ok_img: Image.Image,
    cond_img: Image.Image,
    prompt: str,
    # views.py에서 전달되는 하드코딩된 경로 인자를 그대로 받습니다.
    unet_lora_dir: str, 
    ctrl_lora_dir: str, 
    *, # 키워드 인자만 받습니다.
    mask_img: Optional[Image.Image] = None, 
    steps: int = 30, 
    scale: float = 7.5, 
    strength: float = 0.3,
    cscale: float = 3.0, 
    cstart: float = 0.0, 
    cend: float = 1.0,
    mask_thresh: int = 128, 
    mask_feather: int = 0,
    mask_expand_px: int = 0, # views.py에서 전달됨
    seeds: Optional[List[int]] = None,
    primary_seed: Optional[int] = None,
    debug: bool = False,
    fp16: bool = True,
) -> str:
    """
    장고 뷰에서 호출되는 추론 함수. PIL Image 객체를 받습니다.
    결과 이미지의 로컬 경로를 반환합니다.
    """
    if GLOBAL_PIPE is None:
        raise RuntimeError("AI model not loaded. Check GPU or model files.")
        
    pipe = GLOBAL_PIPE
    
    # 아웃풋 경로 설정 (views.py가 결과를 'results'로 복사하므로, 'samples_infer'에 저장)
    out_dir_rel = "samples_infer"
    out_dir_abs = os.path.join(settings.MEDIA_ROOT, out_dir_rel)
    os.makedirs(out_dir_abs, exist_ok=True)
    
    # Args 객체에 파라미터 설정 (디버그 및 내부 변수 전달용)
    class Args: pass
    # views.py에서 negative 인자는 전달되지 않으므로 빈 문자열로 가정합니다.
    args = Args()
    args.prompt, args.negative, args.steps, args.scale, args.strength, \
    args.cscale, args.cstart, args.cend, args.seed = \
    prompt, "", steps, scale, strength, cscale, cstart, cend, primary_seed
    
    # ------------------ 추론 로직 (본문) ------------------
    
    # 이미지 로드 및 정렬 (views.py가 전달한 객체를 바로 사용)
    # views.py 호출에 따라 use_inpaint는 True로 가정합니다.
    init_rgb, cond_rgb_aligned = align_pair_for_img2img(ok_img, cond_img, target=512)

    # 마스크 준비
    mask = None
    if mask_img:
        # 1. 제공된 마스크 이미지 처리
        mi = ImageOps.exif_transpose(mask_img).convert("L")
        if cond_img and mi.size != ok_img.size:
             mi = mi.resize(ok_img.size, Image.NEAREST)
        mask = resize_square(mi, 512, "NEAREST")
        if mask_feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(mask_feather))
    else:
        # 2. 마스크 생략 시 cond 이미지에서 자동 생성
        mask = auto_mask_from_cond(cond_rgb_aligned, 512, mask_thresh, False, mask_feather)
    
    # 3. 마스크 확장 (views.py 인자에 맞춤)
    if mask_expand_px > 0 and mask:
        # 간단한 가우시안 블러를 이용한 확장 구현
        b = mask.filter(ImageFilter.GaussianBlur(mask_expand_px))
        mask = b.point(lambda v: 255 if v >= 64 else 0)

    # ------------------ 디버그 이미지 저장 (views.py 인자에 맞춤) ------------------
    if debug:
        base = sanitize_filename(prompt)
        init_rgb.save(os.path.join(out_dir_abs, f"__debug_init.png"))
        cond_rgb_aligned.save(os.path.join(out_dir_abs, f"__debug_cond.png"))
        if mask:
            mask.save(os.path.join(out_dir_abs, f"__debug_mask.png"))
    # -----------------------------------------------------------------------------------

    # 시드 & 실행
    seeds = seeds if seeds is not None else [7]
    # views.py가 primary_seed를 전달하므로 이를 사용합니다.
    if primary_seed and primary_seed not in seeds:
        seeds.append(primary_seed)

    base_name = sanitize_filename(prompt)
    result_path = ""
    
    for s in seeds:
        gen = torch.Generator(device=pipe.device.type).manual_seed(s)
        
        # 추론 파이프라인 호출
        result = StableDiffusionControlNetInpaintPipeline.__call__(
            pipe, 
            prompt=args.prompt, 
            negative_prompt=args.negative, 
            image=init_rgb, 
            mask_image=mask,
            control_image=cond_rgb_aligned, 
            num_inference_steps=args.steps, 
            guidance_scale=args.scale,
            strength=float(min(max(args.strength, 0.0), 1.0)), 
            controlnet_conditioning_scale=args.cscale,
            control_guidance_start=args.cstart, 
            control_guidance_end=args.cend, 
            generator=gen,
        )

        out = result.images[0]
        fname = f"{base_name}_seed{s}.png"
        save_path = os.path.join(out_dir_abs, fname)
        out.save(save_path)
        print(f"[\u2713] saved: {save_path}")

        if s == primary_seed or primary_seed is None:
            # primary_seed 결과가 최종 반환 경로가 됩니다.
            result_path = save_path

    return result_path


# ---------------------- Main (기존 CLI 실행을 위해 유지) ----------------------
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

    p.add_argument("--mask", default=None, help="Inpaint mask (white=edit, black=preserve). If omitted, auto from cond.")
    p.add_argument("--mask_thresh", type=int, default=128, help="Threshold for auto mask (0~255)")
    p.add_argument("--mask_invert", action="store_true", help="Invert auto mask after threshold")
    p.add_argument("--feather", type=int, default=6, help="Feather(px) for auto/provided mask")

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

    use_inpaint = bool(args.init)
    mode = "inpaint" if use_inpaint else "txt2img"

    # main 함수에서는 모델을 새로 로드합니다.
    pipe = load_pipe_full(args.sd_id, args.controlnet_id, args.unet_lora_dir,
                          args.ctrl_lora_dir, args.vae_id, args.fp16, not args.no_xformers, mode)

    # CLI는 파일 경로를 받으므로 load_image_rgb를 다시 사용해야 합니다.
    cond_rgb = Image.open(args.cond).convert("RGB") # load_image_rgb가 없으므로 대체
    if use_inpaint:
        init_rgb = Image.open(args.init).convert("RGB") # load_image_rgb가 없으므로 대체
        init_rgb, cond_rgb = align_pair_for_img2img(init_rgb, cond_rgb, target=512)
    else:
        # load_image_rgb가 없으므로 Image.open()만 사용
        cond_rgb = align_for_text2img(cond_rgb, target=512)
        init_rgb = None

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

    gen = torch.Generator(device=pipe.device.type).manual_seed(args.seed)
    base_name = sanitize_filename(args.prompt)
    neg = args.negative or None

    for i in range(args.n):
        g = gen.manual_seed(args.seed + i)

        if use_inpaint:
            result = StableDiffusionControlNetInpaintPipeline.__call__(
                pipe, prompt=args.prompt, negative_prompt=neg, image=init_rgb, mask_image=mask,
                control_image=cond_rgb, num_inference_steps=args.steps, guidance_scale=args.scale,
                strength=float(min(max(args.strength, 0.0), 1.0)), controlnet_conditioning_scale=args.cscale,
                control_guidance_start=args.cstart, control_guidance_end=args.cend, generator=g,
            )
        else:
            result = StableDiffusionControlNetPipeline.__call__(
                pipe, prompt=args.prompt, negative_prompt=neg, image=cond_rgb, num_inference_steps=args.steps,
                guidance_scale=args.scale, controlnet_conditioning_scale=args.cscale, control_guidance_start=args.cstart,
                control_guidance_end=args.cend, generator=g,
            )

        out = result.images[0]
        fname = f"{i:02d}_{base_name}_seed{args.seed+i}.png"
        out.save(os.path.join(args.out, fname))
        print("[saved]", fname)


if __name__ == "__main__":
    main()
