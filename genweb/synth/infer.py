import os, torch, numpy as np
from PIL import Image, ImageOps
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from transformers import CLIPTokenizer, CLIPTextModel

# ==== 사용자 환경에 맞게 경로 설정 ====
BASE_SD_ID        = "runwayml/stable-diffusion-v1-5"
BASE_VAE_ID       = "runwayml/stable-diffusion-v1-5"  # same vae
# 병합(merged) 가중치가 있다면 여기에 디렉토리 지정 (옵션)
MERGED_UNET_DIR   = None   # 예: "./checkpoints_merged/unet_merged"
MERGED_CTRL_DIR   = None   # 예: "./checkpoints_merged/controlnet_merged"
# LoRA 디렉토리 사용하는 경우 (옵션) — 병합 모델이 없을 때만 쓰세요.
UNET_LORA_DIR     = None   # 예: "./checkpoints/unet_lora_step28000"
CTRL_LORA_DIR     = None   # 예: "./checkpoints/ctrl_lora_step28000"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dtype  = torch.float16 if _device.type == "cuda" else torch.float32

# 파이프라인 전역 캐시(서버 시작 시 1회 로드)
_pipe = None

def _binarize_to_rgb(mask_pil: Image.Image, size=(512,512)) -> Image.Image:
    """흑/백 마스크를 0/255 이진화 후 RGB로 변환."""
    m = mask_pil.convert("L").resize(size, Image.NEAREST)
    arr = np.array(m)
    thr = 128
    arr = (arr >= thr).astype(np.uint8) * 255
    m_bin = Image.fromarray(arr, mode="L")
    return Image.merge("RGB", (m_bin, m_bin, m_bin))

def _prepare_inpaint_mask(mask_pil: Image.Image, size=(512,512)) -> Image.Image:
    """
    diffusers inpaint의 mask 규칙: 흰색(255) = 덮어쓸 영역(수정), 검정(0) = 보존.
    사용자가 준 마스크가 '결함 위치 흰색'이면 그대로 사용 가능.
    """
    return mask_pil.convert("L").resize(size, Image.NEAREST)

def get_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    # VAE/텍스트 인코더(옵션): 성능 안정용으로 명시
    vae = AutoencoderKL.from_pretrained(BASE_VAE_ID, subfolder="vae", torch_dtype=_dtype)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=_dtype)
    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # ControlNet 로드 (병합 > 기본)
    if MERGED_CTRL_DIR and os.path.isdir(MERGED_CTRL_DIR):
        controlnet = ControlNetModel.from_pretrained(MERGED_CTRL_DIR, torch_dtype=_dtype)
    else:
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=_dtype)

    # 파이프라인
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        BASE_SD_ID,
        controlnet=controlnet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=_dtype,
    ).to(_device)

    # (선택) xformers/SDPA
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # 병합 UNet이 있으면 교체
    if MERGED_UNET_DIR and os.path.isdir(MERGED_UNET_DIR):
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(MERGED_UNET_DIR, torch_dtype=_dtype)
        pipe.unet = unet.to(_device)

    # (옵션) LoRA 적용 — 병합 모델이 없다면 사용
    # from diffusers import AttnProcsLayers
    # if UNET_LORA_DIR:
    #     pipe.unet.load_attn_procs(UNET_LORA_DIR)
    # if CTRL_LORA_DIR:
    #     pipe.controlnet.load_attn_procs(CTRL_LORA_DIR)

    pipe.safety_checker = None
    pipe.enable_vae_slicing()
    _pipe = pipe
    return _pipe

def run_infer(ok_path, mask_path, prompt, out_dir, steps=30, guidance=7.5, strength=0.35, seed=123):
    os.makedirs(out_dir, exist_ok=True)
    pipe = get_pipeline()

    # 입력 로드 & 리사이즈(512)
    init_img = Image.open(ok_path).convert("RGB").resize((512,512), Image.BICUBIC)
    raw_mask = Image.open(mask_path)
    ctrl_img = _binarize_to_rgb(raw_mask, size=(512,512))     # ControlNet cond (RGB)
    inpaint_mask = _prepare_inpaint_mask(raw_mask, (512,512)) # Inpaint mask (L)

    g = torch.Generator(device=_device).manual_seed(int(seed))

    with torch.autocast(device_type="cuda", dtype=_dtype) if _device.type=="cuda" else torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=init_img,              # 원본 보존
            control_image=ctrl_img,      # 결함 마스크 기반 ControlNet 조건
            mask_image=inpaint_mask,     # 마스크 영역만 수정
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            strength=float(strength),    # 낮을수록 원본 보존↑
            generator=g,
        )
    out_img = result.images[0]
    out_path = os.path.join(out_dir, "result.png")
    out_img.save(out_path)
    return out_path
