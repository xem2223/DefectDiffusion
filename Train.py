# Train.py  ──────────────────────────────────────────
import os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit
from diffusers import (
    ControlNetModel, StableDiffusionControlNetPipeline,
    DDPMScheduler, AutoencoderKL,
)
from Dataset import DefectSynthesisDataset
from tqdm import tqdm

# ───────────── CONFIG ─────────────
DATA_ROOT      = "/opt/dlami/nvme"
BATCH_SIZE     = 2
EPOCHS         = 2
LR             = 1e-4            # ControlNet(LoRA) 학습률
LR_UNET_LORA   = 5e-5            # UNet(LoRA) 학습률 (조금 낮게 권장)
UNET_LORA_RANK = 8               # UNet LoRA rank (4~16 권장)
VAE_ID         = "runwayml/stable-diffusion-v1-5"
SD_ID          = "runwayml/stable-diffusion-v1-5"
CONTROLNET_ID  = "lllyasviel/sd-controlnet-scribble"
OUTPUT_DIR     = "./checkpoints"
SAVE_STEPS     = 2000
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── DATASET ─────────────
dataset = DefectSynthesisDataset(
    DATA_ROOT, cache_in_ram=False,
)
idx2class  = {v: k for k, v in dataset.class2idx.items()}
idx2defect = {v: k for k, v in dataset.defect2idx.items()}

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=False, prefetch_factor=2,
                    persistent_workers=True, timeout=120)

batch0 = next(iter(loader))
print("Batch keys:", list(batch0.keys()))

# ───────────── CUDA 튜닝 ─────────────
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ───────────── MODELS ─────────────
vae = AutoencoderKL.from_pretrained(
    VAE_ID, subfolder="vae", torch_dtype=torch.float16
).to(device).eval()
for p in vae.parameters(): p.requires_grad = False

tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=torch.float16
).to(device).eval()
for p in text_encoder.parameters(): p.requires_grad = False

def find_lora_targets(model):
    leaf_types = (torch.nn.Linear,
                  torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    keep = set()
    for full_name, module in model.named_modules():
        if isinstance(module, leaf_types):
            last = full_name.split(".")[-1]
            if not last.isdigit():
                keep.add(last)
    return sorted(keep)

# 1) ControlNet (PEFT-LoRA)
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_ID, torch_dtype=torch.float16
)
target_mods_ctrl = find_lora_targets(controlnet)
print("ControlNet LoRA 대상(샘플):", target_mods_ctrl[:20], "...")

lora_cfg_ctrl = LoraConfig(r=4, lora_alpha=16, bias="none", target_modules=target_mods_ctrl)
controlnet = get_peft_model(controlnet, lora_cfg_ctrl).to(device)
controlnet.print_trainable_parameters()
controlnet.train()

# 2) 파이프라인 구성
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    SD_ID, controlnet=controlnet, vae=vae,
    text_encoder=text_encoder, tokenizer=tokenizer,
    torch_dtype=torch.float16,
).to(device)

# 메모리 최적화 옵션
pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_attention_slicing()  # 메모리 부족할 때만
pipe.unet.to(memory_format=torch.channels_last)
pipe.controlnet.enable_gradient_checkpointing()
# 선택: UNet도 켤 수 있음(메모리 절약, 속도 약간↓)
# try:
#     pipe.unet.enable_gradient_checkpointing()
# except Exception:
#     pass

# 3) require_grad 기본 설정
for p in pipe.unet.parameters():         p.requires_grad = False
for p in pipe.vae.parameters():          p.requires_grad = False
for p in pipe.text_encoder.parameters(): p.requires_grad = False
for p in pipe.controlnet.parameters():   p.requires_grad = True

# 4) UNet에 PEFT-LoRA 주입 (Diffusers 0.26+ 권장: to_q/to_k/to_v/to_out.0)
#    - 전체 UNet 어텐션 프로젝션층에 LoRA를 삽입 후
#    - down_blocks 하위의 LoRA 파라미터는 requires_grad=False로 꺼서 mid/up만 학습
lora_cfg_unet = LoraConfig(
    r=UNET_LORA_RANK,
    lora_alpha=UNET_LORA_RANK,   # 보통 r와 동일 혹은 2r
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    bias="none",
)
pipe.unet = get_peft_model(pipe.unet, lora_cfg_unet)

# down_blocks의 LoRA 파라미터만 동결 (mid/up 블록 학습 집중)
for name, param in pipe.unet.named_parameters():
    if "lora_" in name and name.startswith("down_blocks"):
        param.requires_grad = False

# (디버그) 학습 파라미터 개수 확인
tn_unet = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
tn_ctrl = sum(p.numel() for p in pipe.controlnet.parameters() if p.requires_grad)
print(f"Trainable params -> UNet(LoRA): {tn_unet:,} | ControlNet(LoRA): {tn_ctrl:,}")

pipe.unet.train()  # LoRA 부분 학습 모드

noise_scheduler = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")

# ───────────── 텍스트 임베딩 캐시(대폭 단축) ─────────────
with torch.no_grad():
    prompt_cache = {}
    MAX_LEN = tokenizer.model_max_length  # 보통 77

    for c_id, c_name in idx2class.items():
        for d_id, d_name in idx2defect.items():
            ptxt = f"{c_name} {d_name} defect"
            enc = tokenizer(
                [ptxt],
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = text_encoder(**enc).last_hidden_state  # [1, MAX_LEN, hidden]
            prompt_cache[(c_id, d_id)] = emb

def get_text_embeds(class_ids, defect_ids):
    embs = [prompt_cache[(int(c.item()), int(d.item()))] for c, d in zip(class_ids, defect_ids)]
    return torch.cat(embs, dim=0)  # (B,77,768)

# ───────────── 마스크 전처리(1→3채널, 512, [-1,1]) ─────────────
def prep_mask(mask_bf16):
    # mask_bf16: (B,1,H,W) or (B,C,H,W)
    if mask_bf16.dim() == 3:
        mask_bf16 = mask_bf16.unsqueeze(1)
    if mask_bf16.size(1) == 1:
        mask_bf16 = mask_bf16.repeat(1, 3, 1, 1)
    mask_bf16 = torch.nn.functional.interpolate(mask_bf16, size=(512, 512), mode="nearest")
    mask_bf16 = mask_bf16.clamp(-1, 1)
    # mask_bf16 = mask_bf16.clamp(0,1) * 2 - 1  # 필요 시
    return mask_bf16

# ───────────── OPTIMIZER ─────────────
params_controlnet = [p for p in pipe.controlnet.parameters() if p.requires_grad]
params_unet_lora  = [p for p in pipe.unet.parameters()       if p.requires_grad]

optimizer = AdamW8bit(
    [
        {"params": params_controlnet, "lr": LR},
        {"params": params_unet_lora,  "lr": LR_UNET_LORA},
    ],
    betas=(0.9, 0.999),
    weight_decay=1e-4
)

# ───────────── TRAIN LOOP (latent/cond 전용) ─────────────
global_step = 0
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        # ① 배치 언팩 (이미 latent/cond가 준비됨)
        lat_in   = batch["ok_lat"].to(device, torch.float16, non_blocking=True)   # [B,4,64,64]
        lat_tgt  = batch["ng_lat"].to(device, torch.float16, non_blocking=True)   # [B,4,64,64]
        cond     = batch["cond"].to(device,   torch.float16, non_blocking=True)   # [B,3,512,512] in [-1,1]
        class_ids  = batch["class_id"]
        defect_ids = batch["defect_id"]
        bsz        = lat_in.size(0)

        # ② 노이즈 & 타임스텝
        noise     = torch.randn_like(lat_in)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (bsz,), device=device)
        noisy_lat = noise_scheduler.add_noise(lat_in, noise, timesteps)

        # ③ 텍스트 임베딩(캐시)
        text_emb = get_text_embeds(class_ids, defect_ids)

        # ④ forward → ControlNet + UNet
        optimizer.zero_grad(set_to_none=True)

        ctrl_out = pipe.controlnet(
            sample=noisy_lat,
            timestep=timesteps,
            encoder_hidden_states=text_emb,
            controlnet_cond=cond,           # 데이터셋에서 이미 3x512x512 [-1,1]
            return_dict=True,
        )
        lat_pred = pipe.unet(
            sample=noisy_lat,
            timestep=timesteps,
            encoder_hidden_states=text_emb,
            down_block_additional_residuals=ctrl_out.down_block_res_samples,
            mid_block_additional_residual=ctrl_out.mid_block_res_sample,
            return_dict=True,
        ).sample

        # ⑤ 손실 & 역전파 (FP32로 계산)
        loss = F.mse_loss(lat_pred.float(), lat_tgt.float())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(params_controlnet + params_unet_lora, 1.0)
        optimizer.step()

        # ⑥ 로그 & 체크포인트
        total_loss  += loss.item()
        global_step += 1
        if global_step % SAVE_STEPS == 0:
            # ControlNet(PEFT LoRA) 저장
            controlnet.save_pretrained(f"{OUTPUT_DIR}/ctrl_lora_step{global_step}")
            # UNet(PEFT LoRA) 저장
            pipe.unet.save_pretrained(f"{OUTPUT_DIR}/unet_lora_step{global_step}")

    avg = total_loss / len(loader)
    print(f"Epoch {epoch} finished | avg loss: {avg:.4f}")

    # 에폭 단위 스냅샷
    controlnet.save_pretrained(f"{OUTPUT_DIR}/ctrl_lora_epoch{epoch}")
    pipe.unet.save_pretrained(f"{OUTPUT_DIR}/unet_lora_epoch{epoch}")

