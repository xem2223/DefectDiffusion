# Train_2.py
"""
학습된 모델로 추론을 했을 때, 결함이 제대로 반영되지 않은 이유 : 
K latent(lat_in)에 노이즈를 얹고 NG latent 자체를 회귀하던 게 문제

변경 사항
NG latent(lat_tgt)에 노이즈를 얹고, 붙인 노이즈(ε)를 예측하도록 바꾸기

SD v1 계열의 표준은 ε-예측(또는 v-예측)
NG latent에 노이즈를 얹고 그 노이즈 자체(또는 velocity)를 맞추게 해야,
ControlNet/LoRA가 “이 cond/프롬프트일 때 어떤 결함 패턴을 주입할지”를 정확히 학습 가능

즉, NG Latent에 노이즈를 더하고, 붙인 노이즈(ε) 또는 v를 예측하도록 교정한 전체 학습 코드

- prediction_type(epsilon/v)에 자동 대응
- 선택: SNR 가중 로스(γ=5 기본, 끄려면 USE_SNR_WEIGHT=False)
- ControlNet/UNet 모두 PEFT-LoRA 학습 (UNet은 mid/up 중심, down 일부 동결 예시 포함)

변경 핵심 요약
노이즈 주입 대상: lat_ok → ❌ / lat_ng → ✅
손실 타깃: ng_lat 자체 회귀 → ❌ / 붙인 노이즈(ε) 또는 velocity) → ✅
옵션: SNR 가중 로스로 timestep 균형(저/고주파) 개선
표현력 강화: UNet LoRA rank ↑, down 일부만 동결(필요시 더 풀 수 있음)
LR: UNet LoRA 1e-4로 상향(경험상 결함 반영력↑), 1k 워밍업

"""
mport os, math, random
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit
from diffusers import (
    ControlNetModel, StableDiffusionControlNetPipeline,
    DDPMScheduler, AutoencoderKL,
)
from tqdm import tqdm
from Dataset2 import DefectSynthesisDataset

# ───────────── CONFIG ─────────────
DATA_ROOT      = "/opt/dlami/nvme"
BATCH_SIZE     = 2
EPOCHS         = 2
LR_CTRL        = 1e-4           # ControlNet(LoRA) 학습률
LR_UNET_LORA   = 1e-4           # UNet(LoRA) 학습률 (기존 5e-5 → 1e-4 권장)
UNET_LORA_RANK = 16             # UNet LoRA rank (표현력 ↑)
VAE_ID         = "runwayml/stable-diffusion-v1-5"
SD_ID          = "runwayml/stable-diffusion-v1-5"
CONTROLNET_ID  = "lllyasviel/sd-controlnet-scribble"
OUTPUT_DIR     = "./checkpoints_rework"
SAVE_STEPS     = 1000
USE_SNR_WEIGHT = True
SNR_GAMMA      = 5.0
GRAD_CLIP_NORM = 1.0
SEED           = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── SEED ─────────────
def seed_everything(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything()

# ───────────── DATASET ─────────────
dataset = DefectSynthesisDataset(DATA_ROOT, cache_in_ram=False)
idx2class  = {v: k for k, v in dataset.class2idx.items()}
idx2defect = {v: k for k, v in dataset.defect2idx.items()}

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=False, prefetch_factor=2,
    persistent_workers=True, timeout=120
)

# 디버그 프린트
try:
    batch0 = next(iter(loader))
    print("Batch keys:", list(batch0.keys()))
except Exception as e:
    print("[warn] loader warmup failed:", e)

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

lora_cfg_ctrl = LoraConfig(r=8, lora_alpha=16, bias="none",
                           target_modules=target_mods_ctrl)
controlnet = get_peft_model(controlnet, lora_cfg_ctrl).to(device)
controlnet.print_trainable_parameters()
controlnet.train()

# 2) 파이프라인 구성
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    SD_ID, controlnet=controlnet, vae=vae,
    text_encoder=text_encoder, tokenizer=tokenizer,
    torch_dtype=torch.float16,
).to(device)

# 메모리 최적화
try: pipe.enable_xformers_memory_efficient_attention()
except Exception: pass
pipe.unet.to(memory_format=torch.channels_last)
pipe.controlnet.enable_gradient_checkpointing()
# 필요 시: pipe.unet.enable_gradient_checkpointing()

# 3) 기본 requires_grad
for p in pipe.unet.parameters():         p.requires_grad = False
for p in pipe.vae.parameters():          p.requires_grad = False
for p in pipe.text_encoder.parameters(): p.requires_grad = False
for p in pipe.controlnet.parameters():   p.requires_grad = True

# 4) UNet에 PEFT-LoRA 주입 (어텐션 투영층)
lora_cfg_unet = LoraConfig(
    r=UNET_LORA_RANK,
    lora_alpha=UNET_LORA_RANK,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    bias="none",
)
pipe.unet = get_peft_model(pipe.unet, lora_cfg_unet)

# down 일부 동결(예: 초기 저주파 특성 과도학습 방지) — 필요 시 완화
for name, param in pipe.unet.named_parameters():
    if "lora_" in name and name.startswith("down_blocks"):
        param.requires_grad = False

tn_unet = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
tn_ctrl = sum(p.numel() for p in pipe.controlnet.parameters() if p.requires_grad)
print(f"Trainable params -> UNet(LoRA): {tn_unet:,} | ControlNet(LoRA): {tn_ctrl:,}")
pipe.unet.train()

# 스케줄러
noise_scheduler = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
print("Scheduler prediction_type:", pred_type)

# ───────────── 텍스트 임베딩 캐시 ─────────────
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

# ───────────── SNR 가중 (선택) ─────────────
if USE_SNR_WEIGHT:
    # alphas_cumprod: [num_train_timesteps]
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)  # float64일 수 있음
    alphas_cumprod = alphas_cumprod.float()
    def get_snr(timesteps):
        # SNR = alpha_t^2 / sigma_t^2 = α / (1-α) (여기서 α=alphas_cumprod)
        a = alphas_cumprod[timesteps]  # (B,)
        snr = a / (1.0 - a).clamp(min=1e-8)
        return snr
    print("SNR weighting enabled (gamma=%.1f)" % SNR_GAMMA)

# ───────────── OPTIMIZER ─────────────
params_controlnet = [p for p in pipe.controlnet.parameters() if p.requires_grad]
params_unet_lora  = [p for p in pipe.unet.parameters()       if p.requires_grad]

optimizer = AdamW8bit(
    [
        {"params": params_controlnet, "lr": LR_CTRL},
        {"params": params_unet_lora,  "lr": LR_UNET_LORA},
    ],
    betas=(0.9, 0.999),
    weight_decay=1e-4
)

# (선택) 간단한 워밍업 스케줄러
warmup_steps = 1000
def lr_lambda(step):
    return min(1.0, step / max(1, warmup_steps))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ───────────── TRAIN LOOP ─────────────
global_step = 0
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        # ① 배치 언팩
        lat_ok   = batch["ok_lat"].to(device, torch.float16, non_blocking=True)   # [B,4,64,64]
        lat_ng   = batch["ng_lat"].to(device, torch.float16, non_blocking=True)   # [B,4,64,64]
        cond     = batch["cond"].to(device,   torch.float16, non_blocking=True)   # [B,3,512,512] in [-1,1]
        class_ids  = batch["class_id"]
        defect_ids = batch["defect_id"]
        bsz        = lat_ok.size(0)

        # ② 노이즈 & 타임스텝  (★ 교정: NG 기준, ε/velocity 예측)
        target_lat = lat_ng
        noise      = torch.randn_like(target_lat)
        timesteps  = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                   (bsz,), device=device)
        noisy_lat  = noise_scheduler.add_noise(target_lat, noise, timesteps)

        # ③ 텍스트 임베딩(캐시)
        with torch.no_grad():
            text_emb = get_text_embeds(class_ids, defect_ids)  # (B,77,768)

        # ④ forward → ControlNet + UNet
        optimizer.zero_grad(set_to_none=True)

        ctrl_out = pipe.controlnet(
            sample=noisy_lat,
            timestep=timesteps,
            encoder_hidden_states=text_emb,
            controlnet_cond=cond,           # [B,3,512,512] in [-1,1]
            return_dict=True,
        )
        lat_pred = pipe.unet(
            sample=noisy_lat,
            timestep=timesteps,
            encoder_hidden_states=text_emb,
            down_block_additional_residuals=ctrl_out.down_block_res_samples,
            mid_block_additional_residual=ctrl_out.mid_block_res_sample,
            return_dict=True,
        ).sample  # 예측값

        # ⑤ 손실 (ε-예측 기본 / v-예측 자동 대응)
        if pred_type == "v_prediction":
            target = noise_scheduler.get_velocity(target_lat, noise, timesteps)
        else:
            target = noise  # epsilon

        loss = F.mse_loss(lat_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=(1,2,3))  # per-sample

        # (선택) SNR 가중
        if USE_SNR_WEIGHT:
            snr = get_snr(timesteps).to(loss.dtype)  # (B,)
            # Imagen에서 제안한 gamma-weighted loss (가벼운 변형)
            # weight = min(snr, gamma) / snr  ≈ 저주파/고주파 균형
            weights = (snr.clamp(max=SNR_GAMMA) / snr).detach()
            loss = loss * weights

        loss = loss.mean()
        loss.backward()

        # 그라디언트 클립 & 스텝
        torch.nn.utils.clip_grad_norm_(params_controlnet + params_unet_lora, GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        # ⑥ 로그 & 체크포인트
        total_loss  += float(loss.item())
        global_step += 1

        if global_step % 100 == 0:
            lr1 = scheduler.get_last_lr()[0]
            lr2 = scheduler.get_last_lr()[-1]
            print(f"[step {global_step}] loss={loss.item():.4f} | lr_ctrl={lr1:.2e} lr_unet={lr2:.2e}")

        if global_step % SAVE_STEPS == 0:
            # ControlNet(LoRA) 저장
            save_dir_c = f"{OUTPUT_DIR}/ctrl_lora_step{global_step}"
            controlnet.save_pretrained(save_dir_c)
            # UNet(LoRA) 저장
            save_dir_u = f"{OUTPUT_DIR}/unet_lora_step{global_step}"
            pipe.unet.save_pretrained(save_dir_u)
            print(f"[ckpt] saved: {save_dir_c} | {save_dir_u}")

    avg = total_loss / max(1, len(loader))
    print(f"Epoch {epoch} finished | avg loss: {avg:.4f}")

    # 에폭 스냅샷
    controlnet.save_pretrained(f"{OUTPUT_DIR}/ctrl_lora_epoch{epoch}")
    pipe.unet.save_pretrained(f"{OUTPUT_DIR}/unet_lora_epoch{epoch}")
    print(f"[ckpt] epoch {epoch} saved.")
