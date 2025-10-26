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
import os, random, csv, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from diffusers import (
    ControlNetModel, StableDiffusionControlNetPipeline,
    DDPMScheduler, AutoencoderKL,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from tqdm import tqdm
from Dataset2 import DefectSynthesisDataset
from torch.optim import AdamW

# ── 무헤드 환경에서 Matplotlib 사용 설정 ─────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ───────────── CONFIG ─────────────
DATA_ROOT      = "/opt/dlami/nvme"
BATCH_SIZE     = 4
EPOCHS         = 6
LR_CTRL        = 1e-4
LR_UNET_LORA   = 1e-4
UNET_LORA_RANK = 16
VAE_ID         = "runwayml/stable-diffusion-v1-5"
SD_ID          = "runwayml/stable-diffusion-v1-5"
CONTROLNET_ID  = "lllyasviel/sd-controlnet-scribble"
OUTPUT_DIR     = "./checkpoints_re"
SAVE_STEPS     = 1000
USE_SNR_WEIGHT = True
SNR_GAMMA      = 5.0
GRAD_CLIP_NORM = 1.0
SEED           = 42

# ── Loss 로깅 설정 ───────────────────────────────────────────────
LOG_DIR    = "./logs_re"
CSV_PATH   = os.path.join(LOG_DIR, "loss_log.csv")
PLOT_PATH  = os.path.join(LOG_DIR, "loss_plot.png")
PLOT_EVERY = 500     # N 스텝마다 loss_plot.png 갱신
EMA_ALPHA  = 0.1     # EMA 스무딩 강도

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV 헤더 생성(최초 1회)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "loss", "lr_ctrl", "lr_unet", "time"])

def ema(prev, x, alpha=EMA_ALPHA):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)

def save_loss_plot(csv_path=CSV_PATH, plot_path=PLOT_PATH):
    """CSV를 읽어 손실 곡선을 PNG로 저장(가능하면 pandas 사용)."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        steps = df["step"].values
        losses = df["loss"].values
        ema_vals = df["loss"].ewm(alpha=EMA_ALPHA).mean().values
    except Exception:
        # pandas 없을 때: 순수 csv로 읽기
        steps, losses, ema_vals = [], [], []
        _ema = None
        with open(csv_path, "r") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                s = int(row["step"]); ls = float(row["loss"])
                _ema = ema(_ema, ls)
                steps.append(s); losses.append(ls); ema_vals.append(_ema)

    plt.figure(figsize=(9,4))
    plt.plot(steps, losses, label="loss", linewidth=1.0)
    plt.plot(steps, ema_vals, label=f"ema({EMA_ALPHA})", linewidth=2.0)
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training Loss")
    plt.grid(True, linewidth=0.4, alpha=0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_path); plt.close()

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

# g5.xlarge(vCPU 4) 권장 로더
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=3, pin_memory=True, prefetch_factor=4,
    persistent_workers=True
)

# ───────────── CUDA/정밀도 튜닝 ─────────────
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

# ───────────── MODELS ─────────────
vae = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae").to(device).eval()
for p in vae.parameters(): p.requires_grad = False

tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
for p in text_encoder.parameters(): p.requires_grad = False

def find_lora_targets(model):
    leaf = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    keep = set()
    for full_name, module in model.named_modules():
        if isinstance(module, leaf):
            last = full_name.split(".")[-1]
            if not last.isdigit():
                keep.add(last)
    return sorted(keep)

# ControlNet base + LoRA 주입
controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID)
lora_cfg_ctrl = LoraConfig(r=8, lora_alpha=16, bias="none",
                           target_modules=find_lora_targets(controlnet))
controlnet = get_peft_model(controlnet, lora_cfg_ctrl)

# 파이프라인
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    SD_ID, controlnet=controlnet, vae=vae,
    text_encoder=text_encoder, tokenizer=tokenizer,
).to(device)

# (중요) xFormers OFF, SDPA ON → dtype 이슈 방지
pipe.unet.set_attn_processor(AttnProcessor2_0())
pipe.controlnet.set_attn_processor(AttnProcessor2_0())

# VAE 메모리 최적화(여유 있으면 disable_slicing()이 속도↑)
pipe.vae.disable_slicing()
pipe.unet.to(memory_format=torch.channels_last)

# ───────────── 동결 정책: LoRA만 학습 ─────────────
# 1) 전체 동결
for p in pipe.unet.parameters():         p.requires_grad = False
for p in pipe.vae.parameters():          p.requires_grad = False
for p in pipe.text_encoder.parameters(): p.requires_grad = False
for p in pipe.controlnet.parameters():   p.requires_grad = False

# 2) UNet LoRA 주입
lora_cfg_unet = LoraConfig(
    r=UNET_LORA_RANK,
    lora_alpha=UNET_LORA_RANK,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    bias="none",
)
pipe.unet = get_peft_model(pipe.unet, lora_cfg_unet)

# 3) gradient checkpointing은 일단 OFF(속도 위해)
# pipe.unet.enable_gradient_checkpointing()
# pipe.controlnet.enable_gradient_checkpointing()

# 4) LoRA 파라미터만 학습 허용
for name, p in pipe.unet.named_parameters():
    if "lora_" in name:
        p.requires_grad = True
for name, p in pipe.controlnet.named_parameters():
    if "lora_" in name:
        p.requires_grad = True

# ── 유틸: 파라미터 카운트
def count_params(*modules, trainable_only=False):
    total = 0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if (not trainable_only) or p.requires_grad:
                total += p.numel()
    return total

# 학습 가능 파라미터 요약
tn_unet  = count_params(pipe.unet, trainable_only=True)
tn_ctrl  = count_params(pipe.controlnet, trainable_only=True)
tn_vae   = count_params(pipe.vae, trainable_only=True)
tn_text  = count_params(pipe.text_encoder, trainable_only=True)
tot_all  = count_params(pipe.unet, pipe.controlnet, pipe.vae, pipe.text_encoder, trainable_only=False)
tot_trn  = tn_unet + tn_ctrl + tn_vae + tn_text
print(f"Trainable -> UNet(LoRA): {tn_unet:,} | ControlNet(LoRA): {tn_ctrl:,} | VAE: {tn_vae:,} | TextEnc: {tn_text:,} | ratio={tot_trn/tot_all:.4%}")

pipe.unet.train(); pipe.controlnet.train()

# ───────────── 스케줄러 ─────────────
noise_scheduler = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
print("Scheduler prediction_type:", pred_type)

# ───────────── 텍스트 임베딩 캐시 ─────────────
with torch.no_grad():
    prompt_cache = {}
    MAX_LEN = tokenizer.model_max_length
    for c_id, c_name in idx2class.items():
        for d_id, d_name in idx2defect.items():
            ptxt = f"{c_name} {d_name} defect"
            enc = tokenizer([ptxt], padding="max_length", truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = text_encoder(**enc).last_hidden_state  # [1,77,H]
            prompt_cache[(c_id, d_id)] = emb  # fp32 캐시

def get_text_embeds(class_ids, defect_ids):
    embs = [prompt_cache[(int(c.item()), int(d.item()))] for c, d in zip(class_ids, defect_ids)]
    return torch.cat(embs, dim=0)  # (B,77,H) — fp32

# ───────────── SNR 가중(선택) ─────────────
if USE_SNR_WEIGHT:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device).float()
    def get_snr(timesteps):
        a = alphas_cumprod[timesteps].clamp(1e-6, 1 - 1e-6)
        return a / (1.0 - a)
    print("SNR weighting enabled (gamma=%.1f)" % SNR_GAMMA)

# ───────────── OPTIMIZER & SCHED ─────────────
params_controlnet = [p for p in pipe.controlnet.parameters() if p.requires_grad]
params_unet_lora  = [p for p in pipe.unet.parameters()       if p.requires_grad]

optimizer = AdamW(
    [{"params": params_controlnet, "lr": LR_CTRL},
     {"params": params_unet_lora,  "lr": LR_UNET_LORA}],
    betas=(0.9, 0.999), weight_decay=1e-4, fused=True
)

warmup_steps = 1000
def lr_lambda(step): return min(1.0, step / max(1, warmup_steps))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# AMP 스케일러
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# ───────────── TRAIN LOOP ─────────────
global_step = 0
ema_loss = None  # EMA 초기화

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch in pbar:
        # ① 배치 언팩
        lat_ok   = batch["ok_lat"].to(device, non_blocking=True)   # (미사용; 향후 대비)
        lat_ng   = batch["ng_lat"].to(device, non_blocking=True)
        cond     = batch["cond"].to(device,   non_blocking=True)   # [-1,1]
        class_ids  = batch["class_id"]
        defect_ids = batch["defect_id"]
        bsz        = lat_ok.size(0)

        # ② 노이즈/타임스텝 (fp32 계산 경로)
        target_lat_f32 = lat_ng.float()
        noise_f32      = torch.randn_like(target_lat_f32)
        timesteps      = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
        noisy_lat_f32  = noise_scheduler.add_noise(target_lat_f32, noise_f32, timesteps)

        # ③ 임베딩(fp32) 준비
        with torch.no_grad():
            text_emb_f32 = get_text_embeds(class_ids, defect_ids)  # fp32

        optimizer.zero_grad(set_to_none=True)

        # ④ forward (AMP fp16)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
            noisy_lat = noisy_lat_f32.half()
            noise     = noise_f32.half()
            cond_h    = cond.half()
            text_emb  = text_emb_f32.half()

            ctrl_out = pipe.controlnet(
                sample=noisy_lat,
                timestep=timesteps,
                encoder_hidden_states=text_emb,
                controlnet_cond=cond_h,
                return_dict=True,
            )
            down_res = [r.to(noisy_lat.dtype) for r in ctrl_out.down_block_res_samples]
            mid_res  = ctrl_out.mid_block_res_sample.to(noisy_lat.dtype)

            lat_pred = pipe.unet(
                sample=noisy_lat,
                timestep=timesteps,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
                return_dict=True,
            ).sample

            # ⑤ 손실 (epsilon / v_prediction)
            if pred_type == "v_prediction":
                target = noise_scheduler.get_velocity(target_lat_f32, noise_f32, timesteps).half()
            else:
                target = noise

            loss = F.mse_loss(lat_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=(1,2,3))
            if USE_SNR_WEIGHT:
                snr = get_snr(timesteps).to(loss.dtype)
                weights = (snr.clamp(max=SNR_GAMMA) / snr).detach()
                loss = loss * weights
            loss = loss.mean()

        # ── 손실/러닝레이트 CSV 기록 + 즉시 곡선 저장 ─────────────
        step_loss = float(loss.item())
        ema_loss = ema(ema_loss, step_loss)
        lr_ctrl_cur = scheduler.get_last_lr()[0]
        lr_unet_cur = scheduler.get_last_lr()[-1]

        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([global_step + 1, epoch, step_loss, lr_ctrl_cur, lr_unet_cur, time.time()])

        if (global_step + 1) % PLOT_EVERY == 0:
            save_loss_plot()
            print(f"[plot] updated: {PLOT_PATH}")

        # ⑥ backward + step
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(params_controlnet + params_unet_lora, GRAD_CLIP_NORM)
        if not torch.isfinite(total_norm):
            print("[skip] non-finite grad norm; zeroing grads")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ⑦ 진행바/로그 & 체크포인트
        total_loss  += step_loss
        global_step += 1
        pbar.set_postfix(loss=f"{step_loss:.4f}", ema=f"{ema_loss:.4f}")

        if global_step % 100 == 0:
            print(f"[step {global_step}] loss={step_loss:.4f} | lr_ctrl={lr_ctrl_cur:.2e} lr_unet={lr_unet_cur:.2e}")

        if global_step % SAVE_STEPS == 0:
            save_dir_c = f"{OUTPUT_DIR}/ctrl_lora_step{global_step}"
            controlnet.save_pretrained(save_dir_c)
            save_dir_u = f"{OUTPUT_DIR}/unet_lora_step{global_step}"
            pipe.unet.save_pretrained(save_dir_u)
            print(f"[ckpt] saved: {save_dir_c} | {save_dir_u}")

    avg = total_loss / max(1, len(loader))
    print(f"Epoch {epoch} finished | avg loss: {avg:.4f}")

    controlnet.save_pretrained(f"{OUTPUT_DIR}/ctrl_lora_epoch{epoch}")
    pipe.unet.save_pretrained(f"{OUTPUT_DIR}/unet_lora_epoch{epoch}")
    print(f"[ckpt] epoch {epoch} saved.")
