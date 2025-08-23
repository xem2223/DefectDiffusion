# train_epsilon_continue.py
# ------------------------------------------------------------
# 이어서 학습 (continual learning for new classes)
# - NG latent에 노이즈 추가 → ε(or v) 예측 (교정된 목적함수)
# - 이전 UNet/ControlNet LoRA에서 resume
# - 클래스 필터링(포함/제외)로 데이터 부분 학습
# - SNR 가중, 워밍업, 그라드클립, 저장 주기 동일
# ------------------------------------------------------------

import os, json, random, argparse
from typing import List, Optional, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from bitsandbytes.optim import AdamW8bit
from diffusers import (
    ControlNetModel, StableDiffusionControlNetPipeline,
    DDPMScheduler, AutoencoderKL,
)
from tqdm import tqdm

# 사용자 데이터셋
from Dataset2 import DefectSynthesisDataset


# ---------------- Utils ----------------

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if not s: return None
    return [x.strip() for x in s.split(",") if x.strip()]

def find_lora_targets(model):
    leaf_types = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    keep = set()
    for full_name, module in model.named_modules():
        if isinstance(module, leaf_types):
            last = full_name.split(".")[-1]
            if not last.isdigit():
                keep.add(last)
    return sorted(keep)

def build_text_cache(tokenizer, text_encoder, classes: Dict[int, str], defects: Dict[int, str], device):
    cache = {}
    MAX_LEN = tokenizer.model_max_length
    with torch.no_grad():
        for c_id, c_name in classes.items():
            for d_id, d_name in defects.items():
                ptxt = f"{c_name} {d_name} defect"
                enc = tokenizer([ptxt], padding="max_length", truncation=True,
                                max_length=MAX_LEN, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                emb = text_encoder(**enc).last_hidden_state  # [1,77,hidden]
                cache[(c_id, d_id)] = emb
    return cache

def filter_dataset(dataset: DefectSynthesisDataset,
                   include_classes: Optional[List[str]] = None,
                   exclude_classes: Optional[List[str]] = None):
    """Dataset2 안에 클래스명 필터 기능이 없다면, 내부 인덱스 리스트를 필터링할 수 있도록 Dataset2에
    filter API가 있다면 활용하세요. 여기서는 Dataset2가 class_name을 sample에 담아준다고 가정."""
    if hasattr(dataset, "filter_by_classes"):
        dataset.filter_by_classes(include=include_classes, exclude=exclude_classes)
        return

    # Fallback: dataset.samples 처럼 접근 가능하면 필터 (필요에 맞게 수정)
    if hasattr(dataset, "samples"):
        def keep(sample):
            cname = sample.get("class_name")
            if include_classes and cname not in include_classes:
                return False
            if exclude_classes and cname in exclude_classes:
                return False
            return True
        dataset.samples = [s for s in dataset.samples if keep(s)]
        print(f"[filter] filtered dataset size: {len(dataset.samples)}")
    else:
        print("[warn] Dataset2에 필터 API가 없고 내부 구조 접근이 불가합니다. "
              "Dataset2에 filter_by_classes(include, exclude) 메서드를 추가하는 것을 권장합니다.")


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser("Continue training on remaining classes")
    # Data
    ap.add_argument("--data_root", default="/opt/dlami/nvme")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)

    # Class filtering
    ap.add_argument("--include_classes", default=None,
                    help="comma-separated class names to train (e.g. 'classA,classB')")
    ap.add_argument("--exclude_classes", default=None,
                    help="comma-separated class names to ignore")

    # Base models
    ap.add_argument("--sd_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--vae_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet_id", default="lllyasviel/sd-controlnet-scribble")

    # LoRA ranks / LR
    ap.add_argument("--unet_lora_rank", type=int, default=16)
    ap.add_argument("--ctrl_lora_rank", type=int, default=8)
    ap.add_argument("--lr_unet", type=float, default=1e-4)
    ap.add_argument("--lr_ctrl", type=float, default=1e-4)

    # Resume from previous LoRA (first 18 classes)
    ap.add_argument("--resume_unet", required=True,
                    help="UNet LoRA dir from phase-1 (e.g. ./checkpoints/unet_lora_step28000)")
    ap.add_argument("--resume_ctrl", required=True,
                    help="ControlNet LoRA dir from phase-1 (e.g. ./checkpoints/ctrl_lora_step28000)")

    # Optional: class/defect mapping to keep IDs consistent across phases
    ap.add_argument("--mapping_json", default=None,
                    help="JSON path with {class2idx, defect2idx} saved in phase-1 (optional)")

    # Tricks
    ap.add_argument("--snr_weight", action="store_true")
    ap.add_argument("--snr_gamma", type=float, default=5.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=1000)

    # Save
    ap.add_argument("--output_dir", default="./checkpoints_phase2")
    ap.add_argument("--save_steps", type=int, default=2000)

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    include_classes = parse_csv_list(args.include_classes)
    exclude_classes = parse_csv_list(args.exclude_classes)

    # ===== Dataset & Filter =====
    dataset = DefectSynthesisDataset(args.data_root, cache_in_ram=False)

    # (선택) 1차 학습 때 저장해둔 매핑 JSON을 로드해 ID 일관성 유지
    # JSON format: {"class2idx": {"classA": 0, ...}, "defect2idx": {"defX": 0, ...}}
    if args.mapping_json and os.path.isfile(args.mapping_json):
        with open(args.mapping_json, "r") as f:
            mapping = json.load(f)
        # Dataset2가 외부 매핑 적용 API가 있다면 사용
        if hasattr(dataset, "set_label_mappings"):
            dataset.set_label_mappings(mapping["class2idx"], mapping["defect2idx"])
            print("[mapping] applied external class/defect mapping")
        else:
            print("[mapping] mapping loaded but Dataset2 has no set_label_mappings API")

    # 클래스 필터링
    filter_dataset(dataset, include_classes, exclude_classes)

    idx2class  = {v: k for k, v in dataset.class2idx.items()}
    idx2defect = {v: k for k, v in dataset.defect2idx.items()}
    print(f"[classes] {len(idx2class)} classes after filter")
    print(f"[defects] {len(idx2defect)} defects")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
        prefetch_factor=2, persistent_workers=True, timeout=120
    )

    # ===== Models =====
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    vae = AutoencoderKL.from_pretrained(args.vae_id, subfolder="vae", torch_dtype=torch.float16).to(device).eval()
    for p in vae.parameters(): p.requires_grad = False

    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device).eval()
    for p in text_encoder.parameters(): p.requires_grad = False

    # ControlNet with LoRA (resume)
    controlnet_base = ControlNetModel.from_pretrained(args.controlnet_id, torch_dtype=torch.float16)
    target_mods_ctrl = find_lora_targets(controlnet_base)
    lora_cfg_ctrl = LoraConfig(r=args.ctrl_lora_rank, lora_alpha=max(16, args.ctrl_lora_rank),
                               bias="none", target_modules=target_mods_ctrl)
    controlnet = get_peft_model(controlnet_base, lora_cfg_ctrl).to(device)

    # resume LoRA weights
    controlnet = PeftModel.from_pretrained(controlnet, args.resume_ctrl, is_trainable=True)
    print(f"[resume] ControlNet LoRA loaded from {args.resume_ctrl}")
    controlnet.print_trainable_parameters()
    controlnet.train()

    # Build pipeline with UNet, then add LoRA & resume
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.sd_id, controlnet=controlnet, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer,
        torch_dtype=torch.float16
    ).to(device)

    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.controlnet.enable_gradient_checkpointing()

    # Freeze base
    for p in pipe.unet.parameters():         p.requires_grad = False
    for p in pipe.vae.parameters():          p.requires_grad = False
    for p in pipe.text_encoder.parameters(): p.requires_grad = False
    for p in pipe.controlnet.parameters():   p.requires_grad = True

    # UNet LoRA + resume
    lora_cfg_unet = LoraConfig(r=args.unet_lora_rank, lora_alpha=args.unet_lora_rank,
                               target_modules=["to_q","to_k","to_v","to_out.0"], bias="none")
    pipe.unet = get_peft_model(pipe.unet, lora_cfg_unet)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.resume_unet, is_trainable=True)
    print(f"[resume] UNet LoRA loaded from {args.resume_unet}")

    # (예시) down_blocks LoRA 동결 유지
    for name, p in pipe.unet.named_parameters():
        if "lora_" in name and name.startswith("down_blocks"):
            p.requires_grad = False

    tn_unet = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    tn_ctrl = sum(p.numel() for p in pipe.controlnet.parameters() if p.requires_grad)
    print(f"Trainable params -> UNet(LoRA): {tn_unet:,} | ControlNet(LoRA): {tn_ctrl:,}")
    pipe.unet.train()

    # Noise scheduler & pred type
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_id, subfolder="scheduler")
    pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    print("Scheduler prediction_type:", pred_type)

    # Text cache (현재 필터된 클래스/결함으로만 구성)
    prompt_cache = build_text_cache(tokenizer, text_encoder, idx2class, idx2defect, device)
    def get_text_embeds(class_ids, defect_ids):
        embs = [prompt_cache[(int(c.item()), int(d.item()))] for c, d in zip(class_ids, defect_ids)]
        return torch.cat(embs, dim=0)

    # SNR weighting
    if args.snr_weight:
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device).float()
        def get_snr(t):
            a = alphas_cumprod[t]
            return a / (1.0 - a).clamp(min=1e-8)
        print(f"SNR weighting enabled (gamma={args.snr_gamma})")

    # Optimizer / warmup
    params_controlnet = [p for p in pipe.controlnet.parameters() if p.requires_grad]
    params_unet_lora  = [p for p in pipe.unet.parameters()       if p.requires_grad]
    optimizer = AdamW8bit(
        [{"params": params_controlnet, "lr": args.lr_ctrl},
         {"params": params_unet_lora,  "lr": args.lr_unet}],
        betas=(0.9, 0.999), weight_decay=1e-4
    )
    warmup_steps = max(0, args.warmup)
    def lr_lambda(step): return min(1.0, step / max(1, warmup_steps))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ============ Train loop ============
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"[Phase-2] Epoch {epoch}/{args.epochs}"):
            lat_ok = batch["ok_lat"].to(device, torch.float16, non_blocking=True)
            lat_ng = batch["ng_lat"].to(device, torch.float16, non_blocking=True)
            cond   = batch["cond"].to(device,   torch.float16, non_blocking=True)
            class_ids, defect_ids = batch["class_id"], batch["defect_id"]
            bsz = lat_ok.size(0)

            # NG 기준 ε/v 예측
            target_lat = lat_ng
            noise      = torch.randn_like(target_lat)
            timesteps  = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            noisy_lat  = noise_scheduler.add_noise(target_lat, noise, timesteps)

            with torch.no_grad():
                text_emb = get_text_embeds(class_ids, defect_ids)

            optimizer.zero_grad(set_to_none=True)

            ctrl_out = pipe.controlnet(
                sample=noisy_lat, timestep=timesteps,
                encoder_hidden_states=text_emb, controlnet_cond=cond,
                return_dict=True,
            )
            lat_pred = pipe.unet(
                sample=noisy_lat, timestep=timesteps,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=ctrl_out.down_block_res_samples,
                mid_block_additional_residual=ctrl_out.mid_block_res_sample,
                return_dict=True,
            ).sample

            # loss
            if pred_type == "v_prediction":
                target = noise_scheduler.get_velocity(target_lat, noise, timesteps)
            else:
                target = noise
            loss = F.mse_loss(lat_pred.float(), target.float(), reduction="none").mean(dim=(1,2,3))

            if args.snr_weight:
                snr = get_snr(timesteps).to(loss.dtype)
                weights = (snr.clamp(max=args.snr_gamma) / snr).detach()
                loss = loss * weights

            loss = loss.mean()
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params_controlnet + params_unet_lora, args.grad_clip)

            optimizer.step()
            lr_sched.step()

            total_loss += float(loss.item())
            global_step += 1

            if global_step % 100 == 0:
                lrs = lr_sched.get_last_lr()
                print(f"[step {global_step}] loss={loss.item():.4f} | lr_ctrl={lrs[0]:.2e} lr_unet={lrs[-1]:.2e}")

            if global_step % args.save_steps == 0:
                cdir = os.path.join(args.output_dir, f"ctrl_lora_step{global_step}")
                udir = os.path.join(args.output_dir, f"unet_lora_step{global_step}")
                pipe.controlnet.save_pretrained(cdir)
                pipe.unet.save_pretrained(udir)
                print(f"[ckpt] saved: {cdir} | {udir}")

        avg = total_loss / max(1, len(loader))
        print(f"[Phase-2] Epoch {epoch} finished | avg loss: {avg:.4f}")

        # epoch snapshot
        cdir = os.path.join(args.output_dir, f"ctrl_lora_epoch{epoch}")
        udir = os.path.join(args.output_dir, f"unet_lora_epoch{epoch}")
        pipe.controlnet.save_pretrained(cdir)
        pipe.unet.save_pretrained(udir)
        print(f"[ckpt] epoch {epoch} saved -> {cdir} | {udir}")


if __name__ == "__main__":
    main()
