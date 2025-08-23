# train_epsilon_continue.py
# ------------------------------------------------------------
# Phase-2 학습 + 리플레이 섞기
#  - 목적함수 교정: NG latent에 노이즈 추가 → ε(or v) 예측
#  - 이전 LoRA에서 resume (UNet/ControlNet)
#  - 메인(새 12클래스) + 리플레이(이전 18클래스 일부) 배치를 확률적으로 섞어 학습
#  - 클래스 필터/매핑 일관성 옵션
# ------------------------------------------------------------

import os, json, random, argparse
from typing import Optional, List, Dict, Tuple
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
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]

def find_lora_targets(model):
    leaf = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    keep = set()
    for name, m in model.named_modules():
        if isinstance(m, leaf):
            last = name.split(".")[-1]
            if not last.isdigit():
                keep.add(last)
    return sorted(keep)

def filter_dataset(dataset: DefectSynthesisDataset,
                   include_classes: Optional[List[str]],
                   exclude_classes: Optional[List[str]]):
    if hasattr(dataset, "filter_by_classes"):
        dataset.filter_by_classes(include=include_classes, exclude=exclude_classes)
        return

    if hasattr(dataset, "samples"):
        def keep(sample):
            cname = sample.get("class_name")
            if include_classes and cname not in include_classes:
                return False
            if exclude_classes and cname in exclude_classes:
                return False
            return True
        old = len(dataset.samples)
        dataset.samples = [s for s in dataset.samples if keep(s)]
        print(f"[filter] {old} -> {len(dataset.samples)}")
    else:
        print("[warn] Dataset2에 filter 메서드가 없고 내부 구조를 모릅니다."
              " Dataset2에 filter_by_classes(include, exclude) 추가 권장.")

def build_text_cache(tokenizer, text_encoder, idx2class: Dict[int, str], idx2defect: Dict[int, str], device):
    cache = {}
    MAX_LEN = tokenizer.model_max_length
    with torch.no_grad():
        for c_id, c_name in idx2class.items():
            for d_id, d_name in idx2defect.items():
                ptxt = f"{c_name} {d_name} defect"
                enc = tokenizer([ptxt], padding="max_length", truncation=True,
                                max_length=MAX_LEN, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                emb = text_encoder(**enc).last_hidden_state  # [1,77,hidden]
                cache[(c_id, d_id)] = emb
    return cache

def get_text_embeds(cache, class_ids, defect_ids):
    embs = [cache[(int(c.item()), int(d.item()))] for c, d in zip(class_ids, defect_ids)]
    return torch.cat(embs, dim=0)  # (B,77,H)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser("Phase-2 training with replay mixing")
    # Data
    ap.add_argument("--data_root", required=True, help="메인(새 12클래스) 데이터 루트")
    ap.add_argument("--replay_root", required=True, help="리플레이(이전 18클래스) 데이터 루트")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)

    # Class filters
    ap.add_argument("--include_classes", default=None, help="메인에 포함할 클래스명 CSV")
    ap.add_argument("--exclude_classes", default=None, help="메인에서 제외할 클래스명 CSV")
    ap.add_argument("--replay_include_classes", default=None, help="리플레이에 포함할 클래스명 CSV")
    ap.add_argument("--replay_exclude_classes", default=None, help="리플레이에서 제외할 클래스명 CSV")

    # Mapping
    ap.add_argument("--mapping_json", default=None, help="phase-1의 {class2idx, defect2idx} JSON (권장)")

    # Base models
    ap.add_argument("--sd_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--vae_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet_id", default="lllyasviel/sd-controlnet-scribble")

    # LoRA ranks / LR
    ap.add_argument("--unet_lora_rank", type=int, default=16)
    ap.add_argument("--ctrl_lora_rank", type=int, default=8)
    ap.add_argument("--lr_unet", type=float, default=1e-4)
    ap.add_argument("--lr_ctrl", type=float, default=1e-4)

    # Resume (from phase-1)
    ap.add_argument("--resume_unet", required=True)
    ap.add_argument("--resume_ctrl", required=True)

    # Tricks
    ap.add_argument("--snr_weight", action="store_true")
    ap.add_argument("--snr_gamma", type=float, default=5.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=1000)

    # Replay mixing
    ap.add_argument("--replay_ratio", type=float, default=0.1,
                    help="배치 단위로 리플레이에서 뽑을 확률 (0.1=10%)")
    ap.add_argument("--max_steps_per_epoch", type=int, default=None,
                    help="에폭당 스텝 수(디폴트: 메인로더 길이). 필요시 제한.")

    # Save
    ap.add_argument("--output_dir", default="./checkpoints_phase2_replay")
    ap.add_argument("--save_steps", type=int, default=2000)

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    inc_main  = parse_csv_list(args.include_classes)
    exc_main  = parse_csv_list(args.exclude_classes)
    inc_rep   = parse_csv_list(args.replay_include_classes)
    exc_rep   = parse_csv_list(args.replay_exclude_classes)

    # ===== Dataset: main (새 12클래스) =====
    ds_main = DefectSynthesisDataset(args.data_root, cache_in_ram=False)
    # mapping 적용(권장)
    if args.mapping_json and os.path.isfile(args.mapping_json):
        with open(args.mapping_json, "r") as f:
            mapping = json.load(f)
        if hasattr(ds_main, "set_label_mappings"):
            ds_main.set_label_mappings(mapping["class2idx"], mapping["defect2idx"])
            print("[mapping] applied to main")
        else:
            print("[mapping] dataset has no set_label_mappings; ensure IDs match")
    filter_dataset(ds_main, inc_main, exc_main)
    idx2class_main  = {v: k for k, v in ds_main.class2idx.items()}
    idx2defect_main = {v: k for k, v in ds_main.defect2idx.items()}
    print(f"[main] classes={len(idx2class_main)} defects={len(idx2defect_main)} size={len(ds_main)}")

    # ===== Dataset: replay (이전 18클래스) =====
    ds_rep = DefectSynthesisDataset(args.replay_root, cache_in_ram=False)
    # 같은 mapping 적용 (강력 권장)
    if args.mapping_json and os.path.isfile(args.mapping_json):
        if hasattr(ds_rep, "set_label_mappings"):
            ds_rep.set_label_mappings(mapping["class2idx"], mapping["defect2idx"])
            print("[mapping] applied to replay")
    filter_dataset(ds_rep, inc_rep, exc_rep)
    print(f"[replay] size={len(ds_rep)}")

    # ===== Loaders & batch mixing =====
    # 동일 배치 크기, 동일 전처리 가정
    loader_main = DataLoader(
        ds_main, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
        prefetch_factor=2, persistent_workers=True, timeout=120
    )
    loader_rep = DataLoader(
        ds_rep, batch_size=args.batch_size, shuffle=True,
        num_workers=max(1, args.num_workers // 2), pin_memory=False,
        prefetch_factor=2, persistent_workers=True, timeout=120
    )
    steps_per_epoch = args.max_steps_per_epoch or len(loader_main)
    print(f"[train] steps_per_epoch={steps_per_epoch} (replay_ratio={args.replay_ratio})")

    # ===== Models =====
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    vae = AutoencoderKL.from_pretrained(args.vae_id, subfolder="vae", torch_dtype=torch.float16).to(device).eval()
    for p in vae.parameters(): p.requires_grad = False

    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device).eval()
    for p in text_encoder.parameters(): p.requires_grad = False

    # ControlNet + LoRA (resume)
    ctrl_base = ControlNetModel.from_pretrained(args.controlnet_id, torch_dtype=torch.float16)
    lora_cfg_ctrl = LoraConfig(r=args.ctrl_lora_rank, lora_alpha=max(16, args.ctrl_lora_rank),
                               bias="none", target_modules=find_lora_targets(ctrl_base))
    controlnet = get_peft_model(ctrl_base, lora_cfg_ctrl).to(device)
    controlnet = PeftModel.from_pretrained(controlnet, args.resume_ctrl, is_trainable=True)
    print(f"[resume] ControlNet LoRA <- {args.resume_ctrl}")
    controlnet.print_trainable_parameters()
    controlnet.train()

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.sd_id, controlnet=controlnet, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer,
        torch_dtype=torch.float16
    ).to(device)

    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.controlnet.enable_gradient_checkpointing()

    for p in pipe.unet.parameters():         p.requires_grad = False
    for p in pipe.vae.parameters():          p.requires_grad = False
    for p in pipe.text_encoder.parameters(): p.requires_grad = False
    for p in pipe.controlnet.parameters():   p.requires_grad = True

    lora_cfg_unet = LoraConfig(r=args.unet_lora_rank, lora_alpha=args.unet_lora_rank,
                               target_modules=["to_q","to_k","to_v","to_out.0"], bias="none")
    pipe.unet = get_peft_model(pipe.unet, lora_cfg_unet)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.resume_unet, is_trainable=True)
    print(f"[resume] UNet LoRA <- {args.resume_unet}")

    # (예시) down_blocks LoRA 동결 유지
    for name, p in pipe.unet.named_parameters():
        if "lora_" in name and name.startswith("down_blocks"):
            p.requires_grad = False

    tn_unet = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    tn_ctrl = sum(p.numel() for p in pipe.controlnet.parameters() if p.requires_grad)
    print(f"Trainable params -> UNet(LoRA): {tn_unet:,} | ControlNet(LoRA): {tn_ctrl:,}")
    pipe.unet.train()

    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_id, subfolder="scheduler")
    pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    print("Scheduler prediction_type:", pred_type)

    # 텍스트 캐시: 메인/리플레이 모두 같은 매핑(IDs)라면 하나만 생성해도 됨
    idx2class = {**idx2class_main}  # (동일 매핑 가정)
    idx2defect = {**idx2defect_main}
    text_cache = build_text_cache(tokenizer, text_encoder, idx2class, idx2defect, device)

    # SNR weight
    if args.snr_weight:
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device).float()
        def get_snr(t):
            a = alphas_cumprod[t]
            return a / (1.0 - a).clamp(min=1e-8)
        print(f"SNR weighting (gamma={args.snr_gamma})")

    # Optimizer/scheduler
    p_ctrl = [p for p in pipe.controlnet.parameters() if p.requires_grad]
    p_unet = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = AdamW8bit(
        [{"params": p_ctrl, "lr": args.lr_ctrl},
         {"params": p_unet, "lr": args.lr_unet}],
        betas=(0.9,0.999), weight_decay=1e-4
    )
    warmup = max(0, args.warmup)
    def lr_lambda(step): return min(1.0, step / max(1, warmup))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ===== Training with replay mixing =====
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0

        it_main = iter(loader_main)
        it_rep  = iter(loader_rep)

        for step in tqdm(range(steps_per_epoch), desc=f"[Phase-2+Replay] Epoch {epoch}/{args.epochs}"):
            # 배치 선택: replay_ratio 확률로 리플레이 배치 선택
            use_replay = (random.random() < args.replay_ratio and len(ds_rep) > 0)

            try:
                batch = next(it_rep if use_replay else it_main)
            except StopIteration:
                # 해당 로더를 재시작
                if use_replay:
                    it_rep = iter(loader_rep)
                    batch = next(it_rep)
                else:
                    it_main = iter(loader_main)
                    batch = next(it_main)

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
                text_emb = get_text_embeds(text_cache, class_ids, defect_ids)

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
                torch.nn.utils.clip_grad_norm_(p_ctrl + p_unet, args.grad_clip)

            optimizer.step()
            lr_sched.step()

            total_loss += float(loss.item())
            global_step += 1

            if global_step % 100 == 0:
                lrs = lr_sched.get_last_lr()
                tag = "REPLAY" if use_replay else "MAIN"
                print(f"[step {global_step}][{tag}] loss={loss.item():.4f} | lr_ctrl={lrs[0]:.2e} lr_unet={lrs[-1]:.2e}")

            if global_step % args.save_steps == 0:
                cdir = os.path.join(args.output_dir, f"ctrl_lora_step{global_step}")
                udir = os.path.join(args.output_dir, f"unet_lora_step{global_step}")
                pipe.controlnet.save_pretrained(cdir)
                pipe.unet.save_pretrained(udir)
                print(f"[ckpt] saved: {cdir} | {udir}")

        avg = total_loss / max(1, steps_per_epoch)
        print(f"[Phase-2+Replay] Epoch {epoch} finished | avg loss: {avg:.4f}")

        cdir = os.path.join(args.output_dir, f"ctrl_lora_epoch{epoch}")
        udir = os.path.join(args.output_dir, f"unet_lora_epoch{epoch}")
        pipe.controlnet.save_pretrained(cdir)
        pipe.unet.save_pretrained(udir)
        print(f"[ckpt] epoch {epoch} saved -> {cdir} | {udir}")


if __name__ == "__main__":
    main()

