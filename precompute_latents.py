# precompute_latents.py
import os, pathlib, torch, argparse
from tqdm import tqdm
from diffusers import AutoencoderKL

def prep_mask(t):  # [B,C,H,W] → [B,3,512,512] in [-1,1]
    if t.dim() == 3: t = t.unsqueeze(0)
    if t.size(1) == 1: t = t.repeat(1, 3, 1, 1)
    t = torch.nn.functional.interpolate(t, size=(512, 512), mode="nearest")
    # 원본이 [0,1]이면 아래 줄 유지, 이미 [-1,1]이면 주석 처리
    # t = t.clamp(0,1) * 2 - 1
    return t

@torch.no_grad()
def encode_list(vae, imgs, device, micro_bs=8):
    out = []
    for i in range(0, len(imgs), micro_bs):
        batch = torch.stack(imgs[i:i+micro_bs]).to(device=device, dtype=torch.float16)
        # 필요 시 512로 맞추기
        if batch.shape[-1] != 512 or batch.shape[-2] != 512:
            batch = torch.nn.functional.interpolate(batch, size=(512,512), mode="bilinear", align_corners=False)
        lat = vae.encode(batch).latent_dist.sample() * vae.config.scaling_factor
        out.extend([x.cpu() for x in lat])
    return out  # list[Tensor: 4x64x64]

def main(root, vae_id, micro_bs):
    root = pathlib.Path(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=torch.float16).to(device).eval()

    classes = [p for p in root.iterdir() if p.is_dir() and p.name!="lost+found"]
    for cls in classes:
        ok_dir, ng_dir, full_dir = cls/"OK", cls/"NG", cls/"Full_NG"
        if not (ok_dir.is_dir() and ng_dir.is_dir() and full_dir.is_dir()):
            print(f"[SKIP] {cls.name}: status dir missing"); continue

        (cls/"OK_lat").mkdir(exist_ok=True)
        (cls/"Full_NG_lat").mkdir(exist_ok=True)
        (cls/"NG_cond").mkdir(exist_ok=True)

        ok_files   = {f.name: f for f in ok_dir.glob("*.pt")}
        ng_files   = {f.name: f for f in ng_dir.glob("*.pt")}
        full_files = {f.name: f for f in full_dir.glob("*.pt")}
        names = sorted(ok_files.keys() & ng_files.keys() & full_files.keys())

        for name in tqdm(names, desc=f"[{cls.name}] precompute"):
            # 1) OK → latent
            ok_list   = torch.load(ok_files[name], map_location="cpu")
            ok_lat    = encode_list(vae, ok_list, device, micro_bs)
            torch.save(ok_lat, cls/"OK_lat"/name)

            # 2) Full_NG → latent
            full_list = torch.load(full_files[name], map_location="cpu")
            full_lat  = encode_list(vae, full_list, device, micro_bs)
            torch.save(full_lat, cls/"Full_NG_lat"/name)

            # 3) NG(mask) → cond(3x512x512, [-1,1])
            ng_list   = torch.load(ng_files[name], map_location="cpu")
            cond_list = []
            for t in ng_list:
                if t.dim()==3 and t.size(0) in (1,3):
                    x = t.unsqueeze(0)  # [1,C,H,W]
                elif t.dim()==4:
                    x = t
                else:
                    raise ValueError(f"Unexpected mask shape: {t.shape}")
                x = prep_mask(x)
                cond_list.append(x.squeeze(0).half().cpu())  # [3,512,512]
            torch.save(cond_list, cls/"NG_cond"/name)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/opt/dlami/nvme")
    ap.add_argument("--vae_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--micro_bs", type=int, default=8)
    args = ap.parse_args()
    main(args.root, args.vae_id, args.micro_bs)
