import os
from django.shortcuts import render
from django.conf import settings
from .forms import UploadForm
from .infer import run_infer

def index(request):
    ctx = {"form": UploadForm()}
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            okf   = form.cleaned_data["ok_image"]
            maskf = form.cleaned_data["mask_image"]
            prompt = form.cleaned_data["prompt"]
            steps    = form.cleaned_data["steps"]
            guidance = form.cleaned_data["guidance"]
            strength = form.cleaned_data["strength"]
            seed     = form.cleaned_data["seed"]

            up_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
            os.makedirs(up_dir, exist_ok=True)
            ok_path   = os.path.join(up_dir, okf.name)
            mask_path = os.path.join(up_dir, maskf.name)
            with open(ok_path, "wb") as f:   [f.write(c) for c in okf.chunks()]
            with open(mask_path, "wb") as f: [f.write(c) for c in maskf.chunks()]

            out_dir  = os.path.join(settings.MEDIA_ROOT, "results")
            out_path = run_infer(ok_path, mask_path, prompt, out_dir,
                                 steps=steps, guidance=guidance, strength=strength, seed=seed)

            rel_out = os.path.relpath(out_path, settings.MEDIA_ROOT).replace("\\", "/")
            ctx.update({
                "form": form,
                "ok_url":   settings.MEDIA_URL + "uploads/" + os.path.basename(ok_path),
                "mask_url": settings.MEDIA_URL + "uploads/" + os.path.basename(mask_path),
                "res_url":  settings.MEDIA_URL + rel_out,
                "prompt": prompt, "steps": steps, "guidance": guidance,
                "strength": strength, "seed": seed,
            })
            return render(request, "index.html", ctx)
        else:
            ctx["form"] = form
    return render(request, "index.html", ctx)
