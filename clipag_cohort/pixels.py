"""Direct pixel optimization against a CLIP-style image encoder (no strokes, no generator).

Follows the `generate()` recipe in the CLIPAG authors' notebook
(https://github.com/royg27/CLIPAG, CLIPAG_T2I.ipynb):

- raw 224x224 RGB pixels, AdamW (lr 0.0175), 100 steps
- each step: 32 whole-image views + 32 random 128px crops resized to 224, each with
  Gaussian noise (std 0.01) and DiffAugment ('color,translation,resize,cutout')
- loss = 1.0 * mean(1 - cos(global views, text)) + 5.0 * mean(1 - cos(crops, text))
- CLIP mean/std normalization before encoding
- gradient-norm clip 0.001; dynamic thresholding (p = 0.95) after every step
- returns the image with the lowest global loss

Two deliberate differences, so pixel runs pair with the stroke runs:
- start image: uniform noise * 0.05 from seed 0 (the authors' commented-out option),
  identical for every word and both models. The authors' default picks a per-prompt
  start from a Tiny-ImageNet mixture model, which would differ between words.
- prompt: "A drawing of {word}." as in prod150 (the authors use the raw caption).

DiffAugment is re-implemented from the parameters in that notebook, not copied verbatim.
All randomness comes from torch's global generator, seeded once per word.
"""

import torch
import torch.nn.functional as F

RECIPE = {
    "source": "royg27/CLIPAG CLIPAG_T2I.ipynb generate()",
    "resolution": 224,
    "optimizer": "AdamW", "lr": 0.0175, "steps": 100,
    "num_views": 32, "num_patches": 32, "patch_size": 128, "noise_std": 0.01,
    "diffaugment_policy": "color,translation,resize,cutout",
    "translation_ratio": 0.125, "resize_scale": [0.8, 1.2], "cutout_ratio": 0.5,
    "lambda_global": 1.0, "lambda_patch": 5.0,
    "grad_clip_max_norm": 0.001, "dynamic_threshold_p": 0.95,
    "normalize_clip_inputs": True,
    "init": "uniform noise * 0.05, seed 0 (authors' commented-out alternative)",
    "selection": "image with the lowest global loss",
}
SMOKE = {"steps": 5, "num_views": 4, "num_patches": 4}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def diff_augment(x, translation_ratio=0.125, resize_scale=(0.8, 1.2), cutout_ratio=0.5):
    """Batch DiffAugment: color, translation, resize, cutout. Differentiable in x."""
    n, _, h, w = x.shape
    dev = x.device
    r = lambda: torch.rand(n, 1, 1, 1, device=dev)
    # color: brightness, saturation, contrast
    x = x + (r() - 0.5)
    mean = x.mean(dim=1, keepdim=True)
    x = (x - mean) * (r() * 2) + mean
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - mean) * (r() + 0.5) + mean
    # translation: shift by up to ratio * size, zero padding
    sx, sy = int(h * translation_ratio + 0.5), int(w * translation_ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, (n, 1, 1), device=dev)
    ty = torch.randint(-sy, sy + 1, (n, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(torch.arange(n, device=dev), torch.arange(h, device=dev),
                                torch.arange(w, device=dev), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    x = F.pad(x, [1, 1, 1, 1]).permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)
    # resize: one scale per batch, then pad or centre-crop back to the original size
    lo, hi = resize_scale
    size = max(1, int(round(h * (lo + (hi - lo) * float(torch.rand(1))))))
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    if size < h:
        p = h - size
        x = F.pad(x, [p // 2, p - p // 2, p // 2, p - p // 2])
    elif size > h:
        o = (size - h) // 2
        x = x[:, :, o:o + h, o:o + w]
    # cutout: zero a (ratio * size) square at a random position per image
    ch, cw = int(h * cutout_ratio + 0.5), int(w * cutout_ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), (n, 1, 1), device=dev)
    oy = torch.randint(0, w + (1 - cw % 2), (n, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(torch.arange(n, device=dev), torch.arange(ch, device=dev),
                                torch.arange(cw, device=dev), indexing="ij")
    gx = torch.clamp(gx + ox - ch // 2, 0, h - 1)
    gy = torch.clamp(gy + oy - cw // 2, 0, w - 1)
    mask = torch.ones(n, h, w, dtype=x.dtype, device=dev)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


def dynamic_threshold(image, p=0.95):
    """Map to [-1, 1], clip at the p-quantile of |x| (at least 1), rescale to [0, 1]."""
    x = image * 2 - 1
    s = torch.quantile(x.abs().reshape(x.shape[0], -1), p, dim=1).clamp(min=1.0).view(-1, 1, 1, 1)
    return (x.clamp(-s, s) / s + 1) / 2


def random_crops(x, n, size):
    """n random size x size crops of a 1-image batch, each resized back to full resolution."""
    _, _, h, w = x.shape
    tops = torch.randint(0, h - size + 1, (n,)).tolist()
    lefts = torch.randint(0, w - size + 1, (n,)).tolist()
    crops = torch.cat([x[:, :, t:t + size, l:l + size] for t, l in zip(tops, lefts)])
    return F.interpolate(crops, size=(h, w), mode="bilinear", align_corners=False)


def initial_image(seed=0, resolution=224, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(1, 3, resolution, resolution, generator=g) * 0.05).to(device)


def optimize_pixels(encode_image, text_features, *, steps, num_views, num_patches, seed=0,
                    recipe=RECIPE, device="cpu", on_step=None):
    """Optimize one image towards text_features. Returns (best_image, best_step, history).

    encode_image: maps a normalized (N, 3, H, W) float batch to (N, D) features.
    best_image: (1, 3, H, W) in [0, 1], the image whose global loss was lowest.
    history: per step {"global", "patch", "total"}, each measured on the image
    before that step's update.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    text = F.normalize(text_features.detach().float().reshape(1, -1), dim=-1)
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    image = initial_image(seed, recipe["resolution"], device).requires_grad_(True)
    optimizer = torch.optim.AdamW([image], lr=recipe["lr"])
    best, best_step, best_loss, history = None, -1, float("inf"), []

    def score(batch):
        batch = diff_augment(batch, recipe["translation_ratio"], recipe["resize_scale"], recipe["cutout_ratio"])
        if recipe["normalize_clip_inputs"]:
            batch = (batch - mean) / std
        features = F.normalize(encode_image(batch).float(), dim=-1)
        return (1 - features @ text.T).mean()

    for step in range(steps):
        before = image.detach().clone()
        optimizer.zero_grad()
        views = image.repeat(num_views, 1, 1, 1)
        global_loss = score(views + recipe["noise_std"] * torch.randn_like(views))
        noisy = image + recipe["noise_std"] * torch.randn_like(image)
        patch_loss = score(random_crops(noisy, num_patches, recipe["patch_size"]))
        loss = recipe["lambda_global"] * global_loss + recipe["lambda_patch"] * patch_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([image], max_norm=recipe["grad_clip_max_norm"])
        optimizer.step()
        with torch.no_grad():
            image.data = dynamic_threshold(image.data, recipe["dynamic_threshold_p"])
        g = float(global_loss.detach())
        history.append({"global": g, "patch": float(patch_loss.detach()), "total": float(loss.detach())})
        if g < best_loss:
            best, best_step, best_loss = before, step, g
        if on_step is not None:
            on_step(step, image.detach(), history[-1])
    return best.clamp(0, 1), best_step, history
