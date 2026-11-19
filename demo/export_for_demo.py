"""
Export CLIPDraw drawings for the Visibility Threshold demo page.

Run this in the same Colab as CLIPDraw (pydiffvg + OpenAI CLIP already installed).
For each word it:
  1. ranks the drawing's strokes by importance (how much CLIP's score drops
     when that one stroke is removed),
  2. renders only the top 8 / 16 / 32 / 64 / 128 / 256 strokes,
  3. asks SigLIP (and optionally Florence) to judge each partial drawing,
  4. writes JSON the demo page can load ("Load results JSON").

No new CLIPDraw runs are needed: it only re-renders and judges the finished drawing.
It reads the clipdraw_final.svg files that batch/batch_runner.py already saves.

Usage in Colab, after notebook Sections 1-4 (so `model` is the loaded CLIP):

    %run -i demo/export_for_demo.py
    words = load_words("/content/Visibility_Thresholds/dataset/processed/sampled_words.csv")
    export_part("/content/drive/MyDrive/clipdraw_results/part1", model, words,
                out_path="/content/drive/MyDrive/clipdraw_results/demo_part1.json",
                limit=10)   # a handful of words is enough for the demo

Or for one drawing still in memory:

    data = export_word("apple", shapes, shape_groups, model,
                       distractors=pick_distractors("apple", words))
    save_json([data], "/content/drive/MyDrive/clipdraw_results/demo_apple.json")
"""

import csv
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import clip
import pydiffvg
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModel, AutoProcessor

BUDGETS = [8, 16, 32, 64, 128, 256]
PROMPT = "A drawing of {}."   # team-agreed CLIPDraw template
CANVAS = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pydiffvg.set_device(device)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)


# ---------- rendering ----------

def render(shapes, groups, canvas=CANVAS):
    """Render diffvg shapes on a white background -> 1x3xHxW tensor in [0, 1]."""
    scene = pydiffvg.RenderFunction.serialize_scene(canvas, canvas, shapes, groups)
    img = pydiffvg.RenderFunction.apply(canvas, canvas, 2, 2, 0, None, *scene)
    img = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4])
    return img.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)


def subset(shapes, groups, idx):
    """Keep only the strokes in idx (each CLIPDraw group holds one path)."""
    sub_shapes = [shapes[i] for i in idx]
    sub_groups = [
        pydiffvg.ShapeGroup(shape_ids=torch.tensor([j]), fill_color=None,
                            stroke_color=groups[i].stroke_color)
        for j, i in enumerate(idx)
    ]
    return sub_shapes, sub_groups


def load_svg(path):
    _, _, shapes, groups = pydiffvg.svg_to_scene(path)
    return shapes, groups


# ---------- 1. stroke importance (CLIP, the model CLIPDraw optimized against) ----------

@torch.no_grad()
def clip_text_feature(clip_model, word):
    t = clip_model.encode_text(clip.tokenize(PROMPT.format(word)).to(device)).float()
    return t / t.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_score(clip_model, img, text_feat):
    x = F.interpolate(img, size=224, mode="bilinear", align_corners=False)
    x = (x - CLIP_MEAN) / CLIP_STD
    f = clip_model.encode_image(x).float()
    f = f / f.norm(dim=-1, keepdim=True)
    return (f @ text_feat.T).item()


@torch.no_grad()
def stroke_order(shapes, groups, clip_model, word, canvas=CANVAS):
    """Indices of strokes, most important first (leave-one-out CLIP score drop)."""
    tf = clip_text_feature(clip_model, word)
    all_idx = list(range(len(shapes)))
    base = clip_score(clip_model, render(*subset(shapes, groups, all_idx), canvas), tf)
    drops = []
    for i in all_idx:
        rest = all_idx[:i] + all_idx[i + 1:]
        drops.append(base - clip_score(clip_model, render(*subset(shapes, groups, rest), canvas), tf))
    return sorted(all_idx, key=lambda i: -drops[i])


# ---------- 2. judges ----------
# Same SigLIP checkpoint as text_score/text_score.py. Change it if the judging stage uses another.

SIGLIP_ID = "google/siglip-so400m-patch14-384"
_siglip, _siglip_proc = None, None


def _load_siglip():
    global _siglip, _siglip_proc
    if _siglip is None:
        _siglip = AutoModel.from_pretrained(SIGLIP_ID).to(device).eval()
        _siglip_proc = AutoProcessor.from_pretrained(SIGLIP_ID)
    return _siglip, _siglip_proc


@torch.no_grad()
def siglip_rank(pil_img, word, distractors):
    """Rank of the target word (1 = best) among word + distractors."""
    model, proc = _load_siglip()
    texts = [PROMPT.format(w) for w in [word] + distractors]
    inputs = proc(text=texts, images=pil_img, padding="max_length", return_tensors="pt").to(device)
    logits = model(**inputs).logits_per_image[0]
    return int((logits > logits[0]).sum().item()) + 1


def pick_distractors(word, word_list, k=99, seed=0):
    """Fixed random distractors, so every stroke budget is judged against the same words."""
    pool = sorted(w for w in set(word_list) if w != word)
    return random.Random(f"{seed}-{word}").sample(pool, min(k, len(pool)))


# florence_fn(pil_img, word) -> float in [0, 1]
# Optional. Hook this up to the Florence judging stage: how well Florence's
# caption of the partial drawing matches the target word.
# (text_score/ is a different measure: how much written text is in the image.)


# ---------- 3. export ----------

def export_word(word, shapes, groups, clip_model, distractors,
                florence_fn=None, canvas=CANVAS, rank_cutoff=5, florence_cutoff=0.5):
    order = stroke_order(shapes, groups, clip_model, word, canvas)

    strokes = []
    for i in order:
        pts = shapes[i].points.detach().cpu().tolist()
        width = float(shapes[i].stroke_width.detach().cpu().reshape(-1)[0])
        color = groups[i].stroke_color.detach().cpu().clamp(0, 1).tolist()
        strokes.append({
            "points": [[round(x, 2), round(y, 2)] for x, y in pts],
            "width": round(width, 2),
            "color": [round(c, 3) for c in color],
        })

    budgets = []
    for n in BUDGETS:
        n = min(n, len(order))
        img = render(*subset(shapes, groups, order[:n]), canvas)
        pil = to_pil_image(img[0].cpu())
        row = {"n": n, "siglip_rank": siglip_rank(pil, word, distractors)}
        if florence_fn is not None:
            row["florence_score"] = round(float(florence_fn(pil, word)), 4)
        budgets.append(row)
        if n == len(order):
            break

    return {
        "word": word,
        "prompt": PROMPT.format(word),
        "canvas": canvas,
        "rank_cutoff": rank_cutoff,
        "num_candidates": len(distractors) + 1,
        "florence_cutoff": florence_cutoff if florence_fn is not None else None,
        "strokes": strokes,
        "budgets": budgets,
    }


def save_json(concepts, path):
    with open(path, "w") as f:
        json.dump(concepts, f)
    print(f"Wrote {len(concepts)} concept(s) to {path}")


def find_drawings(part_folder):
    """(word, svg_path) for every word folder that batch_runner wrote (NNN_slug/clipdraw_final.svg)."""
    part = Path(part_folder)
    names = {}
    summary = part / "batch_summary.csv"
    if summary.exists():
        with summary.open() as f:
            for row in csv.DictReader(f):
                names[Path(row["folder"]).name] = row["word"]
    found = []
    for d in sorted(part.iterdir()):
        svg = d / "clipdraw_final.svg"
        if d.is_dir() and re.match(r"\d{3}_", d.name) and svg.exists():
            word = names.get(d.name, d.name.split("_", 1)[1].replace("_", " "))
            found.append((word, svg))
    return found


def export_part(part_folder, clip_model, word_list, out_path, florence_fn=None, limit=None):
    out = []
    for word, svg in find_drawings(part_folder)[:limit]:
        shapes, groups = load_svg(str(svg))
        out.append(export_word(word, shapes, groups, clip_model,
                               pick_distractors(word, word_list), florence_fn))
        print(f"{word}: " + ", ".join(f"{b['n']}->#{b['siglip_rank']}" for b in out[-1]["budgets"]))
    save_json(out, out_path)
