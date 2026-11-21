"""
Stroke-budget export for the Visibility Threshold demo (Colab, CLIPDraw runtime).

For every finished CLIPDraw drawing (clipdraw_final.svg + experiment.json) it:
  1. ranks the 256 strokes by importance: how much CLIP's score for the prompt
     drops when that one stroke is left out (same CLIP ViT-B/32 and the same
     input normalization setting that CLIPDraw optimized against),
  2. renders the top 8 / 16 / 32 / 64 / 128 / 256 strokes on white, exactly like
     batch_runner saves clipdraw_final.png,
  3. writes each partial drawing as a normal image folder:

         OUT_ROOT/b008/part1/001_word/clipdraw_final.png + experiment.json
         OUT_ROOT/b016/part1/001_word/...
         ...
         OUT_ROOT/strokes/part1/001_word.json   (stroke order, for the demo page)

The b### folders look like ordinary generation output, so the team's evaluation
notebook (visibility_thresholds_evaluation_9F.ipynb, the one that produced
prod150_eval9f_v1) can judge them unchanged: new RUN_ID, EXISTING_ROOTS = [out_root],
SOURCE_LINEAGE = None, then MANIFEST, GPU_EVALUATE and CPU_EVALUATE. demo/build_demo_json.py then joins the
judge results with the stroke order into the JSON the demo page loads.

This prunes strokes from finished 256-stroke drawings. It is not the same as
asking CLIPDraw to draw with fewer strokes (the stroke sweep).

Usage, after clipdraw.ipynb Sections 1-4 (so `model` is CLIP ViT-B/32):

    %run -i demo/export_budgets.py
    export_budgets(
        ["/content/drive/MyDrive/ExperimentResults/runs/prod150_v1/images/part1"],
        model,
        out_root="/content/drive/MyDrive/ExperimentResults/demo_inputs/prune_v1",
        limit=1)   # test one word first, then limit=None
"""

import csv
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import clip
import pydiffvg

BUDGETS = (8, 16, 32, 64, 128, 256)
CANVAS = 224
BATCH = 32
ORDER_METHOD = "clip_vit_b32_leave_one_out_score_drop"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pydiffvg.set_device(device)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)


# ---------- rendering (matches clipdraw.ipynb render_scene: 224 px, 2x2 samples, white background) ----------

def load_svg(path):
    _, _, shapes, groups = pydiffvg.svg_to_scene(str(path))
    return shapes, groups


def render_hwc(shapes, groups, idx, seed=0):
    sub_shapes = [shapes[i] for i in idx]
    sub_groups = [pydiffvg.ShapeGroup(shape_ids=torch.tensor([j]), fill_color=None,
                                      stroke_color=groups[i].stroke_color)
                  for j, i in enumerate(idx)]
    scene = pydiffvg.RenderFunction.serialize_scene(CANVAS, CANVAS, sub_shapes, sub_groups)
    rgba = pydiffvg.RenderFunction.apply(CANVAS, CANVAS, 2, 2, seed, None, *scene)
    return rgba[:, :, 3:4] * rgba[:, :, :3] + (1 - rgba[:, :, 3:4])


# ---------- stroke importance ----------

@torch.no_grad()
def _clip_scores(clip_model, images_hwc, text_feat, normalize):
    x = torch.stack(images_hwc).permute(0, 3, 1, 2).clamp(0, 1)
    if normalize:
        x = (x - CLIP_MEAN) / CLIP_STD
    f = clip_model.encode_image(x).float()
    f = f / f.norm(dim=-1, keepdim=True)
    return (f @ text_feat.T).squeeze(1).tolist()


@torch.no_grad()
def stroke_order(shapes, groups, clip_model, prompt, normalize):
    """Stroke indices, most important first, plus the score drop for each stroke."""
    t = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
    t = t / t.norm(dim=-1, keepdim=True)
    n = len(shapes)
    full = _clip_scores(clip_model, [render_hwc(shapes, groups, range(n))], t, normalize)[0]
    drops = []
    for start in range(0, n, BATCH):
        imgs = [render_hwc(shapes, groups, [j for j in range(n) if j != i])
                for i in range(start, min(n, start + BATCH))]
        drops += [full - s for s in _clip_scores(clip_model, imgs, t, normalize)]
    order = sorted(range(n), key=lambda i: -drops[i])
    return order, drops, full


# ---------- export ----------

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_drawings(part_folder):
    """(folder, word, svg, experiment.json) for each completed word folder."""
    part = Path(part_folder)
    out = []
    for d in sorted(part.iterdir()):
        svg, meta = d / "clipdraw_final.svg", d / "experiment.json"
        if d.is_dir() and re.match(r"\d{3}_", d.name) and svg.exists() and meta.exists():
            info = json.loads(meta.read_text())
            out.append((d, info["word"], svg, info))
    return out


def export_budgets(part_folders, clip_model, out_root, limit=None, budgets=BUDGETS):
    out_root = Path(out_root)
    done = 0
    for part_folder in part_folders:
        if limit is not None and done >= limit:
            break
        part_folder = Path(part_folder)
        part = part_folder.name
        for folder, word, svg, info in find_drawings(part_folder):
            if limit is not None and done >= limit:
                break
            stroke_file = out_root / "strokes" / part / f"{folder.name}.json"
            if stroke_file.exists():          # resume: this word is finished
                print(f"{part}/{folder.name}: already exported, skipping")
                done += 1
                continue
            started = time.perf_counter()
            prompt = info.get("prompt") or info.get("prompt_template", "A drawing of {}.").format(word)
            normalize = bool(info.get("normalize_clip_inputs", False))
            # batch_runner renders clipdraw_final.png with seed = num_iterations; match it
            render_seed = int((info.get("effective") or {}).get("num_iterations", 1000))
            shapes, groups = load_svg(svg)
            order, drops, full_score = stroke_order(shapes, groups, clip_model, prompt, normalize)

            svg_sha = _sha256(svg)
            for n in budgets:
                n = min(n, len(order))
                dest = out_root / f"b{n:03d}" / part / folder.name
                dest.mkdir(parents=True, exist_ok=True)
                # importance picks WHICH strokes; drawing keeps the original layering (index order)
                img = render_hwc(shapes, groups, sorted(order[:n]), seed=render_seed)
                pydiffvg.imwrite(img.detach().cpu(), str(dest / "clipdraw_final.png"), gamma=1.0)
                meta = {
                    "project": info.get("project"),
                    "substrate": info.get("substrate", "vector strokes"),
                    "run_mode": "stroke_prune",
                    "word": word,
                    "prompt": prompt,
                    "stroke_budget": n,
                    "total_strokes": len(order),
                    "stroke_order_method": ORDER_METHOD,
                    "normalize_clip_inputs": normalize,
                    "source_folder": str(folder),
                    "source_svg_sha256": svg_sha,
                    "source_seed": info.get("seed"),
                    "source_effective": info.get("effective"),
                }
                (dest / "experiment.json").write_text(json.dumps(meta, indent=2))
                summary = part_folder / "batch_summary.csv"
                if summary.exists() and not (dest.parent / "batch_summary.csv").exists():
                    shutil.copy(summary, dest.parent / "batch_summary.csv")

            strokes = []
            for i in order:
                strokes.append({
                    "points": [[round(float(x), 2), round(float(y), 2)]
                               for x, y in shapes[i].points.detach().cpu().tolist()],
                    "width": round(float(shapes[i].stroke_width.detach().cpu().reshape(-1)[0]), 2),
                    "color": [round(float(c), 3) for c in groups[i].stroke_color.detach().cpu().clamp(0, 1).tolist()],
                    "score_drop": round(drops[i], 5),
                    "index": i,                     # original paint order
                })
            stroke_file.parent.mkdir(parents=True, exist_ok=True)
            stroke_file.write_text(json.dumps({
                "word": word, "prompt": prompt, "canvas": CANVAS, "part": part, "folder": folder.name,
                "source_svg_sha256": svg_sha, "stroke_order_method": ORDER_METHOD,
                "clip_score_all_strokes": round(full_score, 5), "strokes": strokes,
            }))
            done += 1
            print(f"{part}/{folder.name} ({word}): {len(order)} strokes ranked and "
                  f"{len(budgets)} budgets saved in {time.perf_counter() - started:.1f}s")
    print(f"Finished {done} word(s). In the evaluation notebook set:")
    print(f'  EXISTING_ROOTS = ["{out_root}"]')
    print("  SOURCE_LINEAGE = None   # these are new images, not the prod150 cohort")
