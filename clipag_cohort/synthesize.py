"""Collect every generation run (conditions) into one folder with a single index.

A condition is one run folder of 150 (or fewer) word results in the prod150 layout:
    <root>/part<P>/<NNN>_<word>/clipdraw_final.png + experiment.json

`synthesize()` writes:
    images/<condition>/<word>.png (+ .svg)    every image, flat by word
    metadata/<condition>/<word>.json          its experiment.json, unchanged
    judge_input/<condition>/<NNN>_<word>/     PNG + experiment.json, unchanged, for ONE judge run
                                              over everything (the judge's `part` column = condition)
    index.csv                                 one row per image: condition, word, imageability,
                                              generation stats, judge results if supplied
    wide.csv                                  one row per word, key columns per condition
    grids/words/<group>/<word>.png            all conditions side by side for one word
    grids/overview_<group>.png                words x conditions contact sheet
    summary.json, SUMMARY.md                  completeness, recognition by group, paired comparisons

Judge results are optional. Pass any number of judge `analysis/combined.csv` files; rows
are matched to images by PNG SHA-256, so it does not matter which evaluation run judged what.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from . import compare as cmp

SUBSTRATES = {"strokes", "pixels"}
GUIDANCE = {"clip", "clipag"}
JUDGE_COLUMNS = ["siglip2_target_rank", "siglip2_top1_prediction", "siglip2_top5_predictions",
                 "siglip2_target_score", "florence2_original_caption", "florence2_mentions_apparent_text",
                 "ocr_strings", "text_level_pred", "text_score", "zs_p_text", "target_sim", "target_partial"]
GROUP_ORDER = ["low", "medium", "high"]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recorded_substrate(meta):
    value = meta.get("substrate", "")
    if value == "pixels":
        return "pixels"
    if "stroke" in value:
        return "strokes"
    return None


def _recorded_guidance(meta):
    return (meta.get("guidance") or {}).get("guidance_model") or (meta.get("cohort") or {}).get("guidance_model")


def discover(name, spec, words):
    """Return one row per image in a condition. Stops on duplicates, foreign words or mismatches."""
    root = Path(spec["root"])
    if spec["guidance"] not in GUIDANCE or spec["substrate"] not in SUBSTRATES:
        raise ValueError(f"{name}: guidance must be one of {GUIDANCE}, substrate one of {SUBSTRATES}.")
    if not root.is_dir():
        raise FileNotFoundError(f"{name}: folder not found: {root}")
    info = {w["word"]: w for w in words}
    rows, seen = [], {}
    for meta_path in sorted(root.rglob("experiment.json")):
        if any(part.startswith(".") or part == "staging" for part in meta_path.relative_to(root).parts):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        word = str(meta.get("word", "")).strip().lower()
        png = meta_path.with_name("clipdraw_final.png")
        if not png.is_file():
            raise ValueError(f"{name}: {meta_path.parent} has experiment.json but no clipdraw_final.png.")
        if word not in info:
            raise ValueError(f"{name}: {meta_path} is for {word!r}, which is not one of the 150 words.")
        if word in seen:
            raise ValueError(f"{name}: two results for {word!r}: {seen[word]} and {meta_path.parent}")
        seen[word] = meta_path.parent
        substrate, guidance = _recorded_substrate(meta), _recorded_guidance(meta)
        if substrate and substrate != spec["substrate"]:
            raise ValueError(f"{name}: {meta_path} records substrate {substrate!r}, expected {spec['substrate']!r}.")
        if guidance and guidance != spec["guidance"]:
            raise ValueError(f"{name}: {meta_path} records guidance {guidance!r}, expected {spec['guidance']!r}.")
        effective = meta.get("effective") or {}
        cohort = meta.get("cohort") or {}
        rows.append({
            "condition": name, "guidance": spec["guidance"], "substrate": spec["substrate"],
            "guidance_recorded": guidance is not None,
            "word": word, "imageability_group": info[word]["group"],
            "glasgow_imageability_mean": info[word]["imageability"],
            "run_id": cohort.get("run_id", spec.get("run_id", "")),
            "part": meta_path.parent.parent.name, "folder": meta_path.parent.name,
            "prompt": meta.get("prompt"), "seed": meta.get("seed"),
            "run_mode": meta.get("run_mode"),
            "steps": effective.get("num_iterations", effective.get("steps")),
            "elapsed_seconds": meta.get("elapsed_seconds"), "final_loss": meta.get("final_loss"),
            "png_sha256": sha256_file(png), "source_dir": str(meta_path.parent),
        })
    return rows


def merge_judges(index, judge_tables):
    """Attach judge columns by PNG SHA-256. Earlier tables win if an image was judged twice."""
    index = index.copy()
    for col in JUDGE_COLUMNS + ["judge_source"]:
        index[col] = None
    if not judge_tables:
        return index
    by_hash = {}
    for source, table in judge_tables:
        if "image_sha256" not in table.columns:
            raise ValueError(f"{source} has no image_sha256 column; is it an analysis/combined.csv?")
        for row in table.to_dict("records"):
            by_hash.setdefault(row["image_sha256"], (source, row))
    for i, row in index.iterrows():
        hit = by_hash.get(row["png_sha256"])
        if hit is None:
            continue
        source, judged = hit
        if str(judged.get("target_word", "")).strip().lower() != row["word"]:
            raise ValueError(f"{source}: image {row['png_sha256'][:12]} judged as {judged.get('target_word')!r}, "
                             f"but it is {row['word']!r} in {row['condition']}.")
        for col in JUDGE_COLUMNS:
            index.at[i, col] = judged.get(col)
        index.at[i, "judge_source"] = source
    return index


def collect(conditions, words, judge_tables=()):
    rows = []
    for name, spec in conditions.items():
        rows.extend(discover(name, spec, words))
    if not rows:
        raise ValueError("No results found in any condition.")
    index = pd.DataFrame(rows)
    index = merge_judges(index, list(judge_tables))
    order = {name: i for i, name in enumerate(conditions)}
    group = {g: i for i, g in enumerate(GROUP_ORDER)}
    index["_c"] = index["condition"].map(order)
    index["_g"] = index["imageability_group"].map(group)
    return index.sort_values(["_g", "word", "_c"]).drop(columns=["_c", "_g"]).reset_index(drop=True)


def judged_frame(index, condition):
    sub = index[(index.condition == condition) & index.siglip2_target_rank.notna()].copy()
    if sub.empty:
        return None
    sub = sub.rename(columns={"word": "target_word"})
    for col in ("siglip2_target_rank", "text_level_pred"):
        sub[col] = sub[col].astype(int)
    for col in ("siglip2_target_score", "text_score"):
        sub[col] = sub[col].astype(float)
    return cmp.prepare(sub, condition)


def summarize(index, conditions, words, reference=None, comparisons=None):
    all_words = {w["word"] for w in words}
    out = {"conditions": {}, "comparisons": {}}
    for name, spec in conditions.items():
        sub = index[index.condition == name]
        judged = sub[sub.siglip2_target_rank.notna()]
        entry = {"guidance": spec["guidance"], "substrate": spec["substrate"], "root": str(spec["root"]),
                 "images": int(len(sub)), "missing_words": sorted(all_words - set(sub.word)),
                 "judged": int(len(judged)),
                 "median_elapsed_seconds": float(sub.elapsed_seconds.dropna().median()) if sub.elapsed_seconds.notna().any() else None}
        if len(judged):
            rank = judged.siglip2_target_rank.astype(int)
            entry["top1"] = float((rank == 1).mean())
            entry["top5"] = float((rank <= 5).mean())
            entry["top1_by_group"] = {g: [int((judged[judged.imageability_group == g].siglip2_target_rank.astype(int) == 1).sum()),
                                          int((judged.imageability_group == g).sum())] for g in GROUP_ORDER}
            entry["legible_word_rate"] = float((judged.text_level_pred.astype(int) >= cmp.LEGIBLE_LEVEL).mean())
        out["conditions"][name] = entry
    if comparisons is None and reference in conditions:
        comparisons = [(reference, name) for name in conditions if name != reference]
    for base, treat in comparisons or []:
        try:
            b, t = judged_frame(index, base), judged_frame(index, treat)
            if b is None or t is None:
                out["comparisons"][f"{treat} vs {base}"] = "not judged yet"
                continue
            out["comparisons"][f"{treat} vs {base}"] = cmp.compare(b, t)
        except ValueError as exc:
            out["comparisons"][f"{treat} vs {base}"] = f"not comparable: {exc}"
    return out


def wide_table(index, conditions):
    keep = ["siglip2_target_rank", "siglip2_top1_prediction", "text_level_pred", "text_score"]
    base = index.drop_duplicates("word")[["word", "imageability_group", "glasgow_imageability_mean"]].set_index("word")
    for name in conditions:
        sub = index[index.condition == name].set_index("word")
        for col in keep:
            base[f"{name}__{col}"] = sub[col]
        base[f"{name}__image"] = pd.Series({w: f"images/{name}/{w}.png" for w in sub.index})
    return base.reset_index()


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _caption(row):
    if pd.isna(row.get("siglip2_target_rank")):
        return ""
    rank = int(row["siglip2_target_rank"])
    text = f"rank {rank}" if rank > 1 else "recognized"
    if rank > 1 and isinstance(row.get("siglip2_top1_prediction"), str):
        text += f" ({row['siglip2_top1_prediction']})"
    return text


def word_strip(rows, conditions, out_path, cell=224):
    """One word: every condition side by side, labelled, with the judge's verdict if known."""
    from PIL import Image, ImageDraw
    head, foot, pad = 28, 22, 8
    names = list(conditions)
    canvas = Image.new("RGB", (pad + len(names) * (cell + pad), head + cell + foot + pad), "white")
    draw = ImageDraw.Draw(canvas)
    small = _font(13)
    by_cond = {r["condition"]: r for r in rows}
    for i, name in enumerate(names):
        x = pad + i * (cell + pad)
        draw.text((x, 6), name, fill=(20, 20, 20), font=small)
        r = by_cond.get(name)
        if r is None:
            draw.rectangle([x, head, x + cell - 1, head + cell - 1], outline=(200, 200, 200))
            draw.text((x + 8, head + cell // 2 - 6), "not generated", fill=(150, 150, 150), font=small)
            continue
        with Image.open(r["image_path"]) as im:
            canvas.paste(im.convert("RGB").resize((cell, cell)), (x, head))
        draw.text((x, head + cell + 4), _caption(r), fill=(60, 60, 60), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def overview(index, conditions, group, out_path, cell=112):
    """Contact sheet for one imageability group: words down, conditions across."""
    from PIL import Image, ImageDraw
    sub = index[index.imageability_group == group]
    words = sorted(sub.word.unique())
    names = list(conditions)
    label_w, head, pad = 130, 24, 4
    canvas = Image.new("RGB", (label_w + len(names) * (cell + pad), head + len(words) * (cell + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(12)
    for j, name in enumerate(names):
        draw.text((label_w + j * (cell + pad), 6), name, fill=(20, 20, 20), font=font)
    for i, word in enumerate(words):
        y = head + i * (cell + pad)
        draw.text((6, y + cell // 2 - 6), word, fill=(20, 20, 20), font=font)
        for j, name in enumerate(names):
            r = sub[(sub.word == word) & (sub.condition == name)]
            if r.empty:
                continue
            with Image.open(r.iloc[0]["image_path"]) as im:
                canvas.paste(im.convert("RGB").resize((cell, cell)), (label_w + j * (cell + pad), y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _pct(x):
    return f"{100 * x:.1f}%"


def summary_markdown(summary):
    lines = ["# Synthesis summary", "", "## Conditions", "",
             "| Condition | Guidance | Substrate | Images | Missing | Judged | Top-1 | Low / medium / high top-1 | Legible word | Median seconds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in summary["conditions"].items():
        groups = " / ".join(f"{k}/{n}" for k, n in c["top1_by_group"].values()) if "top1_by_group" in c else "—"
        top1 = _pct(c["top1"]) if "top1" in c else "—"
        legible = _pct(c["legible_word_rate"]) if "legible_word_rate" in c else "—"
        seconds = f"{c['median_elapsed_seconds']:.0f}" if c["median_elapsed_seconds"] is not None else "—"
        lines.append(f"| {name} | {c['guidance']} | {c['substrate']} | {c['images']} | {len(c['missing_words'])} | "
                     f"{c['judged']} | {top1} | {groups} | {legible} | {seconds} |")
    lines += ["", "## Paired comparisons", ""]
    if not summary["comparisons"]:
        lines.append("None requested.")
    for label, r in summary["comparisons"].items():
        lines.append(f"### {label}")
        lines.append("")
        if isinstance(r, str):
            lines.append(r)
        else:
            base, treat = label.split(" vs ")[1], label.split(" vs ")[0]
            lines.append("```")
            lines.append(cmp.report(r, [base, treat]))
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def synthesize(conditions, words, out_dir, judge_tables=(), reference="clip_strokes", comparisons=None,
               make_grids=True, make_judge_input=True):
    """Write the combined folder. Refuses to overwrite an existing one."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists. Choose a new output name.")
    index = collect(conditions, words, judge_tables)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    image_paths = []
    for row in index.to_dict("records"):
        src = Path(row["source_dir"])
        img = tmp / "images" / row["condition"] / f"{row['word']}.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "clipdraw_final.png", img)
        if (src / "clipdraw_final.svg").is_file():
            shutil.copy2(src / "clipdraw_final.svg", img.with_suffix(".svg"))
        meta = tmp / "metadata" / row["condition"] / f"{row['word']}.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "experiment.json", meta)
        if make_judge_input:
            dst = tmp / "judge_input" / row["condition"] / row["folder"]
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "clipdraw_final.png", dst / "clipdraw_final.png")
            shutil.copy2(src / "experiment.json", dst / "experiment.json")
        if sha256_file(img) != row["png_sha256"]:
            raise RuntimeError(f"Copy of {img} differs from its source.")
        image_paths.append(str(img))
    index["image_path"] = image_paths
    if make_grids:
        records = index.to_dict("records")
        for word in index.word.unique():
            rows = [r for r in records if r["word"] == word]
            word_strip(rows, conditions, tmp / "grids" / "words" / rows[0]["imageability_group"] / f"{word}.png")
        for group in GROUP_ORDER:
            if (index.imageability_group == group).any():
                overview(index, conditions, group, tmp / "grids" / f"overview_{group}.png")
    index["image_path"] = [str(Path(p).relative_to(tmp)) for p in image_paths]
    index.to_csv(tmp / "index.csv", index=False)
    wide_table(index, conditions).to_csv(tmp / "wide.csv", index=False)
    summary = summarize(index, conditions, words, reference, comparisons)
    (tmp / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (tmp / "SUMMARY.md").write_text(summary_markdown(summary))
    tmp.rename(out_dir)
    return index, summary
