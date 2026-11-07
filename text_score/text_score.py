# Writes to text_scores.csv, one row per image
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rapidfuzz import fuzz


COLUMNS = [
    "part", "folder", "word",
    "ocr_strings", "ocr_max_conf", "ocr_n_boxes", "text_area_frac",
    "target_sim", "target_partial",
    "zs_p_text", "zs_typo_margin",
    "text_level_pred", "text_score",
]

# Positive ("has text") and negative ("no text") prompts, in that order
PROBE = ["an image with written words or letters", "an image with no text or letters"]

SIGLIP = "google/siglip-so400m-patch14-384"


# Lowercases a string and only keeps a-z
def letters(s):
    return re.sub(r"[^a-z]", "", s.lower())


class Scorer:
    def __init__(self, upscale=3):

        # Imported here so the models only load when a Scorer is made
        import easyocr
        from transformers import AutoModel, AutoProcessor

        self.upscale = upscale
        self.ocr = easyocr.Reader(["en"], gpu=False, verbose=False)

        self.model = AutoModel.from_pretrained(SIGLIP).eval()
        self.proc = AutoProcessor.from_pretrained(SIGLIP)

        # Encode the two probe prompts once, reused for every image
        self.probe = self.text(PROBE)

    @torch.no_grad()
    def text(self, prompts):

        # Encodes a list of prompts with SigLIP
        # Returns normalized embeddings, so a dot product is cosine similarity
        inputs = self.proc(text=prompts, padding="max_length", return_tensors="pt")
        features = self.model.get_text_features(**inputs)

        return torch.nn.functional.normalize(features, dim=-1)

    @torch.no_grad()
    def image(self, im):

        # Encodes an image with SigLIP
        # Returns a normalized image embedding
        inputs = self.proc(images=im, return_tensors="pt")
        features = self.model.get_image_features(**inputs)

        return torch.nn.functional.normalize(features, dim=-1)

    def read(self, im):

        # Upscales the image (default 3x) and runs EasyOCR
        # The strokes are tiny at 224px, so OCR misses a lot without this
        s = self.upscale
        big = im.resize((im.width * s, im.height * s), Image.LANCZOS)

        regions = []

        for box, text, conf in self.ocr.readtext(np.array(big)):

            # Scale the box back down to the original image
            xs = [point[0] / s for point in box]
            ys = [point[1] / s for point in box]

            regions.append((text, float(conf), (min(xs), min(ys), max(xs), max(ys))))

        # Typical text height, used to group boxes into lines
        if regions:
            line_height = np.median([box[3] - box[1] for _, _, box in regions]) or 1
        else:
            line_height = 1

        # Reading order: line by line, then left to right
        def reading_order(region):
            x0, y0, x1, y1 = region[2]
            line = round((y0 + y1) / 2 / line_height)
            return (line, x0)

        # Returns each text region as (string, confidence, box), box in original 224px
        return sorted(regions, key=reading_order)

    # Scores one image and returns every column except part and folder
    def __call__(self, png, word):

        im = Image.open(png).convert("RGB")
        width, height = im.size

        regions = self.read(im)
        strings = [text for text, _, _ in regions]
        target = letters(word)

        # Candidate strings: each OCR string, plus runs of 2-3 neighbours joined
        # (OCR often splits one handwritten word into pieces)
        candidates = []

        for i in range(len(strings)):
            for run in (1, 2, 3):
                if i + run <= len(strings):
                    joined = letters("".join(strings[i:i + run]))

                    if joined:
                        candidates.append(joined)

        # Whole-word match
        target_sim = 0.0

        for c in candidates:
            target_sim = max(target_sim, fuzz.ratio(c, target) / 100)

        # Part-of-word match
        # Needs at least half the word's letters, so a lone "P" can't match "pleasure"
        min_length = max(2, math.ceil(len(target) / 2))
        target_partial = 0.0

        for c in candidates:
            if len(c) >= min_length:
                target_partial = max(target_partial, fuzz.partial_ratio(c, target) / 100)

        # Fill in every text box so overlapping boxes aren't counted twice
        mask = np.zeros((height, width), bool)

        for _, _, (x0, y0, x1, y1) in regions:
            mask[int(y0):math.ceil(y1), int(x0):math.ceil(x1)] = True

        # SigLIP: how much the image matches "text" vs "no text"
        img = self.image(im)
        logits = self.model.logit_scale.exp() * img @ self.probe.T + self.model.logit_bias
        p_text = float(torch.softmax(logits, -1)[0, 0])

        # Does the image look more like the word written out, or a drawing of it?
        written, drawn = self.text([f"the word '{word}' written in letters", f"a drawing of {word}"])
        typo_margin = float(img @ written - img @ drawn)

        max_conf = max((conf for _, conf, _ in regions), default=0.0)

        # Letters read, weighted by how confident the OCR was
        ocr_letters = sum(len(letters(text)) * conf for text, conf, _ in regions)

        r = {
            "word": word,
            "ocr_strings": " | ".join(strings),
            "ocr_max_conf": round(max_conf, 4),
            "ocr_n_boxes": len(regions),
            "text_area_frac": round(float(mask.mean()), 4),
            "target_sim": round(target_sim, 3),
            "target_partial": round(target_partial, 3),
            "zs_p_text": round(p_text, 4),
            "zs_typo_margin": round(typo_margin, 4),
        }

        r["text_level_pred"] = predict_level(r, has_text=bool(regions))

        # Average of SigLIP's probability and the OCR letter count squashed to 0-1
        # (about 4 confident letters gives 0.63)
        letter_part = 1 - math.exp(-ocr_letters / 4)
        r["text_score"] = round((r["zs_p_text"] + letter_part) / 2, 4)

        return r


# 0 none, 1 scribbles, 2 letters, 3 part of the word, 4 full word
# Thresholds are rough guesses for now, need tuning against hand labels
def predict_level(r, has_text):

    if r["target_sim"] >= 0.75:
        return 4

    if r["target_partial"] >= 0.75:
        return 3

    if has_text and r["ocr_max_conf"] >= 0.3:
        return 2

    if has_text or r["zs_p_text"] >= 0.25:
        return 1

    return 0


_scorer = None


# For use from other code: loads the models on the first call, then reuses them
def score_image(png_path, target_word):
    global _scorer

    if _scorer is None:
        _scorer = Scorer()

    return _scorer(png_path, target_word)


# A part folder is one that directly holds NNN_word folders
def is_part_folder(folder):
    for d in folder.iterdir():
        if d.is_dir() and re.match(r"\d{3}_", d.name):
            return True

    return False


# Match words on folder name, not index, since indices restart when a part is rerun
def words_by_folder(part):
    summary = part / "batch_summary.csv"
    words = {}

    if summary.exists():
        with summary.open() as f:
            for row in csv.DictReader(f):
                words[Path(row["folder"]).name] = row["word"]

    return words


# Yields (part, folder, word, png) for every image under root
def find_images(root):
    root = Path(root).expanduser()

    # Root can be one part folder or a folder of parts. Smoke tests are skipped
    if is_part_folder(root):
        parts = [root]
    else:
        parts = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("smoke"))

    for part in parts:
        words = words_by_folder(part)

        for png in sorted(part.glob("[0-9][0-9][0-9]_*/clipdraw_final.png")):
            folder = png.parent.name

            # Fall back to the folder name if the summary doesn't list it
            word = words.get(folder, folder.split("_", 1)[1])

            yield part.name, folder, word, png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)  # Repeat to score several parts
    parser.add_argument("--out", default="text_scores.csv")
    parser.add_argument("--upscale", type=int, default=3)
    args = parser.parse_args()

    global _scorer
    _scorer = Scorer(args.upscale)

    rows = []

    for root in args.root:
        for part, folder, word, png in find_images(root):
            row = {"part": part, "folder": folder, **score_image(png, word)}
            rows.append(row)

            print(part, folder, row["text_level_pred"], row["text_score"])

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
