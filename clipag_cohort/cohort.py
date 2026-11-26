"""Helpers for running CLIPAG-guided CLIPDraw on the 150-word cohort.

Everything here is plain Python, so it can be tested without a GPU. The Colab
notebook `clipag_cohort.ipynb` imports this module and supplies the drawing code
from `clipdraw_clipag.ipynb`.

Outputs use the same layout as the prod150 generation, so the existing judge
pipeline can read them unchanged:

    <run>/images/part<P>/<NNN>_<word>/clipdraw_final.png
                                      clipdraw_final.svg
                                      initial.svg
                                      experiment.json   (has a "word" field)
                                      loss_history.json
"""

import hashlib
import csv
import io
import json
import os
import random
import re
from pathlib import Path

# The 150 nouns, pinned to the commit and bytes used for prod150.
SAMPLED_WORDS_COMMIT = "4717920918c529a4d99456eb598ca98abb822b19"
SAMPLED_WORDS_PATH = "dataset/processed/sampled_words.csv"
SAMPLED_WORDS_SHA256 = "a5412cf7effab6344628c09cab5beb3701e49cf7eaa7879d9709fc9ccc0799fc"

# prod150_v1 settings (unified_colab.py and clipdraw.ipynb at f0417c4). CLIPAG
# runs must match these so each word pairs with its prod150 CLIP drawing.
PROD150 = {
    "run_id": "prod150_v1",
    "generation_source_commit": "f0417c45d276ae1a2d61650343f93346a4d8e7a1",
    "prompt_template": "A drawing of {}.",
    "seed": 0,
    "num_paths": 256,
    "num_iterations": 1000,
    "num_augmentations": 4,
    "max_stroke_width": 50.0,
    "use_negative_prompts": False,
    "normalize_clip_inputs": False,
}

GROUPS = {"[1, 3]": "low", "(3, 5]": "medium", "(5, 7]": "high"}
SUBSETS = ("all", "control15")


def file_sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_words(data, expected_sha256=SAMPLED_WORDS_SHA256):
    """Parse sampled_words.csv bytes into [{word, group, imageability}] in file order."""
    if file_sha256(data) != expected_sha256:
        raise ValueError("sampled_words.csv differs from the pinned prod150 word list.")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    words = [{"word": r["word"], "group": GROUPS[r["score_bin"]],
              "imageability": float(r["glasgow_imageability_mean"])} for r in rows]
    names = [w["word"] for w in words]
    if len(names) != 150 or len(set(names)) != 150:
        raise ValueError(f"Expected 150 unique words, found {len(set(names))}.")
    if any(not re.fullmatch(r"[a-z]+", n) for n in names):
        raise ValueError("Every word must be lowercase letters only.")
    return words


def select_words(words, subset):
    """'all' keeps the 150 words. 'control15' keeps 5 per imageability group (fixed seed).

    control15 is for re-running standard CLIP through this code path, to check it
    reproduces prod150 before reading CLIPAG differences as guidance effects.
    """
    if subset == "all":
        return list(words)
    if subset == "control15":
        keep = set()
        for group in ("low", "medium", "high"):
            names = sorted(w["word"] for w in words if w["group"] == group)
            keep.update(random.Random(0).sample(names, 5))
        return [w for w in words if w["word"] in keep]
    raise ValueError(f"subset must be one of {SUBSETS}.")


def assignment(words, person, people):
    """Same split as prod150: person P of N takes every Nth word starting at P."""
    if type(person) is not int or type(people) is not int or not 1 <= person <= people:
        raise ValueError("Expected 1 <= PERSON <= NUM_PEOPLE.")
    return words[person - 1::people]


def slugify(text, max_len=60):
    """Identical to batch_runner._slugify, so folder names match prod150."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return (slug or "prompt")[:max_len]


def folder_name(index, word):
    return f"{index:03d}_{slugify(word)}"


def run_settings(*, run_id, guidance_model, run_mode, subset, people):
    """Everything that must stay fixed across every session of one run."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("RUN_ID may use letters, digits, underscores and hyphens only.")
    if guidance_model not in ("clip", "clipag"):
        raise ValueError('GUIDANCE_MODEL must be "clip" or "clipag".')
    if run_mode not in ("full", "smoke_test"):
        raise ValueError('RUN_MODE must be "full" or "smoke_test".')
    if subset not in SUBSETS:
        raise ValueError(f"SUBSET must be one of {SUBSETS}.")
    return {"run_id": run_id, "guidance_model": guidance_model, "run_mode": run_mode,
            "subset": subset, "num_people": people, "paired_baseline": dict(PROD150),
            "words_commit": SAMPLED_WORDS_COMMIT, "words_sha256": SAMPLED_WORDS_SHA256}


def check_or_write_json(path, payload):
    """Write payload once; later sessions must find the identical payload."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"{path} was written with different settings. Use a new RUN_ID "
                             "or restore the original settings.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return True


def plan_part(words, part_dir, guidance_model):
    """Return [(index, word_row, folder, done)] for one person's part.

    A folder counts as done only if it holds experiment.json and a nonempty PNG
    whose record names the same word and guidance model. Anything else in the
    part folder stops the run, so a damaged part is never silently mixed in.
    """
    part_dir = Path(part_dir)
    expected = {folder_name(i, w["word"]): (i, w) for i, w in enumerate(words, 1)}
    if part_dir.exists():
        for child in part_dir.iterdir():
            if child.name.startswith("."):
                continue
            if child.name not in expected:
                raise ValueError(f"Unexpected item in part folder: {child}")
            meta, png = child / "experiment.json", child / "clipdraw_final.png"
            if not meta.is_file() or not png.is_file() or png.stat().st_size == 0:
                raise ValueError(f"Incomplete result folder; inspect or remove it: {child}")
            info = json.loads(meta.read_text(encoding="utf-8"))
            _, w = expected[child.name]
            if info.get("word") != w["word"] or info.get("cohort", {}).get("guidance_model") != guidance_model:
                raise ValueError(f"Result folder does not match this run: {child}")
    return [(i, w, part_dir / name, (part_dir / name / "experiment.json").is_file())
            for name, (i, w) in expected.items()]


def cohort_record(base, *, word_row, index, settings, person, source_commit):
    """Add the fields the judge pipeline and the paired analysis need.

    `base` is the experiment.json written by clipdraw_clipag.ipynb's export_run.
    """
    if base["prompt"] != PROD150["prompt_template"].format(word_row["word"]):
        raise ValueError("Prompt differs from the prod150 template.")
    if base["seed"] != PROD150["seed"] or base["normalize_clip_inputs"] or base["use_negative_prompts"]:
        raise ValueError("Seed, normalization or negative prompts differ from prod150.")
    if base["guidance"]["guidance_model"] != settings["guidance_model"]:
        raise ValueError("Guidance model differs from the run settings.")
    if settings["run_mode"] == "full":
        requested = base["requested"]
        for key in ("num_paths", "num_iterations", "num_augmentations", "max_stroke_width"):
            if requested[key] != PROD150[key]:
                raise ValueError(f"{key} differs from prod150.")
    record = dict(base)
    record.update(
        word=word_row["word"],
        prompt_template=PROD150["prompt_template"],
        cohort={"run_id": settings["run_id"], "guidance_model": settings["guidance_model"],
                "subset": settings["subset"], "person": person, "num_people": settings["num_people"],
                "index_in_part": index, "imageability_group": word_row["group"],
                "glasgow_imageability_mean": word_row["imageability"],
                "words_commit": SAMPLED_WORDS_COMMIT, "words_sha256": SAMPLED_WORDS_SHA256,
                "source_commit": source_commit},
        paired_baseline=dict(PROD150),
    )
    return record


def publish(staging_dir, final_dir):
    """Move a finished result into place in one step, so a crash never leaves half a folder."""
    staging_dir, final_dir = Path(staging_dir), Path(final_dir)
    if final_dir.exists():
        raise FileExistsError(f"Result already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    # A rename is atomic only within one filesystem; the notebook keeps staging
    # inside the run folder on Drive. A copy fallback could leave half a folder.
    os.rename(staging_dir, final_dir)
