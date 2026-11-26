"""Synthesis tests on temporary fixture runs (no GPU).

Judge tables are prod150's real judge rows, re-keyed to the fixture PNGs, so these
tests need PROD150_COMBINED (see test_cohort.py); without it they are skipped.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from clipag_cohort import cohort, synthesize
from clipag_cohort.tests.test_cohort import pinned_words_bytes

PROD150_COMBINED = os.environ.get("PROD150_COMBINED")


def write_png(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(root, words, kind, seed_base, people=15):
    """Fixture run in prod150 layout. kind: prod150 | clipag_strokes | clipag_pixels."""
    hashes = {}
    for person in range(1, people + 1):
        for i, w in enumerate(cohort.assignment(words, person, people), 1):
            folder = root / f"part{person}" / cohort.folder_name(i, w["word"])
            folder.mkdir(parents=True)
            hashes[w["word"]] = write_png(folder / "clipdraw_final.png", seed_base + len(hashes))
            if kind == "prod150":  # batch_runner.py schema: no guidance field
                meta = {"word": w["word"], "prompt": f"A drawing of {w['word']}.", "substrate": "vector strokes",
                        "seed": 0, "effective": {"num_iterations": 1000}, "elapsed_seconds": 180.0, "final_loss": -1.2}
            elif kind == "clipag_strokes":
                meta = {"word": w["word"], "prompt": f"A drawing of {w['word']}.", "substrate": "vector strokes",
                        "seed": 0, "effective": {"num_iterations": 1000}, "elapsed_seconds": 170.0, "final_loss": -1.1,
                        "guidance": {"guidance_model": "clipag"},
                        "cohort": {"run_id": "clipag150_v1", "guidance_model": "clipag"}}
            else:
                settings = cohort.pixel_settings(run_id="clipag150px_v1", guidance_model="clipag", run_mode="full",
                                                 subset="all", people=people, recipe={"steps": 100})
                meta = cohort.pixel_record(word_row=w, index=i, settings=settings, person=person, source_commit="x",
                                           prompt=f"A drawing of {w['word']}.", seed=0, effective={"steps": 100},
                                           guidance={"guidance_model": "clipag"}, environment={},
                                           elapsed_seconds=30.0, best_step=90, best_global_loss=0.6, steps_run=100)
            (folder / "experiment.json").write_text(json.dumps(meta))
    return hashes


@unittest.skipUnless(PROD150_COMBINED, "set PROD150_COMBINED to prod150's combined.csv")
class SynthesizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.words = cohort.load_words(pinned_words_bytes())
        cls.prod = pd.read_csv(PROD150_COMBINED)
        cls.h_clip = make_run(root / "prod150", cls.words, "prod150", 0)
        control = cohort.select_words(cls.words, "control15")
        cls.h_ag = make_run(root / "clipag", cls.words, "clipag_strokes", 1000)
        cls.h_px = make_run(root / "pixels", control, "clipag_pixels", 2000, people=1)
        cls.conditions = {
            "clip_strokes": {"root": root / "prod150", "guidance": "clip", "substrate": "strokes"},
            "clipag_strokes": {"root": root / "clipag", "guidance": "clipag", "substrate": "strokes"},
            "clipag_pixels": {"root": root / "pixels", "guidance": "clipag", "substrate": "pixels"},
        }
        judged_clip = cls.prod.copy()
        judged_clip["image_sha256"] = judged_clip.target_word.map(cls.h_clip)
        # CLIPAG fixture: prod150's judge rows with 10 misses turned into hits.
        judged_ag = cls.prod.copy()
        misses = judged_ag.index[judged_ag.siglip2_target_rank > 1][:10]
        judged_ag.loc[misses, "siglip2_target_rank"] = 1
        judged_ag["image_sha256"] = judged_ag.target_word.map(cls.h_ag)
        cls.judges = [("prod150", judged_clip), ("clipag", judged_ag)]
        cls.out = root / "out"
        cls.index, cls.summary = synthesize.synthesize(cls.conditions, cls.words, cls.out, cls.judges)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_folder_contents(self):
        for name in ("index.csv", "wide.csv", "summary.json", "SUMMARY.md",
                     "grids/overview_low.png", "grids/words/high/dome.png",
                     "images/clip_strokes/dome.png", "metadata/clipag_strokes/dome.json"):
            self.assertTrue((self.out / name).is_file(), name)
        self.assertEqual(len(list((self.out / "judge_input" / "clipag_pixels").iterdir())), 15)
        self.assertFalse(self.out.with_name("out.partial").exists())

    def test_index(self):
        self.assertEqual(len(self.index), 150 + 150 + 15)
        self.assertEqual(self.index.groupby("condition").judge_source.count().to_dict(),
                         {"clip_strokes": 150, "clipag_pixels": 0, "clipag_strokes": 150})
        dome = self.index[(self.index.word == "dome") & (self.index.condition == "clip_strokes")].iloc[0]
        self.assertEqual(dome.imageability_group, "high")
        self.assertFalse(dome.guidance_recorded)  # prod150 records do not name the guidance model

    def test_summary_numbers(self):
        c = self.summary["conditions"]
        self.assertAlmostEqual(c["clip_strokes"]["top1"], 122 / 150)
        self.assertEqual(c["clip_strokes"]["top1_by_group"], {"low": [39, 50], "medium": [44, 50], "high": [39, 50]})
        self.assertAlmostEqual(c["clipag_strokes"]["top1"], 132 / 150)
        self.assertEqual(len(c["clipag_pixels"]["missing_words"]), 135)
        self.assertEqual(self.summary["comparisons"]["clipag_pixels vs clip_strokes"], "not judged yet")
        p1 = self.summary["comparisons"]["clipag_strokes vs clip_strokes"]["primary"]["top1_recognition"]
        self.assertEqual((p1["gained"], p1["lost"]), (10, 0))

    def test_refuses_existing_output(self):
        with self.assertRaises(FileExistsError):
            synthesize.synthesize(self.conditions, self.words, self.out)

    def test_rejects_mismatched_runs(self):
        wrong = dict(self.conditions["clipag_strokes"], guidance="clip")
        with self.assertRaises(ValueError):
            synthesize.discover("x", wrong, self.words)
        wrong = dict(self.conditions["clipag_pixels"], substrate="strokes")
        with self.assertRaises(ValueError):
            synthesize.discover("x", wrong, self.words)

    def test_rejects_judge_word_mismatch(self):
        bad = self.judges[0][1].copy()
        bad["target_word"] = bad.target_word.iloc[::-1].values
        with self.assertRaises(ValueError):
            synthesize.collect(self.conditions, self.words, [("bad", bad)])


if __name__ == "__main__":
    unittest.main()
