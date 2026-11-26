"""Local tests (no GPU). Run from the repo root:  python3 -m unittest discover clipag_cohort/tests

The prod150 checks need the evaluation export's combined.csv; point PROD150_COMBINED
at it, or those tests are skipped.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from clipag_cohort import cohort, compare

REPO = Path(__file__).resolve().parents[2]
PROD150_COMBINED = os.environ.get("PROD150_COMBINED")


def pinned_words_bytes():
    return subprocess.check_output(
        ["git", "show", f"{cohort.SAMPLED_WORDS_COMMIT}:{cohort.SAMPLED_WORDS_PATH}"], cwd=REPO)


def base_experiment(word, guidance="clipag", **changes):
    base = {"prompt": f"A drawing of {word}.", "seed": 0, "normalize_clip_inputs": False,
            "use_negative_prompts": False, "guidance": {"guidance_model": guidance},
            "requested": {"num_paths": 256, "num_iterations": 1000, "num_augmentations": 4,
                          "max_stroke_width": 50.0}}
    base.update(changes)
    return base


class WordsTest(unittest.TestCase):
    def setUp(self):
        self.words = cohort.load_words(pinned_words_bytes())

    def test_pinned_list(self):
        self.assertEqual(len(self.words), 150)
        self.assertEqual({w["group"] for w in self.words}, {"low", "medium", "high"})
        with self.assertRaises(ValueError):
            cohort.load_words(pinned_words_bytes() + b"\n")

    def test_control15(self):
        a = cohort.select_words(self.words, "control15")
        b = cohort.select_words(self.words, "control15")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 15)
        for g in ("low", "medium", "high"):
            self.assertEqual(sum(w["group"] == g for w in a), 5)

    def test_assignment_partitions(self):
        parts = [cohort.assignment(self.words, p, 4) for p in range(1, 5)]
        names = [w["word"] for part in parts for w in part]
        self.assertEqual(sorted(names), sorted(w["word"] for w in self.words))
        with self.assertRaises(ValueError):
            cohort.assignment(self.words, 5, 4)

    @unittest.skipUnless(PROD150_COMBINED, "set PROD150_COMBINED to prod150's combined.csv")
    def test_reproduces_prod150_folders(self):
        prod = pd.read_csv(PROD150_COMBINED)
        actual = {(r.part, r.folder) for r in prod.itertuples()}
        matches = []
        for people in range(1, 31):
            planned = {(f"part{p}", cohort.folder_name(i, w["word"]))
                       for p in range(1, people + 1)
                       for i, w in enumerate(cohort.assignment(self.words, p, people), 1)}
            if planned == actual:
                matches.append(people)
        self.assertEqual(len(matches), 1, f"team sizes reproducing prod150's folders: {matches}")


class PartTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.words = cohort.load_words(pinned_words_bytes())[:3]
        self.settings = cohort.run_settings(run_id="t1", guidance_model="clipag", run_mode="full",
                                            subset="all", people=1)

    def tearDown(self):
        self.tmp.cleanup()

    def finish(self, index, word, guidance="clipag"):
        folder = self.root / "part1" / cohort.folder_name(index, word)
        folder.mkdir(parents=True)
        (folder / "clipdraw_final.png").write_bytes(b"png")
        (folder / "experiment.json").write_text(json.dumps({"word": word, "cohort": {"guidance_model": guidance}}))

    def test_plan_and_resume(self):
        plan = cohort.plan_part(self.words, self.root / "part1", "clipag")
        self.assertEqual([p[3] for p in plan], [False, False, False])
        self.finish(1, self.words[0]["word"])
        plan = cohort.plan_part(self.words, self.root / "part1", "clipag")
        self.assertEqual([p[3] for p in plan], [True, False, False])

    def test_rejects_foreign_or_partial_results(self):
        self.finish(1, self.words[0]["word"], guidance="clip")
        with self.assertRaises(ValueError):
            cohort.plan_part(self.words, self.root / "part1", "clipag")

    def test_rejects_incomplete_folder(self):
        (self.root / "part1" / cohort.folder_name(2, self.words[1]["word"])).mkdir(parents=True)
        with self.assertRaises(ValueError):
            cohort.plan_part(self.words, self.root / "part1", "clipag")

    def test_settings_locked(self):
        path = self.root / "cohort_settings.json"
        self.assertTrue(cohort.check_or_write_json(path, self.settings))
        self.assertFalse(cohort.check_or_write_json(path, self.settings))
        changed = dict(self.settings, guidance_model="clip")
        with self.assertRaises(ValueError):
            cohort.check_or_write_json(path, changed)

    def test_record_checks_prod150_match(self):
        w = self.words[0]
        record = cohort.cohort_record(base_experiment(w["word"]), word_row=w, index=1,
                                      settings=self.settings, person=1, source_commit="abc")
        self.assertEqual(record["word"], w["word"])
        self.assertEqual(record["cohort"]["guidance_model"], "clipag")
        bad = [base_experiment(w["word"], prompt=f"A drawing of a {w['word']}."),
               base_experiment(w["word"], seed=1),
               base_experiment(w["word"], guidance="clip"),
               base_experiment(w["word"], requested={"num_paths": 16, "num_iterations": 1000,
                                                     "num_augmentations": 4, "max_stroke_width": 50.0})]
        for base in bad:
            with self.assertRaises(ValueError):
                cohort.cohort_record(base, word_row=w, index=1, settings=self.settings, person=1, source_commit="abc")

    def test_publish_is_one_step(self):
        staging = self.root / "staging" / "x"
        staging.mkdir(parents=True)
        (staging / "experiment.json").write_text("{}")
        final = self.root / "part1" / "001_a"
        cohort.publish(staging, final)
        self.assertTrue((final / "experiment.json").is_file())
        self.assertFalse(staging.exists())
        staging.mkdir()
        with self.assertRaises(FileExistsError):
            cohort.publish(staging, final)


@unittest.skipUnless(PROD150_COMBINED, "set PROD150_COMBINED to prod150's combined.csv")
class CompareTest(unittest.TestCase):
    def setUp(self):
        self.base = compare.load(PROD150_COMBINED)

    def test_identical_runs(self):
        r = compare.compare(self.base, self.base.copy())
        self.assertEqual(r["n_pairs"], 150)
        p = r["primary"]["top1_recognition"]
        self.assertEqual((p["gained"], p["lost"], p["p"]), (0, 0, 1.0))
        self.assertEqual(r["secondary"]["top1_agreement"], 1.0)

    def test_detects_known_change(self):
        treat = self.base.copy()
        misses = treat.index[treat["top1"] == 0][:20]
        treat.loc[misses, "siglip2_target_rank"] = 1
        treat.loc[misses, "top1"] = 1
        r = compare.compare(self.base, treat)
        p = r["primary"]["top1_recognition"]
        self.assertEqual((p["gained"], p["lost"]), (20, 0))
        self.assertAlmostEqual(p["treat_rate"] - p["base_rate"], 20 / 150)
        self.assertLess(p["p"], 1e-5)
        self.assertEqual(len(r["changed_words"]["now_recognized"]), 20)

    def test_partial_overlap_pairs_only_shared_words(self):
        r = compare.compare(self.base, self.base.iloc[:15].copy())
        self.assertEqual((r["n_pairs"], r["n_base_only"]), (15, 135))


if __name__ == "__main__":
    unittest.main()
