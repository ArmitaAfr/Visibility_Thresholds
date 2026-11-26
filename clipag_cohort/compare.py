"""Paired comparison of two judged runs of the same words (e.g. prod150 CLIP vs CLIPAG).

Each input is an `analysis/combined.csv` from the judge pipeline. Rows are paired
by target word; only words present in both runs are compared.

    python3 clipag_cohort/compare.py BASELINE.csv TREATMENT.csv \
        --labels CLIP CLIPAG --json comparison.json

The three primary outcomes are fixed in advance (see README) and Holm-corrected
together. Everything else is reported as secondary / descriptive.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

GROUPS = ["low", "medium", "high"]
LEGIBLE_LEVEL = 3  # text_level_pred >= 3: OCR read at least half of the target word
NEEDED = ["target_word", "imageability_group", "glasgow_imageability_mean", "siglip2_target_rank",
          "siglip2_target_score", "text_level_pred", "text_score", "florence2_mentions_apparent_text"]


def load(path):
    return prepare(pd.read_csv(path), str(path))


def prepare(df, name="table"):
    """Index judged rows by word and add the 0/1 outcome columns."""
    df = df.copy()
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"{name} lacks columns {missing}; is it an analysis/combined.csv?")
    path = name
    df["target_word"] = df["target_word"].str.strip().str.lower()
    if df["target_word"].duplicated().any():
        raise ValueError(f"{path} has more than one image for some words; pairing needs one per word.")
    df["top1"] = (df["siglip2_target_rank"] == 1).astype(int)
    df["legible"] = (df["text_level_pred"] >= LEGIBLE_LEVEL).astype(int)
    df["florence_text"] = df["florence2_mentions_apparent_text"].astype(str).str.lower().eq("true").astype(int)
    return df.set_index("target_word")


def pair(base, treat):
    words = sorted(set(base.index) & set(treat.index))
    if not words:
        raise ValueError("The two runs share no target words.")
    b, t = base.loc[words], treat.loc[words]
    if (b["imageability_group"] != t["imageability_group"]).any():
        raise ValueError("Imageability groups disagree between runs for the same word.")
    return b, t, words


def mcnemar(b, t):
    """Exact McNemar test on paired 0/1 outcomes."""
    gained = int(((b == 0) & (t == 1)).sum())
    lost = int(((b == 1) & (t == 0)).sum())
    n = gained + lost
    p = 1.0 if n == 0 else float(stats.binomtest(gained, n, 0.5).pvalue)
    return {"base_rate": float(b.mean()), "treat_rate": float(t.mean()),
            "difference": float(t.mean() - b.mean()), "gained": gained, "lost": lost, "p": p}


def boot_diff_ci(b, t, rng, reps=10000):
    """Percentile CI for the paired difference in means, resampling words."""
    d = np.asarray(t, float) - np.asarray(b, float)
    idx = rng.integers(0, len(d), (reps, len(d)))
    return [float(x) for x in np.percentile(d[idx].mean(axis=1), [2.5, 97.5])]


def wilcoxon(b, t):
    d = np.asarray(t, float) - np.asarray(b, float)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"base_median": float(np.median(b)), "treat_median": float(np.median(t)),
            "median_difference": float(np.median(d)), "p": p}


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def imageability_effect(df):
    """Top-1 by group within one run, plus the same two-sided tests used for prod150."""
    by_group = {g: [int(df.loc[df.imageability_group == g, "top1"].sum()),
                    int((df.imageability_group == g).sum())] for g in GROUPS}
    table = np.array([[k, n - k] for k, n in by_group.values()])
    chi2_p = float(stats.chi2_contingency(table).pvalue) if (table.sum(0) > 0).all() else 1.0
    rho, p = stats.spearmanr(df["glasgow_imageability_mean"], df["siglip2_target_score"])
    return {"top1_by_group": by_group, "chi2_p": chi2_p,
            "spearman_imageability_vs_score": [float(rho), float(p)]}


def compare(base, treat, seed=0):
    rng = np.random.default_rng(seed)
    b, t, words = pair(base, treat)
    out = {"n_pairs": len(words), "n_base_only": len(set(base.index) - set(words)),
           "n_treat_only": len(set(treat.index) - set(words))}

    # Primary outcomes (pre-registered; Holm across these three).
    p1 = mcnemar(b["top1"], t["top1"])
    p1["difference_ci"] = boot_diff_ci(b["top1"], t["top1"], rng)
    p2 = mcnemar(b["legible"], t["legible"])
    p2["difference_ci"] = boot_diff_ci(b["legible"], t["legible"], rng)
    delta = t["siglip2_target_score"].values - b["siglip2_target_score"].values
    if np.ptp(delta) == 0:  # no change at all: no association to test
        rho, p = 0.0, 1.0
    else:
        rho, p = stats.spearmanr(b["glasgow_imageability_mean"], delta)
    p3 = {"spearman_imageability_vs_score_change": float(rho), "p": float(p),
          "median_score_change_by_group": {g: float(np.median(delta[(b.imageability_group == g).values]))
                                           for g in GROUPS if (b.imageability_group == g).any()}}
    adjusted = holm([p1["p"], p2["p"], p3["p"]])
    for result, a in zip((p1, p2, p3), adjusted):
        result["p_holm"] = a
    out["primary"] = {"top1_recognition": p1, "legible_target_word": p2, "imageability_x_guidance": p3}

    # Secondary / descriptive.
    out["secondary"] = {
        "judge_target_score": wilcoxon(b["siglip2_target_score"], t["siglip2_target_score"]),
        "judge_target_rank": wilcoxon(b["siglip2_target_rank"], t["siglip2_target_rank"]),
        "text_score": wilcoxon(b["text_score"], t["text_score"]),
        "florence_mentions_text": mcnemar(b["florence_text"], t["florence_text"]),
        "text_level_counts": {"base": b["text_level_pred"].value_counts().sort_index().to_dict(),
                              "treat": t["text_level_pred"].value_counts().sort_index().to_dict()},
        "top1_agreement": float((b["top1"] == t["top1"]).mean()),
        "score_correlation": float(stats.spearmanr(b["siglip2_target_score"], t["siglip2_target_score"])[0]),
        "imageability_base": imageability_effect(b),
        "imageability_treat": imageability_effect(t),
        "top1_change_by_group": {g: mcnemar(b.loc[b.imageability_group == g, "top1"],
                                            t.loc[t.imageability_group == g, "top1"]) for g in GROUPS
                                 if (b.imageability_group == g).any()},
    }
    out["changed_words"] = {
        "now_recognized": sorted(w for w in words if b.at[w, "top1"] == 0 and t.at[w, "top1"] == 1),
        "no_longer_recognized": sorted(w for w in words if b.at[w, "top1"] == 1 and t.at[w, "top1"] == 0),
    }
    return out


def report(r, labels):
    a, c = labels
    pct = lambda x: f"{100 * x:.1f}%"
    pr = r["primary"]
    lines = [f"Paired words: {r['n_pairs']} (only in {a}: {r['n_base_only']}, only in {c}: {r['n_treat_only']})", "",
             "PRIMARY (Holm-corrected across the three)"]
    for key, title in (("top1_recognition", "Top-1 recognition"), ("legible_target_word", "Legible target word (level >= 3)")):
        x = pr[key]
        lines.append(f"  {title}: {a} {pct(x['base_rate'])} -> {c} {pct(x['treat_rate'])} | "
                     f"diff {100 * x['difference']:+.1f} pts [{100 * x['difference_ci'][0]:+.1f}, {100 * x['difference_ci'][1]:+.1f}] | "
                     f"{x['gained']} gained, {x['lost']} lost | McNemar p={x['p']:.3g}, Holm p={x['p_holm']:.3g}")
    x = pr["imageability_x_guidance"]
    lines.append(f"  Imageability x guidance: Spearman rho(imageability, judge score change) = "
                 f"{x['spearman_imageability_vs_score_change']:+.3f}, p={x['p']:.3g}, Holm p={x['p_holm']:.3g}")
    lines.append("    median score change by group: " + ", ".join(
        f"{g} {v:+.2f}" for g, v in x["median_score_change_by_group"].items()))
    s = r["secondary"]
    lines += ["", "SECONDARY (uncorrected, descriptive)"]
    for key, title in (("judge_target_score", "Judge score for target"), ("judge_target_rank", "Judge rank of target"),
                       ("text_score", "Text score")):
        x = s[key]
        lines.append(f"  {title}: median {x['base_median']:.3f} -> {x['treat_median']:.3f} "
                     f"(paired median change {x['median_difference']:+.3f}), Wilcoxon p={x['p']:.3g}")
    x = s["florence_mentions_text"]
    lines.append(f"  Florence mentions text: {pct(x['base_rate'])} -> {pct(x['treat_rate'])}, McNemar p={x['p']:.3g}")
    lines.append(f"  Same top-1 outcome in both runs: {pct(s['top1_agreement'])}; "
                 f"judge-score correlation between runs: rho={s['score_correlation']:.3f}")
    for label, key in ((a, "imageability_base"), (c, "imageability_treat")):
        x = s[key]
        groups = ", ".join(f"{g} {k}/{n}" for g, (k, n) in x["top1_by_group"].items())
        lines.append(f"  Imageability within {label}: top-1 {groups}; chi2 p={x['chi2_p']:.3g}; "
                     f"rho(imageability, score)={x['spearman_imageability_vs_score'][0]:+.3f}")
    lines += ["", f"Now recognized with {c}: {', '.join(r['changed_words']['now_recognized']) or 'none'}",
              f"No longer recognized with {c}: {', '.join(r['changed_words']['no_longer_recognized']) or 'none'}"]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("baseline")
    parser.add_argument("treatment")
    parser.add_argument("--labels", nargs=2, default=["CLIP", "CLIPAG"])
    parser.add_argument("--json", help="also write the full results to this JSON file")
    args = parser.parse_args(argv)
    result = compare(load(args.baseline), load(args.treatment))
    print(report(result, args.labels))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
