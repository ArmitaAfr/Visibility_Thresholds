# Comparing saved judge outputs (Phase 4)

This analysis uses only Python's standard library. It reads saved SigLIP2 and
Florence-2 JSON exports and original source artifacts for identity verification;
it never initializes a model or reruns inference. Model predictions and
target-hit rates are **not actual human recognition**. SigLIP2 compatibility
scores are not calibrated probabilities.

## Generate the existing three-image comparison

Exact PowerShell command:

```powershell
Set-Location "C:\Users\orjue\Documents\Visibility_Thresholds"
python -m judges.analyze --siglip2 "judge_results/siglip2_test.json" --florence2 "judge_results/florence2_test.json" --output-dir "judge_results/comparison_test"
```

An example report at that path was generated during implementation. To rerun,
choose a new directory, such as `judge_results/comparison_test_v2`. Existing
reports are never overwritten. No inference dependencies or GPU are required.

## Pilot and larger evaluations

The existing pilot exports cover the same 35 image IDs:

```powershell
python -m judges.analyze --siglip2 "judge_results/siglip2_pilot.json" --florence2 "judge_results/florence2_pilot.json" --output-dir "judge_results/pilot35_analysis"
```

The 35-image report was also generated during implementation; use a new output
directory to rerun. For future cohorts, supply the corresponding saved exports;
this command does not create missing model results. Multiple exports per model
can be supplied after each flag to combine nonoverlapping parts. Duplicate image IDs are rejected even if
the duplicated rows look identical; remove duplicate inputs explicitly.

The default requires identical image-ID sets. If one export is incomplete,
`--allow-unmatched` explicitly analyzes their intersection, lists all omitted IDs
and records input counts and the exact matched denominator. An empty
intersection always fails. Reported SigLIP2 metrics use only matched images.

## Source identity and input validation

- Records join on stable `image_id`, never target words, filename prefixes or
  row order. Multiple images of the same intended word remain separate samples.
- Source PNGs and `experiment.json` must remain available. Phase 1 discovery
  recomputes each ID from canonical metadata and actual PNG bytes. Both models'
  exported IDs and intended words must agree with these sources. Exported image
  paths must also contain the same bytes as the verified metadata's sibling PNG.
  Changed pixels, metadata, wrong source paths, or missing artifacts stop analysis.
- If source files were moved, pass `--source-root "C:\new\batch_results"`
  (multiple roots are supported). Discovery verifies the new artifacts against
  the saved IDs, and the CSV uses their current paths. This is a content-verified
  relocation map, not a skip-verification option. Extra unrelated artifacts in
  those roots still follow normal strict discovery validation.
- Source identity uses the existing Phase 1 hash contract. No image decoding is
  needed. The analysis checks saved identity against current artifacts, not
  cryptographic attestation of what a model saw at an earlier time.
- All supplied rows need valid model/configuration fields and finite,
  nonnegative inference times. Each model must use one configuration profile
  across the supplied exports; only actual `image_batch_size` is ignored because
  final partial batches legitimately differ. Model revisions, prompts, library
  versions or other conditions must be analyzed separately.
- SigLIP2 must contain its complete ranked candidate scores. Vocabulary size,
  hash and actual candidate set must remain constant. Top1/top5, target rank and
  target score are checked against that ranking. Duplicate candidates, nonfinite
  scores, unsorted ranks or inconsistent exported values cause explicit errors.
- Florence captions are taken verbatim from `raw_output.caption`; the separate
  normalized caption is copied from `normalized_prediction`. Neither caption
  is converted to a candidate word. Caption ranks/scores must be null.

## Outputs

The command creates a new directory containing:

| File | Contents |
| --- | --- |
| `comparison.csv` | One row per matched image, both model outputs and inference times. |
| `summary.json` | Exact metrics, denominator, vocabulary size/hash, input file hashes, model/configuration identifiers, unmatched IDs and complete heuristic definitions. |
| `report.md` | Human-readable metrics and caption-pattern summary with interpretation limits. |

CSV columns include image ID, intended word, verified image path and SHA-256,
SigLIP2 top1/top5 words, target rank/score, Florence original/normalized captions,
apparent-text mention flag and matched cues, object/abstract-concept cues, and
both inference times. Lists use JSON inside CSV cells, so captions containing
commas, quotes or newlines remain correctly quoted and recoverable.

Metrics are descriptive over the matched cohort, with a 1-based target rank:

- Top-1 target-hit rate: number with target rank 1 / matched image count.
- Top-5 target-hit rate: number with target rank at most 5 / matched image count
  (or at most vocabulary size for synthetic vocabularies smaller than 5).
- Mean target rank: arithmetic mean of exported, validated target ranks.

The vocabulary size is explicitly reported; the official experiment currently
uses 150 candidate words. These are per-image model retrieval statistics, not
human recognisability measures or benchmark accuracy claims. Multiple images
of one word contribute multiple samples; no word-level balancing is assumed.

## Florence caption patterns: descriptive heuristics

`patterns.py` contains versioned, explicit lexical lists for apparent text,
objects and abstract-concept cues. Matching is case-insensitive and uses whole
tokens. The report counts captions containing at least one cue and the number
of captions mentioning each cue. Categories overlap. This is a transparent
summary of wording rather than comprehensive linguistic or semantic analysis.

Examples: “a glass with the word CAT written on it” has object and writing cues;
“an abstract symbol of freedom” has abstract-concept cues. “No text is visible”
still **mentions** text and is flagged: this flag does not assert text exists.
Words such as “sign” or “letter” can also be ambiguous. Absence of a cue does not
establish absence of objects, writing or concepts. This is **not verified OCR**,
independent text detection, or proof that the image contains writing. Captions
may describe objects or words that are not present in the image.

## Reproducibility and future annotations

Rows are sorted by image ID. Summaries record analysis/heuristic versions,
source export SHA-256s, model names and configuration hashes. No timestamps are
inserted, so unchanged inputs at unchanged paths produce identical reports.
Whole captions and raw input exports are preserved for later review.

The modules separate export validation/joins (`compare.py`), caption lexical
patterns (`patterns.py`) and CLI (`judges/analyze.py`). Later imageability
categories or independent text-presence results can join by `image_id`, with
`image_sha256` available for additional verification. Neither is inferred here;
heuristic text mentions must remain distinct from future verified text-presence
labels. Human study outcomes should likewise be separate annotations.

CPU-only synthetic fixture tests:

```powershell
python -m unittest discover -s judges/tests -p test_analysis.py -v
```

## Phase 6: imageability-stratified analysis

Stratification is opt-in; omitting `--imageability-metadata` preserves the
existing overall analysis and CSV schema. With the flag, `comparison.csv` is
enriched, `summary.json` retains all overall fields and adds `imageability`, and
`report.md` retains the overall report followed by two three-category tables.
Existing output directories are never overwritten.

The actual `dataset/processed/sampled_words.csv` schema contains:

| Source column | Usage |
| --- | --- |
| `word` | Canonical word join after stripping whitespace and lowercasing. |
| `score_bin` | Existing imageability label: `[1, 3]` low, `(3, 5]` medium, `(5, 7]` high. |
| `glasgow_imageability_mean` | Original Glasgow imageability score; required and carried into the CSV. |
| `glasgow_imageability_sd`, `glasgow_imageability_n` | Original rating uncertainty/rater count, carried through when available. |

The current official metadata has 150 unique words, 50 in each recorded bin.
The labels and interval meanings are reused from the dataset builder and its
README; no quantiles, new thresholds or classifications are inferred. Scores
are checked against their **recorded** intervals for consistency. A missing
label is an error even when its score might suggest a bin.

Regenerate the 35-image pilot into a fresh output directory:

```powershell
Set-Location "C:\Users\orjue\Documents\Visibility_Thresholds"
python -m judges.analyze --siglip2 "judge_results/siglip2_pilot.json" --florence2 "judge_results/florence2_pilot.json" --imageability-metadata "dataset/processed/sampled_words.csv" --output-dir "judge_results/pilot35_imageability_v2"
```

The implementation run produced `judge_results/pilot35_imageability/` with
`comparison.csv`, `summary.json` and `report.md`. Use another new directory for
subsequent regenerations. No inference or GPU is needed. Existing multi-export,
source relocation and explicit unmatched-export behavior remain supported.

Join validation is strict and occurs before report files are created:

- Every evaluated canonical target must match exactly one metadata record. No
  fuzzy matching, stemming, guessing from captions or first/last-match choice is
  allowed. Missing targets are all listed in `ImageabilityMetadataError.issues`.
- Duplicate canonical words are rejected even for identical rows; conflicting
  ratings/categories are reported as ambiguous. Case/whitespace variants cannot
  create multiple matches. Duplicate CSV headers are also rejected.
- Required columns, nonblank canonical words, recognized source labels and
  finite, interval-consistent mean scores are required. All supplied metadata
  records are validated, including unused words, rather than silently removing
  incomplete rows. Present SD/count columns must contain valid values. If those
  optional columns are entirely absent, the audit explicitly lists them and
  their enriched CSV values are null; category and mean score are never optional.
- An entire category absent from the metadata is recorded explicitly. If no
  evaluated targets require it, its report row still appears. An evaluated
  target absent from metadata is an error, not an excluded observation.
- All three categories always appear. Empty groups have zero counts, status
  `empty`, and JSON null / Markdown N/A rates and rank statistics, not fabricated
  zeros. The audit reports empty evaluated groups separately from categories
  absent from the metadata file. There is no skip-missing-metadata option.

Added CSV columns are `canonical_target_word`, `imageability_metadata_word`,
`imageability_group`, `score_bin`, `glasgow_imageability_mean`,
`glasgow_imageability_sd` and `glasgow_imageability_n`. Original captions and
existing comparison columns remain unchanged.

`summary.imageability.groups.low/medium/high` contains image count, unique
canonical target count/list, SigLIP2 top-1/top-5 hits and rates, and mean/median
1-based target rank. It also contains caption counts, fractions and percentages
for the existing apparent-text, object and abstract-concept cue lists. These
are reused Phase 4 lexical heuristics: not verified OCR, semantic recognition,
evidence of internal attention, or human recognizability. Negated mentions still
count, categories overlap, and cue absence does not establish visual absence.

**Images remain the observations.** Multiple generated images of one word are
retained and contribute separately to rates and mean/median ranks. Unique word
counts show this repetition; there is no hidden word-level aggregation or
assumption that category image counts are balanced. These are descriptive
summaries only, without causal claims, significance tests or claims about
human recognition.

The metadata path, SHA-256, source column names, label mapping, category word
counts and validation/coverage diagnostics are recorded. The overall and each
group's SigLIP2 vocabulary size/hash stay explicit; existing model identifiers,
configuration hashes and export provenance remain in the summary. The 35-image
pilot is evaluated against 150 candidates, not a 35-word vocabulary. A future
2,000+ **image** cohort may still use 150 candidate words; if the vocabulary
itself changes, its actual exported size/hash will be reported instead. Supply
a matching, complete CSV through the same option for future target vocabularies;
the join is indexed by word and imposes no 150-row or image-count limit.

CPU-only tests cover repeated targets, preserved overall values/captions,
metadata duplicates/conflicts/missing targets, incomplete scores/categories,
source interval boundaries, empty groups, count/rate/median calculations and CLI
output integration:

```powershell
python -m unittest discover -s judges/tests -p test_imageability.py -v
```
