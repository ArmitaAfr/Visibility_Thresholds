# Phase 6.5: standalone local dataset validation

This standard-library utility reads the files already present in a selected
processed directory. It does not import the builder or judges, rebuild files,
repair data, integrate a vocabulary, access main/remotes or modify notebooks.
It cannot certify that local data matches an updated main branch.

From the repository root:

```powershell
python -m dataset.validate_dataset --processed-dir dataset/processed --output judge_results/dataset_validation.json
```

JSON is always printed; `--output` optionally saves the same summary to a **new
.json file outside the processed directory**. Existing reports are never
overwritten. Exit code 0 means complete and valid; 1 means invalid or incomplete;
invalid CLI arguments/output paths return 2. Use a fresh report filename to rerun.
The Python API `validate_dataset(processed_dir)` returns the same dictionary.
No pandas, GPU, model weights or network access are needed.

## Actual local schema and checks

The local noun, sample and split files currently have the same 38-column schema.
Core required metadata is `word`, `length`, `pos`, `pos_share`, `score_bin`,
`glasgow_imageability_mean`, `glasgow_imageability_sd` and
`glasgow_imageability_n`. The validator checks these fields' values and reports
blank cells in every column. Blank optional/non-core norm columns are warnings;
they are not treated as complete required metadata or silently omitted. Other
norm dimensions are not re-derived or scientifically revalidated.

Required inputs are:

- `cleaned_words.csv`: may contain non-noun parts of speech.
- `noun_words.csv`: all rows must have `pos=Noun`.
- `sampled_words.csv`: 150 rows and 150 unique words by default, all nouns.
- `noun_words_split/1.csv` through `6.csv`: all nouns; splits 1/2 are low,
  3/4 medium, 5/6 high, following the existing builder's contract.

Words are matched by strip/lowercase, with duplicate canonical matches reported
as errors, including case/whitespace variants. Literal words such as `null` are
not converted to missing values. Labels must be the existing `[1, 3]`, `(3, 5]`
and `(5, 7]`; finite imageability means must agree with those recorded intervals.
Missing labels are never inferred. SD must be nonnegative, rater count a positive
integer, word length correct and `pos_share` within [0,1]. Decimal comparison
allows equivalent formatting such as `2`, `2.0` and `2.000` without overlooking
actual rating differences.

Cross-file checks verify:

- No word occurs in multiple splits, and their union equals the noun dataset.
- Every sampled word occurs in the noun dataset; sample size is correct.
- When cleaned data exists, its `pos=Noun` subset equals `noun_words.csv`.
- Required metadata agrees for each shared word across valid files, including
  bin, mean, SD, rater count, part of speech, length and POS share.

The summary includes per-file row/unique-word/group counts, columns, SHA-256s,
missing-value counts, duplicate lists, full issues and relationship check states.
Missing files have null counts rather than being represented as empty datasets.
Checks depending on unavailable/unreadable schemas are explicitly `blocked`.
Cross-file consistency distinguishes the files actually compared from skipped
files; an available-files pass is not full validation of missing inputs. Group
counts count labeled rows, not deduplicated words; invalid scores/duplicates are
reported separately and prevent a valid overall result.

`--expected-sample-size N` supports alternate local fixtures/builds; the default
remains 150. This changes only validation expectations, never any dataset file.

## Current feature-branch availability

At implementation, `processed/cleaned_words.csv` was **not available locally**.
It is not assumed to have the README's historical contents or row count. The
validator therefore reports the local dataset as incomplete, validates the
eight available noun/sample/split files, and blocks cleaned-dependent checks.
No pull, merge, cherry-pick, rebuild or copying from main is performed. The
updated dataset on main has not been inspected or certified by this utility.

## CPU-only tests

Run the existing judge suite and the new standalone synthetic dataset suite:

```powershell
python -m unittest discover -s judges/tests -v
python -m unittest discover -s dataset/tests -v
```

Use the existing test environment containing judge dependencies for the first
command so adapter tests run rather than being skipped. Dataset tests need only
the standard library. They cover missing files, blocked checks, overlap/union,
sample membership/counts, duplicates, invalid metadata, mismatched scores,
non-nouns, unreadable CSVs, numeric formatting, read-only inputs and output safety.
