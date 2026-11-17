# Running the manifest text scorer in Colab

Run from the repository root with a Phase 7 schema-v1 manifest created on the same mounted filesystem. The runner checks its CSV fingerprints and requires a nonempty cohort with an empty issues list. Preserve the original images, metadata and CSVs for resume. Give each run a new directory beneath the configured output_root, outside the image inputs.

## Isolated CPU environment

The following is a proposed first setup for Linux x86-64 and Python 3.12. Run it in a shell with uv available. Keep the lock file on persistent storage. These package choices need a real Colab smoke test. Transformers 4.x is deliberate because the scorer consumes tensor-valued get_text_features and get_image_features. No GPU is required.

```bash
ENV=/content/text-score-env
LOCK="/content/drive/MyDrive/text-score-py312-linux.lock"
uv venv --python 3.12.10 "$ENV"
# First setup only: select a new lock filename.
printf '%s\n' torch==2.6.0+cpu torchvision==0.21.0+cpu transformers==4.49.0 easyocr==1.7.2 rapidfuzz sentencepiece protobuf numpy pillow filelock > /content/text-score.in
# The CPU index supplies the PyTorch pair. The resulting lock pins transitives too.
test ! -e "$LOCK" && uv pip compile --python "$ENV/bin/python" --generate-hashes \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  /content/text-score.in -o "$LOCK"
# On reconnect, reuse the saved lock.
uv pip sync --python "$ENV/bin/python" --require-hashes \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match "$LOCK"
```

Set HF_HOME and EASYOCR_MODULE_PATH to dedicated persistent cache directories before starting the process. Use an independently verified full Hugging Face commit for google/siglip-so400m-patch14-384. Both model and processor load at that commit. EasyOCR may download its detector and recognizer on first use. Keep the populated OCR cache for reproducibility. The runner fingerprints its .pth files and records installed package versions.

```bash
"$ENV/bin/python" -m text_score.run_manifest --manifest "$MANIFEST" \
  --output-dir "$RUN_DIR" --revision "$SIGLIP_COMMIT"
```

Default upscale is 3. To resume an interrupted run, repeat the command with --resume. Use one writer per directory. The job uses a file lock and same-directory atomic replacements. Verify locking and replacement behavior on the selected Drive mount before a production run.

run.json binds checkpoints to the manifest digest, model and processor revision, OCR weight hashes, scorer source hash, package versions and scale. checkpoints/ contains one checksummed result per canonical ID. Each attempt checks the source hash, target and canonical metadata-plus-PNG identity, even for cached rows. Scoring receives exactly those verified bytes. A failed attempt writes attempt_errors.json and exits with an error. An interruption can leave just run.json and individual checkpoints. Only results.json marks completion. An earlier attempt_errors.json can remain as a failure history after successful resume.

Join results.json's results array to the judge comparison by image_id and verify image_sha256 and target_word. Each row retains the original scorer columns, including word, part and folder. Historical CSVs have no content hashes. The existing compare.py still compares the two historical CSVs beside it.
