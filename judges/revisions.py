"""Resolve a Hub revision before loading any model/processor weights."""

import re
from pathlib import Path


def is_commit_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def resolve_model_revision(model_id, revision, *, cache_dir=None, local_files_only=False):
    """Read only config.json; its Hub snapshot directory identifies the commit.

    hf_hub_download handles refs, cached snapshots, offline mode and connectivity
    fallback. Do not resolve filesystem symlinks: those point into blobs/, which
    would discard the snapshot's commit identity. No weights are downloaded here.
    """
    from huggingface_hub import hf_hub_download

    try:
        config_path = Path(hf_hub_download(
            repo_id=model_id, filename="config.json", revision=revision,
            cache_dir=cache_dir, local_files_only=local_files_only,
        ))
        commit = config_path.parent.name
        if config_path.parent.parent.name != "snapshots" or not is_commit_sha(commit):
            raise ValueError(f"Config path does not identify a Hub snapshot: {config_path}")
        if is_commit_sha(revision) and commit != revision:
            raise ValueError(f"Requested commit {revision} but resolved {commit}.")
        return commit
    except Exception as exc:
        raise ValueError(
            f"Cannot verify Hugging Face commit for {model_id!r} at {revision!r}: {exc}. "
            "Connect to Hugging Face to cache config.json for this revision, or pass "
            "--revision with the full 40-character commit SHA of an existing cached snapshot. "
            "For offline use, pass --local-files-only and ensure --cache-dir (or HF_HOME) "
            "points to the cache containing that snapshot. No unverified revision will be used."
        ) from exc
