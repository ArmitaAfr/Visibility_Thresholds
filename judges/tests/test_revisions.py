"""Revision verification tests never contact Hugging Face or download weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from judges.revisions import resolve_model_revision


class RevisionTests(unittest.TestCase):
    def download(self, **kwargs):
        return resolve_model_revision("google/siglip2-base-patch16-224", "main", **kwargs)

    def test_branch_resolves_to_snapshot_sha(self):
        download = Mock(return_value=str(Path("cache/models--google--siglip2/snapshots") / ("a" * 40) / "config.json"))
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            self.assertEqual(self.download(), "a" * 40)
        download.assert_called_once_with(repo_id="google/siglip2-base-patch16-224",
                                          filename="config.json", revision="main",
                                          cache_dir=None, local_files_only=False)

    def test_local_cache_snapshot_and_no_symlink_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "models--google--siglip2/snapshots" / ("b" * 40)
            snapshot.mkdir(parents=True)
            config = snapshot / "config.json"
            config.write_text('{}')
            download = Mock(return_value=str(config))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}), \
                 patch.object(Path, "resolve", side_effect=AssertionError("Do not follow cache blob symlinks")):
                self.assertEqual(self.download(cache_dir=directory, local_files_only=True), "b" * 40)
            self.assertTrue(download.call_args.kwargs["local_files_only"])
            self.assertEqual(download.call_args.kwargs["cache_dir"], directory)

    def test_explicit_sha_verified_against_cached_snapshot(self):
        download = Mock(return_value=str(Path("cache/snapshots") / ("c" * 40) / "config.json"))
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            self.assertEqual(resolve_model_revision("repo", "c" * 40), "c" * 40)
            with self.assertRaisesRegex(ValueError, "Requested commit"):
                resolve_model_revision("repo", "d" * 40)

    def test_unknown_or_invalid_revision_provides_actionable_error(self):
        for outcome in (OSError("not cached and offline"), "cache/blobs/hash", "cache/snapshots/main/config.json"):
            download = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
            with self.subTest(outcome=outcome), \
                 patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
                with self.assertRaisesRegex(ValueError, "--revision.*40-character.*--local-files-only"):
                    self.download(local_files_only=True)


if __name__ == "__main__":
    unittest.main()
