"""Build isolated Linux judge interpreters; never install ML packages into the kernel."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

from .checkpoints import atomic_write, file_digest

PYTHON_VERSION = "3.12.10"
UV_VERSION = "0.13.5"
CUDA_INDEX = "https://download.pytorch.org/whl/cu124"


def environment_spec(model, repo):
    names = {"siglip2": ["requirements-siglip2-colab.txt", "requirements-siglip2.txt"],
             "florence2": ["requirements-florence2.txt"]}
    if model not in names:
        raise ValueError("Unknown judge model.")
    return {"model": model, "python": PYTHON_VERSION, "uv": UV_VERSION,
            "torch": "2.6.0", "torchvision": "0.21.0", "cuda_index": CUDA_INDEX,
            "requirements": {n: file_digest(Path(repo) / "judges" / n) for n in names[model]}}


def install_commands(uv, python, requirements):
    return [[uv, "pip", "install", "--python", str(python), "torch==2.6.0", "torchvision==0.21.0",
             "--index-url", CUDA_INDEX],
            [uv, "pip", "install", "--python", str(python), "-r", str(requirements)]]


def validate_lock(lock, spec):
    if (lock.get("spec") != spec or not isinstance(lock.get("freeze"), str)
            or hashlib.sha256(lock["freeze"].encode()).hexdigest() != lock.get("freeze_sha256")):
        raise ValueError("Environment lock/spec is changed or damaged. Preserve it; use a new output/lock root for a new environment.")


def setup_environment(model, repo, env_root, lock_root):
    if platform.system() != "Linux":
        raise RuntimeError("This setup targets Linux Colab. Use the existing local dependency instructions on Windows.")
    uv = shutil.which("uv")
    if uv is None or not subprocess.check_output([uv, "--version"], text=True).startswith(f"uv {UV_VERSION} "):
        raise RuntimeError(f"Install the launcher only into the Colab kernel: python -m pip install uv=={UV_VERSION}")
    repo, lock_root = Path(repo), Path(lock_root)
    spec = environment_spec(model, repo)
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{model}.json"
    lock = None
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        validate_lock(lock, spec)
    env_dir = Path(env_root) / model
    python = env_dir / "bin/python"
    if not env_dir.exists():
        subprocess.run([uv, "venv", "--python", PYTHON_VERSION, str(env_dir)], check=True)
    requirement = repo / "judges" / ("requirements-siglip2-colab.txt" if model == "siglip2" else "requirements-florence2.txt")
    if lock is not None:
        requirement = env_dir / "restore-freeze.txt"
        requirement.write_text(lock["freeze"], encoding="utf-8")
    for command in install_commands(uv, python, requirement):
        subprocess.run(command, check=True)
    freeze = subprocess.check_output([uv, "pip", "freeze", "--python", str(python)], text=True)
    if lock is not None and freeze != lock["freeze"]:
        raise ValueError("Rebuilt environment differs from saved package lock. Stop and inspect before inference.")
    if lock is None:
        atomic_write(lock_path, {"spec": spec, "freeze": freeze,
                                "freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest()})
    subprocess.run([str(python), "-c", "from judges.scaling import cuda_preflight; cuda_preflight()"],
                   cwd=repo, check=True)
    print(f"Ready: {python}")
    return python


def main(argv=None):
    parser = argparse.ArgumentParser(description="Isolated Python/CUDA environments for the judge Colab notebook.")
    parser.add_argument("--model", choices=["siglip2", "florence2"], required=True)
    parser.add_argument("--env-root", type=Path, default=Path("/content/judge_envs"))
    parser.add_argument("--lock-root", type=Path, required=True)
    args = parser.parse_args(argv)
    setup_environment(args.model, Path(__file__).resolve().parents[1], args.env_root, args.lock_root)


if __name__ == "__main__":
    main()
