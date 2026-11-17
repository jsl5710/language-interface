"""Download every model in configs/experiment.yaml into DLSD_MODELS_DIR (default: <project>/models).

Usage:
    python scripts/00_download_models.py --dry-run          # list repos and sizes, download nothing
    python scripts/00_download_models.py                    # download all
    python scripts/00_download_models.py --only google/gemma-3-27b-it
Reads HF_TOKEN from the environment (never stored). Resumable: huggingface_hub skips finished files.
"""
import argparse
import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

WEIGHT_PATTERNS = ["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken", "tokenizer*", "*.jinja"]


def model_ids(cfg: dict) -> list[str]:
    ids = [m["id"] for g in cfg["models"].values() for m in g]
    ids += [c["id"] for c in cfg["classifiers"]]
    ids.append(cfg["judge"]["model"])
    ids += [j["id"] for j in cfg.get("switch_judges", {}).get("panel", [])]
    ids.append(cfg["rendering"]["translation"]["model"])
    return list(dict.fromkeys(ids))  # unique, keep order


def repo_size_gb(api: HfApi, repo: str, token: str | None) -> float | None:
    try:
        info = api.model_info(repo, files_metadata=True, token=token)
    except Exception as e:  # gated without access, missing, offline
        print(f"  cannot read {repo}: {type(e).__name__}")
        return None
    total = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    return total / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = Path(os.environ.get("DLSD_MODELS_DIR", "models"))
    token = os.environ.get("HF_TOKEN")
    ids = args.only or model_ids(cfg)
    api = HfApi()

    total = 0.0
    for repo in ids:
        gb = repo_size_gb(api, repo, token)
        total += gb or 0
        print(f"{repo}: {gb:.1f} GB safetensors" if gb is not None else f"{repo}: size unknown")
        if args.dry_run:
            continue
        dest = root / repo.replace("/", "__")
        snapshot_download(repo, local_dir=dest, token=token, allow_patterns=WEIGHT_PATTERNS,
                          ignore_patterns=["consolidated*", "original/*", "metal/*"])  # duplicate weight copies (Mistral, gpt-oss)
        print(f"  saved to {dest}")
    print(f"total (readable repos): {total:.1f} GB -> {root}")


if __name__ == "__main__":
    main()
