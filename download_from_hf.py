#!/usr/bin/env python3
"""
Download datasets/ and results/ from Hugging Face Hub.

  python download_from_hf.py
  python download_from_hf.py --datasets-dir ./datasets_test --results-dir ./results_test
  python download_from_hf.py --folders datasets

Uses HF_TOKEN env var or --token.
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, login, snapshot_download

REPO_ID = "galkesten/plms_repeats_circuits"
PROJECT_ROOT = Path(__file__).resolve().parent
FOLDERS = ["datasets", "results"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download datasets/ and results/ from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"HuggingFace dataset repo (default: {REPO_ID})",
    )
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Output path for datasets/ (default: PROJECT_ROOT/datasets)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Output path for results/ (default: PROJECT_ROOT/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files (default: skip them)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=FOLDERS,
        choices=FOLDERS,
        help=f"Which folders to download (default: {FOLDERS})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def ensure_authenticated(api: HfApi, token: str | None):
    """Make sure we have a valid HF token (optional for public repos)."""
    if token:
        api.token = token
        return
    api.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if api.token:
        return
    try:
        api.whoami()
        return
    except Exception:
        pass
    print("Not logged in to HuggingFace. Launching interactive login...")
    login()


def copy_tree(src: Path, dst: Path, force: bool) -> tuple[int, int]:
    """Recursively copy files from src to dst. Returns (copied, skipped) counts."""
    copied = 0
    skipped = 0
    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        if dst_file.exists() and not force:
            skipped += 1
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1
    return copied, skipped


def main():
    args = parse_args()
    api = HfApi()
    ensure_authenticated(api, args.token)
    try:
        user = api.whoami().get("name", "anonymous")
        print(f"Authenticated as: {user}")
    except Exception:
        print("Using anonymous access (public repo)")

    # Resolve output dirs
    output_dirs = {
        "datasets": Path(args.datasets_dir) if args.datasets_dir else PROJECT_ROOT / "datasets",
        "results": Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results",
    }

    allow_patterns = [f"{folder}/**" for folder in args.folders]

    print(f"Downloading snapshot from {args.repo_id} ...")
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            token=args.token or os.environ.get("HF_TOKEN"),
        )
    )
    print(f"Snapshot cached at: {snapshot_path}")

    total_copied = 0
    total_skipped = 0

    for folder in args.folders:
        src = snapshot_path / folder
        dst = output_dirs[folder]
        if not src.is_dir():
            print(f"WARNING: {folder}/ not found in repo, skipping.")
            continue

        print(f"Copying {folder}/ -> {dst}")
        copied, skipped = copy_tree(src, dst, force=args.force)
        total_copied += copied
        total_skipped += skipped
        print(f"  {folder}/: {copied} copied, {skipped} skipped")

    print(f"\nDone! {total_copied} files copied, {total_skipped} files skipped.")
    if total_skipped > 0 and not args.force:
        print("  (use --force to overwrite existing files)")


if __name__ == "__main__":
    main()
