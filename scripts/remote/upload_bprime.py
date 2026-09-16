#!/usr/bin/env python3
"""Upload the frozen B' ACT checkpoint to a private Hugging Face model repo.

Requires HF write token: `hf auth login` or env HF_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default=os.environ.get(
            "BPRIME_CKPT", "/root/autodl-tmp/ckpts/v2/act-coffee-30000"
        ),
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("BPRIME_REPO", "davore/act-aloha-static-coffee-bprime"),
    )
    args = parser.parse_args()
    src = Path(args.src)
    if not (src / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing model.safetensors in {src}")
    api = HfApi()
    create_repo(args.repo, private=True, repo_type="model", exist_ok=True)
    info = api.upload_folder(
        repo_id=args.repo,
        folder_path=str(src),
        repo_type="model",
        commit_message="Freeze B' ACT coffee 30k tracking baseline.",
    )
    sha = getattr(info, "oid", None) or getattr(info, "commit_id", None) or str(info)
    print(json.dumps({"repo": args.repo, "sha": sha}, indent=2))


if __name__ == "__main__":
    main()
