"""Resolve checkpoint / cache locations: local paths are returned as-is, `hf://<user>/<repo>/<path>`
is downloaded from the Hugging Face Hub (cached under HF_HOME) on first use.

    fetch("hf://gehaozhang/gam-libero-pro/checkpoints/gam_policy_libero_pro.pt")  -> local file
    fetch("hf://gehaozhang/gam-libero-pro/cache/")                                -> local directory
    fetch("hf://datasets/gehaozhang/gam-libero-pro-assets/")                      -> a dataset repo
"""
from __future__ import annotations

import os


def is_hub(path) -> bool:
    return isinstance(path, str) and path.startswith("hf://")


def fetch(path: str, repo_type: str = "model") -> str:
    if not is_hub(path):
        return os.path.expanduser(path)
    parts = path[len("hf://"):].split("/")
    if parts[0] in ("datasets", "spaces"):                   # hf://datasets/<user>/<repo>/<path>
        repo_type = parts[0][:-1]; parts = parts[1:]
    assert len(parts) >= 2, f"expected hf://[datasets/]<user>/<repo>[/<path>], got {path!r}"
    repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
    from huggingface_hub import hf_hub_download, snapshot_download
    if sub in ("", "/"):                                     # the whole repository
        return _retry(lambda: snapshot_download(repo, repo_type=repo_type))
    if sub.endswith("/"):                                    # a directory (e.g. a training cache)
        root = _retry(lambda: snapshot_download(repo, repo_type=repo_type, allow_patterns=[sub + "*"]))
        return os.path.join(root, sub.rstrip("/"))
    return _retry(lambda: hf_hub_download(repo, sub, repo_type=repo_type))


def _retry(download, tries: int = 30, wait: float = 60.0):
    """A cache is thousands of files: anonymous downloads hit the Hub's request rate limit (HTTP 429) part way
    through. Downloads resume where they stopped, so wait and continue. Logging in (`huggingface-cli login`)
    raises the limit."""
    import time
    from huggingface_hub.errors import HfHubHTTPError
    for k in range(tries):
        try:
            return download()
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) != 429 or k == tries - 1:
                raise
            print(f"[hub] rate limited (429), resuming in {wait:.0f}s ({k + 1}/{tries}); `huggingface-cli login` raises the limit", flush=True)
            time.sleep(wait)
