"""
env_loader.py
-------------
Single place that loads the repo-root `.env` so every section reads the SAME
configuration. Import this before reading any os.getenv().

Lookup order (first hit wins):
  1. Real environment variables already set in the shell (these always win, so
     CI / docker -e overrides still work).
  2. <repo-root>/.env
  3. <section-folder>/.env   (optional per-section override)

Usage (top of any script):
    from env_loader import load_env
    load_env()
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def _find_repo_root(start: Path) -> Path:
    """Walk up until we find the folder containing .env.example (repo root)."""
    for parent in [start, *start.parents]:
        if (parent / ".env.example").exists() or (parent / ".git").exists():
            return parent
    return start


def load_env(verbose: bool = False) -> None:
    """Load .env files into os.environ. Safe to call multiple times."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        if verbose:
            print("[env] python-dotenv not installed; using shell env only. "
                  "pip install python-dotenv")
        return

    here = Path(__file__).resolve().parent
    root = _find_repo_root(here)

    # override=False => real shell env vars take precedence over the file.
    for candidate in (root / ".env", here / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            if verbose:
                print(f"[env] loaded {candidate}")


def require(name: str, hint: str = "") -> str:
    """Fetch a required var with a helpful error instead of a KeyError."""
    val = os.getenv(name)
    if not val:
        msg = f"Missing required env var {name!r}. Set it in your .env file."
        if hint:
            msg += f" ({hint})"
        raise SystemExit(msg)
    return val
