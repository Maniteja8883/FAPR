"""FAPR: Fertility-Normalized Anchored Positional Re-Anchoring (inference-time RoPE patch).

Import this package BEFORE torch is imported anywhere else: the first line below
enables the MPS CPU-fallback for any op Metal doesn't implement, and the env var
must be set before torch initializes its MPS backend.
"""
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Windows' legacy console codepages (cp1252/cp437) can't encode the em-dashes
# and other Unicode punctuation used in this project's log/print strings —
# macOS Terminal and Linux shells default to UTF-8 and never hit this, but a
# stock `cmd.exe`/older PowerShell session will raise UnicodeEncodeError and
# kill the process mid-run. Force UTF-8 stdio (Python >=3.7 stream API) so
# every script behaves identically regardless of host OS/console codepage.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Self-installing first run: if torch/transformers/gradio/... are not present,
# install requirements.txt into THIS interpreter before we import them below,
# so `python app/demo.py` works on a fresh machine with no manual pip step.
# Runs before `.device` (which imports torch). Set FAPR_NO_AUTO_INSTALL=1 to
# opt out. Must not import torch itself — keep it dependency-free.
from ._bootstrap import ensure_dependencies  # noqa: E402

ensure_dependencies()

from .device import pick_device, pick_dtype, empty_cache  # noqa: E402
from .rope_patch import fapr_rope, find_rotary_modules  # noqa: E402
from .fertility import LANGS, ensure_phi, load_phi_table  # noqa: E402

__all__ = [
    "pick_device", "pick_dtype", "empty_cache",
    "fapr_rope", "find_rotary_modules",
    "LANGS", "ensure_phi", "load_phi_table",
]
