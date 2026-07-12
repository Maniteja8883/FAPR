"""First-run dependency bootstrap — makes any FAPR entry point self-installing.

Running `python app/demo.py` (or any script under scripts/) on a fresh machine
should "just work": no manual `pip install -r requirements.txt` step. This
module runs before torch is imported anywhere and, if any third-party package
FAPR needs is not importable, installs the project's requirements.txt into the
CURRENT interpreter (sys.executable) once, then continues in the same process.

Fast path: when everything is already present, every check is an in-memory
importlib lookup and nothing is installed — startup cost is negligible.

Escape hatch: set FAPR_NO_AUTO_INSTALL=1 to skip this entirely and manage the
environment yourself (CI, air-gapped machines, custom torch builds, etc.).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# import-name -> pip label. The pip label is only used for the human-readable
# "missing: ..." message; the actual install is driven by requirements.txt so
# version pins and platform markers (e.g. bitsandbytes; Linux-only) are honored.
_REQUIRED = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "gradio": "gradio",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sentencepiece": "sentencepiece",
    "google.protobuf": "protobuf",
}

_REQ_FILE = Path(__file__).resolve().parents[1] / "requirements.txt"

# Used only if requirements.txt is not sitting next to the package (e.g. FAPR
# was pip-installed as a wheel). Mirrors requirements.txt minus the Linux-only
# bitsandbytes line, which pip would skip on Windows/macOS anyway.
_FALLBACK = [
    "torch>=2.4", "transformers>=4.46,<4.58", "datasets>=2.20",
    "accelerate>=0.33", "gradio>=4.44", "numpy", "pandas", "matplotlib",
    "sentencepiece", "protobuf",
]


def _missing() -> list:
    """Names from _REQUIRED that cannot currently be imported."""
    out = []
    for mod in _REQUIRED:
        try:
            if importlib.util.find_spec(mod) is None:
                out.append(mod)
        except (ImportError, ValueError):
            # A half-installed parent package can raise instead of returning
            # None; treat that as missing so the reinstall repairs it.
            out.append(mod)
    return out


def _ensure_pip() -> None:
    """Bootstrap pip itself if this interpreter somehow lacks it."""
    if importlib.util.find_spec("pip") is None:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])


def ensure_dependencies() -> None:
    """Install any missing requirements into the running interpreter. No-op when
    the environment is already complete or FAPR_NO_AUTO_INSTALL is set."""
    if os.environ.get("FAPR_NO_AUTO_INSTALL"):
        return

    missing = _missing()
    if not missing:
        return

    labels = ", ".join(_REQUIRED[m] for m in missing)
    print(f"[fapr] first-run setup: installing missing dependencies "
          f"({labels}). This happens once and may take a few minutes "
          f"(torch is large)...", flush=True)

    try:
        _ensure_pip()
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
        cmd += ["-r", str(_REQ_FILE)] if _REQ_FILE.exists() else _FALLBACK
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, OSError) as exc:
        target = str(_REQ_FILE) if _REQ_FILE.exists() else "the packages above"
        print(
            "\n[fapr] automatic dependency install failed "
            f"({type(exc).__name__}).\n"
            "       Install them manually with:\n"
            f"           \"{sys.executable}\" -m pip install -r {target}\n"
            "       (Tip: use Python 3.11 or 3.12 — some packages such as torch "
            "do not yet ship wheels for the newest Python releases.)\n",
            file=sys.stderr, flush=True)
        raise

    # New site-packages entries may not be visible to import machinery that has
    # already cached the old state — refresh it before the caller imports them.
    importlib.invalidate_caches()

    still = _missing()
    if still:
        labels = ", ".join(_REQUIRED[m] for m in still)
        print(f"[fapr] warning: these are still not importable after install: "
              f"{labels}. If this persists, restart the process once so the new "
              f"packages are picked up.", file=sys.stderr, flush=True)
    else:
        print("[fapr] dependencies ready.", flush=True)
