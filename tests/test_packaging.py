"""Packaging: identifiable version, declared dependencies, and a README that matches the code."""
from __future__ import annotations

import importlib
import logging
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = PKG_ROOT / "loci_litellm"


def _run(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([sys.executable, "-c", code], cwd=PKG_ROOT, env=env, capture_output=True, text=True, timeout=180)


# --- version ---------------------------------------------------------------------------------------

def test_runtime_version_matches_pyproject() -> None:
    import loci_litellm

    declared = tomllib.loads((PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert loci_litellm.__version__ == declared


def test_import_logs_version_and_path(caplog) -> None:
    import loci_litellm

    with caplog.at_level(logging.INFO, logger="loci_litellm"):
        importlib.reload(loci_litellm)
    messages = [r.getMessage() for r in caplog.records if r.name == "loci_litellm"]
    assert any(loci_litellm.__version__ in m and str(PKG_DIR) in m for m in messages), messages


def test_pyproject_declares_every_third_party_import() -> None:
    deps = tomllib.loads((PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    names = {re.split(r"[<>=!~ ]", d, maxsplit=1)[0].lower() for d in deps}
    assert {"httpx", "litellm", "pyyaml"} <= names


# --- numeric env at import is loud -----------------------------------------------------------------

def test_non_numeric_recall_env_fails_the_import() -> None:
    out = _run("import loci_litellm.grounding_hook", {"LOCI_RECALL_TOP_K": "abc"})
    assert out.returncode != 0 and "ValueError" in out.stderr


# --- README matches the code -----------------------------------------------------------------------

_ENV = r"(?:LOCI|MNEMOSYNE|QDRANT|OLLAMA|EMBED)_[A-Z0-9_]+"


def _code_env_names() -> set[str]:
    found: set[str] = set()
    for path in PKG_DIR.glob("*.py"):
        found |= set(re.findall(rf"""["']({_ENV})["']""", path.read_text(encoding="utf-8")))
    return found - {"LOCI_GROUNDING_STATUS"}  # a label in the injected notice, not an env var


def _readme_names() -> set[str]:
    text = (PKG_ROOT / "README.md").read_text(encoding="utf-8")
    names = set(re.findall(rf"`({_ENV})`", text))
    # suffix shorthand in one table row: `LOCI_RANKER_W_RELEVANCE`, `_RECENCY`, ...
    for prefix in ("LOCI_RANKER_W_", "LOCI_LOCAL_SLO_"):
        for line in text.splitlines():
            if f"`{prefix}" in line:
                names |= {prefix + s for s in re.findall(r"`_([A-Z0-9_]+)`", line.split("|")[1])}
    return names


def test_every_env_var_the_code_reads_is_in_the_readme() -> None:
    missing = sorted(_code_env_names() - _readme_names())
    assert missing == [], f"read by the code but not documented: {missing}"


def test_readme_documents_no_env_var_the_code_does_not_read() -> None:
    phantom = sorted(_readme_names() - _code_env_names())
    assert phantom == [], f"documented but never read: {phantom}"
