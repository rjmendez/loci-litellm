"""Loci grounding, memory write-back and routing callbacks for the LiteLLM proxy.

``__version__`` comes from the adjacent ``pyproject.toml`` in a source checkout, else from installed
distribution metadata, else ``"unknown"``. The package path is logged once at INFO on import.
"""
from __future__ import annotations

import logging
from pathlib import Path


def _read_version() -> str:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        if data.get("project", {}).get("name") == "loci-litellm":
            return str(data["project"]["version"])
    except Exception:  # not a source checkout, or unreadable
        pass
    try:
        from importlib.metadata import version

        return version("loci-litellm")
    except Exception:
        return "unknown"


__version__ = _read_version()

logging.getLogger("loci_litellm").info(
    "loci_litellm %s loaded from %s", __version__, Path(__file__).resolve().parent
)

# Loaded on first attribute access so the offline tools (``python -m loci_litellm.cost_report``)
# run without LiteLLM installed.
_LAZY = {
    "BackendResolver": "loci_litellm.backend_resolver",
    "LociGroundingHook": "loci_litellm.grounding_hook",
    "loci_grounding_hook": "loci_litellm.grounding_hook",
    "LociMemoryWriter": "loci_litellm.memory_writer",
    "loci_memory_writer": "loci_litellm.memory_writer",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'loci_litellm' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "BackendResolver",
    "LociGroundingHook",
    "loci_grounding_hook",
    "LociMemoryWriter",
    "loci_memory_writer",
]
