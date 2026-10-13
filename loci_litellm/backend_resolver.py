"""BackendResolver: backend resolution for the Loci integration, so a LiteLLM install carries no
machine-specific endpoint or key in code.

Resolution order (differs per backend):
  - Embeddings API base (any OpenAI-compatible `/embeddings` endpoint, used as given): LOCI_EMBED_API_BASE,
    MNEMOSYNE_EMBEDDING_API_URL, `[embed] api_base`, else None.
  - Embedding model: LOCI_EMBED_MODEL, MNEMOSYNE_EMBEDDING_MODEL, `[embed] model`, else ''. No default.
  - Embedding request shape (path, auth header and scheme, extra headers and body fields, dimensions, query
    and document prefixes): see embed_options(). Each is an env var, else an `[embed]` key.
  - Qdrant url / key: LOCI_QDRANT_URL / QDRANT_URL, LOCI_QDRANT_API_KEY / QDRANT_API_KEY, then
    `[qdrant] url` / `api_key`, else ''.
  - Other settings: env, then config file, then default.

Config file: $LOCI_CONFIG, else ~/.loci/backends.toml (TOML), read once per process; changes need a
restart. Fail-open: an unreadable, unparseable or missing (when LOCI_CONFIG is set) file yields an empty
config and a WARNING; config_load_error() returns the reason. A section that is not a table is ignored
with a WARNING. Env is read on every call.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("loci_litellm.backends")

_CONFIG_PATH = os.environ.get("LOCI_CONFIG") or str(Path.home() / ".loci" / "backends.toml")

_config_error = ""


def config_load_error() -> str:
    """Why the private config is not in effect ('' when it loaded, or none exists at the default path)."""
    return _config_error


@functools.lru_cache(maxsize=1)
def _config() -> dict:
    """Parse the private TOML config; fail-open to {} with a WARNING and config_load_error() set."""
    global _config_error
    _config_error = ""
    p = Path(_CONFIG_PATH)
    try:
        import tomllib
        if p.exists():
            return tomllib.loads(p.read_text(encoding="utf-8"))
        if os.environ.get("LOCI_CONFIG"):
            _config_error = f"LOCI_CONFIG points at a missing file: {p}"
    except Exception as exc:
        _config_error = f"could not read/parse {p}: {exc!r}"
    if _config_error:
        logger.warning("loci backends: running with an EMPTY config (defaults only): %s", _config_error)
    return {}


_warned_sections: set = set()


def _cfg(section: str, key: str, default=None):
    value = _config().get(section)
    if value is None:
        return default
    if not isinstance(value, dict):
        if section not in _warned_sections:
            _warned_sections.add(section)
            logger.warning("loci backends: config section [%s] is not a table; ignored", section)
        return default
    return value.get(key, default)


class BackendResolver:
    """Backend resolution (per-backend order in the module docstring). Methods return '' or a default rather than raising."""

    @staticmethod
    def embed_model() -> str:
        """Embedding model name sent to the endpoint. '' when unconfigured (no default)."""
        return (os.environ.get("LOCI_EMBED_MODEL")
                or os.environ.get("MNEMOSYNE_EMBEDDING_MODEL")
                or _cfg("embed", "model", "") or "")

    @staticmethod
    def qdrant() -> Tuple[str, str]:
        """(url, api_key) for Qdrant. Env -> config -> ('', '')."""
        url = (os.environ.get("LOCI_QDRANT_URL")
               or os.environ.get("QDRANT_URL")
               or _cfg("qdrant", "url", "") or "")
        key = (os.environ.get("LOCI_QDRANT_API_KEY")
               or os.environ.get("QDRANT_API_KEY")
               or _cfg("qdrant", "api_key", "") or "")
        return url, key

    @staticmethod
    def qdrant_url() -> str:
        return BackendResolver.qdrant()[0]

    @staticmethod
    def qdrant_api_key() -> str:
        return BackendResolver.qdrant()[1]

    @staticmethod
    def embed_api_base() -> Optional[str]:
        """Base URL of an OpenAI-compatible embeddings API; requests go to `<base>/embeddings`."""
        base = (os.environ.get("LOCI_EMBED_API_BASE")
                or os.environ.get("MNEMOSYNE_EMBEDDING_API_URL")
                or _cfg("embed", "api_base", "") or "")
        return base.rstrip("/") or None

    @staticmethod
    def embed_api_key() -> str:
        return (os.environ.get("LOCI_EMBED_API_KEY")
                or os.environ.get("EMBED_API_KEY")
                or _cfg("embed", "api_key", "") or "")

    @staticmethod
    def embed_options() -> dict:
        """Request-shape settings for the embeddings call. Env var first, then the `[embed]` config key.

        api_path            LOCI_EMBED_API_PATH         appended to the base (default "/embeddings"; "" when the
                                                         base is already the full URL)
        auth_header         LOCI_EMBED_AUTH_HEADER      header that carries the key (default "Authorization")
        auth_scheme         LOCI_EMBED_AUTH_SCHEME      value prefix (default "Bearer"; "" sends the raw key)
        extra_headers       LOCI_EMBED_EXTRA_HEADERS    JSON object merged into the headers
        extra_body          LOCI_EMBED_EXTRA_BODY       JSON object merged into the JSON body
        dimensions          LOCI_EMBED_DIMENSIONS       integer sent as `dimensions` when set
        query_prefix        LOCI_EMBED_QUERY_PREFIX     text put before a retrieval query
        document_prefix     LOCI_EMBED_DOCUMENT_PREFIX  text put before a stored record
        """
        def text(env: str, key: str, default: str = "") -> str:
            value = os.environ.get(env)
            if value is None:
                value = _cfg("embed", key, default)
            return default if value is None else str(value)

        def mapping(env: str, key: str) -> dict:
            raw = os.environ.get(env)
            value = _cfg("embed", key, {}) if raw is None else raw
            if isinstance(value, str):
                if not value.strip():
                    return {}
                try:
                    value = json.loads(value)
                except ValueError:
                    logger.warning("loci backends: %s is not valid JSON; ignored", env)
                    return {}
            if not isinstance(value, dict):
                logger.warning("loci backends: %s must be a JSON object; ignored", env)
                return {}
            return value

        dims = text("LOCI_EMBED_DIMENSIONS", "dimensions")
        try:
            dimensions = int(dims) if dims.strip() else 0
        except ValueError:
            logger.warning("loci backends: LOCI_EMBED_DIMENSIONS=%r is not an integer; ignored", dims)
            dimensions = 0
        return {
            "api_path": text("LOCI_EMBED_API_PATH", "api_path", "/embeddings"),
            "auth_header": text("LOCI_EMBED_AUTH_HEADER", "auth_header", "Authorization") or "Authorization",
            "auth_scheme": text("LOCI_EMBED_AUTH_SCHEME", "auth_scheme", "Bearer"),
            "extra_headers": {str(k): str(v) for k, v in mapping("LOCI_EMBED_EXTRA_HEADERS", "extra_headers").items()},
            "extra_body": mapping("LOCI_EMBED_EXTRA_BODY", "extra_body"),
            "dimensions": dimensions,
            "query_prefix": text("LOCI_EMBED_QUERY_PREFIX", "query_prefix"),
            "document_prefix": text("LOCI_EMBED_DOCUMENT_PREFIX", "document_prefix"),
        }

    @staticmethod
    def is_disabled() -> bool:
        """True when LOCI_GROUNDING_DISABLED is set."""
        return os.environ.get("LOCI_GROUNDING_DISABLED", "").lower() in ("1", "true", "yes", "on")

    @staticmethod
    def cost_ledger_file() -> str:
        """`[cost] ledger_file` from the private config ('' when unset). LOCI_COST_LEDGER_FILE is
        handled by cost_ledger.ledger_path()."""
        return str(_cfg("cost", "ledger_file", "") or "")

    @staticmethod
    def grounding_audit_file() -> str:
        return os.environ.get("LOCI_GROUNDING_AUDIT_FILE", "")

    @staticmethod
    def memory_write_collection() -> str:
        """Collection the memory writer upserts into: LOCI_MEMORY_WRITE_COLLECTION, else `[memory]
        write_collection`, else `loci_memory`."""
        return (os.environ.get("LOCI_MEMORY_WRITE_COLLECTION")
                or _cfg("memory", "write_collection", "")
                or "loci_memory")

    @staticmethod
    def client_rag_ok_markers() -> tuple[str, ...]:
        """Strings a client puts in a request when its own retrieval succeeded: `[grounding]
        client_rag_ok_markers = [...]` in the config or LOCI_CLIENT_RAG_OK_MARKERS ('||' separated).
        Empty by default, which turns the client shortcut off."""
        raw = os.environ.get("LOCI_CLIENT_RAG_OK_MARKERS")
        if raw is not None:
            values: list = raw.split("||")
        else:
            configured = _cfg("grounding", "client_rag_ok_markers", [])
            values = configured if isinstance(configured, list) else [configured]
        return tuple(v.strip() for v in values if isinstance(v, str) and v.strip())

    @staticmethod
    def client_rag_trusted() -> bool:
        return os.environ.get("LOCI_TRUST_CLIENT_RAG", "1").lower() in ("1", "true", "yes", "on")

    @staticmethod
    def is_internal_call(*metadata_sources) -> bool:
        """True when the request carries ``metadata.loci_internal_call`` (judges, classifiers, embeddings:
        neither grounded nor written back). LOCI_HONOR_INTERNAL_CALL=0 ignores the marker. Any caller
        can set it."""
        if os.environ.get("LOCI_HONOR_INTERNAL_CALL", "1").lower() not in ("1", "true", "yes", "on"):
            return False
        return any(isinstance(m, dict) and m.get("loci_internal_call") is True for m in metadata_sources)

    @staticmethod
    def pheromone_deposit_disabled() -> bool:
        return os.environ.get("LOCI_PHERO_DEPOSIT_DISABLED", "").lower() in ("1", "true", "yes", "on")

def build_embed_request(resolver, text: str, kind: str = "query"):
    """(url, headers, body) for one embedding call, or None when the endpoint or model is not configured.
    `kind` is "query" for retrieval text and "document" for text being stored."""
    base = resolver.embed_api_base()
    model = resolver.embed_model()
    if not base or not model:
        return None
    opts = getattr(resolver, "embed_options", BackendResolver.embed_options)()
    prefix = opts["query_prefix"] if kind == "query" else opts["document_prefix"]
    body = dict(opts["extra_body"])
    body.update({"model": model, "input": f"{prefix}{text}"})
    if opts["dimensions"]:
        body["dimensions"] = opts["dimensions"]
    headers = {"Content-Type": "application/json", **opts["extra_headers"]}
    key = resolver.embed_api_key()
    if key:
        headers[opts["auth_header"]] = f"{opts['auth_scheme']} {key}".strip()
    return f"{base}{opts['api_path']}", headers, body


def parse_embedding(payload) -> list:
    """The vector from an embeddings response. Accepts the OpenAI shape (`data[0].embedding`), a single
    `embedding`, `embeddings[0]`, and `embeddings.float[0]`. Raises ValueError for anything else."""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict) and "embedding" in data[0]:
            return data[0]["embedding"]
        if isinstance(payload.get("embedding"), list):
            return payload["embedding"]
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, dict):
            embeddings = embeddings.get("float")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
    raise ValueError("unrecognised embeddings response shape")


def ledger_path() -> str:
    """Cost ledger path: LOCI_COST_LEDGER_FILE, else `[cost] ledger_file`, else
    ~/.loci/litellm_cost_ledger.jsonl. Lives here so the offline report needs no LiteLLM import."""
    explicit = os.environ.get("LOCI_COST_LEDGER_FILE") or BackendResolver.cost_ledger_file()
    return str(explicit or (Path.home() / ".loci" / "litellm_cost_ledger.jsonl"))
