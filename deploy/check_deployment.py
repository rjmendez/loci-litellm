#!/usr/bin/env python3
"""Check a LiteLLM deployment (config.yaml and optional .env) against the invariants
this integration depends on. Exit 0 = clean, 1 = violations.

    python check_deployment.py --config config.yaml [--env .env] [--embedding-group embedding]
                               [--local-env-prefix PREFIX ...] [--allow-ip-literals]

Invariants:

  embedding-group     a model group with the embedding name exists, is absent from
                      router_settings.fallbacks / context_window_fallbacks /
                      content_policy_fallbacks, and is not a tier of a complexity_router_config.
                      Not checked: litellm_settings.fallbacks, default_fallbacks, alias targets.
  priced              every deployment declares input_cost_per_token and output_cost_per_token,
                      or LiteLLM's cost map knows the model. Exempt: the embedding group,
                      auto-router deployments, and deployments whose api_base is `os.environ/NAME`
                      with NAME starting with one of --local-env-prefix (default LOCAL, OLLAMA, VLLM,
                      LLAMA; decided by the NAME only).
  no-ip-literal       (skipped with --allow-ip-literals) no literal api_base is an IP address and, with --env, no variable named
                      *_URL/_BASE/_BASE_URL/_ENDPOINT/_HOST (or referenced as an api_base through
                      os.environ/NAME) holds one. Names containing BIND or LISTEN are skipped.
  alias-targets       every model_group_alias target is a model group.
  trusted-proxies     general_settings.trusted_proxy_ranges is set (warning only).

Without --env the environment half of no-ip-literal is skipped, with a NOTE. Callback registration
and LOCI_* settings are not checked. Stdlib plus PyYAML.
"""
from __future__ import annotations

import argparse
import ipaddress
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ENV_REF = re.compile(r"^os\.environ/(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
URL_VAR = re.compile(r"(_URL|_BASE|_BASE_URL|_ENDPOINT|_HOST)$")
BIND_VAR = re.compile(r"(BIND|LISTEN)", re.I)
# Deployments whose api_base is `os.environ/NAME` with NAME starting with one of these are local (exempt from `priced`).
DEFAULT_LOCAL_ENV_PREFIXES = ("LOCAL", "OLLAMA", "VLLM", "LLAMA")


def _host_is_ip(value: str) -> bool:
    """True when the URL's host (or a bare host) is an IP literal."""
    value = value.strip().strip("'\"")
    if not value:
        return False
    host = urlsplit(value if "://" in value else f"//{value}").hostname or ""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _cost_map_prices(model: str) -> bool:
    """True when LiteLLM's cost map has both per-token prices for this model (tries the full name
    and each "/"-suffix). False when litellm cannot be imported."""
    try:
        import litellm
        table = litellm.model_cost
    except Exception:
        return False
    parts = model.split("/")
    for i in range(len(parts)):
        entry = table.get("/".join(parts[i:]))
        if entry and entry.get("input_cost_per_token") is not None and entry.get("output_cost_per_token") is not None:
            return True
    return False


def load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip("'\"")
    return out


def check(config: dict, env: dict[str, str] | None = None, embedding_group: str = "embedding",
          local_env_prefixes: tuple[str, ...] = DEFAULT_LOCAL_ENV_PREFIXES, allow_ip_literals: bool = False):
    """Return (violations, warnings) as lists of strings."""
    violations: list[str] = []
    warnings: list[str] = []
    deployments = config.get("model_list") or []
    groups = {d.get("model_name") for d in deployments if d.get("model_name")}

    # embedding-group
    emb = [d for d in deployments if d.get("model_name") == embedding_group]
    if not emb:
        violations.append(f"embedding-group: no model group named {embedding_group!r}")
    router = config.get("router_settings") or {}
    for key in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks"):
        for entry in router.get(key) or []:
            for src, dsts in (entry or {}).items():
                if embedding_group == src or embedding_group in (dsts or []):
                    violations.append(f"embedding-group: {embedding_group!r} appears in router_settings.{key}")
    for d in deployments:
        params = d.get("litellm_params") or {}
        tiers = (params.get("complexity_router_config") or {}).get("tiers") or {}
        if embedding_group in tiers.values():
            violations.append(f"embedding-group: {embedding_group!r} is a tier of the auto-router")

    # priced
    for d in deployments:
        params = d.get("litellm_params") or {}
        info = d.get("model_info") or {}
        if "complexity_router_config" in params or str(params.get("model", "")).startswith("auto_router/"):
            continue
        if d.get("model_name") == embedding_group:
            continue
        base = str(params.get("api_base", ""))
        m = ENV_REF.match(base)
        if m and m.group("name").upper().startswith(tuple(p.upper() for p in local_env_prefixes)):
            continue
        name = f"{d.get('model_name')} ({info.get('id') or params.get('model')})"
        missing = [f for f in ("input_cost_per_token", "output_cost_per_token") if f not in info and f not in params]
        if missing and not _cost_map_prices(str(params.get("model", ""))):
            for field in missing:
                violations.append(f"priced: {name} has no {field} and LiteLLM's cost map has no entry")

    # no-ip-literal (config)
    for d in deployments:
        base = (d.get("litellm_params") or {}).get("api_base")
        if not allow_ip_literals and isinstance(base, str) and not ENV_REF.match(base) and _host_is_ip(base):
            violations.append(f"no-ip-literal: {d.get('model_name')} api_base is an IP address: {base}")
    # no-ip-literal (env)
    referenced: set[str] = set()
    for d in deployments:
        base = (d.get("litellm_params") or {}).get("api_base")
        m = ENV_REF.match(base) if isinstance(base, str) else None
        if m:
            referenced.add(m.group("name"))
    for key, value in (env or {}).items():
        by_name = URL_VAR.search(key) and not BIND_VAR.search(key)
        if not allow_ip_literals and (by_name or key in referenced) and _host_is_ip(value):
            violations.append(f"no-ip-literal: {key} points at an IP address ({value}); use the DNS name")

    # alias-targets
    for alias, target in (router.get("model_group_alias") or {}).items():
        t = target if isinstance(target, str) else (target or {}).get("model")
        if t not in groups:
            violations.append(f"alias-targets: {alias!r} -> {t!r} is not a model group")

    # trusted-proxies
    if not (config.get("general_settings") or {}).get("trusted_proxy_ranges"):
        warnings.append("trusted-proxies: general_settings.trusted_proxy_ranges is not set")
    return violations, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--env", type=Path)
    ap.add_argument("--embedding-group", default="embedding")
    ap.add_argument("--local-env-prefix", action="append", dest="local_env_prefixes",
                    help="env-name prefix marking a local api_base (repeatable; replaces the defaults)")
    ap.add_argument("--allow-ip-literals", action="store_true", help="do not flag IP addresses in api_base or env")
    args = ap.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    env = load_env(args.env) if args.env else None
    violations, warnings = check(config, env, args.embedding_group,
                                 tuple(args.local_env_prefixes or DEFAULT_LOCAL_ENV_PREFIXES), args.allow_ip_literals)
    if env is None:
        print("NOTE  --env not given: environment variables were not checked for IP addresses")
    for w in warnings:
        print(f"WARN  {w}")
    for v in violations:
        print(f"FAIL  {v}")
    print(f"{len(violations)} violation(s), {len(warnings)} warning(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
