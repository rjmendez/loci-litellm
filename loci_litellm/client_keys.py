"""Per-client identity for the Governor, from a header secret. No LiteLLM database and no virtual keys.

A caller names itself with `x-loci-client` and proves it with `x-loci-client-secret`. The owner envelope holds only
the sha256 of each client's secret (`[clients.<name>] secret_sha256`), compared in constant time. A name that is
not in the envelope, a missing secret or a wrong secret all give the same result: the caller is attributed to the
`anonymous` bucket, which has its own (small) cap, and is flagged `client_unknown`. A caller that sends no client
header at all is the anonymous bucket without the flag. Privileges (`interactive`, `rw`, a per-client cap) come
only from the envelope entry of a client that proved itself; nothing a request says about itself grants any.

Why not LiteLLM virtual keys: in the proxy tree this was written against, a request carrying a non-master key
reaches `ProxyException("No connected db.", type=no_db_connection)` when no database is configured
(litellm/proxy/auth/user_api_key_auth.py), and the key-management endpoints raise "DB not connected". Virtual keys
therefore need a database; per-client caps live in the Governor instead.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional

from loci_litellm.envelope import ClientEntry

ANONYMOUS = "anonymous"
CLIENT_HEADER = "x-loci-client"
SECRET_HEADER = "x-loci-client-secret"
LEASE_HEADER = "x-loci-lease"
_NO_SECRET = "0" * 64


@dataclass(frozen=True)
class ClientIdentity:
    name: str
    known: bool
    interactive: bool
    rw: bool
    cap_day: Optional[Decimal]
    notes: tuple[str, ...] = ()


def _header(headers: Mapping[str, Any], name: str) -> str:
    value = headers.get(name)
    return value.strip() if isinstance(value, str) else ""


def identify(headers: Mapping[str, Any], clients: Mapping[str, ClientEntry], anonymous_cap: Decimal) -> ClientIdentity:
    """The caller's identity. `headers` keys must be lower case."""
    claimed = _header(headers, CLIENT_HEADER)
    anonymous = ClientIdentity(ANONYMOUS, False, False, False, anonymous_cap)
    if not claimed:
        return anonymous
    entry = clients.get(claimed) if claimed != ANONYMOUS else None
    secret = _header(headers, SECRET_HEADER)
    presented = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    expected = entry.secret_sha256 if entry is not None else _NO_SECRET
    matches = hmac.compare_digest(presented, expected)      # always computed, so a wrong name costs the same time
    if entry is None or not secret or not matches:
        return ClientIdentity(ANONYMOUS, False, False, False, anonymous_cap, ("client_unknown",))
    return ClientIdentity(entry.name, True, entry.interactive, entry.rw, entry.cap_day)
