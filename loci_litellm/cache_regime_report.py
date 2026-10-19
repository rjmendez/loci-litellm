"""Cache-read share before and after a regime change, for grounded and ungrounded calls. Measurement only:
read-only on the ledger, it changes nothing.

    python -m loci_litellm.cache_regime_report --split 2026-10-18 [--ledger FILE] [--group NAME]
                                               [--include-internal] [--json]

  - Rows with ts at or after 00:00 UTC of the --split date are "after"; earlier rows are "before".
  - "grounded" is a row whose `grounding` field is `grounded`; every other row (skipped, failed, empty) is
    "ungrounded". Internal calls (the memory system's own judges and embeddings) are left out unless
    --include-internal, because they are never grounded and would skew the ungrounded side.
  - Only successful rows count (a failed call reports no usage).
  - cache_read_share is sum(cache_read_tokens) / sum(prompt_tokens) over the cell, exact (Decimal),
    rounded half up to 4 places; null when the cell has no prompt tokens.
  - lag_pp is (ungrounded share - grounded share) * 100 in percentage points, from the rounded shares, null
    when either side has no tokens; `lags_15pp` is true when it is at least 15.
  - Each period lists the distinct `cache_regime` and `config_hash` values its rows carry.
Unparseable ledger lines are counted on stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Optional

from loci_litellm.cost_report import read_rows

SHARE_PLACES = Decimal("0.0001")
LAG_THRESHOLD_PP = Decimal("15")


def split_timestamp(day: str) -> float:
    """Epoch seconds of 00:00 UTC on YYYY-MM-DD."""
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def _share(read: int, prompt: int) -> Optional[Decimal]:
    if prompt <= 0:
        return None
    return (Decimal(read) / Decimal(prompt)).quantize(SHARE_PLACES, rounding=ROUND_HALF_UP)


def _cell(rows: list[dict]) -> dict:
    prompt = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    read = sum(int(r.get("cache_read_tokens") or 0) for r in rows)
    write = sum(int(r.get("cache_write_tokens") or 0) for r in rows)
    return {"calls": len(rows), "prompt_tokens": prompt, "cache_read_tokens": read, "cache_write_tokens": write,
            "cache_read_share": _share(read, prompt)}


def summarise(rows: Iterable[dict], split_ts: float, group: str = "", include_internal: bool = False) -> dict:
    buckets: dict = {(p, g): [] for p in ("before", "after") for g in ("grounded", "ungrounded")}
    seen: dict = {"before": ([], []), "after": ([], [])}
    for row in rows:
        if not row.get("ok"):
            continue
        if row.get("internal") and not include_internal:
            continue
        if group and row.get("group") != group:
            continue
        period = "after" if float(row["ts"]) >= split_ts else "before"
        kind = "grounded" if row.get("grounding") == "grounded" else "ungrounded"
        buckets[(period, kind)].append(row)
        for slot, name in enumerate(("cache_regime", "config_hash")):
            value = row.get(name)
            if value and value not in seen[period][slot]:
                seen[period][slot].append(value)
    out: dict = {}
    for period in ("before", "after"):
        grounded = _cell(buckets[(period, "grounded")])
        ungrounded = _cell(buckets[(period, "ungrounded")])
        g, u = grounded["cache_read_share"], ungrounded["cache_read_share"]
        lag = None if g is None or u is None else (u - g) * 100
        out[period] = {
            "grounded": grounded,
            "ungrounded": ungrounded,
            "lag_pp": lag,
            "lags_15pp": None if lag is None else lag >= LAG_THRESHOLD_PP,
            "cache_regimes": sorted(seen[period][0]),
            "config_hashes": sorted(seen[period][1]),
        }
    return out


def _jsonable(value):
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"not serialisable: {type(value).__name__}")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, help="YYYY-MM-DD (UTC): rows from this day on are 'after'")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--group", default="", help="only this model group")
    ap.add_argument("--include-internal", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        split_ts = split_timestamp(args.split)
    except ValueError:
        print(f"--split must be YYYY-MM-DD, got {args.split!r}", file=sys.stderr)
        return 2
    from loci_litellm.backend_resolver import ledger_path       # LiteLLM-free: the report runs offline
    path = args.ledger or ledger_path()
    stats: dict = {}
    try:
        rows = list(read_rows(path, 0.0, stats))
    except OSError as exc:
        print(f"cannot read the ledger {path}: {exc}", file=sys.stderr)
        return 2
    if stats.get("skipped"):
        print(f"warning: skipped {stats['skipped']} unreadable ledger line(s) of "
              f"{stats['skipped'] + stats['read']} in {path}", file=sys.stderr)
    summary = summarise(rows, split_ts, args.group, args.include_internal)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable))
        return 0
    print(f"ledger {path}: {stats.get('read', 0)} rows read; split at {args.split} 00:00 UTC")
    print(f"{'period':8s} {'kind':11s} {'calls':>7s} {'prompt':>12s} {'cache_read':>12s} {'share':>8s}")
    for period in ("before", "after"):
        for kind in ("grounded", "ungrounded"):
            cell = summary[period][kind]
            share = "-" if cell["cache_read_share"] is None else f"{cell['cache_read_share']:.4f}"
            print(f"{period:8s} {kind:11s} {cell['calls']:7d} {cell['prompt_tokens']:12d} "
                  f"{cell['cache_read_tokens']:12d} {share:>8s}")
        lag = summary[period]["lag_pp"]
        flag = "" if lag is None else ("  (>= 15 pp: grounded calls lag)" if summary[period]["lags_15pp"] else "")
        print(f"{period:8s} lag (ungrounded - grounded): {'n/a' if lag is None else f'{lag:.2f} pp'}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
