"""cache_regime_report: cache-read share before/after a split date, grounded vs ungrounded."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from loci_litellm import cache_regime_report as crr

SPLIT = "2026-10-18"
SPLIT_TS = datetime(2026, 10, 18, tzinfo=timezone.utc).timestamp()


def _r(ts: float, prompt: int, read: int, grounding: str = "grounded", **kw) -> dict:
    row = {"ts": ts, "group": "g", "ok": True, "internal": False, "grounding": grounding,
           "prompt_tokens": prompt, "cache_read_tokens": read, "cache_write_tokens": 0}
    row.update(kw)
    return row


def _fixture_rows() -> list:
    after = SPLIT_TS + 3600
    before = SPLIT_TS - 3600
    return [
        # after, ungrounded: 672,274 of 1,282,180 prompt tokens were cache reads
        _r(after, 1_000_000, 500_000, "skipped", cache_regime="r2", config_hash="h2"),
        _r(after, 282_180, 172_274, "", cache_regime="r2", config_hash="h2"),
        # after, grounded: 40,000 of 400,000
        _r(after, 400_000, 40_000, "grounded", cache_regime="r2", config_hash="h2"),
        # before, grounded: 0 of 800,000; ungrounded: 600,000 of 1,000,000
        _r(before, 800_000, 0, "grounded"),
        _r(before, 1_000_000, 600_000, "skipped", cache_regime="r1", config_hash="h1"),
    ]


def test_the_fixture_gives_the_exact_shares() -> None:
    s = crr.summarise(_fixture_rows(), SPLIT_TS)
    assert s["after"]["ungrounded"] == {"calls": 2, "prompt_tokens": 1_282_180, "cache_read_tokens": 672_274,
                                        "cache_write_tokens": 0, "cache_read_share": Decimal("0.5243")}
    assert s["after"]["grounded"] == {"calls": 1, "prompt_tokens": 400_000, "cache_read_tokens": 40_000,
                                      "cache_write_tokens": 0, "cache_read_share": Decimal("0.1000")}
    assert s["before"]["grounded"]["cache_read_share"] == Decimal("0.0000")
    assert s["before"]["ungrounded"]["cache_read_share"] == Decimal("0.6000")
    assert s["after"]["lag_pp"] == Decimal("42.43") and s["after"]["lags_15pp"] is True
    assert s["before"]["lag_pp"] == Decimal("60.00") and s["before"]["lags_15pp"] is True
    assert (s["after"]["cache_regimes"], s["after"]["config_hashes"]) == (["r2"], ["h2"])
    assert (s["before"]["cache_regimes"], s["before"]["config_hashes"]) == (["r1"], ["h1"])


def test_a_gap_under_15_points_is_not_flagged() -> None:
    rows = [_r(SPLIT_TS + 1, 1000, 500, "grounded"), _r(SPLIT_TS + 1, 1000, 640, "skipped")]
    after = crr.summarise(rows, SPLIT_TS)["after"]
    assert (after["lag_pp"], after["lags_15pp"]) == (Decimal("14.00"), False)
    rows[1] = _r(SPLIT_TS + 1, 1000, 650, "skipped")
    after = crr.summarise(rows, SPLIT_TS)["after"]
    assert (after["lag_pp"], after["lags_15pp"]) == (Decimal("15.00"), True)       # exactly 15 counts


def test_shares_round_half_up_to_four_places() -> None:
    half = crr.summarise([_r(SPLIT_TS, 20_000, 1)], SPLIT_TS)["after"]["grounded"]["cache_read_share"]
    below = crr.summarise([_r(SPLIT_TS, 30_000, 1)], SPLIT_TS)["after"]["grounded"]["cache_read_share"]
    assert half == Decimal("0.0001")                       # 0.00005 rounds up
    assert below == Decimal("0.0000")                      # 0.0000333 rounds down


def test_a_row_exactly_at_the_split_is_after_and_one_second_earlier_is_before() -> None:
    s = crr.summarise([_r(SPLIT_TS, 100, 10), _r(SPLIT_TS - 1, 100, 20)], SPLIT_TS)
    assert (s["after"]["grounded"]["cache_read_tokens"], s["before"]["grounded"]["cache_read_tokens"]) == (10, 20)


def test_empty_cells_have_no_share_and_no_lag() -> None:
    s = crr.summarise([_r(SPLIT_TS + 1, 100, 10, "grounded")], SPLIT_TS)
    assert s["after"]["ungrounded"]["cache_read_share"] is None
    assert (s["after"]["lag_pp"], s["after"]["lags_15pp"]) == (None, None)
    assert s["before"]["grounded"] == {"calls": 0, "prompt_tokens": 0, "cache_read_tokens": 0,
                                       "cache_write_tokens": 0, "cache_read_share": None}


def test_failed_and_internal_rows_are_left_out_unless_asked_for() -> None:
    rows = [_r(SPLIT_TS + 1, 100, 10), _r(SPLIT_TS + 1, 900, 0, ok=False), _r(SPLIT_TS + 1, 500, 5, internal=True)]
    kept = crr.summarise(rows, SPLIT_TS)["after"]["grounded"]
    assert (kept["calls"], kept["prompt_tokens"], kept["cache_read_tokens"]) == (1, 100, 10)
    with_internal = crr.summarise(rows, SPLIT_TS, include_internal=True)["after"]["grounded"]
    assert (with_internal["calls"], with_internal["prompt_tokens"]) == (2, 600)    # the failed row stays out


def test_the_group_filter_keeps_only_that_group() -> None:
    rows = [_r(SPLIT_TS + 1, 100, 10, group="a"), _r(SPLIT_TS + 1, 300, 90, group="b")]
    assert crr.summarise(rows, SPLIT_TS, group="b")["after"]["grounded"]["cache_read_share"] == Decimal("0.3000")
    assert crr.summarise(rows, SPLIT_TS)["after"]["grounded"]["cache_read_share"] == Decimal("0.2500")


def test_rows_from_before_the_cache_fields_count_as_zero_cache_ungrounded() -> None:
    old = {"ts": SPLIT_TS + 1, "ok": True, "prompt_tokens": 100}
    s = crr.summarise([old], SPLIT_TS)["after"]["ungrounded"]
    assert (s["calls"], s["prompt_tokens"], s["cache_read_tokens"], s["cache_read_share"]) == (
        1, 100, 0, Decimal("0.0000"))


def test_split_timestamp_is_midnight_utc() -> None:
    assert crr.split_timestamp("2026-10-18") == 1792281600.0
    with pytest.raises(ValueError, match="does not match format"):
        crr.split_timestamp("18/10/2026")


# ---------------------------------------------------------------- command line

def _write(tmp_path: Path, rows: list, extra: str = "") -> Path:
    path = tmp_path / "ledger.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n" + extra, encoding="utf-8")
    return path


def test_main_json_prints_the_exact_shares(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = _write(tmp_path, _fixture_rows())
    assert crr.main(["--split", SPLIT, "--ledger", str(path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["after"]["ungrounded"]["cache_read_share"] == 0.5243
    assert out["after"]["grounded"]["cache_read_share"] == 0.1
    assert out["after"]["lag_pp"] == 42.43 and out["after"]["lags_15pp"] is True
    assert out["before"]["cache_regimes"] == ["r1"]


def test_main_text_table_and_a_corrupt_line_warning(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = _write(tmp_path, _fixture_rows(), extra="{not json\n")
    assert crr.main(["--split", SPLIT, "--ledger", str(path)]) == 0
    captured = capsys.readouterr()
    assert "after    ungrounded" in captured.out and "0.5243" in captured.out and "0.1000" in captured.out
    assert "lag (ungrounded - grounded): 42.43 pp  (>= 15 pp: grounded calls lag)" in captured.out
    assert "skipped 1 unreadable ledger line(s) of 6" in captured.err


def test_main_refuses_a_bad_date_and_a_missing_ledger(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = _write(tmp_path, _fixture_rows())
    assert crr.main(["--split", "yesterday", "--ledger", str(path)]) == 2
    assert "--split must be YYYY-MM-DD, got 'yesterday'" in capsys.readouterr().err
    assert crr.main(["--split", SPLIT, "--ledger", str(tmp_path / "missing.jsonl")]) == 2
    assert "cannot read the ledger" in capsys.readouterr().err
