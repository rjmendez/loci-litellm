"""Delivery audit: the final provider body is checked for the injected context."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from loci_litellm.grounding_hook import LociGroundingHook

CONTEXT = 'MEMORY MATCH (1 results from 1 Qdrant collection) for "q":\n[a|0.9] fact'
NOTICE = "LOCI_GROUNDING_STATUS: miss. Grounding did not add retrieved context for this request because no matching context."


class _Resolver:
    def __init__(self, audit_file: Path | None) -> None:
        self._audit_file = str(audit_file) if audit_file else ""

    def grounding_audit_file(self) -> str:
        return self._audit_file


def _kwargs(status: str | None, body: dict | None, call_id: str = "call-1") -> dict:
    metadata = {"loci_grounding_status": status} if status else {}
    out: dict = {"litellm_params": {"metadata": metadata}, "litellm_call_id": call_id}
    if body is not None:
        out["additional_args"] = {"complete_input_dict": body}
    return out


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_a_grounded_request_whose_final_body_carries_the_context_is_recorded_delivered(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    hook = LociGroundingHook(resolver=_Resolver(audit))
    body = {"model": "m", "messages": [{"role": "system", "content": CONTEXT}, {"role": "user", "content": "q"}]}

    hook.log_pre_api_call("model-x", [], _kwargs("grounded", body))

    (row,) = _rows(audit)
    assert (row["status"], row["delivered"], row["model"], row["call_id"]) == ("grounded", True, "model-x", "call-1")
    assert CONTEXT not in audit.read_text()


def test_a_bridge_that_drops_the_slot_is_reported_as_lost(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    audit = tmp_path / "audit.jsonl"
    hook = LociGroundingHook(resolver=_Resolver(audit))
    body = {"model": "m", "messages": [{"role": "user", "content": "q"}]}

    with caplog.at_level(logging.ERROR, logger="loci_litellm.grounding"):
        hook.log_pre_api_call("model-x", [], _kwargs("grounded", body))

    (row,) = _rows(audit)
    assert (row["status"], row["delivered"]) == ("grounded", False)
    assert [r.levelno for r in caplog.records if "LOST" in r.getMessage()] == [logging.ERROR]


def test_responses_instructions_in_the_final_body_count_as_delivered(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    hook = LociGroundingHook(resolver=_Resolver(audit))

    hook.log_pre_api_call("m", [], _kwargs("grounded", {"instructions": CONTEXT, "input": "q"}, "c-ok"))
    hook.log_pre_api_call("m", [], _kwargs("grounded", {"input": "q"}, "c-lost"))

    assert [(r["call_id"], r["delivered"]) for r in _rows(audit)] == [("c-ok", True), ("c-lost", False)]


def test_a_failure_notice_is_judged_by_its_own_marker(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    hook = LociGroundingHook(resolver=_Resolver(audit))

    hook.log_pre_api_call("m", [], _kwargs("miss", {"system": NOTICE, "messages": []}))

    (row,) = _rows(audit)
    assert (row["status"], row["delivered"]) == ("miss", True)


def test_requests_the_hook_did_not_ground_are_not_audited(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    hook = LociGroundingHook(resolver=_Resolver(audit))

    hook.log_pre_api_call("m", [], _kwargs("client_grounded", {"messages": []}))
    hook.log_pre_api_call("m", [], _kwargs("skipped", {"messages": []}))
    hook.log_pre_api_call("m", [], _kwargs(None, {"messages": []}))

    assert _rows(audit) == []


def test_without_a_final_body_the_logged_messages_are_checked(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    hook = LociGroundingHook(resolver=_Resolver(audit))

    hook.log_pre_api_call("m", [{"role": "system", "content": CONTEXT}], _kwargs("grounded", None))

    (row,) = _rows(audit)
    assert row["delivered"] is True


def test_no_audit_file_configured_still_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    hook = LociGroundingHook(resolver=_Resolver(None))

    with caplog.at_level(logging.INFO, logger="loci_litellm.grounding"):
        hook.log_pre_api_call("model-x", [], _kwargs("grounded", {"messages": [{"role": "system", "content": CONTEXT}]}))

    assert [r.getMessage() for r in caplog.records if "delivered" in r.getMessage()] == [
        "loci grounding: delivered status=grounded model=model-x slot=instruction"
    ]
