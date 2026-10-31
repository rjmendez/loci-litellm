"""Hard caller policy for the verified Hermes identity.

Stubs only the trust inputs (the envelope and the live router's model list); the policy logic runs for real.
Each rejection has a positive twin.

Mutation runs (each reverted after):
  - removed the model allowlist check            -> test_invented_model_is_refused FAILED
  - removed the tool-args JSON check             -> test_malformed_tool_arguments_are_refused FAILED
  - made the broad except pass instead of 503    -> test_policy_error_fails_closed_for_a_verified_caller FAILED
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from loci_litellm import caller_policy as cp
from loci_litellm.envelope import Caps, ClientEntry, EnvelopeState

SECRET = "hermes-test-secret-value"
GOOD_MODEL = "auto-router"
MODELS = {"auto-router", "medium-tier", "complex-tier"}


def _env() -> EnvelopeState:
    entry = ClientEntry(name="hermes", secret_sha256=hashlib.sha256(SECRET.encode()).hexdigest(),
                        cap_day=None, interactive=True, rw=False)
    return EnvelopeState(caps=Caps(), enforce=True, day_timezone="local", clients={"hermes": entry},
                         level="L0", alarm="", sha256="", source="envelope")


@pytest.fixture
def stub_identity_and_router(monkeypatch, tmp_path):
    fake_gov = SimpleNamespace(governor=SimpleNamespace(_envelope=lambda: _env()))
    import loci_litellm.governor as gov_mod
    monkeypatch.setattr(gov_mod, "loci_governor", fake_gov)
    monkeypatch.setattr(cp, "_deployments", lambda: [{"model_name": m} for m in sorted(MODELS)])
    monkeypatch.setenv("LOCI_GOVERNOR_DIR", str(tmp_path))
    return tmp_path


def _data(model: str, messages=None, headers=None) -> dict:
    if messages is None:
        messages = [{"role": "user", "content": "hi"}]
    d = {"model": model, "messages": messages}
    if headers is not None:
        d["proxy_server_request"] = {"headers": headers}
    return d


def _hermes_headers(secret: str = SECRET) -> dict:
    return {"x-loci-client": "hermes", "x-loci-client-secret": secret}


def _run(data: dict):
    return asyncio.run(cp.loci_caller_policy.async_pre_call_hook(None, None, data, "acompletion"))


def test_known_model_and_valid_history_pass(stub_identity_and_router) -> None:
    assert _run(_data(GOOD_MODEL, headers=_hermes_headers())) is None


def test_invented_model_is_refused(stub_identity_and_router) -> None:
    with pytest.raises(cp.PolicyRejection, match="model 'gpt-9-invented' is not served") as exc:
        _run(_data("gpt-9-invented", headers=_hermes_headers()))
    assert exc.value.status_code == 400 and exc.value.code == "model_not_allowed"


def test_refusal_is_recorded_as_an_audit_line(stub_identity_and_router) -> None:
    with pytest.raises(cp.PolicyRejection):
        _run(_data("gpt-9-invented", headers=_hermes_headers()))
    lines = (stub_identity_and_router / cp.VIOLATIONS_FILE).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["reason"] for r in rows] == ["model_not_allowed"]
    assert rows[0]["client"] == "hermes" and rows[0]["detail"] == "gpt-9-invented"


def test_malformed_tool_arguments_are_refused(stub_identity_and_router) -> None:
    messages = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "execute_code", "arguments": "{not json"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    with pytest.raises(cp.PolicyRejection, match="message 1: execute_code arguments not JSON") as exc:
        _run(_data(GOOD_MODEL, messages=messages, headers=_hermes_headers()))
    assert exc.value.status_code == 400 and exc.value.code == "malformed_tool_call_history"


def test_valid_tool_arguments_pass(stub_identity_and_router) -> None:
    # positive twin of the malformed-argument refusal: same shape, arguments are a JSON object string
    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{\"cmd\": \"dir\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    assert _run(_data(GOOD_MODEL, messages=messages, headers=_hermes_headers())) is None


def test_wrong_secret_is_not_enforced(stub_identity_and_router) -> None:
    # not a verified caller: the policy does not constrain it (the governor still caps it as anonymous)
    assert _run(_data("gpt-9-invented", headers=_hermes_headers(secret="wrong"))) is None


def test_no_identity_is_not_enforced(stub_identity_and_router) -> None:
    assert _run(_data("gpt-9-invented", headers=None)) is None


def test_router_unavailable_fails_closed_for_a_verified_caller(stub_identity_and_router, monkeypatch) -> None:
    monkeypatch.setattr(cp, "_deployments", lambda: None)
    with pytest.raises(cp.PolicyRejection) as exc:
        _run(_data(GOOD_MODEL, headers=_hermes_headers()))
    assert exc.value.status_code == 503 and exc.value.code == "policy_error"


def test_policy_error_fails_closed_for_a_verified_caller(stub_identity_and_router, monkeypatch) -> None:
    def boom():
        raise RuntimeError("router state corrupt")
    monkeypatch.setattr(cp, "_deployments", boom)
    with pytest.raises(cp.PolicyRejection) as exc:
        _run(_data(GOOD_MODEL, headers=_hermes_headers()))
    assert exc.value.status_code == 503 and exc.value.code == "policy_error"
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_extra_models_from_env_are_allowed(stub_identity_and_router, monkeypatch) -> None:
    # the real _router_models merge runs here: a model absent from the router is allowed only via the env list
    monkeypatch.setenv("LOCI_POLICY_EXTRA_MODELS", "abliterated-model-large-v2")
    assert _run(_data("abliterated-model-large-v2", headers=_hermes_headers())) is None
    with pytest.raises(cp.PolicyRejection, match="not served"):
        _run(_data("abliterated-model-xl", headers=_hermes_headers()))


# --- forced retrieval: a recall question must reach the memory tool before the model answers ---

RETRIEVAL = {"type": "function", "function": {"name": "mcp__loci__rag_context_search", "parameters": {}}}


def _recall_data(text: str, tools=None, extra_messages=(), tool_choice=None) -> dict:
    d = _data(GOOD_MODEL, messages=[{"role": "user", "content": text}, *extra_messages],
              headers=_hermes_headers())
    d["tools"] = [RETRIEVAL] if tools is None else tools
    if tool_choice is not None:
        d["tool_choice"] = tool_choice
    return d


def test_recall_question_is_forced_to_the_memory_tool(stub_identity_and_router) -> None:
    data = _recall_data("What did we decide about the relay earlier?")
    _run(data)
    assert data["tool_choice"] == {"type": "function", "function": {"name": "mcp__loci__rag_context_search"}}


def test_non_recall_question_is_not_forced(stub_identity_and_router) -> None:
    # positive twin: an ordinary request is left alone
    data = _recall_data("Reformat this list into a table.")
    _run(data)
    assert "tool_choice" not in data


def test_no_forcing_once_the_turn_already_used_a_tool(stub_identity_and_router) -> None:
    # the model has gone to a tool this turn; forcing again would loop
    extra = [{"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "mcp__loci__rag_context_search", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "results"}]
    data = _recall_data("What did we decide about the relay earlier?", extra_messages=extra)
    _run(data)
    assert "tool_choice" not in data


def test_callers_own_tool_choice_is_respected(stub_identity_and_router) -> None:
    data = _recall_data("What did we decide earlier?", tool_choice="none")
    _run(data)
    assert data["tool_choice"] == "none"


def test_no_retrieval_tool_offered_means_no_forcing(stub_identity_and_router) -> None:
    data = _recall_data("What did we decide earlier?", tools=[])
    _run(data)
    assert "tool_choice" not in data


def test_recall_patterns_can_be_overridden(stub_identity_and_router, monkeypatch) -> None:
    monkeypatch.setenv("LOCI_POLICY_RECALL_PATTERNS", "zebra")
    data = _recall_data("What did we decide earlier?")
    _run(data)
    assert "tool_choice" not in data
    data = _recall_data("tell me about the zebra incident")
    _run(data)
    assert data["tool_choice"]["function"]["name"] == "mcp__loci__rag_context_search"


# --- act, don't talk: a promised tool action must become a tool call on the next turn ---

def _promise_data(assistant_text: str, tools=None, tool_choice=None, tool_calls=None) -> dict:
    assistant = {"role": "assistant", "content": assistant_text}
    if tool_calls is not None:
        assistant["tool_calls"] = tool_calls
    d = {"model": GOOD_MODEL, "messages": [{"role": "user", "content": "check the store"}, assistant]}
    d["tools"] = [RETRIEVAL] if tools is None else tools
    if tool_choice is not None:
        d["tool_choice"] = tool_choice
    return d


def test_promised_action_without_a_call_requires_a_tool(stub_identity_and_router) -> None:
    data = _promise_data("I'll check the investigation store and report back.")
    _run(data)
    assert data["tool_choice"] == "required"
    assert data["messages"][-1]["role"] == "system"
    assert "made no tool call" in data["messages"][-1]["content"]


def test_promise_is_recorded_as_a_violation(stub_identity_and_router) -> None:
    _run(_promise_data("Let me run the query now."))
    rows = [json.loads(line) for line in (stub_identity_and_router / cp.VIOLATIONS_FILE).read_text(
        encoding="utf-8").splitlines()]
    assert [r["reason"] for r in rows] == ["unfulfilled_promise"]


def test_promise_that_already_made_a_call_is_not_forced(stub_identity_and_router) -> None:
    # positive twin: the assistant turn carries a real tool call, so nothing was promised and not made
    calls = [{"id": "c1", "type": "function", "function": {"name": "mcp__loci__rag_context_search",
                                                           "arguments": "{\"query\": \"x\"}"}}]
    data = _promise_data("I'll check the store.", tool_calls=calls)
    _run(data)
    assert "tool_choice" not in data


def test_plain_reply_without_a_promise_is_not_forced(stub_identity_and_router) -> None:
    data = _promise_data("The relay is on port 4443.")
    _run(data)
    assert "tool_choice" not in data


def test_no_tools_offered_means_no_forcing(stub_identity_and_router) -> None:
    data = _promise_data("I'll check the store.", tools=[])
    _run(data)
    assert "tool_choice" not in data


def test_callers_tool_choice_none_is_respected(stub_identity_and_router) -> None:
    data = _promise_data("I'll check the store.", tool_choice="none")
    _run(data)
    assert data["tool_choice"] == "none"


def test_unidentified_caller_is_also_held_to_the_promise_rule(stub_identity_and_router) -> None:
    # the rule is not identity-gated: no headers at all still gets the required tool call
    data = _promise_data("Let me verify that against the logs.")
    data["proxy_server_request"] = {"headers": {}}
    _run(data)
    assert data["tool_choice"] == "required"


def test_deferred_loci_tools_force_the_discovery_tool(stub_identity_and_router) -> None:
    # Hermes sends the Loci tools deferred: no rag tool in the request, only tool_search. The recall question
    # must still reach a tool first, through tool_search.
    discovery = {"type": "function", "function": {"name": "tool_search", "parameters": {}}}
    data = _recall_data("What did we decide about the relay earlier?", tools=[discovery])
    _run(data)
    assert data["tool_choice"] == {"type": "function", "function": {"name": "tool_search"}}
