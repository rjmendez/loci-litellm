"""prefix_hash: the cacheable-prefix hash and the bounded session-dwell tracker."""
from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path

import pytest

from loci_litellm import backend_resolver, prefix_hash as ph
from loci_litellm.grounding_hook import _OWN_CONTEXT_PREFIX, loci_grounding_hook as hook
from loci_litellm.prefix_hash import DEFAULT_STRIP_MARKERS, DwellTracker, prefix_hash, request_of


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("LOCI_PREFIX_STRIP_MARKERS", "LOCI_CLIENT_RAG_OK_MARKERS", "LOCI_ARM_DWELL",
                 "LOCI_ARM_DWELL_SECONDS", "LOCI_ARM_DWELL_ALARM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(backend_resolver, "_CONFIG_PATH", str(tmp_path / "backends.toml"))
    backend_resolver._config.cache_clear()
    yield
    backend_resolver._config.cache_clear()


SYSTEM = "You are a careful assistant."
USER = "Fix the failing test in parser.py"
HITS = [{"content": "the relay was moved to the second rack", "collection": "loci_memory", "score": 0.91},
        {"content": "decision: keep the old retry limit", "collection": "docs", "score": 0.77}]


def _chat(system=SYSTEM, user=USER) -> dict:
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


# ---------------------------------------------------------------- what is stripped

def test_the_golden_hash_of_a_plain_request() -> None:
    assert prefix_hash(_chat()) == "d1d02821843a8914"


def test_reminder_tags_and_a_client_rag_header_do_not_change_the_hash() -> None:
    plain = _chat()
    noisy = _chat(user=("<system-reminder>The date is 2026-10-18 and the build is red.</system-reminder>"
                        "AUTOMATIC LOCI RAG: 3 notes about parser.py\n[a] one\n[b] two\n\n" + USER))
    other_noise = _chat(user=("<system-reminder>Different reminder\nspanning lines</system-reminder>"
                              "AUTOMATIC LOCI RAG: nothing found\n\n" + USER))
    assert prefix_hash(noisy) == prefix_hash(plain) == prefix_hash(other_noise)


def test_a_different_system_text_or_user_text_changes_the_hash() -> None:
    base = prefix_hash(_chat())
    assert prefix_hash(_chat(system="You are a terse assistant.")) != base
    assert prefix_hash(_chat(user="Fix the failing test in lexer.py")) != base
    # twin: the same text, only reminder noise added to the system message, is equal
    assert prefix_hash(_chat(system="<system-reminder>x</system-reminder>" + SYSTEM)) == base


def test_a_reminder_only_leading_message_is_skipped_for_the_first_conversational_message() -> None:
    a = {"messages": [{"role": "user", "content": "<system-reminder>r</system-reminder>"},
                      {"role": "user", "content": USER}]}
    b = {"messages": [{"role": "user", "content": USER}]}
    assert prefix_hash(a) == prefix_hash(b)


def test_text_parts_are_stripped_per_part_and_images_count_by_type() -> None:
    parts = [{"type": "text", "text": "<system-reminder>r</system-reminder>"}, {"type": "text", "text": USER}]
    assert prefix_hash({"messages": [{"role": "user", "content": parts}]}) == prefix_hash(
        {"messages": [{"role": "user", "content": USER}]})
    with_image = parts + [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert prefix_hash({"messages": [{"role": "user", "content": with_image}]}) != prefix_hash(
        {"messages": [{"role": "user", "content": USER}]})


def test_whitespace_runs_and_tool_key_order_do_not_matter() -> None:
    tools_a = [{"type": "function", "function": {"name": "f", "description": "d"}}]
    tools_b = [{"function": {"description": "d", "name": "f"}, "type": "function"}]
    assert prefix_hash({**_chat(), "tools": tools_a}) == prefix_hash({**_chat(), "tools": tools_b})
    assert prefix_hash(_chat(user="Fix  the failing\n test in parser.py")) == prefix_hash(
        _chat(user="Fix the failing test in parser.py"))


def test_the_tool_set_is_part_of_the_prefix() -> None:
    tools = [{"type": "function", "function": {"name": "f"}}]
    assert prefix_hash({**_chat(), "tools": tools}) != prefix_hash(_chat())
    assert prefix_hash({**_chat(), "tools": tools + [{"type": "function", "function": {"name": "g"}}]}) != \
        prefix_hash({**_chat(), "tools": tools})


def test_nothing_to_hash_gives_an_empty_string() -> None:
    assert prefix_hash({}) == ""
    assert prefix_hash({"messages": []}) == ""
    assert prefix_hash({"messages": [{"role": "user", "content": "<system-reminder>r</system-reminder>"}]}) == ""
    assert prefix_hash("not a mapping") == ""        # type: ignore[arg-type]


def test_an_unterminated_tag_removes_the_rest_of_the_text() -> None:
    open_ended = _chat(user="<system-reminder>never closed " + USER)
    assert prefix_hash(open_ended) == prefix_hash({"messages": [{"role": "system", "content": SYSTEM}]})


def test_only_the_first_16384_characters_of_a_text_are_hashed() -> None:
    head = "x" * ph.TEXT_CAP
    assert ph.TEXT_CAP == 16_384
    assert prefix_hash(_chat(system=head + "tail one")) == prefix_hash(_chat(system=head + "tail two"))
    # twin: a difference inside the cap still counts, down to its last character
    edge = "x" * (ph.TEXT_CAP - 1)
    assert prefix_hash(_chat(system=edge + "a")) != prefix_hash(_chat(system=edge + "b"))
    assert prefix_hash(_chat(system="y" + head[1:] + "tail")) != prefix_hash(_chat(system=head + "tail"))
    big_tools = [{"type": "function", "function": {"name": "f", "description": "d" * ph.TEXT_CAP}}]
    other_tools = [{"type": "function", "function": {"name": "f", "description": "d" * ph.TEXT_CAP + "more"}}]
    assert prefix_hash({**_chat(), "tools": big_tools}) == prefix_hash({**_chat(), "tools": other_tools})


def test_a_leading_marker_drops_a_whole_chat_message_even_with_blank_lines_inside() -> None:
    block = "MEMORY MATCH (2 results from 1 collection) for \"q\" (retrieved reference data):\n[a|0.9] one\n\n[b|0.8] two"
    with_block = {"messages": [{"role": "system", "content": block}, *_chat()["messages"]]}
    assert prefix_hash(with_block) == prefix_hash(_chat())


def test_a_marker_inside_a_text_removes_only_up_to_the_next_blank_line() -> None:
    mixed = _chat(user="AUTOMATIC LOCI RAG: 2 notes\n[a] one\n\n" + USER)
    assert prefix_hash(mixed) == prefix_hash(_chat())
    kept = _chat(user="before the marker, kept. AUTOMATIC LOCI RAG: 2 notes\n\n" + USER)
    assert prefix_hash(kept) != prefix_hash(_chat())      # the text before the marker is part of the prefix


# ---------------------------------------------------------------- what the grounding hook emits

def _surfaces() -> dict:
    return {
        "chat": ({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]}, "chat"),
        "anthropic_str": ({"system": SYSTEM, "messages": [{"role": "user", "content": USER}]}, "anthropic"),
        "anthropic_list": ({"system": [{"type": "text", "text": SYSTEM}],
                            "messages": [{"role": "user", "content": USER}]}, "anthropic"),
        "responses": ({"instructions": SYSTEM, "input": USER}, "responses"),
        "completion": ({"prompt": USER}, "completion"),
    }


@pytest.mark.parametrize("name", ["chat", "anthropic_str", "anthropic_list", "responses", "completion"])
def test_the_default_markers_remove_what_the_grounding_hook_injects(name: str) -> None:
    data, surface = _surfaces()[name]
    context = hook._format_results(HITS, "what did we decide about the relay", 0)
    assert context.startswith(_OWN_CONTEXT_PREFIX)             # the default marker is the hook's own prefix
    grounded = copy.deepcopy(data)
    hook._inject_context(grounded, context, surface)
    failed = copy.deepcopy(data)
    hook._inject_failure_notice(failed, "skipped", "embed_unavailable", surface)
    assert grounded != data and failed != data                  # the hook really changed the request
    assert prefix_hash(grounded) == prefix_hash(failed) == prefix_hash(data) != ""


def test_a_changed_system_text_still_changes_the_hash_after_injection() -> None:
    data, surface = _surfaces()["anthropic_str"]
    other = copy.deepcopy(data)
    other["system"] = "You are a terse assistant."
    context = hook._format_results(HITS, "q", 0)
    for request in (data, other):
        hook._inject_context(request, context, surface)
    assert prefix_hash(data) != prefix_hash(other)


@pytest.mark.xfail(strict=True, reason="known limit: a retrieved block with a blank line inside, prepended to a "
                                       "string field, leaves a remainder in the hash")
def test_a_block_with_a_blank_line_prepended_to_a_string_field_is_fully_removed() -> None:
    data = {"system": SYSTEM, "messages": [{"role": "user", "content": USER}]}
    grounded = copy.deepcopy(data)
    hook._inject_context(grounded, hook._format_results(
        [{"content": "first paragraph\n\nsecond paragraph", "collection": "docs", "score": 0.9}], "q", 0), "anthropic")
    assert prefix_hash(grounded) == prefix_hash(data)


def test_the_default_markers_are_the_documented_four() -> None:
    assert DEFAULT_STRIP_MARKERS == ("<system-reminder>=></system-reminder>", "MEMORY MATCH (",
                                     "LOCI_GROUNDING_STATUS:", "AUTOMATIC LOCI RAG")


# ---------------------------------------------------------------- configuration

def test_markers_passed_in_replace_the_defaults() -> None:
    noisy = _chat(user="[[note]]x[[/note]]" + USER)
    assert prefix_hash(noisy, markers=("[[note]]=>[[/note]]",)) == prefix_hash(_chat(), markers=())
    # twin: the defaults do not know this tag, so it is part of the prefix
    assert prefix_hash(noisy) != prefix_hash(_chat())
    # twin: and the custom list does not know the reminder tag
    reminder = _chat(user="<system-reminder>r</system-reminder>" + USER)
    assert prefix_hash(reminder, markers=("[[note]]=>[[/note]]",)) != prefix_hash(_chat(), markers=())


def test_the_env_var_sets_the_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    noisy = _chat(user="[[note]]x[[/note]]" + USER)
    monkeypatch.setenv("LOCI_PREFIX_STRIP_MARKERS", "[[note]]=>[[/note]]||AUTOMATIC X")
    assert ph.effective_markers() == ("[[note]]=>[[/note]]", "AUTOMATIC X")
    assert prefix_hash(noisy) == prefix_hash(_chat())


def test_the_private_config_sets_the_markers(tmp_path: Path) -> None:
    (tmp_path / "backends.toml").write_text('[cache]\nprefix_strip_markers = ["<<n>>=><</n>>", " "]\n')
    backend_resolver._config.cache_clear()
    assert ph.effective_markers() == ("<<n>>=><</n>>",)
    assert prefix_hash(_chat(user="<<n>>x<</n>>" + USER)) == prefix_hash(_chat())


def test_an_empty_setting_strips_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_PREFIX_STRIP_MARKERS", "")
    assert ph.effective_markers() == ()
    assert prefix_hash(_chat(user="<system-reminder>r</system-reminder>" + USER)) != prefix_hash(_chat())


def test_unset_means_the_defaults() -> None:
    assert ph.effective_markers() == DEFAULT_STRIP_MARKERS


def test_the_client_retrieval_markers_are_always_added(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_CLIENT_RAG_OK_MARKERS", "[CLIENT-RAG: OK]")
    assert ph.effective_markers() == DEFAULT_STRIP_MARKERS + ("[CLIENT-RAG: OK]",)
    noisy = _chat(user="[CLIENT-RAG: OK]\nnotes\n[/CLIENT-RAG]\n\n" + USER)
    assert prefix_hash(noisy) == prefix_hash(_chat())


def test_request_of_reads_a_callback_kwargs_shape() -> None:
    tools = [{"type": "function", "function": {"name": "f"}}]
    kwargs = {"messages": _chat()["messages"], "optional_params": {"tools": tools, "max_tokens": 5}}
    assert request_of(kwargs) == {"messages": _chat()["messages"], "tools": tools}
    assert prefix_hash(request_of(kwargs)) == prefix_hash({**_chat(), "tools": tools})
    assert request_of({}) == {}


# ---------------------------------------------------------------- dwell tracker

class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _tracker(clock: Clock, **kw) -> DwellTracker:
    kw.setdefault("enabled", True)
    return DwellTracker(dwell_s=300.0, clock=clock, **kw)


def test_it_is_off_by_default_and_then_changes_nothing() -> None:
    clock = Clock()
    tracker = DwellTracker(clock=clock)
    assert tracker.enabled is False
    picks = []
    for i in range(5):
        picks.append(tracker.choose("k", ["a", "b"] if i % 2 == 0 else ["b", "a"]))
    assert picks == ["a", "b", "a", "b", "a"]                     # the router's own first choice, every time
    assert tracker.stats() == {"entries": 0, "passthrough": 5}
    assert tracker.choose("k", ["b"], alarms={"b": 0.99}) == "b"   # no pinning, no alarm handling


def test_the_env_flag_turns_it_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ARM_DWELL", "1")
    tracker = DwellTracker(clock=Clock())
    assert tracker.enabled is True
    assert [tracker.choose("k", ["a", "b"]), tracker.choose("k", ["b", "a"])] == ["a", "a"]


def test_the_env_sets_the_dwell_and_alarm_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ARM_DWELL_SECONDS", "60")
    monkeypatch.setenv("LOCI_ARM_DWELL_ALARM", "0.9")
    tracker = DwellTracker(enabled=True, clock=Clock())
    assert (tracker.dwell_s, tracker.alarm_threshold) == (60.0, 0.9)
    monkeypatch.setenv("LOCI_ARM_DWELL_SECONDS", "soon")
    assert DwellTracker(enabled=True, clock=Clock()).dwell_s == 300.0


def test_a_hundred_requests_with_one_prefix_all_pick_the_first_arm() -> None:
    clock = Clock()
    tracker = _tracker(clock)
    picks = []
    for i in range(100):
        # the router's ranking flips on every call; the pinned arm must not
        picks.append(tracker.choose("prefix-1", ["a", "b"] if i == 0 or i % 2 else ["b", "a"]))
        clock.now += 2.0                                           # 198 s in all, inside the 300 s dwell
    assert picks == ["a"] * 100
    assert tracker.stats() == {"entries": 1, "held": 99, "pinned": 1}


def test_other_prefixes_are_pinned_independently() -> None:
    tracker = _tracker(Clock())
    assert tracker.choose("p1", ["a", "b"]) == "a"
    assert tracker.choose("p2", ["b", "a"]) == "b"
    assert tracker.choose("p1", ["b", "a"]) == "a"
    assert tracker.choose("p2", ["a", "b"]) == "b"


def test_an_alarm_above_the_threshold_moves_the_prefix_to_the_next_healthy_arm() -> None:
    clock = Clock()
    tracker = _tracker(clock)
    assert tracker.choose("p", ["a", "b", "c"]) == "a"
    clock.now += 1.0
    assert tracker.choose("p", ["a", "b", "c"], alarms={"a": 0.61, "b": 0.7}) == "c"   # b is alarmed too
    assert tracker.stats()["moved_alarm"] == 1
    clock.now += 1.0
    assert tracker.choose("p", ["a", "b", "c"]) == "c"             # and it now holds the new arm


def test_an_alarm_exactly_at_the_threshold_does_not_move_it() -> None:
    tracker = _tracker(Clock())
    tracker.choose("p", ["a", "b"])
    assert tracker.choose("p", ["a", "b"], alarms={"a": 0.6}) == "a"
    assert tracker.choose("p", ["a", "b"], alarms={"a": 0.6000001}) == "b"


def test_an_alarm_moves_it_whatever_the_cache_loss() -> None:
    tracker = _tracker(Clock())
    tracker.choose("p", ["a", "b"])
    assert tracker.choose("p", ["a", "b"], alarms={"a": 0.9}, gain=Decimal("0"), cache_loss=Decimal("99")) == "b"


def test_with_every_alternative_alarmed_it_stays_and_says_so() -> None:
    tracker = _tracker(Clock())
    tracker.choose("p", ["a", "b"])
    assert tracker.choose("p", ["a", "b"], alarms={"a": 0.9, "b": 0.9}) == "a"
    assert tracker.stats()["stuck_alarm"] == 1


def test_a_budget_breach_or_a_vanished_arm_also_moves_it() -> None:
    tracker = _tracker(Clock())
    tracker.choose("p", ["a", "b"])
    assert tracker.choose("p", ["a", "b"], budget_breach=True) == "b"
    assert tracker.stats()["moved_budget"] == 1
    tracker.choose("q", ["a", "b"])
    assert tracker.choose("q", ["c", "b"]) == "c"                  # "a" is no longer offered
    assert tracker.stats()["moved_unavailable"] == 1


def test_after_the_dwell_a_switch_needs_a_gain_greater_than_the_cache_loss() -> None:
    clock = Clock()
    tracker = _tracker(clock)
    tracker.choose("p", ["a", "b"])
    clock.now += 299.999
    assert tracker.choose("p", ["b", "a"], gain=Decimal("5"), cache_loss=Decimal("0")) == "a"   # still inside
    clock.now += 0.001                                               # exactly 300 s after the pin
    assert tracker.choose("p", ["b", "a"], gain=Decimal("0.0100"), cache_loss=Decimal("0.0100")) == "a"
    assert tracker.stats()["refused_cost"] == 1
    clock.now += 300.0
    assert tracker.choose("p", ["b", "a"], gain=Decimal("0.0100"), cache_loss=Decimal("0.0099")) == "b"
    assert tracker.stats()["moved_gain"] == 1


def test_after_the_dwell_the_same_best_arm_just_renews_the_pin() -> None:
    clock = Clock()
    tracker = _tracker(clock)
    tracker.choose("p", ["a", "b"])
    clock.now += 400.0
    assert tracker.choose("p", ["a", "b"], gain=Decimal("9"), cache_loss=Decimal("0")) == "a"
    assert tracker.stats()["renewed"] == 1
    clock.now += 299.0
    assert tracker.choose("p", ["b", "a"], gain=Decimal("9"), cache_loss=Decimal("0")) == "a"   # new dwell holds


def test_money_must_not_be_a_float() -> None:
    tracker = _tracker(Clock())
    tracker.choose("p", ["a", "b"])
    with pytest.raises(TypeError, match="never float"):
        tracker.choose("p", ["a", "b"], gain=0.5)
    assert tracker.choose("p", ["a", "b"], gain="0.5", cache_loss=1) == "a"      # str and int are fine


def test_an_empty_key_is_not_pinned_and_empty_candidates_are_refused() -> None:
    tracker = _tracker(Clock())
    assert [tracker.choose("", ["a", "b"]), tracker.choose("", ["b", "a"])] == ["a", "b"]
    assert tracker.stats() == {"entries": 0, "unkeyed": 2}
    with pytest.raises(ValueError, match="at least one candidate"):
        tracker.choose("p", [])


def test_the_table_is_bounded_and_evicts_the_least_recently_used_key() -> None:
    tracker = _tracker(Clock(), max_entries=3)
    for key, arm in (("k1", "a1"), ("k2", "a2"), ("k3", "a3")):
        tracker.choose(key, [arm, "z"])
    assert tracker.choose("k1", ["z", "a1"]) == "a1"               # touching k1 makes k2 the oldest
    tracker.choose("k4", ["a4", "z"])
    assert tracker.stats() == {"entries": 3, "evicted": 1, "held": 1, "pinned": 4}
    assert tracker.choose("k2", ["new", "a2"]) == "new"            # k2 was evicted: pinned afresh
    assert tracker.choose("k4", ["z", "a4"]) == "a4"               # k4 survived


def test_the_default_table_holds_at_most_ten_thousand_keys() -> None:
    tracker = _tracker(Clock())
    for i in range(10_050):
        tracker.choose(f"k{i}", ["a"])
    assert tracker.stats() == {"entries": 10_000, "evicted": 50, "pinned": 10_050}
