"""Comm bus: t+1 delivery, addressed/own filtering, window bound, verbatim log (§5.2)."""

from __future__ import annotations

from bus import CommBus, make_message
from contracts.enums import MessageType


def _m(frm, to, content, rnd, type=MessageType.REPORT):
    return make_message(frm, to, type, content, rnd)


def test_posted_message_delivered_next_round_only():
    bus = CommBus(window=8)
    bus.post(_m("agent_1", "team", "found caves N", 0))
    # not visible in the same round (before tick)
    assert bus.recent_for("agent_2") == []
    bus.tick()
    delivered = bus.recent_for("agent_2")
    assert len(delivered) == 1 and delivered[0].content == "found caves N"


def test_sender_does_not_see_own_message():
    bus = CommBus(window=8)
    bus.post(_m("agent_1", "team", "hello", 0))
    bus.tick()
    assert bus.recent_for("agent_1") == []          # excludes own
    assert len(bus.recent_for("agent_2")) == 1       # teammate sees it


def test_addressed_message_only_to_recipient():
    bus = CommBus(window=8)
    bus.post(_m("agent_1", "agent_3", "for you", 0))
    bus.tick()
    assert bus.recent_for("agent_2") == []           # not addressed
    assert len(bus.recent_for("agent_3")) == 1


def test_window_bounds_recent_messages():
    bus = CommBus(window=2)
    for i in range(4):
        bus.post(_m("agent_1", "team", f"m{i}", 0))
    bus.tick()
    recent = bus.recent_for("agent_2")
    assert [m.content for m in recent] == ["m2", "m3"]  # only the last `window`


def test_log_keeps_all_messages_verbatim():
    bus = CommBus(window=2)
    for i in range(5):
        bus.post(_m("agent_1", "team", f"m{i}", 0))
    assert len(bus.log) == 5  # full verbatim log for Weave + Orca trace


def test_recent_message_content_not_truncated():
    bus = CommBus(window=8)
    long_content = "iron-rich caves to the N; bring a stone pickaxe and torches " * 5
    bus.post(_m("agent_1", "team", long_content, 0))
    bus.tick()
    delivered = bus.recent_for("agent_2")
    assert delivered[0].content == long_content  # full content, never truncated


def test_recent_for_bounded_to_window_after_many_messages():
    bus = CommBus(window=3)
    for r in range(10):  # one message per round, across many rounds
        bus.post(_m("agent_1", "team", f"m{r}", r))
        bus.tick()
    recent = bus.recent_for("agent_2")
    assert len(recent) == 3  # bounded to K across the whole delivered history
    assert [m.content for m in recent] == ["m7", "m8", "m9"]


def test_recent_for_dedupes_identical_messages():
    bus = CommBus(window=8)
    bus.post(_m("agent_1", "team", "dup", 0))
    bus.post(_m("agent_1", "team", "dup", 0))  # identical -> surfaced once
    bus.tick()
    assert len(bus.recent_for("agent_2")) == 1


def test_message_persists_in_sliding_window_across_ticks():
    # Stream 2 brief: "each agent sees the last K addressed to it or team" — the
    # window spans delivered history, so a message stays visible across subsequent
    # ticks (until K newer messages evict it), not just the single tick after it.
    bus = CommBus(window=8)
    bus.post(_m("agent_1", "team", "once", 0))
    bus.tick()
    assert len(bus.recent_for("agent_2")) == 1
    bus.tick()  # nothing posted in the interim
    assert [m.content for m in bus.recent_for("agent_2")] == ["once"]  # still in window


def test_window_evicts_oldest_across_history():
    bus = CommBus(window=2)
    bus.post(_m("agent_1", "team", "m0", 0))
    bus.tick()
    bus.post(_m("agent_1", "team", "m1", 1))
    bus.tick()
    bus.post(_m("agent_1", "team", "m2", 2))
    bus.tick()
    # last K across history, oldest evicted
    assert [m.content for m in bus.recent_for("agent_2")] == ["m1", "m2"]
