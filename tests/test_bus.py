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


def test_tick_clears_after_delivery():
    bus = CommBus(window=8)
    bus.post(_m("agent_1", "team", "once", 0))
    bus.tick()
    assert len(bus.recent_for("agent_2")) == 1
    bus.tick()  # nothing posted in the interim
    assert bus.recent_for("agent_2") == []
