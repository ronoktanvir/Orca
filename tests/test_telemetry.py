"""Telemetry: safe fallback (no creds/network) + lazy @op identity (§10)."""

from __future__ import annotations

from telemetry import backend, init_telemetry, op


def test_init_off_is_noop():
    t = init_telemetry(mode="off")
    assert t.backend == "noop"
    assert backend() == "noop"
    assert t.log_episode(object(), object()) is None  # no-op, returns None


def test_init_local_writes_dir(tmp_path):
    t = init_telemetry(mode="local", run_dir=str(tmp_path), run_id="r1")
    assert t.backend == "local"
    assert (tmp_path / "r1").exists()
    t.log_event("hello", {"x": 1})
    assert (tmp_path / "r1" / "events.jsonl").exists()


def test_op_decorator_is_identity_offline():
    init_telemetry(mode="off")

    @op
    def add(a, b):
        return a + b

    @op(name="mul")
    def mul(a, b):
        return a * b

    assert add(2, 3) == 5
    assert mul(2, 3) == 6
    # nesting works (op inside op)
    @op
    def outer(x):
        return add(x, mul(x, 2))

    assert outer(4) == 4 + 4 * 2


def test_auto_falls_back_to_local_without_creds(monkeypatch, tmp_path):
    # No W&B credential and no netrc -> auto must resolve to local, never raise.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "no_netrc"))
    t = init_telemetry(mode="auto", run_dir=str(tmp_path))
    assert t.backend == "local"
