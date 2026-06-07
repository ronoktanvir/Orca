"""Weave instrumentation with a safe fallback (§10).

Everything load-bearing is decorated with :func:`op` so Weave traces nest
automatically (§10). But Weave/W&B may be absent or unauthenticated — so:

  * ``op`` is a *lazy* decorator: it wraps a function and only routes through
    ``weave.op`` once telemetry has successfully initialized in "weave" mode.
    Decoration order therefore never matters, and nesting still works because
    the wrapped op is created/called inside the parent op's call.
  * :func:`init_telemetry` chooses a backend and **never raises**: if Weave
    can't init (not installed, no creds), it falls back to local-JSONL logging,
    or to a pure no-op.

This is what lets ``python run.py`` and ``pytest`` run fully offline.
"""

from __future__ import annotations

import functools
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# Module state set by init_telemetry().
_BACKEND: str = "noop"  # "weave" | "local" | "noop"
_weave: Any = None
_op_cache: dict[Callable, Callable] = {}


def backend() -> str:
    return _BACKEND


def op(fn: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Lazy ``@weave.op`` — identity unless telemetry is live in "weave" mode."""

    def deco(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _BACKEND == "weave" and _weave is not None:
                wrapped = _op_cache.get(f)
                if wrapped is None:
                    try:
                        wrapped = _weave.op(f)
                    except Exception:
                        wrapped = f
                    _op_cache[f] = wrapped
                return wrapped(*args, **kwargs)
            return f(*args, **kwargs)

        wrapper.__orca_op__ = True  # type: ignore[attr-defined]
        return wrapper

    return deco(fn) if fn is not None else deco


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Telemetry:
    """A tiny logger that always produces local artifacts (unless mode 'off')."""

    def __init__(self, backend_mode: str, run_dir: Optional[str], run_id: str) -> None:
        self.backend = backend_mode
        self.run_id = run_id
        self.dir: Optional[Path] = None
        self._events_path: Optional[Path] = None
        # Serializes the local-JSONL writes so the concurrent (asyncio) worker
        # path can log from worker threads without interleaving lines (§5/§10).
        self._write_lock = threading.Lock()
        if backend_mode != "noop" and run_dir is not None:
            self.dir = Path(run_dir) / run_id
            self.dir.mkdir(parents=True, exist_ok=True)
            self._events_path = self.dir / "events.jsonl"

    def log_event(self, name: str, data: dict[str, Any]) -> None:
        if self._events_path is None:
            return
        record = {"ts": _now_stamp(), "event": name, **data}
        line = json.dumps(record, default=str) + "\n"
        with self._write_lock, self._events_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def log_episode(self, trace: Any, metrics: Any) -> Optional[Path]:
        """Write an episode's trace + metrics as JSON; return the path (or None)."""
        if self.dir is None:
            return None
        idx = getattr(trace, "episode_idx", 0)
        path = self.dir / f"episode_{idx:04d}.json"
        payload = {
            "trace": trace.model_dump(mode="json", by_alias=True),
            "metrics": metrics.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self.log_event(
            "episode_logged",
            {
                "episode_idx": idx,
                "frontier": metrics.frontier_milestone.value,
                "team_reward": metrics.team_reward,
            },
        )
        return path

    def summary(self) -> str:
        where = f" -> {self.dir}" if self.dir else ""
        return f"telemetry backend={self.backend}{where}"


def _has_wandb_cred() -> bool:
    """True if a W&B credential is detectable (env var or netrc) — so we can use
    Weave without triggering an interactive login prompt."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path(os.path.expanduser("~/.netrc"))
    if netrc.exists():
        try:
            return "wandb" in netrc.read_text(encoding="utf-8").lower()
        except Exception:
            return False
    return False


def init_telemetry(
    *,
    mode: str = "auto",
    entity: Optional[str] = None,
    project: str = "orca",
    run_dir: str = "runs",
    run_id: Optional[str] = None,
) -> Telemetry:
    """Choose a telemetry backend; never raises.

    mode: "auto" | "weave" | "local" | "off".
      - off   : no files, identity ops.
      - local : local-JSONL artifacts, identity ops.
      - weave : try Weave; fall back to local on any failure.
      - auto  : weave iff importable AND a W&B credential is present, else local.

    entity: W&B entity (team) to log under. Weave needs ``entity/project``; with
      ``None`` it relies on the account's default entity, which is unset for
      fresh accounts ("could not determine a W&B entity"). Set it explicitly.
    """
    global _BACKEND, _weave
    run_id = run_id or _now_stamp()
    _weave = None

    if mode == "off":
        _BACKEND = "noop"
        return Telemetry("noop", None, run_id)

    # Gate Weave on a *detectable* credential for BOTH "weave" and "auto" — a
    # missing credential makes weave.init() drop into an interactive login
    # prompt that would HANG a non-interactive run. "credentials absent ->
    # fallback" is exactly §10's contract.
    if mode in ("weave", "auto"):
        if _has_wandb_cred():
            try:
                import weave as _w  # type: ignore

                target = f"{entity}/{project}" if entity else project
                _w.init(target)
                _weave = _w
                _BACKEND = "weave"
                return Telemetry("weave", run_dir, run_id)
            except Exception as exc:  # pragma: no cover - network/cred dependent
                print(f"[telemetry] weave unavailable ({exc!s:.120}); falling back to local")
        elif mode == "weave":
            print(
                "[telemetry] weave requested but no W&B credential found "
                "(set WANDB_API_KEY); falling back to local"
            )

    _BACKEND = "local"
    return Telemetry("local", run_dir, run_id)


__all__ = ["op", "init_telemetry", "Telemetry", "backend"]
