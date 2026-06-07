#!/usr/bin/env python3
"""Fork-gate checker (workflow plan §1 / §9).

Operationalizes the "give it off" checklist: prints each gate item with a
PASS/FAIL and exits non-zero if any fails. Run from the repo root:

    python scripts/check_fork_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _check(label: str, fn) -> bool:
    try:
        ok, detail = fn()
    except Exception as exc:  # pragma: no cover - defensive
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def gate_contracts_frozen():
    from contracts import CONTRACTS
    from tests.fixtures import SAMPLES

    assert len(CONTRACTS) == 7
    for name, (cls, factory) in SAMPLES.items():
        inst = factory()
        # round-trips through JSON (validates the frozen schema)
        cls.model_validate_json(inst.model_dump_json(by_alias=True))
    return True, f"{len(CONTRACTS)} contracts import + sample-validate + round-trip"


def gate_episode_reaches_iron():
    from config import load_config
    from telemetry import init_telemetry
    from train.loop import run

    settings = load_config()
    settings.telemetry.mode = "off"
    (trace, metrics), = run(settings, telemetry=init_telemetry(mode="off"))
    assert metrics.frontier_milestone.value == "iron", metrics.frontier_milestone
    assert trace.terminated_reason == "frontier_target"
    return True, f"frontier={metrics.frontier_milestone.value}, reward={metrics.team_reward:.3f}, rounds={trace.n_rounds}"


def gate_episode_logs_trace():
    from config import load_config
    from telemetry import init_telemetry
    from train.loop import run

    settings = load_config()
    settings.telemetry.mode = "local"
    tmp_dir = REPO_ROOT / "runs"
    tel = init_telemetry(mode="local", run_dir=str(tmp_dir), run_id="forkgate_check")
    run(settings, telemetry=tel)
    ep = tmp_dir / "forkgate_check" / "episode_0000.json"
    assert ep.exists(), "episode artifact not written"
    return True, f"nested trace + EpisodeTrace/EpisodeMetrics logged ({tel.backend})"


def gate_coord_leak():
    from obs_guard.coord_leak_test import _run_oracle_episode, assert_no_coord_leak

    _env, snapshots = _run_oracle_episode()
    assert snapshots, "no observations produced"
    for i, obs in enumerate(snapshots):
        assert_no_coord_leak(obs, path=f"obs[{i}]")
    return True, f"{len(snapshots)} observations scanned, no leaks"


def gate_folder_ownership():
    expected = ["env", "reward", "agents", "bus", "orca", "eval", "telemetry", "contracts", "configs"]
    missing = [d for d in expected if not (REPO_ROOT / d).is_dir()]
    assert not missing, f"missing folders: {missing}"
    assert (REPO_ROOT / "configs" / "default.yaml").exists()
    return True, "stream folders + configs/default.yaml present"


def main() -> int:
    print("ORCA — Fork Gate checklist\n")
    results = [
        _check("The 7 contracts are committed and frozen", gate_contracts_frozen),
        _check("A shallow episode runs end-to-end and reaches iron", gate_episode_reaches_iron),
        _check("Episode logs a nested trace (Weave or local fallback)", gate_episode_logs_trace),
        _check("coord_leak_test passes", gate_coord_leak),
        _check("Folder ownership + config structure agreed", gate_folder_ownership),
    ]
    all_green = all(results)
    print("\n" + ("FORK GATE: GREEN ✅ — Phase 0 foundation invariants intact." if all_green
                  else "FORK GATE: RED ❌ — see failures above."))
    return 0 if all_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
