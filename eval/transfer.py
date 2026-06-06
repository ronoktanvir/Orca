"""Transfer eval — the money plot (§9) — Stream 3 (O7).

Train Full C2 on {A,T2,T3}; freeze; evaluate all three conditions on held-out
{B,C}. Claim: Full C2 >= baselines on held-out => transferable strategy, not
terrain. Held-out seeds are never trained on. Phase 0 placeholder.
"""

from __future__ import annotations

__all__: list[str] = []
