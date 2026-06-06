"""Coordinate-leak invariant guard (§3.2). See ``coord_leak_test.py``."""

from __future__ import annotations

from .coord_leak_test import assert_no_coord_leak, scan_for_leaks

__all__ = ["assert_no_coord_leak", "scan_for_leaks"]
