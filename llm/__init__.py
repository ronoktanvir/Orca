"""Swappable LLM client interface (§11). Phase 0 uses no real model."""

from __future__ import annotations

from .client import LLMClient, StubLLM

__all__ = ["LLMClient", "StubLLM"]
