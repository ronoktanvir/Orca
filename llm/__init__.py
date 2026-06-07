"""Swappable LLM client interface (§11). Phase 0 uses no real model."""

from __future__ import annotations

from .client import FallbackLLM, LLMClient, OpenAIClient, StubLLM, build_llm

__all__ = ["LLMClient", "StubLLM", "OpenAIClient", "FallbackLLM", "build_llm"]
