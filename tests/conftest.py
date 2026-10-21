"""Keep provider SDKs from seeing a real key during the suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hide_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
