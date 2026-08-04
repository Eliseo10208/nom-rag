"""HTTP-layer unit tests that do not load the model or call a provider."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from nom_rag import api


class _BusySlots:
    def acquire(self, timeout: int) -> bool:
        return False

    def release(self) -> None:  # pragma: no cover - release would be a bug here
        raise AssertionError("released a slot that was never acquired")


class _OpenSlots:
    def __init__(self) -> None:
        self.released = False

    def acquire(self, timeout: int) -> bool:
        return True

    def release(self) -> None:
        self.released = True


def test_favicon_reuses_assistant_cat_asset():
    response = api.favicon()

    assert response.path.name == "assistant-cat-cutout.png"
    assert response.media_type == "image/png"


def test_chat_returns_503_when_capacity_queue_times_out(monkeypatch):
    monkeypatch.setattr(api, "_chat_slots", _BusySlots())
    monkeypatch.setitem(api._state, "retriever", object())
    monkeypatch.setitem(api._state, "client", object())

    with pytest.raises(HTTPException) as excinfo:
        api.chat(api.ChatRequest(question="¿Qué dice la NOM-035?"))

    assert excinfo.value.status_code == 503
    assert "consultas en curso" in excinfo.value.detail


def test_chat_releases_capacity_after_provider_error(monkeypatch):
    slots = _OpenSlots()
    monkeypatch.setattr(api, "_chat_slots", slots)
    monkeypatch.setitem(api._state, "retriever", object())
    monkeypatch.setitem(api._state, "client", object())
    monkeypatch.setattr(
        "nom_rag.pipeline.ask",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider down")),
    )

    with pytest.raises(HTTPException) as excinfo:
        api.chat(api.ChatRequest(question="¿Qué dice la NOM-035?"))

    assert excinfo.value.status_code == 502
    assert slots.released is True
