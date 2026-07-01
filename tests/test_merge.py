"""Tests for adapter merge orchestration."""

from __future__ import annotations

from easylora.lora import merge as merge_mod


def test_merge_adapter_saves_model_and_tokenizer(monkeypatch, tmp_path):
    calls: list[str] = []

    class _Merged:
        def save_pretrained(self, path):
            calls.append(f"model:{path.name}")

    class _Peft:
        def merge_and_unload(self):
            calls.append("merge")
            return _Merged()

    class _Tokenizer:
        def save_pretrained(self, path):
            calls.append(f"tokenizer:{path.name}")

    monkeypatch.setattr(merge_mod, "load_adapter", lambda *_args, **_kwargs: _Peft())
    monkeypatch.setattr(merge_mod, "load_tokenizer", lambda _cfg: _Tokenizer())

    out = merge_mod.merge_adapter("base", "adapter", tmp_path / "merged", device_map=None)

    assert out == tmp_path / "merged"
    assert calls == ["merge", "model:merged", "tokenizer:merged"]
