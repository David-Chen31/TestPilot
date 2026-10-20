from __future__ import annotations

from pathlib import Path

import pytest

from claude_converter.converter import Converter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_converter_is_claude_code():
    c = Converter()
    assert c.converter == "claude-code"


@pytest.mark.parametrize("name", ["claude-code", "codex", "pi"])
def test_constructor_accepts_valid_names(name):
    c = Converter(name)
    assert c.converter == name


def test_constructor_accepts_unknown_name():
    # The constructor does not validate; validation happens per-call.
    c = Converter("bogus")  # type: ignore[arg-type]
    assert c.converter == "bogus"


# ---------------------------------------------------------------------------
# load_session
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "claude_converter.converter.load_session_claude"),
        ("codex", "claude_converter.converter.load_session_codex"),
        ("pi", "claude_converter.converter.load_session_pi"),
    ],
)
def test_load_session_routes(monkeypatch, name, target):
    calls = []

    def fake(path):
        calls.append(path)
        return [{"route": name}]

    monkeypatch.setattr(target, fake)

    c = Converter(name)
    result = c.load_session("some/path.jsonl")

    assert result == [{"route": name}]
    assert calls == ["some/path.jsonl"]


def test_load_session_unknown_raises():
    c = Converter("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown converter: 'nope'"):
        c.load_session("x.jsonl")


# ---------------------------------------------------------------------------
# records_to_messages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "claude_converter.converter.records_to_messages_claude"),
        ("codex", "claude_converter.converter.records_to_messages_codex"),
        ("pi", "claude_converter.converter.records_to_messages_pi"),
    ],
)
def test_records_to_messages_routes(monkeypatch, name, target):
    records = [{"a": 1}, {"b": 2}]
    calls = []

    def fake(recs):
        calls.append(recs)
        return [{"role": "user", "content": name}]

    monkeypatch.setattr(target, fake)

    c = Converter(name)
    result = c.records_to_messages(records)

    assert result == [{"role": "user", "content": name}]
    assert calls == [records]


def test_records_to_messages_unknown_raises():
    c = Converter("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown converter: 'nope'"):
        c.records_to_messages([])


# ---------------------------------------------------------------------------
# session_to_messages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "claude_converter.converter.session_to_messages_claude"),
        ("codex", "claude_converter.converter.session_to_messages_codex"),
        ("pi", "claude_converter.converter.session_to_messages_pi"),
    ],
)
def test_session_to_messages_routes_with_output(monkeypatch, name, target):
    calls = []

    def fake(path, output=None):
        calls.append((path, output))
        return [{"role": "assistant", "content": name}]

    monkeypatch.setattr(target, fake)

    c = Converter(name)
    result = c.session_to_messages("in.jsonl", output="out.json")

    assert result == [{"role": "assistant", "content": name}]
    assert calls == [("in.jsonl", "out.json")]


@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "claude_converter.converter.session_to_messages_claude"),
        ("codex", "claude_converter.converter.session_to_messages_codex"),
        ("pi", "claude_converter.converter.session_to_messages_pi"),
    ],
)
def test_session_to_messages_routes_default_output(monkeypatch, name, target):
    calls = []

    def fake(path, output=None):
        calls.append((path, output))
        return []

    monkeypatch.setattr(target, fake)

    c = Converter(name)
    result = c.session_to_messages(Path("in.jsonl"))

    assert result == []
    assert calls == [(Path("in.jsonl"), None)]


def test_session_to_messages_unknown_raises():
    c = Converter("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown converter: 'nope'"):
        c.session_to_messages("x.jsonl")


# ---------------------------------------------------------------------------
# inspect_session
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "claude_converter.converter.inspect_session_claude"),
        ("codex", "claude_converter.converter.inspect_session_codex"),
        ("pi", "claude_converter.converter.inspect_session_pi"),
    ],
)
def test_inspect_session_routes_defaults(monkeypatch, name, target):
    calls = []

    def fake(path, show_flow=False, show_blocks=False, show_raw=False):
        calls.append(
            {"path": path, "show_flow": show_flow, "show_blocks": show_blocks, "show_raw": show_raw}
        )

    monkeypatch.setattr(target, fake)

    c = Converter(name)
    result = c.inspect_session("in.jsonl")

    assert result is None
    assert calls == [
        {"path": "in.jsonl", "show_flow": False, "show_blocks": False, "show_raw": False}
    ]


@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "claude_converter.converter.inspect_session_claude"),
        ("codex", "claude_converter.converter.inspect_session_codex"),
        ("pi", "claude_converter.converter.inspect_session_pi"),
    ],
)
def test_inspect_session_passes_flags(monkeypatch, name, target):
    calls = []

    def fake(path, show_flow=False, show_blocks=False, show_raw=False):
        calls.append((path, show_flow, show_blocks, show_raw))

    monkeypatch.setattr(target, fake)

    c = Converter(name)
    c.inspect_session("in.jsonl", show_flow=True, show_blocks=True, show_raw=True)

    assert calls == [("in.jsonl", True, True, True)]


@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "claude_converter.converter.inspect_session_claude"),
        ("codex", "claude_converter.converter.inspect_session_codex"),
        ("pi", "claude_converter.converter.inspect_session_pi"),
    ],
)
def test_inspect_session_mixed_flags(monkeypatch, name, target):
    calls = []

    def fake(path, show_flow=False, show_blocks=False, show_raw=False):
        calls.append((show_flow, show_blocks, show_raw))

    monkeypatch.setattr(target, fake)

    c = Converter(name)
    c.inspect_session("in.jsonl", show_flow=True)

    assert calls == [(True, False, False)]


def test_inspect_session_unknown_raises():
    c = Converter("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown converter: 'nope'"):
        c.inspect_session("x.jsonl")
