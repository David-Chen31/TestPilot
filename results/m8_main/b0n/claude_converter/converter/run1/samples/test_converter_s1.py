from __future__ import annotations

from pathlib import Path

import pytest

from claude_converter.converter import Converter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sentinel(name: str):
    """Return a callable that records its args/kwargs and returns a marker."""
    record = {"calls": []}

    def _fn(*args, **kwargs):
        record["calls"].append((args, kwargs))
        return {"sentinel": name}

    _fn.record = record
    return _fn


# ---------------------------------------------------------------------------
# Construction / attribute
# ---------------------------------------------------------------------------

def test_default_converter_is_claude_code():
    conv = Converter()
    assert conv.converter == "claude-code"


@pytest.mark.parametrize("name", ["claude-code", "codex", "pi"])
def test_converter_stores_name(name):
    conv = Converter(name)
    assert conv.converter == name


# ---------------------------------------------------------------------------
# load_session dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "load_session_claude"),
        ("codex", "load_session_codex"),
        ("pi", "load_session_pi"),
    ],
)
def test_load_session_dispatches(monkeypatch, name, target):
    fake = _make_sentinel(target)
    monkeypatch.setattr(f"claude_converter.converter.{target}", fake)

    conv = Converter(name)
    result = conv.load_session("path/to/session.jsonl")

    assert result == {"sentinel": target}
    assert fake.record["calls"] == [(("path/to/session.jsonl",), {})]


def test_load_session_accepts_path_object(monkeypatch):
    fake = _make_sentinel("load_session_claude")
    monkeypatch.setattr("claude_converter.converter.load_session_claude", fake)

    p = Path("/tmp/some/file.jsonl")
    result = Converter("claude-code").load_session(p)

    assert result == {"sentinel": "load_session_claude"}
    args, kwargs = fake.record["calls"][0]
    assert args == (p,)
    assert kwargs == {}


def test_load_session_unknown_converter_raises():
    conv = Converter("claude-code")
    conv.converter = "bogus"  # bypass Literal typing to hit the default branch
    with pytest.raises(ValueError) as excinfo:
        conv.load_session("x.jsonl")
    assert "Unknown converter" in str(excinfo.value)
    assert "'bogus'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# records_to_messages dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "records_to_messages_claude"),
        ("codex", "records_to_messages_codex"),
        ("pi", "records_to_messages_pi"),
    ],
)
def test_records_to_messages_dispatches(monkeypatch, name, target):
    fake = _make_sentinel(target)
    monkeypatch.setattr(f"claude_converter.converter.{target}", fake)

    records = [{"type": "user"}, {"type": "assistant"}]
    result = Converter(name).records_to_messages(records)

    assert result == {"sentinel": target}
    assert fake.record["calls"] == [((records,), {})]


def test_records_to_messages_unknown_converter_raises():
    conv = Converter("claude-code")
    conv.converter = "nope"
    with pytest.raises(ValueError) as excinfo:
        conv.records_to_messages([])
    assert "Unknown converter" in str(excinfo.value)
    assert "'nope'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# session_to_messages dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_dispatches_without_output(monkeypatch, name, target):
    fake = _make_sentinel(target)
    monkeypatch.setattr(f"claude_converter.converter.{target}", fake)

    result = Converter(name).session_to_messages("session.jsonl")

    assert result == {"sentinel": target}
    assert fake.record["calls"] == [(("session.jsonl",), {"output": None})]


@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_dispatches_with_output(monkeypatch, name, target):
    fake = _make_sentinel(target)
    monkeypatch.setattr(f"claude_converter.converter.{target}", fake)

    out = Path("/tmp/out/messages.json")
    result = Converter(name).session_to_messages("session.jsonl", output=out)

    assert result == {"sentinel": target}
    assert fake.record["calls"] == [(("session.jsonl",), {"output": out})]


def test_session_to_messages_unknown_converter_raises():
    conv = Converter("claude-code")
    conv.converter = "unknown"
    with pytest.raises(ValueError) as excinfo:
        conv.session_to_messages("session.jsonl")
    assert "Unknown converter" in str(excinfo.value)
    assert "'unknown'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# inspect_session dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_dispatches_defaults(monkeypatch, name, target):
    fake = _make_sentinel(target)
    monkeypatch.setattr(f"claude_converter.converter.{target}", fake)

    result = Converter(name).inspect_session("session.jsonl")

    # inspect_session returns whatever the target returns (None normally);
    # here we verify forwarding of default flags.
    assert result == {"sentinel": target}
    assert fake.record["calls"] == [
        (
            ("session.jsonl",),
            {"show_flow": False, "show_blocks": False, "show_raw": False},
        )
    ]


@pytest.mark.parametrize(
    "name, target",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_dispatches_flags(monkeypatch, name, target):
    fake = _make_sentinel(target)
    monkeypatch.setattr(f"claude_converter.converter.{target}", fake)

    Converter(name).inspect_session(
        "session.jsonl", show_flow=True, show_blocks=True, show_raw=True
    )

    assert fake.record["calls"] == [
        (
            ("session.jsonl",),
            {"show_flow": True, "show_blocks": True, "show_raw": True},
        )
    ]


def test_inspect_session_accepts_path(monkeypatch):
    fake = _make_sentinel("inspect_session_codex")
    monkeypatch.setattr("claude_converter.converter.inspect_session_codex", fake)

    p = Path("/tmp/x/session.jsonl")
    Converter("codex").inspect_session(p, show_flow=True)

    args, kwargs = fake.record["calls"][0]
    assert args == (p,)
    assert kwargs == {"show_flow": True, "show_blocks": False, "show_raw": False}


def test_inspect_session_unknown_converter_raises():
    conv = Converter("claude-code")
    conv.converter = "mystery"
    with pytest.raises(ValueError) as excinfo:
        conv.inspect_session("session.jsonl")
    assert "Unknown converter" in str(excinfo.value)
    assert "'mystery'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Dispatcher does not cross-wire converters
# ---------------------------------------------------------------------------

def test_dispatch_isolated_per_converter(monkeypatch):
    calls = []
    for target in [
        "load_session_claude",
        "load_session_codex",
        "load_session_pi",
    ]:
        def _make(t=target):
            def _fn(*args, **kwargs):
                calls.append(t)
                return []
            return _fn
        monkeypatch.setattr(f"claude_converter.converter.{target}", _make())

    Converter("codex").load_session("p")
    Converter("pi").load_session("p")
    Converter("claude-code").load_session("p")

    assert calls == ["load_session_codex", "load_session_pi", "load_session_claude"]
