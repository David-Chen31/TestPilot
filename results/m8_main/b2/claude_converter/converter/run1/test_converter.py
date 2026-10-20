from __future__ import annotations

from pathlib import Path

import pytest

from claude_converter.converter import Converter


# ---------------------------------------------------------------------------
# Monkeypatching strategy
#
# ``Converter`` imported the per-tool functions into its own module namespace
# at import time (e.g. ``claude_converter.converter.load_session_claude``).
# To dispatch-verify without touching real files, we monkeypatch those names
# on the converter module and assert each was called with the expected args.
# ---------------------------------------------------------------------------


@pytest.fixture
def patched(monkeypatch):
    """Replace every dispatched-to function in the converter module with a
    recorder that returns a sentinel uniquely identifying the target."""
    import claude_converter.converter as conv

    calls = {}

    def make(name):
        def _fn(*args, **kwargs):
            calls[name] = {"args": args, "kwargs": kwargs}
            return f"result:{name}"
        return _fn

    names = [
        "load_session_claude",
        "load_session_codex",
        "load_session_pi",
        "records_to_messages_claude",
        "records_to_messages_codex",
        "records_to_messages_pi",
        "session_to_messages_claude",
        "session_to_messages_codex",
        "session_to_messages_pi",
        "inspect_session_claude",
        "inspect_session_codex",
        "inspect_session_pi",
    ]
    for name in names:
        monkeypatch.setattr(conv, name, make(name))

    return calls


# ---------------------------------------------------------------------------
# __init__ / attribute
# ---------------------------------------------------------------------------


def test_default_converter_name():
    c = Converter()
    assert c.converter == "claude-code"


def test_explicit_converter_names():
    assert Converter("claude-code").converter == "claude-code"
    assert Converter("codex").converter == "codex"
    assert Converter("pi").converter == "pi"


def test_unknown_converter_accepted_at_construction():
    # The type hint is Literal, but nothing validates at runtime.
    c = Converter("bogus")
    assert c.converter == "bogus"


# ---------------------------------------------------------------------------
# load_session dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,fn",
    [
        ("claude-code", "load_session_claude"),
        ("codex", "load_session_codex"),
        ("pi", "load_session_pi"),
    ],
)
def test_load_session_dispatch(patched, name, fn):
    c = Converter(name)
    result = c.load_session("/some/path.jsonl")
    assert result == f"result:{fn}"
    assert patched[fn]["args"] == ("/some/path.jsonl",)
    assert patched[fn]["kwargs"] == {}


def test_load_session_accepts_path_object(patched):
    c = Converter("claude-code")
    p = Path("/tmp/x.jsonl")
    c.load_session(p)
    assert patched["load_session_claude"]["args"] == (p,)


@pytest.mark.parametrize("method", ["load_session", "records_to_messages", "session_to_messages", "inspect_session"])
def test_unknown_converter_raises(patched, method):
    c = Converter("bogus")
    if method == "load_session":
        args = ("p",)
    elif method == "records_to_messages":
        args = ([],)
    elif method == "session_to_messages":
        args = ("p",)
    else:
        args = ("p",)
    with pytest.raises(ValueError, match="Unknown converter: 'bogus'"):
        getattr(c, method)(*args)
    # Nothing should have been recorded for an unknown converter.
    assert patched == {}


def test_unknown_converter_error_message_contains_repr():
    c = Converter(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError) as excinfo:
        c.load_session("p")
    assert str(excinfo.value) == "Unknown converter: 123"


# ---------------------------------------------------------------------------
# records_to_messages dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,fn",
    [
        ("claude-code", "records_to_messages_claude"),
        ("codex", "records_to_messages_codex"),
        ("pi", "records_to_messages_pi"),
    ],
)
def test_records_to_messages_dispatch(patched, name, fn):
    c = Converter(name)
    records = [{"type": "user"}]
    result = c.records_to_messages(records)
    assert result == f"result:{fn}"
    assert patched[fn]["args"] == (records,)
    assert patched[fn]["kwargs"] == {}


# ---------------------------------------------------------------------------
# session_to_messages dispatch (output kwarg)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,fn",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_dispatch_default_output(patched, name, fn):
    c = Converter(name)
    result = c.session_to_messages("in.jsonl")
    assert result == f"result:{fn}"
    assert patched[fn]["args"] == ("in.jsonl",)
    assert patched[fn]["kwargs"] == {"output": None}


@pytest.mark.parametrize(
    "name,fn",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_dispatch_with_output(patched, name, fn, tmp_path):
    c = Converter(name)
    out = tmp_path / "out.json"
    result = c.session_to_messages("in.jsonl", output=out)
    assert result == f"result:{fn}"
    assert patched[fn]["args"] == ("in.jsonl",)
    assert patched[fn]["kwargs"] == {"output": out}


# ---------------------------------------------------------------------------
# inspect_session dispatch (all flags passed through)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,fn",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_dispatch_defaults(patched, name, fn):
    c = Converter(name)
    result = c.inspect_session("s.jsonl")
    assert result == f"result:{fn}"
    assert patched[fn]["args"] == ("s.jsonl",)
    assert patched[fn]["kwargs"] == {
        "show_flow": False,
        "show_blocks": False,
        "show_raw": False,
    }


@pytest.mark.parametrize(
    "name,fn",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_dispatch_custom_flags(patched, name, fn):
    c = Converter(name)
    c.inspect_session("s.jsonl", show_flow=True, show_blocks=True, show_raw=True)
    assert patched[fn]["kwargs"] == {
        "show_flow": True,
        "show_blocks": True,
        "show_raw": True,
    }


def test_inspect_session_partial_flags(patched):
    c = Converter("codex")
    c.inspect_session("s.jsonl", show_flow=True)
    assert patched["inspect_session_codex"]["kwargs"] == {
        "show_flow": True,
        "show_blocks": False,
        "show_raw": False,
    }
