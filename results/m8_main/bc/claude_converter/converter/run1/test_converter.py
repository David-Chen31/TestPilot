from __future__ import annotations

from pathlib import Path

import pytest

from claude_converter import converter as converter_module
from claude_converter.converter import Converter


# ---------------------------------------------------------------------------
# dispatch tests
# ---------------------------------------------------------------------------
#
# Every method of Converter routes to one of three per-tool functions. We
# monkeypatch the module-level names that `converter` imported, then assert
# that (a) the right function was called, (b) with the right arguments, and
# (c) that its return value flows back to the caller.

TOOL_PREFIX = {
    "claude-code": "claude",
    "codex": "codex",
    "pi": "pi",
}


@pytest.fixture
def stubs(monkeypatch):
    """Replace every tool-specific function with a recording stub."""
    calls: dict[str, list[tuple[str, tuple, dict]]] = {}

    def make(name):
        def stub(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return {"called": name, "args": args, "kwargs": kwargs}

        return stub

    for tool, prefix in TOOL_PREFIX.items():
        for method in (
            "load_session",
            "records_to_messages",
            "session_to_messages",
            "inspect_session",
        ):
            fn_name = f"{method}_{prefix}"
            monkeypatch.setattr(converter_module, fn_name, make(fn_name))

    return calls


# --- __init__ -------------------------------------------------------------


def test_default_converter_is_claude_code():
    assert Converter().converter == "claude-code"


@pytest.mark.parametrize("name", ["claude-code", "codex", "pi"])
def test_init_stores_converter_name(name):
    assert Converter(name).converter == name


def test_init_accepts_any_value_without_validation():
    # No validation happens in __init__; unknown values surface later.
    c = Converter("nope")
    assert c.converter == "nope"


# --- load_session ---------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_fn",
    [
        ("claude-code", "load_session_claude"),
        ("codex", "load_session_codex"),
        ("pi", "load_session_pi"),
    ],
)
def test_load_session_dispatches(stubs, name, expected_fn):
    path = "/some/session.jsonl"
    result = Converter(name).load_session(path)

    assert expected_fn in stubs
    args, kwargs = stubs[expected_fn][0]
    assert args == (path,)
    assert kwargs == {}
    assert result == {"called": expected_fn, "args": (path,), "kwargs": {}}


def test_load_session_accepts_path_object(stubs):
    path = Path("x.jsonl")
    Converter("codex").load_session(path)
    assert stubs["load_session_codex"][0][0] == (path,)


def test_load_session_unknown_converter_raises():
    with pytest.raises(ValueError, match="Unknown converter"):
        Converter("bogus").load_session("x.jsonl")


def test_load_session_unknown_converter_error_message():
    with pytest.raises(ValueError) as excinfo:
        Converter("bogus").load_session("x.jsonl")
    assert str(excinfo.value) == "Unknown converter: 'bogus'"


# --- records_to_messages --------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_fn",
    [
        ("claude-code", "records_to_messages_claude"),
        ("codex", "records_to_messages_codex"),
        ("pi", "records_to_messages_pi"),
    ],
)
def test_records_to_messages_dispatches(stubs, name, expected_fn):
    records = [{"type": "user"}]
    result = Converter(name).records_to_messages(records)

    args, kwargs = stubs[expected_fn][0]
    assert args == (records,)
    assert kwargs == {}
    assert result["called"] == expected_fn


def test_records_to_messages_passes_same_object(stubs):
    records = [{"a": 1}]
    Converter("pi").records_to_messages(records)
    assert stubs["records_to_messages_pi"][0][0][0] is records


def test_records_to_messages_unknown_converter_raises():
    with pytest.raises(ValueError, match="Unknown converter"):
        Converter("bogus").records_to_messages([])


# --- session_to_messages --------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_fn",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_dispatches_with_default_output(stubs, name, expected_fn):
    path = "session.jsonl"
    result = Converter(name).session_to_messages(path)

    args, kwargs = stubs[expected_fn][0]
    assert args == (path,)
    assert kwargs == {"output": None}
    assert result["called"] == expected_fn


@pytest.mark.parametrize(
    "name,expected_fn",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_dispatches_with_output(stubs, name, expected_fn):
    path = Path("in.jsonl")
    output = Path("out.json")
    result = Converter(name).session_to_messages(path, output=output)

    args, kwargs = stubs[expected_fn][0]
    assert args == (path,)
    assert kwargs == {"output": output}
    assert result["called"] == expected_fn


def test_session_to_messages_unknown_converter_raises():
    with pytest.raises(ValueError, match="Unknown converter"):
        Converter("bogus").session_to_messages("x.jsonl")


# --- inspect_session ------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_fn",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_dispatches_defaults(stubs, name, expected_fn):
    path = "session.jsonl"
    result = Converter(name).inspect_session(path)

    args, kwargs = stubs[expected_fn][0]
    assert args == (path,)
    assert kwargs == {"show_flow": False, "show_blocks": False, "show_raw": False}
    assert result["called"] == expected_fn


@pytest.mark.parametrize(
    "name,expected_fn",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_dispatches_with_flags(stubs, name, expected_fn):
    path = Path("session.jsonl")
    result = Converter(name).inspect_session(
        path, show_flow=True, show_blocks=True, show_raw=True
    )

    args, kwargs = stubs[expected_fn][0]
    assert args == (path,)
    assert kwargs == {"show_flow": True, "show_blocks": True, "show_raw": True}
    assert result["called"] == expected_fn


def test_inspect_session_unknown_converter_raises():
    with pytest.raises(ValueError, match="Unknown converter"):
        Converter("bogus").inspect_session("x.jsonl")


# --- error message formatting shared across methods -----------------------


@pytest.mark.parametrize(
    "method,args",
    [
        ("load_session", ("p",)),
        ("records_to_messages", ([],)),
        ("session_to_messages", ("p",)),
        ("inspect_session", ("p",)),
    ],
)
def test_unknown_converter_message_includes_repr(method, args):
    c = Converter("weird")
    with pytest.raises(ValueError) as excinfo:
        getattr(c, method)(*args)
    assert "weird" in str(excinfo.value)
    assert "'weird'" in str(excinfo.value)
