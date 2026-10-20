from __future__ import annotations

import pytest

import claude_converter.converter as converter_module
from claude_converter.converter import Converter


# --------------------------------------------------------------------------- #
# Fixtures: replace the tool-specific functions with recording spies so we can
# assert exactly which one is dispatched to (and with what arguments).
# --------------------------------------------------------------------------- #

CALLS: list[tuple] = []


@pytest.fixture(autouse=True)
def _record_calls(monkeypatch):
    CALLS.clear()

    def make_stub(name, return_value=None, record_return=True):
        def stub(*args, **kwargs):
            CALLS.append((name, args, kwargs))
            return return_value
        return stub

    replaced = {
        "load_session_claude": make_stub("load_session_claude", ["claude-loaded"]),
        "load_session_codex": make_stub("load_session_codex", ["codex-loaded"]),
        "load_session_pi": make_stub("load_session_pi", ["pi-loaded"]),
        "records_to_messages_claude": make_stub("records_to_messages_claude", [{"role": "user", "content": "c"}]),
        "records_to_messages_codex": make_stub("records_to_messages_codex", [{"role": "user", "content": "x"}]),
        "records_to_messages_pi": make_stub("records_to_messages_pi", [{"role": "user", "content": "p"}]),
        "session_to_messages_claude": make_stub("session_to_messages_claude", ["claude-messages"]),
        "session_to_messages_codex": make_stub("session_to_messages_codex", ["codex-messages"]),
        "session_to_messages_pi": make_stub("session_to_messages_pi", ["pi-messages"]),
        "inspect_session_claude": make_stub("inspect_session_claude", None),
        "inspect_session_codex": make_stub("inspect_session_codex", None),
        "inspect_session_pi": make_stub("inspect_session_pi", None),
    }

    for attr, stub in replaced.items():
        monkeypatch.setattr(converter_module, attr, stub)

    yield


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #

def test_default_converter_is_claude_code():
    assert Converter().converter == "claude-code"


@pytest.mark.parametrize("name", ["claude-code", "codex", "pi"])
def test_explicit_converter_name_is_stored(name):
    assert Converter(name).converter == name


def test_converter_accepts_positional_argument():
    c = Converter("codex")
    assert c.converter == "codex"


# --------------------------------------------------------------------------- #
# load_session
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name, expected",
    [
        ("claude-code", ["claude-loaded"]),
        ("codex", ["codex-loaded"]),
        ("pi", ["pi-loaded"]),
    ],
)
def test_load_session_dispatches(name, expected):
    assert Converter(name).load_session("some/path.jsonl") == expected


@pytest.mark.parametrize(
    "name, func_name",
    [
        ("claude-code", "load_session_claude"),
        ("codex", "load_session_codex"),
        ("pi", "load_session_pi"),
    ],
)
def test_load_session_calls_correct_function_with_path(name, func_name):
    Converter(name).load_session("my/path.jsonl")
    assert len(CALLS) == 1
    called_name, args, kwargs = CALLS[0]
    assert called_name == func_name
    assert args == ("my/path.jsonl",)
    assert kwargs == {}


def test_load_session_default_converter_routes_to_claude():
    assert Converter().load_session("p") == ["claude-loaded"]
    assert CALLS[0][0] == "load_session_claude"


def test_load_session_unknown_converter_raises():
    c = Converter("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as exc:
        c.load_session("x")
    assert "Unknown converter: 'nope'" in str(exc.value)
    assert CALLS == []


def test_load_session_unknown_converter_none_raises():
    c = Converter(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError) as exc:
        c.load_session("x")
    assert "Unknown converter: None" in str(exc.value)


# --------------------------------------------------------------------------- #
# records_to_messages
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name, expected",
    [
        ("claude-code", [{"role": "user", "content": "c"}]),
        ("codex", [{"role": "user", "content": "x"}]),
        ("pi", [{"role": "user", "content": "p"}]),
    ],
)
def test_records_to_messages_dispatches(name, expected):
    records = [{"type": "user"}]
    assert Converter(name).records_to_messages(records) == expected


@pytest.mark.parametrize(
    "name, func_name",
    [
        ("claude-code", "records_to_messages_claude"),
        ("codex", "records_to_messages_codex"),
        ("pi", "records_to_messages_pi"),
    ],
)
def test_records_to_messages_calls_correct_function(name, func_name):
    records = [{"a": 1}, {"b": 2}]
    Converter(name).records_to_messages(records)
    called_name, args, kwargs = CALLS[0]
    assert called_name == func_name
    assert args == (records,)
    assert kwargs == {}
    # ensure the very same list object is forwarded
    assert args[0] is records


def test_records_to_messages_unknown_converter_raises():
    c = Converter("bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as exc:
        c.records_to_messages([])
    assert "Unknown converter: 'bogus'" in str(exc.value)
    assert CALLS == []


# --------------------------------------------------------------------------- #
# session_to_messages
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name, expected",
    [
        ("claude-code", ["claude-messages"]),
        ("codex", ["codex-messages"]),
        ("pi", ["pi-messages"]),
    ],
)
def test_session_to_messages_dispatches(name, expected):
    assert Converter(name).session_to_messages("p.jsonl") == expected


@pytest.mark.parametrize(
    "name, func_name",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_default_output_none(name, func_name):
    Converter(name).session_to_messages("p.jsonl")
    called_name, args, kwargs = CALLS[0]
    assert called_name == func_name
    assert args == ("p.jsonl",)
    assert kwargs == {"output": None}


@pytest.mark.parametrize(
    "name, func_name",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_forwards_output(name, func_name):
    Converter(name).session_to_messages("in.jsonl", output="out.json")
    called_name, args, kwargs = CALLS[0]
    assert called_name == func_name
    assert args == ("in.jsonl",)
    assert kwargs == {"output": "out.json"}


def test_session_to_messages_output_positional():
    # output is passed through as a keyword to the underlying function
    Converter("codex").session_to_messages("a", "b")
    called_name, args, kwargs = CALLS[0]
    assert called_name == "session_to_messages_codex"
    assert args == ("a",)
    assert kwargs == {"output": "b"}


def test_session_to_messages_unknown_converter_raises():
    c = Converter("zzz")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as exc:
        c.session_to_messages("p")
    assert "Unknown converter: 'zzz'" in str(exc.value)
    assert CALLS == []


# --------------------------------------------------------------------------- #
# inspect_session
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name, func_name",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_default_flags(name, func_name):
    result = Converter(name).inspect_session("p.jsonl")
    assert result is None
    called_name, args, kwargs = CALLS[0]
    assert called_name == func_name
    assert args == ("p.jsonl",)
    assert kwargs == {"show_flow": False, "show_blocks": False, "show_raw": False}


@pytest.mark.parametrize(
    "name, func_name",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_forwards_flags(name, func_name):
    Converter(name).inspect_session(
        "p.jsonl", show_flow=True, show_blocks=True, show_raw=True
    )
    called_name, args, kwargs = CALLS[0]
    assert called_name == func_name
    assert args == ("p.jsonl",)
    assert kwargs == {"show_flow": True, "show_blocks": True, "show_raw": True}


def test_inspect_session_partial_flags():
    Converter("codex").inspect_session("p", show_raw=True)
    _, args, kwargs = CALLS[0]
    assert args == ("p",)
    assert kwargs == {"show_flow": False, "show_blocks": False, "show_raw": True}


def test_inspect_session_unknown_converter_raises():
    c = Converter("unknown-tool")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as exc:
        c.inspect_session("p")
    assert "Unknown converter: 'unknown-tool'" in str(exc.value)
    assert CALLS == []


# --------------------------------------------------------------------------- #
# Dispatcher state is read at call time
# --------------------------------------------------------------------------- #

def test_converter_attribute_used_at_call_time():
    c = Converter("claude-code")
    assert c.load_session("p") == ["claude-loaded"]
    # mutate the instance attribute; dispatch should follow it
    c.converter = "pi"  # type: ignore[assignment]
    assert c.load_session("p") == ["pi-loaded"]
    assert [call[0] for call in CALLS] == ["load_session_claude", "load_session_pi"]
