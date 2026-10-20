from __future__ import annotations

from pathlib import Path

import pytest

from claude_converter.converter import Converter


# ---------------------------------------------------------------------------
# Helpers to build a "fake" implementation set and inject it via monkeypatch
# ---------------------------------------------------------------------------

CONVERTERS = ("claude-code", "codex", "pi")

# Mapping from converter name -> (module, prefix) used by converter.py
# claude-code uses the claude module aliases, codex/pi use suffixed names.
_EXPECTED = {
    "claude-code": {
        "load_session": "load_session_claude",
        "records_to_messages": "records_to_messages_claude",
        "session_to_messages": "session_to_messages_claude",
        "inspect_session": "inspect_session_claude",
    },
    "codex": {
        "load_session": "load_session_codex",
        "records_to_messages": "records_to_messages_codex",
        "session_to_messages": "session_to_messages_codex",
        "inspect_session": "inspect_session_codex",
    },
    "pi": {
        "load_session": "load_session_pi",
        "records_to_messages": "records_to_messages_pi",
        "session_to_messages": "session_to_messages_pi",
        "inspect_session": "inspect_session_pi",
    },
}


class _CallRecorder:
    def __init__(self, name: str, return_value=None):
        self.name = name
        self.return_value = return_value
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value


@pytest.fixture
def patched(monkeypatch):
    """Patch every dispatch target with a recording callable.

    Returns a dict: {converter_name: {method_alias: recorder}}
    """
    import claude_converter.converter as converter_mod

    recorders: dict[str, dict[str, _CallRecorder]] = {}
    for name in CONVERTERS:
        aliases = _EXPECTED[name]
        per_name: dict[str, _CallRecorder] = {}
        for method, alias in aliases.items():
            recorder = _CallRecorder(alias)
            monkeypatch.setattr(converter_mod, alias, recorder)
            per_name[method] = recorder
        recorders[name] = per_name
    return recorders


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

def test_default_converter_is_claude_code():
    c = Converter()
    assert c.converter == "claude-code"


@pytest.mark.parametrize("name", CONVERTERS)
def test_explicit_converter_stored(name):
    c = Converter(name)
    assert c.converter == name


def test_converter_stores_arbitrary_value():
    # No validation happens at construction time.
    c = Converter("bogus")  # type: ignore[arg-type]
    assert c.converter == "bogus"


# ---------------------------------------------------------------------------
# load_session dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CONVERTERS)
def test_load_session_dispatches(patched, name):
    c = Converter(name)
    recorder = patched[name]["load_session"]
    sentinel = [{"type": "x"}]
    recorder.return_value = sentinel

    result = c.load_session("some/path.jsonl")

    assert result is sentinel
    assert len(recorder.calls) == 1
    assert recorder.calls[0] == (("some/path.jsonl",), {})


def test_load_session_accepts_path_object(patched):
    c = Converter("codex")
    recorder = patched["codex"]["load_session"]
    recorder.return_value = []

    p = Path("/tmp/whatever.jsonl")
    assert c.load_session(p) == []
    assert recorder.calls == [((p,), {})]


@pytest.mark.parametrize("name", CONVERTERS)
def test_load_session_only_calls_own_target(patched, name):
    Converter(name).load_session("p")
    for other in CONVERTERS:
        for method, recorder in patched[other].items():
            if other == name and method == "load_session":
                assert len(recorder.calls) == 1
            else:
                assert recorder.calls == [], (other, method)


# ---------------------------------------------------------------------------
# records_to_messages dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CONVERTERS)
def test_records_to_messages_dispatches(patched, name):
    c = Converter(name)
    recorder = patched[name]["records_to_messages"]
    messages = [{"role": "user", "content": "hi"}]
    recorder.return_value = messages
    records = [{"type": "user"}]

    result = c.records_to_messages(records)

    assert result is messages
    assert recorder.calls == [((records,), {})]


def test_records_to_messages_empty_list(patched):
    c = Converter("pi")
    recorder = patched["pi"]["records_to_messages"]
    recorder.return_value = []

    assert c.records_to_messages([]) == []
    assert recorder.calls == [(([],), {})]


# ---------------------------------------------------------------------------
# session_to_messages dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CONVERTERS)
def test_session_to_messages_with_output(patched, name):
    c = Converter(name)
    recorder = patched[name]["session_to_messages"]
    messages = [{"role": "assistant", "content": "ok"}]
    recorder.return_value = messages

    result = c.session_to_messages("in.jsonl", output="out.json")

    assert result is messages
    assert recorder.calls == [
        (("in.jsonl",), {"output": "out.json"})
    ]


@pytest.mark.parametrize("name", CONVERTERS)
def test_session_to_messages_default_output_none(patched, name):
    c = Converter(name)
    recorder = patched[name]["session_to_messages"]
    recorder.return_value = []

    c.session_to_messages("in.jsonl")

    assert recorder.calls == [(("in.jsonl",), {"output": None})]


def test_session_to_messages_explicit_none_output(patched):
    c = Converter("codex")
    recorder = patched["codex"]["session_to_messages"]
    recorder.return_value = [{"role": "user", "content": "x"}]

    result = c.session_to_messages(Path("in.jsonl"), output=None)

    assert result == [{"role": "user", "content": "x"}]
    assert recorder.calls == [((Path("in.jsonl"),), {"output": None})]


def test_session_to_messages_output_as_path(patched):
    c = Converter("pi")
    recorder = patched["pi"]["session_to_messages"]
    recorder.return_value = []

    out = Path("some/dir/out.json")
    c.session_to_messages("in.jsonl", output=out)

    assert recorder.calls == [(("in.jsonl",), {"output": out})]


# ---------------------------------------------------------------------------
# inspect_session dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CONVERTERS)
def test_inspect_session_defaults(patched, name):
    c = Converter(name)
    recorder = patched[name]["inspect_session"]

    result = c.inspect_session("in.jsonl")

    assert result is None  # inspect_session returns whatever target returns
    assert recorder.calls == [
        (
            ("in.jsonl",),
            {"show_flow": False, "show_blocks": False, "show_raw": False},
        )
    ]


@pytest.mark.parametrize("name", CONVERTERS)
def test_inspect_session_forwards_flags(patched, name):
    c = Converter(name)
    recorder = patched[name]["inspect_session"]

    c.inspect_session("in.jsonl", show_flow=True, show_blocks=True, show_raw=True)

    assert recorder.calls == [
        (
            ("in.jsonl",),
            {"show_flow": True, "show_blocks": True, "show_raw": True},
        )
    ]


@pytest.mark.parametrize("name", CONVERTERS)
def test_inspect_session_mixed_flags(patched, name):
    c = Converter(name)
    recorder = patched[name]["inspect_session"]

    c.inspect_session(Path("p.jsonl"), show_flow=True)

    assert recorder.calls == [
        (
            (Path("p.jsonl"),),
            {"show_flow": True, "show_blocks": False, "show_raw": False},
        )
    ]


def test_inspect_session_returns_target_return_value(patched):
    c = Converter("codex")
    recorder = patched["codex"]["inspect_session"]
    recorder.return_value = "sentinel"

    assert c.inspect_session("p") == "sentinel"


# ---------------------------------------------------------------------------
# Unknown converter error handling for each method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, call",
    [
        ("load_session", lambda c: c.load_session("p")),
        ("records_to_messages", lambda c: c.records_to_messages([])),
        ("session_to_messages", lambda c: c.session_to_messages("p")),
        ("inspect_session", lambda c: c.inspect_session("p")),
    ],
)
def test_unknown_converter_raises_value_error(patched, method, call):
    c = Converter("unknown-tool")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as excinfo:
        call(c)
    assert "Unknown converter" in str(excinfo.value)
    assert "'unknown-tool'" in str(excinfo.value)

    # Nothing should have been dispatched.
    for name in CONVERTERS:
        for recorder in patched[name].values():
            assert recorder.calls == []


@pytest.mark.parametrize("method", ["load_session", "records_to_messages",
                                    "session_to_messages", "inspect_session"])
def test_unknown_converter_empty_string(patched, method):
    c = Converter("")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as excinfo:
        if method == "load_session":
            c.load_session("p")
        elif method == "records_to_messages":
            c.records_to_messages([])
        elif method == "session_to_messages":
            c.session_to_messages("p")
        else:
            c.inspect_session("p")
    assert "Unknown converter" in str(excinfo.value)


# ---------------------------------------------------------------------------
# End-to-end-ish dispatch test without patching the underlying functions:
# with an unknown converter no import-level target is called, so these
# simply confirm the ValueError path again for every entry point, including
# with keyword arguments supplied.
# ---------------------------------------------------------------------------

def test_unknown_converter_with_kwargs_raises():
    c = Converter("not-a-real-tool")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        c.session_to_messages("p.jsonl", output="out.json")
    with pytest.raises(ValueError):
        c.inspect_session("p.jsonl", show_flow=True, show_blocks=True, show_raw=True)
