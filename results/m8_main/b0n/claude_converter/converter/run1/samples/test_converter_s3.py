from __future__ import annotations

from pathlib import Path

import pytest

import claude_converter.converter as conv_mod
from claude_converter.converter import Converter


# ---------------------------------------------------------------------------
# Fixtures: patch the per-tool functions in the converter module namespace
# ---------------------------------------------------------------------------


@pytest.fixture
def calls(monkeypatch):
    """Replace every dispatched function with a recorder and return the log."""
    log: list[tuple] = []

    def make(name):
        def fake(*args, **kwargs):
            log.append((name, args, kwargs))
            return {"tool": name, "args": args, "kwargs": kwargs}
        return fake

    for tool in ("claude", "codex"):
        monkeypatch.setattr(conv_mod, f"load_session_{tool}", make(f"load_session_{tool}"))
        monkeypatch.setattr(conv_mod, f"records_to_messages_{tool}", make(f"records_to_messages_{tool}"))
        monkeypatch.setattr(conv_mod, f"session_to_messages_{tool}", make(f"session_to_messages_{tool}"))
        monkeypatch.setattr(conv_mod, f"inspect_session_{tool}", make(f"inspect_session_{tool}"))

    # pi functions have slightly different public names for load/records/session
    monkeypatch.setattr(conv_mod, "load_session_pi", make("load_session_pi"))
    monkeypatch.setattr(conv_mod, "records_to_messages_pi", make("records_to_messages_pi"))
    monkeypatch.setattr(conv_mod, "session_to_messages_pi", make("session_to_messages_pi"))
    monkeypatch.setattr(conv_mod, "inspect_session_pi", make("inspect_session_pi"))

    return log


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_default_converter_is_claude_code():
    assert Converter().converter == "claude-code"


@pytest.mark.parametrize("name", ["claude-code", "codex", "pi"])
def test_explicit_converter_name_stored(name):
    assert Converter(name).converter == name


def test_unknown_converter_name_stored_verbatim():
    # The constructor does not validate; it simply stores whatever is passed.
    c = Converter("bogus")  # type: ignore[arg-type]
    assert c.converter == "bogus"


# ---------------------------------------------------------------------------
# load_session dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_fn",
    [
        ("claude-code", "load_session_claude"),
        ("codex", "load_session_codex"),
        ("pi", "load_session_pi"),
    ],
)
def test_load_session_dispatch(calls, name, expected_fn):
    result = Converter(name).load_session("some/path.jsonl")
    assert calls == [(expected_fn, ("some/path.jsonl",), {})]
    assert result == {"tool": expected_fn, "args": ("some/path.jsonl",), "kwargs": {}}


def test_load_session_passes_path_object(calls):
    p = Path("/tmp/x/session.jsonl")
    Converter("codex").load_session(p)
    assert calls == [("load_session_codex", (p,), {})]


# ---------------------------------------------------------------------------
# records_to_messages dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_fn",
    [
        ("claude-code", "records_to_messages_claude"),
        ("codex", "records_to_messages_codex"),
        ("pi", "records_to_messages_pi"),
    ],
)
def test_records_to_messages_dispatch(calls, name, expected_fn):
    records = [{"type": "user"}, {"type": "assistant"}]
    result = Converter(name).records_to_messages(records)
    assert calls == [(expected_fn, (records,), {})]
    assert result["tool"] == expected_fn


def test_records_to_messages_passes_empty_list(calls):
    Converter("claude-code").records_to_messages([])
    assert calls == [("records_to_messages_claude", ([],), {})]


# ---------------------------------------------------------------------------
# session_to_messages dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_fn",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_dispatch_default_output(calls, name, expected_fn):
    result = Converter(name).session_to_messages("s.jsonl")
    assert calls == [(expected_fn, ("s.jsonl",), {"output": None})]
    assert result["tool"] == expected_fn


@pytest.mark.parametrize(
    "name, expected_fn",
    [
        ("claude-code", "session_to_messages_claude"),
        ("codex", "session_to_messages_codex"),
        ("pi", "session_to_messages_pi"),
    ],
)
def test_session_to_messages_dispatch_with_output(calls, name, expected_fn):
    out = Path("/tmp/out/messages.json")
    result = Converter(name).session_to_messages("s.jsonl", output=out)
    assert calls == [(expected_fn, ("s.jsonl",), {"output": out})]
    assert result["tool"] == expected_fn


def test_session_to_messages_output_positional_second_arg_is_not_output(calls):
    # The signature is (path, output=None); a second positional argument is output.
    out = "out.json"
    Converter("pi").session_to_messages("s.jsonl", out)
    assert calls == [("session_to_messages_pi", ("s.jsonl",), {"output": out})]


# ---------------------------------------------------------------------------
# inspect_session dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_fn",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_dispatch_defaults(calls, name, expected_fn):
    result = Converter(name).inspect_session("s.jsonl")
    assert calls == [
        (
            expected_fn,
            ("s.jsonl",),
            {"show_flow": False, "show_blocks": False, "show_raw": False},
        )
    ]
    assert result["tool"] == expected_fn


@pytest.mark.parametrize(
    "name, expected_fn",
    [
        ("claude-code", "inspect_session_claude"),
        ("codex", "inspect_session_codex"),
        ("pi", "inspect_session_pi"),
    ],
)
def test_inspect_session_dispatch_with_flags(calls, name, expected_fn):
    Converter(name).inspect_session(
        "s.jsonl", show_flow=True, show_blocks=True, show_raw=True
    )
    assert calls == [
        (
            expected_fn,
            ("s.jsonl",),
            {"show_flow": True, "show_blocks": True, "show_raw": True},
        )
    ]


def test_inspect_session_returns_underlying_return_value(calls):
    # The fake returns a dict; the dispatcher must pass it through unchanged.
    result = Converter("codex").inspect_session("s.jsonl")
    assert result == {
        "tool": "inspect_session_codex",
        "args": ("s.jsonl",),
        "kwargs": {"show_flow": False, "show_blocks": False, "show_raw": False},
    }


# ---------------------------------------------------------------------------
# Error handling for unknown converter names
# ---------------------------------------------------------------------------


def test_unknown_converter_load_session_raises(calls):
    c = Converter("bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as excinfo:
        c.load_session("s.jsonl")
    assert str(excinfo.value) == "Unknown converter: 'bogus'"
    assert calls == []


def test_unknown_converter_records_to_messages_raises(calls):
    c = Converter("bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as excinfo:
        c.records_to_messages([])
    assert str(excinfo.value) == "Unknown converter: 'bogus'"
    assert calls == []


def test_unknown_converter_session_to_messages_raises(calls):
    c = Converter("bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as excinfo:
        c.session_to_messages("s.jsonl")
    assert str(excinfo.value) == "Unknown converter: 'bogus'"
    assert calls == []


def test_unknown_converter_inspect_session_raises(calls):
    c = Converter("bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as excinfo:
        c.inspect_session("s.jsonl")
    assert str(excinfo.value) == "Unknown converter: 'bogus'"
    assert calls == []


def test_error_message_includes_repr_for_odd_names(calls):
    c = Converter("it's weird")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as excinfo:
        c.load_session("s.jsonl")
    assert str(excinfo.value) == "Unknown converter: \"it's weird\"%s" % ()


# ---------------------------------------------------------------------------
# Mutation of the converter name between calls is honored
# ---------------------------------------------------------------------------


def test_converter_name_can_be_changed_between_calls(calls):
    c = Converter("claude-code")
    c.load_session("a.jsonl")
    c.converter = "pi"
    c.load_session("b.jsonl")
    assert calls == [
        ("load_session_claude", ("a.jsonl",), {}),
        ("load_session_pi", ("b.jsonl",), {}),
    ]


# ---------------------------------------------------------------------------
# Integration-style check against the real implementation (no monkeypatch)
# ---------------------------------------------------------------------------


def test_real_claude_code_session_to_messages_on_tmp_file(tmp_path):
    import json

    session = tmp_path / "session.jsonl"
    records = [
        {
            "type": "user",
            "message": {"role": "user", "content": "hello"},
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "hi there"},
        },
    ]
    session.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    messages = Converter("claude-code").session_to_messages(session)
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_real_converter_load_session_reads_jsonl(tmp_path):
    import json

    session = tmp_path / "session.jsonl"
    records = [{"type": "user", "message": {"role": "user", "content": "x"}}]
    session.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

    loaded = Converter("claude-code").load_session(session)
    assert loaded == records


def test_real_converter_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        Converter("claude-code").load_session(missing)
