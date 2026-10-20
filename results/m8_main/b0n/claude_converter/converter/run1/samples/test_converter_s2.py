"""Unit tests for claude_converter.converter."""

from __future__ import annotations

import pytest

import claude_converter.converter as converter_module
from claude_converter.converter import Converter


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SENTINEL = object()


@pytest.fixture
def spy(monkeypatch):
    """Replace every routed-to function with a recorder and return the dict
    of calls keyed by function name."""
    calls: dict[str, list] = {}

    def make_fake(name):
        def fake(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return SENTINEL

        return fake

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
        monkeypatch.setattr(converter_module, name, make_fake(name))
    return calls


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_default_converter_is_claude_code():
    conv = Converter()
    assert conv.converter == "claude-code"


def test_explicit_converter_is_stored():
    assert Converter("codex").converter == "codex"
    assert Converter("pi").converter == "pi"
    assert Converter("claude-code").converter == "claude-code"


# ---------------------------------------------------------------------------
# load_session routing
# ---------------------------------------------------------------------------


def test_load_session_routes_claude(spy):
    conv = Converter("claude-code")
    result = conv.load_session("/tmp/session.jsonl")
    assert result is SENTINEL
    assert spy == {"load_session_claude": [(("/tmp/session.jsonl",), {})]}


def test_load_session_routes_codex(spy):
    conv = Converter("codex")
    result = conv.load_session("/tmp/codex.jsonl")
    assert result is SENTINEL
    assert spy == {"load_session_codex": [(("/tmp/codex.jsonl",), {})]}


def test_load_session_routes_pi(spy):
    conv = Converter("pi")
    result = conv.load_session("/tmp/pi.jsonl")
    assert result is SENTINEL
    assert spy == {"load_session_pi": [(("/tmp/pi.jsonl",), {})]}


def test_load_session_unknown_raises(spy):
    conv = Converter()
    conv.converter = "bogus"  # bypass type checker to hit the fallback branch
    with pytest.raises(ValueError, match="Unknown converter: 'bogus'"):
        conv.load_session("/tmp/x.jsonl")
    assert spy == {}


# ---------------------------------------------------------------------------
# records_to_messages routing
# ---------------------------------------------------------------------------


def test_records_to_messages_routes_claude(spy):
    conv = Converter("claude-code")
    records = [{"type": "user"}]
    result = conv.records_to_messages(records)
    assert result is SENTINEL
    assert spy == {"records_to_messages_claude": [((records,), {})]}


def test_records_to_messages_routes_codex(spy):
    conv = Converter("codex")
    records = [{"type": "response_item"}]
    result = conv.records_to_messages(records)
    assert result is SENTINEL
    assert spy == {"records_to_messages_codex": [((records,), {})]}


def test_records_to_messages_routes_pi(spy):
    conv = Converter("pi")
    records = [{"type": "message"}]
    result = conv.records_to_messages(records)
    assert result is SENTINEL
    assert spy == {"records_to_messages_pi": [((records,), {})]}


def test_records_to_messages_unknown_raises(spy):
    conv = Converter()
    conv.converter = "nope"
    with pytest.raises(ValueError, match="Unknown converter: 'nope'"):
        conv.records_to_messages([])
    assert spy == {}


# ---------------------------------------------------------------------------
# session_to_messages routing
# ---------------------------------------------------------------------------


def test_session_to_messages_routes_claude_default_output(spy):
    conv = Converter("claude-code")
    result = conv.session_to_messages("/tmp/a.jsonl")
    assert result is SENTINEL
    assert spy == {
        "session_to_messages_claude": [(("/tmp/a.jsonl",), {"output": None})]
    }


def test_session_to_messages_routes_codex_with_output(spy):
    conv = Converter("codex")
    result = conv.session_to_messages("/tmp/c.jsonl", output="/tmp/out.json")
    assert result is SENTINEL
    assert spy == {
        "session_to_messages_codex": [
            (("/tmp/c.jsonl",), {"output": "/tmp/out.json"})
        ]
    }


def test_session_to_messages_routes_pi_with_output(spy):
    conv = Converter("pi")
    result = conv.session_to_messages("/tmp/p.jsonl", output="/tmp/p_out.json")
    assert result is SENTINEL
    assert spy == {
        "session_to_messages_pi": [(("/tmp/p.jsonl",), {"output": "/tmp/p_out.json"})]
    }


def test_session_to_messages_unknown_raises(spy):
    conv = Converter()
    conv.converter = "invalid"
    with pytest.raises(ValueError, match="Unknown converter: 'invalid'"):
        conv.session_to_messages("/tmp/x.jsonl")
    assert spy == {}


# ---------------------------------------------------------------------------
# inspect_session routing
# ---------------------------------------------------------------------------


def test_inspect_session_routes_claude_defaults(spy):
    conv = Converter("claude-code")
    result = conv.inspect_session("/tmp/a.jsonl")
    assert result is SENTINEL
    assert spy == {
        "inspect_session_claude": [
            (("/tmp/a.jsonl",), {"show_flow": False, "show_blocks": False, "show_raw": False})
        ]
    }


def test_inspect_session_routes_codex_flags(spy):
    conv = Converter("codex")
    result = conv.inspect_session(
        "/tmp/c.jsonl", show_flow=True, show_blocks=True, show_raw=True
    )
    assert result is SENTINEL
    assert spy == {
        "inspect_session_codex": [
            (("/tmp/c.jsonl",), {"show_flow": True, "show_blocks": True, "show_raw": True})
        ]
    }


def test_inspect_session_routes_pi_partial_flags(spy):
    conv = Converter("pi")
    result = conv.inspect_session("/tmp/p.jsonl", show_flow=True)
    assert result is SENTINEL
    assert spy == {
        "inspect_session_pi": [
            (("/tmp/p.jsonl",), {"show_flow": True, "show_blocks": False, "show_raw": False})
        ]
    }


def test_inspect_session_unknown_raises(spy):
    conv = Converter()
    conv.converter = "xx"
    with pytest.raises(ValueError, match="Unknown converter: 'xx'"):
        conv.inspect_session("/tmp/x.jsonl")
    assert spy == {}
