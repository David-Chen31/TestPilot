"""Tests for claude_converter.codex."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest

from claude_converter.codex import (
    CODEX_SCHEMA,
    _payload_parts,
    _payload_text,
    _record_role,
    _record_type_of,
    _timestamp_of,
    _total_tokens_of,
    inspect_session_codex,
    load_session_codex,
    records_to_messages_codex,
    session_to_messages_codex,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _msg(role: str, blocks: list[dict]) -> dict:
    return {
        "type": "response_item",
        "payload": {"type": "message", "role": role, "content": blocks},
    }


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _input_text(text: str) -> dict:
    return {"type": "input_text", "text": text}


def _output_text(text: str) -> dict:
    return {"type": "output_text", "text": text}


def _record(payload: dict, rtype: str = "response_item", **extra) -> dict:
    rec = {"type": rtype, "payload": payload}
    rec.update(extra)
    return rec


def _tiny_png_b64() -> str:
    """Return a base64-encoded 1x1 PNG (no header)."""
    # 1x1 transparent PNG
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    return base64.b64encode(png_bytes).decode("ascii")


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------

def test_load_session_codex_reads_jsonl(tmp_path: Path):
    p = _write_jsonl(tmp_path / "session.jsonl", [{"a": 1}, {"b": 2}])
    assert load_session_codex(p) == [{"a": 1}, {"b": 2}]


def test_load_session_codex_accepts_str_path(tmp_path: Path):
    p = _write_jsonl(tmp_path / "session.jsonl", [{"a": 1}])
    assert load_session_codex(str(p)) == [{"a": 1}]


def test_load_session_codex_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_session_codex(tmp_path / "nope.jsonl")


def test_load_session_codex_bad_extension(tmp_path: Path):
    p = tmp_path / "session.txt"
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_codex(p)


def test_load_session_codex_empty_file(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_codex(p)


def test_load_session_codex_skips_invalid_json_lines(tmp_path: Path, capsys):
    p = tmp_path / "session.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
    assert load_session_codex(p) == [{"a": 1}, {"b": 2}]
    err = capsys.readouterr().err
    assert "invalid JSON" in err


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------

def test_record_role_message_user():
    rec = _msg("user", [_input_text("hi")])
    assert _record_role(rec) == "user"


def test_record_role_message_assistant():
    rec = _msg("assistant", [_output_text("hello")])
    assert _record_role(rec) == "assistant"


def test_record_role_developer_skipped():
    rec = _msg("developer", [_input_text("sys")])
    assert _record_role(rec) is None


def test_record_role_unknown_message_role():
    rec = _msg("system", [_input_text("sys")])
    assert _record_role(rec) is None


def test_record_role_non_response_item():
    rec = _record({"type": "message", "role": "user"}, rtype="event_msg")
    assert _record_role(rec) is None


@pytest.mark.parametrize(
    "ptype,expected",
    [
        ("reasoning", "assistant"),
        ("function_call", "assistant"),
        ("custom_tool_call", "assistant"),
        ("function_call_output", "user"),
        ("custom_tool_call_output", "user"),
        ("unknown_type", None),
    ],
)
def test_record_role_tool_types(ptype, expected):
    rec = _record({"type": ptype})
    assert _record_role(rec) == expected


def test_record_role_payload_null():
    rec = {"type": "response_item", "payload": None}
    assert _record_role(rec) is None


# ---------------------------------------------------------------------------
# _payload_text
# ---------------------------------------------------------------------------

def test_payload_text_message():
    payload = {
        "type": "message",
        "content": [
            _input_text("hello"),
            _input_text("world"),
        ],
    }
    assert _payload_text(payload) == "hello\nworld"


def test_payload_text_message_skips_blank():
    payload = {
        "type": "message",
        "content": [_input_text("a"), _input_text("   "), _input_text("b")],
    }
    assert _payload_text(payload) == "a\nb"


def test_payload_text_message_image_block():
    payload = {
        "type": "message",
        "content": [_input_text("look"), {"type": "input_image", "image_url": "x"}],
    }
    assert _payload_text(payload) == "look\n<image>"


def test_payload_text_message_ignores_non_dict_blocks():
    payload = {
        "type": "message",
        "content": [_input_text("a"), "garbage", 42],
    }
    assert _payload_text(payload) == "a"


def test_payload_text_message_empty_content():
    assert _payload_text({"type": "message", "content": []}) == ""


def test_payload_text_message_missing_content():
    assert _payload_text({"type": "message"}) == ""


def test_payload_text_reasoning():
    payload = {
        "type": "reasoning",
        "summary": [{"text": "think"}, {"text": "more"}],
    }
    assert _payload_text(payload) == "<thinking>think\nmore</thinking>"


def test_payload_text_reasoning_empty():
    assert _payload_text({"type": "reasoning", "summary": []}) == ""


def test_payload_text_function_call():
    payload = {"type": "function_call", "name": "fn", "arguments": "arg"}
    assert _payload_text(payload) == "<tool_use name='fn'>arg</tool_use>"


def test_payload_text_function_call_output():
    payload = {"type": "function_call_output", "output": "result"}
    assert _payload_text(payload) == "<tool_result>result</tool_result>"


def test_payload_text_custom_tool_call():
    payload = {"type": "custom_tool_call", "name": "ct", "input": "inp"}
    assert _payload_text(payload) == "<tool_use name='ct'>inp</tool_use>"


def test_payload_text_custom_tool_call_output():
    payload = {"type": "custom_tool_call_output", "output": "out"}
    assert _payload_text(payload) == "<tool_result>out</tool_result>"


def test_payload_text_unknown_type():
    assert _payload_text({"type": "whatever"}) == ""


def test_payload_text_missing_type():
    assert _payload_text({}) == ""


# ---------------------------------------------------------------------------
# _payload_parts
# ---------------------------------------------------------------------------

def test_payload_parts_text_message():
    payload = {"type": "message", "content": [_input_text("a"), _output_text("b")]}
    assert _payload_parts(payload) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_payload_parts_skips_blank_text():
    payload = {"type": "message", "content": [_input_text("a"), _input_text("  ")]}
    assert _payload_parts(payload) == [{"type": "text", "text": "a"}]


def test_payload_parts_image(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "claude_converter.codex.convert_base64_to_pil_image",
        lambda uri: sentinel,
    )
    payload = {
        "type": "message",
        "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAA"}],
    }
    assert _payload_parts(payload) == [{"type": "image", "image": sentinel}]


def test_payload_parts_image_missing_url(monkeypatch):
    def _fail(_):
        raise AssertionError("should not be called")

    monkeypatch.setattr("claude_converter.codex.convert_base64_to_pil_image", _fail)
    payload = {"type": "message", "content": [{"type": "input_image", "image_url": ""}]}
    assert _payload_parts(payload) == []


def test_payload_parts_image_import_error(monkeypatch, capsys):
    def _fail(_):
        raise ImportError("no pillow")

    monkeypatch.setattr("claude_converter.codex.convert_base64_to_pil_image", _fail)
    payload = {"type": "message", "content": [{"type": "input_image", "image_url": "x"}]}
    assert _payload_parts(payload) == []
    assert "Skipping unreadable input_image" in capsys.readouterr().err


def test_payload_parts_image_value_error(monkeypatch, capsys):
    def _fail(_):
        raise ValueError("bad image")

    monkeypatch.setattr("claude_converter.codex.convert_base64_to_pil_image", _fail)
    payload = {"type": "message", "content": [{"type": "input_image", "image_url": "x"}]}
    assert _payload_parts(payload) == []
    assert "Skipping unreadable input_image" in capsys.readouterr().err


def test_payload_parts_non_message_collapses_to_text():
    payload = {"type": "function_call", "name": "fn", "arguments": "arg"}
    assert _payload_parts(payload) == [
        {"type": "text", "text": "<tool_use name='fn'>arg</tool_use>"}
    ]


def test_payload_parts_non_message_empty_text():
    assert _payload_parts({"type": "whatever"}) == []


def test_payload_parts_message_ignores_non_dict_blocks():
    payload = {"type": "message", "content": ["junk", _input_text("a")]}
    assert _payload_parts(payload) == [{"type": "text", "text": "a"}]


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------

def test_records_to_messages_simple():
    records = [
        _msg("user", [_input_text("hi")]),
        _msg("assistant", [_output_text("hello")]),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_records_to_messages_merges_same_role():
    records = [
        _msg("user", [_input_text("a")]),
        _msg("user", [_input_text("b")]),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "a\nb"},
    ]


def test_records_to_messages_skips_non_conversational():
    records = [
        _record({"type": "token_count"}),
        _msg("user", [_input_text("hi")]),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "hi"},
    ]


def test_records_to_messages_skips_empty_parts():
    records = [
        _msg("user", []),
        _msg("assistant", [_output_text("a")]),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "assistant", "content": "a"},
    ]


def test_records_to_messages_tool_roles():
    records = [
        _msg("user", [_input_text("hi")]),
        _record({"type": "reasoning", "summary": [{"text": "think"}]}),
        _record({"type": "function_call", "name": "fn", "arguments": "a"}),
        _record({"type": "function_call_output", "output": "r"}),
    ]
    result = records_to_messages_codex(records)
    assert result == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": (
                "<thinking>think</thinking>\n"
                "<tool_use name='fn'>a</tool_use>"
            ),
        },
        {"role": "user", "content": "<tool_result>r</tool_result>"},
    ]


def test_records_to_messages_with_image(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "claude_converter.codex.convert_base64_to_pil_image",
        lambda uri: sentinel,
    )
    records = [
        _msg(
            "user",
            [_input_text("look"), {"type": "input_image", "image_url": "data:x,AAA"}],
        ),
    ]
    result = records_to_messages_codex(records)
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "image": sentinel},
            ],
        }
    ]


def test_records_to_messages_empty():
    assert records_to_messages_codex([]) == []


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------

def test_session_to_messages_codex(tmp_path: Path):
    records = [
        _msg("user", [_input_text("hi")]),
        _msg("assistant", [_output_text("hello")]),
    ]
    p = _write_jsonl(tmp_path / "s.jsonl", records)
    result = session_to_messages_codex(p)
    assert result == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_session_to_messages_codex_output(tmp_path: Path):
    records = [_msg("user", [_input_text("hi")])]
    p = _write_jsonl(tmp_path / "s.jsonl", records)
    out = tmp_path / "sub" / "out.json"
    result = session_to_messages_codex(p, output=out)

    assert result == [{"role": "user", "content": "hi"}]
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == [{"role": "user", "content": "hi"}]


def test_session_to_messages_codex_output_encodes_image(tmp_path: Path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    b64 = _tiny_png_b64()
    data_uri = "data:image/png;base64," + b64
    records = [
        _msg(
            "user",
            [_input_text("look"), {"type": "input_image", "image_url": data_uri}],
        )
    ]
    p = _write_jsonl(tmp_path / "s.jsonl", records)
    out = tmp_path / "out.json"
    result = session_to_messages_codex(p, output=out)

    # in-memory keeps a real PIL image
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look"}
    assert isinstance(content[1]["image"], Image.Image)

    # saved json has data URI
    saved = json.loads(out.read_text(encoding="utf-8"))
    saved_content = saved[0]["content"]
    assert saved_content[0] == {"type": "text", "text": "look"}
    assert saved_content[1]["type"] == "image"
    assert saved_content[1]["image"].startswith("data:image/")


# ---------------------------------------------------------------------------
# inspection helpers
# ---------------------------------------------------------------------------

def test_record_type_of():
    assert _record_type_of({"type": "response_item", "payload": {"type": "message"}}) == (
        "response_item:message"
    )


def test_record_type_of_no_sub():
    assert _record_type_of({"type": "event_msg"}) == "event_msg"


def test_record_type_of_unknown():
    assert _record_type_of({}) == "unknown"


def test_timestamp_of():
    rec = {"timestamp": "2024-01-02T03:04:05.123456789Z"}
    assert _timestamp_of(rec) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_total_tokens_of_keeps_latest():
    records = [
        _record(
            {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 1,
                        "output_tokens": 2,
                        "cached_input_tokens": 3,
                        "total_tokens": 6,
                    }
                },
            }
        ),
        _record(
            {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "cached_input_tokens": 30,
                        "total_tokens": 60,
                    }
                },
            }
        ),
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cached_input_tokens": 30,
        "total_tokens": 60,
    }


def test_total_tokens_of_none_info():
    records = [
        _record({"type": "token_count", "info": None}),
    ]
    assert _total_tokens_of(records) == {}


def test_total_tokens_of_none_usage():
    records = [
        _record({"type": "token_count", "info": {"total_token_usage": None}}),
    ]
    assert _total_tokens_of(records) == {}


def test_total_tokens_of_empty():
    assert _total_tokens_of([]) == {}


def test_total_tokens_of_defaults():
    records = [
        _record({"type": "token_count", "info": {"total_token_usage": {"input_tokens": 5}}}),
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 5,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "total_tokens": 0,
    }


# ---------------------------------------------------------------------------
# CODEX_SCHEMA
# ---------------------------------------------------------------------------

def test_codex_schema_label():
    assert CODEX_SCHEMA.label == "CODEX"


def test_codex_schema_callables():
    rec = _msg("user", [_input_text("hi")])
    assert CODEX_SCHEMA.is_message(rec) is True
    assert CODEX_SCHEMA.role_of(rec) == "user"
    assert CODEX_SCHEMA.text_of(rec) == "hi"

    other = _record({"type": "token_count"})
    assert CODEX_SCHEMA.is_message(other) is False
    assert CODEX_SCHEMA.role_of(other) == "other"


def test_codex_schema_total_tokens():
    records = [
        _record({"type": "token_count", "info": {"total_token_usage": {"input_tokens": 7}}}),
    ]
    assert CODEX_SCHEMA.total_tokens_of(records)["input_tokens"] == 7


# ---------------------------------------------------------------------------
# inspect_session_codex
# ---------------------------------------------------------------------------

def test_inspect_session_codex_runs(tmp_path: Path, capsys):
    records = [
        _msg("user", [_input_text("hi")]),
        _msg("assistant", [_output_text("hello")]),
    ]
    p = _write_jsonl(tmp_path / "s.jsonl", records)
    inspect_session_codex(p)
    out = capsys.readouterr().out
    assert "CODEX" in out


def test_inspect_session_codex_show_flags(tmp_path: Path, capsys):
    records = [
        _msg("user", [_input_text("hi")]),
        _msg("assistant", [_output_text("hello")]),
    ]
    p = _write_jsonl(tmp_path / "s.jsonl", records)
    inspect_session_codex(p, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "CODEX" in out


def test_inspect_session_codex_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        inspect_session_codex(tmp_path / "nope.jsonl")
