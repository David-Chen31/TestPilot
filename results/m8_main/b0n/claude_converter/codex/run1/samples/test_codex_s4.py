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
# Helpers
# ---------------------------------------------------------------------------

def _response_item(payload: dict, timestamp: str = "2024-01-01T00:00:00.000Z") -> dict:
    return {"timestamp": timestamp, "type": "response_item", "payload": payload}


def _msg(role: str, *blocks) -> dict:
    return _response_item({"type": "message", "role": role, "content": list(blocks)})


def _text_block(text: str) -> dict:
    return {"type": "input_text", "text": text}


def _write_jsonl(tmp_path: Path, records: list[dict], name: str = "session.jsonl") -> Path:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _make_png_data_uri() -> str:
    """Build a tiny valid PNG data URI. Requires Pillow (skipped otherwise)."""
    pytest.importorskip("PIL")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------

def test_load_session_codex_reads_records(tmp_path):
    records = [{"type": "session_meta"}, {"type": "response_item", "payload": {"type": "message"}}]
    p = _write_jsonl(tmp_path, records)
    assert load_session_codex(p) == records


def test_load_session_codex_accepts_str_path(tmp_path):
    records = [{"type": "x"}]
    p = _write_jsonl(tmp_path, records)
    assert load_session_codex(str(p)) == records


def test_load_session_codex_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_codex(tmp_path / "nope.jsonl")


def test_load_session_codex_bad_extension(tmp_path):
    p = tmp_path / "session.txt"
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_session_codex(p)


def test_load_session_codex_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or contains no valid records"):
        load_session_codex(p)


# ---------------------------------------------------------------------------
# _payload_text
# ---------------------------------------------------------------------------

def test_payload_text_message_concatenates_text_blocks():
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "hello"},
            {"type": "output_text", "text": "world"},
            {"type": "text", "text": "third"},
        ],
    }
    assert _payload_text(payload) == "hello\nworld\nthird"


def test_payload_text_message_skips_blank_and_non_dict():
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "keep"},
            {"type": "input_text", "text": "   "},
            "not a dict",
            {"type": "input_text", "text": "last"},
        ],
    }
    assert _payload_text(payload) == "keep\nlast"


def test_payload_text_message_image_block():
    payload = {
        "type": "message",
        "content": [{"type": "input_text", "text": "look"}, {"type": "input_image"}],
    }
    assert _payload_text(payload) == "look\n<image>"


def test_payload_text_reasoning():
    payload = {"type": "reasoning", "summary": [{"text": "thinking..."}, {"not text": 1}]}
    assert _payload_text(payload) == "<thinking>thinking...</thinking>"


def test_payload_text_reasoning_empty():
    assert _payload_text({"type": "reasoning", "summary": []}) == ""
    assert _payload_text({"type": "reasoning", "summary": [{"text": "  "}]}) == ""


def test_payload_text_function_call():
    payload = {"type": "function_call", "name": "shell", "arguments": '{"a": 1}'}
    assert _payload_text(payload) == "<tool_use name='shell'>{\"a\": 1}</tool_use>"


def test_payload_text_function_call_output():
    payload = {"type": "function_call_output", "output": "result"}
    assert _payload_text(payload) == "<tool_result>result</tool_result>"


def test_payload_text_custom_tool_call():
    payload = {"type": "custom_tool_call", "name": "my_tool", "input": "raw input"}
    assert _payload_text(payload) == "<tool_use name='my_tool'>raw input</tool_use>"


def test_payload_text_custom_tool_call_output():
    payload = {"type": "custom_tool_call_output", "output": "custom result"}
    assert _payload_text(payload) == "<tool_result>custom result</tool_result>"


def test_payload_text_unknown_type():
    assert _payload_text({"type": "something_else"}) == ""
    assert _payload_text({}) == ""


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------

def test_record_role_non_response_item():
    assert _record_role({"type": "event_msg", "payload": {}}) is None


def test_record_role_message_user():
    assert _record_role(_msg("user", _text_block("hi"))) == "user"


def test_record_role_message_assistant():
    assert _record_role(_msg("assistant", _text_block("hi"))) == "assistant"


def test_record_role_message_developer_skipped():
    assert _record_role(_msg("developer", _text_block("sys"))) is None


def test_record_role_message_unknown_role():
    assert _record_role(_msg("system", _text_block("sys"))) is None


def test_record_role_tool_types():
    assert _record_role(_response_item({"type": "reasoning"})) == "assistant"
    assert _record_role(_response_item({"type": "function_call"})) == "assistant"
    assert _record_role(_response_item({"type": "custom_tool_call"})) == "assistant"
    assert _record_role(_response_item({"type": "function_call_output"})) == "user"
    assert _record_role(_response_item({"type": "custom_tool_call_output"})) == "user"


def test_record_role_unknown_payload_type():
    assert _record_role(_response_item({"type": "weird"})) is None


def test_record_role_null_payload():
    assert _record_role({"type": "response_item", "payload": None}) is None


# ---------------------------------------------------------------------------
# _payload_parts
# ---------------------------------------------------------------------------

def test_payload_parts_message_text_only():
    payload = {"type": "message", "content": [{"type": "input_text", "text": "hi"}]}
    assert _payload_parts(payload) == [{"type": "text", "text": "hi"}]


def test_payload_parts_message_skips_blank_text():
    payload = {
        "type": "message",
        "content": [{"type": "input_text", "text": "  "}, {"type": "input_text", "text": "ok"}],
    }
    assert _payload_parts(payload) == [{"type": "text", "text": "ok"}]


def test_payload_parts_message_skips_missing_image_url():
    payload = {"type": "message", "content": [{"type": "input_image", "image_url": ""}]}
    assert _payload_parts(payload) == []


def test_payload_parts_message_with_image():
    uri = _make_png_data_uri()
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "pic"},
            {"type": "input_image", "image_url": uri},
        ],
    }
    parts = _payload_parts(payload)
    assert parts[0] == {"type": "text", "text": "pic"}
    assert parts[1]["type"] == "image"
    from PIL import Image as PILImage
    assert isinstance(parts[1]["image"], PILImage.Image)


def test_payload_parts_unreadable_image_logged_and_skipped(capsys):
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "keep"},
            {"type": "input_image", "image_url": "data:image/png;base64,!!!notbase64!!!"},
        ],
    }
    parts = _payload_parts(payload)
    assert parts == [{"type": "text", "text": "keep"}]
    assert "Skipping unreadable input_image" in capsys.readouterr().err


def test_payload_parts_function_call():
    payload = {"type": "function_call", "name": "f", "arguments": "{}"}
    assert _payload_parts(payload) == [{"type": "text", "text": "<tool_use name='f'>{}</tool_use>"}]


def test_payload_parts_empty_flattened_text():
    assert _payload_parts({"type": "reasoning", "summary": []}) == []


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------

def test_records_to_messages_basic():
    records = [
        _msg("user", _text_block("hello")),
        _msg("assistant", {"type": "output_text", "text": "hi there"}),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_records_to_messages_merges_consecutive_same_role():
    records = [
        _msg("user", _text_block("a")),
        _msg("user", _text_block("b")),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "a\nb"},
    ]


def test_records_to_messages_skips_non_messages():
    records = [
        {"type": "session_meta", "payload": {"type": "message", "role": "user"}},
        _msg("developer", _text_block("sys")),
        _msg("user", _text_block("real")),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "real"}]


def test_records_to_messages_skips_empty_parts():
    records = [
        _msg("user", _text_block("   ")),
        _msg("user", _text_block("real")),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "real"}]


def test_records_to_messages_tool_call_flattened():
    records = [
        _msg("user", _text_block("run it")),
        _response_item({"type": "function_call", "name": "shell", "arguments": "ls"}),
        _response_item({"type": "function_call_output", "output": "file.txt"}),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "<tool_use name='shell'>ls</tool_use>"},
        {"role": "user", "content": "<tool_result>file.txt</tool_result>"},
    ]


def test_records_to_messages_multimodal_content_is_list():
    uri = _make_png_data_uri()
    records = [
        _msg(
            "user",
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": uri},
        ),
    ]
    messages = records_to_messages_codex(records)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"


def test_records_to_messages_empty_records():
    assert records_to_messages_codex([]) == []


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------

def test_session_to_messages_codex_no_output(tmp_path):
    records = [_msg("user", _text_block("hi"))]
    p = _write_jsonl(tmp_path, records)
    assert session_to_messages_codex(p) == [{"role": "user", "content": "hi"}]


def test_session_to_messages_codex_writes_output(tmp_path):
    records = [_msg("user", _text_block("hi"))]
    p = _write_jsonl(tmp_path, records)
    out = tmp_path / "sub" / "out.json"
    result = session_to_messages_codex(p, output=out)

    assert result == [{"role": "user", "content": "hi"}]
    assert out.exists()
    with open(out, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == [{"role": "user", "content": "hi"}]


def test_session_to_messages_codex_output_encodes_image(tmp_path):
    uri = _make_png_data_uri()
    records = [
        _msg("user", {"type": "input_text", "text": "pic"}, {"type": "input_image", "image_url": uri}),
    ]
    p = _write_jsonl(tmp_path, records)
    out = tmp_path / "out.json"
    result = session_to_messages_codex(p, output=out)

    # In-memory keeps PIL image
    from PIL import Image as PILImage
    assert isinstance(result[0]["content"], list)
    assert isinstance(result[0]["content"][1]["image"], PILImage.Image)

    # On disk, image becomes a base64 data URI string
    with open(out, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved[0]["content"][1]["type"] == "image"
    assert isinstance(saved[0]["content"][1]["image"], str)
    assert saved[0]["content"][1]["image"].startswith("data:image/")


# ---------------------------------------------------------------------------
# _record_type_of / _timestamp_of / _total_tokens_of
# ---------------------------------------------------------------------------

def test_record_type_of_with_payload_subtype():
    assert _record_type_of(_response_item({"type": "message"})) == "response_item:message"


def test_record_type_of_without_payload_subtype():
    assert _record_type_of({"type": "session_meta"}) == "session_meta"
    assert _record_type_of({}) == "unknown"


def test_timestamp_of_truncates_and_formats():
    r = {"timestamp": "2024-01-02T03:04:05.123456Z"}
    assert _timestamp_of(r) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_total_tokens_of_keeps_latest():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count",
                                          "info": {"total_token_usage": {"input_tokens": 1, "output_tokens": 2,
                                                                         "cached_input_tokens": 0,
                                                                         "total_tokens": 3}}}},
        {"type": "event_msg", "payload": {"type": "token_count",
                                          "info": {"total_token_usage": {"input_tokens": 10, "output_tokens": 20,
                                                                         "cached_input_tokens": 5,
                                                                         "total_tokens": 30}}}},
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cached_input_tokens": 5,
        "total_tokens": 30,
    }


def test_total_tokens_of_handles_null_info():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count", "info": None}},
        {"type": "event_msg", "payload": {"type": "token_count",
                                          "info": {"total_token_usage": {"input_tokens": 1, "output_tokens": 2,
                                                                         "total_tokens": 3}}}},
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 1,
        "output_tokens": 2,
        "cached_input_tokens": 0,
        "total_tokens": 3,
    }


def test_total_tokens_of_handles_null_usage():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": None}}},
    ]
    assert _total_tokens_of(records) == {}


def test_total_tokens_of_no_token_count_records():
    assert _total_tokens_of([{"type": "event_msg", "payload": {"type": "other"}}]) == {}


# ---------------------------------------------------------------------------
# InspectorSchema wiring
# ---------------------------------------------------------------------------

def test_codex_schema_callbacks():
    assert CODEX_SCHEMA.label == "CODEX"
    assert CODEX_SCHEMA.is_message(_msg("user", _text_block("x"))) is True
    assert CODEX_SCHEMA.is_message({"type": "session_meta"}) is False
    assert CODEX_SCHEMA.role_of(_msg("user", _text_block("x"))) == "user"
    assert CODEX_SCHEMA.role_of({"type": "session_meta"}) == "other"
    assert CODEX_SCHEMA.text_of(_msg("user", _text_block("x"))) == "x"
    assert CODEX_SCHEMA.record_type_of(_msg("user", _text_block("x"))) == "response_item:message"
    assert CODEX_SCHEMA.timestamp_of({"timestamp": "2024-05-06T07:08:09Z"}) == "2024-05-06 07:08:09"


# ---------------------------------------------------------------------------
# inspect_session_codex
# ---------------------------------------------------------------------------

def test_inspect_session_codex_prints_report(tmp_path, capsys):
    records = [_msg("user", _text_block("hello"))]
    p = _write_jsonl(tmp_path, records)
    inspect_session_codex(p, show_flow=True)
    out = capsys.readouterr().out
    assert "CODEX" in out
    assert "hello" in out


def test_inspect_session_codex_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_session_codex(tmp_path / "nope.jsonl")
