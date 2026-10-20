"""Unit tests for claude_converter.pi."""

from __future__ import annotations

import base64
import io
import json

import pytest

import claude_converter.pi as pi_module
from claude_converter.pi import (
    load_session_pi,
    records_to_messages_pi,
    session_to_messages_pi,
    inspect_session_pi,
    _block_to_text,
    _decode_pi_image,
    _message_of,
    _record_role,
    _text_of,
    _message_parts,
    _record_type_of,
    _timestamp_of,
    _total_tokens_of,
    PI_SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role, content, **extra):
    message = {"role": role, "content": content}
    message.update(extra)
    return {"type": "message", "message": message}


def _write_jsonl(tmp_path, records, name="session.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _png_bytes():
    """A tiny valid 1x1 PNG."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


def _png_b64():
    return base64.b64encode(_png_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# load_session_pi
# ---------------------------------------------------------------------------

def test_load_session_pi_reads_records(tmp_path):
    records = [
        {"type": "session", "id": "abc"},
        _msg("user", "hello"),
    ]
    path = _write_jsonl(tmp_path, records)
    assert load_session_pi(path) == records


def test_load_session_pi_accepts_str_path(tmp_path):
    path = _write_jsonl(tmp_path, [_msg("user", "hi")])
    assert load_session_pi(str(path)) == [_msg("user", "hi")]


def test_load_session_pi_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_pi(tmp_path / "nope.jsonl")


def test_load_session_pi_bad_extension(tmp_path):
    path = tmp_path / "session.txt"
    path.write_text("{}")
    with pytest.raises(ValueError):
        load_session_pi(path)


def test_load_session_pi_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(ValueError):
        load_session_pi(path)


# ---------------------------------------------------------------------------
# _block_to_text
# ---------------------------------------------------------------------------

def test_block_to_text_text():
    assert _block_to_text({"type": "text", "text": "hi"}) == "hi"


def test_block_to_text_thinking():
    assert _block_to_text({"type": "thinking", "thinking": "hmm"}) == "<thinking>hmm</thinking>"


def test_block_to_text_tool_call():
    out = _block_to_text({"type": "toolCall", "name": "ls", "arguments": {"a": 1}})
    assert out == "<tool_use name='ls'>{\"a\": 1}</tool_use>"


def test_block_to_text_tool_call_no_args():
    out = _block_to_text({"type": "toolCall", "name": "ls"})
    assert out == "<tool_use name='ls'>{}</tool_use>"


def test_block_to_text_image():
    assert _block_to_text({"type": "image"}) == "<image>"


def test_block_to_text_unknown():
    assert _block_to_text({"type": "weird"}) == ""


# ---------------------------------------------------------------------------
# _decode_pi_image
# ---------------------------------------------------------------------------

def test_decode_pi_image_empty_prints_warning(capsys):
    assert _decode_pi_image({"type": "image", "data": ""}) is None
    err = capsys.readouterr().err
    assert "empty base64 data" in err


def test_decode_pi_image_bad_data_prints_warning(capsys):
    result = _decode_pi_image({"type": "image", "data": "not-base64!!!"})
    assert result is None
    assert "Skipping unreadable image block" in capsys.readouterr().err


def test_decode_pi_image_valid():
    image = _decode_pi_image({"type": "image", "data": _png_b64()})
    assert image is not None
    assert image.size == (1, 1)


def test_decode_pi_image_import_error(monkeypatch, capsys):
    def boom(data):
        raise ImportError("no pillow")

    monkeypatch.setattr(pi_module, "convert_base64_to_pil_image", boom)
    assert _decode_pi_image({"type": "image", "data": "abc"}) is None
    assert "no pillow" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _message_of / _record_role
# ---------------------------------------------------------------------------

def test_message_of_non_message():
    assert _message_of({"type": "session"}) is None


def test_message_of_message():
    assert _message_of(_msg("user", "x")) == {"role": "user", "content": "x"}


def test_record_role_user():
    assert _record_role(_msg("user", "x")) == "user"


def test_record_role_assistant():
    assert _record_role(_msg("assistant", "x")) == "assistant"


def test_record_role_tool_result_maps_to_user():
    assert _record_role(_msg("toolResult", [])) == "user"


def test_record_role_other():
    assert _record_role(_msg("system", "x")) is None


def test_record_role_non_message():
    assert _record_role({"type": "session"}) is None


def test_record_role_null_message():
    assert _record_role({"type": "message", "message": None}) is None


# ---------------------------------------------------------------------------
# _text_of
# ---------------------------------------------------------------------------

def test_text_of_string_content():
    assert _text_of(_msg("user", "hello")) == "hello"


def test_text_of_list_content():
    content = [
        {"type": "text", "text": "a"},
        {"type": "thinking", "thinking": "b"},
        {"type": "toolCall", "name": "t", "arguments": {}},
    ]
    out = _text_of(_msg("assistant", content))
    assert out == "a\n<thinking>b</thinking>\n<tool_use name='t'>{}</tool_use>"


def test_text_of_skips_blank_parts():
    content = [{"type": "text", "text": "  "}, {"type": "text", "text": "x"}]
    assert _text_of(_msg("user", content)) == "x"


def test_text_of_non_message():
    assert _text_of({"type": "session"}) == ""


def test_text_of_content_other_type():
    assert _text_of(_msg("user", 123)) == ""


def test_text_of_tool_result():
    content = [
        {"type": "text", "text": "out"},
        {"type": "image"},
    ]
    record = _msg("toolResult", content, toolName="bash", toolCallId="id1")
    assert _text_of(record) == "<tool_result name='bash' id='id1'>out\n<image></tool_result>"


def test_text_of_tool_result_empty():
    record = _msg("toolResult", [], toolName="x", toolCallId="y")
    assert _text_of(record) == "<tool_result name='x' id='y'></tool_result>"


# ---------------------------------------------------------------------------
# _message_parts
# ---------------------------------------------------------------------------

def test_message_parts_string_content():
    assert _message_parts(_msg("user", "hi")) == [{"type": "text", "text": "hi"}]


def test_message_parts_string_content_blank():
    assert _message_parts(_msg("user", "   ")) == []


def test_message_parts_list_text_blocks():
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "  "}]
    assert _message_parts(_msg("user", content)) == [{"type": "text", "text": "a"}]


def test_message_parts_image_block():
    content = [{"type": "image", "data": _png_b64()}]
    parts = _message_parts(_msg("user", content))
    assert len(parts) == 1
    assert parts[0]["type"] == "image"
    assert parts[0]["image"].size == (1, 1)


def test_message_parts_tool_result_text_only():
    content = [{"type": "text", "text": "out"}]
    record = _msg("toolResult", content, toolName="bash", toolCallId="id1")
    assert _message_parts(record) == [
        {"type": "text", "text": "<tool_result name='bash' id='id1'>out</tool_result>"}
    ]


def test_message_parts_tool_result_text_only_empty():
    record = _msg("toolResult", [], toolName="x", toolCallId="y")
    assert _message_parts(record) == [
        {"type": "text", "text": "<tool_result name='x' id='y'></tool_result>"}
    ]


def test_message_parts_tool_result_with_image():
    content = [
        {"type": "text", "text": "out"},
        {"type": "image", "data": _png_b64()},
    ]
    record = _msg("toolResult", content, toolName="bash", toolCallId="id1")
    parts = _message_parts(record)
    assert parts[0] == {"type": "text", "text": "<tool_result name='bash' id='id1'>"}
    assert parts[1] == {"type": "text", "text": "out"}
    assert parts[2]["type"] == "image"
    assert parts[2]["image"].size == (1, 1)
    assert parts[3] == {"type": "text", "text": "</tool_result>"}


def test_message_parts_tool_result_with_bad_image(capsys):
    content = [
        {"type": "text", "text": "out"},
        {"type": "image", "data": ""},
    ]
    record = _msg("toolResult", content, toolName="bash", toolCallId="id1")
    parts = _message_parts(record)
    assert [p["type"] for p in parts] == ["text", "text", "text"]
    assert "empty base64 data" in capsys.readouterr().err


def test_message_parts_non_message():
    assert _message_parts({"type": "session"}) == []


# ---------------------------------------------------------------------------
# records_to_messages_pi
# ---------------------------------------------------------------------------

def test_records_to_messages_simple_exchange():
    records = [
        {"type": "session"},
        _msg("user", "hello"),
        _msg("assistant", "world"),
    ]
    assert records_to_messages_pi(records) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


def test_records_to_messages_skips_empty_and_other_roles():
    records = [
        _msg("user", ""),
        _msg("system", "sys"),
        _msg("assistant", "hi"),
    ]
    assert records_to_messages_pi(records) == [{"role": "assistant", "content": "hi"}]


def test_records_to_messages_merges_consecutive_same_role():
    records = [
        _msg("user", "one"),
        _msg("user", "two"),
        _msg("assistant", "three"),
    ]
    assert records_to_messages_pi(records) == [
        {"role": "user", "content": "one\ntwo"},
        {"role": "assistant", "content": "three"},
    ]


def test_records_to_messages_tool_result_merges_with_user():
    records = [
        _msg("user", "ask"),
        _msg("toolResult", [{"type": "text", "text": "res"}], toolName="t", toolCallId="1"),
    ]
    result = records_to_messages_pi(records)
    assert result == [
        {"role": "user", "content": "ask\n<tool_result name='t' id='1'>res</tool_result>"}
    ]


def test_records_to_messages_with_image_becomes_list():
    records = [
        _msg("user", [
            {"type": "text", "text": "look"},
            {"type": "image", "data": _png_b64()},
        ]),
    ]
    result = records_to_messages_pi(records)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"


def test_records_to_messages_empty_records():
    assert records_to_messages_pi([]) == []


# ---------------------------------------------------------------------------
# session_to_messages_pi
# ---------------------------------------------------------------------------

def test_session_to_messages_reads_and_converts(tmp_path):
    records = [_msg("user", "hi"), _msg("assistant", "yo")]
    path = _write_jsonl(tmp_path, records)
    assert session_to_messages_pi(path) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_session_to_messages_writes_output(tmp_path):
    records = [_msg("user", "hi")]
    path = _write_jsonl(tmp_path, records)
    out = tmp_path / "nested" / "out.json"
    messages = session_to_messages_pi(path, output=out)

    assert messages == [{"role": "user", "content": "hi"}]
    assert out.exists()
    with open(out, encoding="utf-8") as f:
        assert json.load(f) == [{"role": "user", "content": "hi"}]


def test_session_to_messages_output_with_image(tmp_path):
    records = [_msg("user", [{"type": "image", "data": _png_b64()}])]
    path = _write_jsonl(tmp_path, records)
    out = tmp_path / "out.json"
    messages = session_to_messages_pi(path, output=out)

    # In-memory keeps the PIL image
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][0]["image"].size == (1, 1)

    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    image_part = data[0]["content"][0]
    assert image_part["type"] == "image"
    assert image_part["image"].startswith("data:image/")


# ---------------------------------------------------------------------------
# Inspector helpers
# ---------------------------------------------------------------------------

def test_record_type_of_message():
    assert _record_type_of(_msg("user", "x")) == "message:user"


def test_record_type_of_other():
    assert _record_type_of({"type": "session"}) == "session"


def test_record_type_of_default():
    assert _record_type_of({}) == "unknown"


def test_timestamp_of_iso():
    assert _timestamp_of({"timestamp": "2024-01-02T03:04:05.123Z"}) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_total_tokens_of_sums():
    records = [
        _msg("user", "x", usage={
            "input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 1,
        }),
        _msg("assistant", "y", usage={"input": 3, "output": 4}),
    ]
    assert _total_tokens_of(records) == {
        "input": 13, "output": 9, "cache_read": 2, "cache_write": 1,
    }


def test_total_tokens_of_no_usage_returns_empty():
    assert _total_tokens_of([_msg("user", "x")]) == {}


def test_total_tokens_of_ignores_non_messages():
    records = [
        {"type": "session", "message": {"usage": {"input": 100}}},
    ]
    assert _total_tokens_of(records) == {}


def test_pi_schema_fields():
    assert PI_SCHEMA.label == "PI"
    assert PI_SCHEMA.is_message(_msg("user", "x")) is True
    assert PI_SCHEMA.is_message({"type": "session"}) is False
    assert PI_SCHEMA.role_of(_msg("assistant", "x")) == "assistant"
    assert PI_SCHEMA.role_of({"type": "session"}) == "other"


# ---------------------------------------------------------------------------
# inspect_session_pi
# ---------------------------------------------------------------------------

def test_inspect_session_pi_outputs(capsys, tmp_path):
    records = [
        {"type": "session", "id": "abc", "timestamp": "2024-01-01T00:00:00Z"},
        _msg("user", "hello", usage={"input": 1, "output": 2}),
        _msg("assistant", "world"),
    ]
    path = _write_jsonl(tmp_path, records)
    inspect_session_pi(path)
    out = capsys.readouterr().out
    assert "PI" in out
    assert "hello" in out


def test_inspect_session_pi_show_flow(capsys, tmp_path):
    path = _write_jsonl(tmp_path, [_msg("user", "hi")])
    inspect_session_pi(path, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "hi" in out
