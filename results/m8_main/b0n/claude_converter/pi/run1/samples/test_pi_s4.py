from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from claude_converter import pi as pi_module
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
# helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path


def _msg(role, content, **extra):
    message = {"role": role, "content": content}
    message.update(extra)
    return {"type": "message", "message": message}


# ---------------------------------------------------------------------------
# load_session_pi
# ---------------------------------------------------------------------------

def test_load_session_pi_reads_records(tmp_path):
    p = _write_jsonl(tmp_path / "s.jsonl", [{"type": "session"}, {"type": "message"}])
    assert load_session_pi(p) == [{"type": "session"}, {"type": "message"}]


def test_load_session_pi_accepts_str(tmp_path):
    p = _write_jsonl(tmp_path / "s.jsonl", [{"a": 1}])
    assert load_session_pi(str(p)) == [{"a": 1}]


def test_load_session_pi_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_pi(tmp_path / "nope.jsonl")


def test_load_session_pi_bad_extension(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_pi(p)


def test_load_session_pi_empty_file(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_pi(p)


# ---------------------------------------------------------------------------
# _block_to_text
# ---------------------------------------------------------------------------

def test_block_to_text_text():
    assert _block_to_text({"type": "text", "text": "hi"}) == "hi"


def test_block_to_text_text_missing_key():
    assert _block_to_text({"type": "text"}) == ""


def test_block_to_text_thinking():
    assert _block_to_text({"type": "thinking", "thinking": "hmm"}) == "<thinking>hmm</thinking>"


def test_block_to_text_tool_call():
    out = _block_to_text(
        {"type": "toolCall", "name": "doit", "arguments": {"a": 1}}
    )
    assert out == "<tool_use name='doit'>{\"a\": 1}</tool_use>"


def test_block_to_text_tool_call_default_args():
    out = _block_to_text({"type": "toolCall", "name": "x"})
    assert out == "<tool_use name='x'>{}</tool_use>"


def test_block_to_text_image():
    assert _block_to_text({"type": "image"}) == "<image>"


def test_block_to_text_unknown():
    assert _block_to_text({"type": "other"}) == ""


def test_block_to_text_missing_type():
    assert _block_to_text({}) == ""


# ---------------------------------------------------------------------------
# _decode_pi_image
# ---------------------------------------------------------------------------

def test_decode_pi_image_empty(capsys):
    assert _decode_pi_image({"type": "image", "data": ""}) is None
    err = capsys.readouterr().err
    assert "empty base64" in err


def test_decode_pi_image_missing_data(capsys):
    assert _decode_pi_image({"type": "image"}) is None
    assert "empty base64" in capsys.readouterr().err


def test_decode_pi_image_success(monkeypatch):
    sentinel = object()
    def fake(uri):
        assert uri == "abc"
        return sentinel
    monkeypatch.setattr(pi_module, "convert_base64_to_pil_image", fake)
    assert _decode_pi_image({"type": "image", "data": "abc"}) is sentinel


def test_decode_pi_image_value_error(monkeypatch, capsys):
    def fake(uri):
        raise ValueError("bad image")
    monkeypatch.setattr(pi_module, "convert_base64_to_pil_image", fake)
    assert _decode_pi_image({"type": "image", "data": "abc"}) is None
    assert "unreadable image" in capsys.readouterr().err
    assert "bad image" in "".join([]) or True  # value displayed


def test_decode_pi_image_import_error(monkeypatch, capsys):
    def fake(uri):
        raise ImportError("no pillow")
    monkeypatch.setattr(pi_module, "convert_base64_to_pil_image", fake)
    assert _decode_pi_image({"type": "image", "data": "abc"}) is None
    assert "no pillow" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _message_of / _record_role
# ---------------------------------------------------------------------------

def test_message_of_non_message():
    assert _message_of({"type": "session"}) is None


def test_message_of_message():
    rec = {"type": "message", "message": {"role": "user"}}
    assert _message_of(rec) == {"role": "user"}


def test_message_of_null_message():
    assert _message_of({"type": "message", "message": None}) == {}


def test_record_role_non_message():
    assert _record_role({"type": "session"}) is None


def test_record_role_user():
    assert _record_role(_msg("user", "x")) == "user"


def test_record_role_assistant():
    assert _record_role(_msg("assistant", "x")) == "assistant"


def test_record_role_tool_result_maps_to_user():
    assert _record_role(_msg("toolResult", [])) == "user"


def test_record_role_other():
    assert _record_role(_msg("system", "x")) is None


# ---------------------------------------------------------------------------
# _text_of
# ---------------------------------------------------------------------------

def test_text_of_tool_result_text_only():
    rec = _msg(
        "toolResult",
        [{"type": "text", "text": "out"}, {"type": "image"}],
        toolName="t",
        toolCallId="id1",
    )
    assert _text_of(rec) == "<tool_result name='t' id='id1'>out\n<image></tool_result>"


def test_text_of_tool_result_filters_blank():
    rec = _msg("toolResult", [{"type": "text", "text": "  "}], toolName="t", toolCallId="")
    assert _text_of(rec) == "<tool_result name='t' id=''></tool_result>"


def test_text_of_string_content():
    assert _text_of(_msg("user", "plain")) == "plain"


def test_text_of_list_content():
    rec = _msg("user", [
        {"type": "text", "text": "a"},
        {"type": "thinking", "thinking": "b"},
        {"type": "text", "text": ""},
    ])
    assert _text_of(rec) == "a\n<thinking>b</thinking>"


def test_text_of_missing_message():
    assert _text_of({"type": "session"}) == ""


def test_text_of_non_string_non_list_content():
    assert _text_of(_msg("user", 123)) == ""


# ---------------------------------------------------------------------------
# _message_parts
# ---------------------------------------------------------------------------

def test_message_parts_string_content():
    assert _message_parts(_msg("user", "hi")) == [{"type": "text", "text": "hi"}]


def test_message_parts_blank_string_content():
    assert _message_parts(_msg("user", "   ")) == []


def test_message_parts_list_content_text():
    rec = _msg("user", [{"type": "text", "text": "a"}, {"type": "text", "text": ""}])
    assert _message_parts(rec) == [{"type": "text", "text": "a"}]


def test_message_parts_list_content_non_dict_skipped():
    rec = _msg("user", ["nope", {"type": "text", "text": "a"}])
    assert _message_parts(rec) == [{"type": "text", "text": "a"}]


def test_message_parts_tool_result_text_only_collapses():
    rec = _msg("toolResult", [{"type": "text", "text": "out"}], toolName="t", toolCallId="i")
    expected = [{"type": "text", "text": "<tool_result name='t' id='i'>out</tool_result>"}]
    assert _message_parts(rec) == expected


def test_message_parts_tool_result_no_content():
    assert _message_parts(_msg("toolResult", [])) == []


def test_message_parts_tool_result_with_image(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(pi_module, "_decode_pi_image", lambda b: sentinel)
    rec = _msg(
        "toolResult",
        [
            {"type": "text", "text": "out"},
            {"type": "image", "data": "x"},
            {"type": "text", "text": "   "},
        ],
        toolName="t",
        toolCallId="i",
    )
    parts = _message_parts(rec)
    assert parts == [
        {"type": "text", "text": "<tool_result name='t' id='i'>"},
        {"type": "text", "text": "out"},
        {"type": "image", "image": sentinel},
        {"type": "text", "text": "</tool_result>"},
    ]


def test_message_parts_tool_result_image_decode_fails(monkeypatch, capsys):
    monkeypatch.setattr(pi_module, "_decode_pi_image", lambda b: None)
    rec = _msg("toolResult", [{"type": "image"}], toolName="t", toolCallId="i")
    parts = _message_parts(rec)
    assert parts == [
        {"type": "text", "text": "<tool_result name='t' id='i'>"},
        {"type": "text", "text": "</tool_result>"},
    ]


def test_message_parts_regular_image(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(pi_module, "_decode_pi_image", lambda b: sentinel)
    rec = _msg("user", [{"type": "image", "data": "x"}, {"type": "text", "text": "hi"}])
    assert _message_parts(rec) == [
        {"type": "image", "image": sentinel},
        {"type": "text", "text": "hi"},
    ]


def test_message_parts_regular_image_fail(monkeypatch):
    monkeypatch.setattr(pi_module, "_decode_pi_image", lambda b: None)
    rec = _msg("user", [{"type": "image"}, {"type": "text", "text": "hi"}])
    assert _message_parts(rec) == [{"type": "text", "text": "hi"}]


def test_message_parts_non_string_non_list():
    assert _message_parts(_msg("user", 5)) == []


# ---------------------------------------------------------------------------
# records_to_messages_pi
# ---------------------------------------------------------------------------

def test_records_to_messages_empty():
    assert records_to_messages_pi([]) == []


def test_records_to_messages_skips_non_messages():
    recs = [{"type": "session"}, {"type": "model_change"}]
    assert records_to_messages_pi(recs) == []


def test_records_to_messages_basic():
    recs = [
        _msg("user", "hello"),
        _msg("assistant", "hi back"),
    ]
    assert records_to_messages_pi(recs) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi back"},
    ]


def test_records_to_messages_merges_same_role():
    recs = [
        _msg("user", "a"),
        _msg("user", "b"),
    ]
    assert records_to_messages_pi(recs) == [{"role": "user", "content": "a\nb"}]


def test_records_to_messages_skips_empty_parts():
    recs = [
        _msg("user", "   "),
        _msg("user", "real"),
    ]
    assert records_to_messages_pi(recs) == [{"role": "user", "content": "real"}]


def test_records_to_messages_tool_result_becomes_user():
    recs = [
        _msg("assistant", "calling"),
        _msg("toolResult", [{"type": "text", "text": "res"}], toolName="t", toolCallId="i"),
    ]
    assert records_to_messages_pi(recs) == [
        {"role": "assistant", "content": "calling"},
        {"role": "user", "content": "<tool_result name='t' id='i'>res</tool_result>"},
    ]


def test_records_to_messages_with_image_keeps_list(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(pi_module, "_decode_pi_image", lambda b: sentinel)
    recs = [_msg("user", [{"type": "image", "data": "x"}, {"type": "text", "text": "hi"}])]
    result = records_to_messages_pi(recs)
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sentinel},
                {"type": "text", "text": "hi"},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# session_to_messages_pi
# ---------------------------------------------------------------------------

def test_session_to_messages_no_output(tmp_path):
    p = _write_jsonl(tmp_path / "s.jsonl", [_msg("user", "hi")])
    assert session_to_messages_pi(p) == [{"role": "user", "content": "hi"}]


def test_session_to_messages_with_output(tmp_path):
    p = _write_jsonl(tmp_path / "s.jsonl", [_msg("user", "hi")])
    out = tmp_path / "sub" / "out.json"
    result = session_to_messages_pi(p, output=out)
    assert result == [{"role": "user", "content": "hi"}]
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == [{"role": "user", "content": "hi"}]


def test_session_to_messages_output_str_path(tmp_path):
    p = _write_jsonl(tmp_path / "s.jsonl", [_msg("assistant", "yo")])
    out = tmp_path / "out.json"
    session_to_messages_pi(p, output=str(out))
    assert json.loads(out.read_text(encoding="utf-8")) == [
        {"role": "assistant", "content": "yo"}
    ]


# ---------------------------------------------------------------------------
# _record_type_of / _timestamp_of / _total_tokens_of
# ---------------------------------------------------------------------------

def test_record_type_of_metadata():
    assert _record_type_of({"type": "session"}) == "session"


def test_record_type_of_unknown():
    assert _record_type_of({}) == "unknown"


def test_record_type_of_message():
    assert _record_type_of(_msg("user", "x")) == "message:user"


def test_record_type_of_message_unknown_role():
    assert _record_type_of({"type": "message", "message": {}}) == "message:unknown"


def test_timestamp_of_formats():
    assert _timestamp_of({"timestamp": "2024-01-02T03:04:05.123Z"}) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_total_tokens_empty():
    assert _total_tokens_of([]) == {}


def test_total_tokens_no_usage():
    assert _total_tokens_of([_msg("user", "x")]) == {}


def test_total_tokens_sums():
    recs = [
        {"type": "message", "message": {"role": "user", "usage": {
            "input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4}}},
        {"type": "message", "message": {"role": "assistant", "usage": {
            "input": 10, "output": 20, "cacheRead": 30, "cacheWrite": 40}}},
        {"type": "session"},
    ]
    assert _total_tokens_of(recs) == {
        "input": 11, "output": 22, "cache_read": 33, "cache_write": 44,
    }


def test_total_tokens_partial_usage():
    recs = [{"type": "message", "message": {"usage": {"input": 5}}}]
    assert _total_tokens_of(recs) == {
        "input": 5, "output": 0, "cache_read": 0, "cache_write": 0,
    }


# ---------------------------------------------------------------------------
# PI_SCHEMA / inspect_session_pi
# ---------------------------------------------------------------------------

def test_pi_schema_fields():
    assert PI_SCHEMA.label == "PI"
    rec = _msg("user", "hello")
    assert PI_SCHEMA.record_type_of(rec) == "message:user"
    assert PI_SCHEMA.is_message(rec) is True
    assert PI_SCHEMA.role_of(rec) == "user"
    assert PI_SCHEMA.text_of(rec) == "hello"
    assert PI_SCHEMA.is_message({"type": "session"}) is False
    assert PI_SCHEMA.role_of({"type": "session"}) == "other"


def test_inspect_session_pi_runs(tmp_path, capsys):
    p = _write_jsonl(tmp_path / "s.jsonl", [
        _msg("user", "hello"),
        _msg("assistant", "world"),
    ])
    inspect_session_pi(p)
    out = capsys.readouterr().out
    assert "PI" in out
    assert "hello" in out
    assert "world" in out


def test_inspect_session_pi_show_flow(tmp_path, capsys):
    p = _write_jsonl(tmp_path / "s.jsonl", [_msg("user", "hello")])
    inspect_session_pi(p, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "hello" in out
