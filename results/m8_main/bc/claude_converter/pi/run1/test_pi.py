import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import pytest

from claude_converter.pi import (
    PI_SCHEMA,
    _block_to_text,
    _decode_pi_image,
    _message_of,
    _message_parts,
    _record_role,
    _record_type_of,
    _text_of,
    _timestamp_of,
    _total_tokens_of,
    inspect_session_pi,
    load_session_pi,
    records_to_messages_pi,
    session_to_messages_pi,
)


# ---------- helpers ----------

def _msg(role, content, **extra):
    message = {"role": role, "content": content}
    message.update(extra)
    return {"type": "message", "id": "x", "parentId": None, "message": message}


def _text_block(text):
    return {"type": "text", "text": text}


def _write_jsonl(path: Path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _tiny_png_base64():
    """Return base64 PNG data, skipping the test if Pillow is unavailable."""
    PIL_Image = pytest.importorskip("PIL.Image")
    buf = BytesIO()
    PIL_Image.new("RGB", (2, 2), "red").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _strip_ansi(s):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---------- load_session_pi ----------

def test_load_session_pi_reads_records(tmp_path):
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [{"type": "session", "id": "a"}, _msg("user", "hi")])
    records = load_session_pi(p)
    assert records == [{"type": "session", "id": "a"}, _msg("user", "hi")]


def test_load_session_pi_accepts_json_extension(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"type": "session"}\n', encoding="utf-8")
    assert load_session_pi(p) == [{"type": "session"}]


def test_load_session_pi_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_pi(tmp_path / "nope.jsonl")


def test_load_session_pi_bad_extension(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_pi(p)


def test_load_session_pi_empty(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_pi(p)


def test_load_session_pi_skips_invalid_lines(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text('{"type": "message"}\nnot json\n{"type": "session"}\n', encoding="utf-8")
    records = load_session_pi(p)
    assert records == [{"type": "message"}, {"type": "session"}]
    err = capsys.readouterr().err
    assert "Line 2 is invalid JSON" in err


# ---------- _block_to_text ----------

def test_block_to_text_text():
    assert _block_to_text({"type": "text", "text": "hi"}) == "hi"


def test_block_to_text_text_missing_key():
    assert _block_to_text({"type": "text"}) == ""


def test_block_to_text_thinking():
    assert _block_to_text({"type": "thinking", "thinking": "hmm"}) == "<thinking>hmm</thinking>"


def test_block_to_text_thinking_missing():
    assert _block_to_text({"type": "thinking"}) == "<thinking></thinking>"


def test_block_to_text_tool_call():
    out = _block_to_text({"type": "toolCall", "name": "n", "arguments": {"a": 1}})
    assert out == "<tool_use name='n'>{\"a\": 1}</tool_use>"


def test_block_to_text_tool_call_ensure_ascii_false():
    out = _block_to_text({"type": "toolCall", "name": "n", "arguments": {"k": "é"}})
    assert out == "<tool_use name='n'>{\"k\": \"é\"}</tool_use>"


def test_block_to_text_image():
    assert _block_to_text({"type": "image"}) == "<image>"


def test_block_to_text_unknown():
    assert _block_to_text({"type": "zzz"}) == ""


def test_block_to_text_empty_type():
    assert _block_to_text({}) == ""


# ---------- _decode_pi_image ----------

def test_decode_image_empty_data(capsys):
    assert _decode_pi_image({"type": "image", "data": ""}) is None
    assert "Skipping image block with empty base64 data" in capsys.readouterr().err


def test_decode_image_missing_data(capsys):
    assert _decode_pi_image({"type": "image"}) is None
    assert "Skipping image block with empty base64 data" in capsys.readouterr().err


def test_decode_image_invalid_base64(capsys):
    assert _decode_pi_image({"type": "image", "data": "!!!notbase64!!!"}) is None
    assert "Skipping unreadable image block" in capsys.readouterr().err


def test_decode_image_ok():
    pytest.importorskip("PIL.Image")
    b64 = _tiny_png_base64()
    img = _decode_pi_image({"type": "image", "data": b64, "mimeType": "image/png"})
    assert img.size == (2, 2)


# ---------- _message_of / _record_role / _record_type_of ----------

def test_message_of_non_message_record():
    assert _message_of({"type": "session"}) is None


def test_message_of_message_record():
    rec = _msg("user", "x")
    assert _message_of(rec) == rec["message"]


def test_message_of_null_message():
    # dget falls back to {} when the key is present but null.
    assert _message_of({"type": "message", "message": None}) == {}


def test_record_role_non_message():
    assert _record_role({"type": "session"}) is None


def test_record_role_user():
    assert _record_role(_msg("user", "x")) == "user"


def test_record_role_assistant():
    assert _record_role(_msg("assistant", "x")) == "assistant"


def test_record_role_tool_result_maps_to_user():
    rec = {"type": "message", "message": {"role": "toolResult", "content": []}}
    assert _record_role(rec) == "user"


def test_record_role_unknown_role():
    rec = {"type": "message", "message": {"role": "system", "content": "x"}}
    assert _record_role(rec) is None


def test_record_type_of_message():
    assert _record_type_of(_msg("user", "x")) == "message:user"


def test_record_type_of_message_default_role():
    assert _record_type_of({"type": "message"}) == "message:unknown"


def test_record_type_of_non_message():
    assert _record_type_of({"type": "session"}) == "session"


def test_record_type_of_missing_type():
    assert _record_type_of({}) == "unknown"


# ---------- _text_of ----------

def test_text_of_string_content():
    assert _text_of(_msg("user", "hello")) == "hello"


def test_text_of_list_content():
    rec = _msg("assistant", [_text_block("a"), _text_block("b")])
    assert _text_of(rec) == "a\nb"


def test_text_of_skips_empty_parts():
    rec = _msg("assistant", [_text_block("a"), _text_block(""), _text_block("b")])
    assert _text_of(rec) == "a\nb"


def test_text_of_non_string_non_list_content():
    assert _text_of(_msg("user", 123)) == ""


def test_text_of_non_message_record():
    assert _text_of({"type": "session"}) == ""


def test_text_of_tool_result():
    rec = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "read",
            "toolCallId": "c1",
            "content": [_text_block("data")],
        },
    }
    assert _text_of(rec) == "<tool_result name='read' id='c1'>data</tool_result>"


def test_text_of_tool_result_image_becomes_tag():
    rec = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "t",
            "toolCallId": "i",
            "content": [{"type": "image", "data": "x"}],
        },
    }
    assert _text_of(rec) == "<tool_result name='t' id='i'><image></tool_result>"


def test_text_of_tool_result_skips_non_dict_blocks():
    rec = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "t",
            "toolCallId": "i",
            "content": ["nope", _text_block("ok")],
        },
    }
    assert _text_of(rec) == "<tool_result name='t' id='i'>ok</tool_result>"


def test_text_of_tool_result_missing_keys():
    rec = {
        "type": "message",
        "message": {"role": "toolResult", "content": [_text_block("x")]},
    }
    assert _text_of(rec) == "<tool_result name='' id=''>x</tool_result>"


# ---------- _message_parts ----------

def test_message_parts_string_content():
    assert _message_parts(_msg("user", "hi")) == [{"type": "text", "text": "hi"}]


def test_message_parts_empty_string():
    assert _message_parts(_msg("user", "")) == []


def test_message_parts_whitespace_string():
    assert _message_parts(_msg("user", "   ")) == []


def test_message_parts_list_text():
    rec = _msg("assistant", [_text_block("a"), _text_block("b")])
    assert _message_parts(rec) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_message_parts_list_skips_empty_and_non_dict():
    rec = _msg("assistant", [_text_block(""), "junk", _text_block("x")])
    assert _message_parts(rec) == [{"type": "text", "text": "x"}]


def test_message_parts_non_string_non_list_content():
    assert _message_parts(_msg("user", 5)) == []


def test_message_parts_non_message():
    assert _message_parts({"type": "session"}) == []


def test_message_parts_tool_result_no_image():
    rec = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "read",
            "toolCallId": "c1",
            "content": [_text_block("file")],
        },
    }
    assert _message_parts(rec) == [
        {"type": "text", "text": "<tool_result name='read' id='c1'>file</tool_result>"}
    ]


def test_message_parts_tool_result_no_image_empty_text():
    rec = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "read",
            "toolCallId": "c1",
            "content": [_text_block("   ")],
        },
    }
    # _text_of produces a non-empty wrapper, so a text part is returned.
    assert _message_parts(rec) == [
        {"type": "text", "text": "<tool_result name='read' id='c1'></tool_result>"}
    ]


def test_message_parts_tool_result_with_image():
    pytest.importorskip("PIL.Image")
    b64 = _tiny_png_base64()
    rec = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "shot",
            "toolCallId": "c9",
            "content": [_text_block("s"), {"type": "image", "data": b64}],
        },
    }
    parts = _message_parts(rec)
    assert parts[0] == {"type": "text", "text": "<tool_result name='shot' id='c9'>"}
    assert parts[1] == {"type": "text", "text": "s"}
    assert parts[2]["type"] == "image"
    assert parts[-1] == {"type": "text", "text": "</tool_result>"}


def test_message_parts_tool_result_with_image_skips_empty_text():
    pytest.importorskip("PIL.Image")
    b64 = _tiny_png_base64()
    rec = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "t",
            "toolCallId": "i",
            "content": [_text_block("  "), {"type": "image", "data": b64}],
        },
    }
    parts = _message_parts(rec)
    assert parts[0]["text"] == "<tool_result name='t' id='i'>"
    assert parts[1]["type"] == "image"
    assert parts[-1] == {"type": "text", "text": "</tool_result>"}


def test_message_parts_tool_result_with_bad_image_skipped(capsys):
    rec = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "t",
            "toolCallId": "i",
            "content": [{"type": "image", "data": ""}],
        },
    }
    parts = _message_parts(rec)
    assert parts == [
        {"type": "text", "text": "<tool_result name='t' id='i'>"},
        {"type": "text", "text": "</tool_result>"},
    ]
    assert "Skipping image block with empty base64 data" in capsys.readouterr().err


def test_message_parts_tool_result_skips_non_dict_blocks():
    pytest.importorskip("PIL.Image")
    b64 = _tiny_png_base64()
    rec = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "t",
            "toolCallId": "i",
            "content": ["junk", _text_block("keep"), {"type": "image", "data": b64}],
        },
    }
    parts = _message_parts(rec)
    assert {"type": "text", "text": "keep"} in parts
    assert any(p["type"] == "image" for p in parts)


# ---------- records_to_messages_pi: basic ----------

def test_simple_user_assistant_text():
    records = [
        _msg("user", "hello"),
        _msg("assistant", "hi there"),
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_skips_non_message_records():
    records = [
        {"type": "session", "id": "x"},
        {"type": "model_change", "model": "m"},
        _msg("user", "hello"),
    ]
    assert records_to_messages_pi(records) == [{"role": "user", "content": "hello"}]


def test_skips_unknown_roles():
    records = [{"type": "message", "message": {"role": "system", "content": "sys"}}]
    assert records_to_messages_pi(records) == []


def test_skips_empty_content():
    records = [
        _msg("user", ""),
        _msg("user", "   "),
        _msg("assistant", "real"),
    ]
    assert records_to_messages_pi(records) == [{"role": "assistant", "content": "real"}]


def test_message_with_no_message_field():
    records = [{"type": "message"}]
    assert records_to_messages_pi(records) == []


def test_message_with_explicit_null_message():
    records = [{"type": "message", "message": None}]
    assert records_to_messages_pi(records) == []


def test_empty_records_returns_empty_list():
    assert records_to_messages_pi([]) == []


# ---------- tool calls ----------

def test_tool_call_rendering():
    records = [
        _msg("assistant", [
            _text_block("calling"),
            {"type": "toolCall", "name": "read", "arguments": {"path": "a.txt"}},
        ]),
    ]
    out = records_to_messages_pi(records)
    assert out == [{
        "role": "assistant",
        "content": "calling\n<tool_use name='read'>{\"path\": \"a.txt\"}</tool_use>",
    }]


def test_tool_call_default_arguments_empty():
    records = [_msg("assistant", [{"type": "toolCall", "name": "ping"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "assistant", "content": "<tool_use name='ping'>{}</tool_use>"}]


def test_thinking_block_rendering():
    records = [_msg("assistant", [{"type": "thinking", "thinking": "hmm"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "assistant", "content": "<thinking>hmm</thinking>"}]


def test_unknown_block_type_dropped():
    records = [_msg("assistant", [{"type": "weird", "x": 1}, _text_block("ok")])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "assistant", "content": "ok"}]


def test_non_dict_blocks_in_list_skipped():
    records = [_msg("user", ["not", "dicts", _text_block("kept")])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "kept"}]


# ---------- toolResult ----------

def test_tool_result_becomes_user_text():
    records = [{
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "read",
            "toolCallId": "c1",
            "content": [_text_block("file contents")],
        },
    }]
    out = records_to_messages_pi(records)
    assert out == [{
        "role": "user",
        "content": "<tool_result name='read' id='c1'>file contents</tool_result>",
    }]


def test_tool_result_empty_text_keeps_wrapper():
    # A whitespace-only text block is dropped from the inner text, but the
    # wrapper tag itself is non-empty so the turn is preserved.
    records = [{
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "read",
            "toolCallId": "c1",
            "content": [_text_block("   ")],
        },
    }]
    out = records_to_messages_pi(records)
    assert out == [{
        "role": "user",
        "content": "<tool_result name='read' id='c1'></tool_result>",
    }]


def test_tool_result_missing_name_and_id():
    records = [{
        "type": "message",
        "message": {
            "role": "toolResult",
            "content": [_text_block("x")],
        },
    }]
    out = records_to_messages_pi(records)
    assert out == [{
        "role": "user",
        "content": "<tool_result name='' id=''>x</tool_result>",
    }]


# ---------- merging ----------

def test_merges_consecutive_same_role():
    records = [
        _msg("user", "a"),
        _msg("user", "b"),
        _msg("assistant", "c"),
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {"role": "user", "content": "a\nb"},
        {"role": "assistant", "content": "c"},
    ]


def test_alternating_roles_not_merged():
    records = [_msg("user", "a"), _msg("assistant", "b"), _msg("user", "c")]
    out = records_to_messages_pi(records)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [m["content"] for m in out] == ["a", "b", "c"]


def test_nested_array_content_merged():
    records = [
        _msg("user", [_text_block("one")]),
        _msg("user", [_text_block("two")]),
    ]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "one\ntwo"}]


def test_tool_result_merges_with_adjacent_user():
    records = [
        _msg("user", "prompt"),
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "read",
                "toolCallId": "c1",
                "content": [_text_block("data")],
            },
        },
    ]
    out = records_to_messages_pi(records)
    assert out == [{
        "role": "user",
        "content": "prompt\n<tool_result name='read' id='c1'>data</tool_result>",
    }]


# ---------- images ----------

def test_image_block_in_user_message():
    pytest.importorskip("PIL.Image")
    b64 = _tiny_png_base64()
    records = [_msg("user", [
        _text_block("look"),
        {"type": "image", "data": b64, "mimeType": "image/png"},
    ])]
    out = records_to_messages_pi(records)
    assert len(out) == 1
    assert out[0]["role"] == "user"
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"
    assert content[1]["image"].size == (2, 2)


def test_image_only_message_becomes_list():
    pytest.importorskip("PIL.Image")
    b64 = _tiny_png_base64()
    records = [_msg("user", [{"type": "image", "data": b64}])]
    out = records_to_messages_pi(records)
    assert len(out) == 1
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"


def test_empty_image_data_skipped(capsys):
    records = [_msg("user", [
        _text_block("see"),
        {"type": "image", "data": ""},
    ])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "see"}]
    assert "Skipping image block with empty base64 data" in capsys.readouterr().err


def test_undecodable_image_skipped(capsys):
    records = [_msg("user", [
        {"type": "image", "data": "!!!notbase64!!!"},
    ])]
    out = records_to_messages_pi(records)
    assert out == []
    # Without Pillow, the ImportError is caught and reported instead;
    # with Pillow, the ValueError from bad base64 is reported.
    err = capsys.readouterr().err
    assert ("Skipping unreadable image block" in err)


def test_image_in_tool_result_kept_multimodal():
    pytest.importorskip("PIL.Image")
    b64 = _tiny_png_base64()
    records = [{
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "screenshot",
            "toolCallId": "c9",
            "content": [
                _text_block("shot"),
                {"type": "image", "data": b64},
            ],
        },
    }]
    out = records_to_messages_pi(records)
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "<tool_result name='screenshot' id='c9'>"}
    assert content[1] == {"type": "text", "text": "shot"}
    assert content[2]["type"] == "image"
    assert content[-1] == {"type": "text", "text": "</tool_result>"}


def test_tool_result_image_only_multimodal():
    pytest.importorskip("PIL.Image")
    b64 = _tiny_png_base64()
    records = [{
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolName": "t",
            "toolCallId": "id",
            "content": [{"type": "image", "data": b64}],
        },
    }]
    out = records_to_messages_pi(records)
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "<tool_result name='t' id='id'>"}
    assert content[1]["type"] == "image"
    assert content[-1] == {"type": "text", "text": "</tool_result>"}


# ---------- session_to_messages_pi ----------

def test_session_to_messages_reads_file(tmp_path):
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [_msg("user", "hey"), _msg("assistant", "yo")])
    out = session_to_messages_pi(p)
    assert out == [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "yo"},
    ]


def test_session_to_messages_writes_output(tmp_path):
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [_msg("user", "hey")])
    out_path = tmp_path / "out" / "res.json"
    out = session_to_messages_pi(p, output=out_path)
    assert out == [{"role": "user", "content": "hey"}]
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved == [{"role": "user", "content": "hey"}]


def test_session_to_messages_output_image_serialized(tmp_path):
    pytest.importorskip("PIL.Image")
    b64 = _tiny_png_base64()
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [_msg("user", [
        _text_block("pic"),
        {"type": "image", "data": b64},
    ])])
    out_path = tmp_path / "res.json"
    out = session_to_messages_pi(p, output=out_path)
    # In-memory keeps a PIL image object
    assert out[0]["content"][1]["type"] == "image"
    assert not isinstance(out[0]["content"][1]["image"], str)

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    img = saved[0]["content"][1]["image"]
    assert isinstance(img, str)
    assert img.startswith("data:image/")


def test_session_to_messages_no_output_no_file(tmp_path):
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [_msg("user", "hey")])
    out = session_to_messages_pi(p)
    assert out == [{"role": "user", "content": "hey"}]
    assert list(tmp_path.glob("*.json")) == []


def test_session_to_messages_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        session_to_messages_pi(tmp_path / "nope.jsonl")


# ---------- inspect-related helpers ----------

def test_timestamp_of_iso():
    assert _timestamp_of({"timestamp": "2024-01-02T03:04:05.678Z"}) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_total_tokens_of_empty():
    assert _total_tokens_of([]) == {}


def test_total_tokens_of_no_usage():
    assert _total_tokens_of([_msg("user", "hi")]) == {}


def test_total_tokens_of_sums_usage():
    records = [
        _msg("user", "hi"),
        _msg("assistant", "yo", usage={"input": 10, "output": 20, "cacheRead": 3, "cacheWrite": 4}),
        _msg("assistant", "again", usage={"input": 1, "output": 2}),
        {"type": "session"},
    ]
    totals = _total_tokens_of(records)
    assert totals == {"input": 11, "output": 22, "cache_read": 3, "cache_write": 4}


def test_total_tokens_of_ignores_null_usage():
    records = [_msg("assistant", "x", usage=None)]
    assert _total_tokens_of(records) == {}


# ---------- PI_SCHEMA wiring ----------

def test_pi_schema_label():
    assert PI_SCHEMA.label == "PI"


def test_pi_schema_callbacks():
    user_rec = _msg("user", "hi")
    assert PI_SCHEMA.record_type_of(user_rec) == "message:user"
    assert PI_SCHEMA.is_message(user_rec) is True
    assert PI_SCHEMA.role_of(user_rec) == "user"
    assert PI_SCHEMA.text_of(user_rec) == "hi"
    assert PI_SCHEMA.timestamp_of({"timestamp": "2024-01-02T03:04:05Z"}) == "2024-01-02 03:04:05"

    session_rec = {"type": "session"}
    assert PI_SCHEMA.is_message(session_rec) is False
    assert PI_SCHEMA.role_of(session_rec) == "other"
    assert PI_SCHEMA.total_tokens_of is not None


# ---------- inspect_session_pi ----------

def test_inspect_session_pi_prints_report(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [
        {"type": "session", "id": "s1"},
        _msg("user", "hi"),
        _msg("assistant", "hello", usage={"input": 3, "output": 4,
                                          "cacheRead": 1, "cacheWrite": 2}),
    ])
    inspect_session_pi(p, show_flow=True)
    out = _strip_ansi(capsys.readouterr().out)
    # The report header and the message-flow section are always printed.
    assert "PI SESSION" in out
    # Turn roles/record types are always summarized.
    assert "message:user" in out
    assert "message:assistant" in out


def test_inspect_session_pi_show_blocks_renders_text(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [
        _msg("user", "hi"),
        _msg("assistant", "hello"),
    ])
    inspect_session_pi(p, show_flow=True, show_blocks=True)
    # show_blocks relies on schema.blocks_of, which PI_SCHEMA does not
    # provide, so text is not rendered inline. The flow still reports roles.
    out = _strip_ansi(capsys.readouterr().out)
    assert "PI SESSION" in out
    assert "message:user" in out
    assert "message:assistant" in out


def test_inspect_session_pi_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_session_pi(tmp_path / "nope.jsonl")


def test_inspect_session_pi_token_usage(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [
        _msg("user", "hi"),
        _msg("assistant", "hello", usage={"input": 10, "output": 20,
                                          "cacheRead": 3, "cacheWrite": 4}),
    ])
    inspect_session_pi(p)
    out = _strip_ansi(capsys.readouterr().out)
    assert "PI SESSION" in out


def test_inspect_session_pi_show_flow_prints_message_text(tmp_path, capsys):
    # Without schema.blocks_of, print_flow does not inline block text, so
    # the message text is not echoed in the flow. Only roles are shown.
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [_msg("user", "unique_prompt_xyz")])
    inspect_session_pi(p, show_flow=True)
    out = _strip_ansi(capsys.readouterr().out)
    assert "PI SESSION" in out
    assert "message:user" in out
