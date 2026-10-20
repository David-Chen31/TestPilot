from __future__ import annotations

import json
import sys

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


def _response_item(payload: dict) -> dict:
    return {"type": "response_item", "payload": payload}


def _message(role: str, *blocks: dict) -> dict:
    return _response_item({"type": "message", "role": role, "content": list(blocks)})


def _text_block(text: str, btype: str = "input_text") -> dict:
    return {"type": btype, "text": text}


def _write_jsonl(tmp_path, records: list[dict], name: str = "session.jsonl"):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------


def test_load_session_codex_reads_records(tmp_path):
    records = [{"type": "event_msg", "payload": {"type": "x"}},
               {"type": "response_item", "payload": {"type": "message"}}]
    path = _write_jsonl(tmp_path, records)
    assert load_session_codex(path) == records


def test_load_session_codex_str_path(tmp_path):
    records = [{"a": 1}]
    path = _write_jsonl(tmp_path, records)
    assert load_session_codex(str(path)) == records


def test_load_session_codex_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_codex(tmp_path / "nope.jsonl")


def test_load_session_codex_bad_extension(tmp_path):
    p = tmp_path / "session.txt"
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_codex(p)


def test_load_session_codex_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_codex(p)


def test_load_session_codex_skips_bad_json(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
    records = load_session_codex(p)
    assert records == [{"a": 1}, {"b": 2}]
    assert "invalid JSON" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _payload_text
# ---------------------------------------------------------------------------


def test_payload_text_message_plain():
    payload = {"type": "message", "content": [
        {"type": "input_text", "text": "hello"},
        {"type": "output_text", "text": "world"},
        {"type": "text", "text": "third"},
    ]}
    assert _payload_text(payload) == "hello\nworld\nthird"


def test_payload_text_message_skips_blank_and_non_dict():
    payload = {"type": "message", "content": [
        {"type": "input_text", "text": "   "},
        "not a dict",
        {"type": "text", "text": "keep"},
    ]}
    assert _payload_text(payload) == "keep"


def test_payload_text_message_image_tag():
    payload = {"type": "message", "content": [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": "data:..."},
    ]}
    assert _payload_text(payload) == "look\n<image>"


def test_payload_text_message_null_content():
    assert _payload_text({"type": "message", "content": None}) == ""


def test_payload_text_reasoning():
    payload = {"type": "reasoning", "summary": [
        {"text": "step one"},
        {"text": "step two"},
        {"text": ""},
        "not a dict",
    ]}
    assert _payload_text(payload) == "<thinking>step one\nstep two</thinking>"


def test_payload_text_reasoning_empty():
    assert _payload_text({"type": "reasoning", "summary": []}) == ""


def test_payload_text_function_call():
    payload = {"type": "function_call", "name": "shell", "arguments": '{"x":1}'}
    assert _payload_text(payload) == "<tool_use name='shell'>{\"x\":1}</tool_use>"


def test_payload_text_function_call_defaults():
    assert _payload_text({"type": "function_call"}) == "<tool_use name=''></tool_use>"


def test_payload_text_function_call_output():
    payload = {"type": "function_call_output", "output": "result"}
    assert _payload_text(payload) == "<tool_result>result</tool_result>"


def test_payload_text_function_call_output_default():
    assert _payload_text({"type": "function_call_output"}) == "<tool_result></tool_result>"


def test_payload_text_custom_tool_call():
    payload = {"type": "custom_tool_call", "name": "custom", "input": "abc"}
    assert _payload_text(payload) == "<tool_use name='custom'>abc</tool_use>"


def test_payload_text_custom_tool_call_output():
    payload = {"type": "custom_tool_call_output", "output": "out"}
    assert _payload_text(payload) == "<tool_result>out</tool_result>"


def test_payload_text_unknown_type():
    assert _payload_text({"type": "nonsense"}) == ""
    assert _payload_text({}) == ""


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------


def test_record_role_not_response_item():
    assert _record_role({"type": "event_msg", "payload": {"type": "message"}}) is None


def test_record_role_user():
    assert _record_role(_message("user")) == "user"


def test_record_role_assistant():
    assert _record_role(_message("assistant")) == "assistant"


def test_record_role_developer_skipped():
    assert _record_role(_message("developer")) is None


def test_record_role_unknown_message_role():
    assert _record_role(_message("system")) is None


def test_record_role_tool_types():
    for ptype, expected in [
        ("reasoning", "assistant"),
        ("function_call", "assistant"),
        ("custom_tool_call", "assistant"),
        ("function_call_output", "user"),
        ("custom_tool_call_output", "user"),
    ]:
        assert _record_role(_response_item({"type": ptype})) == expected


def test_record_role_unknown_payload_type():
    assert _record_role(_response_item({"type": "other"})) is None


def test_record_role_null_payload():
    assert _record_role({"type": "response_item", "payload": None}) is None


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------


def test_records_to_messages_basic_alternation():
    records = [
        _message("user", _text_block("hi")),
        _message("assistant", _text_block("hello", "output_text")),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_records_to_messages_merges_same_role():
    records = [
        _message("user", _text_block("a")),
        _message("user", _text_block("b")),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "a\nb"}]


def test_records_to_messages_skips_empty_and_non_message():
    records = [
        {"type": "event_msg", "payload": {"type": "message", "role": "user"}},
        _message("user"),
        _response_item({"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "  "}
        ]}),
        _message("user", _text_block("real")),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "real"}]


def test_records_to_messages_skips_developer():
    records = [
        _message("developer", _text_block("instructions")),
        _message("user", _text_block("hi")),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "hi"}]


def test_records_to_messages_tool_flattening():
    records = [
        _message("user", _text_block("do it")),
        _response_item({"type": "reasoning", "summary": [{"text": "think"}]}),
        _response_item({"type": "function_call", "name": "shell", "arguments": "ls"}),
        _response_item({"type": "function_call_output", "output": "files"}),
        _message("assistant", _text_block("done", "output_text")),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "do it"},
        {"role": "assistant",
         "content": "<thinking>think</thinking>\n<tool_use name='shell'>ls</tool_use>"},
        {"role": "user", "content": "<tool_result>files</tool_result>"},
        {"role": "assistant", "content": "done"},
    ]


def test_records_to_messages_empty_records():
    assert records_to_messages_codex([]) == []


def test_records_to_messages_custom_tool_call():
    records = [
        _response_item({"type": "custom_tool_call", "name": "c", "input": "in"}),
        _response_item({"type": "custom_tool_call_output", "output": "out"}),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "assistant", "content": "<tool_use name='c'>in</tool_use>"},
        {"role": "user", "content": "<tool_result>out</tool_result>"},
    ]


def test_payload_parts_non_message_single_text():
    payload = {"type": "function_call", "name": "f", "arguments": "a"}
    assert _payload_parts(payload) == [
        {"type": "text", "text": "<tool_use name='f'>a</tool_use>"}
    ]


def test_payload_parts_empty_non_message():
    assert _payload_parts({"type": "unknown"}) == []


def test_payload_parts_message_blank_dropped():
    payload = {"type": "message", "content": [
        {"type": "input_text", "text": "   "},
        {"type": "input_text", "text": "ok"},
    ]}
    assert _payload_parts(payload) == [{"type": "text", "text": "ok"}]


def test_payload_parts_message_non_dict_block():
    payload = {"type": "message", "content": [42, {"type": "text", "text": "x"}]}
    assert _payload_parts(payload) == [{"type": "text", "text": "x"}]


def test_payload_parts_image_unreadable(capsys):
    payload = {"type": "message", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,!!!!notbase64"},
        {"type": "text", "text": "keep"},
    ]}
    parts = _payload_parts(payload)
    assert parts == [{"type": "text", "text": "keep"}]
    assert "Skipping unreadable input_image" in capsys.readouterr().err


def test_payload_parts_image_missing_url():
    payload = {"type": "message", "content": [
        {"type": "input_image"},
        {"type": "text", "text": "keep"},
    ]}
    assert _payload_parts(payload) == [{"type": "text", "text": "keep"}]


# ---------------------------------------------------------------------------
# records_to_messages_codex with image parts (requires Pillow)
# ---------------------------------------------------------------------------


def _tiny_png_data_uri() -> str:
    import base64
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (len(data).to_bytes(4, "big") + tag + data
                + zlib.crc32(tag + data).to_bytes(4, "big"))

    import struct

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\xff\x00\x00"  # filter byte + RGB pixel
    idat = zlib.compress(raw)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", idat) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def test_records_to_messages_multimodal_image():
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        pytest.skip("Pillow not installed")

    uri = _tiny_png_data_uri()
    records = [
        _message("user",
                 _text_block("see image"),
                 {"type": "input_image", "image_url": uri}),
        _message("assistant", _text_block("ok", "output_text")),
    ]
    messages = records_to_messages_codex(records)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "see image"}
    assert content[1]["type"] == "image"
    from PIL import Image
    assert isinstance(content[1]["image"], Image.Image)
    assert messages[1] == {"role": "assistant", "content": "ok"}


def test_records_to_messages_merge_across_image():
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        pytest.skip("Pillow not installed")

    uri = _tiny_png_data_uri()
    records = [
        _message("user", _text_block("first")),
        _message("user",
                 _text_block("second"),
                 {"type": "input_image", "image_url": uri}),
    ]
    messages = records_to_messages_codex(records)
    assert len(messages) == 1
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "first\nsecond"}
    assert content[1]["type"] == "image"


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------


def test_session_to_messages_no_output(tmp_path):
    records = [
        _message("user", _text_block("hi")),
        _message("assistant", _text_block("yo", "output_text")),
    ]
    path = _write_jsonl(tmp_path, records)
    messages = session_to_messages_codex(path)
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_session_to_messages_with_output(tmp_path):
    records = [
        _message("user", _text_block("hi")),
        _message("assistant", _text_block("yo", "output_text")),
    ]
    path = _write_jsonl(tmp_path, records)
    out = tmp_path / "nested" / "out.json"
    messages = session_to_messages_codex(path, output=out)
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert out.exists()
    with open(out, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == messages


def test_session_to_messages_output_str_path(tmp_path):
    records = [_message("user", _text_block("hi"))]
    path = _write_jsonl(tmp_path, records)
    out = tmp_path / "out.json"
    session_to_messages_codex(path, output=str(out))
    assert json.loads(out.read_text(encoding="utf-8")) == [
        {"role": "user", "content": "hi"}
    ]


def test_session_to_messages_output_image_serialized(tmp_path):
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        pytest.skip("Pillow not installed")

    uri = _tiny_png_data_uri()
    records = [
        _message("user",
                 _text_block("pic"),
                 {"type": "input_image", "image_url": uri}),
    ]
    path = _write_jsonl(tmp_path, records)
    out = tmp_path / "out.json"
    messages = session_to_messages_codex(path, output=out)

    # In-memory result keeps real PIL image.
    from PIL import Image
    assert isinstance(messages[0]["content"][1]["image"], Image.Image)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved[0]["content"][1]["type"] == "image"
    assert saved[0]["content"][1]["image"].startswith("data:image/")


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------


def test_record_type_of_with_sub():
    assert _record_type_of({"type": "response_item", "payload": {"type": "message"}}) == \
        "response_item:message"


def test_record_type_of_no_sub():
    assert _record_type_of({"type": "event_msg"}) == "event_msg"


def test_record_type_of_missing_type():
    assert _record_type_of({}) == "unknown"


def test_record_type_of_null_payload():
    assert _record_type_of({"type": "x", "payload": None}) == "x"


def test_timestamp_of_truncated():
    rec = {"timestamp": "2024-01-02T03:04:05.123456Z"}
    assert _timestamp_of(rec) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_total_tokens_of_latest_wins():
    records = [
        {"type": "response_item", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 1, "output_tokens": 2,
                                  "cached_input_tokens": 3, "total_tokens": 5}}}},
        {"type": "response_item", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 10, "output_tokens": 20,
                                  "cached_input_tokens": 30, "total_tokens": 50}}}},
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cached_input_tokens": 30,
        "total_tokens": 50,
    }


def test_total_tokens_of_none_info():
    records = [
        {"type": "response_item", "payload": {"type": "token_count", "info": None}},
    ]
    assert _total_tokens_of(records) == {}


def test_total_tokens_of_null_usage():
    records = [
        {"type": "response_item", "payload": {"type": "token_count",
                                               "info": {"total_token_usage": None}}},
    ]
    assert _total_tokens_of(records) == {}


def test_total_tokens_of_no_events():
    assert _total_tokens_of([{"type": "response_item", "payload": {}}]) == {}


def test_total_tokens_of_missing_keys():
    records = [
        {"type": "response_item", "payload": {"type": "token_count",
                                               "info": {"total_token_usage": {}}}},
    ]
    assert _total_tokens_of(records) == {}


def test_total_tokens_of_partial_keys():
    records = [
        {"type": "response_item", "payload": {"type": "token_count",
                                               "info": {"total_token_usage": {
                                                   "input_tokens": 7}}}},
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 7,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "total_tokens": 0,
    }


# ---------------------------------------------------------------------------
# CODEX_SCHEMA + inspect_session_codex
# ---------------------------------------------------------------------------


def test_codex_schema_fields():
    assert CODEX_SCHEMA.label == "CODEX"
    user = _message("user", _text_block("hi"))
    assert CODEX_SCHEMA.is_message(user) is True
    assert CODEX_SCHEMA.role_of(user) == "user"
    assert CODEX_SCHEMA.text_of(user) == "hi"

    other = {"type": "event_msg", "payload": {"type": "x"}}
    assert CODEX_SCHEMA.is_message(other) is False
    assert CODEX_SCHEMA.role_of(other) == "other"
    assert CODEX_SCHEMA.text_of(other) == ""


def test_inspect_session_codex_runs(tmp_path, capsys):
    records = [
        {"type": "event_msg", "payload": {"type": "task_started"},
         "timestamp": "2024-01-01T00:00:00Z"},
        _message("user", _text_block("hello")),
        _message("assistant", _text_block("world", "output_text")),
        {"type": "response_item", "payload": {"type": "token_count",
                                              "info": {"total_token_usage": {
                                                  "input_tokens": 3,
                                                  "output_tokens": 4,
                                                  "total_tokens": 7}}}},
    ]
    path = _write_jsonl(tmp_path, records)
    inspect_session_codex(path)
    out = capsys.readouterr().out
    assert "CODEX" in out


def test_inspect_session_codex_show_flow(tmp_path, capsys):
    records = [_message("user", _text_block("hello"))]
    path = _write_jsonl(tmp_path, records)
    inspect_session_codex(path, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "CODEX" in out
