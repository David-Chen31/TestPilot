import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_converter import codex
from claude_converter.codex import (
    load_session_codex,
    records_to_messages_codex,
    session_to_messages_codex,
    inspect_session_codex,
    _payload_text,
    _record_role,
    _payload_parts,
    _record_type_of,
    _timestamp_of,
    _total_tokens_of,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path, records, name="session.jsonl") -> Path:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _msg(role, content, ts="2024-01-01T00:00:00Z"):
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "message", "role": role, "content": content},
    }


def _input_text(text):
    return {"type": "input_text", "text": text}


def _output_text(text):
    return {"type": "output_text", "text": text}


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------

def test_load_session_codex_reads_jsonl(tmp_path):
    p = _write_jsonl(tmp_path, [
        {"type": "response_item", "payload": {"type": "message"}},
        {"type": "event_msg", "payload": {}},
    ])
    records = load_session_codex(p)
    assert len(records) == 2
    assert records[0]["type"] == "response_item"
    assert records[1]["type"] == "event_msg"


def test_load_session_codex_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_codex(tmp_path / "nope.jsonl")


def test_load_session_codex_bad_extension(tmp_path):
    p = tmp_path / "session.txt"
    p.write_text("{}\n")
    with pytest.raises(ValueError):
        load_session_codex(p)


# ---------------------------------------------------------------------------
# _payload_text
# ---------------------------------------------------------------------------

def test_payload_text_message_text_blocks():
    payload = {
        "type": "message",
        "content": [
            _input_text("hello"),
            _output_text("world"),
            {"type": "text", "text": "extra"},
        ],
    }
    assert _payload_text(payload) == "hello\nworld\nextra"


def test_payload_text_message_skips_whitespace_and_adds_image_placeholder():
    payload = {
        "type": "message",
        "content": [
            _input_text("   "),
            {"type": "input_image"},
            _input_text("hi"),
        ],
    }
    assert _payload_text(payload) == "<image>\nhi"


def test_payload_text_message_non_dict_blocks_skipped():
    payload = {"type": "message", "content": ["junk", _input_text("ok")]}
    assert _payload_text(payload) == "ok"


def test_payload_text_reasoning():
    payload = {"type": "reasoning", "summary": [{"text": "think"}, {"text": "more"}]}
    assert _payload_text(payload) == "<thinking>think\nmore</thinking>"


def test_payload_text_reasoning_empty():
    assert _payload_text({"type": "reasoning", "summary": []}) == ""


def test_payload_text_function_call():
    payload = {"type": "function_call", "name": "foo", "arguments": '{"a":1}'}
    assert _payload_text(payload) == "<tool_use name='foo'>{\"a\":1}</tool_use>"


def test_payload_text_function_call_output():
    payload = {"type": "function_call_output", "output": "result"}
    assert _payload_text(payload) == "<tool_result>result</tool_result>"


def test_payload_text_custom_tool_call():
    payload = {"type": "custom_tool_call", "name": "bar", "input": "inp"}
    assert _payload_text(payload) == "<tool_use name='bar'>inp</tool_use>"


def test_payload_text_custom_tool_call_output():
    payload = {"type": "custom_tool_call_output", "output": "out"}
    assert _payload_text(payload) == "<tool_result>out</tool_result>"


def test_payload_text_unknown_type():
    assert _payload_text({"type": "weird"}) == ""


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------

def test_record_role_non_response_item():
    assert _record_role({"type": "event_msg", "payload": {"type": "message"}}) is None


def test_record_role_user_and_assistant():
    assert _record_role(_msg("user", [])) == "user"
    assert _record_role(_msg("assistant", [])) == "assistant"


def test_record_role_developer_skipped():
    assert _record_role(_msg("developer", [])) is None


def test_record_role_unknown_message_role():
    assert _record_role(_msg("tool", [])) is None


def test_record_role_tool_types():
    assert _record_role({"type": "response_item", "payload": {"type": "reasoning"}}) == "assistant"
    assert _record_role({"type": "response_item", "payload": {"type": "function_call"}}) == "assistant"
    assert _record_role({"type": "response_item", "payload": {"type": "custom_tool_call"}}) == "assistant"
    assert _record_role({"type": "response_item", "payload": {"type": "function_call_output"}}) == "user"
    assert _record_role({"type": "response_item", "payload": {"type": "custom_tool_call_output"}}) == "user"


def test_record_role_unknown_ptype():
    assert _record_role({"type": "response_item", "payload": {"type": "unknown"}}) is None


def test_record_role_handles_null_payload():
    assert _record_role({"type": "response_item", "payload": None}) is None


# ---------------------------------------------------------------------------
# _payload_parts
# ---------------------------------------------------------------------------

def test_payload_parts_message_text_only():
    payload = {"type": "message", "content": [_input_text("hi"), _input_text("  ")]}
    assert _payload_parts(payload) == [{"type": "text", "text": "hi"}]


def test_payload_parts_message_with_image():
    fake_image = object()
    payload = {
        "type": "message",
        "content": [
            _input_text("look"),
            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
        ],
    }
    with patch.object(codex, "convert_base64_to_pil_image", return_value=fake_image) as m:
        parts = _payload_parts(payload)
    m.assert_called_once_with("data:image/png;base64,abc")
    assert parts == [
        {"type": "text", "text": "look"},
        {"type": "image", "image": fake_image},
    ]


def test_payload_parts_image_missing_url_skipped():
    payload = {"type": "message", "content": [{"type": "input_image", "image_url": ""}]}
    assert _payload_parts(payload) == []


def test_payload_parts_image_decode_error_skipped(capsys):
    payload = {
        "type": "message",
        "content": [{"type": "input_image", "image_url": "bad"}],
    }
    with patch.object(codex, "convert_base64_to_pil_image", side_effect=ValueError("nope")):
        parts = _payload_parts(payload)
    assert parts == []
    err = capsys.readouterr().err
    assert "Skipping unreadable input_image" in err
    assert "nope" in err


def test_payload_parts_image_import_error_skipped(capsys):
    payload = {
        "type": "message",
        "content": [{"type": "input_image", "image_url": "bad"}],
    }
    with patch.object(codex, "convert_base64_to_pil_image", side_effect=ImportError("no pil")):
        parts = _payload_parts(payload)
    assert parts == []
    assert "Skipping unreadable input_image" in capsys.readouterr().err


def test_payload_parts_non_message_uses_text():
    payload = {"type": "function_call", "name": "f", "arguments": "a"}
    assert _payload_parts(payload) == [
        {"type": "text", "text": "<tool_use name='f'>a</tool_use>"}
    ]


def test_payload_parts_non_message_empty_returns_empty():
    assert _payload_parts({"type": "unknown"}) == []


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------

def test_records_to_messages_basic():
    records = [
        _msg("user", [_input_text("hello")]),
        _msg("assistant", [_output_text("hi")]),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_records_to_messages_merges_consecutive_same_role():
    records = [
        _msg("user", [_input_text("a")]),
        _msg("user", [_input_text("b")]),
        _msg("assistant", [_output_text("c")]),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "a\nb"},
        {"role": "assistant", "content": "c"},
    ]


def test_records_to_messages_skips_non_message_and_empty():
    records = [
        {"type": "event_msg", "payload": {"type": "x"}},
        _msg("user", [_input_text("   ")]),  # blank -> no parts
        _msg("user", [_input_text("real")]),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "real"},
    ]


def test_records_to_messages_tool_calls_flattened():
    records = [
        _msg("user", [_input_text("run")]),
        {
            "type": "response_item",
            "payload": {"type": "reasoning", "summary": [{"text": "thinking"}]},
        },
        {
            "type": "response_item",
            "payload": {"type": "function_call", "name": "f", "arguments": "{}"},
        },
        {
            "type": "response_item",
            "payload": {"type": "function_call_output", "output": "done"},
        },
    ]
    messages = records_to_messages_codex(records)
    assert messages == [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": "<thinking>thinking</thinking>\n<tool_use name='f'>{}</tool_use>",
        },
        {"role": "user", "content": "<tool_result>done</tool_result>"},
    ]


def test_records_to_messages_with_image_produces_list_content():
    fake_image = object()
    records = [
        _msg("user", [
            _input_text("look"),
            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
        ]),
    ]
    with patch.object(codex, "convert_base64_to_pil_image", return_value=fake_image):
        messages = records_to_messages_codex(records)
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "image": fake_image},
            ],
        }
    ]


def test_records_to_messages_empty_records():
    assert records_to_messages_codex([]) == []


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------

def test_session_to_messages_no_output(tmp_path):
    p = _write_jsonl(tmp_path, [
        _msg("user", [_input_text("hello")]),
        _msg("assistant", [_output_text("hi")]),
    ])
    messages = session_to_messages_codex(p)
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_session_to_messages_writes_output(tmp_path):
    p = _write_jsonl(tmp_path, [
        _msg("user", [_input_text("hello")]),
    ])
    out = tmp_path / "sub" / "out.json"
    messages = session_to_messages_codex(p, output=out)

    assert messages == [{"role": "user", "content": "hello"}]
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == [{"role": "user", "content": "hello"}]


def test_session_to_messages_output_serializes_images(tmp_path):
    fake_image = object()
    p = _write_jsonl(tmp_path, [
        _msg("user", [
            _input_text("look"),
            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
        ]),
    ])
    out = tmp_path / "out.json"
    with patch.object(codex, "convert_base64_to_pil_image", return_value=fake_image), \
            patch.object(codex, "messages_to_json_safe",
                         side_effect=lambda msgs: [{"role": m["role"], "content": "SAFE"} for m in msgs]) as mjs:
        messages = session_to_messages_codex(p, output=out)
    # in-memory keeps real image
    assert messages[0]["content"][1]["image"] is fake_image
    # saved file used the safe conversion
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == [{"role": "user", "content": "SAFE"}]
    mjs.assert_called_once()


# ---------------------------------------------------------------------------
# _record_type_of / _timestamp_of / _total_tokens_of
# ---------------------------------------------------------------------------

def test_record_type_of_with_subtype():
    r = {"type": "response_item", "payload": {"type": "message"}}
    assert _record_type_of(r) == "response_item:message"


def test_record_type_of_without_subtype():
    assert _record_type_of({"type": "event_msg"}) == "event_msg"


def test_record_type_of_unknown():
    assert _record_type_of({}) == "unknown"


def test_timestamp_of_formats():
    r = {"timestamp": "2024-01-02T03:04:05.123456+00:00"}
    assert _timestamp_of(r) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_total_tokens_of_latest_snapshot():
    records = [
        {"type": "response_item", "payload": {"type": "token_count", "info": {
            "total_token_usage": {
                "input_tokens": 1, "output_tokens": 2,
                "cached_input_tokens": 0, "total_tokens": 3,
            }
        }}},
        {"type": "response_item", "payload": {"type": "token_count", "info": {
            "total_token_usage": {
                "input_tokens": 10, "output_tokens": 20,
                "cached_input_tokens": 5, "total_tokens": 30,
            }
        }}},
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cached_input_tokens": 5,
        "total_tokens": 30,
    }


def test_total_tokens_of_null_info():
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


def test_total_tokens_of_no_token_count():
    assert _total_tokens_of([{"type": "response_item", "payload": {"type": "message"}}]) == {}


# ---------------------------------------------------------------------------
# inspect_session_codex
# ---------------------------------------------------------------------------

def test_inspect_session_codex(tmp_path):
    p = _write_jsonl(tmp_path, [
        _msg("user", [_input_text("hello")]),
        _msg("assistant", [_output_text("hi")]),
    ])
    with patch.object(codex, "run_inspection") as ri:
        inspect_session_codex(p, show_flow=True, show_blocks=True, show_raw=True)
    ri.assert_called_once()
    args, kwargs = ri.call_args
    assert args[0] == p
    assert args[2] is codex.CODEX_SCHEMA
    assert kwargs == {"show_flow": True, "show_blocks": True}


def test_inspect_session_codex_defaults(tmp_path):
    p = _write_jsonl(tmp_path, [_msg("user", [_input_text("hello")])])
    with patch.object(codex, "run_inspection") as ri:
        inspect_session_codex(p)
    _, kwargs = ri.call_args
    assert kwargs == {"show_flow": False, "show_blocks": False}
