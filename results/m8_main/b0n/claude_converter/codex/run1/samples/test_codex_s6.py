from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest

from claude_converter import codex
from claude_converter.codex import (
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

def _rec(rtype: str, payload: dict, timestamp: str = "2024-01-01T00:00:00Z") -> dict:
    return {"timestamp": timestamp, "type": rtype, "payload": payload}


def _msg(role: str, *blocks: dict) -> dict:
    return _rec("response_item", {"type": "message", "role": role, "content": list(blocks)})


def _text_block(text: str) -> dict:
    return {"type": "input_text", "text": text}


def _png_b64() -> str:
    # 1x1 transparent PNG
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


pil = pytest.importorskip("PIL.Image")


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------

class TestLoadSessionCodex:
    def test_loads_records(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [_rec("event_msg", {"type": "x"})])
        records = load_session_codex(p)
        assert records == [_rec("event_msg", {"type": "x"})]

    def test_loads_json_extension(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.json", [_rec("event_msg", {"type": "x"})])
        assert len(load_session_codex(p)) == 1

    def test_accepts_str_path(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [_rec("event_msg", {})])
        assert len(load_session_codex(str(p))) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session_codex(tmp_path / "nope.jsonl")

    def test_bad_extension_raises(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text("{}")
        with pytest.raises(ValueError):
            load_session_codex(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("\n\n")
        with pytest.raises(ValueError):
            load_session_codex(p)


# ---------------------------------------------------------------------------
# _payload_text
# ---------------------------------------------------------------------------

class TestPayloadText:
    def test_message_blocks(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "output_text", "text": "world"},
                {"type": "text", "text": "!"},
            ],
        }
        assert _payload_text(payload) == "hello\nworld\n!"

    def test_message_skips_blank_and_bad_blocks(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "  "},
                "not-a-dict",
                {"type": "output_text", "text": "ok"},
            ],
        }
        assert _payload_text(payload) == "ok"

    def test_message_image_marker(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "before"},
                {"type": "input_image"},
                {"type": "input_text", "text": "after"},
            ],
        }
        assert _payload_text(payload) == "before\n<image>\nafter"

    def test_message_null_content(self):
        assert _payload_text({"type": "message", "content": None}) == ""

    def test_reasoning_wrapped(self):
        payload = {
            "type": "reasoning",
            "summary": [{"text": "thinking..."}, {"text": "more"}],
        }
        assert _payload_text(payload) == "<thinking>thinking...\nmore</thinking>"

    def test_reasoning_empty(self):
        assert _payload_text({"type": "reasoning", "summary": []}) == ""

    def test_function_call(self):
        payload = {"type": "function_call", "name": "foo", "arguments": '{"a":1}'}
        assert _payload_text(payload) == "<tool_use name='foo'>{\"a\":1}</tool_use>"

    def test_function_call_output(self):
        payload = {"type": "function_call_output", "output": "res"}
        assert _payload_text(payload) == "<tool_result>res</tool_result>"

    def test_custom_tool_call(self):
        payload = {"type": "custom_tool_call", "name": "bar", "input": "inp"}
        assert _payload_text(payload) == "<tool_use name='bar'>inp</tool_use>"

    def test_custom_tool_call_output(self):
        payload = {"type": "custom_tool_call_output", "output": "out"}
        assert _payload_text(payload) == "<tool_result>out</tool_result>"

    def test_unknown_type(self):
        assert _payload_text({"type": "unknown"}) == ""
        assert _payload_text({}) == ""


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------

class TestRecordRole:
    def test_non_response_item(self):
        assert _record_role({"type": "event_msg", "payload": {}}) is None

    def test_message_user(self):
        assert _record_role(_msg("user", _text_block("hi"))) == "user"

    def test_message_assistant(self):
        assert _record_role(_msg("assistant", _text_block("hi"))) == "assistant"

    def test_message_developer_skipped(self):
        assert _record_role(_msg("developer", _text_block("x"))) is None

    def test_message_unknown_role(self):
        assert _record_role(_msg("system", _text_block("x"))) is None

    def test_null_payload(self):
        assert _record_role({"type": "response_item", "payload": None}) is None

    @pytest.mark.parametrize("ptype,role", [
        ("reasoning", "assistant"),
        ("function_call", "assistant"),
        ("custom_tool_call", "assistant"),
        ("function_call_output", "user"),
        ("custom_tool_call_output", "user"),
    ])
    def test_tool_roles(self, ptype, role):
        assert _record_role(_rec("response_item", {"type": ptype})) == role

    def test_unknown_payload_type(self):
        assert _record_role(_rec("response_item", {"type": "weird"})) is None


# ---------------------------------------------------------------------------
# _payload_parts
# ---------------------------------------------------------------------------

class TestPayloadParts:
    def test_text_parts(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "output_text", "text": "  "},
            ],
        }
        assert _payload_parts(payload) == [{"type": "text", "text": "hello"}]

    def test_image_part(self):
        payload = {
            "type": "message",
            "content": [{"type": "input_image", "image_url": _png_b64()}],
        }
        parts = _payload_parts(payload)
        assert len(parts) == 1
        assert parts[0]["type"] == "image"
        assert isinstance(parts[0]["image"], pil.Image)

    def test_image_empty_url_skipped(self):
        payload = {
            "type": "message",
            "content": [{"type": "input_image", "image_url": ""}],
        }
        assert _payload_parts(payload) == []

    def test_image_bad_data_skipped(self, capsys):
        payload = {
            "type": "message",
            "content": [{"type": "input_image", "image_url": "not-an-image"}],
        }
        assert _payload_parts(payload) == []
        err = capsys.readouterr().err
        assert "Skipping unreadable input_image" in err

    def test_non_message_payload(self):
        payload = {"type": "function_call", "name": "n", "arguments": "a"}
        assert _payload_parts(payload) == [
            {"type": "text", "text": "<tool_use name='n'>a</tool_use>"}
        ]

    def test_non_message_empty_text(self):
        assert _payload_parts({"type": "reasoning", "summary": []}) == []


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------

class TestRecordsToMessages:
    def test_simple_conversation(self):
        records = [
            _msg("user", _text_block("hi")),
            _msg("assistant", {"type": "output_text", "text": "hello"}),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_merges_consecutive_same_role(self):
        records = [
            _msg("user", _text_block("a")),
            _msg("user", _text_block("b")),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "a\nb"},
        ]

    def test_skips_non_conversational(self):
        records = [
            {"type": "event_msg", "payload": {"type": "token_count"}},
            _msg("user", _text_block("hi")),
            _msg("developer", _text_block("ignored")),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "hi"},
        ]

    def test_skips_empty_parts(self):
        records = [
            _msg("user", _text_block("   ")),
            _msg("assistant", _text_block("real")),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "assistant", "content": "real"},
        ]

    def test_tool_calls_flattened(self):
        records = [
            _msg("user", _text_block("go")),
            _rec("response_item", {"type": "reasoning", "summary": [{"text": "think"}]}),
            _rec("response_item", {"type": "function_call", "name": "f", "arguments": "{}"}),
            _rec("response_item", {"type": "function_call_output", "output": "done"}),
        ]
        result = records_to_messages_codex(records)
        assert result == [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "<thinking>think</thinking>\n<tool_use name='f'>{}</tool_use>",
            },
            {"role": "user", "content": "<tool_result>done</tool_result>"},
        ]

    def test_images_produce_list_content(self):
        records = [
            _msg(
                "user",
                _text_block("look"),
                {"type": "input_image", "image_url": _png_b64()},
            ),
        ]
        result = records_to_messages_codex(records)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "look"}
        assert content[1]["type"] == "image"
        assert isinstance(content[1]["image"], pil.Image)

    def test_text_only_merges_into_string(self):
        records = [
            _msg("user", _text_block("a")),
            _msg("user", _text_block("b")),
        ]
        result = records_to_messages_codex(records)
        assert result[0]["content"] == "a\nb"

    def test_empty_records(self):
        assert records_to_messages_codex([]) == []


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------

class TestSessionToMessages:
    def test_converts_file(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [
            _msg("user", _text_block("hi")),
            _msg("assistant", _text_block("yo")),
        ])
        result = session_to_messages_codex(p)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]

    def test_output_writes_json(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [
            _msg("user", _text_block("hi")),
        ])
        out = tmp_path / "sub" / "out.json"
        session_to_messages_codex(p, output=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data == [{"role": "user", "content": "hi"}]

    def test_output_encodes_images(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [
            _msg(
                "user",
                _text_block("pic"),
                {"type": "input_image", "image_url": _png_b64()},
            ),
        ])
        out = tmp_path / "out.json"
        session_to_messages_codex(p, output=out)
        data = json.loads(out.read_text())
        content = data[0]["content"]
        assert content[0] == {"type": "text", "text": "pic"}
        assert content[1]["type"] == "image"
        assert content[1]["image"].startswith("data:image/png;base64,")

    def test_output_none_keeps_pil_objects(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [
            _msg("user", {"type": "input_image", "image_url": _png_b64()}),
        ])
        result = session_to_messages_codex(p)
        assert isinstance(result[0]["content"][0]["image"], pil.Image)


# ---------------------------------------------------------------------------
# Inspection helpers & schema
# ---------------------------------------------------------------------------

class TestInspectionHelpers:
    def test_record_type_of_with_sub(self):
        assert _record_type_of(_rec("response_item", {"type": "message"})) == "response_item:message"

    def test_record_type_of_without_sub(self):
        assert _record_type_of({"type": "event_msg", "payload": {}}) == "event_msg"

    def test_record_type_of_unknown(self):
        assert _record_type_of({}) == "unknown"

    def test_record_type_of_null_payload(self):
        assert _record_type_of({"type": "response_item", "payload": None}) == "response_item"

    def test_timestamp_of(self):
        assert _timestamp_of({"timestamp": "2024-01-02T03:04:05.678Z"}) == "2024-01-02 03:04:05"

    def test_timestamp_of_missing(self):
        assert _timestamp_of({}) == ""

    def test_total_tokens_latest_snapshot(self):
        records = [
            _rec("event_msg", {"type": "token_count", "info": {
                "total_token_usage": {"input_tokens": 1, "output_tokens": 2,
                                      "cached_input_tokens": 0, "total_tokens": 3}}}),
            _rec("event_msg", {"type": "token_count", "info": {
                "total_token_usage": {"input_tokens": 10, "output_tokens": 20,
                                      "cached_input_tokens": 5, "total_tokens": 35}}}),
        ]
        assert _total_tokens_of(records) == {
            "input_tokens": 10,
            "output_tokens": 20,
            "cached_input_tokens": 5,
            "total_tokens": 35,
        }

    def test_total_tokens_null_info(self):
        records = [_rec("event_msg", {"type": "token_count", "info": None})]
        assert _total_tokens_of(records) == {}

    def test_total_tokens_missing_usage(self):
        records = [_rec("event_msg", {"type": "token_count", "info": {}})]
        assert _total_tokens_of(records) == {}

    def test_total_tokens_no_token_records(self):
        assert _total_tokens_of([_msg("user", _text_block("hi"))]) == {}

    def test_total_tokens_defaults(self):
        records = [_rec("event_msg", {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 7}}})]
        assert _total_tokens_of(records) == {
            "input_tokens": 7,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "total_tokens": 0,
        }


class TestSchema:
    def test_schema_label(self):
        assert codex.CODEX_SCHEMA.label == "CODEX"

    def test_schema_role_of(self):
        rec = _msg("user", _text_block("hi"))
        assert codex.CODEX_SCHEMA.role_of(rec) == "user"

    def test_schema_role_of_other(self):
        rec = {"type": "event_msg", "payload": {}}
        assert codex.CODEX_SCHEMA.role_of(rec) == "other"

    def test_schema_is_message(self):
        assert codex.CODEX_SCHEMA.is_message(_msg("user", _text_block("hi"))) is True
        assert codex.CODEX_SCHEMA.is_message({"type": "event_msg", "payload": {}}) is False

    def test_schema_text_of(self):
        rec = _msg("user", _text_block("hello"))
        assert codex.CODEX_SCHEMA.text_of(rec) == "hello"


# ---------------------------------------------------------------------------
# inspect_session_codex
# ---------------------------------------------------------------------------

class TestInspectSessionCodex:
    def test_smoke(self, tmp_path, capsys):
        p = _write_jsonl(tmp_path / "s.jsonl", [
            _msg("user", _text_block("hi")),
            _msg("assistant", _text_block("yo")),
            _rec("event_msg", {"type": "token_count", "info": {
                "total_token_usage": {"input_tokens": 1, "output_tokens": 2,
                                      "cached_input_tokens": 0, "total_tokens": 3}}}),
        ])
        inspect_session_codex(p)
        out = capsys.readouterr().out
        assert "CODEX" in out

    def test_with_flow(self, tmp_path, capsys):
        p = _write_jsonl(tmp_path / "s.jsonl", [
            _msg("user", _text_block("hi")),
            _msg("assistant", _text_block("yo")),
        ])
        inspect_session_codex(p, show_flow=True)
        out = capsys.readouterr().out
        assert "hi" in out
        assert "yo" in out
