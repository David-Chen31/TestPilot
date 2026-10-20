from __future__ import annotations

import base64
import io
import json

import pytest

from claude_converter import codex
from claude_converter.codex import (
    load_session_codex,
    records_to_messages_codex,
    session_to_messages_codex,
    inspect_session_codex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _msg(role, *blocks):
    return {"type": "response_item", "payload": {"type": "message", "role": role, "content": list(blocks)}}


def _text(text):
    return {"type": "input_text", "text": text}


def _out(text):
    return {"type": "output_text", "text": text}


def _tiny_png_data_uri():
    """A 1x1 transparent PNG as a base64 data URI."""
    # Hardcoded 1x1 PNG bytes
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------

class TestLoadSessionCodex:
    def test_loads_records(self, tmp_path):
        p = tmp_path / "s.jsonl"
        _write_jsonl(p, [{"a": 1}, {"b": 2}])
        assert load_session_codex(p) == [{"a": 1}, {"b": 2}]

    def test_accepts_str_path(self, tmp_path):
        p = tmp_path / "s.jsonl"
        _write_jsonl(p, [{"a": 1}])
        assert load_session_codex(str(p)) == [{"a": 1}]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session_codex(tmp_path / "nope.jsonl")

    def test_bad_extension(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session_codex(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session_codex(p)


# ---------------------------------------------------------------------------
# _payload_text
# ---------------------------------------------------------------------------

class TestPayloadText:
    def test_message_input_output_and_plain_text(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "output_text", "text": "world"},
                {"type": "text", "text": "plain"},
            ],
        }
        assert codex._payload_text(payload) == "hello\nworld\nplain"

    def test_message_with_image_placeholder(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:..."},
            ],
        }
        assert codex._payload_text(payload) == "look\n<image>"

    def test_message_skips_non_dict_blocks_and_blank_text(self):
        payload = {
            "type": "message",
            "content": [
                "not a dict",
                {"type": "input_text", "text": "  "},
                {"type": "input_text", "text": "keep"},
                {"type": "unknown", "text": "ignored"},
            ],
        }
        assert codex._payload_text(payload) == "keep"

    def test_message_null_content(self):
        assert codex._payload_text({"type": "message", "content": None}) == ""

    def test_reasoning(self):
        payload = {"type": "reasoning", "summary": [{"text": "thinking"}, {"text": "more"}]}
        assert codex._payload_text(payload) == "<thinking>thinking\nmore</thinking>"

    def test_reasoning_empty(self):
        assert codex._payload_text({"type": "reasoning", "summary": []}) == ""

    def test_reasoning_skips_non_dict(self):
        payload = {"type": "reasoning", "summary": ["x", {"text": "y"}]}
        assert codex._payload_text(payload) == "<thinking>y</thinking>"

    def test_function_call(self):
        payload = {"type": "function_call", "name": "shell", "arguments": "{}"}
        assert codex._payload_text(payload) == "<tool_use name='shell'>{}</tool_use>"

    def test_function_call_defaults(self):
        assert codex._payload_text({"type": "function_call"}) == "<tool_use name=''></tool_use>"

    def test_function_call_output(self):
        payload = {"type": "function_call_output", "output": "result"}
        assert codex._payload_text(payload) == "<tool_result>result</tool_result>"

    def test_custom_tool_call(self):
        payload = {"type": "custom_tool_call", "name": "editor", "input": "x"}
        assert codex._payload_text(payload) == "<tool_use name='editor'>x</tool_use>"

    def test_custom_tool_call_output(self):
        payload = {"type": "custom_tool_call_output", "output": "ok"}
        assert codex._payload_text(payload) == "<tool_result>ok</tool_result>"

    def test_unknown_type(self):
        assert codex._payload_text({"type": "weird"}) == ""


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------

class TestRecordRole:
    def test_non_response_item(self):
        assert codex._record_role({"type": "event_msg", "payload": {}}) is None

    def test_user_message(self):
        assert codex._record_role(_msg("user", _text("hi"))) == "user"

    def test_assistant_message(self):
        assert codex._record_role(_msg("assistant", _out("hi"))) == "assistant"

    def test_developer_skipped(self):
        assert codex._record_role(_msg("developer", _text("sys"))) is None

    def test_unknown_role(self):
        assert codex._record_role(_msg("tool", _text("x"))) is None

    @pytest.mark.parametrize("ptype,role", [
        ("reasoning", "assistant"),
        ("function_call", "assistant"),
        ("custom_tool_call", "assistant"),
        ("function_call_output", "user"),
        ("custom_tool_call_output", "user"),
    ])
    def test_tool_roles(self, ptype, role):
        rec = {"type": "response_item", "payload": {"type": ptype}}
        assert codex._record_role(rec) == role

    def test_unknown_payload_type(self):
        rec = {"type": "response_item", "payload": {"type": "mystery"}}
        assert codex._record_role(rec) is None

    def test_payload_null(self):
        rec = {"type": "response_item", "payload": None}
        assert codex._record_role(rec) is None


# ---------------------------------------------------------------------------
# _payload_parts
# ---------------------------------------------------------------------------

class TestPayloadParts:
    def test_message_text_parts(self):
        payload = {"type": "message", "content": [{"type": "input_text", "text": "a"},
                                                  {"type": "output_text", "text": "b"}]}
        assert codex._payload_parts(payload) == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_message_skips_blank_text(self):
        payload = {"type": "message", "content": [{"type": "input_text", "text": "  "},
                                                  {"type": "input_text", "text": "ok"}]}
        assert codex._payload_parts(payload) == [{"type": "text", "text": "ok"}]

    def test_message_image(self):
        uri = _tiny_png_data_uri()
        payload = {"type": "message",
                   "content": [{"type": "input_image", "image_url": uri}]}
        parts = codex._payload_parts(payload)
        assert len(parts) == 1
        assert parts[0]["type"] == "image"
        assert parts[0]["image"].size == (1, 1)

    def test_message_image_missing_url_skipped(self):
        payload = {"type": "message", "content": [{"type": "input_image"}]}
        assert codex._payload_parts(payload) == []

    def test_message_image_bad_data_skipped(self, capsys):
        payload = {"type": "message",
                   "content": [{"type": "input_image", "image_url": "data:image/png;base64,@@@@bad"}]}
        parts = codex._payload_parts(payload)
        assert parts == []
        err = capsys.readouterr().err
        assert "Skipping unreadable input_image" in err

    def test_message_non_dict_block_skipped(self):
        payload = {"type": "message", "content": ["str", {"type": "input_text", "text": "x"}]}
        assert codex._payload_parts(payload) == [{"type": "text", "text": "x"}]

    def test_tool_call_flattens(self):
        payload = {"type": "function_call", "name": "n", "arguments": "a"}
        assert codex._payload_parts(payload) == [{"type": "text", "text": "<tool_use name='n'>a</tool_use>"}]

    def test_empty_text_returns_empty(self):
        assert codex._payload_parts({"type": "reasoning", "summary": []}) == []


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------

class TestRecordsToMessages:
    def test_basic_user_assistant(self):
        records = [_msg("user", _text("hi")), _msg("assistant", _out("hello"))]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_merges_consecutive_same_role(self):
        records = [_msg("user", _text("a")), _msg("user", _text("b"))]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "a\nb"},
        ]

    def test_skips_non_message_records(self):
        records = [
            {"type": "event_msg", "payload": {"type": "x"}},
            _msg("user", _text("hi")),
            {"type": "response_item", "payload": {"type": "development"}},
        ]
        assert records_to_messages_codex(records) == [{"role": "user", "content": "hi"}]

    def test_skips_empty_parts(self):
        records = [
            _msg("user", _text("   ")),
            _msg("assistant", _out("real")),
        ]
        assert records_to_messages_codex(records) == [{"role": "assistant", "content": "real"}]

    def test_developer_skipped(self):
        records = [
            _msg("developer", _text("system")),
            _msg("user", _text("hi")),
        ]
        assert records_to_messages_codex(records) == [{"role": "user", "content": "hi"}]

    def test_tool_call_goes_to_assistant(self):
        records = [
            _msg("user", _text("do it")),
            {"type": "response_item",
             "payload": {"type": "function_call", "name": "shell", "arguments": "ls"}},
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "<tool_use name='shell'>ls</tool_use>"},
        ]

    def test_tool_output_merges_as_user(self):
        records = [
            _msg("assistant", _out("calling")),
            {"type": "response_item",
             "payload": {"type": "function_call_output", "output": "done"}},
            _msg("user", _text("more")),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "assistant", "content": "calling"},
            {"role": "user", "content": "<tool_result>done</tool_result>\nmore"},
        ]

    def test_reasoning_merges_with_assistant(self):
        records = [
            _msg("user", _text("q")),
            {"type": "response_item",
             "payload": {"type": "reasoning", "summary": [{"text": "think"}]}},
            _msg("assistant", _out("a")),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<thinking>think</thinking>\na"},
        ]

    def test_image_produces_list_content(self):
        uri = _tiny_png_data_uri()
        records = [_msg("user", _text("look"), {"type": "input_image", "image_url": uri})]
        result = records_to_messages_codex(records)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "look"}
        assert content[1]["type"] == "image"
        assert content[1]["image"].size == (1, 1)

    def test_empty_records(self):
        assert records_to_messages_codex([]) == []

    def test_custom_tool_call(self):
        records = [
            {"type": "response_item",
             "payload": {"type": "custom_tool_call", "name": "ed", "input": "code"}},
        ]
        assert records_to_messages_codex(records) == [
            {"role": "assistant", "content": "<tool_use name='ed'>code</tool_use>"},
        ]

    def test_custom_tool_output(self):
        records = [
            {"type": "response_item",
             "payload": {"type": "custom_tool_call_output", "output": "res"}},
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "<tool_result>res</tool_result>"},
        ]


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------

class TestSessionToMessagesCodex:
    def test_basic(self, tmp_path):
        p = tmp_path / "s.jsonl"
        _write_jsonl(p, [_msg("user", _text("hi")), _msg("assistant", _out("hello"))])
        result = session_to_messages_codex(p)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_output_file_text(self, tmp_path):
        p = tmp_path / "s.jsonl"
        _write_jsonl(p, [_msg("user", _text("hi"))])
        out = tmp_path / "nested" / "out.json"
        result = session_to_messages_codex(p, output=out)
        assert result == [{"role": "user", "content": "hi"}]
        assert out.exists()
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved == [{"role": "user", "content": "hi"}]

    def test_output_file_with_image(self, tmp_path):
        uri = _tiny_png_data_uri()
        p = tmp_path / "s.jsonl"
        _write_jsonl(p, [_msg("user", _text("look"), {"type": "input_image", "image_url": uri})])
        out = tmp_path / "out.json"
        result = session_to_messages_codex(p, output=out)
        # In-memory still has PIL image
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][1]["type"] == "image"
        assert result[0]["content"][1]["image"].size == (1, 1)
        # On disk, image replaced with data URI string
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(saved[0]["content"], list)
        assert saved[0]["content"][1]["type"] == "image"
        assert isinstance(saved[0]["content"][1]["image"], str)
        assert saved[0]["content"][1]["image"].startswith("data:image/")

    def test_accepts_string_path(self, tmp_path):
        p = tmp_path / "s.jsonl"
        _write_jsonl(p, [_msg("user", _text("hi"))])
        assert session_to_messages_codex(str(p)) == [{"role": "user", "content": "hi"}]

    def test_does_not_write_when_output_none(self, tmp_path):
        p = tmp_path / "s.jsonl"
        _write_jsonl(p, [_msg("user", _text("hi"))])
        before = set(x.name for x in tmp_path.iterdir())
        session_to_messages_codex(p)
        after = set(x.name for x in tmp_path.iterdir())
        assert before == after


# ---------------------------------------------------------------------------
# _record_type_of / _timestamp_of / _total_tokens_of
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_record_type_of_with_sub(self):
        rec = {"type": "response_item", "payload": {"type": "message"}}
        assert codex._record_type_of(rec) == "response_item:message"

    def test_record_type_of_without_sub(self):
        assert codex._record_type_of({"type": "event"}) == "event"

    def test_record_type_of_missing(self):
        assert codex._record_type_of({}) == "unknown"

    def test_timestamp_of(self):
        rec = {"timestamp": "2024-01-02T03:04:05.123Z"}
        assert codex._timestamp_of(rec) == "2024-01-02 03:04:05"

    def test_timestamp_of_missing(self):
        assert codex._timestamp_of({}) == ""

    def test_total_tokens_latest_snapshot(self):
        records = [
            {"payload": {"type": "token_count",
                         "info": {"total_token_usage": {
                             "input_tokens": 1, "output_tokens": 2,
                             "cached_input_tokens": 3, "total_tokens": 6}}}},
            {"payload": {"type": "token_count",
                         "info": {"total_token_usage": {
                             "input_tokens": 10, "output_tokens": 20,
                             "cached_input_tokens": 30, "total_tokens": 60}}}},
        ]
        assert codex._total_tokens_of(records) == {
            "input_tokens": 10, "output_tokens": 20,
            "cached_input_tokens": 30, "total_tokens": 60,
        }

    def test_total_tokens_empty(self):
        assert codex._total_tokens_of([]) == {}

    def test_total_tokens_no_token_count(self):
        assert codex._total_tokens_of([{"payload": {"type": "message"}}]) == {}

    def test_total_tokens_info_null(self):
        records = [{"payload": {"type": "token_count", "info": None}}]
        assert codex._total_tokens_of(records) == {}

    def test_total_tokens_usage_null(self):
        records = [{"payload": {"type": "token_count",
                                "info": {"total_token_usage": None}}}]
        assert codex._total_tokens_of(records) == {}

    def test_total_tokens_missing_fields(self):
        records = [{"payload": {"type": "token_count",
                                "info": {"total_token_usage": {}}}}]
        assert codex._total_tokens_of(records) == {}  # empty usage -> {} -> latest stays {}

    def test_total_tokens_partial_fields(self):
        records = [{"payload": {"type": "token_count",
                                "info": {"total_token_usage": {"input_tokens": 5}}}}]
        assert codex._total_tokens_of(records) == {
            "input_tokens": 5, "output_tokens": 0,
            "cached_input_tokens": 0, "total_tokens": 0,
        }


# ---------------------------------------------------------------------------
# CODEX_SCHEMA wiring
# ---------------------------------------------------------------------------

class TestCodexSchema:
    def test_label(self):
        assert codex.CODEX_SCHEMA.label == "CODEX"

    def test_is_message(self):
        assert codex.CODEX_SCHEMA.is_message(_msg("user", _text("x"))) is True
        assert codex.CODEX_SCHEMA.is_message({"type": "event_msg", "payload": {}}) is False

    def test_role_of(self):
        assert codex.CODEX_SCHEMA.role_of(_msg("user", _text("x"))) == "user"
        assert codex.CODEX_SCHEMA.role_of({"type": "event_msg", "payload": {}}) == "other"

    def test_text_of(self):
        assert codex.CODEX_SCHEMA.text_of(_msg("user", _text("hello"))) == "hello"

    def test_record_type_of(self):
        rec = {"type": "response_item", "payload": {"type": "message"}}
        assert codex.CODEX_SCHEMA.record_type_of(rec) == "response_item:message"

    def test_total_tokens_of(self):
        records = [{"payload": {"type": "token_count",
                                "info": {"total_token_usage": {"total_tokens": 9}}}}]
        assert codex.CODEX_SCHEMA.total_tokens_of(records)["total_tokens"] == 9


# ---------------------------------------------------------------------------
# inspect_session_codex
# ---------------------------------------------------------------------------

class TestInspectSessionCodex:
    def test_prints_report(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        _write_jsonl(p, [
            {"timestamp": "2024-01-01T00:00:00Z", "type": "response_item",
             "payload": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "hello"}]}},
        ])
        inspect_session_codex(p)
        out = capsys.readouterr().out
        assert "CODEX" in out
        assert "hello" in out

    def test_show_flow_and_blocks(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        _write_jsonl(p, [
            {"timestamp": "2024-01-01T00:00:00Z", "type": "response_item",
             "payload": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "flowtext"}]}},
        ])
        inspect_session_codex(p, show_flow=True, show_blocks=True)
        out = capsys.readouterr().out
        assert "flowtext" in out
