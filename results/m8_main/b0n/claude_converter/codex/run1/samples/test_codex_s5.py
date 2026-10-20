import base64
import io
import json

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

def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _response_item(payload):
    return {"type": "response_item", "payload": payload}


def _message(role, blocks):
    return _response_item({"type": "message", "role": role, "content": blocks})


def _text_block(text):
    return {"type": "input_text", "text": text}


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------

class TestLoadSessionCodex:
    def test_loads_records(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [{"a": 1}, {"b": 2}])
        assert load_session_codex(p) == [{"a": 1}, {"b": 2}]

    def test_str_path(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [{"a": 1}])
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
    def test_message_text_blocks(self):
        payload = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": "world"},
            ],
        }
        assert _payload_text(payload) == "hello\nworld"

    def test_message_output_text_and_text(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "output_text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        }
        assert _payload_text(payload) == "a\nb"

    def test_message_skips_blank(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "  "},
                {"type": "input_text", "text": "keep"},
            ],
        }
        assert _payload_text(payload) == "keep"

    def test_message_with_image(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:..."},
            ],
        }
        assert _payload_text(payload) == "look\n<image>"

    def test_message_non_dict_block_skipped(self):
        payload = {
            "type": "message",
            "content": ["junk", {"type": "input_text", "text": "ok"}],
        }
        assert _payload_text(payload) == "ok"

    def test_reasoning(self):
        payload = {
            "type": "reasoning",
            "summary": [{"text": "thinking hard"}, {"text": "more"}],
        }
        assert _payload_text(payload) == "<thinking>thinking hard\nmore</thinking>"

    def test_reasoning_empty(self):
        payload = {"type": "reasoning", "summary": []}
        assert _payload_text(payload) == ""

    def test_reasoning_blank_summaries(self):
        payload = {"type": "reasoning", "summary": [{"text": "   "}]}
        assert _payload_text(payload) == ""

    def test_function_call(self):
        payload = {"type": "function_call", "name": "ls", "arguments": '{"x":1}'}
        assert _payload_text(payload) == "<tool_use name='ls'>{\"x\":1}</tool_use>"

    def test_function_call_output(self):
        payload = {"type": "function_call_output", "output": "result"}
        assert _payload_text(payload) == "<tool_result>result</tool_result>"

    def test_custom_tool_call(self):
        payload = {"type": "custom_tool_call", "name": "grep", "input": "abc"}
        assert _payload_text(payload) == "<tool_use name='grep'>abc</tool_use>"

    def test_custom_tool_call_output(self):
        payload = {"type": "custom_tool_call_output", "output": "res"}
        assert _payload_text(payload) == "<tool_result>res</tool_result>"

    def test_unknown_type(self):
        assert _payload_text({"type": "whatever"}) == ""

    def test_missing_type(self):
        assert _payload_text({}) == ""


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------

class TestRecordRole:
    def test_non_response_item(self):
        assert _record_role({"type": "event_msg", "payload": {}}) is None

    def test_missing_type(self):
        assert _record_role({}) is None

    def test_user_message(self):
        assert _record_role(_message("user", [])) == "user"

    def test_assistant_message(self):
        assert _record_role(_message("assistant", [])) == "assistant"

    def test_developer_skipped(self):
        assert _record_role(_message("developer", [])) is None

    def test_unknown_role(self):
        assert _record_role(_message("system", [])) is None

    def test_tool_roles(self):
        assert _record_role(_response_item({"type": "reasoning"})) == "assistant"
        assert _record_role(_response_item({"type": "function_call"})) == "assistant"
        assert _record_role(_response_item({"type": "custom_tool_call"})) == "assistant"
        assert _record_role(_response_item({"type": "function_call_output"})) == "user"
        assert _record_role(_response_item({"type": "custom_tool_call_output"})) == "user"

    def test_unknown_payload_type(self):
        assert _record_role(_response_item({"type": "mystery"})) is None

    def test_null_payload(self):
        assert _record_role({"type": "response_item", "payload": None}) is None


# ---------------------------------------------------------------------------
# _payload_parts
# ---------------------------------------------------------------------------

class TestPayloadParts:
    def test_text_parts(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "a"},
                {"type": "output_text", "text": "b"},
            ],
        }
        assert _payload_parts(payload) == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_blank_text_skipped(self):
        payload = {"type": "message", "content": [{"type": "input_text", "text": "  "}]}
        assert _payload_parts(payload) == []

    def test_non_dict_block_skipped(self):
        payload = {"type": "message", "content": ["x", {"type": "input_text", "text": "ok"}]}
        assert _payload_parts(payload) == [{"type": "text", "text": "ok"}]

    def test_non_message_uses_payload_text(self):
        payload = {"type": "function_call", "name": "ls", "arguments": "{}"}
        assert _payload_parts(payload) == [{"type": "text", "text": "<tool_use name='ls'>{}</tool_use>"}]

    def test_non_message_blank(self):
        assert _payload_parts({"type": "mystery"}) == []

    def test_missing_image_url_skipped(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_image", "image_url": ""},
                {"type": "input_text", "text": "t"},
            ],
        }
        assert _payload_parts(payload) == [{"type": "text", "text": "t"}]

    def test_unreadable_image_skipped(self, capsys):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,notb64"},
                {"type": "input_text", "text": "t"},
            ],
        }
        parts = _payload_parts(payload)
        assert parts == [{"type": "text", "text": "t"}]
        # message printed to stderr about skipping
        assert "Skipping unreadable input_image" in capsys.readouterr().err

    def test_valid_image(self, monkeypatch):
        sentinel = object()

        import claude_converter.codex as codex_mod

        def fake_convert(uri):
            assert uri == "data:image/png;base64,AAAA"
            return sentinel

        monkeypatch.setattr(codex_mod, "convert_base64_to_pil_image", fake_convert)

        payload = {
            "type": "message",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ],
        }
        assert _payload_parts(payload) == [{"type": "image", "image": sentinel}]


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------

class TestRecordsToMessages:
    def test_empty(self):
        assert records_to_messages_codex([]) == []

    def test_simple_alternation(self):
        records = [
            _message("user", [_text_block("hi")]),
            _message("assistant", [{"type": "output_text", "text": "hello"}]),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_merge_consecutive_same_role(self):
        records = [
            _message("user", [_text_block("a")]),
            _message("user", [_text_block("b")]),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "a\nb"},
        ]

    def test_tool_call_merged_into_assistant(self):
        records = [
            _message("assistant", [{"type": "output_text", "text": "let me check"}]),
            _response_item({"type": "function_call", "name": "ls", "arguments": "{}"}),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "assistant", "content": "let me check\n<tool_use name='ls'>{}</tool_use>"},
        ]

    def test_developer_skipped(self):
        records = [
            _message("developer", [_text_block("sys")]),
            _message("user", [_text_block("hi")]),
        ]
        assert records_to_messages_codex(records) == [{"role": "user", "content": "hi"}]

    def test_empty_parts_skipped(self):
        records = [
            _message("user", []),
            _message("user", [_text_block("hi")]),
        ]
        assert records_to_messages_codex(records) == [{"role": "user", "content": "hi"}]

    def test_non_message_records_skipped(self):
        records = [
            {"type": "event_msg", "payload": {"type": "message", "role": "user", "content": []}},
            _message("user", [_text_block("hi")]),
        ]
        assert records_to_messages_codex(records) == [{"role": "user", "content": "hi"}]

    def test_reasoning_and_tool_output_flow(self):
        records = [
            _message("user", [_text_block("do it")]),
            _response_item({"type": "reasoning", "summary": [{"text": "plan"}]}),
            _response_item({"type": "function_call", "name": "run", "arguments": "x"}),
            _response_item({"type": "function_call_output", "output": "done"}),
        ]
        result = records_to_messages_codex(records)
        assert result == [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": "<thinking>plan</thinking>\n<tool_use name='run'>x</tool_use>",
            },
            {"role": "user", "content": "<tool_result>done</tool_result>"},
        ]

    def test_multimodal_content_kept_as_list(self, monkeypatch):
        import claude_converter.codex as codex_mod

        sentinel = object()
        monkeypatch.setattr(
            codex_mod, "convert_base64_to_pil_image", lambda uri: sentinel
        )
        records = [
            _message(
                "user",
                [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
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


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------

class TestSessionToMessages:
    def test_returns_messages(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "s.jsonl",
            [
                _message("user", [_text_block("hi")]),
                _message("assistant", [{"type": "output_text", "text": "yo"}]),
            ],
        )
        assert session_to_messages_codex(p) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]

    def test_writes_output_json(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [_message("user", [_text_block("hi")])])
        out = tmp_path / "out" / "result.json"
        result = session_to_messages_codex(p, output=out)
        assert result == [{"role": "user", "content": "hi"}]
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == [{"role": "user", "content": "hi"}]

    def test_output_with_image_serialized(self, tmp_path, monkeypatch):
        import claude_converter.codex as codex_mod
        from PIL import Image

        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        monkeypatch.setattr(codex_mod, "convert_base64_to_pil_image", lambda uri: img)

        p = _write_jsonl(
            tmp_path / "s.jsonl",
            [_message("user", [{"type": "input_image", "image_url": "data:x"}])],
        )
        out = tmp_path / "out.json"
        result = session_to_messages_codex(p, output=out)

        # in memory still has PIL image
        assert isinstance(result[0]["content"][0]["image"], Image.Image)

        loaded = json.loads(out.read_text(encoding="utf-8"))
        img_part = loaded[0]["content"][0]
        assert img_part["type"] == "image"
        assert isinstance(img_part["image"], str)
        assert img_part["image"].startswith("data:image/")


# ---------------------------------------------------------------------------
# inspection helpers
# ---------------------------------------------------------------------------

class TestRecordTypeOf:
    def test_with_subtype(self):
        assert _record_type_of({"type": "response_item", "payload": {"type": "message"}}) == "response_item:message"

    def test_without_subtype(self):
        assert _record_type_of({"type": "event_msg", "payload": {}}) == "event_msg"

    def test_unknown_type(self):
        assert _record_type_of({}) == "unknown"


class TestTimestampOf:
    def test_iso(self):
        assert _timestamp_of({"timestamp": "2024-01-02T03:04:05.678Z"}) == "2024-01-02 03:04:05"

    def test_missing(self):
        assert _timestamp_of({}) == ""


class TestTotalTokensOf:
    def test_empty(self):
        assert _total_tokens_of([]) == {}

    def test_latest_snapshot_wins(self):
        records = [
            {"payload": {"type": "token_count", "info": {"total_token_usage": {
                "input_tokens": 1, "output_tokens": 2, "cached_input_tokens": 0, "total_tokens": 3,
            }}}},
            {"payload": {"type": "token_count", "info": {"total_token_usage": {
                "input_tokens": 10, "output_tokens": 20, "cached_input_tokens": 5, "total_tokens": 30,
            }}}},
        ]
        assert _total_tokens_of(records) == {
            "input_tokens": 10,
            "output_tokens": 20,
            "cached_input_tokens": 5,
            "total_tokens": 30,
        }

    def test_defaults_filled(self):
        records = [
            {"payload": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 7}}}},
        ]
        assert _total_tokens_of(records) == {
            "input_tokens": 7,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "total_tokens": 0,
        }

    def test_null_info_ignored(self):
        records = [
            {"payload": {"type": "token_count", "info": None}},
        ]
        assert _total_tokens_of(records) == {}

    def test_null_usage_ignored(self):
        records = [
            {"payload": {"type": "token_count", "info": {"total_token_usage": None}}},
        ]
        assert _total_tokens_of(records) == {}

    def test_non_token_count_ignored(self):
        records = [{"payload": {"type": "message"}}]
        assert _total_tokens_of(records) == {}


# ---------------------------------------------------------------------------
# CODEX_SCHEMA
# ---------------------------------------------------------------------------

class TestCodexSchema:
    def test_label(self):
        assert CODEX_SCHEMA.label == "CODEX"

    def test_is_message(self):
        assert CODEX_SCHEMA.is_message(_message("user", [])) is True
        assert CODEX_SCHEMA.is_message({"type": "event_msg", "payload": {}}) is False

    def test_role_of(self):
        assert CODEX_SCHEMA.role_of(_message("user", [])) == "user"
        assert CODEX_SCHEMA.role_of({"type": "event_msg", "payload": {}}) == "other"

    def test_text_of(self):
        assert CODEX_SCHEMA.text_of(_message("user", [_text_block("hi")])) == "hi"


# ---------------------------------------------------------------------------
# inspect_session_codex
# ---------------------------------------------------------------------------

class TestInspectSession:
    def test_prints_report(self, tmp_path, capsys):
        p = _write_jsonl(
            tmp_path / "s.jsonl",
            [
                _message("user", [_text_block("hi")]),
                _message("assistant", [{"type": "output_text", "text": "yo"}]),
            ],
        )
        inspect_session_codex(p)
        out = capsys.readouterr().out
        assert "CODEX" in out

    def test_show_flow(self, tmp_path, capsys):
        p = _write_jsonl(tmp_path / "s.jsonl", [_message("user", [_text_block("hi")])])
        inspect_session_codex(p, show_flow=True, show_blocks=True)
        out = capsys.readouterr().out
        assert "hi" in out
