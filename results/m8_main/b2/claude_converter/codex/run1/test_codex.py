from __future__ import annotations

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


def _record(payload: dict, rtype: str = "response_item", timestamp: str = "2024-01-01T00:00:00Z") -> dict:
    return {"type": rtype, "payload": payload, "timestamp": timestamp}


def _msg(content, role="user") -> dict:
    return _record({"type": "message", "role": role, "content": content})


def _text(content: str, role: str = "user") -> dict:
    return _msg([{"type": "input_text", "text": content}], role=role)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------


class TestLoadSessionCodex:
    def test_loads_records(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [{"type": "a"}, {"type": "b"}])
        assert load_session_codex(p) == [{"type": "a"}, {"type": "b"}]

    def test_accepts_str_path(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [{"x": 1}])
        assert load_session_codex(str(p)) == [{"x": 1}]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session_codex(tmp_path / "nope.jsonl")

    def test_bad_extension(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text("{}\n")
        with pytest.raises(ValueError):
            load_session_codex(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("\n\n")
        with pytest.raises(ValueError):
            load_session_codex(p)

    def test_skips_blank_lines_and_invalid_json(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text('{"a": 1}\n\nnot json\n{"b": 2}\n')
        assert load_session_codex(p) == [{"a": 1}, {"b": 2}]
        err = capsys.readouterr().err
        assert "Line 3 is invalid JSON" in err


# ---------------------------------------------------------------------------
# _payload_text
# ---------------------------------------------------------------------------


class TestPayloadText:
    def test_message_text_blocks(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "output_text", "text": "world"},
                {"type": "text", "text": "extra"},
            ],
        }
        assert _payload_text(payload) == "hello\nworld\nextra"

    def test_message_image(self):
        payload = {"type": "message", "content": [{"type": "input_image", "image_url": "x"}]}
        assert _payload_text(payload) == "<image>"

    def test_message_skips_empty_and_non_dict(self):
        payload = {
            "type": "message",
            "content": [
                {"type": "input_text", "text": "   "},
                "not a dict",
                {"type": "input_text", "text": "kept"},
            ],
        }
        assert _payload_text(payload) == "kept"

    def test_message_null_content(self):
        assert _payload_text({"type": "message", "content": None}) == ""

    def test_reasoning(self):
        payload = {"type": "reasoning", "summary": [{"text": "think"}, {"text": "more"}]}
        assert _payload_text(payload) == "<thinking>think\nmore</thinking>"

    def test_reasoning_empty(self):
        assert _payload_text({"type": "reasoning", "summary": []}) == ""

    def test_function_call(self):
        payload = {"type": "function_call", "name": "foo", "arguments": '{"a":1}'}
        assert _payload_text(payload) == "<tool_use name='foo'>{\"a\":1}</tool_use>"

    def test_function_call_output(self):
        payload = {"type": "function_call_output", "output": "result"}
        assert _payload_text(payload) == "<tool_result>result</tool_result>"

    def test_custom_tool_call(self):
        payload = {"type": "custom_tool_call", "name": "bar", "input": "in"}
        assert _payload_text(payload) == "<tool_use name='bar'>in</tool_use>"

    def test_custom_tool_call_output(self):
        payload = {"type": "custom_tool_call_output", "output": "out"}
        assert _payload_text(payload) == "<tool_result>out</tool_result>"

    def test_unknown_type(self):
        assert _payload_text({"type": "weird"}) == ""


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------


class TestRecordRole:
    def test_non_response_item(self):
        assert _record_role({"type": "event_msg", "payload": {"type": "message"}}) is None

    def test_user_message(self):
        assert _record_role(_msg([], role="user")) == "user"

    def test_assistant_message(self):
        assert _record_role(_msg([], role="assistant")) == "assistant"

    def test_developer_message_skipped(self):
        assert _record_role(_msg([], role="developer")) is None

    def test_other_message_role(self):
        assert _record_role(_msg([], role="system")) is None

    @pytest.mark.parametrize(
        "ptype,role",
        [
            ("reasoning", "assistant"),
            ("function_call", "assistant"),
            ("custom_tool_call", "assistant"),
            ("function_call_output", "user"),
            ("custom_tool_call_output", "user"),
        ],
    )
    def test_tool_roles(self, ptype, role):
        assert _record_role(_record({"type": ptype})) == role

    def test_unknown_payload_type(self):
        assert _record_role(_record({"type": "mystery"})) is None

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
                {"type": "input_text", "text": "hi"},
                {"type": "text", "text": "  "},
                {"type": "output_text", "text": "bye"},
            ],
        }
        assert _payload_parts(payload) == [
            {"type": "text", "text": "hi"},
            {"type": "text", "text": "bye"},
        ]

    def test_image_missing_url_skipped(self):
        payload = {"type": "message", "content": [{"type": "input_image"}]}
        assert _payload_parts(payload) == []

    def test_image_unreadable_skipped(self, monkeypatch, capsys):
        import claude_converter.codex as codex

        def boom(_url):
            raise ValueError("bad")

        monkeypatch.setattr(codex, "convert_base64_to_pil_image", boom)
        payload = {"type": "message", "content": [{"type": "input_image", "image_url": "x"}]}
        assert _payload_parts(payload) == []
        assert "Skipping unreadable input_image" in capsys.readouterr().err

    def test_image_import_error_skipped(self, monkeypatch, capsys):
        import claude_converter.codex as codex

        def boom(_url):
            raise ImportError("no pillow")

        monkeypatch.setattr(codex, "convert_base64_to_pil_image", boom)
        payload = {"type": "message", "content": [{"type": "input_image", "image_url": "x"}]}
        assert _payload_parts(payload) == []
        assert "Skipping unreadable input_image" in capsys.readouterr().err

    def test_image_success(self, monkeypatch):
        import claude_converter.codex as codex

        sentinel = object()
        monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda _u: sentinel)
        payload = {"type": "message", "content": [{"type": "input_image", "image_url": "x"}]}
        result = _payload_parts(payload)
        assert result == [{"type": "image", "image": sentinel}]

    def test_non_message_flat_text(self):
        payload = {"type": "function_call", "name": "f", "arguments": "a"}
        assert _payload_parts(payload) == [
            {"type": "text", "text": "<tool_use name='f'>a</tool_use>"}
        ]

    def test_non_message_empty_text(self):
        assert _payload_parts({"type": "reasoning", "summary": []}) == []


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------


class TestRecordsToMessages:
    def test_empty(self):
        assert records_to_messages_codex([]) == []

    def test_simple_user_assistant(self):
        records = [
            _text("hello", "user"),
            _text("hi there", "assistant"),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_skips_non_conversational(self):
        records = [
            {"type": "event_msg", "payload": {"type": "message"}},
            _msg([], role="developer"),
            _text("kept", "user"),
        ]
        assert records_to_messages_codex(records) == [{"role": "user", "content": "kept"}]

    def test_skips_empty_parts(self):
        records = [
            _msg([{"type": "input_text", "text": "   "}], role="user"),
            _text("real", "user"),
        ]
        assert records_to_messages_codex(records) == [{"role": "user", "content": "real"}]

    def test_merges_consecutive_same_role(self):
        records = [
            _text("one", "user"),
            _text("two", "user"),
            _text("reply", "assistant"),
        ]
        assert records_to_messages_codex(records) == [
            {"role": "user", "content": "one\ntwo"},
            {"role": "assistant", "content": "reply"},
        ]

    def test_tool_call_flattening(self):
        records = [
            _text("question", "user"),
            _record({"type": "reasoning", "summary": [{"text": "plan"}]}),
            _record({"type": "function_call", "name": "run", "arguments": "a"}),
            _record({"type": "function_call_output", "output": "out"}),
            _text("done", "assistant"),
        ]
        messages = records_to_messages_codex(records)
        assert messages == [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": (
                    "<thinking>plan</thinking>\n"
                    "<tool_use name='run'>a</tool_use>"
                ),
            },
            {"role": "user", "content": "<tool_result>out</tool_result>"},
            {"role": "assistant", "content": "done"},
        ]

    def test_image_produces_list_content(self, monkeypatch):
        import claude_converter.codex as codex

        sentinel = object()
        monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda _u: sentinel)
        records = [
            _msg(
                [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                ],
                role="user",
            )
        ]
        assert records_to_messages_codex(records) == [
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
    def test_roundtrip_without_output(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "s.jsonl",
            [_text("hello", "user"), _text("hi", "assistant")],
        )
        assert session_to_messages_codex(p) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_writes_output(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [_text("hello", "user")])
        out = tmp_path / "sub" / "out.json"
        messages = session_to_messages_codex(p, output=out)
        assert messages == [{"role": "user", "content": "hello"}]
        assert out.exists()
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved == [{"role": "user", "content": "hello"}]

    def test_output_encodes_images(self, tmp_path, monkeypatch):
        import claude_converter.codex as codex

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not available")

        img = Image.new("RGB", (2, 2), color="red")
        monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda _u: img)

        records = [
            _msg(
                [
                    {"type": "input_text", "text": "pic"},
                    {"type": "input_image", "image_url": "x"},
                ],
                role="user",
            )
        ]
        p = _write_jsonl(tmp_path / "s.jsonl", records)
        out = tmp_path / "out.json"
        messages = session_to_messages_codex(p, output=out)

        # In-memory keeps the real image.
        assert messages[0]["content"][1]["image"] is img

        saved = json.loads(out.read_text(encoding="utf-8"))
        image_part = saved[0]["content"][1]
        assert image_part["type"] == "image"
        assert isinstance(image_part["image"], str)
        assert image_part["image"].startswith("data:image/")

    def test_output_replaces_image_with_repr_when_pillow_missing(
        self, tmp_path, monkeypatch
    ):
        """When Pillow unavailable, messages_to_json_safe falls back to repr(image)."""
        import builtins
        import claude_converter.codex as codex

        sentinel = "FAKE_IMAGE"
        monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda _u: sentinel)

        # Simulate Pillow being unimportable inside messages_to_json_safe.
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("no pillow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        records = [
            _msg(
                [
                    {"type": "input_text", "text": "pic"},
                    {"type": "input_image", "image_url": "x"},
                ],
                role="user",
            )
        ]
        p = _write_jsonl(tmp_path / "s.jsonl", records)
        out = tmp_path / "out.json"
        messages = session_to_messages_codex(p, output=out)

        saved = json.loads(out.read_text(encoding="utf-8"))
        image_part = saved[0]["content"][1]
        assert image_part == {"type": "image", "image": repr(sentinel)}

        # In-memory list is unmodified.
        assert messages[0]["content"][1]["image"] == sentinel

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            session_to_messages_codex(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------------------
# Small helpers used by the schema
# ---------------------------------------------------------------------------


class TestRecordHelpers:
    def test_record_type_of_with_sub(self):
        assert _record_type_of({"type": "response_item", "payload": {"type": "message"}}) == (
            "response_item:message"
        )

    def test_record_type_of_no_sub(self):
        assert _record_type_of({"type": "event_msg", "payload": {}}) == "event_msg"

    def test_record_type_of_payload_not_dict(self):
        assert _record_type_of({"type": "response_item", "payload": None}) == "response_item"

    def test_record_type_of_missing_type(self):
        assert _record_type_of({"payload": {}}) == "unknown"

    def test_timestamp_of(self):
        assert _timestamp_of({"timestamp": "2024-01-01T12:34:56.789Z"}) == "2024-01-01 12:34:56"

    def test_timestamp_of_missing(self):
        assert _timestamp_of({}) == ""

    def test_timestamp_of_null(self):
        # str(None) -> "None", sliced to 19 chars and T->space is a no-op.
        assert _timestamp_of({"timestamp": None}) == "None"

    def test_total_tokens_keeps_latest(self):
        records = [
            _record(
                {
                    "type": "token_count",
                    "info": {"total_token_usage": {"input_tokens": 1, "output_tokens": 2}},
                }
            ),
            _record(
                {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 10,
                            "output_tokens": 20,
                            "cached_input_tokens": 5,
                            "total_tokens": 30,
                        }
                    },
                }
            ),
        ]
        assert _total_tokens_of(records) == {
            "input_tokens": 10,
            "output_tokens": 20,
            "cached_input_tokens": 5,
            "total_tokens": 30,
        }

    def test_total_tokens_partial_usage_uses_defaults(self):
        records = [
            _record(
                {
                    "type": "token_count",
                    "info": {"total_token_usage": {"input_tokens": 7}},
                }
            ),
        ]
        assert _total_tokens_of(records) == {
            "input_tokens": 7,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "total_tokens": 0,
        }

    def test_total_tokens_ignores_non_token_records(self):
        records = [
            _record({"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1}}}),
            _record({"type": "message", "role": "user", "content": []}),
        ]
        assert _total_tokens_of(records) == {
            "input_tokens": 1,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "total_tokens": 0,
        }

    def test_total_tokens_empty(self):
        assert _total_tokens_of([]) == {}

    def test_total_tokens_null_info(self):
        records = [_record({"type": "token_count", "info": None})]
        assert _total_tokens_of(records) == {}

    def test_total_tokens_null_usage(self):
        records = [_record({"type": "token_count", "info": {"total_token_usage": None}})]
        assert _total_tokens_of(records) == {}


# ---------------------------------------------------------------------------
# Schema + inspect
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_fields(self):
        assert CODEX_SCHEMA.label == "CODEX"
        assert CODEX_SCHEMA.is_message(_text("hi")) is True
        assert CODEX_SCHEMA.is_message({"type": "event_msg", "payload": {}}) is False
        assert CODEX_SCHEMA.role_of(_text("hi", "assistant")) == "assistant"
        assert CODEX_SCHEMA.role_of({"type": "event_msg", "payload": {}}) == "other"
        assert CODEX_SCHEMA.text_of(_text("hi")) == "hi"
        assert CODEX_SCHEMA.timestamp_of({"timestamp": "2024-01-01T00:00:00Z"}) == (
            "2024-01-01 00:00:00"
        )

    def test_schema_token_summary(self):
        records = [
            _record(
                {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 3,
                            "output_tokens": 4,
                            "cached_input_tokens": 1,
                            "total_tokens": 7,
                        }
                    },
                }
            )
        ]
        assert CODEX_SCHEMA.total_tokens_of(records) == {
            "input_tokens": 3,
            "output_tokens": 4,
            "cached_input_tokens": 1,
            "total_tokens": 7,
        }


class TestInspectSessionCodex:
    def test_prints_report(self, tmp_path, capsys):
        p = _write_jsonl(
            tmp_path / "s.jsonl",
            [_text("hello", "user"), _text("hi", "assistant")],
        )
        inspect_session_codex(p)
        out = capsys.readouterr().out
        assert "CODEX" in out

    def test_show_flow(self, tmp_path, capsys):
        p = _write_jsonl(
            tmp_path / "s.jsonl",
            [_text("hello", "user"), _text("hi", "assistant")],
        )
        inspect_session_codex(p, show_flow=True)
        out = capsys.readouterr().out
        assert "CODEX" in out
        # The flow section shows timestamps, not message bodies by default.
        assert "2024-01-01 00:00:00" in out

    def test_show_blocks(self, tmp_path, capsys):
        p = _write_jsonl(
            tmp_path / "s.jsonl",
            [_text("visible-body", "user")],
        )
        inspect_session_codex(p, show_flow=True, show_blocks=True)
        out = capsys.readouterr().out
        assert "CODEX" in out

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_session_codex(tmp_path / "missing.jsonl")
