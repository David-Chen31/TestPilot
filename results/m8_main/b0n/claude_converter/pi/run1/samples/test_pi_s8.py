"""Unit tests for claude_converter.pi."""

from __future__ import annotations

import base64
import json
from io import BytesIO

import pytest

from claude_converter import pi
from claude_converter.pi import (
    load_session_pi,
    records_to_messages_pi,
    session_to_messages_pi,
    inspect_session_pi,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _png_bytes() -> bytes:
    PIL = pytest.importorskip("PIL.Image")
    buf = BytesIO()
    PIL.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _b64_png() -> str:
    return base64.b64encode(_png_bytes()).decode("ascii")


def _write_jsonl(tmp_path, lines, name="session.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def _msg(role, content, **extra):
    message = {"role": role, "content": content}
    message.update(extra)
    return {"type": "message", "message": message}


# ---------------------------------------------------------------------------
# load_session_pi
# ---------------------------------------------------------------------------


class TestLoadSessionPi:
    def test_loads_records(self, tmp_path):
        path = _write_jsonl(
            tmp_path,
            [
                {"type": "session", "id": "abc"},
                _msg("user", "hello"),
                _msg("assistant", "hi"),
            ],
        )
        records = load_session_pi(path)
        assert records == [
            {"type": "session", "id": "abc"},
            {"type": "message", "message": {"role": "user", "content": "hello"}},
            {"type": "message", "message": {"role": "assistant", "content": "hi"}},
        ]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text(
            '\n{"type": "session", "id": "x"}\n\n\n{"type": "message", "message": {"role": "user", "content": "a"}}\n',
            encoding="utf-8",
        )
        records = load_session_pi(str(path))
        assert records == [
            {"type": "session", "id": "x"},
            {"type": "message", "message": {"role": "user", "content": "a"}},
        ]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_session_pi(tmp_path / "nope.jsonl")

    def test_bad_extension_raises(self, tmp_path):
        path = tmp_path / "session.txt"
        path.write_text('{"a": 1}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Unrecognized extension"):
            load_session_pi(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty or contains no valid records"):
            load_session_pi(path)

    def test_invalid_json_lines_skipped(self, tmp_path, capsys):
        path = tmp_path / "s.jsonl"
        path.write_text(
            '{"type": "session", "id": "x"}\nNOT JSON\n{"type": "message", "message": {"role": "user", "content": "a"}}\n',
            encoding="utf-8",
        )
        records = load_session_pi(path)
        assert records == [
            {"type": "session", "id": "x"},
            {"type": "message", "message": {"role": "user", "content": "a"}},
        ]
        err = capsys.readouterr().err
        assert "X Line 2 is invalid JSON" in err


# ---------------------------------------------------------------------------
# _block_to_text
# ---------------------------------------------------------------------------


class TestBlockToText:
    def test_text(self):
        assert pi._block_to_text({"type": "text", "text": "hi"}) == "hi"

    def test_text_missing_value(self):
        assert pi._block_to_text({"type": "text"}) == ""

    def test_thinking(self):
        assert (
            pi._block_to_text({"type": "thinking", "thinking": "pondering"})
            == "<thinking>pondering</thinking>"
        )

    def test_thinking_missing_value(self):
        assert pi._block_to_text({"type": "thinking"}) == "<thinking></thinking>"

    def test_tool_call(self):
        out = pi._block_to_text(
            {"type": "toolCall", "name": "ls", "arguments": {"path": "/tmp"}}
        )
        assert out == "<tool_use name='ls'>{\"path\": \"/tmp\"}</tool_use>"

    def test_tool_call_default_arguments(self):
        out = pi._block_to_text({"type": "toolCall", "name": "noop"})
        assert out == "<tool_use name='noop'>{}</tool_use>"

    def test_tool_call_unicode_not_escaped(self):
        out = pi._block_to_text(
            {"type": "toolCall", "name": "echo", "arguments": {"s": "héllo"}}
        )
        assert "héllo" in out

    def test_image(self):
        assert pi._block_to_text({"type": "image", "data": "xxx"}) == "<image>"

    def test_unknown(self):
        assert pi._block_to_text({"type": "wat"}) == ""

    def test_missing_type(self):
        assert pi._block_to_text({}) == ""


# ---------------------------------------------------------------------------
# _decode_pi_image
# ---------------------------------------------------------------------------


class TestDecodePiImage:
    def test_empty_data_warns_and_returns_none(self, capsys):
        pytest.importorskip("PIL.Image")
        assert pi._decode_pi_image({"type": "image", "data": ""}) is None
        assert "Skipping image block with empty base64 data" in capsys.readouterr().err

    def test_missing_data_warns_and_returns_none(self, capsys):
        pytest.importorskip("PIL.Image")
        assert pi._decode_pi_image({"type": "image"}) is None
        assert "Skipping image block with empty base64 data" in capsys.readouterr().err

    def test_valid_image(self):
        PIL = pytest.importorskip("PIL.Image")
        img = pi._decode_pi_image({"type": "image", "data": _b64_png()})
        assert isinstance(img, PIL.Image)
        assert img.size == (2, 2)

    def test_data_uri(self):
        PIL = pytest.importorskip("PIL.Image")
        uri = "data:image/png;base64," + _b64_png()
        img = pi._decode_pi_image({"type": "image", "data": uri})
        assert isinstance(img, PIL.Image)

    def test_bad_base64_warns_and_returns_none(self, capsys):
        pytest.importorskip("PIL.Image")
        assert pi._decode_pi_image({"type": "image", "data": "not-an-image!!"}) is None
        assert "Skipping unreadable image block" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _message_of / _record_role
# ---------------------------------------------------------------------------


class TestMessageOfAndRole:
    def test_message_of_non_message(self):
        assert pi._message_of({"type": "session"}) is None

    def test_message_of_message(self):
        rec = {"type": "message", "message": {"role": "user"}}
        assert pi._message_of(rec) == {"role": "user"}

    def test_message_of_null_message(self):
        assert pi._message_of({"type": "message", "message": None}) is None

    def test_record_role_tool_result_maps_to_user(self):
        rec = {"type": "message", "message": {"role": "toolResult"}}
        assert pi._record_role(rec) == "user"

    def test_record_role_user(self):
        assert pi._record_role(_msg("user", "x")) == "user"

    def test_record_role_assistant(self):
        assert pi._record_role(_msg("assistant", "x")) == "assistant"

    def test_record_role_other_role(self):
        assert pi._record_role(_msg("system", "x")) is None

    def test_record_role_non_message(self):
        assert pi._record_role({"type": "session"}) is None

    def test_record_role_missing_role(self):
        assert pi._record_role({"type": "message", "message": {}}) is None


# ---------------------------------------------------------------------------
# _text_of
# ---------------------------------------------------------------------------


class TestTextOf:
    def test_string_content(self):
        assert pi._text_of(_msg("user", "hello")) == "hello"

    def test_non_message_returns_empty(self):
        assert pi._text_of({"type": "session"}) == ""

    def test_list_content_joined(self):
        rec = _msg(
            "assistant",
            [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        )
        assert pi._text_of(rec) == "a\nb"

    def test_list_content_skips_blank_parts(self):
        rec = _msg(
            "user",
            [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "   "},
                "not a dict",
                {"type": "thinking", "thinking": "t"},
            ],
        )
        assert pi._text_of(rec) == "a\n<thinking>t</thinking>"

    def test_unknown_content_type(self):
        assert pi._text_of(_msg("user", 123)) == ""

    def test_tool_result(self):
        rec = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "bash",
                "toolCallId": "call1",
                "content": [
                    {"type": "text", "text": "out1"},
                    {"type": "text", "text": "out2"},
                ],
            },
        }
        assert pi._text_of(rec) == (
            "<tool_result name='bash' id='call1'>out1\nout2</tool_result>"
        )

    def test_tool_result_with_image_marker(self):
        rec = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "view",
                "toolCallId": "c",
                "content": [
                    {"type": "text", "text": "see"},
                    {"type": "image", "data": "x"},
                ],
            },
        }
        assert pi._text_of(rec) == "<tool_result name='view' id='c'>see\n<image></tool_result>"

    def test_tool_result_blank_parts_omitted(self):
        rec = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "t",
                "toolCallId": "i",
                "content": [
                    {"type": "text", "text": "  "},
                    {"type": "text", "text": "real"},
                ],
            },
        }
        assert pi._text_of(rec) == "<tool_result name='t' id='i'>real</tool_result>"

    def test_tool_result_missing_name_id(self):
        rec = {
            "type": "message",
            "message": {"role": "toolResult", "content": "ignored-string"},
        }
        assert pi._text_of(rec) == "<tool_result name='' id=''></tool_result>"


# ---------------------------------------------------------------------------
# _message_parts
# ---------------------------------------------------------------------------


class TestMessageParts:
    def test_string_content(self):
        assert pi._message_parts(_msg("user", "hello")) == [
            {"type": "text", "text": "hello"}
        ]

    def test_string_content_blank_returns_empty(self):
        assert pi._message_parts(_msg("user", "   ")) == []

    def test_string_content_empty_returns_empty(self):
        assert pi._message_parts(_msg("user", "")) == []

    def test_list_content(self):
        rec = _msg(
            "assistant",
            [
                {"type": "text", "text": "a"},
                {"type": "toolCall", "name": "n", "arguments": {"k": 1}},
            ],
        )
        parts = pi._message_parts(rec)
        assert parts == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "<tool_use name='n'>{\"k\": 1}</tool_use>"},
        ]

    def test_list_content_skips_non_dicts_and_blanks(self):
        rec = _msg("user", ["notdict", {"type": "text", "text": " "}, {"type": "text", "text": "ok"}])
        assert pi._message_parts(rec) == [{"type": "text", "text": "ok"}]

    def test_unknown_content_type_returns_empty(self):
        assert pi._message_parts(_msg("user", 5)) == []

    def test_non_message_returns_empty(self):
        assert pi._message_parts({"type": "session"}) == []

    def test_non_dict_block_for_tool_result_no_image(self):
        # non-message-part branch
        rec = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "t",
                "toolCallId": "i",
                "content": [{"type": "text", "text": "x"}, "zzz"],
            },
        }
        assert pi._message_parts(rec) == [
            {"type": "text", "text": "<tool_result name='t' id='i'>x</tool_result>"}
        ]

    def test_tool_result_no_image_blank_text(self):
        rec = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "t",
                "toolCallId": "i",
                "content": [{"type": "text", "text": "  "}],
            },
        }
        assert pi._message_parts(rec) == []

    def test_tool_result_with_image(self):
        PIL = pytest.importorskip("PIL.Image")
        rec = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "view",
                "toolCallId": "c",
                "content": [
                    {"type": "text", "text": "  "},
                    {"type": "text", "text": "note"},
                    {"type": "image", "data": _b64_png()},
                    "junk",
                ],
            },
        }
        parts = pi._message_parts(rec)
        assert parts[0] == {"type": "text", "text": "<tool_result name='view' id='c'>"}
        assert parts[1] == {"type": "text", "text": "note"}
        assert parts[2]["type"] == "image"
        assert isinstance(parts[2]["image"], PIL.Image)
        assert parts[3] == {"type": "text", "text": "</tool_result>"}
        assert len(parts) == 4

    def test_tool_result_with_undecodable_image_omitted(self, capsys):
        pytest.importorskip("PIL.Image")
        rec = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "view",
                "toolCallId": "c",
                "content": [
                    {"type": "text", "text": "note"},
                    {"type": "image", "data": "garbage"},
                ],
            },
        }
        parts = pi._message_parts(rec)
        assert parts == [
            {"type": "text", "text": "<tool_result name='view' id='c'>"},
            {"type": "text", "text": "note"},
            {"type": "text", "text": "</tool_result>"},
        ]
        assert "Skipping unreadable image block" in capsys.readouterr().err

    def test_list_content_with_image(self):
        PIL = pytest.importorskip("PIL.Image")
        rec = _msg(
            "user",
            [
                {"type": "text", "text": "look"},
                {"type": "image", "data": _b64_png()},
            ],
        )
        parts = pi._message_parts(rec)
        assert parts[0] == {"type": "text", "text": "look"}
        assert parts[1]["type"] == "image"
        assert isinstance(parts[1]["image"], PIL.Image)

    def test_list_content_with_empty_image_skipped(self, capsys):
        pytest.importorskip("PIL.Image")
        rec = _msg("user", [{"type": "text", "text": "x"}, {"type": "image", "data": ""}])
        parts = pi._message_parts(rec)
        assert parts == [{"type": "text", "text": "x"}]
        assert "Skipping image block with empty base64 data" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# records_to_messages_pi
# ---------------------------------------------------------------------------


class TestRecordsToMessagesPi:
    def test_simple_conversation(self):
        records = [_msg("user", "hello"), _msg("assistant", "hi")]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_metadata_records_skipped(self):
        records = [
            {"type": "session", "id": "x"},
            {"type": "model_change"},
            _msg("user", "hi"),
        ]
        assert records_to_messages_pi(records) == [{"role": "user", "content": "hi"}]

    def test_empty_parts_skipped(self):
        records = [_msg("user", "   "), _msg("user", "real")]
        assert records_to_messages_pi(records) == [{"role": "user", "content": "real"}]

    def test_consecutive_same_role_merged(self):
        records = [_msg("user", "a"), _msg("user", "b")]
        assert records_to_messages_pi(records) == [{"role": "user", "content": "a\nb"}]

    def test_tool_result_maps_to_user_and_merges(self):
        records = [
            _msg("user", "q"),
            _msg("assistant", "thinking"),
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolName": "bash",
                    "toolCallId": "c1",
                    "content": [{"type": "text", "text": "output"}],
                },
            },
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "thinking"},
            {"role": "user", "content": "<tool_result name='bash' id='c1'>output</tool_result>"},
        ]

    def test_multimodal_keeps_list_content(self):
        pytest.importorskip("PIL.Image")
        records = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "describe"},
                    {"type": "image", "data": _b64_png()},
                ],
            )
        ]
        out = records_to_messages_pi(records)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        content = out[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image"

    def test_merge_parts_with_images_keeps_list(self):
        pytest.importorskip("PIL.Image")
        records = [
            _msg("user", "text first"),
            _msg("user", [{"type": "image", "data": _b64_png()}]),
        ]
        out = records_to_messages_pi(records)
        content = out[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "text first"}
        assert content[1]["type"] == "image"

    def test_empty_records(self):
        assert records_to_messages_pi([]) == []

    def test_tool_call_flattened(self):
        records = [
            _msg(
                "assistant",
                [{"type": "toolCall", "name": "ls", "arguments": {"path": "."}}],
            )
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {
                "role": "assistant",
                "content": "<tool_use name='ls'>{\"path\": \".\"}</tool_use>",
            }
        ]


# ---------------------------------------------------------------------------
# session_to_messages_pi
# ---------------------------------------------------------------------------


class TestSessionToMessagesPi:
    def test_returns_messages(self, tmp_path):
        path = _write_jsonl(
            tmp_path,
            [_msg("user", "hi"), _msg("assistant", "hello")],
        )
        out = session_to_messages_pi(path)
        assert out == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_output_writes_json(self, tmp_path):
        path = _write_jsonl(tmp_path, [_msg("user", "hi")])
        out_path = tmp_path / "nested" / "out.json"
        result = session_to_messages_pi(path, output=out_path)

        assert out_path.exists()
        with open(out_path, encoding="utf-8") as f:
            written = json.load(f)
        assert written == [{"role": "user", "content": "hi"}]
        assert result == [{"role": "user", "content": "hi"}]

    def test_output_serializes_images(self, tmp_path):
        PIL = pytest.importorskip("PIL.Image")
        path = _write_jsonl(
            tmp_path,
            [
                _msg(
                    "user",
                    [
                        {"type": "text", "text": "look"},
                        {"type": "image", "data": _b64_png()},
                    ],
                )
            ],
        )
        out_path = tmp_path / "out.json"
        result = session_to_messages_pi(path, output=out_path)

        # in-memory retains PIL image
        content = result[0]["content"]
        assert isinstance(content, list)
        assert isinstance(content[1]["image"], PIL.Image)

        with open(out_path, encoding="utf-8") as f:
            written = json.load(f)
        wc = written[0]["content"]
        assert wc[0] == {"type": "text", "text": "look"}
        assert wc[1]["type"] == "image"
        assert isinstance(wc[1]["image"], str)
        assert wc[1]["image"].startswith("data:image/")

    def test_no_output_returns_without_writing(self, tmp_path):
        path = _write_jsonl(tmp_path, [_msg("user", "hi")])
        before = set(p.name for p in tmp_path.iterdir())
        session_to_messages_pi(path)
        after = set(p.name for p in tmp_path.iterdir())
        assert before == after

    def test_missing_file_propagates(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            session_to_messages_pi(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------------------
# inspection helper schema functions
# ---------------------------------------------------------------------------


class TestSchemaHelpers:
    def test_record_type_of_message(self):
        rec = _msg("user", "x")
        assert pi._record_type_of(rec) == "message:user"

    def test_record_type_of_non_message(self):
        assert pi._record_type_of({"type": "session"}) == "session"

    def test_record_type_of_missing_type(self):
        assert pi._record_type_of({}) == "unknown"

    def test_record_type_of_message_missing_role(self):
        rec = {"type": "message", "message": {}}
        assert pi._record_type_of(rec) == "message:unknown"

    def test_timestamp_of_truncation(self):
        assert pi._timestamp_of({"timestamp": "2024-01-02T03:04:05.678Z"}) == "2024-01-02 03:04:05"

    def test_timestamp_of_missing(self):
        assert pi._timestamp_of({}) == ""

    def test_total_tokens_of_no_usage(self):
        assert pi._total_tokens_of([{"type": "message", "message": {"role": "user"}}]) == {}

    def test_total_tokens_of_ignores_non_messages(self):
        assert pi._total_tokens_of([{"type": "session"}]) == {}

    def test_total_tokens_of_sums(self):
        records = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "usage": {
                        "input": 10,
                        "output": 5,
                        "cacheRead": 2,
                        "cacheWrite": 1,
                    },
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "usage": {
                        "input": 1,
                        "output": 2,
                        "cacheRead": 0,
                        "cacheWrite": 3,
                    },
                },
            },
        ]
        assert pi._total_tokens_of(records) == {
            "input": 11,
            "output": 7,
            "cache_read": 2,
            "cache_write": 4,
        }

    def test_pi_schema_shape(self):
        assert pi.PI_SCHEMA.label == "PI"
        assert pi.PI_SCHEMA.record_type_of(_msg("user", "x")) == "message:user"
        assert pi.PI_SCHEMA.is_message(_msg("user", "x")) is True
        assert pi.PI_SCHEMA.is_message({"type": "session"}) is False
        assert pi.PI_SCHEMA.role_of(_msg("assistant", "x")) == "assistant"
        assert pi.PI_SCHEMA.role_of({"type": "session"}) == "other"
        assert pi.PI_SCHEMA.text_of(_msg("user", "hi")) == "hi"
        assert pi.PI_SCHEMA.total_tokens_of is pi._total_tokens_of


# ---------------------------------------------------------------------------
# inspect_session_pi (smoke test against real run_inspection)
# ---------------------------------------------------------------------------


class TestInspectSessionPi:
    def test_inspect_prints_report(self, tmp_path, capsys):
        path = _write_jsonl(
            tmp_path,
            [
                {"type": "session", "id": "x"},
                _msg("user", "hello"),
                _msg("assistant", "hi"),
            ],
        )
        inspect_session_pi(path)
        out = capsys.readouterr().out
        assert "PI" in out
        assert "hello" in out

    def test_inspect_show_flow(self, tmp_path, capsys):
        path = _write_jsonl(tmp_path, [_msg("user", "flowtext")])
        inspect_session_pi(path, show_flow=True)
        out = capsys.readouterr().out
        assert "flowtext" in out

    def test_inspect_uses_load_session(self, tmp_path, monkeypatch, recwarn=None):
        path = _write_jsonl(tmp_path, [_msg("user", "x")])
        calls = {"n": 0}
        real_load = pi.load_session_pi

        def spy(p):
            calls["n"] += 1
            return real_load(p)

        monkeypatch.setattr(pi, "load_session_pi", spy)
        inspect_session_pi(path)
        assert calls["n"] == 1
