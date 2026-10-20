from __future__ import annotations

import base64
import io
import json

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path, records, name="session.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _make_png_base64() -> str:
    PIL = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    PIL.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _user_record(text):
    return {"type": "message", "message": {"role": "user", "content": text}}


def _assistant_record(text):
    return {"type": "message", "message": {"role": "assistant", "content": text}}


# ---------------------------------------------------------------------------
# load_session_pi
# ---------------------------------------------------------------------------

class TestLoadSessionPi:
    def test_loads_records(self, tmp_path):
        path = _write_jsonl(tmp_path, [
            {"type": "session", "id": "abc"},
            {"type": "message", "message": {"role": "user", "content": "hi"}},
        ])
        records = load_session_pi(path)
        assert records == [
            {"type": "session", "id": "abc"},
            {"type": "message", "message": {"role": "user", "content": "hi"}},
        ]

    def test_accepts_str_path(self, tmp_path):
        path = _write_jsonl(tmp_path, [{"type": "session"}])
        assert load_session_pi(str(path)) == [{"type": "session"}]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session_pi(tmp_path / "nope.jsonl")

    def test_bad_extension_raises(self, tmp_path):
        path = tmp_path / "session.txt"
        path.write_text('{"type": "session"}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            load_session_pi(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session_pi(path)

    def test_invalid_json_lines_skipped(self, tmp_path):
        path = tmp_path / "partial.jsonl"
        path.write_text(
            '{"type": "session"}\n'
            'not json\n'
            '{"type": "message", "message": {"role": "user", "content": "x"}}\n',
            encoding="utf-8",
        )
        records = load_session_pi(path)
        assert len(records) == 2
        assert records[0] == {"type": "session"}
        assert records[1]["type"] == "message"


# ---------------------------------------------------------------------------
# _block_to_text
# ---------------------------------------------------------------------------

class TestBlockToText:
    def test_text(self):
        assert _block_to_text({"type": "text", "text": "hello"}) == "hello"

    def test_text_missing_key(self):
        assert _block_to_text({"type": "text"}) == ""

    def test_thinking(self):
        assert _block_to_text({"type": "thinking", "thinking": "pondering"}) == \
            "<thinking>pondering</thinking>"

    def test_tool_call(self):
        out = _block_to_text({
            "type": "toolCall",
            "name": "read_file",
            "arguments": {"path": "a.txt"},
        })
        assert out == "<tool_use name='read_file'>{\"path\": \"a.txt\"}</tool_use>"

    def test_tool_call_default_args(self):
        out = _block_to_text({"type": "toolCall", "name": "noop"})
        assert out == "<tool_use name='noop'>{}</tool_use>"

    def test_image(self):
        assert _block_to_text({"type": "image"}) == "<image>"

    def test_unknown_type(self):
        assert _block_to_text({"type": "weird"}) == ""

    def test_no_type(self):
        assert _block_to_text({}) == ""


# ---------------------------------------------------------------------------
# _decode_pi_image
# ---------------------------------------------------------------------------

class TestDecodePiImage:
    def test_empty_data_returns_none_and_warns(self, capsys):
        assert _decode_pi_image({"type": "image", "data": ""}) is None
        err = capsys.readouterr().err
        assert "empty base64 data" in err

    def test_missing_data_returns_none(self, capsys):
        assert _decode_pi_image({"type": "image"}) is None
        err = capsys.readouterr().err
        assert "empty base64 data" in err

    def test_valid_image_decoded(self):
        pytest.importorskip("PIL.Image")
        b64 = _make_png_base64()
        img = _decode_pi_image({"type": "image", "data": b64, "mimeType": "image/png"})
        assert img is not None
        assert img.size == (2, 2)

    def test_undecodable_returns_none_and_warns(self, capsys):
        pytest.importorskip("PIL.Image")
        out = _decode_pi_image({"type": "image", "data": "not-valid-base64!!!"})
        assert out is None
        err = capsys.readouterr().err
        assert "Skipping unreadable image block" in err


# ---------------------------------------------------------------------------
# _message_of / _record_role
# ---------------------------------------------------------------------------

class TestMessageAndRole:
    def test_message_of_non_message(self):
        assert _message_of({"type": "session"}) is None

    def test_message_of_message(self):
        rec = {"type": "message", "message": {"role": "user"}}
        assert _message_of(rec) == {"role": "user"}

    def test_message_of_null_message(self):
        assert _message_of({"type": "message", "message": None}) == {}

    def test_role_user(self):
        assert _record_role(_user_record("hi")) == "user"

    def test_role_assistant(self):
        assert _record_role(_assistant_record("yo")) == "assistant"

    def test_role_tool_result_maps_to_user(self):
        rec = {"type": "message", "message": {"role": "toolResult"}}
        assert _record_role(rec) == "user"

    def test_role_other_returns_none(self):
        assert _record_role({"type": "message", "message": {"role": "system"}}) is None

    def test_role_non_message_returns_none(self):
        assert _record_role({"type": "session"}) is None


# ---------------------------------------------------------------------------
# _text_of
# ---------------------------------------------------------------------------

class TestTextOf:
    def test_plain_string_content(self):
        assert _text_of(_user_record("hello world")) == "hello world"

    def test_list_text_blocks(self):
        rec = {"type": "message", "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"},
            ],
        }}
        assert _text_of(rec) == "line one\nline two"

    def test_blank_blocks_filtered(self):
        rec = {"type": "message", "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "keep"},
                {"type": "text", "text": "   "},
            ],
        }}
        assert _text_of(rec) == "keep"

    def test_non_dict_blocks_skipped(self):
        rec = {"type": "message", "message": {
            "role": "user",
            "content": ["not a dict", {"type": "text", "text": "ok"}],
        }}
        assert _text_of(rec) == "ok"

    def test_tool_result_formatting(self):
        rec = {"type": "message", "message": {
            "role": "toolResult",
            "toolName": "read",
            "toolCallId": "c1",
            "content": [{"type": "text", "text": "contents"}],
        }}
        assert _text_of(rec) == "<tool_result name='read' id='c1'>contents</tool_result>"

    def test_tool_result_with_image_placeholder(self):
        rec = {"type": "message", "message": {
            "role": "toolResult",
            "toolName": "shot",
            "toolCallId": "c2",
            "content": [
                {"type": "text", "text": "img:"},
                {"type": "image"},
            ],
        }}
        assert _text_of(rec) == "<tool_result name='shot' id='c2'>img:\n<image></tool_result>"

    def test_tool_result_empty_content(self):
        rec = {"type": "message", "message": {
            "role": "toolResult",
            "toolName": "n",
            "toolCallId": "i",
            "content": [],
        }}
        assert _text_of(rec) == "<tool_result name='n' id='i'></tool_result>"

    def test_unknown_role_returns_content_string(self):
        rec = {"type": "message", "message": {"role": "system", "content": "sys"}}
        assert _text_of(rec) == "sys"

    def test_non_string_non_list_content(self):
        rec = {"type": "message", "message": {"role": "user", "content": 42}}
        assert _text_of(rec) == ""

    def test_non_message_record(self):
        assert _text_of({"type": "session"}) == ""


# ---------------------------------------------------------------------------
# _message_parts
# ---------------------------------------------------------------------------

class TestMessageParts:
    def test_string_content(self):
        assert _message_parts(_user_record("hi")) == [{"type": "text", "text": "hi"}]

    def test_blank_string_content_empty_parts(self):
        assert _message_parts(_user_record("   ")) == []

    def test_list_text_blocks(self):
        rec = {"type": "message", "message": {
            "role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        }}
        assert _message_parts(rec) == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_empty_list(self):
        rec = {"type": "message", "message": {"role": "user", "content": []}}
        assert _message_parts(rec) == []

    def test_non_str_non_list_content(self):
        rec = {"type": "message", "message": {"role": "user", "content": 5}}
        assert _message_parts(rec) == []

    def test_tool_result_without_images_matches_text_of(self):
        rec = {"type": "message", "message": {
            "role": "toolResult",
            "toolName": "n",
            "toolCallId": "i",
            "content": [{"type": "text", "text": "x"}],
        }}
        parts = _message_parts(rec)
        assert parts == [{"type": "text", "text": _text_of(rec)}]

    def test_tool_result_empty_no_images(self):
        rec = {"type": "message", "message": {
            "role": "toolResult",
            "toolName": "n",
            "toolCallId": "i",
            "content": [],
        }}
        assert _message_parts(rec) == []

    def test_tool_result_with_image_multimodal(self):
        pytest.importorskip("PIL.Image")
        b64 = _make_png_base64()
        rec = {"type": "message", "message": {
            "role": "toolResult",
            "toolName": "shot",
            "toolCallId": "c",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "data": b64},
                {"type": "text", "text": "   "},
            ],
        }}
        parts = _message_parts(rec)
        assert parts[0] == {"type": "text", "text": "<tool_result name='shot' id='c'>"}
        assert parts[1] == {"type": "text", "text": "before"}
        assert parts[2]["type"] == "image"
        assert parts[2]["image"].size == (2, 2)
        assert parts[3] == {"type": "text", "text": "</tool_result>"}

    def test_tool_result_with_undecodable_image_skipped(self, capsys):
        pytest.importorskip("PIL.Image")
        rec = {"type": "message", "message": {
            "role": "toolResult",
            "toolName": "shot",
            "toolCallId": "c",
            "content": [{"type": "image", "data": "bad!!!"}],
        }}
        parts = _message_parts(rec)
        assert parts == [
            {"type": "text", "text": "<tool_result name='shot' id='c'>"},
            {"type": "text", "text": "</tool_result>"},
        ]
        assert "Skipping unreadable image block" in capsys.readouterr().err

    def test_tool_result_with_empty_image_data_warns(self, capsys):
        rec = {"type": "message", "message": {
            "role": "toolResult",
            "toolName": "t",
            "toolCallId": "i",
            "content": [{"type": "image", "data": ""}],
        }}
        parts = _message_parts(rec)
        assert parts == [
            {"type": "text", "text": "<tool_result name='t' id='i'>"},
            {"type": "text", "text": "</tool_result>"},
        ]
        assert "empty base64 data" in capsys.readouterr().err

    def test_user_message_with_image(self):
        pytest.importorskip("PIL.Image")
        b64 = _make_png_base64()
        rec = {"type": "message", "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "data": b64},
            ],
        }}
        parts = _message_parts(rec)
        assert parts[0] == {"type": "text", "text": "look"}
        assert parts[1]["type"] == "image"
        assert parts[1]["image"].size == (2, 2)

    def test_user_message_blank_text_and_image(self):
        pytest.importorskip("PIL.Image")
        b64 = _make_png_base64()
        rec = {"type": "message", "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "  "},
                {"type": "image", "data": b64},
            ],
        }}
        parts = _message_parts(rec)
        assert len(parts) == 1
        assert parts[0]["type"] == "image"

    def test_non_dict_entry_in_list_skipped(self):
        rec = {"type": "message", "message": {
            "role": "user",
            "content": ["bogus", {"type": "text", "text": "ok"}],
        }}
        assert _message_parts(rec) == [{"type": "text", "text": "ok"}]


# ---------------------------------------------------------------------------
# records_to_messages_pi
# ---------------------------------------------------------------------------

class TestRecordsToMessages:
    def test_empty(self):
        assert records_to_messages_pi([]) == []

    def test_metadata_only_ignored(self):
        records = [{"type": "session"}, {"type": "model_change"}]
        assert records_to_messages_pi(records) == []

    def test_basic_alternating(self):
        records = [
            _user_record("hi"),
            _assistant_record("hello"),
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_consecutive_same_role_merged(self):
        records = [_user_record("one"), _user_record("two")]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "one\ntwo"},
        ]

    def test_empty_content_records_skipped(self):
        records = [
            _user_record("hi"),
            _user_record("   "),
            _assistant_record("bye"),
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "bye"},
        ]

    def test_tool_result_becomes_user(self):
        records = [
            {"type": "message", "message": {
                "role": "toolResult",
                "toolName": "read",
                "toolCallId": "c",
                "content": [{"type": "text", "text": "data"}],
            }},
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "<tool_result name='read' id='c'>data</tool_result>"},
        ]

    def test_multimodal_content_preserved(self):
        pytest.importorskip("PIL.Image")
        b64 = _make_png_base64()
        records = [{"type": "message", "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "pic"},
                {"type": "image", "data": b64},
            ],
        }}]
        out = records_to_messages_pi(records)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        content = out[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "pic"}
        assert content[1]["type"] == "image"
        assert content[1]["image"].size == (2, 2)

    def test_merge_adjacent_text_parts(self):
        records = [_user_record("one"), _user_record("two")]
        out = records_to_messages_pi(records)
        assert out == [{"role": "user", "content": "one\ntwo"}]


# ---------------------------------------------------------------------------
# session_to_messages_pi
# ---------------------------------------------------------------------------

class TestSessionToMessages:
    def test_no_output(self, tmp_path):
        path = _write_jsonl(tmp_path, [_user_record("hi")])
        out = session_to_messages_pi(path)
        assert out == [{"role": "user", "content": "hi"}]

    def test_output_written(self, tmp_path):
        path = _write_jsonl(tmp_path, [
            _user_record("hi"),
            _assistant_record("yo"),
        ])
        out_path = tmp_path / "nested" / "out.json"
        result = session_to_messages_pi(path, output=out_path)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        assert out_path.exists()
        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert written == result

    def test_output_with_image_serialized_as_data_uri(self, tmp_path):
        pytest.importorskip("PIL.Image")
        b64 = _make_png_base64()
        path = _write_jsonl(tmp_path, [{"type": "message", "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "data": b64},
            ],
        }}])
        out_path = tmp_path / "out.json"
        result = session_to_messages_pi(path, output=out_path)
        # in-memory keeps PIL image
        assert result[0]["content"][1]["type"] == "image"
        assert result[0]["content"][1]["image"].size == (2, 2)

        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert written[0]["content"][1]["type"] == "image"
        assert written[0]["content"][1]["image"].startswith("data:image/")


# ---------------------------------------------------------------------------
# Inspector helpers
# ---------------------------------------------------------------------------

class TestInspectorHelpers:
    def test_record_type_message(self):
        assert _record_type_of(_user_record("hi")) == "message:user"

    def test_record_type_message_default_role(self):
        assert _record_type_of({"type": "message"}) == "message:unknown"

    def test_record_type_other(self):
        assert _record_type_of({"type": "session"}) == "session"

    def test_record_type_missing(self):
        assert _record_type_of({}) == "unknown"

    def test_timestamp_of(self):
        rec = {"timestamp": "2024-01-02T03:04:05.123456Z"}
        assert _timestamp_of(rec) == "2024-01-02 03:04:05"

    def test_timestamp_missing(self):
        assert _timestamp_of({}) == ""

    def test_total_tokens_empty(self):
        assert _total_tokens_of([{"type": "session"}]) == {}

    def test_total_tokens_sum(self):
        records = [
            {"type": "message", "message": {"role": "assistant", "usage": {
                "input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 1,
            }}},
            {"type": "message", "message": {"role": "assistant", "usage": {
                "input": 3, "output": 4, "cacheRead": 0, "cacheWrite": 1,
            }}},
        ]
        assert _total_tokens_of(records) == {
            "input": 13, "output": 9, "cache_read": 2, "cache_write": 2,
        }

    def test_total_tokens_null_usage_ignored(self):
        records = [
            {"type": "message", "message": {"role": "user", "usage": None}},
        ]
        assert _total_tokens_of(records) == {}

    def test_schema_fields(self):
        assert PI_SCHEMA.label == "PI"
        assert PI_SCHEMA.is_message(_user_record("hi")) is True
        assert PI_SCHEMA.is_message({"type": "session"}) is False
        assert PI_SCHEMA.role_of(_user_record("hi")) == "user"
        assert PI_SCHEMA.role_of({"type": "session"}) == "other"


# ---------------------------------------------------------------------------
# inspect_session_pi
# ---------------------------------------------------------------------------

class TestInspectSessionPi:
    def test_prints_report(self, tmp_path, capsys):
        path = _write_jsonl(tmp_path, [
            {"type": "session", "id": "s1"},
            _user_record("hello there"),
            _assistant_record("response"),
        ])
        inspect_session_pi(path)
        out = capsys.readouterr().out
        assert "PI" in out

    def test_with_flow_and_blocks(self, tmp_path, capsys):
        path = _write_jsonl(tmp_path, [
            _user_record("hello"),
            _assistant_record("world"),
        ])
        inspect_session_pi(path, show_flow=True, show_blocks=True)
        out = capsys.readouterr().out
        assert "hello" in out
        assert "world" in out

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_session_pi(tmp_path / "nope.jsonl")
