"""Unit tests for claude_converter.pi."""

from __future__ import annotations

import base64
import io
import json
import sys
import types
from pathlib import Path

import pytest

from claude_converter import pi
from claude_converter.pi import (
    load_session_pi,
    records_to_messages_pi,
    session_to_messages_pi,
    inspect_session_pi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_base64() -> str:
    """Create a tiny valid PNG encoded as base64 (no network, deterministic)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _msg(role: str, content, **extra) -> dict:
    message = {"role": role, "content": content}
    message.update(extra)
    return {"type": "message", "message": message}


def _user(text: str) -> dict:
    return _msg("user", text)


def _assistant(text: str) -> dict:
    return _msg("assistant", text)


def _write_jsonl(tmp_path: Path, records: list[dict], name: str = "session.jsonl") -> Path:
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _pil_available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# load_session_pi
# ---------------------------------------------------------------------------

class TestLoadSessionPi:
    def test_loads_records(self, tmp_path):
        records = [{"type": "session"}, _user("hi"), _assistant("hello")]
        path = _write_jsonl(tmp_path, records)
        assert load_session_pi(path) == records

    def test_accepts_str_path(self, tmp_path):
        records = [_user("hi")]
        path = _write_jsonl(tmp_path, records)
        assert load_session_pi(str(path)) == records

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session_pi(tmp_path / "nope.jsonl")

    def test_bad_extension_raises(self, tmp_path):
        path = tmp_path / "session.txt"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session_pi(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session_pi(path)

    def test_skips_invalid_json_lines(self, tmp_path, capsys):
        path = tmp_path / "mixed.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_user("ok")) + "\n")
            f.write("{bad json\n")
            f.write(json.dumps(_assistant("fine")) + "\n")
        records = load_session_pi(path)
        assert records == [_user("ok"), _assistant("fine")]
        assert "is invalid json" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Basic content handling via records_to_messages_pi
# ---------------------------------------------------------------------------

class TestRecordsToMessagesBasic:
    def test_string_content(self):
        records = [_user("hello"), _assistant("world")]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_text_blocks_joined(self):
        records = [
            _msg("assistant", [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ])
        ]
        assert records_to_messages_pi(records) == [
            {"role": "assistant", "content": "a\nb"},
        ]

    def test_thinking_block(self):
        records = [
            _msg("assistant", [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"},
            ])
        ]
        assert records_to_messages_pi(records) == [
            {"role": "assistant", "content": "<thinking>hmm</thinking>\nanswer"},
        ]

    def test_tool_call_block(self):
        records = [
            _msg("assistant", [
                {"type": "toolCall", "name": "read", "arguments": {"path": "x"}},
            ])
        ]
        out = records_to_messages_pi(records)
        assert len(out) == 1
        expected_args = json.dumps({"path": "x"}, ensure_ascii=False)
        assert out[0] == {
            "role": "assistant",
            "content": f"<tool_use name='read'>{expected_args}</tool_use>",
        }

    def test_tool_call_block_empty_arguments(self):
        records = [
            _msg("assistant", [
                {"type": "toolCall", "name": "noop"},
            ])
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {"role": "assistant", "content": "<tool_use name='noop'>{}</tool_use>"},
        ]

    def test_image_block_with_empty_data_skipped(self, capsys):
        records = [_msg("user", [{"type": "image", "data": "", "mimeType": "image/png"}])]
        out = records_to_messages_pi(records)
        assert out == []
        assert "empty base64" in capsys.readouterr().err

    def test_unknown_block_type_skipped(self):
        records = [
            _msg("user", [
                {"type": "mystery", "x": 1},
                {"type": "text", "text": "kept"},
            ])
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "kept"},
        ]

    def test_non_message_records_ignored(self):
        records = [
            {"type": "session", "id": "abc"},
            _user("hi"),
            {"type": "model_change", "model": "x"},
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "hi"},
        ]

    def test_unknown_role_ignored(self):
        records = [
            _msg("system", "ignore me"),
            _user("keep"),
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "keep"},
        ]

    def test_missing_message_null_safe(self):
        records = [{"type": "message", "message": None}, _user("ok")]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "ok"},
        ]

    def test_empty_content_skipped(self):
        records = [
            _msg("user", ""),
            _msg("assistant", "   "),
            _msg("user", []),
            _user("real"),
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "real"},
        ]

    def test_blank_text_blocks_dropped(self):
        records = [
            _msg("user", [
                {"type": "text", "text": ""},
                {"type": "text", "text": "  "},
            ])
        ]
        assert records_to_messages_pi(records) == []

    def test_non_dict_blocks_ignored(self):
        records = [
            _msg("user", [
                "not-a-dict",
                {"type": "text", "text": "ok"},
            ])
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "ok"},
        ]

    def test_empty_records_list(self):
        assert records_to_messages_pi([]) == []

    def test_non_string_non_list_content_returns_empty(self):
        records = [_msg("user", 12345), _user("real")]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "real"},
        ]


# ---------------------------------------------------------------------------
# Consecutive same-role merging
# ---------------------------------------------------------------------------

class TestRecordsToMessagesMerging:
    def test_consecutive_user_merged(self):
        records = [_user("one"), _user("two")]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "one\ntwo"},
        ]

    def test_alternating_roles_kept_separate(self):
        records = [_user("u"), _assistant("a"), _user("u2")]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u2"},
        ]

    def test_three_consecutive_merged(self):
        records = [_assistant("a"), _assistant("b"), _assistant("c")]
        assert records_to_messages_pi(records) == [
            {"role": "assistant", "content": "a\nb\nc"},
        ]


# ---------------------------------------------------------------------------
# toolResult handling
# ---------------------------------------------------------------------------

class TestToolResults:
    def test_tool_result_text_only(self):
        records = [
            _msg(
                "toolResult",
                [{"type": "text", "text": "output"}],
                toolName="bash",
                toolCallId="call-1",
            )
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {
                "role": "user",
                "content": "<tool_result name='bash' id='call-1'>output</tool_result>",
            },
        ]

    def test_tool_result_multiple_text_blocks_joined(self):
        records = [
            _msg(
                "toolResult",
                [
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ],
                toolName="t",
                toolCallId="id",
            )
        ]
        out = records_to_messages_pi(records)
        assert out[0]["role"] == "user"
        assert "line1\nline2" in out[0]["content"]

    def test_tool_result_empty_text_still_wraps(self):
        records = [
            _msg("toolResult", [], toolName="t", toolCallId="id"),
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {"role": "user", "content": "<tool_result name='t' id='id'></tool_result>"},
        ]

    def test_tool_result_image_empty_data_warns(self, capsys):
        records = [
            _msg(
                "toolResult",
                [
                    {"type": "text", "text": "see image"},
                    {"type": "image", "data": "", "mimeType": "image/png"},
                ],
                toolName="t",
                toolCallId="id",
            )
        ]
        out = records_to_messages_pi(records)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        content = out[0]["content"]
        assert isinstance(content, str)
        assert content.startswith("<tool_result name='t' id='id'>")
        assert content.endswith("</tool_result>")
        assert "see image" in content
        assert "Skipping image" in capsys.readouterr().err

    def test_tool_result_tool_result_role_maps_to_user(self):
        records = [_assistant("a"), _msg("toolResult", "x", toolName="t", toolCallId="i")]
        out = records_to_messages_pi(records)
        assert [m["role"] for m in out] == ["assistant", "user"]

    def test_tool_result_blank_text_only_no_image_yields_empty(self):
        # No image -> collapses to _text_of output, which is non-empty wrapper.
        records = [
            _msg("toolResult", [{"type": "text", "text": "  "}], toolName="t", toolCallId="id"),
        ]
        out = records_to_messages_pi(records)
        # _text_of drops blank text parts, producing an empty wrapper, which
        # .strip() is non-empty (has tags), so it is kept.
        assert out == [
            {"role": "user", "content": "<tool_result name='t' id='id'></tool_result>"},
        ]

    def test_tool_result_no_tool_name_or_id(self):
        records = [
            _msg("toolResult", [{"type": "text", "text": "x"}]),
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {"role": "user", "content": "<tool_result name='' id=''>x</tool_result>"},
        ]


# ---------------------------------------------------------------------------
# Image decoding paths
# ---------------------------------------------------------------------------

class TestImages:
    def test_user_image_kept_as_multimodal_part(self):
        if not _pil_available():
            pytest.skip("PIL not installed")
        b64 = _make_png_base64()
        records = [
            _msg("user", [
                {"type": "text", "text": "look"},
                {"type": "image", "data": b64, "mimeType": "image/png"},
            ])
        ]
        out = records_to_messages_pi(records)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        content = out[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "look"}
        assert content[1]["type"] == "image"
        from PIL import Image
        assert isinstance(content[1]["image"], Image.Image)

    def test_data_uri_image(self):
        if not _pil_available():
            pytest.skip("PIL not installed")
        b64 = _make_png_base64()
        uri = "data:image/png;base64," + b64
        records = [_msg("user", [{"type": "image", "data": uri, "mimeType": "image/png"}])]
        out = records_to_messages_pi(records)
        assert isinstance(out[0]["content"], list)
        assert out[0]["content"][0]["type"] == "image"

    def test_undecodable_image_warns_and_skips(self, capsys):
        records = [_msg("user", [{"type": "image", "data": "!!!not base64!!!", "mimeType": "image/png"}])]
        out = records_to_messages_pi(records)
        assert out == []
        assert "Skipping" in capsys.readouterr().err

    def test_image_partial_when_undecodable_with_text(self, capsys):
        records = [
            _msg("user", [
                {"type": "text", "text": "before"},
                {"type": "image", "data": "ffff", "mimeType": "image/png"},
            ])
        ]
        out = records_to_messages_pi(records)
        assert out == [{"role": "user", "content": "before"}]
        assert "Skipping" in capsys.readouterr().err

    def _run_without_pil(self, records, capsys):
        """Force convert_base64_to_pil_image's ImportError branch by removing PIL.

        Temporarily removes any 'PIL' modules from sys.modules and blocks a
        fresh import so convert_base64_to_pil_image raises ImportError.
        """
        saved = {k: sys.modules[k] for k in list(sys.modules) if k == "PIL" or k.startswith("PIL.")}
        for k in saved:
            del sys.modules[k]
        sys.modules["PIL"] = None  # type: ignore[assignment]
        try:
            out = records_to_messages_pi(records)
        finally:
            if "PIL" in sys.modules:
                del sys.modules["PIL"]
            sys.modules.update(saved)
        return out

    def test_image_requires_pillow_warns(self, capsys):
        # Without Pillow installed, a valid-looking non-empty base64 payload
        # causes convert_base64_to_pil_image to raise ImportError, which
        # _decode_pi_image catches and prints a warning for.
        records = [_msg("user", [{"type": "image", "data": "AAAA", "mimeType": "image/png"}])]
        out = self._run_without_pil(records, capsys)
        assert out == []
        assert "Skipping unreadable image block" in capsys.readouterr().err

    def test_tool_result_with_image_kept_multimodal(self):
        if not _pil_available():
            pytest.skip("PIL not installed")
        b64 = _make_png_base64()
        records = [
            _msg(
                "toolResult",
                [
                    {"type": "text", "text": "here"},
                    {"type": "image", "data": b64, "mimeType": "image/png"},
                ],
                toolName="t",
                toolCallId="id",
            )
        ]
        out = records_to_messages_pi(records)
        assert len(out) == 1
        content = out[0]["content"]
        assert isinstance(content, list)
        texts = [p["text"] for p in content if p["type"] == "text"]
        assert texts[0] == "<tool_result name='t' id='id'>"
        assert "here" in texts
        assert texts[-1] == "</tool_result>"
        assert any(p["type"] == "image" for p in content)


# ---------------------------------------------------------------------------
# session_to_messages_pi
# ---------------------------------------------------------------------------

class TestSessionToMessagesPi:
    def test_round_trip_no_output(self, tmp_path):
        records = [_user("hi"), _assistant("hello")]
        path = _write_jsonl(tmp_path, records)
        out = session_to_messages_pi(path)
        assert out == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_writes_output_file(self, tmp_path):
        records = [_user("hi"), _assistant("hello")]
        path = _write_jsonl(tmp_path, records)
        out_path = tmp_path / "sub" / "out.json"
        result = session_to_messages_pi(path, output=out_path)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert out_path.exists()
        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert written == result

    def test_output_encodes_images(self, tmp_path):
        if not _pil_available():
            pytest.skip("PIL not installed")
        b64 = _make_png_base64()
        records = [
            _msg("user", [
                {"type": "text", "text": "see"},
                {"type": "image", "data": b64, "mimeType": "image/png"},
            ])
        ]
        path = _write_jsonl(tmp_path, records)
        out_path = tmp_path / "out.json"
        result = session_to_messages_pi(path, output=out_path)
        from PIL import Image
        assert isinstance(result[0]["content"][1]["image"], Image.Image)
        written = json.loads(out_path.read_text(encoding="utf-8"))
        image_part = written[0]["content"][1]
        assert image_part["type"] == "image"
        assert isinstance(image_part["image"], str)
        assert image_part["image"].startswith("data:image/")

    def test_propagates_load_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            session_to_messages_pi(tmp_path / "missing.jsonl")

    def test_output_accepts_str_path(self, tmp_path):
        records = [_user("hi")]
        path = _write_jsonl(tmp_path, records)
        out_path = tmp_path / "out.json"
        session_to_messages_pi(path, output=str(out_path))
        assert json.loads(out_path.read_text(encoding="utf-8")) == [
            {"role": "user", "content": "hi"},
        ]


# ---------------------------------------------------------------------------
# inspect_session_pi
# ---------------------------------------------------------------------------

class TestInspectSessionPi:
    def test_prints_report(self, tmp_path, capsys):
        records = [
            {"type": "session", "id": "s1", "timestamp": "2024-01-01T00:00:00Z"},
            _user("hello"),
            _assistant("world"),
        ]
        path = _write_jsonl(tmp_path, records)
        inspect_session_pi(path)
        out = capsys.readouterr().out
        assert "PI" in out

    def test_show_flow_prints_text(self, tmp_path, capsys):
        records = [_user("hi there"), _assistant("yo back")]
        path = _write_jsonl(tmp_path, records)
        inspect_session_pi(path, show_flow=True, show_blocks=True, show_raw=True)
        out = capsys.readouterr().out
        assert "hi there" in out
        assert "yo back" in out

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_session_pi(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------------------
# PI_SCHEMA / internal helpers via public surface
# ---------------------------------------------------------------------------

class TestSchemaHelpers:
    def test_schema_label(self):
        assert pi.PI_SCHEMA.label == "PI"

    def test_record_type_of(self):
        assert pi._record_type_of({"type": "session"}) == "session"
        assert pi._record_type_of({"type": "message", "message": {"role": "user"}}) == "message:user"
        assert pi._record_type_of({}) == "unknown"
        assert pi._record_type_of({"type": "message", "message": None}) == "message:unknown"

    def test_timestamp_of(self):
        assert pi._timestamp_of({"timestamp": "2024-01-02T03:04:05.123Z"}) == "2024-01-02 03:04:05"
        assert pi._timestamp_of({}) == ""

    def test_total_tokens_of(self):
        records = [
            {"type": "message", "message": {"role": "user", "usage": {
                "input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 1}}},
            {"type": "message", "message": {"role": "assistant", "usage": {
                "input": 3, "output": 7}}},
            {"type": "session"},
        ]
        assert pi._total_tokens_of(records) == {
            "input": 13, "output": 12, "cache_read": 2, "cache_write": 1,
        }

    def test_total_tokens_of_no_usage(self):
        assert pi._total_tokens_of([{"type": "session"}, {"type": "message", "message": {}}]) == {}

    def test_total_tokens_of_empty_records(self):
        assert pi._total_tokens_of([]) == {}

    def test_is_message_and_role(self):
        schema = pi.PI_SCHEMA
        assert schema.is_message(_user("x")) is True
        assert schema.is_message({"type": "session"}) is False
        assert schema.role_of(_user("x")) == "user"
        assert schema.role_of({"type": "session"}) == "other"

    def test_schema_text_of(self):
        assert pi.PI_SCHEMA.text_of(_user("hello")) == "hello"

    def test_schema_text_of_tool_result(self):
        rec = _msg("toolResult", [{"type": "text", "text": "out"}],
                   toolName="t", toolCallId="id")
        assert pi.PI_SCHEMA.text_of(rec) == "<tool_result name='t' id='id'>out</tool_result>"

    def test_schema_text_of_tool_result_with_image(self):
        rec = _msg("toolResult", [
            {"type": "text", "text": "out"},
            {"type": "image", "data": "x"},
        ], toolName="t", toolCallId="id")
        assert pi.PI_SCHEMA.text_of(rec) == "<tool_result name='t' id='id'>out\n<image></tool_result>"

    def test_record_role_tool_result(self):
        assert pi._record_role(_msg("toolResult", "x")) == "user"

    def test_record_role_unknown(self):
        assert pi._record_role(_msg("system", "x")) is None

    def test_record_role_non_message(self):
        assert pi._record_role({"type": "session"}) is None

    def test_block_to_text_unknown(self):
        assert pi._block_to_text({"type": "mystery"}) == ""

    def test_block_to_text_image_placeholder(self):
        assert pi._block_to_text({"type": "image"}) == "<image>"

    def test_block_to_text_text(self):
        assert pi._block_to_text({"type": "text", "text": "abc"}) == "abc"

    def test_block_to_text_thinking(self):
        assert pi._block_to_text({"type": "thinking", "thinking": "hm"}) == "<thinking>hm</thinking>"

    def test_block_to_text_tool_call_unicode(self):
        out = pi._block_to_text({"type": "toolCall", "name": "n", "arguments": {"k": "é"}})
        assert out == "<tool_use name='n'>" + json.dumps({"k": "é"}, ensure_ascii=False) + "</tool_use>"

    def test_message_of_non_message(self):
        assert pi._message_of({"type": "session"}) is None

    def test_message_of_missing(self):
        assert pi._message_of({"type": "message"}) == {}
