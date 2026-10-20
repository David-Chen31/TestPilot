"""Tests for claude_converter.claude — public API and internal helpers."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

import claude_converter.claude as claude_mod
from claude_converter.claude import (
    load_session,
    records_to_messages,
    session_to_messages,
    inspect_session,
    _block_to_str,
    _block_to_parts,
    _content_to_str,
    _content_to_parts,
    _collect_block_types,
    _decode_claude_image,
    _render_block,
    _record_type_of,
    _is_message,
    _role_of,
    _text_of,
    _timestamp_of,
    _tokens_of,
    _block_types_of,
    _blocks_of,
    _total_tokens_of,
    _print_session_metadata,
    _print_raw_examples,
    CLAUDE_SCHEMA,
)
from claude_converter.utils import messages_to_json_safe


# ── helpers ──────────────────────────────────────────────────────────────

def write_jsonl(tmp_path: Path, records: list[dict], name: str = "session.jsonl") -> Path:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def user_record(content, session_id="sess-1", cwd="/home/x", branch=None):
    rec = {"type": "user", "message": {"role": "user", "content": content},
           "sessionId": session_id, "cwd": cwd}
    if branch is not None:
        rec["gitBranch"] = branch
    return rec


def assistant_record(content, usage=None):
    msg = {"role": "assistant", "content": content}
    if usage is not None:
        msg["usage"] = usage
    return {"type": "assistant", "message": msg}


def _fake_png_image():
    """Return a real 1x1 PIL image, or None if Pillow is unavailable."""
    try:
        from PIL import Image
    except ImportError:
        return None
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buf, format="PNG")
    return Image.open(io.BytesIO(buf.getvalue()))


# ── _block_to_str ────────────────────────────────────────────────────────

class TestBlockToStr:
    def test_text(self):
        assert _block_to_str({"type": "text", "text": "hello"}) == "hello"

    def test_text_missing(self):
        assert _block_to_str({"type": "text"}) == ""

    def test_thinking(self):
        assert _block_to_str({"type": "thinking", "thinking": "pondering"}) == \
            "<thinking>pondering</thinking>"

    def test_tool_use(self):
        out = _block_to_str({"type": "tool_use", "name": "Read", "input": {"file": "a"}})
        assert out == "<tool_use name='Read'>{\"file\": \"a\"}</tool_use>"

    def test_tool_use_non_ascii(self):
        out = _block_to_str({"type": "tool_use", "name": "x", "input": {"k": "ñ"}})
        assert "ñ" in out

    def test_tool_use_no_input(self):
        out = _block_to_str({"type": "tool_use", "name": "x"})
        assert out == "<tool_use name='x'>{}</tool_use>"

    def test_tool_result_string(self):
        assert _block_to_str({"type": "tool_result", "content": "ok"}) == \
            "<tool_result>ok</tool_result>"

    def test_tool_result_list_text(self):
        block = {"type": "tool_result",
                 "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        assert _block_to_str(block) == "<tool_result>a b</tool_result>"

    def test_tool_result_list_image(self):
        block = {"type": "tool_result",
                 "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}
        assert _block_to_str(block) == "<tool_result><image> hi</tool_result>"

    def test_tool_result_list_ignores_non_dict(self):
        block = {"type": "tool_result", "content": ["junk", {"type": "text", "text": "x"}]}
        assert _block_to_str(block) == "<tool_result>x</tool_result>"

    def test_image_block(self):
        assert _block_to_str({"type": "image"}) == "<image>"

    def test_unknown(self):
        assert _block_to_str({"type": "mystery"}) == ""

    def test_no_type(self):
        assert _block_to_str({}) == ""


# ── _decode_claude_image ─────────────────────────────────────────────────

class TestDecodeClaudeImage:
    def test_unsupported_source_type(self, capsys):
        result = _decode_claude_image({"source": {"type": "url"}})
        assert result is None
        err = capsys.readouterr().err
        assert "unsupported source type" in err
        assert "'url'" in err

    def test_empty_data(self, capsys):
        result = _decode_claude_image({"source": {"type": "base64", "data": ""}})
        assert result is None
        assert "empty base64 data" in capsys.readouterr().err

    def test_missing_data(self, capsys):
        result = _decode_claude_image({"source": {"type": "base64"}})
        assert result is None
        assert "empty base64 data" in capsys.readouterr().err

    def test_no_source(self, capsys):
        result = _decode_claude_image({})
        assert result is None
        # source={} -> type None != base64
        assert "unsupported source type" in capsys.readouterr().err

    def test_success(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(claude_mod, "convert_base64_to_pil_image", lambda d: sentinel)
        result = _decode_claude_image(
            {"source": {"type": "base64", "data": "abc", "media_type": "image/png"}})
        assert result is sentinel

    def test_value_error_returns_none(self, monkeypatch, capsys):
        def boom(_):
            raise ValueError("bad decode")
        monkeypatch.setattr(claude_mod, "convert_base64_to_pil_image", boom)
        result = _decode_claude_image({"source": {"type": "base64", "data": "abc"}})
        assert result is None
        err = capsys.readouterr().err
        assert "unreadable image block" in err
        assert "bad decode" in err

    def test_import_error_returns_none(self, monkeypatch, capsys):
        def boom(_):
            raise ImportError("no pillow")
        monkeypatch.setattr(claude_mod, "convert_base64_to_pil_image", boom)
        result = _decode_claude_image({"source": {"type": "base64", "data": "abc"}})
        assert result is None
        assert "unreadable image block" in capsys.readouterr().err

    def test_real_png_success(self):
        img = _fake_png_image()
        if img is None:
            pytest.skip("Pillow is not installed")
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow is not installed")
        # build the base64 string for a genuine PNG
        import base64
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), (0, 0, 255)).save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        result = _decode_claude_image(
            {"source": {"type": "base64", "data": data, "media_type": "image/png"}})
        assert isinstance(result, Image.Image)
        assert result.size == (2, 2)


# ── _block_to_parts ──────────────────────────────────────────────────────

class TestBlockToParts:
    def test_text_block(self):
        assert _block_to_parts({"type": "text", "text": "hi"}) == \
            [{"type": "text", "text": "hi"}]

    def test_text_whitespace_dropped(self):
        assert _block_to_parts({"type": "text", "text": "   "}) == []

    def test_image_block_success(self, monkeypatch):
        pil = object()
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: pil)
        assert _block_to_parts({"type": "image"}) == [{"type": "image", "image": pil}]

    def test_image_block_failure(self, monkeypatch):
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: None)
        assert _block_to_parts({"type": "image"}) == []

    def test_tool_result_no_image(self):
        block = {"type": "tool_result", "content": "result text"}
        assert _block_to_parts(block) == [{"type": "text", "text": "<tool_result>result text</tool_result>"}]

    def test_tool_result_empty_text(self):
        block = {"type": "tool_result", "content": ""}
        # _block_to_str -> "<tool_result></tool_result>" which has .strip()
        # true, so it returns a text part
        assert _block_to_parts(block) == [
            {"type": "text", "text": "<tool_result></tool_result>"}
        ]

    def test_tool_result_string_content_no_image(self):
        block = {"type": "tool_result", "content": "x"}
        parts = _block_to_parts(block)
        assert parts == [{"type": "text", "text": "<tool_result>x</tool_result>"}]

    def test_tool_result_with_image(self, monkeypatch):
        pil = object()
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: pil)
        block = {
            "type": "tool_result",
            "tool_use_id": "tu-1",
            "content": [
                {"type": "text", "text": "see"},
                {"type": "image"},
                {"type": "text", "text": "  "},  # whitespace -> dropped
            ],
        }
        parts = _block_to_parts(block)
        assert parts == [
            {"type": "text", "text": "<tool_result id='tu-1'>"},
            {"type": "text", "text": "see"},
            {"type": "image", "image": pil},
            {"type": "text", "text": "</tool_result>"},
        ]

    def test_tool_result_with_image_no_id(self, monkeypatch):
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: None)
        block = {"type": "tool_result", "content": [{"type": "image"}]}
        parts = _block_to_parts(block)
        assert parts[0] == {"type": "text", "text": "<tool_result id='?'>"}
        assert parts[-1] == {"type": "text", "text": "</tool_result>"}
        # no image part added because decode returned None
        assert len(parts) == 2

    def test_tool_result_with_image_skips_non_dict(self, monkeypatch):
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: object())
        block = {"type": "tool_result", "tool_use_id": "t",
                 "content": ["junk", {"type": "image"}]}
        parts = _block_to_parts(block)
        # opening + image + closing
        assert len(parts) == 3
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image"
        assert parts[2]["type"] == "text"

    def test_thinking_block_uses_block_to_str(self):
        assert _block_to_parts({"type": "thinking", "thinking": "x"}) == \
            [{"type": "text", "text": "<thinking>x</thinking>"}]

    def test_unknown_block(self):
        assert _block_to_parts({"type": "mystery"}) == []


# ── _content_to_str / _content_to_parts / _collect_block_types ──────────

class TestContentFunctions:
    def test_content_to_str_string(self):
        assert _content_to_str("hello") == "hello"

    def test_content_to_str_list(self):
        content = [{"type": "text", "text": "a"},
                   {"type": "thinking", "thinking": "b"},
                   "not-a-dict"]
        assert _content_to_str(content) == "a\n<thinking>b</thinking>"

    def test_content_to_str_none(self):
        assert _content_to_str(None) == ""

    def test_content_to_str_skips_empty(self):
        content = [{"type": "text", "text": ""},
                   {"type": "text", "text": "keep"}]
        assert _content_to_str(content) == "keep"

    def test_content_to_parts_string(self):
        assert _content_to_parts("hi") == [{"type": "text", "text": "hi"}]

    def test_content_to_parts_empty_string(self):
        assert _content_to_parts("   ") == []

    def test_content_to_parts_none(self):
        assert _content_to_parts(None) == []

    def test_content_to_parts_list(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _content_to_parts(content) == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_content_to_parts_skips_non_dict(self):
        assert _content_to_parts(["junk", {"type": "text", "text": "x"}]) == \
            [{"type": "text", "text": "x"}]

    def test_collect_block_types_string(self):
        assert _collect_block_types("hi") == ["raw_string"]

    def test_collect_block_types_list(self):
        content = [{"type": "text"}, {"type": "image"}, {}]
        assert _collect_block_types(content) == ["text", "image", "unknown"]

    def test_collect_block_types_skips_non_dict(self):
        assert _collect_block_types([{"type": "text"}, "x"]) == ["text"]

    def test_collect_block_types_none(self):
        assert _collect_block_types(None) == ["unknown"]


# ── _render_block ───────────────────────────────────────────────────────

class TestRenderBlock:
    def test_text_block(self):
        out = _render_block({"type": "text", "text": "hi"}, indent="")
        assert "text" in out
        assert "hi" in out

    def test_thinking_block(self):
        out = _render_block({"type": "thinking", "thinking": "deep"}, indent="")
        assert "thinking" in out
        assert "deep" in out

    def test_tool_use_block(self):
        out = _render_block({"type": "tool_use", "name": "Bash", "id": "t1",
                             "input": {"cmd": "ls"}}, indent="")
        assert "Bash" in out
        assert "t1" in out
        assert "cmd" in out

    def test_tool_result_string_content(self):
        out = _render_block({"type": "tool_result", "tool_use_id": "t2",
                             "content": "output"}, indent="")
        assert "t2" in out
        assert "output" in out

    def test_tool_result_list_content(self):
        block = {"type": "tool_result", "tool_use_id": "t3",
                 "content": [
                     {"type": "image", "source": {"media_type": "image/png", "data": "abcd"}},
                     {"type": "text", "text": "note"},
                 ]}
        out = _render_block(block, indent="")
        assert "image/png" in out
        assert "4 base64 chars" in out
        assert "note" in out

    def test_image_block(self):
        block = {"type": "image", "source": {"media_type": "image/jpeg", "data": "xyz"}}
        out = _render_block(block, indent="")
        assert "image/jpeg" in out
        assert "3 base64 chars" in out

    def test_unknown_block(self):
        out = _render_block({"type": "weird"}, indent="")
        assert "weird" in out


# ── Inspector schema helpers ────────────────────────────────────────────

class TestSchemaHelpers:
    def test_record_type_of(self):
        assert _record_type_of({"type": "user"}) == "user"
        assert _record_type_of({}) == "unknown"

    def test_is_message(self):
        assert _is_message({"type": "user"}) is True
        assert _is_message({"type": "assistant"}) is True
        assert _is_message({"type": "system"}) is False

    def test_role_of_from_message(self):
        assert _role_of({"type": "user", "message": {"role": "assistant"}}) == "assistant"

    def test_role_of_fallback(self):
        assert _role_of({"type": "user", "message": {}}) == "user"
        assert _role_of({"type": "user"}) == "user"

    def test_text_of(self):
        r = {"message": {"content": [{"type": "text", "text": "hi"}]}}
        assert _text_of(r) == "hi"

    def test_text_of_string(self):
        assert _text_of({"message": {"content": "plain"}}) == "plain"

    def test_timestamp_of(self):
        r = {"timestamp": "2024-01-02T03:04:05.123Z"}
        assert _timestamp_of(r) == "2024-01-02 03:04:05"

    def test_timestamp_of_missing(self):
        assert _timestamp_of({}) == ""

    def test_tokens_of(self):
        r = {"message": {"usage": {"output_tokens": 42}}}
        assert _tokens_of(r) == 42

    def test_tokens_of_missing(self):
        assert _tokens_of({}) == 0

    def test_block_types_of(self):
        r = {"message": {"content": [{"type": "text"}, {"type": "image"}]}}
        assert _block_types_of(r) == ["text", "image"]

    def test_blocks_of_list(self):
        r = {"message": {"content": [{"type": "text"}, "junk"]}}
        assert _blocks_of(r) == [{"type": "text"}]

    def test_blocks_of_non_list(self):
        assert _blocks_of({"message": {"content": "str"}}) == []

    def test_total_tokens_of(self):
        records = [
            {"message": {"usage": {"input_tokens": 10, "output_tokens": 5,
                                   "cache_read_input_tokens": 2,
                                   "cache_creation_input_tokens": 3}}},
            {"message": {"usage": {"input_tokens": 1, "output_tokens": 1}}},
        ]
        assert _total_tokens_of(records) == {
            "input": 11, "output": 6, "cache_read": 2, "cache_write": 3, "total": 17,
        }

    def test_total_tokens_all_zero(self):
        assert _total_tokens_of([{"message": {}}]) == {}


# ── CLAUDE_SCHEMA ───────────────────────────────────────────────────────

class TestClaudeSchema:
    def test_label(self):
        assert CLAUDE_SCHEMA.label == "CLAUDE CODE"

    def test_callables_bound(self):
        assert CLAUDE_SCHEMA.record_type_of is _record_type_of
        assert CLAUDE_SCHEMA.is_message is _is_message
        assert CLAUDE_SCHEMA.role_of is _role_of
        assert CLAUDE_SCHEMA.text_of is _text_of
        assert CLAUDE_SCHEMA.total_tokens_of is _total_tokens_of
        assert CLAUDE_SCHEMA.tokens_of is _tokens_of
        assert CLAUDE_SCHEMA.block_types_of is _block_types_of
        assert CLAUDE_SCHEMA.blocks_of is _blocks_of
        assert CLAUDE_SCHEMA.render_block is _render_block


# ── load_session ────────────────────────────────────────────────────────

class TestLoadSession:
    def test_loads_records(self, tmp_path):
        recs = [{"type": "user"}, {"type": "assistant"}]
        p = write_jsonl(tmp_path, recs)
        assert load_session(p) == recs

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nope.jsonl")

    def test_bad_extension(self, tmp_path):
        p = tmp_path / "session.txt"
        p.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(p)

    def test_skips_invalid_json_lines(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text('{"type":"user"}\nnot json\n{"type":"assistant"}\n', encoding="utf-8")
        result = load_session(p)
        assert result == [{"type": "user"}, {"type": "assistant"}]
        assert "invalid JSON" in capsys.readouterr().err


# ── records_to_messages ─────────────────────────────────────────────────

class TestRecordsToMessages:
    def test_skips_non_message_records(self):
        records = [
            {"type": "system", "message": {"content": "sys"}},
            user_record("hello"),
        ]
        result = records_to_messages(records)
        assert result == [{"role": "user", "content": "hello"}]

    def test_simple_user_assistant(self):
        records = [
            user_record("hi"),
            assistant_record("hello there"),
        ]
        result = records_to_messages(records)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]

    def test_role_from_message(self):
        records = [
            {"type": "user", "message": {"role": "assistant", "content": "x"}},
        ]
        assert records_to_messages(records) == [{"role": "assistant", "content": "x"}]

    def test_role_falls_back_to_record_type(self):
        records = [{"type": "user", "message": {"content": "x"}}]
        assert records_to_messages(records) == [{"role": "user", "content": "x"}]

    def test_empty_content_dropped(self):
        records = [user_record("")]
        assert records_to_messages(records) == []

    def test_empty_records(self):
        assert records_to_messages([]) == []

    def test_merges_consecutive_same_role(self):
        records = [
            user_record("first"),
            user_record("second"),
        ]
        result = records_to_messages(records)
        assert result == [{"role": "user", "content": "first\nsecond"}]

    def test_does_not_merge_different_roles(self):
        records = [user_record("u"), assistant_record("a")]
        assert records_to_messages(records) == [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]

    def test_tool_use_block_flattened(self):
        records = [assistant_record([
            {"type": "text", "text": "using tool"},
            {"type": "tool_use", "name": "Read", "input": {"path": "f"}},
        ])]
        result = records_to_messages(records)
        expected_content = (
            "using tool\n"
            "<tool_use name='Read'>{\"path\": \"f\"}</tool_use>"
        )
        assert result == [{"role": "assistant", "content": expected_content}]

    def test_image_kept_as_multimodal_part(self, monkeypatch):
        pil = object()
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: pil)
        records = [user_record([
            {"type": "text", "text": "look"},
            {"type": "image", "source": {}},
        ])]
        result = records_to_messages(records)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "look"}
        assert content[1] == {"type": "image", "image": pil}

    def test_image_failure_returns_string_when_only_text_remains(self, monkeypatch):
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: None)
        records = [user_record([
            {"type": "text", "text": "no image"},
            {"type": "image", "source": {}},
        ])]
        result = records_to_messages(records)
        assert result == [{"role": "user", "content": "no image"}]

    def test_null_message_handled(self):
        records = [{"type": "user", "message": None}]
        # content defaults to "", no parts -> skipped
        assert records_to_messages(records) == []

    def test_string_content(self):
        records = [user_record("plain text")]
        assert records_to_messages(records) == [{"role": "user", "content": "plain text"}]

    def test_merge_text_parts_joined_with_newline(self):
        records = [
            user_record([{"type": "text", "text": "a"}]),
            user_record([{"type": "text", "text": "b"}]),
        ]
        result = records_to_messages(records)
        assert result == [{"role": "user", "content": "a\nb"}]

    def test_tool_result_with_nested_image(self, monkeypatch):
        pil = object()
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: pil)
        records = [user_record([
            {"type": "tool_result", "tool_use_id": "tu", "content": [
                {"type": "text", "text": "here"},
                {"type": "image", "source": {}},
            ]},
        ])]
        result = records_to_messages(records)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "<tool_result id='tu'>"}
        assert content[1] == {"type": "text", "text": "here"}
        assert content[2] == {"type": "image", "image": pil}
        assert content[3] == {"type": "text", "text": "</tool_result>"}


# ── session_to_messages ─────────────────────────────────────────────────

class TestSessionToMessages:
    def test_basic_roundtrip(self, tmp_path):
        records = [user_record("hi"), assistant_record("yo")]
        p = write_jsonl(tmp_path, records)
        msgs = session_to_messages(p)
        assert msgs == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]

    def test_output_written(self, tmp_path):
        records = [user_record("hi")]
        p = write_jsonl(tmp_path, records)
        out = tmp_path / "sub" / "messages.json"
        msgs = session_to_messages(p, output=out)
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == [{"role": "user", "content": "hi"}]
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_output_converts_pil_to_data_uri(self, tmp_path, monkeypatch):
        img = _fake_png_image()
        if img is None:
            pytest.skip("Pillow is not installed")

        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: img)
        records = [user_record([
            {"type": "text", "text": "look"},
            {"type": "image", "source": {}},
        ])]
        p = write_jsonl(tmp_path, records)
        out = tmp_path / "messages.json"
        msgs = session_to_messages(p, output=out)

        # In-memory keeps the real image
        content = msgs[0]["content"]
        assert content[1]["image"] is img

        saved = json.loads(out.read_text(encoding="utf-8"))
        saved_image = saved[0]["content"][1]["image"]
        assert isinstance(saved_image, str)
        assert saved_image.startswith("data:image/")

    def test_output_serializes_non_pil_image_as_repr(self, tmp_path, monkeypatch):
        """messages_to_json_safe leaves non-PIL image objects untouched."""
        # Directly verify the passthrough behavior on messages_to_json_safe,
        # since a bare object() is not JSON serializable and json.dump would
        # raise. This documents the branch where the image is not a real PIL
        # image and PIL is importable.
        pytest.importorskip("PIL.Image")
        sentinel = object()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image", "image": sentinel},
        ]}]
        converted = messages_to_json_safe(messages)
        assert converted[0]["content"][1] == {"type": "image", "image": sentinel}
        # Original messages untouched (shallow copy)
        assert messages[0]["content"][1]["image"] is sentinel

    def test_output_string_content_untouched(self, tmp_path):
        p = write_jsonl(tmp_path, [user_record("hello")])
        out = tmp_path / "messages.json"
        msgs = session_to_messages(p, output=out)
        assert msgs == [{"role": "user", "content": "hello"}]
        assert json.loads(out.read_text(encoding="utf-8")) == [
            {"role": "user", "content": "hello"}
        ]


# ── inspect_session (smoke + metadata) ─────────────────────────────────

class TestInspectSession:
    def test_full_report_runs(self, tmp_path, capsys):
        records = [
            user_record("hi", branch="main"),
            assistant_record("yo", usage={"input_tokens": 1, "output_tokens": 2}),
        ]
        p = write_jsonl(tmp_path, records)
        inspect_session(p, show_flow=True, show_blocks=True, show_raw=True)
        out = capsys.readouterr().out
        assert "CLAUDE CODE" in out
        assert "Session" in out
        assert "sess-1" in out
        assert "CWD" in out
        assert "/home/x" in out
        assert "Branch" in out
        assert "main" in out
        assert "RAW RECORD EXAMPLES" in out

    def test_no_flow_no_raw(self, tmp_path, capsys):
        p = write_jsonl(tmp_path, [user_record("hi")])
        inspect_session(p)
        out = capsys.readouterr().out
        assert "CLAUDE CODE" in out
        assert "RAW RECORD EXAMPLES" not in out

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_session(tmp_path / "missing.jsonl")


# ── _print_session_metadata / _print_raw_examples ──────────────────────

class TestPrintHelpers:
    def test_metadata_none_user(self, capsys):
        _print_session_metadata([{"type": "assistant"}])
        assert capsys.readouterr().out == ""

    def test_metadata_defaults(self, capsys):
        _print_session_metadata([{"type": "user"}])
        out = capsys.readouterr().out
        assert "Session" in out
        assert "CWD" in out
        # The '?' markers are colored with DIM/reset escape codes around them.
        assert "?" in out
        assert "Branch" not in out

    def test_metadata_with_branch(self, capsys):
        _print_session_metadata([{"type": "user", "sessionId": "s", "cwd": "c",
                                 "gitBranch": "feat"}])
        out = capsys.readouterr().out
        assert "Session" in out
        assert "s" in out
        assert "CWD" in out
        assert "c" in out
        assert "Branch" in out
        assert "feat" in out

    def test_raw_examples_one_per_type(self, capsys):
        records = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {"type": "user", "message": {"content": "stringy"}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "x", "id": "1"}]}},
        ]
        _print_raw_examples(records)
        out = capsys.readouterr().out
        assert "RAW RECORD EXAMPLES" in out
        assert out.count("type = user") == 1
        assert "type = assistant" in out
        assert "content[0]:" in out

    def test_raw_examples_string_content_branch(self, capsys):
        records = [
            {"type": "user", "message": {"content": "a plain string content"}},
        ]
        _print_raw_examples(records)
        out = capsys.readouterr().out
        assert "content (string):" in out
        assert "a plain string content" in out

    def test_raw_examples_no_message(self, capsys):
        records = [{"type": "summary"}]
        _print_raw_examples(records)
        out = capsys.readouterr().out
        assert "type = summary" in out
        # No message -> no "message keys:" line for it
        assert "message keys:" not in out


# ── dead-code branch after return in session_to_messages ──────────────

class TestSessionToMessagesOutputShape:
    def test_output_parent_created(self, tmp_path):
        p = write_jsonl(tmp_path, [user_record("x")])
        out = tmp_path / "a" / "b" / "out.json"
        session_to_messages(p, output=out)
        assert out.parent.is_dir()
        assert out.is_file()
