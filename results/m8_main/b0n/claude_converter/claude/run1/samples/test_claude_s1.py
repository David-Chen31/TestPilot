"""Unit tests for claude_converter.claude."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_converter import claude
from claude_converter.claude import (
    load_session,
    records_to_messages,
    session_to_messages,
    inspect_session,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _write_jsonl(tmp_path: Path, records: list[dict], name: str = "session.jsonl") -> Path:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _user(content, **extra):
    rec = {"type": "user", "message": {"role": "user", "content": content}}
    rec.update(extra)
    return rec


def _assistant(content, **extra):
    rec = {"type": "assistant", "message": {"role": "assistant", "content": content}}
    rec.update(extra)
    return rec


# ── _block_to_str ──────────────────────────────────────────────────────

class TestBlockToStr:
    def test_text(self):
        assert claude._block_to_str({"type": "text", "text": "hello"}) == "hello"

    def test_text_missing(self):
        assert claude._block_to_str({"type": "text"}) == ""

    def test_thinking(self):
        out = claude._block_to_str({"type": "thinking", "thinking": "ponder"})
        assert out == "<thinking>ponder</thinking>"

    def test_tool_use(self):
        out = claude._block_to_str(
            {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}}
        )
        assert out.startswith("<tool_use name='Bash'>")
        assert out.endswith("</tool_use>")
        assert '{"cmd": "ls"}' in out

    def test_tool_use_empty_input(self):
        out = claude._block_to_str({"type": "tool_use", "name": "X"})
        assert out == "<tool_use name='X'>{}</tool_use>"

    def test_tool_result_string_content(self):
        out = claude._block_to_str({"type": "tool_result", "content": "ok"})
        assert out == "<tool_result>ok</tool_result>"

    def test_tool_result_list_content(self):
        block = {
            "type": "tool_result",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "b"},
                "not-a-dict",
            ],
        }
        assert claude._block_to_str(block) == "<tool_result>a <image> b</tool_result>"

    def test_image(self):
        assert claude._block_to_str({"type": "image"}) == "<image>"

    def test_unknown(self):
        assert claude._block_to_str({"type": "weird"}) == ""

    def test_no_type(self):
        assert claude._block_to_str({}) == ""


# ── _decode_claude_image ──────────────────────────────────────────────

class TestDecodeClaudeImage:
    def test_unsupported_source_type(self, capsys):
        result = claude._decode_claude_image(
            {"type": "image", "source": {"type": "url", "data": "x"}}
        )
        assert result is None
        err = capsys.readouterr().err
        assert "unsupported source type" in err

    def test_empty_data(self, capsys):
        result = claude._decode_claude_image(
            {"type": "image", "source": {"type": "base64", "data": ""}}
        )
        assert result is None
        assert "empty base64 data" in capsys.readouterr().err

    def test_missing_source(self, capsys):
        result = claude._decode_claude_image({"type": "image"})
        assert result is None
        assert "unsupported source type" in capsys.readouterr().err

    def test_success(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda d: sentinel)
        result = claude._decode_claude_image(
            {"type": "image", "source": {"type": "base64", "data": "abc"}}
        )
        assert result is sentinel

    def test_value_error_caught(self, monkeypatch, capsys):
        def boom(data):
            raise ValueError("bad img")
        monkeypatch.setattr(claude, "convert_base64_to_pil_image", boom)
        result = claude._decode_claude_image(
            {"type": "image", "source": {"type": "base64", "data": "abc"}}
        )
        assert result is None
        assert "unreadable image block: bad img" in capsys.readouterr().err

    def test_import_error_caught(self, monkeypatch):
        def boom(data):
            raise ImportError("no pillow")
        monkeypatch.setattr(claude, "convert_base64_to_pil_image", boom)
        result = claude._decode_claude_image(
            {"type": "image", "source": {"type": "base64", "data": "abc"}}
        )
        assert result is None


# ── _block_to_parts ────────────────────────────────────────────────────

class TestBlockToParts:
    def test_text(self):
        assert claude._block_to_parts({"type": "text", "text": "hi"}) == [
            {"type": "text", "text": "hi"}
        ]

    def test_whitespace_text_dropped(self):
        assert claude._block_to_parts({"type": "text", "text": "   "}) == []

    def test_image_decoded(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: sentinel)
        parts = claude._block_to_parts({"type": "image"})
        assert parts == [{"type": "image", "image": sentinel}]

    def test_image_failed_returns_empty(self, monkeypatch):
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: None)
        assert claude._block_to_parts({"type": "image"}) == []

    def test_tool_result_no_images(self):
        parts = claude._block_to_parts({"type": "tool_result", "content": "out"})
        assert parts == [{"type": "text", "text": "<tool_result>out</tool_result>"}]

    def test_tool_result_empty_text_dropped(self):
        assert claude._block_to_parts({"type": "tool_result", "content": ""}) == []

    def test_tool_result_with_images(self, monkeypatch):
        calls = []

        def fake_decode(b):
            calls.append(b)
            return "IMG"

        monkeypatch.setattr(claude, "_decode_claude_image", fake_decode)
        block = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "   "},
                {"type": "text", "text": "after"},
                "not-a-dict",
            ],
        }
        parts = claude._block_to_parts(block)
        assert parts == [
            {"type": "text", "text": "<tool_result id='t1'>"},
            {"type": "text", "text": "before"},
            {"type": "image", "image": "IMG"},
            {"type": "text", "text": "after"},
            {"type": "text", "text": "</tool_result>"},
        ]
        assert len(calls) == 1

    def test_tool_result_image_decode_fails_skipped(self, monkeypatch):
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: None)
        block = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [{"type": "image"}],
        }
        parts = claude._block_to_parts(block)
        assert parts == [
            {"type": "text", "text": "<tool_result id='t1'>"},
            {"type": "text", "text": "</tool_result>"},
        ]

    def test_tool_result_default_id(self, monkeypatch):
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: "IMG")
        block = {"type": "tool_result", "content": [{"type": "image"}]}
        parts = claude._block_to_parts(block)
        assert parts[0] == {"type": "text", "text": "<tool_result id='?'>"}


# ── _content_to_str ────────────────────────────────────────────────────

class TestContentToStr:
    def test_string(self):
        assert claude._content_to_str("hello") == "hello"

    def test_list(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert claude._content_to_str(content) == "a\nb"

    def test_list_drops_empty(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": ""}]
        assert claude._content_to_str(content) == "a"

    def test_list_skips_non_dict(self):
        assert claude._content_to_str(["x", {"type": "text", "text": "a"}]) == "a"

    def test_none(self):
        assert claude._content_to_str(None) == ""

    def test_number(self):
        assert claude._content_to_str(42) == ""


# ── _content_to_parts ──────────────────────────────────────────────────

class TestContentToParts:
    def test_string(self):
        assert claude._content_to_parts("hi") == [{"type": "text", "text": "hi"}]

    def test_empty_string(self):
        assert claude._content_to_parts("   ") == []

    def test_list(self, monkeypatch):
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: "IMG")
        content = [{"type": "text", "text": "a"}, {"type": "image"}]
        parts = claude._content_to_parts(content)
        assert parts == [
            {"type": "text", "text": "a"},
            {"type": "image", "image": "IMG"},
        ]

    def test_list_skips_non_dict(self):
        assert claude._content_to_parts(["x"]) == []

    def test_none(self):
        assert claude._content_to_parts(None) == []


# ── _collect_block_types ───────────────────────────────────────────────

class TestCollectBlockTypes:
    def test_string(self):
        assert claude._collect_block_types("x") == ["raw_string"]

    def test_list(self):
        content = [{"type": "text"}, {"type": "image"}, "not-a-dict", {}]
        assert claude._collect_block_types(content) == ["text", "image", "unknown"]

    def test_none(self):
        assert claude._collect_block_types(None) == ["unknown"]


# ── _render_block ──────────────────────────────────────────────────────

class TestRenderBlock:
    def test_text(self):
        out = claude._render_block({"type": "text", "text": "hello"})
        assert "text" in out
        assert "hello" in out

    def test_text_truncated(self):
        out = claude._render_block({"type": "text", "text": "x" * 300})
        assert "..." in out

    def test_thinking(self):
        out = claude._render_block({"type": "thinking", "thinking": "deep"})
        assert "thinking" in out
        assert "deep" in out

    def test_tool_use(self):
        out = claude._render_block(
            {"type": "tool_use", "name": "Bash", "id": "id1", "input": {"a": 1}}
        )
        assert "tool_use" in out
        assert "Bash" in out
        assert "id1" in out
        assert "input:" in out
        assert '{"a": 1}' in out

    def test_tool_result_with_list_and_image(self):
        block = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [
                {"type": "image", "source": {"media_type": "image/png", "data": "abcd"}},
                {"type": "text", "text": "note"},
            ],
        }
        out = claude._render_block(block)
        assert "tool_result" in out
        assert "t1" in out
        assert "image.png".replace("image.", "image.") in out or "image/png" in out
        assert "4 base64 chars" in out
        assert "note" in out

    def test_tool_result_string_content(self):
        out = claude._render_block({"type": "tool_result", "content": "plain"})
        assert "content: plain" in out

    def test_image(self):
        block = {"type": "image", "source": {"media_type": "image/jpeg", "data": "abc"}}
        out = claude._render_block(block)
        assert "image" in out
        assert "image/jpeg" in out
        assert "3 base64 chars" in out

    def test_image_missing_source(self):
        out = claude._render_block({"type": "image"})
        assert "?" in out
        assert "0 base64 chars" in out

    def test_unknown_type(self):
        out = claude._render_block({"type": "mystery", "foo": "bar"})
        assert "mystery" in out


# ── load_session ───────────────────────────────────────────────────────

class TestLoadSession:
    def test_loads_records(self, tmp_path):
        p = _write_jsonl(tmp_path, [{"type": "user"}, {"type": "assistant"}])
        assert load_session(p) == [{"type": "user"}, {"type": "assistant"}]

    def test_accepts_str_path(self, tmp_path):
        p = _write_jsonl(tmp_path, [{"a": 1}])
        assert load_session(str(p)) == [{"a": 1}]

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
        p.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(p)

    def test_invalid_json_line_skipped(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        assert load_session(p) == [{"a": 1}]
        assert "invalid JSON" in capsys.readouterr().err

    def test_json_extension(self, tmp_path):
        p = _write_jsonl(tmp_path, [{"a": 1}], name="s.json")
        assert load_session(p) == [{"a": 1}]


# ── records_to_messages ────────────────────────────────────────────────

class TestRecordsToMessages:
    def test_basic_alternation(self):
        records = [_user("hi"), _assistant("hello")]
        result = records_to_messages(records)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_skips_system(self):
        records = [
            {"type": "system", "message": {"role": "system", "content": "sys"}},
            _user("hi"),
        ]
        assert records_to_messages(records) == [{"role": "user", "content": "hi"}]

    def test_skips_unknown_type(self):
        records = [{"type": "summary", "message": {"role": "user", "content": "x"}}]
        assert records_to_messages(records) == []

    def test_empty_content_skipped(self):
        records = [_user(""), _assistant("  ")]
        assert records_to_messages(records) == []

    def test_merges_consecutive_same_role(self):
        records = [_user("a"), _user("b")]
        result = records_to_messages(records)
        assert result == [{"role": "user", "content": "a\nb"}]

    def test_does_not_merge_different_roles(self):
        records = [_user("a"), _assistant("b")]
        result = records_to_messages(records)
        assert result == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    def test_role_from_message(self):
        records = [{"type": "user", "message": {"role": "assistant", "content": "hi"}}]
        assert records_to_messages(records) == [{"role": "assistant", "content": "hi"}]

    def test_role_falls_back_to_type(self):
        records = [{"type": "user", "message": {"content": "hi"}}]
        assert records_to_messages(records) == [{"role": "user", "content": "hi"}]

    def test_missing_message_ignored(self):
        assert records_to_messages([{"type": "user"}]) == []

    def test_message_null_ignored(self):
        assert records_to_messages([{"type": "user", "message": None}]) == []

    def test_flatten_blocks_to_string(self):
        records = [
            _user(
                [
                    {"type": "text", "text": "open"},
                    {"type": "thinking", "thinking": "hmm"},
                ]
            )
        ]
        result = records_to_messages(records)
        assert result == [
            {"role": "user", "content": "open\n<thinking>hmm</thinking>"}
        ]

    def test_tool_use_flattened(self):
        records = [
            _assistant(
                [{"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}}]
            )
        ]
        result = records_to_messages(records)
        assert result[0]["role"] == "assistant"
        assert "<tool_use name='Bash'>" in result[0]["content"]

    def test_tool_result_flattened(self):
        records = [
            _user([{"type": "tool_result", "content": "done"}])
        ]
        result = records_to_messages(records)
        assert result == [
            {"role": "user", "content": "<tool_result>done</tool_result>"}
        ]

    def test_image_kept_as_parts(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: sentinel)
        records = [
            _user(
                [
                    {"type": "text", "text": "look"},
                    {"type": "image"},
                ]
            )
        ]
        result = records_to_messages(records)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [
            {"type": "text", "text": "look"},
            {"type": "image", "image": sentinel},
        ]

    def test_image_only_message(self, monkeypatch):
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: "IMG")
        records = [_user([{"type": "image"}])]
        result = records_to_messages(records)
        assert result == [
            {"role": "user", "content": [{"type": "image", "image": "IMG"}]}
        ]

    def test_merges_image_parts(self, monkeypatch):
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: "IMG")
        records = [
            _user([{"type": "text", "text": "a"}]),
            _user([{"type": "image"}]),
        ]
        result = records_to_messages(records)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "a"},
            {"type": "image", "image": "IMG"},
        ]

    def test_merged_text_joined_with_newline(self):
        records = [_user("a"), _user("b"), _user("c")]
        assert records_to_messages(records) == [{"role": "user", "content": "a\nb\nc"}]

    def test_multiple_turns(self):
        records = [
            _user("u1"),
            _assistant("a1"),
            _user("u2"),
            _assistant("a2"),
        ]
        result = records_to_messages(records)
        assert result == [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]

    def test_empty_records(self):
        assert records_to_messages([]) == []


# ── session_to_messages ────────────────────────────────────────────────

class TestSessionToMessages:
    def test_basic(self, tmp_path):
        p = _write_jsonl(tmp_path, [_user("hi"), _assistant("hello")])
        result = session_to_messages(p)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_output_written(self, tmp_path):
        p = _write_jsonl(tmp_path, [_user("hi")])
        out = tmp_path / "sub" / "messages.json"
        result = session_to_messages(p, output=out)
        assert result == [{"role": "user", "content": "hi"}]
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == [{"role": "user", "content": "hi"}]

    def test_output_nested_dir_created(self, tmp_path):
        p = _write_jsonl(tmp_path, [_assistant("x")])
        out = tmp_path / "a" / "b" / "c.json"
        session_to_messages(p, output=out)
        assert out.parent.is_dir()

    def test_output_str_path(self, tmp_path):
        p = _write_jsonl(tmp_path, [_user("hi")])
        out = tmp_path / "msg.json"
        session_to_messages(p, output=str(out))
        assert out.read_text(encoding="utf-8").strip().startswith("[")

    def test_no_output_no_file(self, tmp_path):
        p = _write_jsonl(tmp_path, [_user("hi")])
        before = set(tmp_path.iterdir())
        session_to_messages(p)
        after = set(tmp_path.iterdir())
        assert before == after

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            session_to_messages(tmp_path / "nope.jsonl")

    def test_messages_to_json_safe_used(self, tmp_path, monkeypatch):
        p = _write_jsonl(tmp_path, [_user("hi")])

        called = {}

        def fake_json_safe(messages):
            called["messages"] = messages
            return [{"role": "user", "content": "[converted]"}]

        monkeypatch.setattr(claude, "messages_to_json_safe", fake_json_safe)
        out = tmp_path / "out.json"
        result = session_to_messages(p, output=out)
        # Returned in-memory list is the real one, not json-safe copy.
        assert result == [{"role": "user", "content": "hi"}]
        assert "messages" in called
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == [{"role": "user", "content": "[converted]"}]


# ── Schema helpers ─────────────────────────────────────────────────────

class TestSchemaHelpers:
    def test_record_type_of(self):
        assert claude._record_type_of({"type": "user"}) == "user"
        assert claude._record_type_of({}) == "unknown"

    def test_is_message(self):
        assert claude._is_message({"type": "user"}) is True
        assert claude._is_message({"type": "assistant"}) is True
        assert claude._is_message({"type": "system"}) is False

    def test_role_of_message(self):
        r = {"type": "user", "message": {"role": "assistant"}}
        assert claude._role_of(r) == "assistant"

    def test_role_of_fallback(self):
        r = {"type": "user"}
        assert claude._role_of(r) == "user"

    def test_text_of(self):
        r = {"message": {"content": "hello"}}
        assert claude._text_of(r) == "hello"

    def test_text_of_list(self):
        r = {"message": {"content": [{"type": "text", "text": "a"}]}}
        assert claude._text_of(r) == "a"

    def test_timestamp_of(self):
        r = {"timestamp": "2024-01-02T03:04:05.678Z"}
        assert claude._timestamp_of(r) == "2024-01-02 03:04:05"

    def test_timestamp_of_missing(self):
        assert claude._timestamp_of({}) == ""

    def test_tokens_of(self):
        r = {"message": {"usage": {"output_tokens": 42}}}
        assert claude._tokens_of(r) == 42

    def test_tokens_of_missing(self):
        assert claude._tokens_of({}) == 0

    def test_block_types_of(self):
        r = {"message": {"content": [{"type": "text"}]}}
        assert claude._block_types_of(r) == ["text"]

    def test_blocks_of(self):
        blocks = [{"type": "text"}, "skip"]
        r = {"message": {"content": blocks}}
        assert claude._blocks_of(r) == [{"type": "text"}]

    def test_blocks_of_string_content(self):
        r = {"message": {"content": "hi"}}
        assert claude._blocks_of(r) == []

    def test_total_tokens_of(self):
        records = [
            {"message": {"usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 3,
            }}},
            {"message": {"usage": {"input_tokens": 1, "output_tokens": 1}}},
        ]
        totals = claude._total_tokens_of(records)
        assert totals == {
            "input": 11,
            "output": 6,
            "cache_read": 2,
            "cache_write": 3,
            "total": 17,
        }

    def test_total_tokens_of_empty(self):
        assert claude._total_tokens_of([]) == {}

    def test_total_tokens_of_zero(self):
        records = [{"message": {"usage": {"input_tokens": 0, "output_tokens": 0}}}]
        assert claude._total_tokens_of(records) == {}


# ── CLAUDE_SCHEMA ──────────────────────────────────────────────────────

class TestClaudeSchema:
    def test_label(self):
        assert claude.CLAUDE_SCHEMA.label == "CLAUDE CODE"

    def test_wired_callables(self):
        s = claude.CLAUDE_SCHEMA
        assert s.record_type_of is claude._record_type_of
        assert s.is_message is claude._is_message
        assert s.role_of is claude._role_of
        assert s.text_of is claude._text_of
        assert s.timestamp_of is claude._timestamp_of
        assert s.tokens_of is claude._tokens_of
        assert s.block_types_of is claude._block_types_of
        assert s.blocks_of is claude._blocks_of
        assert s.render_block is claude._render_block
        assert s.total_tokens_of is claude._total_tokens_of


# ── _print_session_metadata ────────────────────────────────────────────

class TestPrintSessionMetadata:
    def test_no_user(self, capsys):
        claude._print_session_metadata([{"type": "assistant"}])
        assert capsys.readouterr().out == ""

    def test_with_metadata(self, capsys):
        records = [
            {"type": "assistant"},
            {"type": "user", "sessionId": "sid", "cwd": "/tmp", "gitBranch": "main"},
        ]
        claude._print_session_metadata(records)
        out = capsys.readouterr().out
        assert "sid" in out
        assert "/tmp" in out
        assert "main" in out

    def test_no_branch(self, capsys):
        records = [{"type": "user", "sessionId": "s", "cwd": "c"}]
        claude._print_session_metadata(records)
        out = capsys.readouterr().out
        assert "Branch" not in out


# ── _print_raw_examples ────────────────────────────────────────────────

class TestPrintRawExamples:
    def test_prints_each_type_once(self, capsys):
        records = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "x"}]}},
            {"type": "user"},
            {"type": "assistant", "message": {"content": "string content"}},
        ]
        claude._print_raw_examples(records)
        out = capsys.readouterr().out
        assert "RAW RECORD EXAMPLES" in out
        assert "type = user" in out
        assert "type = assistant" in out
        assert "content (string):" in out

    def test_empty_records(self, capsys):
        claude._print_raw_examples([])
        out = capsys.readouterr().out
        assert "RAW RECORD EXAMPLES" in out


# ── inspect_session ────────────────────────────────────────────────────

class TestInspectSession:
    def test_calls_run_inspection(self, tmp_path, monkeypatch):
        p = _write_jsonl(tmp_path, [_user("hi")])
        captured = {}

        def fake_run_inspection(path, records, schema, **kwargs):
            captured["path"] = path
            captured["records"] = records
            captured["schema"] = schema
            captured["kwargs"] = kwargs

        monkeypatch.setattr(claude, "run_inspection", fake_run_inspection)
        inspect_session(p)
        assert captured["schema"] is claude.CLAUDE_SCHEMA
        assert captured["records"] == [_user("hi")]
        assert captured["kwargs"]["show_flow"] is False
        assert captured["kwargs"]["show_blocks"] is False
        assert captured["kwargs"]["extra_header"] is claude._print_session_metadata

    def test_passes_flags(self, tmp_path, monkeypatch):
        p = _write_jsonl(tmp_path, [_user("hi")])
        captured = {}

        def fake_run_inspection(path, records, schema, **kwargs):
            captured["kwargs"] = kwargs

        monkeypatch.setattr(claude, "run_inspection", fake_run_inspection)
        monkeypatch.setattr(claude, "_print_raw_examples", lambda r: None)
        inspect_session(p, show_flow=True, show_blocks=True, show_raw=True)
        assert captured["kwargs"]["show_flow"] is True
        assert captured["kwargs"]["show_blocks"] is True

    def test_show_raw_calls_printer(self, tmp_path, monkeypatch, capsys):
        p = _write_jsonl(tmp_path, [_user("hi")])
        monkeypatch.setattr(claude, "run_inspection", lambda *a, **k: None)
        called = {}

        def fake_raw(records):
            called["records"] = records

        monkeypatch.setattr(claude, "_print_raw_examples", fake_raw)
        inspect_session(p, show_raw=True)
        assert called["records"] == [_user("hi")]

    def test_no_raw_does_not_call(self, tmp_path, monkeypatch):
        p = _write_jsonl(tmp_path, [_user("hi")])
        monkeypatch.setattr(claude, "run_inspection", lambda *a, **k: None)
        called = {"hit": False}

        def fake_raw(records):
            called["hit"] = True

        monkeypatch.setattr(claude, "_print_raw_examples", fake_raw)
        inspect_session(p)
        assert called["hit"] is False

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_session(tmp_path / "missing.jsonl")
