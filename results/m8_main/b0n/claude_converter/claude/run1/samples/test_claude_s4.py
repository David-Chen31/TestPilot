"""Unit tests for claude_converter.claude."""

from __future__ import annotations

import base64
import json
from io import BytesIO

import pytest

from claude_converter import claude

# Pillow is optional in the package; skip image tests if it's unavailable.
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


# ─────────────────────────── helpers ───────────────────────────

def _make_image_block(image_format="PNG", data=None):
    """Build a Claude Code image block, with real base64 data by default."""
    if data is None:
        buf = BytesIO()
        Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buf, format=image_format)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": f"image/{image_format.lower()}",
            "data": data,
        },
    }


def _write_jsonl(tmp_path, records, name="session.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ─────────────────────── _block_to_str ───────────────────────

class TestBlockToStr:
    def test_text(self):
        assert claude._block_to_str({"type": "text", "text": "hello"}) == "hello"

    def test_text_missing_text(self):
        assert claude._block_to_str({"type": "text"}) == ""

    def test_thinking(self):
        block = {"type": "thinking", "thinking": "pondering"}
        assert claude._block_to_str(block) == "<thinking>pondering</thinking>"

    def test_tool_use(self):
        block = {"type": "tool_use", "name": "Read", "input": {"path": "a.txt"}}
        out = claude._block_to_str(block)
        assert out.startswith("<tool_use name='Read'>")
        assert out.endswith("</tool_use>")
        assert '{"path": "a.txt"}' in out

    def test_tool_use_empty_input(self):
        block = {"type": "tool_use", "name": "X"}
        assert claude._block_to_str(block) == "<tool_use name='X'>{}</tool_use>"

    def test_tool_result_string_content(self):
        block = {"type": "tool_result", "content": "output"}
        assert claude._block_to_str(block) == "<tool_result>output</tool_result>"

    def test_tool_result_list_with_text_and_image(self):
        block = {
            "type": "tool_result",
            "content": [
                {"type": "text", "text": "see"},
                {"type": "image"},
                "not-a-dict",
            ],
        }
        assert claude._block_to_str(block) == "<tool_result>see <image></tool_result>"

    def test_tool_result_list_only_nontext(self):
        block = {"type": "tool_result", "content": [{"type": "other"}]}
        assert claude._block_to_str(block) == "<tool_result></tool_result>"

    def test_image(self):
        assert claude._block_to_str({"type": "image"}) == "<image>"

    def test_unknown_returns_empty(self):
        assert claude._block_to_str({"type": "wat"}) == ""

    def test_missing_type(self):
        assert claude._block_to_str({}) == ""


# ─────────────────────── _decode_claude_image ───────────────────────

class TestDecodeClaudeImage:
    def test_valid_image_returns_pil(self, capsys):
        block = _make_image_block()
        img = claude._decode_claude_image(block)
        assert isinstance(img, Image.Image)
        assert img.size == (4, 4)
        assert capsys.readouterr().err == ""

    def test_unsupported_source_type(self, capsys):
        block = {"type": "image", "source": {"type": "url", "data": "x"}}
        assert claude._decode_claude_image(block) is None
        err = capsys.readouterr().err
        assert "unsupported source type" in err
        assert "'url'" in err

    def test_empty_data(self, capsys):
        block = {"type": "image", "source": {"type": "base64", "data": ""}}
        assert claude._decode_claude_image(block) is None
        assert "empty base64 data" in capsys.readouterr().err

    def test_missing_data_treated_as_empty(self, capsys):
        block = {"type": "image", "source": {"type": "base64"}}
        assert claude._decode_claude_image(block) is None
        assert "empty base64 data" in capsys.readouterr().err

    def test_corrupted_base64(self, capsys):
        block = {"type": "image", "source": {"type": "base64", "data": "!!!notb64"}}
        assert claude._decode_claude_image(block) is None
        assert "unreadable image block" in capsys.readouterr().err

    def test_missing_source(self, capsys):
        assert claude._decode_claude_image({"type": "image"}) is None
        assert "unsupported source type" in capsys.readouterr().err

    def test_pillow_import_error(self, monkeypatch, capsys):
        def raise_import(_data):
            raise ImportError("no pillow")
        monkeypatch.setattr(claude, "convert_base64_to_pil_image", raise_import)
        block = _make_image_block()
        assert claude._decode_claude_image(block) is None
        assert "unreadable image block" in capsys.readouterr().err


# ─────────────────────── _block_to_parts ───────────────────────

class TestBlockToParts:
    def test_text_block(self):
        assert claude._block_to_parts({"type": "text", "text": "hi"}) == [
            {"type": "text", "text": "hi"}
        ]

    def test_whitespace_text_dropped(self):
        assert claude._block_to_parts({"type": "text", "text": "   "}) == []

    def test_image_block(self):
        block = _make_image_block()
        parts = claude._block_to_parts(block)
        assert len(parts) == 1
        assert parts[0]["type"] == "image"
        assert isinstance(parts[0]["image"], Image.Image)

    def test_image_block_undecodable_gives_empty(self, capsys):
        block = {"type": "image", "source": {"type": "base64", "data": ""}}
        assert claude._block_to_parts(block) == []
        assert "empty base64 data" in capsys.readouterr().err

    def test_tool_result_text_only(self):
        block = {"type": "tool_result", "content": "done"}
        assert claude._block_to_parts(block) == [
            {"type": "text", "text": "<tool_result>done</tool_result>"}
        ]

    def test_tool_result_text_only_empty(self):
        block = {"type": "tool_result", "content": "   "}
        assert claude._block_to_parts(block) == []

    def test_tool_result_with_nested_image(self):
        block = {
            "type": "tool_result",
            "tool_use_id": "abc",
            "content": [
                {"type": "text", "text": "here"},
                _make_image_block(),
                {"type": "text", "text": "   "},
                "junk",
            ],
        }
        parts = claude._block_to_parts(block)
        assert parts[0] == {"type": "text", "text": "<tool_result id='abc'>"}
        assert parts[1] == {"type": "text", "text": "here"}
        assert parts[2]["type"] == "image"
        assert isinstance(parts[2]["image"], Image.Image)
        # whitespace text dropped, junk skipped
        assert parts[-1] == {"type": "text", "text": "</tool_result>"}
        assert len(parts) == 4

    def test_tool_result_with_nested_image_default_id(self):
        block = {
            "type": "tool_result",
            "content": [_make_image_block()],
        }
        parts = claude._block_to_parts(block)
        assert parts[0] == {"type": "text", "text": "<tool_result id='?'>"}

    def test_tool_result_with_nested_image_bad_image_skipped(self, capsys):
        bad_img = {"type": "image", "source": {"type": "base64", "data": ""}}
        block = {
            "type": "tool_result",
            "tool_use_id": "z",
            "content": [bad_img, {"type": "text", "text": "t"}],
        }
        parts = claude._block_to_parts(block)
        assert parts[0] == {"type": "text", "text": "<tool_result id='z'>"}
        assert parts[1] == {"type": "text", "text": "t"}
        assert parts[2] == {"type": "text", "text": "</tool_result>"}
        assert len(parts) == 3
        assert "empty base64 data" in capsys.readouterr().err


# ─────────────────────── _content_to_str ───────────────────────

class TestContentToStr:
    def test_string(self):
        assert claude._content_to_str("abc") == "abc"

    def test_none(self):
        assert claude._content_to_str(None) == ""

    def test_list_of_blocks(self):
        content = [
            {"type": "text", "text": "a"},
            "ignored",
            {"type": "thinking", "thinking": "th"},
            {"type": "text", "text": "  "},
        ]
        assert claude._content_to_str(content) == "a\n<thinking>th</thinking>"

    def test_other_type(self):
        assert claude._content_to_str(42) == ""


# ─────────────────────── _content_to_parts ───────────────────────

class TestContentToParts:
    def test_string(self):
        assert claude._content_to_parts("hi") == [{"type": "text", "text": "hi"}]

    def test_empty_string(self):
        assert claude._content_to_parts("   ") == []

    def test_list(self):
        content = [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]
        assert claude._content_to_parts(content) == [
            {"type": "text", "text": "x"},
            {"type": "text", "text": "y"},
        ]

    def test_none(self):
        assert claude._content_to_parts(None) == []


# ─────────────────────── _collect_block_types ───────────────────────

class TestCollectBlockTypes:
    def test_string(self):
        assert claude._collect_block_types("x") == ["raw_string"]

    def test_list(self):
        content = [{"type": "text"}, {"no_type": 1}, "junk"]
        assert claude._collect_block_types(content) == ["text", "unknown"]

    def test_none(self):
        assert claude._collect_block_types(None) == ["unknown"]


# ─────────────────────── _render_block ───────────────────────

class TestRenderBlock:
    def test_text(self):
        out = claude._render_block({"type": "text", "text": "hello"})
        assert "text: hello" in out

    def test_thinking(self):
        out = claude._render_block({"type": "thinking", "thinking": "idea"})
        assert "thinking: idea" in out

    def test_tool_use(self):
        block = {"type": "tool_use", "name": "Read", "id": "id1", "input": {"a": 1}}
        out = claude._render_block(block)
        assert "tool_use: Read" in out
        assert "id1" in out
        assert 'input: {"a": 1}' in out

    def test_tool_result_string_content(self):
        block = {"type": "tool_result", "tool_use_id": "t1", "content": "abc"}
        out = claude._render_block(block)
        assert "tool_result:" in out
        assert "t1" in out
        assert "content: abc" in out

    def test_tool_result_list_content(self):
        block = {
            "type": "tool_result",
            "tool_use_id": "t2",
            "content": [
                {
                    "type": "image",
                    "source": {"media_type": "image/png", "data": "abcd"},
                },
                {"type": "text", "text": "hi"},
                "junk",
            ],
        }
        out = claude._render_block(block)
        assert "image: image/png (4 base64 chars)" in out
        assert "content: hi" in out

    def test_image(self):
        block = {
            "type": "image",
            "source": {"media_type": "image/jpeg", "data": "xyz"},
        }
        out = claude._render_block(block)
        assert "image: image/jpeg (3 base64 chars)" in out

    def test_unknown(self):
        out = claude._render_block({"type": "mystery", "foo": "bar"})
        assert "mystery" in out


# ─────────────────────── load_session ───────────────────────

class TestLoadSession:
    def test_loads_records(self, tmp_path):
        records = [{"type": "user"}, {"type": "assistant"}]
        path = _write_jsonl(tmp_path, records)
        assert claude.load_session(path) == records

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            claude.load_session(tmp_path / "nope.jsonl")

    def test_bad_extension(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("{}")
        with pytest.raises(ValueError):
            claude.load_session(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        with pytest.raises(ValueError):
            claude.load_session(p)


# ─────────────────────── records_to_messages ───────────────────────

class TestRecordsToMessages:
    def test_skips_non_message_types(self):
        records = [
            {"type": "summary", "summary": "x"},
            {"type": "user", "message": {"role": "user", "content": "hi"}},
        ]
        assert claude.records_to_messages(records) == [
            {"role": "user", "content": "hi"}
        ]

    def test_empty(self):
        assert claude.records_to_messages([]) == []

    def test_simple_string_content(self):
        records = [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "world"}},
        ]
        assert claude.records_to_messages(records) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_uses_record_type_when_role_missing(self):
        records = [{"type": "user", "message": {"content": "hi"}}]
        assert claude.records_to_messages(records) == [
            {"role": "user", "content": "hi"}
        ]

    def test_skips_empty_content(self):
        records = [
            {"type": "user", "message": {"role": "user", "content": "   "}},
            {"type": "assistant", "message": {"role": "assistant", "content": "x"}},
        ]
        assert claude.records_to_messages(records) == [
            {"role": "assistant", "content": "x"}
        ]

    def test_merges_consecutive_same_role(self):
        records = [
            {"type": "user", "message": {"role": "user", "content": "a"}},
            {"type": "user", "message": {"role": "user", "content": "b"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "c"}},
        ]
        assert claude.records_to_messages(records) == [
            {"role": "user", "content": "a\nb"},
            {"role": "assistant", "content": "c"},
        ]

    def test_block_content_flattened(self):
        records = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "say"},
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"cmd": "ls"},
                        },
                    ],
                },
            }
        ]
        msgs = claude.records_to_messages(records)
        assert msgs[0]["role"] == "assistant"
        content = msgs[0]["content"]
        assert isinstance(content, str)
        assert "say" in content
        assert "tool_use name='Bash'" in content

    def test_text_only_keeps_string_shape(self):
        records = [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
        ]
        msgs = claude.records_to_messages(records)
        assert isinstance(msgs[0]["content"], str)

    def test_image_content_produces_list(self):
        records = [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        _make_image_block(),
                    ],
                },
            }
        ]
        msgs = claude.records_to_messages(records)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "look"}
        assert content[1]["type"] == "image"
        assert isinstance(content[1]["image"], Image.Image)

    def test_null_message_treated_as_empty(self):
        # dget guards against null; should not crash, record skipped.
        records = [
            {"type": "user", "message": None},
            {"type": "assistant", "message": {"role": "assistant", "content": "x"}},
        ]
        assert claude.records_to_messages(records) == [
            {"role": "assistant", "content": "x"}
        ]


# ─────────────────────── session_to_messages ───────────────────────

class TestSessionToMessages:
    def test_basic(self, tmp_path):
        records = [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "yo"}},
        ]
        path = _write_jsonl(tmp_path, records)
        assert claude.session_to_messages(path) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]

    def test_output_written(self, tmp_path):
        records = [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
        ]
        path = _write_jsonl(tmp_path, records)
        out = tmp_path / "sub" / "messages.json"
        result = claude.session_to_messages(path, output=out)
        assert result == [{"role": "user", "content": "hi"}]
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == [{"role": "user", "content": "hi"}]

    def test_output_with_image_serializes(self, tmp_path):
        records = [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "img"}, _make_image_block()],
                },
            }
        ]
        path = _write_jsonl(tmp_path, records)
        out = tmp_path / "m.json"
        result = claude.session_to_messages(path, output=out)

        # In-memory keeps real PIL image
        content = result[0]["content"]
        assert isinstance(content, list)
        assert isinstance(content[1]["image"], Image.Image)

        # On-disk is JSON safe
        loaded = json.loads(out.read_text(encoding="utf-8"))
        img_part = loaded[0]["content"][1]
        assert img_part["type"] == "image"
        assert isinstance(img_part["image"], str)
        assert img_part["image"].startswith("data:image/")

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            claude.session_to_messages(tmp_path / "missing.jsonl")


# ─────────────────────── schema helpers ───────────────────────

class TestSchemaHelpers:
    def test_record_type_of(self):
        assert claude._record_type_of({"type": "user"}) == "user"
        assert claude._record_type_of({}) == "unknown"

    def test_is_message(self):
        assert claude._is_message({"type": "user"}) is True
        assert claude._is_message({"type": "assistant"}) is True
        assert claude._is_message({"type": "summary"}) is False

    def test_role_of(self):
        assert claude._role_of({"type": "user", "message": {"role": "user"}}) == "user"

    def test_role_of_fallback(self):
        assert claude._role_of({"type": "user", "message": {}}) == "user"

    def test_text_of(self):
        r = {"type": "user", "message": {"content": [{"type": "text", "text": "x"}]}}
        assert claude._text_of(r) == "x"

    def test_timestamp_of(self):
        r = {"timestamp": "2024-01-02T03:04:05.123Z"}
        assert claude._timestamp_of(r) == "2024-01-02 03:04:05"

    def test_timestamp_of_missing(self):
        assert claude._timestamp_of({}) == ""

    def test_tokens_of(self):
        r = {"message": {"usage": {"output_tokens": 42}}}
        assert claude._tokens_of(r) == 42

    def test_tokens_of_missing(self):
        assert claude._tokens_of({}) == 0

    def test_block_types_of(self):
        r = {"message": {"content": [{"type": "text"}, {"type": "image"}]}}
        assert claude._block_types_of(r) == ["text", "image"]

    def test_blocks_of(self):
        content = [{"type": "text"}, "junk"]
        r = {"message": {"content": content}}
        assert claude._blocks_of(r) == [{"type": "text"}]

    def test_blocks_of_string_content(self):
        assert claude._blocks_of({"message": {"content": "str"}}) == []

    def test_total_tokens_of(self):
        records = [
            {"message": {"usage": {"input_tokens": 10, "output_tokens": 5,
                                   "cache_read_input_tokens": 2,
                                   "cache_creation_input_tokens": 3}}},
            {"message": {"usage": {"input_tokens": 1, "output_tokens": 4}}},
        ]
        assert claude._total_tokens_of(records) == {
            "input": 11, "output": 9, "cache_read": 2,
            "cache_write": 3, "total": 20,
        }

    def test_total_tokens_of_empty(self):
        assert claude._total_tokens_of([]) == {}

    def test_total_tokens_of_zero_returns_empty(self):
        records = [{"message": {"usage": {"input_tokens": 0, "output_tokens": 0}}}]
        assert claude._total_tokens_of(records) == {}


# ─────────────────────── CLAUDE_SCHEMA ───────────────────────

class TestClaudeSchema:
    def test_schema_fields(self):
        s = claude.CLAUDE_SCHEMA
        assert s.label == "CLAUDE CODE"
        assert s.record_type_of is claude._record_type_of
        assert s.is_message is claude._is_message
        assert s.role_of is claude._role_of
        assert s.text_of is claude._text_of
        assert s.timestamp_of is claude._timestamp_of
        assert s.total_tokens_of is claude._total_tokens_of
        assert s.tokens_of is claude._tokens_of
        assert s.block_types_of is claude._block_types_of
        assert s.blocks_of is claude._blocks_of
        assert s.render_block is claude._render_block


# ─────────────────────── printing helpers ───────────────────────

class TestPrintSessionMetadata:
    def test_prints_session_cwd_and_branch(self, capsys):
        records = [{
            "type": "user",
            "sessionId": "sid",
            "cwd": "/home/x",
            "gitBranch": "main",
        }]
        claude._print_session_metadata(records)
        out = capsys.readouterr().out
        assert "Session : sid" in out
        assert "CWD     : /home/x" in out
        assert "Branch  : main" in out

    def test_no_branch_line(self, capsys):
        records = [{"type": "user", "sessionId": "sid", "cwd": "/c"}]
        claude._print_session_metadata(records)
        out = capsys.readouterr().out
        assert "Session : sid" in out
        assert "Branch" not in out

    def test_no_user(self, capsys):
        claude._print_session_metadata([{"type": "assistant"}])
        assert capsys.readouterr().out == ""


class TestPrintRawExamples:
    def test_prints_one_per_type(self, capsys):
        records = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "a"}]}},
            {"type": "user", "message": {"content": "second"}},
            {"type": "assistant", "message": {"content": "plain"}},
        ]
        claude._print_raw_examples(records)
        out = capsys.readouterr().out
        assert "RAW RECORD EXAMPLES" in out
        assert "type = user" in out
        assert "type = assistant" in out
        # content[0] block shown once for first user
        assert "content[0]" in out
        assert "content (string)" in out


# ─────────────────────── inspect_session ───────────────────────

class TestInspectSession:
    def test_runs_without_error(self, tmp_path, capsys):
        records = [
            {"type": "user", "sessionId": "s", "cwd": "/c",
             "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "yo",
                                              "usage": {"output_tokens": 3}}},
        ]
        path = _write_jsonl(tmp_path, records)
        claude.inspect_session(path)
        out = capsys.readouterr().out
        assert "CLAUDE CODE" in out
        assert "Session : s" in out

    def test_show_flow_and_blocks(self, tmp_path, capsys):
        records = [
            {"type": "user", "message": {"role": "user",
             "content": [{"type": "text", "text": "hi"}]}},
        ]
        path = _write_jsonl(tmp_path, records)
        claude.inspect_session(path, show_flow=True, show_blocks=True)
        # Just assert it produced output without raising.
        assert capsys.readouterr().out != ""

    def test_show_raw(self, tmp_path, capsys):
        records = [{"type": "user", "message": {"role": "user", "content": "hi"}}]
        path = _write_jsonl(tmp_path, records)
        claude.inspect_session(path, show_raw=True)
        out = capsys.readouterr().out
        assert "RAW RECORD EXAMPLES" in out

    def test_show_raw_not_shown_by_default(self, tmp_path, capsys):
        records = [{"type": "user", "message": {"role": "user", "content": "hi"}}]
        path = _write_jsonl(tmp_path, records)
        claude.inspect_session(path)
        assert "RAW RECORD EXAMPLES" not in capsys.readouterr().out

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            claude.inspect_session(tmp_path / "nope.jsonl")
