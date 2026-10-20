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


# ── helpers ──────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _user(content):
    return {"type": "user", "message": {"role": "user", "content": content}}


def _assistant(content):
    return {"type": "assistant", "message": {"role": "assistant", "content": content}}


# A tiny valid PNG (1x1 transparent) as base64, for image tests.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _pil_available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


class _FakeImage:
    """Stand-in for a PIL.Image.Image when Pillow is not installed."""


def _fake_convert(_data):
    return _FakeImage()


@pytest.fixture
def fake_pil(monkeypatch):
    """Simulate a working Pillow-backed decoder without requiring Pillow.

    Patches convert_base64_to_pil_image in the utils module (which the claude
    module imports at module load) to return a sentinel object.
    """
    monkeypatch.setattr(
        claude, "convert_base64_to_pil_image", _fake_convert
    )
    return _FakeImage


# ── _block_to_str ────────────────────────────────────────────────────────

class TestBlockToStr:
    def test_text(self):
        assert claude._block_to_str({"type": "text", "text": "hi"}) == "hi"

    def test_thought(self):
        assert claude._block_to_str({"type": "thinking", "thinking": "hm"}) == (
            "<thinking>hm</thinking>"
        )

    def test_tool_use(self):
        block = {"type": "tool_use", "name": "Read", "input": {"path": "a"}}
        assert claude._block_to_str(block) == (
            '<tool_use name=\'Read\'>{"path": "a"}</tool_use>'
        )

    def test_tool_result_string_content(self):
        block = {"type": "tool_result", "content": "result"}
        assert claude._block_to_str(block) == "<tool_result>result</tool_result>"

    def test_tool_result_list_content(self):
        block = {
            "type": "tool_result",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "b"},
                "notadict",
            ],
        }
        assert claude._block_to_str(block) == "<tool_result>a <image> b</tool_result>"

    def test_image(self):
        assert claude._block_to_str({"type": "image"}) == "<image>"

    def test_unknown(self):
        assert claude._block_to_str({"type": "weird"}) == ""

    def test_missing_type(self):
        assert claude._block_to_str({}) == ""


# ── _decode_claude_image ─────────────────────────────────────────────────

class TestDecodeClaudeImage:
    def test_unsupported_source_type(self, capsys):
        block = {"source": {"type": "url", "data": "x"}}
        assert claude._decode_claude_image(block) is None
        err = capsys.readouterr().err
        assert "unsupported source type" in err
        assert "'url'" in err

    def test_empty_data(self, capsys):
        block = {"source": {"type": "base64", "data": ""}}
        assert claude._decode_claude_image(block) is None
        assert "empty base64 data" in capsys.readouterr().err

    def test_missing_source_treated_as_unsupported(self, capsys):
        block = {}
        assert claude._decode_claude_image(block) is None
        err = capsys.readouterr().err
        assert "unsupported source type" in err
        assert "None" in err

    def test_no_pil_returns_none(self, capsys):
        # Pillow is not required for the base64 path when it fails to import:
        # _decode_claude_image catches ImportError and returns None.
        if _pil_available():
            pytest.skip("Pillow installed; ImportError path not exercised")
        block = {
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64}
        }
        assert claude._decode_claude_image(block) is None
        assert "unreadable image block" in capsys.readouterr().err

    def test_bad_base64(self, capsys, monkeypatch):
        # Simulate a decoder that raises ValueError for corrupt base64,
        # without depending on Pillow being installed.
        def _boom(_data):
            raise ValueError("Could not decode base64 image: corrupt")

        monkeypatch.setattr(claude, "convert_base64_to_pil_image", _boom)
        block = {"source": {"type": "base64", "data": "!!!not base64!!!"}}
        assert claude._decode_claude_image(block) is None
        assert "unreadable image block" in capsys.readouterr().err

    def test_importerror_returns_none(self, capsys, monkeypatch):
        def _boom(_data):
            raise ImportError("Pillow missing")

        monkeypatch.setattr(claude, "convert_base64_to_pil_image", _boom)
        block = {"source": {"type": "base64", "data": _PNG_B64}}
        assert claude._decode_claude_image(block) is None
        assert "unreadable image block" in capsys.readouterr().err

    def test_valid(self, monkeypatch):
        sentinel = object()

        def _decode(_data):
            return sentinel

        monkeypatch.setattr(claude, "convert_base64_to_pil_image", _decode)
        block = {
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64}
        }
        assert claude._decode_claude_image(block) is sentinel

    def test_valid_real_pil(self):
        if not _pil_available():
            pytest.skip("Pillow not installed")
        from PIL import Image

        block = {
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64}
        }
        assert isinstance(claude._decode_claude_image(block), Image.Image)


# ── _block_to_parts ──────────────────────────────────────────────────────

class TestBlockToParts:
    def test_text(self):
        assert claude._block_to_parts({"type": "text", "text": "hi"}) == [
            {"type": "text", "text": "hi"}
        ]

    def test_empty_text(self):
        assert claude._block_to_parts({"type": "text", "text": "   "}) == []

    def test_image_valid(self, fake_pil):
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64},
        }
        parts = claude._block_to_parts(block)
        assert len(parts) == 1
        assert parts[0]["type"] == "image"
        assert isinstance(parts[0]["image"], fake_pil)

    def test_image_invalid(self, capsys):
        block = {"type": "image", "source": {"type": "base64", "data": ""}}
        assert claude._block_to_parts(block) == []
        assert "empty base64 data" in capsys.readouterr().err

    def test_image_no_pil_returns_empty(self, capsys):
        if _pil_available():
            pytest.skip("Pillow installed; no-PIL path not exercised")
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64},
        }
        assert claude._block_to_parts(block) == []
        assert "unreadable image block" in capsys.readouterr().err

    def test_tool_result_text_only(self):
        block = {"type": "tool_result", "content": "output"}
        assert claude._block_to_parts(block) == [
            {"type": "text", "text": "<tool_result>output</tool_result>"}
        ]

    def test_tool_result_blank_text_only(self):
        # The whitespace-only tool_result is wrapped in tags by _block_to_str,
        # which is not blank (contains "<tool_result>"), so a text part is kept.
        block = {"type": "tool_result", "content": "   "}
        assert claude._block_to_parts(block) == [
            {"type": "text", "text": "<tool_result>   </tool_result>"}
        ]

    def test_tool_result_with_image(self, fake_pil):
        block = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [
                {"type": "text", "text": "first"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _PNG_B64,
                    },
                },
                {"type": "text", "text": "  "},
            ],
        }
        parts = claude._block_to_parts(block)
        assert parts[0] == {"type": "text", "text": "<tool_result id='t1'>"}
        assert parts[1] == {"type": "text", "text": "first"}
        assert parts[2]["type"] == "image"
        assert isinstance(parts[2]["image"], fake_pil)
        assert parts[-1] == {"type": "text", "text": "</tool_result>"}
        # empty text block skipped
        assert len(parts) == 4

    def test_tool_result_with_image_no_id(self, fake_pil):
        block = {
            "type": "tool_result",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _PNG_B64,
                    },
                }
            ],
        }
        parts = claude._block_to_parts(block)
        assert parts[0] == {"type": "text", "text": "<tool_result id='?'>"}


# ── _content_to_str / _content_to_parts / _collect_block_types ────────────

class TestContentHelpers:
    def test_content_to_str_string(self):
        assert claude._content_to_str("hello") == "hello"

    def test_content_to_str_list(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert claude._content_to_str(content) == "a\nb"

    def test_content_to_str_skips_empty(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": " "}]
        assert claude._content_to_str(content) == "a"

    def test_content_to_str_none(self):
        assert claude._content_to_str(None) == ""

    def test_content_to_parts_string(self):
        assert claude._content_to_parts("hi") == [{"type": "text", "text": "hi"}]

    def test_content_to_parts_blank_string(self):
        assert claude._content_to_parts("   ") == []

    def test_content_to_parts_none(self):
        assert claude._content_to_parts(None) == []

    def test_collect_block_types_string(self):
        assert claude._collect_block_types("raw") == ["raw_string"]

    def test_collect_block_types_list(self):
        content = [{"type": "text"}, {"type": "tool_use"}, "notdict"]
        assert claude._collect_block_types(content) == ["text", "tool_use"]

    def test_collect_block_types_none(self):
        assert claude._collect_block_types(None) == ["unknown"]


# ── load_session ─────────────────────────────────────────────────────────

class TestLoadSession:
    def test_valid(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [{"type": "user"}, {"type": "assistant"}])
        records = load_session(p)
        assert records == [{"type": "user"}, {"type": "assistant"}]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nope.jsonl")

    def test_bad_extension(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text("{}\n")
        with pytest.raises(ValueError, match="Unrecognized extension"):
            load_session(p)

    def test_empty(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("\n\n")
        with pytest.raises(ValueError, match="no valid records"):
            load_session(p)

    def test_skips_invalid_json_lines(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({"type": "user"}) + "\nnotjson\n")
        records = load_session(p)
        assert records == [{"type": "user"}]
        assert "invalid JSON" in capsys.readouterr().err


# ── records_to_messages ──────────────────────────────────────────────────

class TestRecordsToMessages:
    def test_skips_non_message_records(self):
        records = [
            {"type": "system", "message": {"content": "x"}},
            _user("hi"),
        ]
        assert records_to_messages(records) == [{"role": "user", "content": "hi"}]

    def test_role_from_message(self):
        records = [{"type": "user", "message": {"role": "assistant", "content": "yo"}}]
        assert records_to_messages(records) == [{"role": "assistant", "content": "yo"}]

    def test_role_fallback_to_rtype(self):
        records = [{"type": "user", "message": {"content": "yo"}}]
        assert records_to_messages(records) == [{"role": "user", "content": "yo"}]

    def test_skips_empty_parts(self):
        records = [{"type": "user", "message": {"content": "   "}}]
        assert records_to_messages(records) == []

    def test_merges_consecutive_same_role(self):
        records = [_user("a"), _user("b")]
        result = records_to_messages(records)
        assert result == [{"role": "user", "content": "a\nb"}]

    def test_alternating_roles_not_merged(self):
        records = [_user("a"), _assistant("b")]
        result = records_to_messages(records)
        assert result == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    def test_list_content_flattened(self):
        records = [_user([{"type": "text", "text": "hello"}])]
        assert records_to_messages(records) == [{"role": "user", "content": "hello"}]

    def test_tool_use_flattened(self):
        block = {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}
        records = [_assistant([block])]
        result = records_to_messages(records)
        assert result == [
            {
                "role": "assistant",
                "content": '<tool_use name=\'Bash\'>{"command": "ls"}</tool_use>',
            }
        ]

    def test_image_kept_as_part(self, fake_pil):
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64},
        }
        records = [_user([{"type": "text", "text": "look"}, block])]
        result = records_to_messages(records)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "look"}
        assert content[1]["type"] == "image"
        assert isinstance(content[1]["image"], fake_pil)

    def test_null_message_safe(self):
        records = [{"type": "user", "message": None}]
        assert records_to_messages(records) == []


# ── session_to_messages ──────────────────────────────────────────────────

class TestSessionToMessages:
    def test_basic(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [_user("hi"), _assistant("hello")])
        result = session_to_messages(p)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_output_written(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [_user("hi")])
        out = tmp_path / "sub" / "out.json"
        result = session_to_messages(p, output=out)
        assert result == [{"role": "user", "content": "hi"}]
        assert out.exists()
        saved = json.loads(out.read_text())
        assert saved == [{"role": "user", "content": "hi"}]

    def test_output_with_image(self, tmp_path):
        if not _pil_available():
            pytest.skip("Pillow not installed; cannot build a real PIL image")
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64},
        }
        p = _write_jsonl(tmp_path / "s.jsonl", [_user([block])])
        out = tmp_path / "out.json"
        result = session_to_messages(p, output=out)
        # in-memory still PIL
        assert isinstance(result[0]["content"], list)
        saved = json.loads(out.read_text())
        image_part = saved[0]["content"][0]
        assert image_part["type"] == "image"
        assert isinstance(image_part["image"], str)
        assert image_part["image"].startswith("data:image/")


# ── inspect_session ──────────────────────────────────────────────────────

class TestInspectSession:
    def test_runs_and_prints(self, tmp_path, capsys):
        records = [
            {
                "type": "user",
                "sessionId": "sess-1",
                "cwd": "/tmp/project",
                "gitBranch": "main",
                "message": {
                    "role": "user",
                    "content": "hello",
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "cache_read_input_tokens": 1,
                        "cache_creation_input_tokens": 2,
                    },
                },
            },
            _assistant("world"),
        ]
        p = _write_jsonl(tmp_path / "s.jsonl", records)
        inspect_session(p, show_flow=True, show_blocks=True, show_raw=True)
        out = capsys.readouterr().out
        assert "CLAUDE CODE" in out
        assert "sess-1" in out
        assert "/tmp/project" in out
        assert "main" in out
        assert "RAW RECORD EXAMPLES" in out

    def test_metadata_no_user(self, tmp_path, capsys):
        p = _write_jsonl(tmp_path / "s.jsonl", [_assistant("x")])
        inspect_session(p)
        out = capsys.readouterr().out
        assert "Session" not in out


# ── schema callbacks ─────────────────────────────────────────────────────

class TestSchemaCallbacks:
    def test_record_type(self):
        assert claude._record_type_of({"type": "user"}) == "user"
        assert claude._record_type_of({}) == "unknown"

    def test_is_message(self):
        assert claude._is_message({"type": "user"})
        assert claude._is_message({"type": "assistant"})
        assert not claude._is_message({"type": "system"})

    def test_role_of(self):
        assert claude._role_of({"message": {"role": "user"}}) == "user"
        assert claude._role_of({"type": "assistant"}) == "assistant"

    def test_text_of(self):
        assert claude._text_of({"message": {"content": "hi"}}) == "hi"

    def test_timestamp_of(self):
        assert claude._timestamp_of({"timestamp": "2024-01-01T12:00:00Z"}) == (
            "2024-01-01 12:00:00"
        )
        assert claude._timestamp_of({}) == ""

    def test_tokens_of(self):
        assert claude._tokens_of({"message": {"usage": {"output_tokens": 7}}}) == 7
        assert claude._tokens_of({"message": {}}) == 0

    def test_block_types_of(self):
        assert claude._block_types_of(
            {"message": {"content": [{"type": "text"}]}}
        ) == ["text"]

    def test_blocks_of(self):
        r = {"message": {"content": [{"type": "text"}, "x"]}}
        assert claude._blocks_of(r) == [{"type": "text"}]

    def test_blocks_of_string_content(self):
        assert claude._blocks_of({"message": {"content": "hi"}}) == []

    def test_total_tokens_of(self):
        records = [
            {
                "message": {
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 4,
                    }
                }
            }
        ]
        totals = claude._total_tokens_of(records)
        assert totals == {
            "input": 1,
            "output": 2,
            "cache_read": 3,
            "cache_write": 4,
            "total": 3,
        }

    def test_total_tokens_of_empty(self):
        assert claude._total_tokens_of([]) == {}


# ── _render_block ────────────────────────────────────────────────────────

class TestRenderBlock:
    def test_text(self):
        out = claude._render_block({"type": "text", "text": "hi"})
        assert "text" in out
        assert "hi" in out

    def test_tool_use(self):
        block = {"type": "tool_use", "name": "Bash", "id": "abc", "input": {"x": 1}}
        out = claude._render_block(block)
        assert "Bash" in out
        assert "abc" in out

    def test_tool_result_list(self):
        block = {
            "type": "tool_result",
            "tool_use_id": "t",
            "content": [
                {"type": "image", "source": {"media_type": "image/png", "data": "abcd"}},
                {"type": "text", "text": "hi"},
            ],
        }
        out = claude._render_block(block)
        assert "image: image/png (4 base64 chars)" in out
        assert "content: hi" in out

    def test_tool_result_string(self):
        block = {"type": "tool_result", "content": "raw"}
        out = claude._render_block(block)
        assert "content: raw" in out

    def test_image(self):
        block = {
            "type": "image",
            "source": {"media_type": "image/jpeg", "data": "12345"},
        }
        out = claude._render_block(block)
        assert "image/jpeg" in out
        assert "(5 base64 chars)" in out

    def test_unknown(self):
        out = claude._render_block({"type": "mystery", "x": 1})
        assert "mystery" in out
