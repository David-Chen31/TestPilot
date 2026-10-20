"""Tests for claude_converter.claude public API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_converter.claude import (
    load_session,
    records_to_messages,
    session_to_messages,
    inspect_session,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _user(content, role="user"):
    return {
        "type": "user",
        "message": {"role": role, "content": content},
    }


def _assistant(content, role="assistant", usage=None):
    msg = {"role": role, "content": content}
    if usage is not None:
        msg["usage"] = usage
    return {"type": "assistant", "message": msg}


def _write_jsonl(tmp_path: Path, records: list[dict], name="session.jsonl") -> Path:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


# ── load_session ─────────────────────────────────────────────────────────


class TestLoadSession:
    def test_loads_valid_records(self, tmp_path):
        records = [_user("hello"), _assistant("hi there")]
        p = _write_jsonl(tmp_path, records)
        loaded = load_session(p)
        assert loaded == records

    def test_accepts_string_path(self, tmp_path):
        records = [_user("hi")]
        p = _write_jsonl(tmp_path, records)
        loaded = load_session(str(p))
        assert loaded == records

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(
            json.dumps(_user("a")) + "\n\n" + json.dumps(_assistant("b")) + "\n",
            encoding="utf-8",
        )
        loaded = load_session(p)
        assert len(loaded) == 2
        assert loaded[0]["type"] == "user"
        assert loaded[1]["type"] == "assistant"

    def test_accepts_json_extension(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps(_user("a")) + "\n", encoding="utf-8")
        assert load_session(p) == [_user("a")]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nope.jsonl")

    def test_bad_extension_raises(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text(json.dumps(_user("a")) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(p)

    def test_invalid_json_lines_skipped_but_valid_kept(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text(
            json.dumps(_user("a")) + "\n" + "{not json}\n" + json.dumps(_assistant("b")) + "\n",
            encoding="utf-8",
        )
        loaded = load_session(p)
        assert len(loaded) == 2
        assert loaded[0] == _user("a")
        assert loaded[1] == _assistant("b")
        err = capsys.readouterr().err
        assert "invalid JSON" in err


# ── records_to_messages ──────────────────────────────────────────────────


class TestRecordsToMessages:
    def test_empty_records(self):
        assert records_to_messages([]) == []

    def test_skips_non_message_types(self):
        records = [
            {"type": "system", "message": {"role": "system", "content": "sys"}},
            _user("hi"),
        ]
        msgs = records_to_messages(records)
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_string_content_basic(self):
        records = [_user("hello"), _assistant("world")]
        msgs = records_to_messages(records)
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_empty_content_skipped(self):
        records = [_user(""), _assistant("   ")]
        assert records_to_messages(records) == []

    def test_text_blocks_joined(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        msgs = records_to_messages([_user(content)])
        assert msgs == [{"role": "user", "content": "first\nsecond"}]

    def test_thinking_block_flattened(self):
        content = [{"type": "thinking", "thinking": "ponder"}]
        msgs = records_to_messages([_assistant(content)])
        assert msgs == [
            {"role": "assistant", "content": "<thinking>ponder</thinking>"}
        ]

    def test_tool_use_block_flattened(self):
        content = [
            {
                "type": "tool_use",
                "name": "Read",
                "input": {"path": "/x"},
            }
        ]
        msgs = records_to_messages([_assistant(content)])
        expected_input = json.dumps({"path": "/x"}, ensure_ascii=False)
        assert msgs == [
            {
                "role": "assistant",
                "content": f"<tool_use name='Read'>{expected_input}</tool_use>",
            }
        ]

    def test_tool_result_string_content(self):
        content = [
            {"type": "tool_result", "tool_use_id": "t1", "content": "output"},
        ]
        msgs = records_to_messages([_user(content)])
        assert msgs == [{"role": "user", "content": "<tool_result>output</tool_result>"}]

    def test_tool_result_nested_text_blocks(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ]
        msgs = records_to_messages([_user(content)])
        assert msgs == [{"role": "user", "content": "<tool_result>a b</tool_result>"}]

    def test_tool_result_nested_image_text_marked(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [
                    {"type": "text", "text": "img follows"},
                    {"type": "image", "source": {"type": "base64", "data": ""}},
                ],
            }
        ]
        msgs = records_to_messages([_user(content)])
        assert msgs == [
            {"role": "user", "content": "<tool_result>img follows <image></tool_result>"}
        ]

    def test_empty_tool_result_skipped(self):
        content = [{"type": "tool_result", "tool_use_id": "t1", "content": ""}]
        assert records_to_messages([_user(content)]) == []

    def test_consecutive_same_role_merged(self):
        records = [_user("one"), _user("two")]
        msgs = records_to_messages(records)
        assert msgs == [{"role": "user", "content": "one\ntwo"}]

    def test_alternating_roles_not_merged(self):
        records = [_user("a"), _assistant("b"), _user("c")]
        msgs = records_to_messages(records)
        assert msgs == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]

    def test_role_defaults_to_record_type_when_missing(self):
        record = {"type": "user", "message": {"content": "hi"}}
        msgs = records_to_messages([record])
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_null_message_handled(self):
        record = {"type": "user", "message": None}
        assert records_to_messages([record]) == []

    def test_unknown_block_type_produces_empty(self):
        content = [{"type": "mystery", "data": 1}]
        assert records_to_messages([_assistant(content)]) == []

    def test_image_block_with_bad_base64_skipped(self, tmp_path, capsys):
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "!!!notb64"},
            }
        ]
        # Only well-formed if Pillow is importable; unwrap to check the
        # empty-part skip behavior in either case.
        try:
            import PIL  # noqa: F401
        except ImportError:
            pytest.skip("Pillow not installed")
        msgs = records_to_messages([_user(content)])
        assert msgs == []
        err = capsys.readouterr().err
        assert "Skipping unreadable image block" in err

    def test_image_block_unsupported_source_type(self, capsys):
        content = [
            {"type": "image", "source": {"type": "url", "url": "http://x"}}
        ]
        msgs = records_to_messages([_user(content)])
        assert msgs == []
        err = capsys.readouterr().err
        assert "unsupported source type" in err

    def test_image_block_empty_data_warns(self, capsys):
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}
        ]
        msgs = records_to_messages([_user(content)])
        assert msgs == []
        err = capsys.readouterr().err
        assert "empty base64 data" in err

    def test_image_block_valid_kept_as_multimodal(self, capsys):
        pytest.importorskip("PIL")
        import base64
        from io import BytesIO

        from PIL import Image as PILImage

        buf = BytesIO()
        PILImage.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")

        content = [
            {"type": "text", "text": "look:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
        ]
        msgs = records_to_messages([_user(content)])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        parts = msgs[0]["content"]
        assert isinstance(parts, list)
        assert parts[0] == {"type": "text", "text": "look:"}
        assert parts[1]["type"] == "image"
        assert isinstance(parts[1]["image"], PILImage.Image)
        assert parts[1]["image"].size == (2, 2)


# ── session_to_messages ──────────────────────────────────────────────────


class TestSessionToMessages:
    def test_basic_conversion(self, tmp_path):
        p = _write_jsonl(tmp_path, [_user("a"), _assistant("b")])
        msgs = session_to_messages(p)
        assert msgs == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    def test_output_written(self, tmp_path):
        p = _write_jsonl(tmp_path, [_user("a"), _assistant("b")])
        out = tmp_path / "sub" / "messages.json"
        msgs = session_to_messages(p, output=out)

        assert out.exists()
        with open(out, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == msgs
        assert saved == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    def test_output_creates_parent_dirs(self, tmp_path):
        p = _write_jsonl(tmp_path, [_user("x")])
        out = tmp_path / "deep" / "nested" / "out.json"
        session_to_messages(p, output=out)
        assert out.exists()

    def test_output_accepts_string_path(self, tmp_path):
        p = _write_jsonl(tmp_path, [_user("x")])
        out = tmp_path / "out.json"
        session_to_messages(p, output=str(out))
        assert out.exists()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            session_to_messages(tmp_path / "missing.jsonl")

    def test_output_with_image_serialized_safely(self, tmp_path):
        pytest.importorskip("PIL")
        import base64
        from io import BytesIO

        from PIL import Image as PILImage

        buf = BytesIO()
        PILImage.new("RGB", (2, 2), color=(0, 255, 0)).save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")

        record = _user(
            [
                {"type": "text", "text": "pic"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": data},
                },
            ]
        )
        p = _write_jsonl(tmp_path, [record])
        out = tmp_path / "messages.json"
        msgs = session_to_messages(p, output=out)

        # In-memory keeps PIL objects
        parts = msgs[0]["content"]
        assert isinstance(parts, list)
        assert parts[1]["type"] == "image"
        assert isinstance(parts[1]["image"], PILImage.Image)

        # On-disk is JSON-safe
        with open(out, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved[0]["content"][1]["type"] == "image"
        assert saved[0]["content"][1]["image"].startswith("data:image/")


# ── inspect_session ──────────────────────────────────────────────────────


class TestInspectSession:
    def test_runs_and_prints_header(self, tmp_path, capsys):
        records = [
            {
                "type": "user",
                "sessionId": "abc",
                "cwd": "/home",
                "gitBranch": "main",
                "message": {"role": "user", "content": "hello"},
            },
            _assistant("reply", usage={"output_tokens": 5}),
        ]
        p = _write_jsonl(tmp_path, records)
        inspect_session(p)
        out = capsys.readouterr().out
        assert "CLAUDE CODE" in out

    def test_show_flow_and_blocks(self, tmp_path, capsys):
        records = [_user("hello"), _assistant([{"type": "text", "text": "hi"}])]
        p = _write_jsonl(tmp_path, records)
        inspect_session(p, show_flow=True, show_blocks=True)
        out = capsys.readouterr().out
        assert "CLAUDE CODE" in out

    def test_show_raw_prints_examples(self, tmp_path, capsys):
        records = [_user("hello"), _assistant("hi")]
        p = _write_jsonl(tmp_path, records)
        inspect_session(p, show_raw=True)
        out = capsys.readouterr().out
        assert "RAW RECORD EXAMPLES" in out
        assert "type = user" in out
        assert "type = assistant" in out

    def test_prints_session_metadata(self, tmp_path, capsys):
        records = [
            {
                "type": "user",
                "sessionId": "sid-123",
                "cwd": "/work/dir",
                "gitBranch": "feature",
                "message": {"role": "user", "content": "hi"},
            }
        ]
        p = _write_jsonl(tmp_path, records)
        inspect_session(p)
        out = capsys.readouterr().out
        assert "sid-123" in out or "Session" in out
        assert "/work/dir" in out

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_session(tmp_path / "missing.jsonl")
