"""Unit tests for claude_converter.pi (public API only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_converter.pi import (
    load_session_pi,
    records_to_messages_pi,
    session_to_messages_pi,
    inspect_session_pi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _msg(role: str, content, **extra) -> dict:
    message = {"role": role, "content": content}
    message.update(extra)
    return {"type": "message", "message": message}


# ---------------------------------------------------------------------------
# load_session_pi
# ---------------------------------------------------------------------------

class TestLoadSessionPi:
    def test_loads_records(self, tmp_path):
        path = tmp_path / "s.jsonl"
        records = [
            {"type": "session", "id": "abc"},
            _msg("user", "hello"),
        ]
        _write_jsonl(path, records)
        loaded = load_session_pi(path)
        assert loaded == records

    def test_accepts_str_path(self, tmp_path):
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [{"type": "session"}])
        assert load_session_pi(str(path)) == [{"type": "session"}]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "s.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write(json.dumps({"type": "session"}) + "\n")
            f.write("   \n")
        assert load_session_pi(path) == [{"type": "session"}]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session_pi(tmp_path / "nope.jsonl")

    def test_bad_extension_raises(self, tmp_path):
        path = tmp_path / "s.txt"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session_pi(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session_pi(path)


# ---------------------------------------------------------------------------
# records_to_messages_pi
# ---------------------------------------------------------------------------

class TestRecordsToMessagesPi:
    def test_skips_non_message_records(self):
        records = [
            {"type": "session", "id": "x"},
            {"type": "model_change", "model": "m"},
            _msg("user", "hi"),
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "hi"}
        ]

    def test_string_content(self):
        records = [_msg("user", "hello world")]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "hello world"}
        ]

    def test_text_blocks_joined(self):
        records = [
            _msg("assistant", [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ])
        ]
        assert records_to_messages_pi(records) == [
            {"role": "assistant", "content": "first\nsecond"}
        ]

    def test_empty_text_block_dropped(self):
        records = [
            _msg("user", [
                {"type": "text", "text": ""},
                {"type": "text", "text": "kept"},
            ])
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "kept"}
        ]

    def test_thinking_block(self):
        records = [
            _msg("assistant", [{"type": "thinking", "thinking": "ponder"}])
        ]
        assert records_to_messages_pi(records) == [
            {"role": "assistant", "content": "<thinking>ponder</thinking>"}
        ]

    def test_tool_call_block(self):
        records = [
            _msg("assistant", [
                {"type": "toolCall", "name": "run", "arguments": {"a": 1}},
            ])
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {"role": "assistant", "content": "<tool_use name='run'>{\"a\": 1}</tool_use>"}
        ]

    def test_tool_call_default_arguments(self):
        records = [
            _msg("assistant", [{"type": "toolCall", "name": "noargs"}])
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {"role": "assistant", "content": "<tool_use name='noargs'>{}</tool_use>"}
        ]

    def test_unknown_block_type_dropped(self):
        records = [
            _msg("user", [
                {"type": "weird", "data": "x"},
                {"type": "text", "text": "ok"},
            ])
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "ok"}
        ]

    def test_unknown_role_skipped(self):
        records = [_msg("system", "ignored")]
        assert records_to_messages_pi(records) == []

    def test_missing_message_skipped(self):
        records = [{"type": "message", "message": None}]
        assert records_to_messages_pi(records) == []

    def test_empty_parts_skipped(self):
        records = [
            _msg("user", ""),
            _msg("assistant", []),
        ]
        assert records_to_messages_pi(records) == []

    def test_non_dict_content_block_skipped(self):
        records = [_msg("user", ["not-a-dict", {"type": "text", "text": "x"}])]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "x"}
        ]

    def test_consecutive_same_role_merged(self):
        records = [
            _msg("user", "one"),
            _msg("user", "two"),
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "one\ntwo"}
        ]

    def test_alternating_roles_not_merged(self):
        records = [
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("user", "u2"),
        ]
        assert records_to_messages_pi(records) == [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]

    def test_tool_result_becomes_user(self):
        records = [
            _msg(
                "toolResult",
                [{"type": "text", "text": "result text"}],
                toolName="exec",
                toolCallId="call-1",
            )
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {
                "role": "user",
                "content": "<tool_result name='exec' id='call-1'>result text</tool_result>",
            }
        ]

    def test_tool_result_with_image_text_flatten(self):
        records = [
            _msg(
                "toolResult",
                [
                    {"type": "text", "text": "out"},
                    {"type": "image", "data": ""},
                ],
                toolName="t",
                toolCallId="id",
            )
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {
                "role": "user",
                "content": "<tool_result name='t' id='id'>out\n<image></tool_result>",
            }
        ]

    def test_tool_result_empty_content_parts(self):
        records = [
            _msg("toolResult", [], toolName="t", toolCallId="i")
        ]
        assert records_to_messages_pi(records) == []

    def test_tool_result_merges_with_user(self):
        records = [
            _msg("user", "before"),
            _msg("toolResult", [{"type": "text", "text": "out"}],
                 toolName="t", toolCallId="i"),
        ]
        out = records_to_messages_pi(records)
        assert out == [
            {
                "role": "user",
                "content": (
                    "before\n"
                    "<tool_result name='t' id='i'>out</tool_result>"
                ),
            }
        ]


# ---------------------------------------------------------------------------
# Image handling (via records_to_messages_pi)
# ---------------------------------------------------------------------------

class TestImageHandling:
    def test_empty_image_data_skipped(self, capsys):
        records = [
            _msg("user", [
                {"type": "image", "data": "", "mimeType": "image/png"},
                {"type": "text", "text": "after"},
            ])
        ]
        out = records_to_messages_pi(records)
        assert out == [{"role": "user", "content": "after"}]
        err = capsys.readouterr().err
        assert "Skipping image block with empty base64 data" in err

    def test_undecodable_image_skipped(self, capsys):
        records = [
            _msg("user", [
                {"type": "image", "data": "!!!notbase64!!!", "mimeType": "image/png"},
                {"type": "text", "text": "after"},
            ])
        ]
        out = records_to_messages_pi(records)
        assert out == [{"role": "user", "content": "after"}]
        err = capsys.readouterr().err
        # Either Pillow is missing (ImportError) or the payload cannot be
        # decoded (ValueError); both produce the "Skipping unreadable" line.
        assert "Skipping unreadable image block" in err

    def test_valid_image_kept_as_multimodal(self):
        pytest.importorskip("PIL.Image")
        import base64
        import io
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        records = [
            _msg("user", [
                {"type": "image", "data": b64, "mimeType": "image/png"},
                {"type": "text", "text": "caption"},
            ])
        ]
        out = records_to_messages_pi(records)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        content = out[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image"
        assert isinstance(content[0]["image"], Image.Image)
        assert content[1] == {"type": "text", "text": "caption"}

    def test_tool_result_with_valid_image(self):
        pytest.importorskip("PIL.Image")
        import base64
        import io
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (2, 2), (0, 255, 0)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        records = [
            _msg(
                "toolResult",
                [
                    {"type": "text", "text": "see image"},
                    {"type": "image", "data": b64, "mimeType": "image/png"},
                ],
                toolName="snap",
                toolCallId="c9",
            )
        ]
        out = records_to_messages_pi(records)
        assert len(out) == 1
        content = out[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {
            "type": "text",
            "text": "<tool_result name='snap' id='c9'>",
        }
        assert content[1] == {"type": "text", "text": "see image"}
        assert content[2]["type"] == "image"
        assert isinstance(content[2]["image"], Image.Image)
        assert content[3] == {"type": "text", "text": "</tool_result>"}


# ---------------------------------------------------------------------------
# session_to_messages_pi
# ---------------------------------------------------------------------------

class TestSessionToMessagesPi:
    def test_in_memory_only(self, tmp_path):
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [_msg("user", "hi")])
        out = session_to_messages_pi(path)
        assert out == [{"role": "user", "content": "hi"}]

    def test_writes_output_file(self, tmp_path):
        src = tmp_path / "s.jsonl"
        _write_jsonl(src, [_msg("user", "hi"), _msg("assistant", "yo")])
        out_path = tmp_path / "nested" / "out.json"
        result = session_to_messages_pi(src, output=out_path)

        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        assert out_path.exists()
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))
        assert on_disk == result

    def test_output_serializes_images(self, tmp_path):
        pytest.importorskip("PIL.Image")
        import base64
        import io
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (2, 2), (0, 0, 255)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        src = tmp_path / "s.jsonl"
        _write_jsonl(src, [
            _msg("user", [
                {"type": "image", "data": b64, "mimeType": "image/png"},
                {"type": "text", "text": "cap"},
            ])
        ])
        out_path = tmp_path / "out.json"
        result = session_to_messages_pi(src, output=out_path)

        # In-memory keeps a real PIL.Image
        assert isinstance(result[0]["content"][0]["image"], Image.Image)

        # On-disk is a base64 data URI string
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))
        image_part = on_disk[0]["content"][0]
        assert image_part["type"] == "image"
        assert isinstance(image_part["image"], str)
        assert image_part["image"].startswith("data:image/")
        assert ";base64," in image_part["image"]

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            session_to_messages_pi(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# inspect_session_pi
# ---------------------------------------------------------------------------

class TestInspectSessionPi:
    def _sample(self, tmp_path: Path) -> Path:
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [
            {"type": "session", "id": "abc", "timestamp": "2024-01-01T00:00:00Z"},
            _msg("user", "hello", usage={"input": 5, "output": 2,
                                         "cacheRead": 1, "cacheWrite": 0}),
            _msg("assistant", "hi", usage={"input": 3, "output": 4}),
        ])
        return path

    def test_basic_run(self, tmp_path, capsys):
        path = self._sample(tmp_path)
        inspect_session_pi(path)
        out = capsys.readouterr().out
        assert isinstance(out, str)
        assert len(out) > 0

    def test_with_all_flags(self, tmp_path, capsys):
        path = self._sample(tmp_path)
        inspect_session_pi(path, show_flow=True, show_blocks=True, show_raw=True)
        out = capsys.readouterr().out
        assert isinstance(out, str)
        assert len(out) > 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_session_pi(tmp_path / "nope.jsonl")
