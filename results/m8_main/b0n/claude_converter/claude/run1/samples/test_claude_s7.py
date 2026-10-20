"""Unit tests for claude_converter.claude."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from claude_converter.claude import (
    _block_to_parts,
    _block_to_str,
    _collect_block_types,
    _content_to_parts,
    _content_to_str,
    _decode_claude_image,
    _is_message,
    _record_type_of,
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
    inspect_session,
    load_session,
    records_to_messages,
    session_to_messages,
)


# ── helpers ────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _make_png_b64() -> str:
    """Return a tiny valid PNG as a base64 string (uses Pillow if available)."""
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── _block_to_str ──────────────────────────────────────────────────────


def test_block_to_str_text():
    assert _block_to_str({"type": "text", "text": "hello"}) == "hello"


def test_block_to_str_text_missing_text():
    assert _block_to_str({"type": "text"}) == ""


def test_block_to_str_thinking():
    assert _block_to_str({"type": "thinking", "thinking": "hmm"}) == "<thinking>hmm</thinking>"


def test_block_to_str_tool_use():
    out = _block_to_str(
        {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}}
    )
    assert out == "<tool_use name='Bash'>{\"cmd\": \"ls\"}</tool_use>"


def test_block_to_str_tool_use_no_input():
    out = _block_to_str({"type": "tool_use", "name": "X"})
    assert out == "<tool_use name='X'>{}</tool_use>"


def test_block_to_str_tool_result_string_content():
    out = _block_to_str({"type": "tool_result", "content": "ok"})
    assert out == "<tool_result>ok</tool_result>"


def test_block_to_str_tool_result_list_mixed():
    out = _block_to_str(
        {
            "type": "tool_result",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image"},
                {"type": "text", "text": "b"},
                "not-a-dict",
            ],
        }
    )
    assert out == "<tool_result>a <image> b</tool_result>"


def test_block_to_str_image():
    assert _block_to_str({"type": "image"}) == "<image>"


def test_block_to_str_unknown():
    assert _block_to_str({"type": "something"}) == ""
    assert _block_to_str({}) == ""


# ── _decode_claude_image ──────────────────────────────────────────────


def test_decode_image_wrong_source_type(capsys):
    result = _decode_claude_image({"source": {"type": "url"}})
    assert result is None
    err = capsys.readouterr().err
    assert "unsupported source type" in err


def test_decode_image_empty_data(capsys):
    result = _decode_claude_image({"source": {"type": "base64", "data": ""}})
    assert result is None
    err = capsys.readouterr().err
    assert "empty base64 data" in err


def test_decode_image_missing_source(capsys):
    # dget returns {} → source.get("type") is None, unsupported
    result = _decode_claude_image({})
    assert result is None
    err = capsys.readouterr().err
    assert "unsupported source type" in err


def test_decode_image_bad_base64(capsys):
    result = _decode_claude_image(
        {"source": {"type": "base64", "data": "!!!not-base64!!!"}}
    )
    assert result is None
    err = capsys.readouterr().err
    assert "unreadable image block" in err


def test_decode_image_valid():
    b64 = _make_png_b64()
    img = _decode_claude_image({"source": {"type": "base64", "data": b64}})
    assert img is not None
    assert img.size == (1, 1)


# ── _block_to_parts ───────────────────────────────────────────────────


def test_block_to_parts_text():
    assert _block_to_parts({"type": "text", "text": "hi"}) == [
        {"type": "text", "text": "hi"}
    ]


def test_block_to_parts_empty_text_dropped():
    assert _block_to_parts({"type": "text", "text": "   "}) == []


def test_block_to_parts_unknown():
    assert _block_to_parts({"type": "weird"}) == []


def test_block_to_parts_image_valid():
    b64 = _make_png_b64()
    parts = _block_to_parts({"type": "image", "source": {"type": "base64", "data": b64}})
    assert len(parts) == 1
    assert parts[0]["type"] == "image"
    assert parts[0]["image"].size == (1, 1)


def test_block_to_parts_image_invalid(capsys):
    parts = _block_to_parts({"type": "image", "source": {"type": "base64", "data": ""}})
    assert parts == []


def test_block_to_parts_tool_result_text_only():
    parts = _block_to_parts(
        {"type": "tool_result", "content": "done", "tool_use_id": "t1"}
    )
    assert parts == [{"type": "text", "text": "<tool_result>done</tool_result>"}]


def test_block_to_parts_tool_result_text_only_blank():
    parts = _block_to_parts({"type": "tool_result", "content": ""})
    assert parts == []


def test_block_to_parts_tool_result_with_nested_image():
    b64 = _make_png_b64()
    parts = _block_to_parts(
        {
            "type": "tool_result",
            "tool_use_id": "t42",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "source": {"type": "base64", "data": b64}},
                {"type": "text", "text": "  "},  # blank dropped
                {"type": "text", "text": "after"},
                "not-a-dict",
            ],
        }
    )
    assert parts[0] == {"type": "text", "text": "<tool_result id='t42'>"}
    assert parts[1] == {"type": "text", "text": "before"}
    assert parts[2]["type"] == "image"
    assert parts[3] == {"type": "text", "text": "after"}
    assert parts[-1] == {"type": "text", "text": "</tool_result>"}


def test_block_to_parts_tool_result_with_bad_image():
    parts = _block_to_parts(
        {
            "type": "tool_result",
            "tool_use_id": "tX",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": ""}},
                {"type": "text", "text": "kept"},
            ],
        }
    )
    # opening tag, kept text, closing tag; bad image dropped
    assert parts[0] == {"type": "text", "text": "<tool_result id='tX'>"}
    assert parts[1] == {"type": "text", "text": "kept"}
    assert parts[2] == {"type": "text", "text": "</tool_result>"}


def test_block_to_parts_tool_result_default_id():
    parts = _block_to_parts(
        {
            "type": "tool_result",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": ""}},
            ],
        }
    )
    assert parts[0] == {"type": "text", "text": "<tool_result id='?'>"}


# ── _content_to_str / _content_to_parts ───────────────────────────────


def test_content_to_str_string():
    assert _content_to_str("abc") == "abc"


def test_content_to_str_list():
    content = [
        {"type": "text", "text": "one"},
        {"type": "text", "text": "  "},
        {"type": "thinking", "thinking": "t"},
        "not-a-dict",
    ]
    assert _content_to_str(content) == "one\n<thinking>t</thinking>"


def test_content_to_str_other():
    assert _content_to_str(None) == ""
    assert _content_to_str(42) == ""


def test_content_to_parts_string():
    assert _content_to_parts("hello") == [{"type": "text", "text": "hello"}]


def test_content_to_parts_string_blank():
    assert _content_to_parts("  ") == []


def test_content_to_parts_list():
    parts = _content_to_parts(
        [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    )
    assert parts == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


def test_content_to_parts_other():
    assert _content_to_parts(None) == []


# ── _collect_block_types ──────────────────────────────────────────────


def test_collect_block_types_string():
    assert _collect_block_types("hi") == ["raw_string"]


def test_collect_block_types_list():
    assert _collect_block_types(
        [{"type": "text"}, {"type": "image"}, "x", {}]
    ) == ["text", "image", "unknown"]


def test_collect_block_types_other():
    assert _collect_block_types(None) == ["unknown"]


# ── schema accessor helpers ───────────────────────────────────────────


def test_record_type_of():
    assert _record_type_of({"type": "user"}) == "user"
    assert _record_type_of({}) == "unknown"


def test_is_message():
    assert _is_message({"type": "user"}) is True
    assert _is_message({"type": "assistant"}) is True
    assert _is_message({"type": "system"}) is False
    assert _is_message({}) is False


def test_role_of():
    assert _role_of({"type": "user", "message": {"role": "assistant"}}) == "assistant"
    assert _role_of({"type": "user"}) == "user"
    assert _role_of({"type": "weird"}) == "weird"


def test_text_of():
    assert _text_of({"message": {"content": "hi"}}) == "hi"
    assert _text_of({"message": {"content": [{"type": "text", "text": "a"}]}}) == "a"
    assert _text_of({}) == ""


def test_timestamp_of():
    r = {"timestamp": "2024-01-02T03:04:05.678Z"}
    assert _timestamp_of(r) == "2024-01-02 03:04:05"
    assert _timestamp_of({}) == ""


def test_tokens_of():
    r = {"message": {"usage": {"output_tokens": 17}}}
    assert _tokens_of(r) == 17
    assert _tokens_of({}) == 0


def test_block_types_of():
    r = {"message": {"content": [{"type": "text"}, {"type": "tool_use"}]}}
    assert _block_types_of(r) == ["text", "tool_use"]


def test_blocks_of():
    blocks = [{"type": "text"}, "not-a-dict"]
    r = {"message": {"content": blocks}}
    assert _blocks_of(r) == [{"type": "text"}]
    assert _blocks_of({"message": {"content": "str"}}) == []


# ── _total_tokens_of ──────────────────────────────────────────────────


def test_total_tokens_of_empty():
    assert _total_tokens_of([]) == {}
    assert _total_tokens_of([{"message": {}}]) == {}


def test_total_tokens_of_sums():
    records = [
        {
            "message": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 2,
                    "cache_creation_input_tokens": 3,
                }
            }
        },
        {
            "message": {
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
            }
        },
    ]
    totals = _total_tokens_of(records)
    assert totals == {
        "input": 11,
        "output": 6,
        "cache_read": 2,
        "cache_write": 3,
        "total": 17,
    }


def test_total_tokens_of_handles_null_usage():
    assert _total_tokens_of([{"message": {"usage": None}}]) == {}


# ── CLAUDE_SCHEMA ────────────────────────────────────────────────────


def test_claude_schema_label():
    assert CLAUDE_SCHEMA.label == "CLAUDE CODE"
    assert CLAUDE_SCHEMA.is_message({"type": "user"}) is True
    assert CLAUDE_SCHEMA.blocks_of is not None
    assert CLAUDE_SCHEMA.render_block is not None


# ── load_session ─────────────────────────────────────────────────────


def test_load_session_ok(tmp_path):
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [{"type": "user", "x": 1}, {"type": "assistant", "y": 2}])
    assert load_session(p) == [{"type": "user", "x": 1}, {"type": "assistant", "y": 2}]


def test_load_session_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "nope.jsonl")


def test_load_session_bad_extension(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("{}\n")
    with pytest.raises(ValueError):
        load_session(p)


def test_load_session_empty(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError):
        load_session(p)


# ── records_to_messages ─────────────────────────────────────────────


def test_records_to_messages_basic():
    records = [
        {"type": "system", "message": {"role": "system", "content": "sys"}},
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "hey"}},
    ]
    msgs = records_to_messages(records)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]


def test_records_to_messages_merges_consecutive_same_role():
    records = [
        {"type": "user", "message": {"role": "user", "content": "a"}},
        {"type": "user", "message": {"role": "user", "content": "b"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "c"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "d"}},
    ]
    msgs = records_to_messages(records)
    assert msgs == [
        {"role": "user", "content": "a\nb"},
        {"role": "assistant", "content": "c\nd"},
    ]


def test_records_to_messages_skips_empty_parts():
    records = [
        {"type": "user", "message": {"role": "user", "content": "  "}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": ""}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": "ok"}},
    ]
    msgs = records_to_messages(records)
    assert msgs == [{"role": "assistant", "content": "ok"}]


def test_records_to_messages_role_fallback_to_type():
    records = [{"type": "user", "message": {"content": "hi"}}]
    msgs = records_to_messages(records)
    assert msgs == [{"role": "user", "content": "hi"}]


def test_records_to_messages_with_image_keeps_list():
    b64 = _make_png_b64()
    records = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "data": b64}},
                ],
            },
        }
    ]
    msgs = records_to_messages(records)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"
    assert content[1]["image"].size == (1, 1)


def test_records_to_messages_tool_blocks_text():
    records = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"a": 1}},
                ],
            },
        },
    ]
    msgs = records_to_messages(records)
    assert msgs == [
        {
            "role": "assistant",
            "content": "<tool_use name='Bash'>{\"a\": 1}</tool_use>",
        }
    ]


def test_records_to_messages_merge_with_image_joins_text():
    b64 = _make_png_b64()
    records = [
        {"type": "user", "message": {"role": "user", "content": "first"}},
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second"},
                    {"type": "image", "source": {"type": "base64", "data": b64}},
                ],
            },
        },
    ]
    msgs = records_to_messages(records)
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "first\nsecond"}
    assert content[1]["type"] == "image"


def test_records_to_messages_empty():
    assert records_to_messages([]) == []


# ── session_to_messages ─────────────────────────────────────────────


def test_session_to_messages_basic(tmp_path):
    p = tmp_path / "s.jsonl"
    _write_jsonl(
        p,
        [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "hey"}},
        ],
    )
    msgs = session_to_messages(p)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]


def test_session_to_messages_writes_output(tmp_path):
    p = tmp_path / "s.jsonl"
    _write_jsonl(
        p,
        [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "hey"}},
        ],
    )
    out = tmp_path / "sub" / "out.json"
    msgs = session_to_messages(p, output=out)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == msgs


def test_session_to_messages_output_image_json_safe(tmp_path):
    pytest.importorskip("PIL")
    b64 = _make_png_b64()
    p = tmp_path / "s.jsonl"
    _write_jsonl(
        p,
        [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {"type": "image", "source": {"type": "base64", "data": b64}},
                    ],
                },
            }
        ],
    )
    out = tmp_path / "out.json"
    msgs = session_to_messages(p, output=out)
    # In-memory keeps PIL object
    assert msgs[0]["content"][1]["type"] == "image"
    assert not isinstance(msgs[0]["content"][1]["image"], str)
    # On disk becomes base64 data URI
    loaded = json.loads(out.read_text(encoding="utf-8"))
    ondisk_img = loaded[0]["content"][1]["image"]
    assert isinstance(ondisk_img, str)
    assert ondisk_img.startswith("data:image/")


def test_session_to_messages_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        session_to_messages(tmp_path / "nope.jsonl")


# ── inspect_session ─────────────────────────────────────────────────


def test_inspect_session_runs(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    _write_jsonl(
        p,
        [
            {
                "type": "user",
                "sessionId": "sess-1",
                "cwd": "/tmp/x",
                "gitBranch": "main",
                "message": {"role": "user", "content": "hi"},
                "timestamp": "2024-01-02T03:04:05.000Z",
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": "hey",
                    "usage": {"input_tokens": 3, "output_tokens": 4},
                },
            },
        ],
    )
    inspect_session(p, show_flow=True, show_blocks=True, show_raw=True)
    out = capsys.readouterr().out
    assert "CLAUDE CODE" in out
    assert "sess-1" in out
    assert "main" in out
    assert "RAW RECORD EXAMPLES" in out


def test_print_session_metadata_output(capsys):
    records = [
        {
            "type": "user",
            "sessionId": "abc",
            "cwd": "/home",
            "gitBranch": "dev",
        }
    ]
    _print_session_metadata(records)
    out = capsys.readouterr().out
    assert "abc" in out
    assert "/home" in out
    assert "dev" in out


def test_print_session_metadata_no_user(capsys):
    _print_session_metadata([{"type": "assistant"}])
    assert capsys.readouterr().out == ""


def test_print_raw_examples_output(capsys):
    records = [
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        },
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "dup"}]},
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "string-content"},
        },
    ]
    _print_raw_examples(records)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    # user printed once despite duplicate
    assert out.count("type = user") == 1
    assert "type = assistant" in out
    assert "content (string)" in out
