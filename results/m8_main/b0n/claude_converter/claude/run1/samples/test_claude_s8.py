import json
import sys

import pytest

from claude_converter import claude
from claude_converter.claude import (
    load_session,
    records_to_messages,
    session_to_messages,
    inspect_session,
)


# ── helpers ────────────────────────────────────────────────────────────────


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return str(path)


class FakeImage:
    """Stand-in for a PIL.Image so tests don't require Pillow."""


# ── _block_to_str ──────────────────────────────────────────────────────────


def test_block_to_str_text():
    assert claude._block_to_str({"type": "text", "text": "hello"}) == "hello"


def test_block_to_str_thinking():
    assert (
        claude._block_to_str({"type": "thinking", "thinking": "deep"})
        == "<thinking>deep</thinking>"
    )


def test_block_to_str_tool_use():
    block = {"type": "tool_use", "name": "Read", "input": {"path": "a.txt"}}
    result = claude._block_to_str(block)
    assert result == "<tool_use name='Read'>{\"path\": \"a.txt\"}</tool_use>"


def test_block_to_str_tool_result_string_content():
    block = {"type": "tool_result", "content": "ok"}
    assert claude._block_to_str(block) == "<tool_result>ok</tool_result>"


def test_block_to_str_tool_result_list_content():
    block = {
        "type": "tool_result",
        "content": [
            {"type": "text", "text": "line1"},
            {"type": "image"},
            {"type": "text", "text": "line2"},
            "not-a-dict",
        ],
    }
    assert (
        claude._block_to_str(block)
        == "<tool_result>line1 <image> line2</tool_result>"
    )


def test_block_to_str_image():
    assert claude._block_to_str({"type": "image"}) == "<image>"


def test_block_to_str_unknown():
    assert claude._block_to_str({"type": "mystery"}) == ""


def test_block_to_str_missing_type():
    assert claude._block_to_str({}) == ""


# ── _decode_claude_image ───────────────────────────────────────────────────


def test_decode_claude_image_unsupported_source_type(capsys):
    block = {"type": "image", "source": {"type": "url", "data": "x"}}
    assert claude._decode_claude_image(block) is None
    err = capsys.readouterr().err
    assert "unsupported source type" in err


def test_decode_claude_image_empty_data(capsys):
    block = {"type": "image", "source": {"type": "base64", "data": ""}}
    assert claude._decode_claude_image(block) is None
    err = capsys.readouterr().err
    assert "empty base64 data" in err


def test_decode_claude_image_missing_source(capsys):
    block = {"type": "image"}
    assert claude._decode_claude_image(block) is None
    err = capsys.readouterr().err
    assert "unsupported source type" in err


def test_decode_claude_image_import_error(monkeypatch, capsys):
    def boom(_data):
        raise ImportError("no pillow")

    monkeypatch.setattr(claude, "convert_base64_to_pil_image", boom)
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert claude._decode_claude_image(block) is None
    assert "Skipping unreadable image block" in capsys.readouterr().err


def test_decode_claude_image_value_error(monkeypatch, capsys):
    def boom(_data):
        raise ValueError("bad base64")

    monkeypatch.setattr(claude, "convert_base64_to_pil_image", boom)
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert claude._decode_claude_image(block) is None
    assert "bad base64" in capsys.readouterr().err


def test_decode_claude_image_success(monkeypatch):
    sentinel = FakeImage()
    monkeypatch.setattr(
        claude, "convert_base64_to_pil_image", lambda _d: sentinel
    )
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert claude._decode_claude_image(block) is sentinel


# ── _block_to_parts ────────────────────────────────────────────────────────


def test_block_to_parts_image_ok(monkeypatch):
    sentinel = FakeImage()
    monkeypatch.setattr(claude, "_decode_claude_image", lambda b: sentinel)
    parts = claude._block_to_parts({"type": "image"})
    assert parts == [{"type": "image", "image": sentinel}]


def test_block_to_parts_image_failed(monkeypatch):
    monkeypatch.setattr(claude, "_decode_claude_image", lambda b: None)
    assert claude._block_to_parts({"type": "image"}) == []


def test_block_to_parts_text():
    parts = claude._block_to_parts({"type": "text", "text": "hi"})
    assert parts == [{"type": "text", "text": "hi"}]


def test_block_to_parts_blank_text():
    assert claude._block_to_parts({"type": "text", "text": "   "}) == []


def test_block_to_parts_tool_result_text_only():
    block = {"type": "tool_result", "content": "plain"}
    parts = claude._block_to_parts(block)
    assert parts == [{"type": "text", "text": "<tool_result>plain</tool_result>"}]


def test_block_to_parts_tool_result_empty():
    block = {"type": "tool_result", "content": ""}
    assert claude._block_to_parts(block) == []


def test_block_to_parts_tool_result_with_images(monkeypatch):
    sentinel = FakeImage()

    def fake_decode(b):
        if b is nested_img:
            return sentinel
        return None

    nested_img = {"type": "image", "source": {"type": "base64", "data": "x"}}
    bad_img = {"type": "image", "source": {"type": "base64", "data": ""}}
    monkeypatch.setattr(claude, "_decode_claude_image", fake_decode)

    block = {
        "type": "tool_result",
        "tool_use_id": "tid-1",
        "content": [
            {"type": "text", "text": "before"},
            nested_img,
            bad_img,
            {"type": "text", "text": "  "},
            {"type": "text", "text": "after"},
            "not-a-dict",
        ],
    }
    parts = claude._block_to_parts(block)
    assert parts == [
        {"type": "text", "text": "<tool_result id='tid-1'>"},
        {"type": "text", "text": "before"},
        {"type": "image", "image": sentinel},
        {"type": "text", "text": "after"},
        {"type": "text", "text": "</tool_result>"},
    ]


def test_block_to_parts_tool_result_image_missing_id(monkeypatch):
    monkeypatch.setattr(claude, "_decode_claude_image", lambda b: FakeImage())
    block = {
        "type": "tool_result",
        "content": [{"type": "image"}],
    }
    parts = claude._block_to_parts(block)
    assert parts[0] == {"type": "text", "text": "<tool_result id='?'>"}
    assert parts[-1] == {"type": "text", "text": "</tool_result>"}


# ── _content_to_str / _content_to_parts ────────────────────────────────────


def test_content_to_str_string():
    assert claude._content_to_str("direct") == "direct"


def test_content_to_str_list():
    content = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": ""},
        {"type": "text", "text": "b"},
        "skipped",
    ]
    assert claude._content_to_str(content) == "a\nb"


def test_content_to_str_none():
    assert claude._content_to_str(None) == ""


def test_content_to_parts_string():
    assert claude._content_to_parts("hi") == [{"type": "text", "text": "hi"}]


def test_content_to_parts_empty_string():
    assert claude._content_to_parts("   ") == []


def test_content_to_parts_list():
    content = [
        {"type": "text", "text": "a"},
        "skipped",
        {"type": "text", "text": "b"},
    ]
    assert claude._content_to_parts(content) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_content_to_parts_none():
    assert claude._content_to_parts(None) == []


# ── _collect_block_types ───────────────────────────────────────────────────


def test_collect_block_types_string():
    assert claude._collect_block_types("s") == ["raw_string"]


def test_collect_block_types_list():
    content = [{"type": "text"}, {}, "skip", {"type": "image"}]
    assert claude._collect_block_types(content) == ["text", "unknown", "image"]


def test_collect_block_types_other():
    assert claude._collect_block_types(None) == ["unknown"]


# ── _render_block ──────────────────────────────────────────────────────────


def test_render_block_text():
    out = claude._render_block({"type": "text", "text": "hello"})
    assert "hello" in out
    assert "text" in out


def test_render_block_thinking():
    out = claude._render_block({"type": "thinking", "thinking": "ponder"})
    assert "ponder" in out
    assert "thinking" in out


def test_render_block_tool_use():
    block = {"type": "tool_use", "name": "Read", "id": "abc", "input": {"a": 1}}
    out = claude._render_block(block)
    assert "Read" in out
    assert "abc" in out
    assert '{"a": 1}' in out


def test_render_block_tool_result_list():
    block = {
        "type": "tool_result",
        "tool_use_id": "tid",
        "content": [
            {"type": "image", "source": {"media_type": "image/png", "data": "abcd"}},
            {"type": "text", "text": "nested text"},
        ],
    }
    out = claude._render_block(block)
    assert "tid" in out
    assert "image/png" in out
    assert "4 base64 chars" in out
    assert "nested text" in out


def test_render_block_tool_result_string():
    block = {"type": "tool_result", "tool_use_id": "tid", "content": "plain"}
    out = claude._render_block(block)
    assert "plain" in out


def test_render_block_image():
    block = {
        "type": "image",
        "source": {"media_type": "image/jpeg", "data": "abc"},
    }
    out = claude._render_block(block)
    assert "image/jpeg" in out
    assert "3 base64 chars" in out


def test_render_block_unknown():
    out = claude._render_block({"type": "weird", "x": 1})
    assert "weird" in out


# ── load_session ───────────────────────────────────────────────────────────


def test_load_session_roundtrip(tmp_path):
    p = tmp_path / "session.jsonl"
    write_jsonl(p, [{"type": "user"}, {"type": "assistant"}])
    assert load_session(str(p)) == [{"type": "user"}, {"type": "assistant"}]


def test_load_session_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session(str(tmp_path / "nope.jsonl"))


def test_load_session_bad_extension(tmp_path):
    p = tmp_path / "session.txt"
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session(str(p))


def test_load_session_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session(str(p))


# ── records_to_messages ────────────────────────────────────────────────────


def test_records_to_messages_basic():
    records = [
        {"type": "user", "message": {"role": "user", "content": "Hi"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "Hello"}},
    ]
    assert records_to_messages(records) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_records_to_messages_skips_non_message_types():
    records = [
        {"type": "system", "message": {"role": "system", "content": "sys"}},
        {"type": "user", "message": {"role": "user", "content": "Hi"}},
        {"type": "summary", "message": {"role": "user", "content": "no"}},
    ]
    assert records_to_messages(records) == [{"role": "user", "content": "Hi"}]


def test_records_to_messages_merges_same_role():
    records = [
        {"type": "user", "message": {"role": "user", "content": "one"}},
        {"type": "user", "message": {"role": "user", "content": "two"}},
    ]
    assert records_to_messages(records) == [
        {"role": "user", "content": "one\ntwo"}
    ]


def test_records_to_messages_skips_empty_parts():
    records = [
        {"type": "user", "message": {"role": "user", "content": ""}},
        {"type": "user", "message": {"role": "user", "content": "kept"}},
    ]
    assert records_to_messages(records) == [{"role": "user", "content": "kept"}]


def test_records_to_messages_uses_record_type_as_fallback_role():
    records = [{"type": "user", "message": {"content": "hi"}}]
    assert records_to_messages(records) == [{"role": "user", "content": "hi"}]


def test_records_to_messages_with_image_part(monkeypatch):
    sentinel = FakeImage()
    monkeypatch.setattr(claude, "_decode_claude_image", lambda b: sentinel)
    records = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "data": "x"}},
                ],
            },
        }
    ]
    result = records_to_messages(records)
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "image": sentinel},
            ],
        }
    ]


def test_records_to_messages_flattens_blocks_to_text():
    records = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "answer"},
                ],
            },
        }
    ]
    assert records_to_messages(records) == [
        {
            "role": "assistant",
            "content": "<thinking>hmm</thinking>\nanswer",
        }
    ]


def test_records_to_messages_empty_list():
    assert records_to_messages([]) == []


# ── session_to_messages ────────────────────────────────────────────────────


def test_session_to_messages_basic(tmp_path):
    p = tmp_path / "session.jsonl"
    write_jsonl(
        p,
        [
            {"type": "user", "message": {"role": "user", "content": "Hi"}},
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": "Hello"},
            },
        ],
    )
    result = session_to_messages(str(p))
    assert result == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_session_to_messages_writes_output(tmp_path):
    p = tmp_path / "session.jsonl"
    write_jsonl(
        p, [{"type": "user", "message": {"role": "user", "content": "Hi"}}]
    )
    out = tmp_path / "nested" / "messages.json"
    result = session_to_messages(str(p), output=str(out))

    assert result == [{"role": "user", "content": "Hi"}]
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == [{"role": "user", "content": "Hi"}]


def test_session_to_messages_no_output_does_not_write(tmp_path):
    p = tmp_path / "session.jsonl"
    write_jsonl(
        p, [{"type": "user", "message": {"role": "user", "content": "Hi"}}]
    )
    session_to_messages(str(p))
    assert list(tmp_path.glob("**/*.json")) == []


def test_session_to_messages_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        session_to_messages(str(tmp_path / "missing.jsonl"))


# ── schema helper functions ────────────────────────────────────────────────


def test_record_type_of():
    assert claude._record_type_of({"type": "user"}) == "user"
    assert claude._record_type_of({}) == "unknown"


def test_is_message():
    assert claude._is_message({"type": "user"}) is True
    assert claude._is_message({"type": "assistant"}) is True
    assert claude._is_message({"type": "system"}) is False


def test_role_of():
    assert (
        claude._role_of({"type": "user", "message": {"role": "assistant"}})
        == "assistant"
    )
    assert claude._role_of({"type": "user", "message": {}}) == "user"


def test_text_of():
    record = {"type": "user", "message": {"content": "plain text"}}
    assert claude._text_of(record) == "plain text"


def test_timestamp_of():
    record = {"timestamp": "2024-01-02T03:04:05.678Z"}
    assert claude._timestamp_of(record) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert claude._timestamp_of({}) == ""


def test_tokens_of():
    record = {"message": {"usage": {"output_tokens": 42}}}
    assert claude._tokens_of(record) == 42


def test_tokens_of_missing():
    assert claude._tokens_of({}) == 0


def test_block_types_of():
    record = {"message": {"content": [{"type": "text"}, {"type": "image"}]}}
    assert claude._block_types_of(record) == ["text", "image"]


def test_blocks_of_list():
    record = {"message": {"content": [{"type": "text"}, "skip"]}}
    assert claude._blocks_of(record) == [{"type": "text"}]


def test_blocks_of_string():
    record = {"message": {"content": "string"}}
    assert claude._blocks_of(record) == []


def test_total_tokens_of():
    records = [
        {
            "message": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 2,
                    "cache_creation_input_tokens": 1,
                }
            }
        },
        {"message": {"usage": {"input_tokens": 3, "output_tokens": 4}}},
    ]
    assert claude._total_tokens_of(records) == {
        "input": 13,
        "output": 9,
        "cache_read": 2,
        "cache_write": 1,
        "total": 22,
    }


def test_total_tokens_of_empty_when_all_zero():
    assert claude._total_tokens_of([{"message": {}}]) == {}


# ── CLAUDE_SCHEMA ──────────────────────────────────────────────────────────


def test_claude_schema_label():
    assert claude.CLAUDE_SCHEMA.label == "CLAUDE CODE"


# ── _print_session_metadata ────────────────────────────────────────────────


def test_print_session_metadata(capsys):
    records = [
        {
            "type": "user",
            "sessionId": "sid-123",
            "cwd": "/work",
            "gitBranch": "main",
        }
    ]
    claude._print_session_metadata(records)
    out = capsys.readouterr().out
    assert "sid-123" in out
    assert "/work" in out
    assert "main" in out


def test_print_session_metadata_no_user(capsys):
    claude._print_session_metadata([{"type": "assistant"}])
    assert capsys.readouterr().out == ""


def test_print_session_metadata_no_branch(capsys):
    records = [{"type": "user", "sessionId": "s", "cwd": "c"}]
    claude._print_session_metadata(records)
    out = capsys.readouterr().out
    assert "Branch" not in out


# ── _print_raw_examples ────────────────────────────────────────────────────


def test_print_raw_examples(capsys):
    records = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            },
        },
        {"type": "user", "message": {"content": "string content"}},
        {"type": "assistant"},
    ]
    claude._print_raw_examples(records)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    assert "type = user" in out
    assert "type = assistant" in out
    assert "content (string)" in out


# ── inspect_session ────────────────────────────────────────────────────────


def test_inspect_session_calls_run_inspection(tmp_path, monkeypatch):
    p = tmp_path / "session.jsonl"
    write_jsonl(
        p, [{"type": "user", "message": {"role": "user", "content": "Hi"}}]
    )

    captured = {}

    def fake_run_inspection(path, records, schema, **kwargs):
        captured["path"] = path
        captured["records"] = records
        captured["schema"] = schema
        captured["kwargs"] = kwargs

    monkeypatch.setattr(claude, "run_inspection", fake_run_inspection)
    inspect_session(str(p), show_flow=True, show_blocks=True, show_raw=False)

    assert captured["schema"] is claude.CLAUDE_SCHEMA
    assert captured["records"] == [
        {"type": "user", "message": {"role": "user", "content": "Hi"}}
    ]
    assert captured["kwargs"]["show_flow"] is True
    assert captured["kwargs"]["show_blocks"] is True
    assert captured["kwargs"]["extra_header"] is claude._print_session_metadata


def test_inspect_session_show_raw(tmp_path, monkeypatch, capsys):
    p = tmp_path / "session.jsonl"
    write_jsonl(
        p, [{"type": "user", "message": {"role": "user", "content": "Hi"}}]
    )
    monkeypatch.setattr(claude, "run_inspection", lambda *a, **k: None)

    inspect_session(str(p), show_raw=True)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out


def test_inspect_session_no_raw(tmp_path, monkeypatch, capsys):
    p = tmp_path / "session.jsonl"
    write_jsonl(
        p, [{"type": "user", "message": {"role": "user", "content": "Hi"}}]
    )
    monkeypatch.setattr(claude, "run_inspection", lambda *a, **k: None)

    inspect_session(str(p))
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" not in out


def test_inspect_session_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_session(str(tmp_path / "nope.jsonl"))
