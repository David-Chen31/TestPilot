from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_converter import claude
from claude_converter.claude import (
    _block_to_parts,
    _block_to_str,
    _collect_block_types,
    _content_to_parts,
    _content_to_str,
    _decode_claude_image,
    _render_block,
    _total_tokens_of,
    CLAUDE_SCHEMA,
    inspect_session,
    load_session,
    records_to_messages,
    session_to_messages,
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class FakeImage:
    """Stand-in for a PIL.Image.Image produced by the decoder."""

    def __init__(self, name="fake"):
        self.name = name

    def __repr__(self):
        return f"FakeImage({self.name!r})"


def _write_jsonl(path: Path, records: list) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _user(text_or_blocks, **extra):
    rec = {
        "type": "user",
        "message": {"role": "user", "content": text_or_blocks},
    }
    rec.update(extra)
    return rec


def _assistant(content, **extra):
    rec = {
        "type": "assistant",
        "message": {"role": "assistant", "content": content},
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# _block_to_str
# ---------------------------------------------------------------------------

def test_block_to_str_text():
    assert _block_to_str({"type": "text", "text": "hello"}) == "hello"


def test_block_to_str_text_missing_key():
    assert _block_to_str({"type": "text"}) == ""


def test_block_to_str_thinking():
    assert _block_to_str({"type": "thinking", "thinking": "hmm"}) == "<thinking>hmm</thinking>"


def test_block_to_str_tool_use():
    block = {"type": "tool_use", "name": "Read", "input": {"path": "a.txt"}}
    out = _block_to_str(block)
    assert out == "<tool_use name='Read'>{\"path\": \"a.txt\"}</tool_use>"


def test_block_to_str_tool_use_defaults():
    out = _block_to_str({"type": "tool_use"})
    assert out == "<tool_use name=''>{}</tool_use>"


def test_block_to_str_tool_result_string():
    out = _block_to_str({"type": "tool_result", "content": "raw output"})
    assert out == "<tool_result>raw output</tool_result>"


def test_block_to_str_tool_result_list_text_and_image():
    block = {
        "type": "tool_result",
        "content": [
            {"type": "text", "text": "part1"},
            {"type": "image", "source": {}},
            {"type": "text", "text": "part2"},
        ],
    }
    out = _block_to_str(block)
    assert out == "<tool_result>part1 <image> part2</tool_result>"


def test_block_to_str_tool_result_list_ignores_non_dicts():
    block = {
        "type": "tool_result",
        "content": ["not a dict", {"type": "text", "text": "ok"}],
    }
    assert _block_to_str(block) == "<tool_result>ok</tool_result>"


def test_block_to_str_image():
    assert _block_to_str({"type": "image"}) == "<image>"


def test_block_to_str_unknown():
    assert _block_to_str({"type": "mystery"}) == ""


def test_block_to_str_no_type():
    assert _block_to_str({}) == ""


# ---------------------------------------------------------------------------
# _decode_claude_image
# ---------------------------------------------------------------------------

def test_decode_image_unsupported_source_type(monkeypatch, capsys):
    block = {"type": "image", "source": {"type": "url", "data": "x"}}
    assert _decode_claude_image(block) is None
    err = capsys.readouterr().err
    assert "unsupported source type" in err
    assert "'url'" in err


def test_decode_image_missing_source(capsys):
    assert _decode_claude_image({"type": "image"}) is None
    assert "unsupported source type" in capsys.readouterr().err


def test_decode_image_empty_data(capsys):
    block = {"type": "image", "source": {"type": "base64", "data": ""}}
    assert _decode_claude_image(block) is None
    assert "empty base64 data" in capsys.readouterr().err


def test_decode_image_missing_data(capsys):
    block = {"type": "image", "source": {"type": "base64"}}
    assert _decode_claude_image(block) is None
    assert "empty base64 data" in capsys.readouterr().err


def test_decode_image_success(monkeypatch):
    fake = FakeImage()
    monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda data: fake)
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert _decode_claude_image(block) is fake


def test_decode_image_value_error(monkeypatch, capsys):
    def boom(_data):
        raise ValueError("bad base64")

    monkeypatch.setattr(claude, "convert_base64_to_pil_image", boom)
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert _decode_claude_image(block) is None
    assert "unreadable image block" in capsys.readouterr().err


def test_decode_image_import_error(monkeypatch, capsys):
    def boom(_data):
        raise ImportError("no pillow")

    monkeypatch.setattr(claude, "convert_base64_to_pil_image", boom)
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert _decode_claude_image(block) is None
    assert "unreadable image block" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _block_to_parts
# ---------------------------------------------------------------------------

def test_block_to_parts_text():
    assert _block_to_parts({"type": "text", "text": "hi"}) == [
        {"type": "text", "text": "hi"}
    ]


def test_block_to_parts_blank_text():
    assert _block_to_parts({"type": "text", "text": "   "}) == []


def test_block_to_parts_image_success(monkeypatch):
    fake = FakeImage()
    monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda data: fake)
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert _block_to_parts(block) == [{"type": "image", "image": fake}]


def test_block_to_parts_image_failure(capsys):
    block = {"type": "image", "source": {"type": "base64", "data": ""}}
    assert _block_to_parts(block) == []
    assert "empty base64 data" in capsys.readouterr().err


def test_block_to_parts_tool_result_no_image():
    block = {"type": "tool_result", "content": "plain", "tool_use_id": "t1"}
    assert _block_to_parts(block) == [
        {"type": "text", "text": "<tool_result>plain</tool_result>"}
    ]


def test_block_to_parts_tool_result_blank():
    block = {"type": "tool_result", "content": ""}
    assert _block_to_parts(block) == []


def test_block_to_parts_tool_result_with_image(monkeypatch):
    fake = FakeImage()
    monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda data: fake)
    block = {
        "type": "tool_result",
        "tool_use_id": "t42",
        "content": [
            {"type": "text", "text": "before"},
            {"type": "image", "source": {"type": "base64", "data": "abc"}},
            {"type": "text", "text": "   "},
            {"type": "text", "text": "after"},
        ],
    }
    parts = _block_to_parts(block)
    assert parts == [
        {"type": "text", "text": "<tool_result id='t42'>"},
        {"type": "text", "text": "before"},
        {"type": "image", "image": fake},
        {"type": "text", "text": "after"},
        {"type": "text", "text": "</tool_result>"},
    ]


def test_block_to_parts_tool_result_with_image_default_id(monkeypatch):
    fake = FakeImage()
    monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda data: fake)
    block = {
        "type": "tool_result",
        "content": [
            {"type": "image", "source": {"type": "base64", "data": "abc"}},
        ],
    }
    parts = _block_to_parts(block)
    assert parts[0] == {"type": "text", "text": "<tool_result id='?'>"}
    assert parts[-1] == {"type": "text", "text": "</tool_result>"}


def test_block_to_parts_tool_result_with_failed_image(monkeypatch, capsys):
    def boom(_data):
        raise ValueError("bad")

    monkeypatch.setattr(claude, "convert_base64_to_pil_image", boom)
    block = {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": [
            {"type": "image", "source": {"type": "base64", "data": "abc"}},
            {"type": "text", "text": "txt"},
        ],
    }
    parts = _block_to_parts(block)
    assert parts == [
        {"type": "text", "text": "<tool_result id='t1'>"},
        {"type": "text", "text": "txt"},
        {"type": "text", "text": "</tool_result>"},
    ]
    assert "unreadable image block" in capsys.readouterr().err


def test_block_to_parts_tool_result_with_image_skips_non_dict(monkeypatch):
    fake = FakeImage()
    monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda data: fake)
    block = {
        "type": "tool_result",
        "content": [
            {"type": "image", "source": {"type": "base64", "data": "abc"}},
            "junk",
        ],
    }
    parts = _block_to_parts(block)
    assert parts == [
        {"type": "text", "text": "<tool_result id='?'>"},
        {"type": "image", "image": fake},
        {"type": "text", "text": "</tool_result>"},
    ]


# ---------------------------------------------------------------------------
# _content_to_str / _content_to_parts / _collect_block_types
# ---------------------------------------------------------------------------

def test_content_to_str_string():
    assert _content_to_str("hello") == "hello"


def test_content_to_str_none():
    assert _content_to_str(None) == ""


def test_content_to_str_list():
    content = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": ""},
        {"type": "thinking", "thinking": "b"},
        "not a dict",
    ]
    assert _content_to_str(content) == "a\n<thinking>b</thinking>"


def test_content_to_parts_string():
    assert _content_to_parts("hi") == [{"type": "text", "text": "hi"}]


def test_content_to_parts_blank_string():
    assert _content_to_parts("   ") == []


def test_content_to_parts_none():
    assert _content_to_parts(None) == []


def test_content_to_parts_list():
    content = [
        {"type": "text", "text": "a"},
        "skip",
        {"type": "text", "text": "b"},
    ]
    assert _content_to_parts(content) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_collect_block_types():
    assert _collect_block_types("raw") == ["raw_string"]
    assert _collect_block_types(None) == ["unknown"]
    assert _collect_block_types([{"type": "text"}, {}, "x"]) == ["text", "unknown"]


# ---------------------------------------------------------------------------
# _render_block
# ---------------------------------------------------------------------------

def test_render_block_text():
    out = _render_block({"type": "text", "text": "hello"}, indent="")
    assert "text" in out
    assert "hello" in out


def test_render_block_text_truncation():
    long = "x" * 300
    out = _render_block({"type": "text", "text": long}, indent="")
    assert "..." in out
    assert long not in out


def test_render_block_thinking():
    out = _render_block({"type": "thinking", "thinking": "pondering"}, indent="")
    assert "thinking" in out
    assert "pondering" in out


def test_render_block_tool_use():
    block = {"type": "tool_use", "name": "Read", "id": "id1", "input": {"a": 1}}
    out = _render_block(block, indent="")
    assert "Read" in out
    assert "id1" in out
    assert "input" in out


def test_render_block_tool_result_list():
    block = {
        "type": "tool_result",
        "tool_use_id": "t9",
        "content": [
            {"type": "image", "source": {"media_type": "image/png", "data": "abcd"}},
            {"type": "text", "text": "text here"},
            "junk",
        ],
    }
    out = _render_block(block, indent="")
    assert "t9" in out
    assert "image/png" in out
    assert "4 base64 chars" in out
    assert "text here" in out


def test_render_block_tool_result_string():
    block = {"type": "tool_result", "content": "plain output", "tool_use_id": "t1"}
    out = _render_block(block, indent="")
    assert "plain output" in out


def test_render_block_image():
    block = {
        "type": "image",
        "source": {"media_type": "image/jpeg", "data": "abcdef"},
    }
    out = _render_block(block, indent="")
    assert "image/jpeg" in out
    assert "6 base64 chars" in out


def test_render_block_image_missing_source():
    out = _render_block({"type": "image"}, indent="")
    assert "?" in out
    assert "0 base64 chars" in out


def test_render_block_unknown():
    out = _render_block({"type": "weird", "foo": "bar"}, indent="")
    assert "weird" in out
    assert "bar" in out


# ---------------------------------------------------------------------------
# load_session
# ---------------------------------------------------------------------------

def test_load_session_roundtrip(tmp_path):
    records = [{"type": "user"}, {"type": "assistant"}]
    path = _write_jsonl(tmp_path / "s.jsonl", records)
    assert load_session(path) == records


def test_load_session_accepts_str_path(tmp_path):
    path = _write_jsonl(tmp_path / "s.jsonl", [{"type": "user"}])
    assert load_session(str(path)) == [{"type": "user"}]


def test_load_session_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "missing.jsonl")


def test_load_session_bad_extension(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session(p)


def test_load_session_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session(p)


# ---------------------------------------------------------------------------
# records_to_messages
# ---------------------------------------------------------------------------

def test_records_to_messages_simple_text():
    records = [
        _user("hello"),
        _assistant("hi there"),
    ]
    assert records_to_messages(records) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_records_to_messages_skips_system():
    records = [
        {"type": "system", "message": {"role": "system", "content": "sys"}},
        _user("hello"),
    ]
    assert records_to_messages(records) == [{"role": "user", "content": "hello"}]


def test_records_to_messages_merges_consecutive_same_role():
    records = [
        _user("first"),
        _user("second"),
    ]
    assert records_to_messages(records) == [
        {"role": "user", "content": "first\nsecond"}
    ]


def test_records_to_messages_skips_empty_content():
    records = [
        _user(""),
        _assistant("real"),
    ]
    assert records_to_messages(records) == [
        {"role": "assistant", "content": "real"}
    ]


def test_records_to_messages_role_from_message():
    records = [
        {
            "type": "user",
            "message": {"role": "assistant", "content": "weird"},
        }
    ]
    assert records_to_messages(records) == [
        {"role": "assistant", "content": "weird"}
    ]


def test_records_to_messages_role_fallback_to_type():
    records = [{"type": "user", "message": {"content": "hi"}}]
    assert records_to_messages(records) == [{"role": "user", "content": "hi"}]


def test_records_to_messages_flattens_blocks():
    records = [
        _user([
            {"type": "text", "text": "line1"},
            {"type": "tool_use", "name": "Read", "input": {}},
        ]),
    ]
    out = records_to_messages(records)
    assert out[0]["role"] == "user"
    assert "<tool_use name='Read'>" in out[0]["content"]
    assert "line1" in out[0]["content"]


def test_records_to_messages_missing_message():
    records = [{"type": "user"}]
    assert records_to_messages(records) == []


def test_records_to_messages_empty_list():
    assert records_to_messages([]) == []


def test_records_to_messages_tool_result_text():
    records = [
        _user([
            {"type": "tool_result", "content": "done", "tool_use_id": "t1"},
        ]),
    ]
    out = records_to_messages(records)
    assert out == [
        {"role": "user", "content": "<tool_result>done</tool_result>"}
    ]


def test_records_to_messages_text_only_merge_no_leading_newline():
    # merge_content_parts joins with a newline; finalize_content joins parts
    # with newlines as well. Two text pieces on the same role join.
    records = [
        _user("a"),
        _user("b"),
        _assistant("c"),
    ]
    assert records_to_messages(records) == [
        {"role": "user", "content": "a\nb"},
        {"role": "assistant", "content": "c"},
    ]


# ---------------------------------------------------------------------------
# session_to_messages
# ---------------------------------------------------------------------------

def test_session_to_messages_basic(tmp_path):
    records = [_user("hello"), _assistant("hi")]
    path = _write_jsonl(tmp_path / "s.jsonl", records)
    out = session_to_messages(path)
    assert out == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_session_to_messages_writes_output(tmp_path):
    records = [_user("hello"), _assistant("hi")]
    path = _write_jsonl(tmp_path / "s.jsonl", records)
    out_path = tmp_path / "nested" / "messages.json"

    result = session_to_messages(path, output=out_path)

    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert result == written


def test_session_to_messages_output_as_str(tmp_path):
    path = _write_jsonl(tmp_path / "s.jsonl", [_user("x")])
    out_path = tmp_path / "out.json"
    session_to_messages(path, output=str(out_path))
    assert json.loads(out_path.read_text(encoding="utf-8")) == [
        {"role": "user", "content": "x"}
    ]


def test_session_to_messages_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        session_to_messages(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Inspector schema helpers
# ---------------------------------------------------------------------------

def test_record_type_of():
    assert claude._record_type_of({"type": "user"}) == "user"
    assert claude._record_type_of({}) == "unknown"


def test_is_message():
    assert claude._is_message({"type": "user"}) is True
    assert claude._is_message({"type": "assistant"}) is True
    assert claude._is_message({"type": "system"}) is False


def test_role_of():
    assert claude._role_of(_user("x")) == "user"
    assert claude._role_of({"type": "system", "message": {}}) == "system"


def test_text_of():
    assert claude._text_of(_user("hello")) == "hello"


def test_timestamp_of():
    rec = {"timestamp": "2024-01-02T03:04:05Z"}
    assert claude._timestamp_of(rec) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert claude._timestamp_of({}) == ""


def test_tokens_of():
    rec = {"message": {"usage": {"output_tokens": 42}}}
    assert claude._tokens_of(rec) == 42


def test_tokens_of_missing():
    assert claude._tokens_of({}) == 0


def test_block_types_of():
    rec = _user([{"type": "text"}, {"type": "image"}])
    assert claude._block_types_of(rec) == ["text", "image"]


def test_block_types_of_string_content():
    rec = _user("plain")
    assert claude._block_types_of(rec) == ["raw_string"]


def test_blocks_of():
    rec = _user([{"type": "text"}, "junk"])
    assert claude._blocks_of(rec) == [{"type": "text"}]


def test_blocks_of_string_content():
    rec = _user("x")
    assert claude._blocks_of(rec) == []


def test_total_tokens_of():
    records = [
        {"message": {"usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        }}},
        {"message": {"usage": {"input_tokens": 1, "output_tokens": 1}}},
    ]
    totals = _total_tokens_of(records)
    assert totals == {
        "input": 11,
        "output": 6,
        "cache_read": 3,
        "cache_write": 2,
        "total": 17,
    }


def test_total_tokens_of_empty():
    assert _total_tokens_of([
        {"message": {"usage": {"cache_read_input_tokens": 5}}},
    ]) == {}


def test_claude_schema_configured():
    assert CLAUDE_SCHEMA.label == "CLAUDE CODE"
    assert CLAUDE_SCHEMA.record_type_of is claude._record_type_of
    assert CLAUDE_SCHEMA.is_message is claude._is_message
    assert CLAUDE_SCHEMA.role_of is claude._role_of
    assert CLAUDE_SCHEMA.text_of is claude._text_of
    assert CLAUDE_SCHEMA.timestamp_of is claude._timestamp_of
    assert CLAUDE_SCHEMA.total_tokens_of is claude._total_tokens_of
    assert CLAUDE_SCHEMA.tokens_of is claude._tokens_of
    assert CLAUDE_SCHEMA.block_types_of is claude._block_types_of
    assert CLAUDE_SCHEMA.blocks_of is claude._blocks_of
    assert CLAUDE_SCHEMA.render_block is claude._render_block


# ---------------------------------------------------------------------------
# _print_session_metadata / _print_raw_examples
# ---------------------------------------------------------------------------

def test_print_session_metadata(capsys):
    records = [{
        "type": "user",
        "sessionId": "sess-1",
        "cwd": "/tmp/x",
        "gitBranch": "main",
    }]
    claude._print_session_metadata(records)
    out = capsys.readouterr().out
    assert "sess-1" in out
    assert "/tmp/x" in out
    assert "main" in out


def test_print_session_metadata_no_user(capsys):
    claude._print_session_metadata([{"type": "system"}])
    assert capsys.readouterr().out == ""


def test_print_session_metadata_no_branch(capsys):
    records = [{"type": "user", "sessionId": "s", "cwd": "c"}]
    claude._print_session_metadata(records)
    out = capsys.readouterr().out
    assert "s" in out
    assert "Branch" not in out


def test_print_raw_examples_list_content(capsys):
    records = [
        _user([{"type": "text", "text": "hi"}]),
        _user("second user same type"),
        {"type": "system"},
    ]
    claude._print_raw_examples(records)
    out = capsys.readouterr().out
    assert "type = user" in out
    assert "type = system" in out
    assert "content[0]" in out


def test_print_raw_examples_string_content(capsys):
    records = [{"type": "assistant", "message": {"content": "plain string"}}]
    claude._print_raw_examples(records)
    out = capsys.readouterr().out
    assert "content (string)" in out
    assert "plain string" in out


# ---------------------------------------------------------------------------
# inspect_session
# ---------------------------------------------------------------------------

def test_inspect_session_runs(tmp_path, capsys):
    records = [
        _user("hello", sessionId="s1", cwd="/c"),
        _assistant([{"type": "text", "text": "hi"}]),
    ]
    path = _write_jsonl(tmp_path / "s.jsonl", records)

    inspect_session(path)

    out = capsys.readouterr().out
    assert "CLAUDE CODE" in out
    assert "s1" in out


def test_inspect_session_with_flow_and_blocks_and_raw(tmp_path, capsys):
    records = [
        _user("hello"),
        _assistant("hi"),
    ]
    path = _write_jsonl(tmp_path / "s.jsonl", records)

    inspect_session(path, show_flow=True, show_blocks=True, show_raw=True)

    out = capsys.readouterr().out
    assert "CLAUDE CODE" in out
    assert "RAW RECORD EXAMPLES" in out


def test_inspect_session_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_session(tmp_path / "missing.jsonl")
