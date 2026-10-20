import json
import sys

import pytest

from claude_converter.claude import (
    CLAUDE_SCHEMA,
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
    inspect_session,
    load_session,
    records_to_messages,
    session_to_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# _block_to_str
# ---------------------------------------------------------------------------

def test_block_to_str_text():
    assert _block_to_str({"type": "text", "text": "hello"}) == "hello"


def test_block_to_str_text_missing():
    assert _block_to_str({"type": "text"}) == ""


def test_block_to_str_thinking():
    out = _block_to_str({"type": "thinking", "thinking": "hmm"})
    assert out == "<thinking>hmm</thinking>"


def test_block_to_str_tool_use():
    block = {"type": "tool_use", "name": "Read", "input": {"path": "a.txt"}}
    out = _block_to_str(block)
    assert out == "<tool_use name='Read'>" + json.dumps({"path": "a.txt"}, ensure_ascii=False) + "</tool_use>"


def test_block_to_str_tool_use_missing_input():
    out = _block_to_str({"type": "tool_use", "name": "Bash"})
    assert out == "<tool_use name='Bash'>{}</tool_use>"


def test_block_to_str_tool_result_string_content():
    out = _block_to_str({"type": "tool_result", "content": "result"})
    assert out == "<tool_result>result</tool_result>"


def test_block_to_str_tool_result_list_content():
    block = {
        "type": "tool_result",
        "content": [
            {"type": "text", "text": "a"},
            {"type": "image"},
            {"type": "text", "text": "b"},
            "notadict",
        ],
    }
    out = _block_to_str(block)
    assert out == "<tool_result>a <image> b</tool_result>"


def test_block_to_str_image():
    assert _block_to_str({"type": "image"}) == "<image>"


def test_block_to_str_unknown():
    assert _block_to_str({"type": "weird"}) == ""


def test_block_to_str_no_type():
    assert _block_to_str({}) == ""


# ---------------------------------------------------------------------------
# _decode_claude_image (mock convert_base64_to_pil_image)
# ---------------------------------------------------------------------------

def test_decode_claude_image_unsupported_source_type(monkeypatch, capsys):
    block = {"type": "image", "source": {"type": "url", "data": "x"}}
    assert _decode_claude_image(block) is None
    err = capsys.readouterr().err
    assert "unsupported source type" in err


def test_decode_claude_image_empty_data(monkeypatch, capsys):
    block = {"type": "image", "source": {"type": "base64", "data": ""}}
    assert _decode_claude_image(block) is None
    err = capsys.readouterr().err
    assert "empty base64 data" in err


def test_decode_claude_image_success(monkeypatch):
    sentinel = object()
    import claude_converter.claude as mod
    monkeypatch.setattr(mod, "convert_base64_to_pil_image", lambda data: sentinel)
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert _decode_claude_image(block) is sentinel


def test_decode_claude_image_value_error(monkeypatch, capsys):
    import claude_converter.claude as mod

    def boom(data):
        raise ValueError("bad image")

    monkeypatch.setattr(mod, "convert_base64_to_pil_image", boom)
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert _decode_claude_image(block) is None
    err = capsys.readouterr().err
    assert "unreadable image block" in err


def test_decode_claude_image_import_error(monkeypatch, capsys):
    import claude_converter.claude as mod

    def boom(data):
        raise ImportError("no pillow")

    monkeypatch.setattr(mod, "convert_base64_to_pil_image", boom)
    block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
    assert _decode_claude_image(block) is None
    err = capsys.readouterr().err
    assert "unreadable image block" in err


# ---------------------------------------------------------------------------
# _block_to_parts
# ---------------------------------------------------------------------------

def test_block_to_parts_text():
    parts = _block_to_parts({"type": "text", "text": "hello"})
    assert parts == [{"type": "text", "text": "hello"}]


def test_block_to_parts_empty_text():
    assert _block_to_parts({"type": "text", "text": "  "}) == []


def test_block_to_parts_image_success(monkeypatch):
    sentinel = object()
    import claude_converter.claude as mod
    monkeypatch.setattr(mod, "_decode_claude_image", lambda b: sentinel)
    parts = _block_to_parts({"type": "image"})
    assert parts == [{"type": "image", "image": sentinel}]


def test_block_to_parts_image_none(monkeypatch):
    import claude_converter.claude as mod
    monkeypatch.setattr(mod, "_decode_claude_image", lambda b: None)
    assert _block_to_parts({"type": "image"}) == []


def test_block_to_parts_tool_result_no_image():
    block = {"type": "tool_result", "content": "plain result"}
    parts = _block_to_parts(block)
    assert parts == [{"type": "text", "text": "<tool_result>plain result</tool_result>"}]


def test_block_to_parts_tool_result_empty():
    # No nested image, so it falls back to _block_to_str, which always
    # emits "<tool_result>...</tool_result>" even for empty content.
    block = {"type": "tool_result", "content": ""}
    assert _block_to_parts(block) == [
        {"type": "text", "text": "<tool_result></tool_result>"}
    ]


def test_block_to_parts_tool_result_with_images(monkeypatch):
    import claude_converter.claude as mod
    sentinel = object()
    monkeypatch.setattr(mod, "_decode_claude_image", lambda b: sentinel)

    block = {
        "type": "tool_result",
        "tool_use_id": "tid1",
        "content": [
            {"type": "text", "text": "before"},
            {"type": "image"},
            {"type": "text", "text": "   "},  # whitespace skipped
            {"type": "text", "text": "after"},
        ],
    }
    parts = _block_to_parts(block)
    assert parts == [
        {"type": "text", "text": "<tool_result id='tid1'>"},
        {"type": "text", "text": "before"},
        {"type": "image", "image": sentinel},
        {"type": "text", "text": "after"},
        {"type": "text", "text": "</tool_result>"},
    ]


def test_block_to_parts_tool_result_with_unreadable_image(monkeypatch):
    import claude_converter.claude as mod
    monkeypatch.setattr(mod, "_decode_claude_image", lambda b: None)
    block = {
        "type": "tool_result",
        "tool_use_id": "tid1",
        "content": [{"type": "image"}],
    }
    parts = _block_to_parts(block)
    assert parts == [
        {"type": "text", "text": "<tool_result id='tid1'>"},
        {"type": "text", "text": "</tool_result>"},
    ]


def test_block_to_parts_tool_result_missing_id(monkeypatch):
    import claude_converter.claude as mod
    monkeypatch.setattr(mod, "_decode_claude_image", lambda b: object())
    block = {"type": "tool_result", "content": [{"type": "image"}]}
    parts = _block_to_parts(block)
    assert parts[0] == {"type": "text", "text": "<tool_result id='?'>"}


def test_block_to_parts_unknown():
    assert _block_to_parts({"type": "weird"}) == []


# ---------------------------------------------------------------------------
# _content_to_str / _content_to_parts / _collect_block_types
# ---------------------------------------------------------------------------

def test_content_to_str_string():
    assert _content_to_str("hi") == "hi"


def test_content_to_str_list():
    content = [
        {"type": "text", "text": "a"},
        {"type": "thinking", "thinking": "t"},
        "notadict",
    ]
    assert _content_to_str(content) == "a\n<thinking>t</thinking>"


def test_content_to_str_list_empty_parts_filtered():
    content = [{"type": "text", "text": "  "}, {"type": "text", "text": "x"}]
    assert _content_to_str(content) == "x"


def test_content_to_str_none():
    assert _content_to_str(None) == ""


def test_content_to_parts_string():
    assert _content_to_parts("hi") == [{"type": "text", "text": "hi"}]


def test_content_to_parts_string_empty():
    assert _content_to_parts("   ") == []


def test_content_to_parts_list():
    content = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
    assert _content_to_parts(content) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_content_to_parts_none():
    assert _content_to_parts(None) == []


def test_collect_block_types_string():
    assert _collect_block_types("x") == ["raw_string"]


def test_collect_block_types_list():
    content = [{"type": "text"}, {}, "notadict"]
    assert _collect_block_types(content) == ["text", "unknown"]


def test_collect_block_types_none():
    assert _collect_block_types(None) == ["unknown"]


# ---------------------------------------------------------------------------
# record accessor helpers
# ---------------------------------------------------------------------------

def test_record_type_of():
    assert _record_type_of({"type": "user"}) == "user"
    assert _record_type_of({}) == "unknown"


def test_is_message():
    assert _is_message({"type": "user"})
    assert _is_message({"type": "assistant"})
    assert not _is_message({"type": "system"})
    assert not _is_message({})


def test_role_of():
    r = {"type": "user", "message": {"role": "assistant"}}
    assert _role_of(r) == "assistant"
    assert _role_of({"type": "user"}) == "user"
    assert _role_of({}) == "unknown"


def test_text_of():
    r = {"message": {"content": "hello"}}
    assert _text_of(r) == "hello"


def test_text_of_missing():
    assert _text_of({}) == ""


def test_timestamp_of():
    assert _timestamp_of({"timestamp": "2024-01-02T03:04:05.678Z"}) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_tokens_of():
    r = {"message": {"usage": {"output_tokens": 42}}}
    assert _tokens_of(r) == 42
    assert _tokens_of({}) == 0


def test_block_types_of():
    r = {"message": {"content": [{"type": "text"}, {"type": "image"}]}}
    assert _block_types_of(r) == ["text", "image"]


def test_block_types_of_missing():
    # Missing content defaults to [] (a list), so _collect_block_types
    # returns one entry per dict block in it -- i.e. [] for an empty list.
    assert _block_types_of({}) == []


def test_blocks_of_list():
    r = {"message": {"content": [{"type": "text"}, "x"]}}
    assert _blocks_of(r) == [{"type": "text"}]


def test_blocks_of_non_list():
    assert _blocks_of({"message": {"content": "string"}}) == []
    assert _blocks_of({}) == []


def test_total_tokens_of():
    records = [
        {"message": {"usage": {"input_tokens": 10, "output_tokens": 5}}},
        {"message": {"usage": {"input_tokens": 2, "output_tokens": 3,
                               "cache_read_input_tokens": 1,
                               "cache_creation_input_tokens": 4}}},
    ]
    totals = _total_tokens_of(records)
    assert totals == {
        "input": 12,
        "output": 8,
        "cache_read": 1,
        "cache_write": 4,
        "total": 20,
    }


def test_total_tokens_of_empty():
    assert _total_tokens_of([]) == {}


def test_total_tokens_of_zero_nop_output():
    records = [{"message": {"usage": {"cache_read_input_tokens": 5}}}]
    assert _total_tokens_of(records) == {}


# ---------------------------------------------------------------------------
# CLAUDE_SCHEMA wiring
# ---------------------------------------------------------------------------

def test_claude_schema_wiring():
    assert CLAUDE_SCHEMA.label == "CLAUDE CODE"
    r = {"type": "user", "message": {"role": "user", "content": "hi",
                                     "usage": {"output_tokens": 1}}}
    assert CLAUDE_SCHEMA.record_type_of(r) == "user"
    assert CLAUDE_SCHEMA.is_message(r) is True
    assert CLAUDE_SCHEMA.role_of(r) == "user"
    assert CLAUDE_SCHEMA.text_of(r) == "hi"
    assert CLAUDE_SCHEMA.tokens_of(r) == 1
    assert CLAUDE_SCHEMA.block_types_of(r) == ["raw_string"]
    assert CLAUDE_SCHEMA.blocks_of(r) == []


# ---------------------------------------------------------------------------
# load_session
# ---------------------------------------------------------------------------

def test_load_session_reads_records(tmp_path):
    path = tmp_path / "session.jsonl"
    write_jsonl(path, [{"type": "user"}, {"type": "assistant"}])
    records = load_session(path)
    assert records == [{"type": "user"}, {"type": "assistant"}]


def test_load_session_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "missing.jsonl")


def test_load_session_bad_extension(tmp_path):
    path = tmp_path / "session.txt"
    path.write_text("{}\n")
    with pytest.raises(ValueError):
        load_session(path)


def test_load_session_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(ValueError):
        load_session(path)


# ---------------------------------------------------------------------------
# records_to_messages
# ---------------------------------------------------------------------------

def test_records_to_messages_basic():
    records = [
        {"type": "user", "message": {"role": "user", "content": "hello"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "hi"}},
    ]
    msgs = records_to_messages(records)
    assert msgs == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_records_to_messages_skips_system():
    records = [
        {"type": "system", "message": {"content": "sys"}},
        {"type": "user", "message": {"role": "user", "content": "hello"}},
    ]
    msgs = records_to_messages(records)
    assert msgs == [{"role": "user", "content": "hello"}]


def test_records_to_messages_skips_empty_parts():
    records = [
        {"type": "user", "message": {"role": "user", "content": "   "}},
        {"type": "user", "message": {"role": "user", "content": "hello"}},
    ]
    msgs = records_to_messages(records)
    assert msgs == [{"role": "user", "content": "hello"}]


def test_records_to_messages_merges_same_role():
    records = [
        {"type": "user", "message": {"role": "user", "content": "a"}},
        {"type": "user", "message": {"role": "user", "content": "b"}},
    ]
    msgs = records_to_messages(records)
    assert msgs == [{"role": "user", "content": "a\nb"}]


def test_records_to_messages_role_fallback_to_type():
    records = [
        {"type": "user", "message": {"content": "x"}},
    ]
    msgs = records_to_messages(records)
    assert msgs == [{"role": "user", "content": "x"}]


def test_records_to_messages_tool_use_flattened():
    records = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "using tool"},
                    {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}},
                ],
            },
        },
    ]
    msgs = records_to_messages(records)
    expected_tool = "<tool_use name='Bash'>" + json.dumps({"cmd": "ls"}, ensure_ascii=False) + "</tool_use>"
    assert msgs == [{"role": "assistant", "content": f"using tool\n{expected_tool}"}]


def test_records_to_messages_with_image_keeps_list(monkeypatch):
    import claude_converter.claude as mod
    sentinel = object()
    monkeypatch.setattr(mod, "_decode_claude_image", lambda b: sentinel)
    records = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look:"},
                    {"type": "image", "source": {"type": "base64", "data": "x"}},
                ],
            },
        },
    ]
    msgs = records_to_messages(records)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look:"}
    assert content[1]["type"] == "image"
    assert content[1]["image"] is sentinel


def test_records_to_messages_empty():
    assert records_to_messages([]) == []


# ---------------------------------------------------------------------------
# session_to_messages
# ---------------------------------------------------------------------------

def test_session_to_messages_basic(tmp_path):
    path = tmp_path / "s.jsonl"
    write_jsonl(path, [
        {"type": "user", "message": {"role": "user", "content": "hello"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "hi"}},
    ])
    msgs = session_to_messages(path)
    assert msgs == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_session_to_messages_with_output(tmp_path):
    path = tmp_path / "s.jsonl"
    write_jsonl(path, [
        {"type": "user", "message": {"role": "user", "content": "hello"}},
    ])
    out = tmp_path / "out" / "messages.json"
    msgs = session_to_messages(path, output=out)
    assert msgs == [{"role": "user", "content": "hello"}]
    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == [{"role": "user", "content": "hello"}]


def test_session_to_messages_output_with_image(monkeypatch, tmp_path):
    import claude_converter.claude as mod
    sentinel = object()
    monkeypatch.setattr(mod, "_decode_claude_image", lambda b: sentinel)

    captured = {}

    def fake_safe(messages):
        captured["messages"] = messages
        return [{"role": m["role"], "content": "safe"} for m in messages]

    monkeypatch.setattr(mod, "messages_to_json_safe", fake_safe)

    path = tmp_path / "s.jsonl"
    write_jsonl(path, [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look:"},
                    {"type": "image"},
                ],
            },
        },
    ])
    out = tmp_path / "messages.json"
    msgs = session_to_messages(path, output=out)

    # In-memory keeps sentinel image
    assert isinstance(msgs[0]["content"], list)
    assert msgs[0]["content"][1]["image"] is sentinel

    # messages_to_json_safe was called with the raw messages
    assert captured["messages"] is msgs

    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == [{"role": "user", "content": "safe"}]


def test_session_to_messages_creates_parent_dirs(tmp_path):
    path = tmp_path / "s.jsonl"
    write_jsonl(path, [
        {"type": "user", "message": {"role": "user", "content": "hello"}},
    ])
    out = tmp_path / "a" / "b" / "c.json"
    session_to_messages(path, output=out)
    assert out.exists()


# ---------------------------------------------------------------------------
# _print_session_metadata
# ---------------------------------------------------------------------------

def test_print_session_metadata(capsys):
    records = [
        {"type": "system"},
        {"type": "user", "sessionId": "sid", "cwd": "/tmp", "gitBranch": "main"},
    ]
    _print_session_metadata(records)
    out = capsys.readouterr().out
    assert "sid" in out
    assert "/tmp" in out
    assert "main" in out


def test_print_session_metadata_no_user(capsys):
    _print_session_metadata([{"type": "system"}])
    assert capsys.readouterr().out == ""


def test_print_session_metadata_no_branch(capsys):
    records = [{"type": "user", "sessionId": "sid", "cwd": "/tmp"}]
    _print_session_metadata(records)
    out = capsys.readouterr().out
    assert "Branch" not in out


# ---------------------------------------------------------------------------
# _print_raw_examples
# ---------------------------------------------------------------------------

def test_print_raw_examples(capsys):
    records = [
        {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
        {"type": "user", "message": {"content": "dup"}},
        {"type": "assistant", "message": {"content": "hey"}},
    ]
    _print_raw_examples(records)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    assert "type = user" in out
    assert "type = assistant" in out
    # Only one example printed per unique type
    assert out.count("type = user") == 1


# ---------------------------------------------------------------------------
# inspect_session
# ---------------------------------------------------------------------------

def test_inspect_session_calls_run_inspection(monkeypatch, tmp_path):
    import claude_converter.claude as mod

    captured = {}

    def fake_run_inspection(path, records, schema, show_flow=False,
                           show_blocks=False, extra_header=None):
        captured.update(
            path=path,
            records=records,
            schema=schema,
            show_flow=show_flow,
            show_blocks=show_blocks,
            extra_header=extra_header,
        )

    monkeypatch.setattr(mod, "run_inspection", fake_run_inspection)

    p = tmp_path / "s.jsonl"
    write_jsonl(p, [{"type": "user", "message": {"role": "user", "content": "x"}}])

    inspect_session(p, show_flow=True, show_blocks=True)

    assert captured["path"] == p
    assert captured["schema"] is CLAUDE_SCHEMA
    assert captured["show_flow"] is True
    assert captured["show_blocks"] is True
    assert captured["extra_header"] is mod._print_session_metadata
    assert captured["records"] == [{"type": "user", "message": {"role": "user", "content": "x"}}]


def test_inspect_session_show_raw(monkeypatch, tmp_path, capsys):
    import claude_converter.claude as mod

    monkeypatch.setattr(mod, "run_inspection", lambda *a, **k: None)

    p = tmp_path / "s.jsonl"
    write_jsonl(p, [{"type": "user", "message": {"role": "user", "content": "x"}}])

    inspect_session(p, show_raw=True)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out


def test_inspect_session_no_raw(monkeypatch, tmp_path, capsys):
    import claude_converter.claude as mod

    monkeypatch.setattr(mod, "run_inspection", lambda *a, **k: None)

    p = tmp_path / "s.jsonl"
    write_jsonl(p, [{"type": "user", "message": {"role": "user", "content": "x"}}])

    inspect_session(p)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" not in out
