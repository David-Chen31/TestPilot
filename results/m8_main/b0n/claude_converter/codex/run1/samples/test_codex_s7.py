import json
from pathlib import Path

import pytest

import claude_converter.codex as codex
from claude_converter.codex import (
    load_session_codex,
    records_to_messages_codex,
    session_to_messages_codex,
    inspect_session_codex,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _msg(role, content):
    return {
        "type": "response_item",
        "payload": {"type": "message", "role": role, "content": content},
    }


def _text_block(text):
    return {"type": "output_text", "text": text}


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------

def test_load_session_codex_roundtrip(tmp_path):
    records = [
        {"type": "response_item", "payload": {"type": "message"}},
        {"type": "event_msg", "payload": {"type": "token_count"}},
    ]
    path = _write_jsonl(tmp_path / "session.jsonl", records)
    assert load_session_codex(path) == records


def test_load_session_codex_accepts_str_path(tmp_path):
    path = _write_jsonl(tmp_path / "s.jsonl", [{"a": 1}])
    assert load_session_codex(str(path)) == [{"a": 1}]


def test_load_session_codex_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_codex(tmp_path / "nope.jsonl")


def test_load_session_codex_bad_extension(tmp_path):
    path = tmp_path / "session.txt"
    path.write_text("{}\n")
    with pytest.raises(ValueError):
        load_session_codex(path)


def test_load_session_codex_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(ValueError):
        load_session_codex(path)


# ---------------------------------------------------------------------------
# _payload_text (private helper, exercised directly)
# ---------------------------------------------------------------------------

def test_payload_text_message():
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "hello"},
            {"type": "text", "text": "world"},
            {"type": "unknown_type", "text": "ignore"},
        ],
    }
    assert codex._payload_text(payload) == "hello\nworld"


def test_payload_text_message_blank_skipped():
    payload = {
        "type": "message",
        "content": [
            {"type": "output_text", "text": "hi"},
            {"type": "output_text", "text": "   "},
        ],
    }
    assert codex._payload_text(payload) == "hi"


def test_payload_text_message_non_dict_block_skipped():
    payload = {"type": "message", "content": ["notadict", {"type": "text", "text": "ok"}]}
    assert codex._payload_text(payload) == "ok"


def test_payload_text_message_image_marker():
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "before"},
            {"type": "input_image", "image_url": "data:..."},
        ],
    }
    assert codex._payload_text(payload) == "before\n<image>"


def test_payload_text_reasoning():
    payload = {"type": "reasoning", "summary": [{"text": "thinking"}, {"text": "more"}]}
    assert codex._payload_text(payload) == "<thinking>thinking\nmore</thinking>"


def test_payload_text_reasoning_empty():
    assert codex._payload_text({"type": "reasoning", "summary": []}) == ""
    assert codex._payload_text({"type": "reasoning"}) == ""


def test_payload_text_function_call():
    payload = {"type": "function_call", "name": "do_it", "arguments": '{"x": 1}'}
    assert codex._payload_text(payload) == "<tool_use name='do_it'>{\"x\": 1}</tool_use>"


def test_payload_text_function_call_output():
    payload = {"type": "function_call_output", "output": "result"}
    assert codex._payload_text(payload) == "<tool_result>result</tool_result>"


def test_payload_text_custom_tool_call():
    payload = {"type": "custom_tool_call", "name": "c", "input": "in"}
    assert codex._payload_text(payload) == "<tool_use name='c'>in</tool_use>"


def test_payload_text_custom_tool_call_output():
    payload = {"type": "custom_tool_call_output", "output": "out"}
    assert codex._payload_text(payload) == "<tool_result>out</tool_result>"


def test_payload_text_unknown():
    assert codex._payload_text({"type": "whatever"}) == ""
    assert codex._payload_text({}) == ""


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------

def test_record_role_message_user():
    assert codex._record_role(_msg("user", [])) == "user"


def test_record_role_message_assistant():
    assert codex._record_role(_msg("assistant", [])) == "assistant"


def test_record_role_developer_skipped():
    assert codex._record_role(_msg("developer", [])) is None


def test_record_role_message_unknown_role():
    assert codex._record_role(_msg("system", [])) is None


def test_record_role_non_response_item():
    assert codex._record_role({"type": "event_msg", "payload": {}}) is None


def test_record_role_tool_roles():
    assert codex._record_role({"type": "response_item", "payload": {"type": "reasoning"}}) == "assistant"
    assert codex._record_role({"type": "response_item", "payload": {"type": "function_call"}}) == "assistant"
    assert codex._record_role({"type": "response_item", "payload": {"type": "custom_tool_call"}}) == "assistant"
    assert codex._record_role({"type": "response_item", "payload": {"type": "function_call_output"}}) == "user"
    assert codex._record_role({"type": "response_item", "payload": {"type": "custom_tool_call_output"}}) == "user"


def test_record_role_unknown_payload_type():
    assert codex._record_role({"type": "response_item", "payload": {"type": "mystery"}}) is None


def test_record_role_null_payload():
    assert codex._record_role({"type": "response_item", "payload": None}) is None


# ---------------------------------------------------------------------------
# _payload_parts
# ---------------------------------------------------------------------------

def test_payload_parts_message_text():
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "a"},
            {"type": "input_text", "text": "  "},
            {"type": "output_text", "text": "b"},
        ],
    }
    assert codex._payload_parts(payload) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_payload_parts_message_no_image_url_skipped():
    payload = {
        "type": "message",
        "content": [
            {"type": "input_image", "image_url": ""},
            {"type": "input_image"},
        ],
    }
    assert codex._payload_parts(payload) == []


def test_payload_parts_image_decode_error_skipped(capsys):
    payload = {
        "type": "message",
        "content": [{"type": "input_image", "image_url": "not-a-valid-data-uri"}],
    }
    assert codex._payload_parts(payload) == []
    err = capsys.readouterr().err
    assert "Skipping unreadable input_image" in err


def test_payload_parts_image_import_error(monkeypatch, capsys):
    def boom(uri):
        raise ImportError("no pillow")

    monkeypatch.setattr(codex, "convert_base64_to_pil_image", boom)
    payload = {
        "type": "message",
        "content": [{"type": "input_image", "image_url": "x"}],
    }
    assert codex._payload_parts(payload) == []
    assert "Skipping unreadable input_image" in capsys.readouterr().err


def test_payload_parts_image_success(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda uri: sentinel)
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
        ],
    }
    assert codex._payload_parts(payload) == [
        {"type": "text", "text": "hi"},
        {"type": "image", "image": sentinel},
    ]


def test_payload_parts_non_message_uses_text():
    payload = {"type": "function_call", "name": "f", "arguments": "{}"}
    assert codex._payload_parts(payload) == [
        {"type": "text", "text": "<tool_use name='f'>{}</tool_use>"}
    ]


def test_payload_parts_non_message_empty_text():
    assert codex._payload_parts({"type": "reasoning", "summary": []}) == []


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------

def test_records_to_messages_simple():
    records = [
        _msg("user", [_text_block("hello")]),
        _msg("assistant", [_text_block("hi there")]),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_records_to_messages_skips_non_messages():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count"}},
        _msg("user", [_text_block("x")]),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "x"}]


def test_records_to_messages_skips_developer():
    records = [
        _msg("developer", [_text_block("sys")]),
        _msg("user", [_text_block("hello")]),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "hello"}]


def test_records_to_messages_skips_empty_parts():
    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}},
        _msg("user", [_text_block("real")]),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "real"}]


def test_records_to_messages_merges_same_role():
    records = [
        _msg("user", [_text_block("part1")]),
        _msg("user", [_text_block("part2")]),
        _msg("assistant", [_text_block("reply")]),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "part1\npart2"},
        {"role": "assistant", "content": "reply"},
    ]


def test_records_to_messages_tool_call_flattened():
    records = [
        _msg("user", [_text_block("go")]),
        {"type": "response_item", "payload": {"type": "function_call", "name": "f", "arguments": "{}"}},
        {"type": "response_item", "payload": {"type": "function_call_output", "output": "done"}},
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "<tool_use name='f'>{}</tool_use>"},
        {"role": "user", "content": "<tool_result>done</tool_result>"},
    ]


def test_records_to_messages_reasoning_merged_into_assistant():
    records = [
        _msg("assistant", [_text_block("answer")]),
        {"type": "response_item", "payload": {"type": "reasoning", "summary": [{"text": "hmm"}]}},
    ]
    assert records_to_messages_codex(records) == [
        {"role": "assistant", "content": "answer\n<thinking>hmm</thinking>"},
    ]


def test_records_to_messages_empty():
    assert records_to_messages_codex([]) == []


def test_records_to_messages_multimodal_image(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda uri: sentinel)
    records = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
                ],
            },
        }
    ]
    result = records_to_messages_codex(records)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image", "image": sentinel},
    ]


def test_records_to_messages_multimodal_merge_with_image(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda uri: sentinel)
    records = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:x"}],
            },
        },
        _msg("user", [_text_block("text after image")]),
    ]
    result = records_to_messages_codex(records)
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sentinel},
                {"type": "text", "text": "text after image"},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------

def test_session_to_messages_codex(tmp_path):
    records = [
        _msg("user", [_text_block("hello")]),
        _msg("assistant", [_text_block("hi")]),
    ]
    path = _write_jsonl(tmp_path / "s.jsonl", records)
    assert session_to_messages_codex(path) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_session_to_messages_codex_writes_output(tmp_path):
    records = [_msg("user", [_text_block("hello")])]
    path = _write_jsonl(tmp_path / "s.jsonl", records)
    out = tmp_path / "sub" / "out.json"
    result = session_to_messages_codex(path, output=out)
    assert result == [{"role": "user", "content": "hello"}]
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == [{"role": "user", "content": "hello"}]


def test_session_to_messages_codex_output_serializes_images(tmp_path, monkeypatch):
    # Use a real PIL image if Pillow is present; otherwise skip.
    PIL = pytest.importorskip("PIL.Image")
    import base64
    from io import BytesIO

    buf = BytesIO()
    PIL.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    uri = f"data:image/png;base64,{b64}"

    records = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "img"},
                    {"type": "input_image", "image_url": uri},
                ],
            },
        }
    ]
    path = _write_jsonl(tmp_path / "s.jsonl", records)
    out = tmp_path / "out.json"
    result = session_to_messages_codex(path, output=out)

    # in-memory keeps real PIL image
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "img"}
    assert content[1]["type"] == "image"
    assert isinstance(content[1]["image"], PIL.Image)

    # saved output is JSON-serializable data URI
    saved = json.loads(out.read_text(encoding="utf-8"))
    saved_content = saved[0]["content"]
    assert saved_content[0] == {"type": "text", "text": "img"}
    assert saved_content[1]["type"] == "image"
    assert saved_content[1]["image"].startswith("data:image/")


# ---------------------------------------------------------------------------
# inspection helpers
# ---------------------------------------------------------------------------

def test_record_type_of():
    assert codex._record_type_of({"type": "response_item", "payload": {"type": "message"}}) == "response_item:message"
    assert codex._record_type_of({"type": "event_msg"}) == "event_msg"
    assert codex._record_type_of({}) == "unknown"
    assert codex._record_type_of({"type": "x", "payload": {"type": ""}}) == "x"


def test_timestamp_of():
    assert codex._timestamp_of({"timestamp": "2024-01-02T03:04:05.678Z"}) == "2024-01-02 03:04:05"
    assert codex._timestamp_of({}) == ""


def test_total_tokens_of_empty():
    assert codex._total_tokens_of([]) == []
    assert codex._total_tokens_of([{"payload": {"type": "message"}}]) == {}


def test_total_tokens_of_latest_snapshot():
    records = [
        {"payload": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 5, "total_tokens": 5}}}},
        {
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "cached_input_tokens": 3,
                        "total_tokens": 33,
                    }
                },
            }
        },
    ]
    assert codex._total_tokens_of(records) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cached_input_tokens": 3,
        "total_tokens": 33,
    }


def test_total_tokens_of_null_info_and_none_usage():
    records = [
        {"payload": {"type": "token_count", "info": None}},
        {"payload": {"type": "token_count", "info": {"total_token_usage": None}}},
        {"payload": {"type": "token_count", "info": {"total_token_usage": {"total_tokens": 7}}}},
    ]
    assert codex._total_tokens_of(records) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "total_tokens": 7,
    }


# ---------------------------------------------------------------------------
# CODEX_SCHEMA wiring
# ---------------------------------------------------------------------------

def test_codex_schema_fields():
    schema = codex.CODEX_SCHEMA
    assert schema.label == "CODEX"
    rec = {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hey"}]}}
    assert schema.is_message(rec) is True
    assert schema.role_of(rec) == "user"
    assert schema.text_of(rec) == "hey"
    assert schema.record_type_of(rec) == "response_item:message"

    other = {"type": "event_msg", "payload": {"type": "token_count"}}
    assert schema.is_message(other) is False
    assert schema.role_of(other) == "other"


# ---------------------------------------------------------------------------
# inspect_session_codex
# ---------------------------------------------------------------------------

def test_inspect_session_codex_runs(tmp_path, capsys):
    records = [
        _msg("user", [_text_block("hello")]),
        _msg("assistant", [_text_block("hi")]),
    ]
    path = _write_jsonl(tmp_path / "s.jsonl", records)

    calls = {}
    real_run = codex.run_inspection

    def spy_run(p, recs, schema, show_flow=False, show_blocks=False, **kw):
        calls["path"] = p
        calls["records"] = recs
        calls["schema"] = schema
        calls["show_flow"] = show_flow
        calls["show_blocks"] = show_blocks
        return real_run(p, recs, schema, show_flow=show_flow, show_blocks=show_blocks, **kw)

    codex.run_inspection = spy_run
    try:
        inspect_session_codex(path)
    finally:
        codex.run_inspection = real_run

    assert calls["path"] == path
    assert calls["records"] == records
    assert calls["schema"] is codex.CODEX_SCHEMA
    assert calls["show_flow"] is False
    assert calls["show_blocks"] is False


def test_inspect_session_codex_show_flow(tmp_path, capsys):
    records = [_msg("user", [_text_block("hello")])]
    path = _write_jsonl(tmp_path / "s.jsonl", records)

    calls = {}
    real_run = codex.run_inspection

    def spy_run(p, recs, schema, show_flow=False, show_blocks=False, **kw):
        calls["show_flow"] = show_flow
        calls["show_blocks"] = show_blocks
        return None

    codex.run_inspection = spy_run
    try:
        inspect_session_codex(path, show_flow=True, show_blocks=True, show_raw=True)
    finally:
        codex.run_inspection = real_run

    assert calls["show_flow"] is True
    assert calls["show_blocks"] is True
