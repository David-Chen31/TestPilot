from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_converter import codex
from claude_converter.codex import (
    CODEX_SCHEMA,
    inspect_session_codex,
    load_session_codex,
    records_to_messages_codex,
    session_to_messages_codex,
    _payload_text,
    _payload_parts,
    _record_role,
    _record_type_of,
    _timestamp_of,
    _total_tokens_of,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _msg(role: str, content: list[dict], ts: str = "2024-01-01T00:00:00Z") -> dict:
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "message", "role": role, "content": content},
    }


def _text_block(text: str) -> dict:
    return {"type": "input_text", "text": text}


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------


def test_load_session_codex_returns_records(tmp_path):
    path = tmp_path / "s.jsonl"
    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}},
        {"type": "event_msg", "payload": {"type": "token_count"}},
    ]
    _write_jsonl(path, records)
    loaded = load_session_codex(path)
    assert loaded == records


def test_load_session_codex_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_codex(tmp_path / "nope.jsonl")


def test_load_session_codex_bad_extension(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_codex(path)


def test_load_session_codex_empty_file(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_codex(path)


# ---------------------------------------------------------------------------
# _payload_text
# ---------------------------------------------------------------------------


def test_payload_text_message_text_blocks():
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "hello"},
            {"type": "output_text", "text": "world"},
            {"type": "text", "text": "third"},
            "not a dict",
            {"type": "input_text", "text": "   "},  # whitespace-only dropped
        ],
    }
    assert _payload_text(payload) == "hello\nworld\nthird"


def test_payload_text_message_image_block():
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "data:..."},
        ],
    }
    assert _payload_text(payload) == "look\n<image>"


def test_payload_text_message_empty_content():
    assert _payload_text({"type": "message", "content": []}) == ""


def test_payload_text_message_content_null():
    assert _payload_text({"type": "message", "content": None}) == ""


def test_payload_text_reasoning():
    payload = {
        "type": "reasoning",
        "summary": [
            {"text": "thinking hard"},
            {"text": "more"},
            "not a dict",
            {"text": "  "},
        ],
    }
    assert _payload_text(payload) == "<thinking>thinking hard\nmore</thinking>"


def test_payload_text_reasoning_empty():
    assert _payload_text({"type": "reasoning", "summary": []}) == ""
    assert _payload_text({"type": "reasoning", "summary": [{"text": " "}]}) == ""


def test_payload_text_function_call():
    payload = {"type": "function_call", "name": "search", "arguments": '{"q": "x"}'}
    assert _payload_text(payload) == "<tool_use name='search'>{\"q\": \"x\"}</tool_use>"


def test_payload_text_function_call_defaults():
    payload = {"type": "function_call"}
    assert _payload_text(payload) == "<tool_use name=''></tool_use>"


def test_payload_text_function_call_output():
    payload = {"type": "function_call_output", "output": "result!"}
    assert _payload_text(payload) == "<tool_result>result!</tool_result>"


def test_payload_text_function_call_output_default():
    assert _payload_text({"type": "function_call_output"}) == "<tool_result></tool_result>"


def test_payload_text_custom_tool_call():
    payload = {"type": "custom_tool_call", "name": "run", "input": "echo hi"}
    assert _payload_text(payload) == "<tool_use name='run'>echo hi</tool_use>"


def test_payload_text_custom_tool_call_output():
    payload = {"type": "custom_tool_call_output", "output": "done"}
    assert _payload_text(payload) == "<tool_result>done</tool_result>"


def test_payload_text_unknown_returns_empty():
    assert _payload_text({"type": "something_else"}) == ""
    assert _payload_text({}) == ""


# ---------------------------------------------------------------------------
# _record_role
# ---------------------------------------------------------------------------


def test_record_role_non_response_item():
    assert _record_role({"type": "event_msg", "payload": {}}) is None


def test_record_role_message_user():
    r = {"type": "response_item", "payload": {"type": "message", "role": "user"}}
    assert _record_role(r) == "user"


def test_record_role_message_assistant():
    r = {"type": "response_item", "payload": {"type": "message", "role": "assistant"}}
    assert _record_role(r) == "assistant"


def test_record_role_message_developer_skipped():
    r = {"type": "response_item", "payload": {"type": "message", "role": "developer"}}
    assert _record_role(r) is None


def test_record_role_message_unknown_role():
    r = {"type": "response_item", "payload": {"type": "message", "role": "system"}}
    assert _record_role(r) is None


@pytest.mark.parametrize(
    "ptype,expected",
    [
        ("reasoning", "assistant"),
        ("function_call", "assistant"),
        ("custom_tool_call", "assistant"),
        ("function_call_output", "user"),
        ("custom_tool_call_output", "user"),
    ],
)
def test_record_role_tool_types(ptype, expected):
    r = {"type": "response_item", "payload": {"type": ptype}}
    assert _record_role(r) == expected


def test_record_role_payload_null():
    r = {"type": "response_item", "payload": None}
    assert _record_role(r) is None


# ---------------------------------------------------------------------------
# _payload_parts
# ---------------------------------------------------------------------------


def test_payload_parts_message_text_only():
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "hello"},
            {"type": "output_text", "text": "world"},
            {"type": "text", "text": "  "},  # dropped
            "junk",
        ],
    }
    assert _payload_parts(payload) == [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]


def test_payload_parts_message_image(monkeypatch):
    class FakeImage:
        pass

    fake = FakeImage()
    monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda uri: fake)

    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ],
    }
    parts = _payload_parts(payload)
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image"
    assert parts[1]["image"] is fake


def test_payload_parts_image_no_url_skipped(monkeypatch):
    called = []
    monkeypatch.setattr(
        codex, "convert_base64_to_pil_image", lambda uri: called.append(uri)
    )
    payload = {
        "type": "message",
        "content": [{"type": "input_image", "image_url": ""}],
    }
    assert _payload_parts(payload) == []
    assert called == []


def test_payload_parts_image_decode_error_skipped(monkeypatch, capsys):
    def boom(uri):
        raise ValueError("bad")

    monkeypatch.setattr(codex, "convert_base64_to_pil_image", boom)
    payload = {
        "type": "message",
        "content": [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "nope"},
        ],
    }
    assert _payload_parts(payload) == [{"type": "text", "text": "hi"}]
    captured = capsys.readouterr()
    assert "Skipping unreadable input_image" in captured.err


def test_payload_parts_image_import_error_skipped(monkeypatch, capsys):
    def boom(uri):
        raise ImportError("no pillow")

    monkeypatch.setattr(codex, "convert_base64_to_pil_image", boom)
    payload = {
        "type": "message",
        "content": [{"type": "input_image", "image_url": "nope"}],
    }
    assert _payload_parts(payload) == []
    captured = capsys.readouterr()
    assert "Skipping unreadable input_image" in captured.err


def test_payload_parts_non_message_text():
    payload = {"type": "function_call", "name": "x", "arguments": "y"}
    assert _payload_parts(payload) == [
        {"type": "text", "text": "<tool_use name='x'>y</tool_use>"}
    ]


def test_payload_parts_non_message_empty():
    # _payload_text for function_call_output always emits the
    # "<tool_result>...</tool_result>" wrapper, which is non-empty even
    # when the output field is empty, so a single text part is produced.
    assert _payload_parts({"type": "function_call_output", "output": ""}) == [
        {"type": "text", "text": "<tool_result></tool_result>"}
    ]


# ---------------------------------------------------------------------------
# records_to_messages_codex
# ---------------------------------------------------------------------------


def test_records_to_messages_basic_conversation():
    records = [
        _msg("user", [_text_block("hi")]),
        _msg("assistant", [{"type": "output_text", "text": "hello"}]),
        _msg("user", [_text_block("bye")]),
    ]
    messages = records_to_messages_codex(records)
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]


def test_records_to_messages_merges_same_role():
    records = [
        _msg("user", [_text_block("one")]),
        _msg("user", [_text_block("two")]),
    ]
    messages = records_to_messages_codex(records)
    assert messages == [{"role": "user", "content": "one\ntwo"}]


def test_records_to_messages_skips_non_messages():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count"}},
        _msg("user", [_text_block("hi")]),
        {"type": "response_item", "payload": {"type": "message", "role": "developer", "content": []}},
    ]
    messages = records_to_messages_codex(records)
    assert messages == [{"role": "user", "content": "hi"}]


def test_records_to_messages_skips_empty_parts():
    records = [
        _msg("user", [_text_block("   ")]),
        _msg("assistant", [{"type": "output_text", "text": "ok"}]),
    ]
    messages = records_to_messages_codex(records)
    assert messages == [{"role": "assistant", "content": "ok"}]


def test_records_to_messages_tool_calls_and_outputs():
    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                               "content": [{"type": "input_text", "text": "run it"}]}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "arguments": "ls"}},
        {"type": "response_item", "payload": {"type": "function_call_output", "output": "file.txt"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                               "content": [{"type": "output_text", "text": "done"}]}},
    ]
    messages = records_to_messages_codex(records)
    assert messages == [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "<tool_use name='shell'>ls</tool_use>"},
        {"role": "user", "content": "<tool_result>file.txt</tool_result>"},
        {"role": "assistant", "content": "done"},
    ]


def test_records_to_messages_reasoning_merged_into_assistant():
    records = [
        {"type": "response_item", "payload": {"type": "reasoning", "summary": [{"text": "hmm"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                               "content": [{"type": "output_text", "text": "answer"}]}},
    ]
    messages = records_to_messages_codex(records)
    assert messages == [
        {"role": "assistant", "content": "<thinking>hmm</thinking>\nanswer"},
    ]


def test_records_to_messages_multimodal(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda uri: sentinel)

    records = [
        _msg("user", [
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]),
    ]
    messages = records_to_messages_codex(records)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image"
    assert content[1]["image"] is sentinel


def test_records_to_messages_empty_records():
    assert records_to_messages_codex([]) == []


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------


def test_session_to_messages_no_output(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [
        _msg("user", [_text_block("hi")]),
        _msg("assistant", [{"type": "output_text", "text": "yo"}]),
    ])
    messages = session_to_messages_codex(path)
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_session_to_messages_writes_output(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [
        _msg("user", [_text_block("hi")]),
    ])
    out = tmp_path / "sub" / "out.json"
    messages = session_to_messages_codex(path, output=out)
    assert messages == [{"role": "user", "content": "hi"}]
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == [{"role": "user", "content": "hi"}]


def test_session_to_messages_output_serializes_images(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (2, 2), color="red")
    monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda uri: img)

    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [
        _msg("user", [
            {"type": "input_text", "text": "img"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]),
    ])
    out = tmp_path / "out.json"
    messages = session_to_messages_codex(path, output=out)

    # In-memory list keeps the real PIL image.
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[1]["image"] is img

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded[0]["content"][0] == {"type": "text", "text": "img"}
    assert loaded[0]["content"][1]["type"] == "image"
    assert loaded[0]["content"][1]["image"].startswith("data:image/")


def test_session_to_messages_output_non_pil_image_falls_back_to_repr(tmp_path, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(codex, "convert_base64_to_pil_image", lambda uri: sentinel)

    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [
        _msg("user", [
            {"type": "input_text", "text": "img"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]),
    ])
    out = tmp_path / "out.json"
    messages = session_to_messages_codex(path, output=out)

    assert messages[0]["content"][1]["image"] is sentinel

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded[0]["content"][1]["type"] == "image"
    assert loaded[0]["content"][1]["image"] == repr(sentinel)


# ---------------------------------------------------------------------------
# Inspector helpers
# ---------------------------------------------------------------------------


def test_record_type_of_with_sub():
    r = {"type": "response_item", "payload": {"type": "message"}}
    assert _record_type_of(r) == "response_item:message"


def test_record_type_of_without_sub():
    r = {"type": "event_msg", "payload": {"other": 1}}
    assert _record_type_of(r) == "event_msg"


def test_record_type_of_missing_type():
    assert _record_type_of({}) == "unknown"


def test_timestamp_of():
    r = {"timestamp": "2024-01-02T03:04:05.123Z"}
    assert _timestamp_of(r) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_total_tokens_of_latest_snapshot():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 1, "output_tokens": 2,
                                  "cached_input_tokens": 3, "total_tokens": 5},
        }}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 10, "output_tokens": 20,
                                  "cached_input_tokens": 30, "total_tokens": 60},
        }}},
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cached_input_tokens": 30,
        "total_tokens": 60,
    }


def test_total_tokens_of_ignores_null_info():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 1, "output_tokens": 2,
                                  "cached_input_tokens": 3, "total_tokens": 5},
        }}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": None}},
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 1,
        "output_tokens": 2,
        "cached_input_tokens": 3,
        "total_tokens": 5,
    }


def test_total_tokens_of_missing_fields():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 4},
        }}},
    ]
    assert _total_tokens_of(records) == {
        "input_tokens": 4,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "total_tokens": 0,
    }


def test_total_tokens_of_none():
    assert _total_tokens_of([]) == {}
    assert _total_tokens_of([{"type": "x", "payload": {}}]) == {}


def test_total_tokens_of_null_usage():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": None,
        }}},
    ]
    assert _total_tokens_of(records) == {}


# ---------------------------------------------------------------------------
# CODEX_SCHEMA
# ---------------------------------------------------------------------------


def test_codex_schema_fields():
    assert CODEX_SCHEMA.label == "CODEX"

    r = {"type": "response_item", "payload": {"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": "hi"}]}}
    assert CODEX_SCHEMA.is_message(r) is True
    assert CODEX_SCHEMA.role_of(r) == "user"
    assert CODEX_SCHEMA.text_of(r) == "hi"

    other = {"type": "event_msg", "payload": {"type": "token_count"}}
    assert CODEX_SCHEMA.is_message(other) is False
    assert CODEX_SCHEMA.role_of(other) == "other"


def test_codex_schema_timestamp():
    r = {"timestamp": "2024-01-02T03:04:05Z"}
    assert CODEX_SCHEMA.timestamp_of(r) == "2024-01-02 03:04:05"


# ---------------------------------------------------------------------------
# inspect_session_codex
# ---------------------------------------------------------------------------


def test_inspect_session_codex_smoke(tmp_path, capsys):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [
        _msg("user", [_text_block("hi")]),
        _msg("assistant", [{"type": "output_text", "text": "yo"}]),
    ])
    inspect_session_codex(path)
    out = capsys.readouterr().out
    assert "CODEX" in out


def test_inspect_session_codex_flow_mode(tmp_path, capsys):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [
        _msg("user", [_text_block("hi")]),
    ])
    inspect_session_codex(path, show_flow=True)
    out = capsys.readouterr().out
    assert "CODEX" in out
