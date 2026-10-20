from __future__ import annotations

import base64
import io
import json
import sys
import types
from pathlib import Path

import pytest

from claude_converter import codex
from claude_converter.codex import (
    inspect_session_codex,
    load_session_codex,
    records_to_messages_codex,
    session_to_messages_codex,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path: Path, records: list[dict], name: str = "session.jsonl") -> Path:
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _msg(role: str, *blocks) -> dict:
    return {"type": "response_item", "payload": {"type": "message", "role": role, "content": list(blocks)}}


def _text_block(text: str, btype: str = "input_text") -> dict:
    return {"type": btype, "text": text}


# A tiny valid 1x1 PNG, base64-encoded. This lets us build an image data URI
# without needing Pillow installed (Pillow is only needed to *decode*).
_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLv"
    "AAAAAElFTkSuQmCC"
)


def _png_data_uri() -> str:
    return "data:image/png;base64," + _PNG_1X1_B64


class _FakePILImage:
    """Stand-in for PIL.Image.Image when Pillow is not installed."""


def _has_pillow() -> bool:
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        return False
    return True


def _install_fake_pillow(monkeypatch):
    """
    Make image tests runnable even without Pillow installed.

    The code under test lazily imports PIL inside
    convert_base64_to_pil_image (which codex.py imports as a module-level
    name). When Pillow is unavailable we stub codex.convert_base64_to_pil_image
    to return a fake image object.
    """
    if _has_pillow():
        return  # Pillow available: real path works.

    def _fake_convert(uri):
        return _FakePILImage()

    monkeypatch.setattr(codex, "convert_base64_to_pil_image", _fake_convert)


# ---------------------------------------------------------------------------
# load_session_codex
# ---------------------------------------------------------------------------

def test_load_session_codex_roundtrip(tmp_path):
    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}},
        {"type": "event_msg", "payload": {"type": "token_count"}},
    ]
    path = _write_jsonl(tmp_path, records)
    assert load_session_codex(path) == records


def test_load_session_codex_accepts_str(tmp_path):
    path = _write_jsonl(tmp_path, [{"type": "x"}])
    assert load_session_codex(str(path)) == [{"type": "x"}]


def test_load_session_codex_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_codex(tmp_path / "nope.jsonl")


def test_load_session_codex_bad_extension(tmp_path):
    p = tmp_path / "session.txt"
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_codex(p)


def test_load_session_codex_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_codex(p)


def test_load_session_codex_skips_invalid_lines(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text('{"type": "a"}\nnot json\n{"type": "b"}\n', encoding="utf-8")
    assert load_session_codex(p) == [{"type": "a"}, {"type": "b"}]
    err = capsys.readouterr().err
    assert "Line 2 is invalid JSON" in err


# ---------------------------------------------------------------------------
# _payload_text / _payload_parts indirect via records_to_messages_codex
# ---------------------------------------------------------------------------

def test_message_text_flattening_and_roles():
    records = [
        _msg("user", _text_block("hello", "input_text")),
        _msg("assistant", _text_block("hi there", "output_text")),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_message_developer_role_skipped():
    records = [_msg("developer", _text_block("sys")), _msg("user", _text_block("u"))]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "u"}]


def test_message_unknown_role_skipped():
    records = [_msg("tool", _text_block("x")), _msg("user", _text_block("u"))]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "u"}]


def test_message_text_block_types():
    records = [
        _msg("user", _text_block("a", "input_text"), {"type": "text", "text": "b"}),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "a\nb"}]


def test_message_empty_blocks_skipped():
    records = [
        _msg("user", _text_block("   "), {"type": "unknown"}),
    ]
    assert records_to_messages_codex(records) == []


def test_message_non_dict_block_skipped():
    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                               "content": ["nope", {"type": "input_text", "text": "ok"}]}},
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "ok"}]


def test_reasoning_wrapped_in_thinking():
    records = [
        {"type": "response_item", "payload": {"type": "reasoning",
                                              "summary": [{"text": "pondering"}, {"text": "more"}]}},
    ]
    assert records_to_messages_codex(records) == [
        {"role": "assistant", "content": "<thinking>pondering\nmore</thinking>"}
    ]


def test_reasoning_empty_yields_nothing():
    records = [
        {"type": "response_item", "payload": {"type": "reasoning", "summary": [{"text": "  "}]}},
    ]
    assert records_to_messages_codex(records) == []


def test_reasoning_non_dict_summary_skipped():
    records = [
        {"type": "response_item", "payload": {"type": "reasoning", "summary": ["x", {"text": "ok"}]}},
    ]
    assert records_to_messages_codex(records) == [
        {"role": "assistant", "content": "<thinking>ok</thinking>"}
    ]


def test_function_call_formatted():
    records = [
        {"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                              "arguments": '{"cmd": "ls"}'}},
    ]
    assert records_to_messages_codex(records) == [
        {"role": "assistant", "content": "<tool_use name='shell'>{\"cmd\": \"ls\"}</tool_use>"}
    ]


def test_function_call_output_formatted():
    records = [
        {"type": "response_item", "payload": {"type": "function_call_output", "output": "result"}},
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "<tool_result>result</tool_result>"}
    ]


def test_custom_tool_call_and_output():
    records = [
        {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "edit",
                                              "input": "data"}},
        {"type": "response_item", "payload": {"type": "custom_tool_call_output", "output": "done"}},
    ]
    assert records_to_messages_codex(records) == [
        {"role": "assistant", "content": "<tool_use name='edit'>data</tool_use>"},
        {"role": "user", "content": "<tool_result>done</tool_result>"},
    ]


def test_non_response_item_ignored():
    records = [
        {"type": "event_msg", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "ignored"}]}},
        _msg("user", _text_block("kept")),
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "kept"}]


def test_unknown_payload_type_yields_nothing():
    records = [{"type": "response_item", "payload": {"type": "weird"}}]
    assert records_to_messages_codex(records) == []


def test_payload_null_handled():
    records = [{"type": "response_item", "payload": None}]
    assert records_to_messages_codex(records) == []


def test_empty_records():
    assert records_to_messages_codex([]) == []


# ---------------------------------------------------------------------------
# merging consecutive same-role records
# ---------------------------------------------------------------------------

def test_consecutive_same_role_merged():
    records = [
        _msg("user", _text_block("one")),
        _msg("user", _text_block("two")),
        _msg("assistant", _text_block("resp", "output_text")),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "user", "content": "one\ntwo"},
        {"role": "assistant", "content": "resp"},
    ]


def test_reasoning_merges_into_assistant_message():
    records = [
        {"type": "response_item", "payload": {"type": "reasoning", "summary": [{"text": "think"}]}},
        _msg("assistant", _text_block("answer", "output_text")),
    ]
    assert records_to_messages_codex(records) == [
        {"role": "assistant", "content": "<thinking>think</thinking>\nanswer"},
    ]


# ---------------------------------------------------------------------------
# images / multimodal
# ---------------------------------------------------------------------------

def test_image_part_in_content(monkeypatch):
    _install_fake_pillow(monkeypatch)
    uri = _png_data_uri()
    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": uri},
        ]}},
    ]
    result = records_to_messages_codex(records)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"
    if _has_pillow():
        from PIL import Image
        assert isinstance(content[1]["image"], Image.Image)
    else:
        assert isinstance(content[1]["image"], _FakePILImage)


def test_input_image_without_url_skipped():
    records = [
        _msg("user", _text_block("only text")),
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": ""},
        ]}},
    ]
    assert records_to_messages_codex(records) == [{"role": "user", "content": "only text"}]


def test_input_image_text_only_payload_text_uses_tag():
    # _payload_text renders the placeholder, used only when no image parts remain
    assert codex._payload_text(
        {"type": "message", "content": [{"type": "input_image"}, {"type": "input_text", "text": "x"}]}
    ) == "<image>\nx"


def test_bad_image_skipped_with_stderr(capsys):
    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,notbase64!!"},
        ]}},
    ]
    result = records_to_messages_codex(records)
    assert result == []
    err = capsys.readouterr().err
    # Without Pillow installed this hits the ImportError branch; with Pillow
    # installed it hits the ValueError branch. Either way it prints a skip
    # message mentioning the readable-image skip.
    assert "Skipping unreadable input_image" in err


def test_bad_image_skipped_with_value_error(capsys, monkeypatch):
    # Force the ValueError path deterministically by stubbing the converter.
    def _raise_value_error(uri):
        raise ValueError("boom")

    monkeypatch.setattr(codex, "convert_base64_to_pil_image", _raise_value_error)

    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "keep"},
            {"type": "input_image", "image_url": "data:image/png;base64,xyz"},
        ]}},
    ]
    result = records_to_messages_codex(records)
    assert result == [{"role": "user", "content": "keep"}]
    err = capsys.readouterr().err
    assert "Skipping unreadable input_image" in err


def test_bad_image_import_error_skipped(capsys, monkeypatch):
    # Simulate Pillow being absent -> ImportError branch of _payload_parts.
    def _raise_import_error(uri):
        raise ImportError("Pillow is required")

    monkeypatch.setattr(codex, "convert_base64_to_pil_image", _raise_import_error)

    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,xyz"},
            {"type": "input_text", "text": "after"},
        ]}},
    ]
    result = records_to_messages_codex(records)
    assert result == [{"role": "user", "content": "after"}]
    err = capsys.readouterr().err
    assert "Skipping unreadable input_image" in err


def test_image_merge_across_same_role(monkeypatch):
    _install_fake_pillow(monkeypatch)
    uri = _png_data_uri()
    records = [
        _msg("user", _text_block("first")),
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": uri},
        ]}},
    ]
    result = records_to_messages_codex(records)
    assert len(result) == 1
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "first"}
    assert content[1]["type"] == "image"
    if _has_pillow():
        from PIL import Image
        assert isinstance(content[1]["image"], Image.Image)
    else:
        assert isinstance(content[1]["image"], _FakePILImage)


# ---------------------------------------------------------------------------
# session_to_messages_codex
# ---------------------------------------------------------------------------

def test_session_to_messages_codex_no_output(tmp_path):
    records = [_msg("user", _text_block("hey"))]
    path = _write_jsonl(tmp_path, records)
    assert session_to_messages_codex(path) == [{"role": "user", "content": "hey"}]


def test_session_to_messages_codex_writes_output(tmp_path):
    records = [_msg("user", _text_block("hey"))]
    path = _write_jsonl(tmp_path, records)
    out = tmp_path / "nested" / "out.json"
    result = session_to_messages_codex(path, output=out)
    assert result == [{"role": "user", "content": "hey"}]
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == [{"role": "user", "content": "hey"}]


def test_session_to_messages_codex_output_image_serialized(tmp_path, monkeypatch):
    if not _has_pillow():
        pytest.skip("Pillow not installed; real image serialization path unavailable")
    from PIL import Image

    uri = _png_data_uri()
    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": uri},
        ]}},
    ]
    path = _write_jsonl(tmp_path, records)
    out = tmp_path / "out.json"
    result = session_to_messages_codex(path, output=out)

    # in-memory result keeps PIL images
    assert isinstance(result[0]["content"][0]["image"], Image.Image)

    # on disk becomes a data URI string
    saved = json.loads(out.read_text(encoding="utf-8"))
    img_repr = saved[0]["content"][0]["image"]
    assert isinstance(img_repr, str)
    assert img_repr.startswith("data:image/")


def test_session_to_messages_codex_output_image_without_pillow(tmp_path, monkeypatch):
    # When Pillow is absent, convert_base64_to_pil_image raises ImportError;
    # the image is dropped and the message is text-only (or absent if no text).
    def _raise_import_error(uri):
        raise ImportError("Pillow is required")

    monkeypatch.setattr(codex, "convert_base64_to_pil_image", _raise_import_error)

    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "keep"},
            {"type": "input_image", "image_url": "data:image/png;base64,xyz"},
        ]}},
    ]
    path = _write_jsonl(tmp_path, records)
    out = tmp_path / "out.json"
    result = session_to_messages_codex(path, output=out)
    assert result == [{"role": "user", "content": "keep"}]
    assert json.loads(out.read_text(encoding="utf-8")) == [{"role": "user", "content": "keep"}]


def test_session_to_messages_codex_accepts_str_output(tmp_path):
    records = [_msg("user", _text_block("hi"))]
    path = _write_jsonl(tmp_path, records)
    out = tmp_path / "o.json"
    result = session_to_messages_codex(str(path), output=str(out))
    assert result == [{"role": "user", "content": "hi"}]
    assert json.loads(out.read_text(encoding="utf-8")) == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# inspector internals / schema
# ---------------------------------------------------------------------------

def test_record_type_of():
    assert codex._record_type_of({"type": "response_item", "payload": {"type": "message"}}) == \
        "response_item:message"
    assert codex._record_type_of({"type": "event_msg", "payload": {}}) == "event_msg"
    assert codex._record_type_of({}) == "unknown"


def test_timestamp_of():
    r = {"timestamp": "2024-01-02T03:04:05.123456Z"}
    assert codex._timestamp_of(r) == "2024-01-02 03:04:05"
    assert codex._timestamp_of({}) == ""


def test_total_tokens_of_latest_snapshot():
    records = [
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 1, "output_tokens": 2,
                                  "cached_input_tokens": 3, "total_tokens": 3}}}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 10, "output_tokens": 20,
                                  "cached_input_tokens": 5, "total_tokens": 30}}}},
    ]
    assert codex._total_tokens_of(records) == {
        "input_tokens": 10, "output_tokens": 20,
        "cached_input_tokens": 5, "total_tokens": 30,
    }


def test_total_tokens_of_no_tokens():
    assert codex._total_tokens_of([]) == {}
    assert codex._total_tokens_of([{"type": "event_msg", "payload": {"type": "other"}}]) == {}


def test_total_tokens_of_null_info():
    records = [{"type": "event_msg", "payload": {"type": "token_count", "info": None}}]
    assert codex._total_tokens_of(records) == {}


def test_total_tokens_of_missing_keys_default_zero():
    records = [{"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": 4}}}}]
    assert codex._total_tokens_of(records) == {
        "input_tokens": 4, "output_tokens": 0,
        "cached_input_tokens": 0, "total_tokens": 0,
    }


def test_total_tokens_of_falsey_usage_ignored():
    # usage value of null should not clobber a previous snapshot.
    records = [
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 1, "output_tokens": 1,
                                  "cached_input_tokens": 1, "total_tokens": 2}}}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": None}}},
    ]
    assert codex._total_tokens_of(records) == {
        "input_tokens": 1, "output_tokens": 1,
        "cached_input_tokens": 1, "total_tokens": 2,
    }


def test_codex_schema_helpers():
    schema = codex.CODEX_SCHEMA
    assert schema.label == "CODEX"

    msg = {"type": "response_item", "payload": {"type": "message", "role": "user",
                                                "content": [{"type": "input_text", "text": "hi"}]}}
    assert schema.is_message(msg) is True
    assert schema.role_of(msg) == "user"
    assert schema.text_of(msg) == "hi"

    other = {"type": "event_msg", "payload": {"type": "x"}}
    assert schema.is_message(other) is False
    assert schema.role_of(other) == "other"
    assert schema.text_of(other) == ""


def test_codex_schema_tool_role_and_timestamp():
    schema = codex.CODEX_SCHEMA
    fc = {"type": "response_item", "payload": {"type": "function_call", "name": "n", "arguments": "{}"}}
    assert schema.is_message(fc) is True
    assert schema.role_of(fc) == "assistant"
    assert schema.text_of(fc) == "<tool_use name='n'>{}</tool_use>"

    ts = {"type": "event_msg", "timestamp": "2024-01-02T03:04:05.999Z"}
    assert schema.timestamp_of(ts) == "2024-01-02 03:04:05"


# ---------------------------------------------------------------------------
# inspect_session_codex
# ---------------------------------------------------------------------------

def test_inspect_session_codex_runs(tmp_path, capsys):
    records = [
        _msg("user", _text_block("hello")),
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 2, "output_tokens": 3,
                                  "cached_input_tokens": 0, "total_tokens": 5}}}},
    ]
    path = _write_jsonl(tmp_path, records)
    inspect_session_codex(path, show_flow=True)
    out = capsys.readouterr().out
    assert "CODEX" in out


def test_inspect_session_codex_minimal(tmp_path, capsys):
    records = [_msg("user", _text_block("hi"))]
    path = _write_jsonl(tmp_path, records)
    inspect_session_codex(path)
    out = capsys.readouterr().out
    assert "CODEX" in out


def test_inspect_session_codex_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_session_codex(tmp_path / "missing.jsonl")
