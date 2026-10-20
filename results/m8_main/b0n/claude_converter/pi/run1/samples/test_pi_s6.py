"""Unit tests for claude_converter.pi."""

from __future__ import annotations

import base64
import io
import json

import pytest

from claude_converter.pi import (
    load_session_pi,
    records_to_messages_pi,
    session_to_messages_pi,
    inspect_session_pi,
    _block_to_text,
    _record_role,
    _text_of,
    _message_parts,
    _decode_pi_image,
    _record_type_of,
    _timestamp_of,
    _total_tokens_of,
    PI_SCHEMA,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _msg_record(role, content, **extra):
    message = {"role": role}
    if content is not None:
        message["content"] = content
    message.update(extra)
    return {"type": "message", "message": message}


def _png_bytes():
    """Smallest valid-ish PNG payload: a 1x1 image."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAAA"
        "AAAJAAH4c3nBAAAAAElFTkSuQmCC"
    )


def _png_b64():
    return base64.b64encode(_png_bytes()).decode("ascii")


def _image_block():
    return {"type": "image", "data": _png_b64(), "mimeType": "image/png"}


# --------------------------------------------------------------------------- #
# load_session_pi
# --------------------------------------------------------------------------- #

def test_load_session_pi_returns_records(tmp_path):
    path = tmp_path / "s.jsonl"
    records = [
        {"type": "session", "id": "s1"},
        {"type": "message", "message": {"role": "user", "content": "hi"}},
    ]
    _write_jsonl(path, records)
    assert load_session_pi(path) == records


def test_load_session_pi_accepts_str_path(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [{"type": "session"}])
    assert load_session_pi(str(path)) == [{"type": "session"}]


def test_load_session_pi_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_pi(tmp_path / "nope.jsonl")


def test_load_session_pi_bad_extension(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_session_pi(path)


def test_load_session_pi_empty_file(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_session_pi(path)


# --------------------------------------------------------------------------- #
# _block_to_text
# --------------------------------------------------------------------------- #

def test_block_to_text_text():
    assert _block_to_text({"type": "text", "text": "hello"}) == "hello"


def test_block_to_text_text_missing():
    assert _block_to_text({"type": "text"}) == ""


def test_block_to_text_thinking():
    assert _block_to_text({"type": "thinking", "thinking": "hmm"}) == "<thinking>hmm</thinking>"


def test_block_to_text_thinking_missing():
    assert _block_to_text({"type": "thinking"}) == "<thinking></thinking>"


def test_block_to_text_tool_call():
    out = _block_to_text({"type": "toolCall", "name": "Bash", "arguments": {"cmd": "ls"}})
    assert out == "<tool_use name='Bash'>{\"cmd\": \"ls\"}</tool_use>"


def test_block_to_text_tool_call_no_args():
    out = _block_to_text({"type": "toolCall", "name": "X"})
    assert out == "<tool_use name='X'>{}</tool_use>"


def test_block_to_text_image():
    assert _block_to_text({"type": "image"}) == "<image>"


def test_block_to_text_unknown():
    assert _block_to_text({"type": "whatever"}) == ""


def test_block_to_text_no_type():
    assert _block_to_text({}) == ""


# --------------------------------------------------------------------------- #
# _record_role / _text_of
# --------------------------------------------------------------------------- #

def test_record_role_non_message():
    assert _record_role({"type": "session"}) is None


def test_record_role_user():
    assert _record_role(_msg_record("user", "hi")) == "user"


def test_record_role_assistant():
    assert _record_role(_msg_record("assistant", "yo")) == "assistant"


def test_record_role_tool_result_maps_to_user():
    assert _record_role(_msg_record("toolResult", [])) == "user"


def test_record_role_unknown_role():
    assert _record_role(_msg_record("system", "x")) is None


def test_record_role_null_message():
    assert _record_role({"type": "message", "message": None}) is None


def test_record_role_missing_message():
    assert _record_role({"type": "message"}) is None


def test_text_of_string_content():
    assert _text_of(_msg_record("user", "hello")) == "hello"


def test_text_of_list_content():
    rec = _msg_record(
        "assistant",
        [
            {"type": "text", "text": "a"},
            {"type": "thinking", "thinking": "b"},
        ],
    )
    assert _text_of(rec) == "a\n<thinking>b</thinking>"


def test_text_of_list_skips_empty_parts():
    rec = _msg_record("user", [{"type": "text", "text": "  "}, {"type": "text", "text": "x"}])
    assert _text_of(rec) == "x"


def test_text_of_tool_result():
    rec = _msg_record(
        "toolResult",
        [
            {"type": "text", "text": "line1"},
            {"type": "image"},
            {"type": "text", "text": "line2"},
        ],
        toolName="Bash",
        toolCallId="c1",
    )
    expected = "<tool_result name='Bash' id='c1'>line1\n<image>\nline2</tool_result>"
    assert _text_of(rec) == expected


def test_text_of_tool_result_no_content():
    rec = _msg_record("toolResult", [], toolName="T", toolCallId="i")
    assert _text_of(rec) == "<tool_result name='T' id='i'></tool_result>"


def test_text_of_tool_result_skips_non_dict():
    rec = _msg_record("toolResult", ["raw", {"type": "text", "text": "ok"}], toolName="T", toolCallId="i")
    assert _text_of(rec) == "<tool_result name='T' id='i'>ok</tool_result>"


def test_text_of_non_message_record():
    assert _text_of({"type": "session"}) == ""


def test_text_of_unexpected_content_type():
    assert _text_of(_msg_record("user", 123)) == ""


# --------------------------------------------------------------------------- #
# _decode_pi_image
# --------------------------------------------------------------------------- #

def test_decode_pi_image_empty_warns(capsys):
    assert _decode_pi_image({"type": "image", "data": ""}) is None
    err = capsys.readouterr().err
    assert "empty base64 data" in err


def test_decode_pi_image_missing_data_warns(capsys):
    assert _decode_pi_image({"type": "image"}) is None
    assert "empty base64 data" in capsys.readouterr().err


def test_decode_pi_image_invalid_warns(capsys):
    assert _decode_pi_image({"type": "image", "data": "not-base64!!!"}) is None
    err = capsys.readouterr().err
    assert "Skipping unreadable image block" in err


def test_decode_pi_image_valid():
    pytest.importorskip("PIL")
    img = _decode_pi_image(_image_block())
    assert img is not None
    assert img.size == (1, 1)


def test_decode_pi_image_importerror(monkeypatch, capsys):
    import claude_converter.pi as pi_mod

    def boom(data):
        raise ImportError("no pillow")

    monkeypatch.setattr(pi_mod, "convert_base64_to_pil_image", boom)
    assert _decode_pi_image({"type": "image", "data": "abc"}) is None
    assert "Skipping unreadable image block" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# _message_parts
# --------------------------------------------------------------------------- #

def test_message_parts_plain_string():
    assert _message_parts(_msg_record("user", "hello")) == [{"type": "text", "text": "hello"}]


def test_message_parts_blank_string():
    assert _message_parts(_msg_record("user", "   ")) == []


def test_message_parts_empty_string():
    assert _message_parts(_msg_record("user", "")) == []


def test_message_parts_list_text_blocks():
    rec = _msg_record("assistant", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert _message_parts(rec) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_message_parts_skips_blank_and_non_dict():
    rec = _msg_record("user", ["raw", {"type": "text", "text": "x"}, {"type": "text", "text": " "}])
    assert _message_parts(rec) == [{"type": "text", "text": "x"}]


def test_message_parts_tool_result_text_only():
    rec = _msg_record(
        "toolResult",
        [{"type": "text", "text": "out"}],
        toolName="Bash",
        toolCallId="c1",
    )
    assert _message_parts(rec) == [
        {"type": "text", "text": "<tool_result name='Bash' id='c1'>out</tool_result>"}
    ]


def test_message_parts_tool_result_text_only_empty():
    rec = _msg_record("toolResult", [], toolName="T", toolCallId="i")
    # text produced is "" -> no parts
    assert _message_parts(rec) == []


def test_message_parts_tool_result_with_image(monkeypatch):
    import claude_converter.pi as pi_mod

    sentinel = object()
    monkeypatch.setattr(pi_mod, "convert_base64_to_pil_image", lambda d: sentinel)

    rec = _msg_record(
        "toolResult",
        [
            {"type": "text", "text": "before"},
            _image_block(),
            {"type": "text", "text": "  "},
            {"type": "text", "text": "after"},
        ],
        toolName="T",
        toolCallId="i",
    )
    parts = _message_parts(rec)
    assert parts == [
        {"type": "text", "text": "<tool_result name='T' id='i'>"},
        {"type": "text", "text": "before"},
        {"type": "image", "image": sentinel},
        {"type": "text", "text": "after"},
        {"type": "text", "text": "</tool_result>"},
    ]


def test_message_parts_tool_result_with_undecodable_image(monkeypatch, capsys):
    import claude_converter.pi as pi_mod

    monkeypatch.setattr(
        pi_mod, "convert_base64_to_pil_image", lambda d: (_ for _ in ()).throw(ValueError("bad"))
    )
    rec = _msg_record(
        "toolResult",
        [{"type": "text", "text": "x"}, _image_block()],
        toolName="T",
        toolCallId="i",
    )
    parts = _message_parts(rec)
    assert parts == [
        {"type": "text", "text": "<tool_result name='T' id='i'>"},
        {"type": "text", "text": "x"},
        {"type": "text", "text": "</tool_result>"},
    ]


def test_message_parts_plain_with_image(monkeypatch):
    import claude_converter.pi as pi_mod

    sentinel = object()
    monkeypatch.setattr(pi_mod, "convert_base64_to_pil_image", lambda d: sentinel)
    rec = _msg_record("user", [{"type": "text", "text": "hi"}, _image_block()])
    parts = _message_parts(rec)
    assert parts == [
        {"type": "text", "text": "hi"},
        {"type": "image", "image": sentinel},
    ]


def test_message_parts_unexpected_content_type():
    assert _message_parts(_msg_record("user", 42)) == []


def test_message_parts_non_message():
    assert _message_parts({"type": "session"}) == []


# --------------------------------------------------------------------------- #
# records_to_messages_pi
# --------------------------------------------------------------------------- #

def test_records_to_messages_basic():
    records = [
        {"type": "session"},
        _msg_record("user", "hello"),
        _msg_record("assistant", "world"),
    ]
    assert records_to_messages_pi(records) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


def test_records_to_messages_merges_consecutive_same_role():
    records = [
        _msg_record("user", "a"),
        _msg_record("user", "b"),
    ]
    assert records_to_messages_pi(records) == [{"role": "user", "content": "a\nb"}]


def test_records_to_messages_skips_empty_parts():
    records = [
        _msg_record("user", "   "),
        _msg_record("assistant", "reply"),
    ]
    assert records_to_messages_pi(records) == [{"role": "assistant", "content": "reply"}]


def test_records_to_messages_skips_non_message():
    records = [{"type": "session"}, {"type": "model_change"}]
    assert records_to_messages_pi(records) == []


def test_records_to_messages_tool_result_then_assistant():
    records = [
        _msg_record("user", "run"),
        _msg_record("assistant", [{"type": "toolCall", "name": "Bash", "arguments": {"cmd": "ls"}}]),
        _msg_record("toolResult", [{"type": "text", "text": "out"}], toolName="Bash", toolCallId="c1"),
    ]
    result = records_to_messages_pi(records)
    assert result == [
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": "<tool_use name='Bash'>{\"cmd\": \"ls\"}</tool_use>"},
        {"role": "user", "content": "<tool_result name='Bash' id='c1'>out</tool_result>"},
    ]


def test_records_to_messages_empty():
    assert records_to_messages_pi([]) == []


def test_records_to_messages_multimodal(monkeypatch):
    import claude_converter.pi as pi_mod

    sentinel = object()
    monkeypatch.setattr(pi_mod, "convert_base64_to_pil_image", lambda d: sentinel)
    records = [_msg_record("user", [{"type": "text", "text": "look"}, _image_block()])]
    result = records_to_messages_pi(records)
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "image": sentinel},
            ],
        }
    ]


# --------------------------------------------------------------------------- #
# session_to_messages_pi
# --------------------------------------------------------------------------- #

def test_session_to_messages_no_output(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [_msg_record("user", "hi"), _msg_record("assistant", "yo")])
    result = session_to_messages_pi(path)
    assert result == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_session_to_messages_with_output(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [_msg_record("user", "hi")])
    out = tmp_path / "nested" / "out.json"
    result = session_to_messages_pi(path, output=out)
    assert result == [{"role": "user", "content": "hi"}]
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == [{"role": "user", "content": "hi"}]


def test_session_to_messages_output_multimodal(tmp_path):
    pytest.importorskip("PIL")
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [_msg_record("user", [{"type": "text", "text": "x"}, _image_block()])])
    out = tmp_path / "out.json"
    result = session_to_messages_pi(path, output=out)
    # in-memory result holds a PIL image
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][1]["type"] == "image"
    assert not isinstance(result[0]["content"][1]["image"], str)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved[0]["content"][1]["type"] == "image"
    assert saved[0]["content"][1]["image"].startswith("data:image/")


def test_session_to_messages_output_overwrites_dir(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [_msg_record("user", "hello")])
    out = tmp_path / "a" / "b" / "c.json"
    session_to_messages_pi(path, output=out)
    assert out.exists()


# --------------------------------------------------------------------------- #
# schema helpers
# --------------------------------------------------------------------------- #

def test_record_type_of_message():
    assert _record_type_of(_msg_record("user", "x")) == "message:user"


def test_record_type_of_message_missing_role():
    rec = {"type": "message", "message": {}}
    assert _record_type_of(rec) == "message:unknown"


def test_record_type_of_non_message():
    assert _record_type_of({"type": "session"}) == "session"


def test_record_type_of_missing_type():
    assert _record_type_of({}) == "unknown"


def test_timestamp_of_truncates_and_replaces():
    assert _timestamp_of({"timestamp": "2024-01-02T03:04:05.123Z"}) == "2024-01-02 03:04:05"


def test_timestamp_of_missing():
    assert _timestamp_of({}) == ""


def test_total_tokens_of_empty_no_usage():
    assert _total_tokens_of([]) == {}
    assert _total_tokens_of([{"type": "session"}]) == {}
    assert _total_tokens_of([{"type": "message", "message": {"role": "user"}}]) == {}


def test_total_tokens_of_sums():
    records = [
        _msg_record("user", "x", usage={"input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 1}),
        _msg_record("assistant", "y", usage={"input": 3, "output": 7}),
        {"type": "session"},
    ]
    assert _total_tokens_of(records) == {
        "input": 13,
        "output": 12,
        "cache_read": 2,
        "cache_write": 1,
    }


def test_pi_schema_fields():
    assert PI_SCHEMA.label == "PI"
    rec = _msg_record("user", "hello")
    assert PI_SCHEMA.is_message(rec) is True
    assert PI_SCHEMA.role_of(rec) == "user"
    assert PI_SCHEMA.text_of(rec) == "hello"
    assert PI_SCHEMA.is_message({"type": "session"}) is False
    assert PI_SCHEMA.role_of({"type": "session"}) == "other"
    assert PI_SCHEMA.timestamp_of({"timestamp": "2024-01-01T00:00:00Z"}) == "2024-01-01 00:00:00"
    assert PI_SCHEMA.total_tokens_of([]) == {}


# --------------------------------------------------------------------------- #
# inspect_session_pi
# --------------------------------------------------------------------------- #

def test_inspect_session_pi_prints(tmp_path, capsys):
    path = tmp_path / "s.jsonl"
    _write_jsonl(
        path,
        [
            {"type": "session", "id": "s1"},
            _msg_record("user", "hello", usage={"input": 5, "output": 0}),
            _msg_record("assistant", "world", usage={"input": 1, "output": 2}),
        ],
    )
    inspect_session_pi(path)
    out = capsys.readouterr().out
    assert "PI" in out
    assert "user" in out
    assert "assistant" in out


def test_inspect_session_pi_show_flow(tmp_path, capsys):
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [_msg_record("user", "hello there")])
    inspect_session_pi(path, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "hello there" in out
