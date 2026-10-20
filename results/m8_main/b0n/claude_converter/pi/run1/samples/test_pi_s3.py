import json
import sys
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

def write_jsonl(path: Path, records: list) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def msg(role, content, **extra):
    message = {"role": role, "content": content}
    message.update(extra)
    return {"type": "message", "message": message}


# A tiny valid PNG (1x1 transparent), base64-encoded.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# load_session_pi
# ---------------------------------------------------------------------------

def test_load_session_pi_returns_records(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "hi"), {"type": "session"}])
    records = load_session_pi(path)
    assert records == [msg("user", "hi"), {"type": "session"}]


def test_load_session_pi_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_pi(tmp_path / "nope.jsonl")


def test_load_session_pi_bad_extension(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("{}")
    with pytest.raises(ValueError):
        load_session_pi(path)


def test_load_session_pi_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError):
        load_session_pi(path)


# ---------------------------------------------------------------------------
# records_to_messages_pi: text-only / structure
# ---------------------------------------------------------------------------

def test_records_empty():
    assert records_to_messages_pi([]) == []


def test_records_skips_non_message_records():
    records = [
        {"type": "session"},
        {"type": "model_change"},
        msg("user", "hello"),
    ]
    result = records_to_messages_pi(records)
    assert result == [{"role": "user", "content": "hello"}]


def test_records_string_content():
    records = [msg("user", "hi there")]
    assert records_to_messages_pi(records) == [{"role": "user", "content": "hi there"}]


def test_records_blank_string_content_dropped():
    records = [msg("user", "   ")]
    assert records_to_messages_pi(records) == []


def test_records_list_text_blocks():
    records = [msg("user", [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ])]
    assert records_to_messages_pi(records) == [
        {"role": "user", "content": "first\nsecond"}
    ]


def test_records_assistant_thinking_and_toolcall():
    records = [msg("assistant", [
        {"type": "thinking", "thinking": "pondering"},
        {"type": "toolCall", "name": "search", "arguments": {"q": "x"}},
    ])]
    result = records_to_messages_pi(records)
    assert result[0]["role"] == "assistant"
    content = result[0]["content"]
    assert "<thinking>pondering</thinking>" in content
    assert "<tool_use name='search'>" in content
    assert json.dumps({"q": "x"}, ensure_ascii=False) in content


def test_records_toolcall_default_arguments():
    records = [msg("assistant", [{"type": "toolCall", "name": "noop"}])]
    content = records_to_messages_pi(records)[0]["content"]
    assert content == "<tool_use name='noop'>{}</tool_use>"


def test_records_unrecognized_role_dropped():
    records = [msg("system", "you are a bot")]
    assert records_to_messages_pi(records) == []


def test_records_non_dict_content_blocks_ignored():
    records = [msg("user", [{"type": "text", "text": "keep"}, "not-a-dict", 42])]
    assert records_to_messages_pi(records) == [{"role": "user", "content": "keep"}]


def test_records_unknown_block_type_ignored():
    records = [msg("user", [{"type": "mystery", "x": 1}, {"type": "text", "text": "ok"}])]
    assert records_to_messages_pi(records) == [{"role": "user", "content": "ok"}]


# ---------------------------------------------------------------------------
# records_to_messages_pi: merging consecutive same-role
# ---------------------------------------------------------------------------

def test_records_merge_consecutive_user():
    records = [msg("user", "one"), msg("user", "two")]
    assert records_to_messages_pi(records) == [{"role": "user", "content": "one\ntwo"}]


def test_records_merge_consecutive_assistant():
    records = [msg("assistant", "a"), msg("assistant", "b")]
    assert records_to_messages_pi(records) == [{"role": "assistant", "content": "a\nb"}]


def test_records_alternating_roles_preserved():
    records = [msg("user", "q"), msg("assistant", "a"), msg("user", "q2")]
    assert records_to_messages_pi(records) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q2"},
    ]


# ---------------------------------------------------------------------------
# records_to_messages_pi: toolResult handling
# ---------------------------------------------------------------------------

def test_records_toolresult_text_only():
    records = [msg("toolResult", [
        {"type": "text", "text": "output line"},
    ], toolName="grep", toolCallId="call-1")]
    result = records_to_messages_pi(records)
    assert result == [
        {
            "role": "user",
            "content": "<tool_result name='grep' id='call-1'>output line</tool_result>",
        }
    ]


def test_records_toolresult_empty_text_dropped():
    records = [msg("toolResult", [{"type": "text", "text": ""}], toolName="t", toolCallId="c")]
    assert records_to_messages_pi(records) == []


def test_records_toolresult_joins_text_parts_skipping_blanks():
    records = [msg("toolResult", [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "  "},
        {"type": "text", "text": "b"},
    ], toolName="t", toolCallId="c")]
    result = records_to_messages_pi(records)
    assert result == [
        {"role": "user", "content": "<tool_result name='t' id='c'>a\nb</tool_result>"}
    ]


def test_records_toolresult_image_renders_placeholder_in_text_of():
    # _text_of puts "<image>" for image blocks; with no decoding it's just text
    records = [msg("toolResult", [
        {"type": "text", "text": "see"},
        {"type": "image", "data": ""},
    ], toolName="t", toolCallId="c")]
    # empty image data => has_image True, decoding fails => skip; text parts kept
    result = records_to_messages_pi(records)
    content = result[0]["content"]
    # no image decoded, so should collapse to a string
    assert isinstance(content, str)
    assert content.startswith("<tool_result name='t' id='c'>")
    assert "see" in content
    assert content.endswith("</tool_result>")


# ---------------------------------------------------------------------------
# records_to_messages_pi: image handling
# ---------------------------------------------------------------------------

def test_records_user_image_multimodal():
    records = [msg("user", [
        {"type": "text", "text": "describe"},
        {"type": "image", "data": PNG_B64, "mimeType": "image/png"},
    ])]
    result = records_to_messages_pi(records)
    assert len(result) == 1
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image"
    assert "image" in content[1]
    # a decoded PIL image object
    assert content[1]["image"].size == (1, 1)


def test_records_empty_image_data_warns_and_skips(tmp_path, capsys):
    records = [msg("user", [
        {"type": "text", "text": "hi"},
        {"type": "image", "data": ""},
    ])]
    result = records_to_messages_pi(records)
    # image skipped, only text remains -> collapses to string
    assert result == [{"role": "user", "content": "hi"}]
    err = capsys.readouterr().err
    assert "Skipping image block with empty base64 data" in err


def test_records_undecodable_image_warns_and_skips(capsys):
    records = [msg("user", [
        {"type": "text", "text": "hi"},
        {"type": "image", "data": "!!!not-base64!!!"},
    ])]
    result = records_to_messages_pi(records)
    assert result == [{"role": "user", "content": "hi"}]
    err = capsys.readouterr().err
    assert "Skipping unreadable image block" in err


def test_records_toolresult_with_image_multimodal():
    records = [msg("toolResult", [
        {"type": "text", "text": "look"},
        {"type": "image", "data": PNG_B64},
    ], toolName="shot", toolCallId="c9")]
    result = records_to_messages_pi(records)
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "<tool_result name='shot' id='c9'>"}
    assert content[1] == {"type": "text", "text": "look"}
    assert content[2]["type"] == "image"
    assert content[2]["image"].size == (1, 1)
    assert content[-1] == {"type": "text", "text": "</tool_result>"}


def test_records_toolresult_image_only_bad_data():
    records = [msg("toolResult", [
        {"type": "image", "data": ""},
    ], toolName="t", toolCallId="c")]
    result = records_to_messages_pi(records)
    # All images skipped and no text parts; but wrapper text parts remain
    content = result[0]["content"]
    assert content == [
        {"type": "text", "text": "<tool_result name='t' id='c'>"},
        {"type": "text", "text": "</tool_result>"},
    ]


# ---------------------------------------------------------------------------
# records_to_messages_pi: message null/non-dict content
# ---------------------------------------------------------------------------

def test_records_message_missing_content():
    records = [{"type": "message", "message": {"role": "user"}}]
    # content defaults to "", blank, dropped
    assert records_to_messages_pi(records) == []


def test_records_message_null_message():
    records = [{"type": "message", "message": None}]
    # _message_of uses dget -> {} so role "" -> dropped
    assert records_to_messages_pi(records) == []


def test_records_content_other_type():
    records = [{"type": "message", "message": {"role": "user", "content": 123}}]
    assert records_to_messages_pi(records) == []


# ---------------------------------------------------------------------------
# session_to_messages_pi
# ---------------------------------------------------------------------------

def test_session_to_messages_pi(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [
        {"type": "session"},
        msg("user", "hello"),
        msg("assistant", "hi"),
    ])
    result = session_to_messages_pi(path)
    assert result == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_session_to_messages_pi_writes_output(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "hello")])
    out = tmp_path / "sub" / "out.json"
    result = session_to_messages_pi(path, output=out)
    assert result == [{"role": "user", "content": "hello"}]
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == [{"role": "user", "content": "hello"}]


def test_session_to_messages_pi_output_with_image(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [
        msg("user", [
            {"type": "text", "text": "pic"},
            {"type": "image", "data": PNG_B64},
        ])
    ])
    out = tmp_path / "out.json"
    result = session_to_messages_pi(path, output=out)
    # in-memory keeps PIL image
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][1]["type"] == "image"
    assert result[0]["content"][1]["image"].size == (1, 1)
    # on disk re-encoded as data URI
    saved = json.loads(out.read_text(encoding="utf-8"))
    img_part = saved[0]["content"][1]
    assert img_part["type"] == "image"
    assert img_part["image"].startswith("data:image/")


# ---------------------------------------------------------------------------
# inspect_session_pi
# ---------------------------------------------------------------------------

def test_inspect_session_pi_runs(tmp_path, capsys):
    path = write_jsonl(tmp_path / "s.jsonl", [
        {"type": "session"},
        msg("user", "hello"),
        msg("assistant", "hi"),
    ])
    inspect_session_pi(path)
    captured = capsys.readouterr()
    assert "PI" in captured.out or "hello" in captured.out


def test_inspect_session_pi_show_flow(tmp_path, capsys):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "hello world")])
    inspect_session_pi(path, show_flow=True)
    captured = capsys.readouterr()
    assert "hello world" in captured.out


def test_inspect_session_pi_show_blocks(tmp_path, capsys):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "blocky")])
    inspect_session_pi(path, show_flow=True, show_blocks=True)
    captured = capsys.readouterr()
    assert "blocky" in captured.out


# ---------------------------------------------------------------------------
# Token totals via PI_SCHEMA
# ---------------------------------------------------------------------------

def test_total_tokens_via_schema():
    from claude_converter.pi import PI_SCHEMA

    records = [
        {"type": "message", "message": {"role": "assistant", "content": "x",
                                        "usage": {"input": 10, "output": 5,
                                                  "cacheRead": 2, "cacheWrite": 1}}},
        {"type": "message", "message": {"role": "assistant", "content": "y",
                                        "usage": {"input": 3, "output": 4}}},
        {"type": "session"},
    ]
    totals = PI_SCHEMA.total_tokens_of(records)
    assert totals == {"input": 13, "output": 9, "cache_read": 2, "cache_write": 1}


def test_total_tokens_no_usage():
    from claude_converter.pi import PI_SCHEMA

    records = [msg("user", "hi")]
    assert PI_SCHEMA.total_tokens_of(records) == {}


def test_record_type_of_and_role_of():
    from claude_converter.pi import PI_SCHEMA

    user_rec = msg("user", "hi")
    assert PI_SCHEMA.record_type_of(user_rec) == "message:user"
    assert PI_SCHEMA.is_message(user_rec) is True
    assert PI_SCHEMA.role_of(user_rec) == "user"

    session_rec = {"type": "session"}
    assert PI_SCHEMA.record_type_of(session_rec) == "session"
    assert PI_SCHEMA.is_message(session_rec) is False
    assert PI_SCHEMA.role_of(session_rec) == "other"

    tool_rec = msg("toolResult", [], toolName="t", toolCallId="c")
    assert PI_SCHEMA.role_of(tool_rec) == "user"


def test_timestamp_of():
    from claude_converter.pi import PI_SCHEMA

    rec = {"timestamp": "2024-01-02T03:04:05.678Z"}
    assert PI_SCHEMA.timestamp_of(rec) == "2024-01-02 03:04:05"


def test_text_of_assistant_toolcall():
    from claude_converter.pi import PI_SCHEMA

    rec = msg("assistant", [{"type": "toolCall", "name": "f", "arguments": {}}])
    assert PI_SCHEMA.text_of(rec) == "<tool_use name='f'>{}</tool_use>"
