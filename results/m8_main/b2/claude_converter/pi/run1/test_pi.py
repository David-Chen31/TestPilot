from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from claude_converter import pi
from claude_converter.pi import (
    inspect_session_pi,
    load_session_pi,
    records_to_messages_pi,
    session_to_messages_pi,
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
    return {"type": "message", "id": "x", "parentId": None, "message": message}


# ---------------------------------------------------------------------------
# load_session_pi
# ---------------------------------------------------------------------------


def test_load_session_pi_returns_records(tmp_path):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [{"type": "session", "id": "abc"}, msg("user", "hello")],
    )
    records = load_session_pi(path)
    assert records == [{"type": "session", "id": "abc"}, msg("user", "hello")]


def test_load_session_pi_accepts_str_path(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [{"type": "session"}])
    assert load_session_pi(str(path)) == [{"type": "session"}]


def test_load_session_pi_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session_pi(tmp_path / "nope.jsonl")


def test_load_session_pi_bad_extension(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_pi(p)


def test_load_session_pi_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_session_pi(p)


def test_load_session_pi_skips_invalid_lines(tmp_path, capsys):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"type": "session"}\nnot json\n{"type": "model_change"}\n', encoding="utf-8")
    records = load_session_pi(p)
    assert records == [{"type": "session"}, {"type": "model_change"}]
    err = capsys.readouterr().err
    assert "invalid JSON" in err


# ---------------------------------------------------------------------------
# records_to_messages_pi - basic text handling
# ---------------------------------------------------------------------------


def test_simple_user_assistant_string_content():
    records = [msg("user", "hi"), msg("assistant", "hello there")]
    out = records_to_messages_pi(records)
    assert out == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]


def test_skips_non_message_and_unknown_roles():
    records = [
        {"type": "session", "id": "a"},
        {"type": "model_change", "model": "m"},
        {"type": "message", "message": {"role": "system", "content": "x"}},
        msg("user", "real"),
    ]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "real"}]


def test_empty_content_is_skipped():
    records = [msg("user", ""), msg("assistant", "   "), msg("user", "ok")]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "ok"}]


def test_empty_records_returns_empty_list():
    assert records_to_messages_pi([]) == []


def test_message_with_missing_message_key():
    records = [{"type": "message"}, msg("user", "after")]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "after"}]


def test_message_with_null_message_key():
    records = [{"type": "message", "message": None}, msg("user", "after")]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "after"}]


# ---------------------------------------------------------------------------
# _block_to_text default-argument coverage
# ---------------------------------------------------------------------------


def test_block_to_text_missing_type_defaults_to_empty():
    assert pi._block_to_text({}) == ""
    assert pi._block_to_text({"text": "ignored"}) == ""


def test_block_to_text_missing_text_returns_empty_string():
    result = pi._block_to_text({"type": "text"})
    assert result == ""
    assert isinstance(result, str)


def test_block_to_text_missing_thinking_returns_empty_string_wrapped():
    result = pi._block_to_text({"type": "thinking"})
    assert result == "<thinking></thinking>"


def test_block_to_text_text_block_null_text_returns_none_default():
    # text: None -> .get("text", "") returns None when key present with null,
    # so the block yields None (not ""). This distinguishes the "" default
    # from a None default only if the key is absent... check absent case:
    assert pi._block_to_text({"type": "text", "text": None}) is None


def test_block_to_text_tool_call_missing_name_defaults_empty():
    result = pi._block_to_text({"type": "toolCall", "arguments": {}})
    assert result == "<tool_use name=''>{}</tool_use>"


def test_block_to_text_tool_call_missing_arguments_defaults_empty_dict():
    result = pi._block_to_text({"type": "toolCall", "name": "n"})
    assert result == "<tool_use name='n'>{}</tool_use>"


def test_tool_call_arguments_json_preserves_unicode():
    records = [msg("assistant", [{"type": "toolCall", "name": "n", "arguments": {"k": "é"}}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "assistant", "content": "<tool_use name='n'>{\"k\": \"é\"}</tool_use>"}]
    assert "\\u00e9" not in out[0]["content"]


# ---------------------------------------------------------------------------
# Content list blocks: text / thinking / toolCall / image markers
# ---------------------------------------------------------------------------


def test_text_blocks_joined_with_newline():
    records = [msg("user", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "a\nb"}]


def test_blank_text_blocks_filtered():
    records = [
        msg("user", [{"type": "text", "text": "a"}, {"type": "text", "text": "  "},
                     {"type": "text", "text": ""}])
    ]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "a"}]


def test_thinking_block_wrapped():
    records = [msg("assistant", [{"type": "thinking", "thinking": "hmm"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "assistant", "content": "<thinking>hmm</thinking>"}]


def test_tool_call_block_rendered_as_tag():
    records = [
        msg("assistant", [{"type": "toolCall", "name": "read", "arguments": {"path": "x"}}])
    ]
    out = records_to_messages_pi(records)
    expected_args = json.dumps({"path": "x"}, ensure_ascii=False)
    assert out == [
        {"role": "assistant", "content": f"<tool_use name='read'>{expected_args}</tool_use>"}
    ]


def test_tool_call_default_arguments():
    records = [msg("assistant", [{"type": "toolCall", "name": "noop"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "assistant", "content": "<tool_use name='noop'>{}</tool_use>"}]


def test_tool_call_missing_name_defaults_empty():
    records = [msg("assistant", [{"type": "toolCall", "arguments": {}}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "assistant", "content": "<tool_use name=''>{}</tool_use>"}]


def test_unknown_block_type_ignored():
    records = [msg("user", [{"type": "weird", "text": "x"}, {"type": "text", "text": "y"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "y"}]


def test_non_dict_blocks_ignored():
    records = [msg("user", ["raw string", {"type": "text", "text": "y"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "y"}]


def test_content_neither_str_nor_list_returns_empty():
    records = [msg("user", 123), msg("assistant", "after")]
    out = records_to_messages_pi(records)
    assert out == [{"role": "assistant", "content": "after"}]


# ---------------------------------------------------------------------------
# Merging consecutive same-role turns
# ---------------------------------------------------------------------------


def test_consecutive_same_role_merged_as_text():
    records = [msg("user", "one"), msg("user", "two")]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "one\ntwo"}]


def test_alternating_roles_not_merged():
    records = [msg("user", "u"), msg("assistant", "a"), msg("user", "u2")]
    out = records_to_messages_pi(records)
    assert out == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u2"},
    ]


def test_skipped_empty_turn_between_same_role_allows_merge():
    records = [msg("user", "one"), msg("assistant", ""), msg("user", "two")]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "one\ntwo"}]


# ---------------------------------------------------------------------------
# toolResult handling (~user role)
# ---------------------------------------------------------------------------


def test_tool_result_becomes_user_with_tag():
    records = [
        msg(
            "toolResult",
            [{"type": "text", "text": "output"}],
            toolName="read",
            toolCallId="call-1",
        )
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {
            "role": "user",
            "content": "<tool_result name='read' id='call-1'>output</tool_result>",
        }
    ]


def test_tool_result_missing_metadata_defaults_empty():
    records = [msg("toolResult", [{"type": "text", "text": "out"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "<tool_result name='' id=''>out</tool_result>"}]


def test_tool_result_without_text_is_skipped():
    records = [
        msg("toolResult", [], toolName="read", toolCallId="c"),
        msg("user", "hi"),
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {
            "role": "user",
            "content": "<tool_result name='read' id='c'></tool_result>\nhi",
        }
    ]


def test_tool_result_merges_with_consecutive_user():
    records = [
        msg("user", "question"),
        msg("toolResult", [{"type": "text", "text": "res"}], toolName="t", toolCallId="1"),
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {
            "role": "user",
            "content": "question\n<tool_result name='t' id='1'>res</tool_result>",
        }
    ]


def test_tool_result_text_blocks_joined_and_blank_filtered(monkeypatch, capsys):
    called = {"n": 0}

    def fake_convert(data):
        called["n"] += 1
        return FakeImage("x")

    monkeypatch.setattr(pi, "convert_base64_to_pil_image", fake_convert)

    records = [
        msg(
            "toolResult",
            [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "   "},
                {"type": "text", "text": "b"},
                {"type": "image"},
            ],
            toolName="t",
            toolCallId="1",
        )
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {
            "role": "user",
            "content": "<tool_result name='t' id='1'>\na\nb\n</tool_result>",
        }
    ]
    assert called["n"] == 0
    assert "empty base64" in capsys.readouterr().err


def test_tool_result_non_dict_blocks_ignored_in_text_of():
    records = [
        msg(
            "toolResult",
            ["raw", {"type": "text", "text": "x"}],
            toolName="t",
            toolCallId="1",
        )
    ]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "<tool_result name='t' id='1'>x</tool_result>"}]


def test_tool_result_text_missing_text_key_defaults_empty():
    # text block with no "text" key -> b.get("text", "") -> ""
    records = [
        msg(
            "toolResult",
            [{"type": "text"}, {"type": "text", "text": "x"}],
            toolName="t",
            toolCallId="1",
        )
    ]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "<tool_result name='t' id='1'>x</tool_result>"}]


def test_tool_result_image_marker_in_text_of():
    # The _text_of path (no images triggered → only reached when there is no
    # image in content...). Use _text_of directly to exercise the "<image>"
    # marker branch on line 96-97.
    record = msg(
        "toolResult",
        [{"type": "text", "text": "before"}, {"type": "image"}, {"type": "text", "text": "after"}],
        toolName="t",
        toolCallId="1",
    )
    assert pi._text_of(record) == "<tool_result name='t' id='1'>before\n<image>\nafter</tool_result>"


# ---------------------------------------------------------------------------
# Image handling via _decode_pi_image (monkeypatched)
# ---------------------------------------------------------------------------


class FakeImage:
    def __init__(self, tag):
        self.tag = tag

    def __eq__(self, other):
        return isinstance(other, FakeImage) and other.tag == self.tag

    def __repr__(self):
        return f"FakeImage({self.tag!r})"


def test_user_message_image_block_kept_as_part(monkeypatch):
    sentinel = FakeImage("img")
    monkeypatch.setattr(pi, "convert_base64_to_pil_image", lambda data: sentinel)

    records = [
        msg(
            "user",
            [
                {"type": "text", "text": "look"},
                {"type": "image", "data": "abc", "mimeType": "image/png"},
            ],
        )
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "image": sentinel},
            ],
        }
    ]


def test_image_only_message_produces_list_content(monkeypatch):
    sentinel = FakeImage("img")
    monkeypatch.setattr(pi, "convert_base64_to_pil_image", lambda data: sentinel)
    records = [msg("user", [{"type": "image", "data": "abc"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": [{"type": "image", "image": sentinel}]}]


def test_decode_pi_image_passes_data_to_converter(monkeypatch):
    received = {}

    def fake_convert(data):
        received["data"] = data
        return FakeImage("x")

    monkeypatch.setattr(pi, "convert_base64_to_pil_image", fake_convert)
    result = pi._decode_pi_image({"type": "image", "data": "payload123"})
    assert result == FakeImage("x")
    assert received["data"] == "payload123"


def test_decode_pi_image_empty_data_returns_none(monkeypatch, capsys):
    monkeypatch.setattr(pi, "convert_base64_to_pil_image", lambda data: FakeImage("x"))
    assert pi._decode_pi_image({"type": "image"}) is None
    assert pi._decode_pi_image({"type": "image", "data": ""}) is None
    err = capsys.readouterr().err
    # Exact message text (kills case/char mutants of the warning string).
    assert "X Skipping image block with empty base64 data" in err


def test_decode_pi_image_exact_warning_text(monkeypatch, capsys):
    monkeypatch.setattr(pi, "convert_base64_to_pil_image", lambda data: FakeImage("x"))
    assert pi._decode_pi_image({"type": "image", "data": ""}) is None
    err = capsys.readouterr().err.strip()
    assert err == "X Skipping image block with empty base64 data"


def test_decode_pi_image_unreadable_exact_warning(monkeypatch, capsys):
    def fake_convert(data):
        raise ValueError("boom-detail")

    monkeypatch.setattr(pi, "convert_base64_to_pil_image", fake_convert)
    assert pi._decode_pi_image({"type": "image", "data": "x"}) is None
    err = capsys.readouterr().err.strip()
    assert err == "X Skipping unreadable image block: boom-detail"


def test_decode_pi_image_import_error_exact_warning(monkeypatch, capsys):
    def fake_convert(data):
        raise ImportError("no pillow-detail")

    monkeypatch.setattr(pi, "convert_base64_to_pil_image", fake_convert)
    assert pi._decode_pi_image({"type": "image", "data": "x"}) is None
    err = capsys.readouterr().err.strip()
    assert err == "X Skipping unreadable image block: no pillow-detail"


def test_empty_base64_image_skipped_with_warning(monkeypatch, capsys):
    called = {"n": 0}

    def fake_convert(data):
        called["n"] += 1
        return FakeImage("x")

    monkeypatch.setattr(pi, "convert_base64_to_pil_image", fake_convert)
    records = [msg("user", [{"type": "image"}, {"type": "text", "text": "hi"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "hi"}]
    assert called["n"] == 0
    assert "empty base64" in capsys.readouterr().err


def test_undecodable_image_skipped_with_warning(monkeypatch, capsys):
    def fake_convert(data):
        raise ValueError("bad")

    monkeypatch.setattr(pi, "convert_base64_to_pil_image", fake_convert)
    records = [msg("user", [{"type": "image", "data": "zzz"}, {"type": "text", "text": "hi"}])]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "hi"}]
    assert "unreadable image" in capsys.readouterr().err


def test_import_error_image_skipped(monkeypatch, capsys):
    def fake_convert(data):
        raise ImportError("no pillow")

    monkeypatch.setattr(pi, "convert_base64_to_pil_image", fake_convert)
    records = [msg("user", [{"type": "image", "data": "zzz"}]), msg("user", "hi")]
    out = records_to_messages_pi(records)
    assert out == [{"role": "user", "content": "hi"}]
    assert "unreadable image" in capsys.readouterr().err


def test_tool_result_with_image_keeps_multimodal_parts(monkeypatch):
    sentinel = FakeImage("img")
    monkeypatch.setattr(pi, "convert_base64_to_pil_image", lambda data: sentinel)
    records = [
        msg(
            "toolResult",
            [
                {"type": "text", "text": "pic"},
                {"type": "image", "data": "abc"},
                {"type": "text", "text": "   "},
            ],
            toolName="shot",
            toolCallId="c1",
        )
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<tool_result name='shot' id='c1'>"},
                {"type": "text", "text": "pic"},
                {"type": "image", "image": sentinel},
                {"type": "text", "text": "</tool_result>"},
            ],
        }
    ]


def test_tool_result_with_image_uses_empty_metadata_defaults(monkeypatch):
    sentinel = FakeImage("img")
    monkeypatch.setattr(pi, "convert_base64_to_pil_image", lambda data: sentinel)
    records = [msg("toolResult", [{"type": "image", "data": "abc"}])]
    out = records_to_messages_pi(records)
    assert out == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<tool_result name='' id=''>"},
                {"type": "image", "image": sentinel},
                {"type": "text", "text": "</tool_result>"},
            ],
        }
    ]


def test_tool_result_image_ignores_non_dict_content(monkeypatch):
    sentinel = FakeImage("img")
    monkeypatch.setattr(pi, "convert_base64_to_pil_image", lambda data: sentinel)
    records = [
        msg(
            "toolResult",
            ["raw", {"type": "image", "data": "abc"}],
            toolName="t",
            toolCallId="1",
        )
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<tool_result name='t' id='1'>"},
                {"type": "image", "image": sentinel},
                {"type": "text", "text": "</tool_result>"},
            ],
        }
    ]


def test_tool_result_image_decoding_failure_still_wraps(monkeypatch, capsys):
    def fake_convert(data):
        raise ValueError("bad")

    monkeypatch.setattr(pi, "convert_base64_to_pil_image", fake_convert)
    records = [
        msg(
            "toolResult",
            [{"type": "image", "data": "zzz"}, {"type": "text", "text": "ok"}],
            toolName="t",
            toolCallId="1",
        )
    ]
    out = records_to_messages_pi(records)
    assert out == [
        {
            "role": "user",
            "content": "<tool_result name='t' id='1'>\nok\n</tool_result>",
        }
    ]
    assert "unreadable image" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# session_to_messages_pi
# ---------------------------------------------------------------------------


def test_session_to_messages_pi_returns_messages(tmp_path):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [{"type": "session"}, msg("user", "hi"), msg("assistant", "yo")],
    )
    out = session_to_messages_pi(path)
    assert out == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_session_to_messages_pi_writes_output(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "hi")])
    out_path = tmp_path / "out" / "result.json"
    msgs = session_to_messages_pi(path, output=out_path)
    assert msgs == [{"role": "user", "content": "hi"}]
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved == [{"role": "user", "content": "hi"}]


def test_session_to_messages_pi_output_creates_parent_dirs(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "hi")])
    out_path = tmp_path / "a" / "b" / "c" / "result.json"
    assert not (tmp_path / "a").exists()
    session_to_messages_pi(path, output=out_path)
    assert out_path.exists()


def test_session_to_messages_pi_output_without_path(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "hi")])
    before = set(tmp_path.iterdir())
    session_to_messages_pi(path, output=None)
    after = set(tmp_path.iterdir())
    assert before == after


def test_session_to_messages_pi_output_serializes_images(tmp_path, monkeypatch):
    sentinel = FakeImage("img")
    monkeypatch.setattr(pi, "convert_base64_to_pil_image", lambda data: sentinel)

    path = write_jsonl(
        tmp_path / "s.jsonl",
        [msg("user", [{"type": "text", "text": "hi"}])],
    )
    out_path = tmp_path / "out.json"
    msgs = session_to_messages_pi(path, output=out_path)
    assert msgs == [{"role": "user", "content": "hi"}]
    assert json.loads(out_path.read_text(encoding="utf-8")) == msgs


def test_session_to_messages_pi_output_pretty_printed(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "héllo")])
    out_path = tmp_path / "out.json"
    session_to_messages_pi(path, output=out_path)
    raw = out_path.read_text(encoding="utf-8")
    # ensure_ascii=False: the unicode char is written literally.
    assert "héllo" in raw
    assert "\\u00e9" not in raw


def test_session_to_messages_pi_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        session_to_messages_pi(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# inspect_session_pi
# ---------------------------------------------------------------------------


def test_inspect_session_pi_runs_and_prints(tmp_path, capsys):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            {"type": "session", "id": "s1"},
            msg("user", "hello"),
            msg("assistant", [{"type": "text", "text": "world"}]),
        ],
    )
    inspect_session_pi(path)
    out = capsys.readouterr().out
    assert "PI" in out
    assert "hello" not in out


def test_inspect_session_pi_show_flow(tmp_path, capsys):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "hello")])
    inspect_session_pi(path, show_flow=True)
    out = capsys.readouterr().out
    assert "PI" in out


def test_inspect_session_pi_show_flow_with_text_flag(tmp_path, capsys):
    path = write_jsonl(tmp_path / "s.jsonl", [msg("user", "hello")])
    inspect_session_pi(path, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "PI" in out


def test_inspect_session_pi_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_session_pi(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Internal helpers / schema exposure sanity
# ---------------------------------------------------------------------------


def test_pi_schema_attributes():
    assert pi.PI_SCHEMA.label == "PI"
    assert pi.PI_SCHEMA.is_message(msg("user", "hi")) is True
    assert pi.PI_SCHEMA.is_message({"type": "session"}) is False
    assert pi.PI_SCHEMA.role_of(msg("assistant", "x")) == "assistant"
    assert pi.PI_SCHEMA.role_of({"type": "session"}) == "other"
    assert pi.PI_SCHEMA.role_of(msg("toolResult", "x")) == "user"


def test_record_type_of():
    assert pi._record_type_of({"type": "session"}) == "session"
    assert pi._record_type_of({"type": "message", "message": {"role": "user"}}) == "message:user"
    assert pi._record_type_of({}) == "unknown"


def test_record_type_of_message_default_role():
    assert pi._record_type_of({"type": "message", "message": {}}) == "message:unknown"


def test_record_type_of_message_null_message():
    assert pi._record_type_of({"type": "message", "message": None}) == "message:unknown"


def test_timestamp_of_truncates():
    assert pi._timestamp_of({"timestamp": "2024-01-02T03:04:05.678Z"}) == "2024-01-02 03:04:05"
    assert pi._timestamp_of({}) == ""


def test_timestamp_of_non_string_value_stringified():
    assert pi._timestamp_of({"timestamp": 12345}) == "12345"


def test_total_tokens_of():
    records = [
        {"type": "message", "message": {"role": "user", "usage": {
            "input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 1}}},
        {"type": "message", "message": {"role": "assistant", "usage": {
            "input": 3, "output": 4}}},
        {"type": "session"},
    ]
    assert pi._total_tokens_of(records) == {
        "input": 13, "output": 9, "cache_read": 2, "cache_write": 1,
    }


def test_total_tokens_of_only_cached_keys():
    records = [
        {"type": "message", "message": {"role": "assistant", "usage": {
            "cacheRead": 7, "cacheWrite": 8}}},
    ]
    assert pi._total_tokens_of(records) == {
        "input": 0, "output": 0, "cache_read": 7, "cache_write": 8,
    }


def test_total_tokens_of_no_usage():
    assert pi._total_tokens_of([{"type": "session"}, msg("user", "x")]) == {}


def test_total_tokens_of_null_message_ignored():
    records = [{"type": "message", "message": None}]
    assert pi._total_tokens_of(records) == {}


def test_message_of_helpers():
    assert pi._message_of({"type": "message", "message": {"role": "user"}}) == {"role": "user"}
    assert pi._message_of({"type": "session"}) is None
    assert pi._message_of({"type": "message", "message": None}) == {}


def test_record_role_unknown_role_returns_none():
    assert pi._record_role({"type": "message", "message": {"role": "system"}}) is None
    assert pi._record_role({"type": "message", "message": {"role": ""}}) is None


def test_record_role_missing_role_returns_none():
    # role absent -> default "" -> not in ("user","assistant") -> None
    assert pi._record_role({"type": "message", "message": {}}) is None


def test_record_role_tool_result_returns_user():
    assert pi._record_role(msg("toolResult", "x")) == "user"


def test_record_role_user_and_assistant():
    assert pi._record_role(msg("user", "x")) == "user"
    assert pi._record_role(msg("assistant", "x")) == "assistant"


def test_text_of_string_content():
    assert pi._text_of(msg("user", "abc")) == "abc"


def test_text_of_missing_message_returns_empty():
    assert pi._text_of({"type": "session"}) == ""


def test_text_of_non_str_non_list_returns_empty():
    assert pi._text_of(msg("user", 123)) == ""


def test_text_of_string_content_missing_role():
    # message has content but no role -> default role "" -> falls through to
    # content handling (string) path.
    record = {"type": "message", "message": {"content": "abc"}}
    assert pi._text_of(record) == "abc"


def test_text_of_tool_result_missing_role_uses_string_branch():
    # role missing => not "toolResult"; content handled normally.
    record = {"type": "message", "message": {"content": [{"type": "text", "text": "z"}]}}
    assert pi._text_of(record) == "z"


def test_text_of_tool_result_text_missing_key():
    # A text block without "text" key defaults to "" and is filtered.
    record = msg(
        "toolResult",
        [{"type": "text"}, {"type": "text", "text": "kept"}],
        toolName="t",
        toolCallId="1",
    )
    assert pi._text_of(record) == "<tool_result name='t' id='1'>kept</tool_result>"


def test_text_of_tool_result_no_matching_blocks():
    record = msg("toolResult", [], toolName="t", toolCallId="1")
    assert pi._text_of(record) == "<tool_result name='t' id='1'></tool_result>"


def test_block_to_text_dispatch():
    assert pi._block_to_text({"type": "text", "text": "t"}) == "t"
    assert pi._block_to_text({"type": "thinking", "thinking": "th"}) == "<thinking>th</thinking>"
    assert pi._block_to_text({"type": "image"}) == "<image>"
    assert pi._block_to_text({"type": "other"}) == ""


def test_block_to_text_wrong_type_key_ignored():
    # block with a different type key uses "" default -> no branch matches
    assert pi._block_to_text({"XXtypeXX": "text", "text": "nope"}) == ""
    assert pi._block_to_text({"type": None, "text": "nope"}) == ""


def test_block_to_text_text_default_empty_on_missing_key():
    # missing text key -> "" default
    assert pi._block_to_text({"type": "text"}) == ""


def test_block_to_text_toolcall_default_name_and_args():
    # name missing -> "" default; arguments missing -> {} default
    assert pi._block_to_text({"type": "toolCall"}) == "<tool_use name=''>{}</tool_use>"


def test_block_to_text_thinking_default_empty():
    assert pi._block_to_text({"type": "thinking"}) == "<thinking></thinking>"
