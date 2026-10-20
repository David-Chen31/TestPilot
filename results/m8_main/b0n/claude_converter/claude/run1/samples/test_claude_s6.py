"""Unit tests for claude_converter.claude."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import claude_converter.claude as claude
from claude_converter.claude import (
    inspect_session,
    load_session,
    records_to_messages,
    session_to_messages,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _user(text: str, **extra) -> dict:
    rec = {"type": "user", "message": {"role": "user", "content": text}}
    rec.update(extra)
    return rec


def _assistant(blocks, **extra) -> dict:
    rec = {"type": "assistant", "message": {"role": "assistant", "content": blocks}}
    rec.update(extra)
    return rec


class _FakeImage:
    """Stand-in for a PIL.Image so tests don't need Pillow."""

    def __init__(self, name: str = "img"):
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_FakeImage({self.name!r})"


# ---------------------------------------------------------------------------
# load_session
# ---------------------------------------------------------------------------


class TestLoadSession:
    def test_loads_records(self, tmp_path):
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [_user("hi"), _assistant([{"type": "text", "text": "yo"}])])
        records = load_session(path)
        assert records == [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
            },
        ]

    def test_accepts_str_path(self, tmp_path):
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [_user("hi")])
        assert load_session(str(path)) == [_user("hi")]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nope.jsonl")

    def test_bad_extension_raises(self, tmp_path):
        path = tmp_path / "s.txt"
        _write_jsonl(path, [_user("hi")])
        with pytest.raises(ValueError):
            load_session(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(path)

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text(
            "\n" + json.dumps(_user("hi")) + "\n   \n", encoding="utf-8"
        )
        assert load_session(path) == [_user("hi")]

    def test_skips_invalid_json_lines(self, tmp_path, capsys):
        path = tmp_path / "s.jsonl"
        path.write_text(
            json.dumps(_user("hi")) + "\nNOT JSON\n", encoding="utf-8"
        )
        records = load_session(path)
        assert records == [_user("hi")]
        err = capsys.readouterr().err
        assert "invalid JSON" in err


# ---------------------------------------------------------------------------
# _block_to_str / _content_to_str (through public paths)
# ---------------------------------------------------------------------------


class TestBlockFlattening:
    def test_text_block(self):
        assert claude._block_to_str({"type": "text", "text": "hello"}) == "hello"

    def test_thinking_block(self):
        assert (
            claude._block_to_str({"type": "thinking", "thinking": "hmm"})
            == "<thinking>hmm</thinking>"
        )

    def test_tool_use_block(self):
        out = claude._block_to_str(
            {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}}
        )
        assert out == "<tool_use name='Bash'>{\"cmd\": \"ls\"}</tool_use>"

    def test_tool_use_block_empty_input(self):
        out = claude._block_to_str({"type": "tool_use", "name": "X"})
        assert out == "<tool_use name='X'>{}</tool_use>"

    def test_tool_result_string_content(self):
        out = claude._block_to_str({"type": "tool_result", "content": "ok"})
        assert out == "<tool_result>ok</tool_result>"

    def test_tool_result_list_content(self):
        block = {
            "type": "tool_result",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image"},
                "skip-non-dict",
                {"type": "text", "text": "b"},
            ],
        }
        assert claude._block_to_str(block) == "<tool_result>a <image> b</tool_result>"

    def test_image_block(self):
        assert claude._block_to_str({"type": "image"}) == "<image>"

    def test_unknown_block(self):
        assert claude._block_to_str({"type": "weird"}) == ""
        assert claude._block_to_str({}) == ""

    def test_content_to_str_str(self):
        assert claude._content_to_str("direct") == "direct"

    def test_content_to_str_list_joins_nonempty(self):
        content = [
            {"type": "text", "text": "a"},
            {"type": "weird"},
            {"type": "text", "text": "b"},
            "not-a-dict",
        ]
        assert claude._content_to_str(content) == "a\nb"

    def test_content_to_str_none(self):
        assert claude._content_to_str(None) == ""


# ---------------------------------------------------------------------------
# _decode_claude_image
# ---------------------------------------------------------------------------


class TestDecodeClaudeImage:
    def test_unsupported_source_type(self, capsys, monkeypatch):
        monkeypatch.setattr(
            claude, "convert_base64_to_pil_image", lambda d: pytest.fail("should not be called")
        )
        result = claude._decode_claude_image(
            {"source": {"type": "url", "data": "abc"}}
        )
        assert result is None
        assert "unsupported source type" in capsys.readouterr().err

    def test_missing_source(self, capsys):
        result = claude._decode_claude_image({})
        assert result is None
        assert "unsupported source type" in capsys.readouterr().err

    def test_empty_data(self, capsys):
        result = claude._decode_claude_image(
            {"source": {"type": "base64", "data": ""}}
        )
        assert result is None
        assert "empty base64" in capsys.readouterr().err

    def test_missing_data(self, capsys):
        result = claude._decode_claude_image({"source": {"type": "base64"}})
        assert result is None
        assert "empty base64" in capsys.readouterr().err

    def test_success(self, monkeypatch):
        sentinel = _FakeImage("ok")
        monkeypatch.setattr(
            claude, "convert_base64_to_pil_image", lambda d: sentinel
        )
        assert claude._decode_claude_image(
            {"source": {"type": "base64", "data": "aGk="}}
        ) is sentinel

    def test_value_error_caught(self, monkeypatch, capsys):
        def boom(_):
            raise ValueError("bad")

        monkeypatch.setattr(claude, "convert_base64_to_pil_image", boom)
        result = claude._decode_claude_image(
            {"source": {"type": "base64", "data": "x"}}
        )
        assert result is None
        assert "unreadable image block" in capsys.readouterr().err

    def test_import_error_caught(self, monkeypatch, capsys):
        def boom(_):
            raise ImportError("no pillow")

        monkeypatch.setattr(claude, "convert_base64_to_pil_image", boom)
        result = claude._decode_claude_image(
            {"source": {"type": "base64", "data": "x"}}
        )
        assert result is None
        assert "unreadable image block" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _block_to_parts
# ---------------------------------------------------------------------------


class TestBlockToParts:
    def test_text_block(self):
        assert claude._block_to_parts({"type": "text", "text": "hi"}) == [
            {"type": "text", "text": "hi"}
        ]

    def test_blank_text_dropped(self):
        assert claude._block_to_parts({"type": "text", "text": "  "}) == []

    def test_image_block_success(self, monkeypatch):
        sentinel = _FakeImage()
        monkeypatch.setattr(
            claude, "convert_base64_to_pil_image", lambda d: sentinel
        )
        parts = claude._block_to_parts(
            {"type": "image", "source": {"type": "base64", "data": "aGk="}}
        )
        assert parts == [{"type": "image", "image": sentinel}]

    def test_image_block_failure_empty(self, capsys):
        parts = claude._block_to_parts(
            {"type": "image", "source": {"type": "base64", "data": ""}}
        )
        assert parts == []

    def test_tool_result_text_only_matches_block_to_str(self):
        block = {"type": "tool_result", "content": "plain text"}
        assert claude._block_to_parts(block) == [
            {"type": "text", "text": "<tool_result>plain text</tool_result>"}
        ]

    def test_tool_result_blank_produces_empty(self):
        block = {"type": "tool_result", "content": ""}
        # _block_to_str -> "<tool_result></tool_result>" which is non-blank...
        parts = claude._block_to_parts(block)
        assert parts == [
            {"type": "text", "text": "<tool_result></tool_result>"}
        ]

    def test_tool_result_with_nested_image(self, monkeypatch):
        sentinel1 = _FakeImage("1")
        sentinel2 = _FakeImage("2")

        def decoder(block):
            # return different sentinels for the two image blocks
            if block.get("marker") == "one":
                return sentinel1
            return sentinel2

        monkeypatch.setattr(claude, "_decode_claude_image", decoder)

        block = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "marker": "one"},
                "not-a-dict",
                {"type": "text", "text": "   "},  # blank, dropped
                {"type": "image", "marker": "two"},
                {"type": "text", "text": "after"},
            ],
        }
        parts = claude._block_to_parts(block)
        assert parts == [
            {"type": "text", "text": "<tool_result id='t1'>"},
            {"type": "text", "text": "before"},
            {"type": "image", "image": sentinel1},
            {"type": "image", "image": sentinel2},
            {"type": "text", "text": "after"},
            {"type": "text", "text": "</tool_result>"},
        ]

    def test_tool_result_missing_id_defaults(self, monkeypatch):
        monkeypatch.setattr(
            claude, "_decode_claude_image", lambda b: _FakeImage()
        )
        block = {
            "type": "tool_result",
            "content": [{"type": "image"}],
        }
        parts = claude._block_to_parts(block)
        assert parts[0] == {"type": "text", "text": "<tool_result id='?'>"}
        assert parts[-1] == {"type": "text", "text": "</tool_result>"}

    def test_tool_result_image_decoder_returns_none(self, monkeypatch):
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: None)
        block = {
            "type": "tool_result",
            "tool_use_id": "x",
            "content": [{"type": "image"}],
        }
        parts = claude._block_to_parts(block)
        assert parts == [
            {"type": "text", "text": "<tool_result id='x'>"},
            {"type": "text", "text": "</tool_result>"},
        ]


# ---------------------------------------------------------------------------
# _content_to_parts / _collect_block_types
# ---------------------------------------------------------------------------


class TestContentToParts:
    def test_string_content(self):
        assert claude._content_to_parts("hi") == [{"type": "text", "text": "hi"}]

    def test_string_blank(self):
        assert claude._content_to_parts("   ") == []

    def test_list_content(self):
        content = [
            {"type": "text", "text": "a"},
            "skip",
            {"type": "text", "text": "b"},
        ]
        assert claude._content_to_parts(content) == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_none_content(self):
        assert claude._content_to_parts(None) == []


class TestCollectBlockTypes:
    def test_string(self):
        assert claude._collect_block_types("hi") == ["raw_string"]

    def test_list(self):
        content = [{"type": "text"}, {"type": "tool_use"}, "skip", {}]
        assert claude._collect_block_types(content) == [
            "text",
            "tool_use",
            "unknown",
        ]

    def test_none(self):
        assert claude._collect_block_types(None) == ["unknown"]


# ---------------------------------------------------------------------------
# records_to_messages
# ---------------------------------------------------------------------------


class TestRecordsToMessages:
    def test_skips_non_message_records(self):
        records = [
            {"type": "system", "message": {"role": "system", "content": "x"}},
            _user("hi"),
        ]
        assert records_to_messages(records) == [{"role": "user", "content": "hi"}]

    def test_empty_records(self):
        assert records_to_messages([]) == []

    def test_text_only_produces_string_content(self):
        records = [
            _user("hello"),
            _assistant([{"type": "text", "text": "world"}]),
        ]
        assert records_to_messages(records) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_merges_consecutive_same_role(self):
        records = [_user("a"), _user("b")]
        assert records_to_messages(records) == [{"role": "user", "content": "a\nb"}]

    def test_alternating_roles_not_merged(self):
        records = [_user("a"), {"type": "assistant", "message": {"role": "assistant", "content": "b"}}]
        assert records_to_messages(records) == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    def test_empty_content_skipped(self):
        records = [
            {"type": "user", "message": {"role": "user", "content": []}},
            _user("real"),
        ]
        assert records_to_messages(records) == [{"role": "user", "content": "real"}]

    def test_role_defaults_to_type(self):
        records = [
            {"type": "user", "message": {"content": "hi"}},
        ]
        assert records_to_messages(records) == [{"role": "user", "content": "hi"}]

    def test_missing_message_uses_empty(self):
        records = [{"type": "user"}]
        assert records_to_messages(records) == []

    def test_thinking_and_tool_use_flattened(self):
        records = [
            _assistant(
                [
                    {"type": "thinking", "thinking": "thought"},
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "name": "Bash", "input": {"a": 1}},
                ]
            )
        ]
        content = records_to_messages(records)[0]["content"]
        assert content == (
            "<thinking>thought</thinking>\n"
            "answer\n"
            '<tool_use name=\'Bash\'>{"a": 1}</tool_use>'
        )

    def test_image_kept_as_multimodal_part(self, monkeypatch):
        sentinel = _FakeImage()
        monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda d: sentinel)
        records = [
            _assistant(
                [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "data": "aGk="}},
                ]
            )
        ]
        messages = records_to_messages(records)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], list)
        assert msg["content"] == [
            {"type": "text", "text": "look"},
            {"type": "image", "image": sentinel},
        ]

    def test_user_string_then_assistant_image_merge_keeps_list(self, monkeypatch):
        sentinel = _FakeImage()
        monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda d: sentinel)
        records = [
            _user("first"),
            _assistant(
                [
                    {"type": "text", "text": "text"},
                    {"type": "image", "source": {"type": "base64", "data": "aGk="}},
                ]
            ),
        ]
        messages = records_to_messages(records)
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"] == [
            {"type": "text", "text": "text"},
            {"type": "image", "image": sentinel},
        ]

    def test_merge_same_role_with_image_merges_text(self, monkeypatch):
        sentinel = _FakeImage()
        monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda d: sentinel)
        records = [
            _assistant([{"type": "text", "text": "one"}]),
            _assistant(
                [
                    {"type": "text", "text": "two"},
                    {"type": "image", "source": {"type": "base64", "data": "aGk="}},
                ]
            ),
        ]
        messages = records_to_messages(records)
        assert messages == [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "one\ntwo"},
                    {"type": "image", "image": sentinel},
                ],
            }
        ]

    def test_tool_result_with_nested_image_produces_parts(self, monkeypatch):
        sentinel = _FakeImage()
        monkeypatch.setattr(claude, "_decode_claude_image", lambda b: sentinel)
        records = [
            _user(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "see"},
                            {"type": "image"},
                        ],
                    }
                ]
            )
        ]
        messages = records_to_messages(records)
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == [
            {"type": "text", "text": "<tool_result id='t1'>"},
            {"type": "text", "text": "see"},
            {"type": "image", "image": sentinel},
            {"type": "text", "text": "</tool_result>"},
        ]

    def test_null_message_handled(self):
        # dget tolerates explicit null message
        records = [{"type": "user", "message": None}]
        assert records_to_messages(records) == []


# ---------------------------------------------------------------------------
# session_to_messages
# ---------------------------------------------------------------------------


class TestSessionToMessages:
    def test_basic_roundtrip(self, tmp_path):
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [_user("hi"), _assistant([{"type": "text", "text": "yo"}])])
        messages = session_to_messages(path)
        assert messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]

    def test_output_written(self, tmp_path):
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [_user("hi")])
        out = tmp_path / "out" / "messages.json"
        messages = session_to_messages(path, output=out)
        assert messages == [{"role": "user", "content": "hi"}]
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == [{"role": "user", "content": "hi"}]

    def test_output_serializes_images(self, tmp_path, monkeypatch):
        sentinel = _FakeImage()
        monkeypatch.setattr(claude, "convert_base64_to_pil_image", lambda d: sentinel)
        # Provide a json_safe that converts image to a marker string so the
        # write path works with our fake image.
        def fake_safe(messages):
            out = []
            for m in messages:
                content = m["content"]
                if isinstance(content, list):
                    new = []
                    for part in content:
                        if part.get("type") == "image":
                            new.append({"type": "image", "image": "DATA_URI"})
                        else:
                            new.append(part)
                    out.append({"role": m["role"], "content": new})
                else:
                    out.append(m)
            return out

        monkeypatch.setattr(claude, "messages_to_json_safe", fake_safe)

        path = tmp_path / "s.jsonl"
        _write_jsonl(
            path,
            [
                _assistant(
                    [
                        {"type": "image", "source": {"type": "base64", "data": "aGk="}},
                    ]
                )
            ],
        )
        out = tmp_path / "messages.json"
        messages = session_to_messages(path, output=out)
        # in-memory keeps real object
        assert messages[0]["content"][0]["image"] is sentinel
        # on-disk is serialized
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == [
            {"role": "assistant", "content": [{"type": "image", "image": "DATA_URI"}]}
        ]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            session_to_messages(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# schema helpers (used by inspect_session)
# ---------------------------------------------------------------------------


class TestSchemaHelpers:
    def test_record_type_of(self):
        assert claude._record_type_of({"type": "user"}) == "user"
        assert claude._record_type_of({}) == "unknown"

    def test_is_message(self):
        assert claude._is_message({"type": "user"})
        assert claude._is_message({"type": "assistant"})
        assert not claude._is_message({"type": "system"})

    def test_role_of(self):
        assert claude._role_of({"type": "user", "message": {"role": "assistant"}}) == "assistant"

    def test_role_of_defaults_to_type(self):
        assert claude._role_of({"type": "user", "message": {}}) == "user"

    def test_text_of(self):
        rec = {"message": {"content": "hello"}}
        assert claude._text_of(rec) == "hello"

    def test_timestamp_of(self):
        rec = {"timestamp": "2024-01-02T03:04:05.999Z"}
        assert claude._timestamp_of(rec) == "2024-01-02 03:04:05"

    def test_timestamp_of_missing(self):
        assert claude._timestamp_of({}) == ""

    def test_tokens_of(self):
        rec = {"message": {"usage": {"output_tokens": 42}}}
        assert claude._tokens_of(rec) == 42

    def test_tokens_of_missing(self):
        assert claude._tokens_of({}) == 0

    def test_block_types_of(self):
        rec = {"message": {"content": [{"type": "text"}]}}
        assert claude._block_types_of(rec) == ["text"]

    def test_blocks_of(self):
        rec = {"message": {"content": [{"type": "text"}, "skip"]}}
        assert claude._blocks_of(rec) == [{"type": "text"}]

    def test_blocks_of_string_content(self):
        rec = {"message": {"content": "str"}}
        assert claude._blocks_of(rec) == []

    def test_total_tokens_of_empty_when_no_tokens(self):
        assert claude._total_tokens_of([{"message": {"usage": {}}}]) == {}

    def test_total_tokens_of_sums(self):
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
        totals = claude._total_tokens_of(records)
        assert totals == {
            "input": 11,
            "output": 6,
            "cache_read": 2,
            "cache_write": 3,
            "total": 17,
        }


# ---------------------------------------------------------------------------
# _render_block
# ---------------------------------------------------------------------------


class TestRenderBlock:
    def test_text_block(self):
        out = claude._render_block({"type": "text", "text": "hi"})
        assert "text" in out
        assert "hi" in out

    def test_thinking_block(self):
        out = claude._render_block({"type": "thinking", "thinking": "deep"})
        assert "thinking" in out
        assert "deep" in out

    def test_tool_use_block(self):
        out = claude._render_block(
            {"type": "tool_use", "name": "Bash", "id": "abc", "input": {"x": 1}}
        )
        assert "tool_use" in out
        assert "Bash" in out
        assert "abc" in out
        assert '{"x": 1}' in out

    def test_tool_result_string(self):
        out = claude._render_block(
            {"type": "tool_result", "tool_use_id": "id1", "content": "result text"}
        )
        assert "tool_result" in out
        assert "id1" in out
        assert "result text" in out

    def test_tool_result_list_with_image_and_text(self):
        block = {
            "type": "tool_result",
            "tool_use_id": "id1",
            "content": [
                {"type": "image", "source": {"media_type": "image/png", "data": "abcd"}},
                {"type": "text", "text": "caption"},
                "skip",
            ],
        }
        out = claude._render_block(block)
        assert "image/png" in out
        assert "4 base64 chars" in out
        assert "caption" in out

    def test_image_block(self):
        block = {
            "type": "image",
            "source": {"media_type": "image/jpeg", "data": "12345"},
        }
        out = claude._render_block(block)
        assert "image/jpeg" in out
        assert "5 base64 chars" in out

    def test_image_block_missing_source(self):
        out = claude._render_block({"type": "image"})
        assert "? (0 base64 chars)" in out

    def test_unknown_block(self):
        out = claude._render_block({"type": "weird", "x": 1})
        assert "weird" in out


# ---------------------------------------------------------------------------
# inspect_session
# ---------------------------------------------------------------------------


class TestInspectSession:
    def test_calls_run_inspection(self, tmp_path, monkeypatch):
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [_user("hi")])
        calls = {}

        def fake_run_inspection(p, records, schema, **kwargs):
            calls["path"] = p
            calls["records"] = records
            calls["schema"] = schema
            calls["kwargs"] = kwargs

        monkeypatch.setattr(claude, "run_inspection", fake_run_inspection)

        inspect_session(path)
        assert Path(calls["path"]) == path
        assert calls["records"] == [_user("hi")]
        assert calls["schema"] is claude.CLAUDE_SCHEMA
        assert calls["kwargs"]["show_flow"] is False
        assert calls["kwargs"]["show_blocks"] is False
        assert calls["kwargs"]["extra_header"] is claude._print_session_metadata

    def test_show_raw_calls_printer(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [_user("hi")])
        monkeypatch.setattr(claude, "run_inspection", lambda *a, **k: None)
        called = {}
        monkeypatch.setattr(
            claude, "_print_raw_examples", lambda records: called.setdefault("records", records)
        )

        inspect_session(path, show_raw=True)
        assert called["records"] == [_user("hi")]

    def test_show_raw_false_skips(self, tmp_path, monkeypatch):
        path = tmp_path / "s.jsonl"
        _write_jsonl(path, [_user("hi")])
        monkeypatch.setattr(claude, "run_inspection", lambda *a, **k: None)
        monkeypatch.setattr(
            claude, "_print_raw_examples", lambda records: pytest.fail("should not be called")
        )
        inspect_session(path)

    def test_full_run_does_not_crash(self, tmp_path, capsys):
        path = tmp_path / "s.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user",
                    "sessionId": "sess-1",
                    "cwd": "/tmp",
                    "gitBranch": "main",
                    "message": {"role": "user", "content": "hello"},
                    "timestamp": "2024-01-02T03:04:05Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "hi"},
                            {"type": "tool_use", "name": "Bash", "id": "x", "input": {}},
                        ],
                        "usage": {"input_tokens": 3, "output_tokens": 4},
                    },
                },
            ],
        )
        inspect_session(path, show_flow=True, show_blocks=True, show_raw=True)
        out = capsys.readouterr().out
        assert "CLAUDE CODE" in out
        assert "sess-1" in out
        assert "/tmp" in out
        assert "main" in out
        assert "RAW RECORD EXAMPLES" in out


# ---------------------------------------------------------------------------
# _print_session_metadata / _print_raw_examples
# ---------------------------------------------------------------------------


class TestPrintHelpers:
    def test_metadata_no_user(self, capsys):
        claude._print_session_metadata([{"type": "assistant"}])
        assert capsys.readouterr().out == ""

    def test_metadata_prints_fields(self, capsys):
        claude._print_session_metadata(
            [
                {
                    "type": "user",
                    "sessionId": "sid",
                    "cwd": "/cwd",
                    "gitBranch": "dev",
                }
            ]
        )
        out = capsys.readouterr().out
        assert "sid" in out
        assert "/cwd" in out
        assert "dev" in out

    def test_metadata_omits_missing_branch(self, capsys):
        claude._print_session_metadata([{"type": "user", "sessionId": "s", "cwd": "c"}])
        out = capsys.readouterr().out
        assert "Branch" not in out

    def test_raw_examples_one_per_type(self, capsys):
        records = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "x"}]}},
            {"type": "user", "message": {"content": "y"}},
        ]
        claude._print_raw_examples(records)
        out = capsys.readouterr().out
        # type printed only once
        assert out.count("type = user") == 1
        assert "content (string)" in out
