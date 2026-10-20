"""Tests for claude_converter.claude public API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_converter.claude import (
    load_session,
    records_to_messages,
    session_to_messages,
    inspect_session,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def _user(text, **extra):
    rec = {"type": "user", "message": {"role": "user", "content": text}}
    rec.update(extra)
    return rec


def _assistant(content, **extra):
    rec = {"type": "assistant", "message": {"role": "assistant", "content": content}}
    rec.update(extra)
    return rec


# ── load_session ─────────────────────────────────────────────────────────


class TestLoadSession:
    def test_loads_records(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "s.jsonl",
            [_user("hi"), _assistant("hello")],
        )
        records = load_session(p)
        assert records == [_user("hi"), _assistant("hello")]

    def test_loads_from_str_path(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [_user("hi")])
        records = load_session(str(p))
        assert len(records) == 1
        assert records[0]["type"] == "user"

    def test_accepts_dot_json_extension(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.json", [_user("hi")])
        records = load_session(p)
        assert records == [_user("hi")]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nope.jsonl")

    def test_bad_extension_raises(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text('{"type": "user"}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(p)

    def test_skips_invalid_lines(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text(
            json.dumps(_user("ok")) + "\nNOT JSON\n" + json.dumps(_assistant("a")) + "\n",
            encoding="utf-8",
        )
        records = load_session(p)
        assert [r["type"] for r in records] == ["user", "assistant"]
        assert "invalid JSON" in capsys.readouterr().err


# ── records_to_messages ──────────────────────────────────────────────────


class TestRecordsToMessages:
    def test_basic_text_turns(self):
        records = [_user("hi"), _assistant("hello")]
        assert records_to_messages(records) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_skips_non_message_types(self):
        records = [
            {"type": "system", "message": {"role": "system", "content": "sys"}},
            _user("hi"),
            {"type": "summary", "summary": "x"},
        ]
        assert records_to_messages(records) == [{"role": "user", "content": "hi"}]

    def test_empty_content_skipped(self):
        records = [_user(""), _user("   "), _user("real")]
        assert records_to_messages(records) == [{"role": "user", "content": "real"}]

    def test_empty_list_content_skipped(self):
        records = [_user([]), _user("real")]
        assert records_to_messages(records) == [{"role": "user", "content": "real"}]

    def test_role_falls_back_to_record_type(self):
        records = [{"type": "assistant", "message": {"content": "x"}}]
        assert records_to_messages(records) == [{"role": "assistant", "content": "x"}]

    def test_consecutive_same_role_merged(self):
        records = [_user("a"), _user("b"), _assistant("c")]
        assert records_to_messages(records) == [
            {"role": "user", "content": "a\nb"},
            {"role": "assistant", "content": "c"},
        ]

    def test_alternating_roles_not_merged(self):
        records = [_user("a"), _assistant("b"), _user("c")]
        assert records_to_messages(records) == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]

    def test_text_blocks_flattened(self):
        content = [
            {"type": "text", "text": "line1"},
            {"type": "text", "text": "line2"},
        ]
        records = [_user(content)]
        assert records_to_messages(records) == [
            {"role": "user", "content": "line1\nline2"}
        ]

    def test_thinking_block(self):
        content = [{"type": "thinking", "thinking": "deep"}]
        records = [_assistant(content)]
        assert records_to_messages(records) == [
            {"role": "assistant", "content": "<thinking>deep</thinking>"}
        ]

    def test_tool_use_block(self):
        content = [
            {"type": "tool_use", "name": "Read", "input": {"file": "a.txt"}},
        ]
        records = [_assistant(content)]
        messages = records_to_messages(records)
        assert messages == [
            {
                "role": "assistant",
                "content": "<tool_use name='Read'>{'file': 'a.txt'}</tool_use>".replace(
                    "{'file': 'a.txt'}", '{"file": "a.txt"}'
                ),
            }
        ]

    def test_tool_result_string_content(self):
        content = [
            {"type": "tool_result", "tool_use_id": "t1", "content": "output"},
        ]
        records = [_user(content)]
        assert records_to_messages(records) == [
            {"role": "user", "content": "<tool_result>output</tool_result>"}
        ]

    def test_tool_result_nested_text_list(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [
                    {"type": "text", "text": "part1"},
                    {"type": "text", "text": "part2"},
                ],
            }
        ]
        records = [_user(content)]
        assert records_to_messages(records) == [
            {"role": "user", "content": "<tool_result>part1 part2</tool_result>"}
        ]

    def test_tool_result_nested_image_marker(self):
        content = [
            {
                "type": "tool_result",
                "content": [
                    {"type": "text", "text": "see"},
                    {"type": "image"},
                ],
            }
        ]
        records = [_user(content)]
        assert records_to_messages(records) == [
            {"role": "user", "content": "<tool_result>see <image></tool_result>"}
        ]

    def test_image_block_without_pil_skipped(self, monkeypatch):
        import claude_converter.claude as claude_mod

        def fake_decode(block):
            return None

        monkeypatch.setattr(claude_mod, "_decode_claude_image", fake_decode)
        content = [{"type": "image", "source": {"type": "base64", "data": "x"}}]
        records = [_user(content)]
        assert records_to_messages(records) == []

    def test_image_block_with_pil_kept(self, monkeypatch):
        import claude_converter.claude as claude_mod

        sentinel = object()

        def fake_decode(block):
            return sentinel

        monkeypatch.setattr(claude_mod, "_decode_claude_image", fake_decode)
        content = [{"type": "image", "source": {"type": "base64", "data": "x"}}]
        records = [_user(content)]
        messages = records_to_messages(records)
        assert messages == [
            {
                "role": "user",
                "content": [{"type": "image", "image": sentinel}],
            }
        ]
        assert messages[0]["content"][0]["image"] is sentinel

    def test_tool_result_nested_image_kept(self, monkeypatch):
        import claude_converter.claude as claude_mod

        sentinel = object()
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: sentinel)
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "t9",
                "content": [
                    {"type": "text", "text": "here"},
                    {"type": "image", "source": {"type": "base64", "data": "x"}},
                ],
            }
        ]
        records = [_user(content)]
        messages = records_to_messages(records)
        assert len(messages) == 1
        parts = messages[0]["content"]
        assert parts[0] == {"type": "text", "text": "<tool_result id='t9'>"}
        assert parts[1] == {"type": "text", "text": "here"}
        assert parts[2] == {"type": "image", "image": sentinel}
        assert parts[3] == {"type": "text", "text": "</tool_result>"}

    def test_tool_result_nested_image_only_failed_decode(self, monkeypatch):
        import claude_converter.claude as claude_mod

        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: None)
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}],
            }
        ]
        records = [_user(content)]
        messages = records_to_messages(records)
        parts = messages[0]["content"]
        assert parts[0]["text"] == "<tool_result id='t1'>"
        assert parts[-1]["text"] == "</tool_result>"
        assert all(p["type"] == "text" for p in parts)

    def test_non_dict_content_items_ignored(self):
        content = ["raw string", {"type": "text", "text": "ok"}, 42]
        records = [_user(content)]
        assert records_to_messages(records) == [{"role": "user", "content": "ok"}]

    def test_null_message_handled(self):
        records = [{"type": "user", "message": None}]
        assert records_to_messages(records) == []

    def test_raw_string_content_in_list_message(self):
        records = [_user("plain")]
        assert records_to_messages(records) == [{"role": "user", "content": "plain"}]

    def test_non_str_list_content_returns_empty(self):
        records = [{"type": "user", "message": {"role": "user", "content": 123}}]
        assert records_to_messages(records) == []

    def test_unknown_block_type_contributes_nothing(self):
        content = [{"type": "weird", "data": "x"}, {"type": "text", "text": "ok"}]
        records = [_user(content)]
        assert records_to_messages(records) == [{"role": "user", "content": "ok"}]

    def test_mixed_image_and_text_merges_consecutive(self, monkeypatch):
        import claude_converter.claude as claude_mod

        sentinel = object()
        monkeypatch.setattr(claude_mod, "_decode_claude_image", lambda b: sentinel)
        img = {"type": "image", "source": {"type": "base64", "data": "x"}}
        records = [
            _user([img]),
            _user("follow up"),
        ]
        messages = records_to_messages(records)
        assert len(messages) == 1
        parts = messages[0]["content"]
        assert parts[0] == {"type": "image", "image": sentinel}
        assert parts[1] == {"type": "text", "text": "follow up"}


# ── session_to_messages ──────────────────────────────────────────────────


class TestSessionToMessages:
    def test_reads_and_converts(self, tmp_path):
        p = _write_jsonl(tmp_path / "s.jsonl", [_user("hi"), _assistant("hello")])
        assert session_to_messages(p) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_writes_output_file(self, tmp_path):
        src = _write_jsonl(tmp_path / "s.jsonl", [_user("hi")])
        out = tmp_path / "nested" / "out.json"
        result = session_to_messages(src, output=out)
        assert result == [{"role": "user", "content": "hi"}]
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == [{"role": "user", "content": "hi"}]

    def test_output_str_path(self, tmp_path):
        src = _write_jsonl(tmp_path / "s.jsonl", [_user("hi")])
        out = tmp_path / "out.json"
        session_to_messages(src, output=str(out))
        assert out.exists()

    def test_output_serializes_images(self, tmp_path, monkeypatch):
        import claude_converter.claude as claude_mod

        monkeypatch.setattr(
            claude_mod,
            "messages_to_json_safe",
            lambda msgs: [{"role": m["role"], "content": "SAFE"} for m in msgs],
        )
        src = _write_jsonl(tmp_path / "s.jsonl", [_user("hi")])
        out = tmp_path / "out.json"
        result = session_to_messages(src, output=out)
        # in-memory untouched
        assert result == [{"role": "user", "content": "hi"}]
        assert json.loads(out.read_text(encoding="utf-8")) == [
            {"role": "user", "content": "SAFE"}
        ]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            session_to_messages(tmp_path / "nope.jsonl")


# ── inspect_session ──────────────────────────────────────────────────────


class TestInspectSession:
    def test_calls_run_inspection(self, tmp_path, monkeypatch):
        import claude_converter.claude as claude_mod

        calls = {}

        def fake_run(path, records, schema, **kwargs):
            calls["path"] = path
            calls["records"] = records
            calls["schema"] = schema
            calls["kwargs"] = kwargs

        monkeypatch.setattr(claude_mod, "run_inspection", fake_run)

        p = _write_jsonl(tmp_path / "s.jsonl", [_user("hi")])
        inspect_session(p, show_flow=True, show_blocks=True, show_raw=False)

        assert calls["path"] == p
        assert calls["schema"] is claude_mod.CLAUDE_SCHEMA
        assert len(calls["records"]) == 1
        assert calls["kwargs"]["show_flow"] is True
        assert calls["kwargs"]["show_blocks"] is True
        assert calls["kwargs"]["extra_header"] is claude_mod._print_session_metadata

    def test_show_raw_triggers_examples(self, tmp_path, monkeypatch, capsys):
        import claude_converter.claude as claude_mod

        monkeypatch.setattr(claude_mod, "run_inspection", lambda *a, **k: None)
        raw_calls = []
        monkeypatch.setattr(
            claude_mod,
            "_print_raw_examples",
            lambda records: raw_calls.append(records),
        )

        p = _write_jsonl(tmp_path / "s.jsonl", [_user("hi")])
        inspect_session(p, show_raw=True)
        assert len(raw_calls) == 1
        assert raw_calls[0][0]["type"] == "user"

    def test_no_raw_by_default(self, tmp_path, monkeypatch):
        import claude_converter.claude as claude_mod

        monkeypatch.setattr(claude_mod, "run_inspection", lambda *a, **k: None)
        raw_calls = []
        monkeypatch.setattr(
            claude_mod, "_print_raw_examples", lambda records: raw_calls.append(records)
        )
        p = _write_jsonl(tmp_path / "s.jsonl", [_user("hi")])
        inspect_session(p)
        assert raw_calls == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_session(tmp_path / "nope.jsonl")
