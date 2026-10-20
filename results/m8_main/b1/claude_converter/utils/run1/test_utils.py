from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import pytest

from claude_converter.utils import (
    BOLD,
    CYAN,
    DEFAULT_ROLE_COLORS,
    DIM,
    GREEN,
    RESET,
    YELLOW,
    InspectorSchema,
    _pil_to_data_uri,
    _scaled_bar,
    _truncate,
    convert_base64_to_pil_image,
    dget,
    dlist,
    finalize_content,
    load_jsonl,
    merge_content_parts,
    messages_to_json_safe,
    print_block_summary,
    print_flow,
    print_header,
    print_message_roles,
    print_raw_examples,
    print_record_types,
    print_token_usage,
    run_inspection,
)

PIL_AVAILABLE = True
try:
    import PIL.Image  # noqa: F401
except ImportError:
    PIL_AVAILABLE = False

requires_pil = pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def _make_schema(**overrides) -> InspectorSchema:
    defaults = dict(
        label="TEST",
        record_type_of=lambda r: r.get("type", "unknown"),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
    )
    defaults.update(overrides)
    return InspectorSchema(**defaults)


# --------------------------------------------------------------------------- #
# load_jsonl
# --------------------------------------------------------------------------- #

class TestLoadJsonl:
    def test_loads_valid_jsonl(self, tmp_path):
        p = tmp_path / "session.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        assert load_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_accepts_string_path(self, tmp_path):
        p = tmp_path / "session.jsonl"
        p.write_text('{"x": 1}\n', encoding="utf-8")
        assert load_jsonl(str(p)) == [{"x": 1}]

    def test_loads_json_extension(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text('{"a": 1}\n', encoding="utf-8")
        assert load_jsonl(p) == [{"a": 1}]

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text('\n{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
        assert load_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_invalid_lines_logged_to_stderr_and_skipped(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
        result = load_jsonl(p)
        assert result == [{"a": 1}, {"b": 2}]
        err = capsys.readouterr().err
        assert "Line 2 is invalid JSON" in err

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_jsonl(tmp_path / "nope.jsonl")

    def test_bad_extension_raises(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text('{"a": 1}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Unrecognized extension"):
            load_jsonl(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no valid records"):
            load_jsonl(p)

    def test_all_invalid_lines_raises(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("garbage\nmore garbage\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no valid records"):
            load_jsonl(p)


# --------------------------------------------------------------------------- #
# convert_base64_to_pil_image
# --------------------------------------------------------------------------- #

class TestConvertBase64ToPilImage:
    def test_missing_pillow_raises_import_error(self, monkeypatch):
        # Simulate PIL not being importable.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("no PIL")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="Pillow is required"):
            convert_base64_to_pil_image("abc")

    @requires_pil
    def test_invalid_payload_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not decode"):
            convert_base64_to_pil_image("!!!not-base64!!!")

    @requires_pil
    def test_valid_base64_but_not_image_raises(self):
        b64 = base64.b64encode(b"not an image at all").decode()
        with pytest.raises(ValueError, match="Could not decode"):
            convert_base64_to_pil_image(b64)

    @requires_pil
    def test_decodes_data_uri(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (3, 3), color=(1, 2, 3))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        uri = f"data:image/png;base64,{b64}"
        result = convert_base64_to_pil_image(uri)
        assert result.size == (3, 3)

    @requires_pil
    def test_decodes_bare_base64(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (2, 5), color=(9, 8, 7))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        result = convert_base64_to_pil_image(b64)
        assert result.size == (2, 5)


# --------------------------------------------------------------------------- #
# _pil_to_data_uri
# --------------------------------------------------------------------------- #

class TestPilToDataUri:
    @requires_pil
    def test_data_uri_prefix_and_roundtrip(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (4, 4), color=(255, 0, 0))
        uri = _pil_to_data_uri(img)
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",", 1)[1])
        reopened = PIL.open(BytesIO(decoded))
        assert reopened.size == (4, 4)

    @requires_pil
    def test_falls_back_to_png_when_format_fails(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (2, 2))
        img.format = "BOGUS"
        uri = _pil_to_data_uri(img)
        assert uri.startswith("data:image/png;base64,")


# --------------------------------------------------------------------------- #
# messages_to_json_safe
# --------------------------------------------------------------------------- #

class TestMessagesToJsonSafe:
    def test_string_content_unchanged(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = messages_to_json_safe(msgs)
        assert out == [{"role": "user", "content": "hi"}]
        assert out[0] is not msgs[0]

    def test_text_only_parts_unchanged(self):
        content = [{"type": "text", "text": "hi"}]
        msgs = [{"role": "user", "content": content}]
        out = messages_to_json_safe(msgs)
        assert out == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    @requires_pil
    def test_pil_image_converted_to_data_uri(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (4, 4), color=(255, 0, 0))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        out = messages_to_json_safe(msgs)
        part = out[0]["content"][0]
        assert part["type"] == "image"
        assert isinstance(part["image"], str)
        assert part["image"].startswith("data:image/png;base64,")
        json.dumps(out)

    def test_non_image_dicts_preserved(self):
        content = [{"type": "image", "image": "url-not-pil"}]
        msgs = [{"role": "user", "content": content}]
        out = messages_to_json_safe(msgs)
        # Without PIL installed, non-PIL images are repr()'d; with PIL, kept.
        if PIL_AVAILABLE:
            assert out[0]["content"][0] == {"type": "image", "image": "url-not-pil"}
        else:
            assert out[0]["content"][0] == {
                "type": "image",
                "image": repr("url-not-pil"),
            }

    @requires_pil
    def test_original_messages_not_mutated(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (4, 4), color=(255, 0, 0))
        content = [{"type": "image", "image": img}]
        msgs = [{"role": "user", "content": content}]
        messages_to_json_safe(msgs)
        assert msgs[0]["content"][0]["image"] is img

    def test_missing_content_key(self):
        out = messages_to_json_safe([{"role": "user"}])
        assert out == [{"role": "user", "content": None}]

    def test_non_list_content_passthrough(self):
        out = messages_to_json_safe([{"role": "user", "content": 42}])
        assert out == [{"role": "user", "content": 42}]

    def test_pil_import_error_path(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("no PIL")
            return real_import(name, *args, **kwargs)

        # Use a non-PIL object as the image; when PIL import fails, the
        # code falls back to repr(image).
        obj = object()
        msgs = [{"role": "user", "content": [{"type": "image", "image": obj}]}]
        monkeypatch.setattr(builtins, "__import__", fake_import)
        out = messages_to_json_safe(msgs)
        part = out[0]["content"][0]
        assert part["type"] == "image"
        assert part["image"] == repr(obj)


# --------------------------------------------------------------------------- #
# merge_content_parts
# --------------------------------------------------------------------------- #

class TestMergeContentParts:
    def test_appends_when_existing_empty(self):
        existing = []
        merge_content_parts(existing, [{"type": "text", "text": "a"}])
        assert existing == [{"type": "text", "text": "a"}]

    def test_joins_adjacent_text_with_newline(self):
        existing = [{"type": "text", "text": "a"}]
        merge_content_parts(existing, [{"type": "text", "text": "b"}])
        assert existing == [{"type": "text", "text": "a\nb"}]

    def test_multiple_new_parts(self):
        existing = [{"type": "text", "text": "a"}]
        merge_content_parts(existing, [
            {"type": "text", "text": "b"},
            {"type": "text", "text": "c"},
        ])
        assert existing == [{"type": "text", "text": "a\nb\nc"}]

    def test_images_not_merged(self):
        existing = [{"type": "image", "image": "x"}]
        merge_content_parts(existing, [{"type": "image", "image": "y"}])
        assert existing == [
            {"type": "image", "image": "x"},
            {"type": "image", "image": "y"},
        ]

    def test_text_then_image(self):
        existing = [{"type": "text", "text": "a"}]
        merge_content_parts(existing, [{"type": "image", "image": "y"}])
        merge_content_parts(existing, [{"type": "text", "text": "b"}])
        assert existing == [
            {"type": "text", "text": "a"},
            {"type": "image", "image": "y"},
            {"type": "text", "text": "b"},
        ]

    def test_appended_parts_are_copied(self):
        part = {"type": "text", "text": "z"}
        existing = [{"type": "image", "image": "x"}]
        merge_content_parts(existing, [part])
        assert existing[-1] == part
        assert existing[-1] is not part

    def test_returns_none(self):
        existing = []
        assert merge_content_parts(existing, []) is None


# --------------------------------------------------------------------------- #
# finalize_content
# --------------------------------------------------------------------------- #

class TestFinalizeContent:
    def test_text_only_collapses_to_string(self):
        parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert finalize_content(parts) == "a\nb"

    def test_single_text(self):
        assert finalize_content([{"type": "text", "text": "hello"}]) == "hello"

    def test_with_image_returns_list(self):
        parts = [
            {"type": "text", "text": "a"},
            {"type": "image", "image": "x"},
        ]
        assert finalize_content(parts) == parts

    def test_empty_list_returns_empty_string(self):
        assert finalize_content([]) == ""


# --------------------------------------------------------------------------- #
# dget / dlist
# --------------------------------------------------------------------------- #

class TestDget:
    def test_returns_nested_dict(self):
        assert dget({"a": {"b": 1}}, "a") == {"b": 1}

    def test_missing_key(self):
        assert dget({"a": 1}, "b") == {}

    def test_explicit_null_returns_empty(self):
        assert dget({"a": None}, "a") == {}

    def test_non_dict_value_returns_empty(self):
        assert dget({"a": [1, 2]}, "a") == {}
        assert dget({"a": "string"}, "a") == {}

    def test_non_dict_parent(self):
        assert dget(None, "a") == {}
        assert dget([], "a") == {}
        assert dget("x", "a") == {}


class TestDlist:
    def test_returns_list(self):
        assert dlist({"a": [1, 2]}, "a") == [1, 2]

    def test_missing_key(self):
        assert dlist({}, "a") == []

    def test_explicit_null(self):
        assert dlist({"a": None}, "a") == []

    def test_non_list_value(self):
        assert dlist({"a": {"x": 1}}, "a") == []

    def test_non_dict_parent(self):
        assert dlist(None, "a") == []
        assert dlist([1], "a") == []


# --------------------------------------------------------------------------- #
# InspectorSchema
# --------------------------------------------------------------------------- #

class TestInspectorSchema:
    def test_defaults(self):
        schema = _make_schema()
        assert schema.label == "TEST"
        assert schema.role_colors == DEFAULT_ROLE_COLORS
        assert schema.role_colors is not DEFAULT_ROLE_COLORS
        assert schema.total_tokens_of is None
        assert schema.tokens_of is None
        assert schema.block_types_of is None
        assert schema.blocks_of is None
        assert schema.render_block is None
        assert schema.timestamp_of({"anything": 1}) == ""

    def test_role_colors_are_independent_copies(self):
        s1 = _make_schema()
        s2 = _make_schema()
        s1.role_colors["user"] = "x"
        assert s2.role_colors["user"] == GREEN


# --------------------------------------------------------------------------- #
# _scaled_bar
# --------------------------------------------------------------------------- #

class TestScaledBar:
    def test_zero_max_returns_empty(self):
        assert _scaled_bar(0, 0) == ""

    def test_negative_max_returns_empty(self):
        assert _scaled_bar(5, -1) == ""

    def test_max_count_gives_full_bar(self):
        assert _scaled_bar(40, 40) == "█" * 40

    def test_proportional(self):
        assert _scaled_bar(20, 40) == "█" * 20

    def test_minimum_one_bar(self):
        assert _scaled_bar(1, 1000) == "█"


# --------------------------------------------------------------------------- #
# print_* functions
# --------------------------------------------------------------------------- #

class TestPrintHeader:
    def test_prints_label_and_stats(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text("{}\n", encoding="utf-8")
        schema = _make_schema()
        print_header(p, [{}, {}], schema)
        out = capsys.readouterr().out
        assert "TEST SESSION INSPECTOR" in out
        assert "File" in out
        assert "Size" in out
        assert "Lines" in out


class TestPrintRecordTypes:
    def test_counts_and_ordering(self, capsys):
        records = [{"type": "a"}, {"type": "a"}, {"type": "b"}]
        print_record_types(records, _make_schema())
        out = capsys.readouterr().out
        assert "RECORD TYPES" in out
        assert "a" in out
        assert "b" in out

    def test_empty_records(self, capsys):
        print_record_types([], _make_schema())
        out = capsys.readouterr().out
        assert "RECORD TYPES" in out


class TestPrintBlockSummary:
    def test_noop_without_block_types_of(self, capsys):
        schema = _make_schema()
        print_block_summary([{"type": "message"}], schema)
        assert capsys.readouterr().out == ""

    def test_noop_when_no_records(self, capsys):
        schema = _make_schema(block_types_of=lambda r: ["text"])
        print_block_summary([], schema)
        assert capsys.readouterr().out == ""

    def test_prints_breakdown(self, capsys):
        schema = _make_schema(
            block_types_of=lambda r: r.get("blocks", []),
        )
        records = [
            {"type": "message", "role": "user", "blocks": ["text"]},
            {"type": "message", "role": "user", "blocks": ["text", "tool_use"]},
        ]
        print_block_summary(records, schema)
        out = capsys.readouterr().out
        assert "CONTENT BLOCKS" in out
        assert "text" in out
        assert "tool_use" in out
        assert "user" in out

    def test_non_message_uses_record_type(self, capsys):
        schema = _make_schema(
            block_types_of=lambda r: r.get("blocks", []),
        )
        records = [{"type": "meta", "blocks": ["info"]}]
        print_block_summary(records, schema)
        out = capsys.readouterr().out
        assert "meta" in out
        assert "info" in out


class TestPrintMessageRoles:
    def test_noop_without_messages(self, capsys):
        schema = _make_schema()
        print_message_roles([{"type": "other"}], schema)
        assert capsys.readouterr().out == ""

    def test_prints_roles(self, capsys):
        schema = _make_schema()
        records = [
            {"type": "message", "role": "user"},
            {"type": "message", "role": "user"},
            {"type": "message", "role": "assistant"},
        ]
        print_message_roles(records, schema)
        out = capsys.readouterr().out
        assert "MESSAGE ROLES" in out
        assert "user" in out
        assert "assistant" in out


class TestPrintTokenUsage:
    def test_noop_without_total_tokens_of(self, capsys):
        print_token_usage([{}], _make_schema())
        assert capsys.readouterr().out == ""

    def test_noop_when_totals_empty(self, capsys):
        schema = _make_schema(total_tokens_of=lambda recs: {})
        print_token_usage([{}], schema)
        assert capsys.readouterr().out == ""

    def test_prints_totals(self, capsys):
        schema = _make_schema(total_tokens_of=lambda recs: {"input_tokens": 1234})
        print_token_usage([{}], schema)
        out = capsys.readouterr().out
        assert "TOKEN USAGE" in out
        assert "Input Tokens" in out
        assert "1,234" in out


class TestPrintFlow:
    def test_noop_without_messages(self, capsys):
        print_flow([{"type": "other"}], _make_schema())
        assert capsys.readouterr().out == ""

    def test_basic_flow(self, capsys):
        schema = _make_schema()
        records = [
            {"type": "message", "role": "user", "text": "hello"},
            {"type": "message", "role": "assistant", "text": "hi"},
        ]
        print_flow(records, schema)
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" in out
        assert "[USER]" in out
        assert "[ASSISTANT]" in out

    def test_show_text_truncates(self, capsys):
        schema = _make_schema()
        records = [{"type": "message", "role": "user", "text": "x" * 300}]
        print_flow(records, schema, show_text=True)
        out = capsys.readouterr().out
        assert "xxx" in out
        assert "..." in out

    def test_show_text_false_hides_text(self, capsys):
        schema = _make_schema()
        records = [{"type": "message", "role": "user", "text": "SECRET"}]
        print_flow(records, schema, show_text=False)
        out = capsys.readouterr().out
        assert "SECRET" not in out

    def test_tokens_label(self, capsys):
        schema = _make_schema(tokens_of=lambda r: r.get("tk", 0))
        records = [{"type": "message", "role": "user", "tk": 42}]
        print_flow(records, schema)
        out = capsys.readouterr().out
        assert "42tk" in out

    def test_zero_tokens_no_label(self, capsys):
        schema = _make_schema(tokens_of=lambda r: 0)
        records = [{"type": "message", "role": "user"}]
        print_flow(records, schema)
        out = capsys.readouterr().out
        assert "0tk" not in out

    def test_block_types_printed(self, capsys):
        schema = _make_schema(block_types_of=lambda r: ["text", "tool_use"])
        records = [{"type": "message", "role": "user"}]
        print_flow(records, schema)
        out = capsys.readouterr().out
        assert "blocks: text, tool_use" in out

    def test_render_blocks_when_available(self, capsys):
        schema = _make_schema(
            blocks_of=lambda r: [{"type": "text", "text": "rendered"}],
            render_block=lambda b: "RENDERED-BLOCK",
        )
        records = [{"type": "message", "role": "user"}]
        print_flow(records, schema, show_text=True)
        out = capsys.readouterr().out
        assert "RENDERED-BLOCK" in out
        # When blocks render, the flattened text fallback is not used.
        assert "rendered" not in out.replace("RENDERED-BLOCK", "")

    def test_render_blocks_not_used_when_show_text_false(self, capsys):
        schema = _make_schema(
            blocks_of=lambda r: [{"type": "text", "text": "rendered"}],
            render_block=lambda b: "RENDERED-BLOCK",
        )
        records = [{"type": "message", "role": "user"}]
        print_flow(records, schema, show_text=False)
        out = capsys.readouterr().out
        assert "RENDERED-BLOCK" not in out

    def test_show_text_without_render_block_uses_text_of(self, capsys):
        schema = _make_schema(
            blocks_of=lambda r: [{"type": "text", "text": "rendered"}],
        )
        records = [{"type": "message", "role": "user", "text": "fallback-text"}]
        print_flow(records, schema, show_text=True)
        out = capsys.readouterr().out
        assert "fallback-text" in out


class TestPrintRawExamples:
    def test_one_example_per_type(self, capsys):
        schema = _make_schema()
        records = [
            {"type": "a", "text": "t1"},
            {"type": "a", "text": "t2"},
            {"type": "b"},
        ]
        print_raw_examples(records, schema)
        out = capsys.readouterr().out
        assert "RAW RECORD EXAMPLES" in out
        assert out.count("type = a") == 1
        assert out.count("type = b") == 1

    def test_message_prints_flattened_text(self, capsys):
        schema = _make_schema()
        records = [{"type": "message", "role": "user", "text": "flattened"}]
        print_raw_examples(records, schema)
        out = capsys.readouterr().out
        assert "flattened" in out

    def test_non_message_omits_flattened_text(self, capsys):
        schema = _make_schema()
        records = [{"type": "meta"}]
        print_raw_examples(records, schema)
        out = capsys.readouterr().out
        assert "type = meta" in out
        assert "flattened text:" not in out


class TestRunInspection:
    def test_full_run(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text("{}\n", encoding="utf-8")
        schema = _make_schema()
        records = [{"type": "message", "role": "user", "text": "hi"}]
        run_inspection(p, records, schema)
        out = capsys.readouterr().out
        assert "TEST SESSION INSPECTOR" in out
        assert "RECORD TYPES" in out
        assert "MESSAGE ROLES" in out

    def test_extra_header_called(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("{}\n", encoding="utf-8")
        schema = _make_schema()
        records = [{"type": "message", "role": "user"}]
        captured = []

        def extra(recs):
            captured.append(recs)

        run_inspection(p, records, schema, extra_header=extra)
        assert captured == [records]

    def test_show_flow_calls_print_flow(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text("{}\n", encoding="utf-8")
        schema = _make_schema()
        records = [{"type": "message", "role": "user", "text": "hi"}]
        run_inspection(p, records, schema, show_flow=True)
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" in out

    def test_show_flow_false_no_flow(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text("{}\n", encoding="utf-8")
        schema = _make_schema()
        records = [{"type": "message", "role": "user", "text": "hi"}]
        run_inspection(p, records, schema, show_flow=False)
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" not in out

    def test_accepts_string_path(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text("{}\n", encoding="utf-8")
        schema = _make_schema()
        run_inspection(str(p), [{"type": "other"}], schema)
        out = capsys.readouterr().out
        assert "TEST SESSION INSPECTOR" in out


# --------------------------------------------------------------------------- #
# _truncate
# --------------------------------------------------------------------------- #

class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("x" * 20, 5)
        assert result.startswith("xxxxx")
        assert "..." in result

    def test_newlines_escaped(self):
        assert _truncate("a\nb", 100) == "a\\nb"

    def test_non_string_input(self):
        assert _truncate(123, 100) == "123"
