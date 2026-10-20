from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import pytest

from claude_converter.utils import (
    RESET,
    BOLD,
    DIM,
    GREEN,
    CYAN,
    YELLOW,
    DEFAULT_ROLE_COLORS,
    _c,
    _hr,
    _truncate,
    _scaled_bar,
    load_jsonl,
    convert_base64_to_pil_image,
    _pil_to_data_uri,
    messages_to_json_safe,
    merge_content_parts,
    finalize_content,
    dget,
    dlist,
    InspectorSchema,
    print_header,
    print_record_types,
    print_block_summary,
    print_message_roles,
    print_token_usage,
    print_flow,
    print_raw_examples,
    run_inspection,
)


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture()
def tiny_png_bytes() -> bytes:
    """A 1x1 PNG, generated in-memory using Pillow."""
    PIL = pytest.importorskip("PIL.Image")
    buf = BytesIO()
    PIL.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def tiny_png_b64(tiny_png_bytes) -> str:
    return base64.b64encode(tiny_png_bytes).decode()


# ---------------------------------------------------------------------------
# ANSI helper functions
# ---------------------------------------------------------------------------

class TestColorHelpers:
    def test_c_wraps_with_codes_and_reset(self):
        assert _c("hi") == "hi" + RESET
        assert _c("hi", BOLD) == BOLD + "hi" + RESET
        assert _c("hi", BOLD, GREEN) == BOLD + GREEN + "hi" + RESET

    def test_c_stringifies_non_str(self):
        assert _c(42) == "42" + RESET

    def test_hr_default(self):
        assert _hr() == DIM + ("-" * 70) + RESET

    def test_hr_custom(self):
        assert _hr("=", 5) == DIM + "=====" + RESET

    def test_truncate_short_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_truncate_exact_length_unchanged(self):
        assert _truncate("hello", 5) == "hello"

    def test_truncate_long_appends_ellipsis(self):
        out = _truncate("abcdefghij", 5)
        assert out == "abcde" + DIM + "..." + RESET

    def test_truncate_replaces_newlines(self):
        assert _truncate("a\nb", 10) == "a\\nb"

    def test_truncate_non_str(self):
        assert _truncate(123, 10) == "123"

    def test_scaled_bar_zero_max(self):
        assert _scaled_bar(5, 0) == ""

    def test_scaled_bar_negative_max(self):
        assert _scaled_bar(5, -1) == ""

    def test_scaled_bar_full(self):
        assert _scaled_bar(10, 10, width=40) == "█" * 40

    def test_scaled_bar_min_one(self):
        assert _scaled_bar(1, 1000, width=40) == "█"

    def test_scaled_bar_half(self):
        assert _scaled_bar(5, 10, width=40) == "█" * 20


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_loads_records(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        assert load_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_accepts_str_path(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text('{"a": 1}\n', encoding="utf-8")
        assert load_jsonl(str(p)) == [{"a": 1}]

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
        assert load_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_invalid_lines_skipped_and_reported(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        p.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
        result = load_jsonl(p)
        assert result == [{"a": 1}, {"b": 2}]
        err = capsys.readouterr().err
        assert "Line 2 is invalid JSON" in err

    def test_json_extension_accepted(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text('{"a": 1}\n', encoding="utf-8")
        assert load_jsonl(p) == [{"a": 1}]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nope.jsonl")

    def test_bad_extension(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text('{"a": 1}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Unrecognized extension"):
            load_jsonl(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty or contains no valid records"):
            load_jsonl(p)

    def test_only_invalid_lines_raises(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text("nope\nbad\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_jsonl(p)


# ---------------------------------------------------------------------------
# convert_base64_to_pil_image
# ---------------------------------------------------------------------------

class TestConvertBase64ToPilImage:
    def test_bare_base64(self, tiny_png_b64):
        PIL = pytest.importorskip("PIL.Image")
        img = convert_base64_to_pil_image(tiny_png_b64)
        assert isinstance(img, PIL.Image)

    def test_data_uri(self, tiny_png_b64):
        PIL = pytest.importorskip("PIL.Image")
        uri = f"data:image/png;base64,{tiny_png_b64}"
        img = convert_base64_to_pil_image(uri)
        assert isinstance(img, PIL.Image)

    def test_invalid_payload_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not decode base64 image"):
            convert_base64_to_pil_image("not-an-image!!!")


# ---------------------------------------------------------------------------
# _pil_to_data_uri
# ---------------------------------------------------------------------------

class TestPilToDataUri:
    def test_roundtrip_png(self, tiny_png_b64):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.open(BytesIO(base64.b64decode(tiny_png_b64)))
        uri = _pil_to_data_uri(img)
        assert uri.startswith("data:image/png;base64,")
        _, _, b64 = uri.partition(",")
        decoded = base64.b64decode(b64)
        assert decoded.startswith(b"\x89PNG")

    def test_none_format_defaults_to_png(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (1, 1))
        img.format = None
        uri = _pil_to_data_uri(img)
        assert uri.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# messages_to_json_safe
# ---------------------------------------------------------------------------

class TestMessagesToJsonSafe:
    def test_string_content_untouched(self):
        msgs = [{"role": "user", "content": "hello"}]
        out = messages_to_json_safe(msgs)
        assert out == [{"role": "user", "content": "hello"}]
        assert out is not msgs

    def test_missing_content_key(self):
        msgs = [{"role": "user"}]
        out = messages_to_json_safe(msgs)
        assert out == [{"role": "user", "content": None}]

    def test_text_parts_preserved(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        out = messages_to_json_safe(msgs)
        assert out == msgs

    def test_pil_image_converted_to_data_uri(self, tiny_png_bytes):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.open(BytesIO(tiny_png_bytes))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        out = messages_to_json_safe(msgs)
        part = out[0]["content"][0]
        assert part["type"] == "image"
        assert part["image"].startswith("data:image/")
        assert "base64," in part["image"]

    def test_image_part_without_image_key_preserved(self):
        msgs = [{"role": "user", "content": [{"type": "image"}]}]
        out = messages_to_json_safe(msgs)
        assert out == msgs

    def test_image_part_with_non_pil_image_preserved(self):
        msgs = [{"role": "user", "content": [{"type": "image", "image": "not-a-pil"}]}]
        out = messages_to_json_safe(msgs)
        assert out[0]["content"][0]["image"] == "not-a-pil"

    def test_does_not_mutate_original_list(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (1, 1))
        original_part = {"type": "image", "image": img}
        msgs = [{"role": "user", "content": [original_part]}]
        messages_to_json_safe(msgs)
        assert msgs[0]["content"][0] is original_part
        assert msgs[0]["content"][0]["image"] is img

    def test_extra_keys_preserved(self):
        msgs = [{"role": "user", "content": "hi", "name": "x"}]
        out = messages_to_json_safe(msgs)
        assert out == [{"role": "user", "content": "hi", "name": "x"}]

    def test_output_is_json_serializable(self, tiny_png_bytes):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.open(BytesIO(tiny_png_bytes))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        out = messages_to_json_safe(msgs)
        dumped = json.dumps(out)
        assert "data:image/" in dumped


# ---------------------------------------------------------------------------
# merge_content_parts
# ---------------------------------------------------------------------------

class TestMergeContentParts:
    def test_extends_empty(self):
        existing = []
        merge_content_parts(existing, [{"type": "text", "text": "a"}])
        assert existing == [{"type": "text", "text": "a"}]

    def test_adjacent_text_joined_with_newline(self):
        existing = [{"type": "text", "text": "a"}]
        merge_content_parts(existing, [{"type": "text", "text": "b"}])
        assert existing == [{"type": "text", "text": "a\nb"}]

    def test_non_text_not_joined(self):
        existing = [{"type": "image", "image": "x"}]
        merge_content_parts(existing, [{"type": "text", "text": "a"}])
        assert existing == [{"type": "image", "image": "x"}, {"type": "text", "text": "a"}]

    def test_copies_part_dicts(self):
        existing = []
        new = [{"type": "text", "text": "a"}]
        merge_content_parts(existing, new)
        existing[0]["text"] = "changed"
        assert new[0]["text"] == "a"

    def test_multiple_parts(self):
        existing = [{"type": "text", "text": "a"}]
        merge_content_parts(existing, [
            {"type": "text", "text": "b"},
            {"type": "image", "image": "x"},
            {"type": "text", "text": "c"},
        ])
        assert existing == [
            {"type": "text", "text": "a\nb"},
            {"type": "image", "image": "x"},
            {"type": "text", "text": "c"},
        ]

    def test_returns_none(self):
        existing = []
        result = merge_content_parts(existing, [])
        assert result is None


# ---------------------------------------------------------------------------
# finalize_content
# ---------------------------------------------------------------------------

class TestFinalizeContent:
    def test_text_only_returns_string(self):
        parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert finalize_content(parts) == "a\nb"

    def test_single_text(self):
        assert finalize_content([{"type": "text", "text": "x"}]) == "x"

    def test_empty_list_returns_empty_string(self):
        assert finalize_content([]) == ""

    def test_with_image_returns_list(self):
        parts = [{"type": "text", "text": "a"}, {"type": "image", "image": "x"}]
        assert finalize_content(parts) is parts

    def test_only_image_returns_list(self):
        parts = [{"type": "image", "image": "x"}]
        assert finalize_content(parts) is parts


# ---------------------------------------------------------------------------
# dget / dlist
# ---------------------------------------------------------------------------

class TestDget:
    def test_nested_dict(self):
        assert dget({"a": {"b": 1}}, "a") == {"b": 1}

    def test_missing_key(self):
        assert dget({}, "a") == {}

    def test_explicit_null(self):
        assert dget({"a": None}, "a") == {}

    def test_non_dict_value(self):
        assert dget({"a": 5}, "a") == {}

    def test_non_dict_parent(self):
        assert dget(None, "a") == {}
        assert dget("str", "a") == {}
        assert dget([], "a") == {}

    def test_returns_actual_dict_object(self):
        inner = {"x": 1}
        assert dget({"a": inner}, "a") is inner


class TestDlist:
    def test_list_value(self):
        assert dlist({"a": [1, 2]}, "a") == [1, 2]

    def test_missing_key(self):
        assert dlist({}, "a") == []

    def test_explicit_null(self):
        assert dlist({"a": None}, "a") == []

    def test_non_list_value(self):
        assert dlist({"a": "s"}, "a") == []
        assert dlist({"a": {"x": 1}}, "a") == []

    def test_non_dict_parent(self):
        assert dlist(None, "a") == []

    def test_returns_actual_list_object(self):
        inner = [1]
        assert dlist({"a": inner}, "a") is inner


# ---------------------------------------------------------------------------
# InspectorSchema
# ---------------------------------------------------------------------------

class TestInspectorSchema:
    def test_minimal_construction(self):
        s = _make_schema()
        assert s.label == "TEST"
        assert s.timestamp_of({"x": 1}) == ""
        assert s.role_colors == DEFAULT_ROLE_COLORS
        assert s.role_colors is not DEFAULT_ROLE_COLORS
        assert s.total_tokens_of is None
        assert s.tokens_of is None
        assert s.block_types_of is None
        assert s.blocks_of is None
        assert s.render_block is None

    def test_default_role_colors_are_copied(self):
        s1 = _make_schema()
        s2 = _make_schema()
        s1.role_colors["user"] = "changed"
        assert s2.role_colors["user"] == GREEN

    def test_optional_callables(self):
        s = _make_schema(
            total_tokens_of=lambda r: {"total": 1},
            tokens_of=lambda r: 2,
            block_types_of=lambda r: ["text"],
            blocks_of=lambda r: [],
            render_block=lambda b: "rendered",
        )
        assert s.total_tokens_of([]) == {"total": 1}
        assert s.tokens_of({}) == 2
        assert s.block_types_of({}) == ["text"]
        assert s.blocks_of({}) == []
        assert s.render_block({}) == "rendered"


# ---------------------------------------------------------------------------
# print_* functions
# ---------------------------------------------------------------------------

class TestPrintHeader:
    def test_basic_output(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x" * 2048, encoding="utf-8")
        schema = _make_schema(label="TEST")
        print_header(p, [{}, {}], schema)
        out = capsys.readouterr().out
        assert "TEST SESSION INSPECTOR" in out
        assert str(p) in out
        assert "2.0 KB" in out
        assert "Lines" in out


class TestPrintRecordTypes:
    def test_counts_and_bars(self, capsys):
        schema = _make_schema(record_type_of=lambda r: r["type"])
        records = [{"type": "a"}, {"type": "a"}, {"type": "b"}]
        print_record_types(records, schema)
        out = capsys.readouterr().out
        assert "RECORD TYPES" in out
        assert "a" in out
        assert "b" in out

    def test_empty_records(self, capsys):
        schema = _make_schema()
        print_record_types([], schema)
        out = capsys.readouterr().out
        assert "RECORD TYPES" in out


class TestPrintBlockSummary:
    def test_noop_without_block_types(self, capsys):
        schema = _make_schema()
        print_block_summary([{"type": "message", "role": "user"}], schema)
        assert capsys.readouterr().out == ""

    def test_summary_by_role(self, capsys):
        schema = _make_schema(
            block_types_of=lambda r: r.get("blocks", []),
        )
        records = [
            {"type": "message", "role": "user", "blocks": ["text", "text"]},
            {"type": "message", "role": "assistant", "blocks": ["tool_use"]},
        ]
        print_block_summary(records, schema)
        out = capsys.readouterr().out
        assert "CONTENT BLOCKS" in out
        assert "user" in out
        assert "assistant" in out
        assert "text" in out
        assert "tool_use" in out

    def test_non_message_uses_record_type(self, capsys):
        schema = _make_schema(
            block_types_of=lambda r: r.get("blocks", []),
        )
        records = [{"type": "metadata", "blocks": ["meta"]}]
        print_block_summary(records, schema)
        out = capsys.readouterr().out
        assert "metadata" in out
        assert "meta" in out

    def test_empty_records_noop(self, capsys):
        schema = _make_schema(block_types_of=lambda r: [])
        print_block_summary([], schema)
        assert capsys.readouterr().out == ""


class TestPrintMessageRoles:
    def test_no_messages_noop(self, capsys):
        schema = _make_schema()
        print_message_roles([{"type": "other"}], schema)
        assert capsys.readouterr().out == ""

    def test_roles_counted(self, capsys):
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
    def test_noop_without_callback(self, capsys):
        schema = _make_schema()
        print_token_usage([{}], schema)
        assert capsys.readouterr().out == ""

    def test_noop_when_empty_totals(self, capsys):
        schema = _make_schema(total_tokens_of=lambda r: {})
        print_token_usage([{}], schema)
        assert capsys.readouterr().out == ""

    def test_prints_totals(self, capsys):
        schema = _make_schema(total_tokens_of=lambda r: {"input_tokens": 1234})
        print_token_usage([{}], schema)
        out = capsys.readouterr().out
        assert "TOKEN USAGE" in out
        assert "Input Tokens" in out
        assert "1,234" in out


class TestPrintFlow:
    def test_no_messages_noop(self, capsys):
        schema = _make_schema()
        print_flow([{"type": "other"}], schema)
        assert capsys.readouterr().out == ""

    def test_flow_basic(self, capsys):
        schema = _make_schema()
        records = [
            {"type": "message", "role": "user", "text": "hi"},
            {"type": "message", "role": "assistant", "text": "yo"},
        ]
        print_flow(records, schema)
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" in out
        assert "[USER]" in out
        assert "[ASSISTANT]" in out

    def test_flow_with_text_truncated(self, capsys):
        schema = _make_schema()
        records = [{"type": "message", "role": "user", "text": "hello world"}]
        print_flow(records, schema, show_text=True)
        out = capsys.readouterr().out
        assert "hello world" in out

    def test_flow_with_blocks_rendered(self, capsys):
        schema = _make_schema(
            blocks_of=lambda r: [{"kind": "x"}],
            render_block=lambda b: "RENDERED-" + b["kind"],
        )
        records = [{"type": "message", "role": "user"}]
        print_flow(records, schema, show_text=True)
        out = capsys.readouterr().out
        assert "RENDERED-x" in out

    def test_flow_with_block_types_label(self, capsys):
        schema = _make_schema(block_types_of=lambda r: ["text", "image"])
        records = [{"type": "message", "role": "user"}]
        print_flow(records, schema)
        out = capsys.readouterr().out
        assert "blocks: text, image" in out

    def test_flow_with_tokens(self, capsys):
        schema = _make_schema(tokens_of=lambda r: 42)
        records = [{"type": "message", "role": "user"}]
        print_flow(records, schema)
        out = capsys.readouterr().out
        assert "42tk" in out

    def test_flow_zero_tokens_no_label(self, capsys):
        schema = _make_schema(tokens_of=lambda r: 0)
        records = [{"type": "message", "role": "user"}]
        print_flow(records, schema)
        out = capsys.readouterr().out
        assert "0tk" not in out


class TestPrintRawExamples:
    def test_one_per_type(self, capsys):
        schema = _make_schema(record_type_of=lambda r: r["type"])
        records = [
            {"type": "a", "x": 1},
            {"type": "a", "x": 2},
            {"type": "b", "y": 3},
        ]
        print_raw_examples(records, schema)
        out = capsys.readouterr().out
        assert out.count("type = a") == 1
        assert out.count("type = b") == 1

    def test_includes_flattened_text_for_messages(self, capsys):
        schema = _make_schema()
        records = [{"type": "message", "role": "user", "text": "hello callers"}]
        print_raw_examples(records, schema)
        out = capsys.readouterr().out
        assert "flattened text" in out
        assert "hello callers" in out


# ---------------------------------------------------------------------------
# run_inspection
# ---------------------------------------------------------------------------

class TestRunInspection:
    def test_basic(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x", encoding="utf-8")
        schema = _make_schema()
        records = [{"type": "message", "role": "user", "text": "hi"}]
        run_inspection(p, records, schema)
        out = capsys.readouterr().out
        assert "TEST SESSION INSPECTOR" in out
        assert "RECORD TYPES" in out
        assert "MESSAGE ROLES" in out

    def test_extra_header_invoked(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x", encoding="utf-8")
        schema = _make_schema()
        seen = {}

        def extra(records):
            seen["called"] = True
            seen["count"] = len(records)
            print("EXTRA-HEADER")

        run_inspection(p, [{"type": "message"}], schema, extra_header=extra)
        out = capsys.readouterr().out
        assert seen == {"called": True, "count": 1}
        assert "EXTRA-HEADER" in out

    def test_show_flow_with_blocks(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x", encoding="utf-8")
        schema = _make_schema(
            blocks_of=lambda r: [{"t": "z"}],
            render_block=lambda b: "BLOCK-" + b["t"],
        )
        run_inspection(
            p, [{"type": "message", "role": "user"}], schema,
            show_flow=True, show_blocks=True,
        )
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" in out
        assert "BLOCK-z" in out

    def test_show_blocks_without_flow_ignored(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x", encoding="utf-8")
        schema = _make_schema()
        run_inspection(p, [{"type": "message"}], schema, show_blocks=True)
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" not in out

    def test_accepts_str_path(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x", encoding="utf-8")
        schema = _make_schema()
        run_inspection(str(p), [], schema)
        out = capsys.readouterr().out
        assert "TEST SESSION INSPECTOR" in out
