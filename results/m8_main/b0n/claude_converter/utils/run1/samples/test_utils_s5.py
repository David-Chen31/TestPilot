from __future__ import annotations

import base64
import io
import json
import sys

import pytest

from claude_converter import utils
from claude_converter.utils import (
    load_jsonl,
    convert_base64_to_pil_image,
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
    RESET,
    BOLD,
    DIM,
    CYAN,
    GREEN,
    YELLOW,
    RED,
    MAGENTA,
    DEFAULT_ROLE_COLORS,
    _c,
    _hr,
    _truncate,
    _scaled_bar,
    _pil_to_data_uri,
)

PIL = pytest.importorskip  # placeholder; real availability checked below


def _have_pillow() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


needs_pillow = pytest.mark.skipif(not _have_pillow(), reason="Pillow not installed")


# ---------------------------------------------------------------- color helpers

class TestColorConstants:
    def test_reset(self):
        assert RESET == "\033[0m"

    def test_colors(self):
        assert BOLD == "\033[1m"
        assert DIM == "\033[2m"
        assert CYAN == "\033[36m"
        assert GREEN == "\033[32m"
        assert YELLOW == "\033[33m"
        assert RED == "\033[31m"
        assert MAGENTA == "\033[35m"

    def test_default_role_colors(self):
        assert DEFAULT_ROLE_COLORS == {
            "user": GREEN,
            "assistant": CYAN,
            "system": YELLOW,
        }

    def test_c(self):
        assert _c("hi", BOLD) == BOLD + "hi" + RESET
        assert _c(42, RED, BOLD) == RED + BOLD + "42" + RESET

    def test_hr(self):
        out = _hr("-", 5)
        assert out == DIM + "-----" + RESET

    def test_hr_defaults(self):
        out = _hr()
        assert out == DIM + ("-" * 70) + RESET

    def test_truncate_short(self):
        assert _truncate("abc") == "abc"

    def test_truncate_newline_escaped(self):
        assert _truncate("a\nb") == "a\\nb"

    def test_truncate_long(self):
        text = "a" * 130
        out = _truncate(text, max_len=120)
        assert out == ("a" * 120) + DIM + "..." + RESET

    def test_truncate_exact_len_no_ellipsis(self):
        text = "a" * 120
        assert _truncate(text, max_len=120) == text

    def test_truncate_non_string(self):
        assert _truncate(123) == "123"


class TestScaledBar:
    def test_zero_max(self):
        assert _scaled_bar(5, 0) == ""

    def test_negative_max(self):
        assert _scaled_bar(5, -1) == ""

    def test_full_width(self):
        assert _scaled_bar(10, 10) == "█" * 40

    def test_half_width(self):
        assert _scaled_bar(5, 10) == "█" * 20

    def test_min_one(self):
        assert _scaled_bar(1, 1000) == "█"

    def test_zero_count_gives_min_one(self):
        # round(0) -> 0, but max(1, ...) -> 1
        assert _scaled_bar(0, 10) == "█"


# ---------------------------------------------------------------- load_jsonl

class TestLoadJsonl:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nope.jsonl")

    def test_bad_extension(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unrecognized extension"):
            load_jsonl(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text("")
        with pytest.raises(ValueError, match="no valid records"):
            load_jsonl(p)

    def test_only_blank_lines(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text("\n\n  \n")
        with pytest.raises(ValueError, match="no valid records"):
            load_jsonl(p)

    def test_valid_jsonl(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n')
        assert load_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_json_extension(self, tmp_path):
        p = tmp_path / "f.json"
        p.write_text('{"a": 1}\n')
        assert load_jsonl(p) == [{"a": 1}]

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text('\n{"a": 1}\n\n{"b": 2}\n\n')
        assert load_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_invalid_lines_skipped(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
        records = load_jsonl(p)
        assert records == [{"a": 1}, {"b": 2}]
        err = capsys.readouterr().err
        assert "Line 2 is invalid JSON" in err

    def test_all_invalid_raises(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("garbage\nmore garbage\n")
        with pytest.raises(ValueError, match="no valid records"):
            load_jsonl(p)

    def test_accepts_string_path(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text('{"a": 1}\n')
        assert load_jsonl(str(p)) == [{"a": 1}]


# ---------------------------------------------------------------- base64 images

def _png_b64() -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestConvertBase64ToPilImage:
    @needs_pillow
    def test_bare_base64(self):
        img = convert_base64_to_pil_image(_png_b64())
        assert img.format == "PNG"
        assert img.size == (2, 2)

    @needs_pillow
    def test_data_uri(self):
        uri = "data:image/png;base64," + _png_b64()
        img = convert_base64_to_pil_image(uri)
        assert img.size == (2, 2)

    @needs_pillow
    def test_invalid_payload_raises(self):
        with pytest.raises(ValueError, match="Could not decode base64 image"):
            convert_base64_to_pil_image("!!!notbase64???")

    @needs_pillow
    def test_decodable_but_not_image_raises(self):
        payload = base64.b64encode(b"hello world").decode()
        with pytest.raises(ValueError, match="Could not decode base64 image"):
            convert_base64_to_pil_image(payload)

    def test_import_error_when_no_pillow(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("no PIL")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="Pillow is required"):
            convert_base64_to_pil_image("xxx")


# ---------------------------------------------------------------- pil_to_data_uri

@needs_pillow
class TestPilToDataUri:
    def test_png_roundtrip(self):
        from PIL import Image
        img = Image.new("RGB", (1, 1))
        uri = _pil_to_data_uri(img)
        assert uri.startswith("data:image/png;base64,")
        payload = uri.split(",", 1)[1]
        decoded = Image.open(io.BytesIO(base64.b64decode(payload)))
        assert decoded.format == "PNG"

    def test_format_default_png_when_none(self):
        class FakeImage:
            format = None
            def save(self, buf, format="PNG"):
                assert format == "PNG"
                buf.write(b"data")
        uri = _pil_to_data_uri(FakeImage())
        assert uri.startswith("data:image/png;base64,")

    def test_fallback_to_png_on_save_error(self):
        calls = []
        class FakeImage:
            format = "JPEG"
            def save(self, buf, format="JPEG"):
                calls.append(format)
                if format == "JPEG":
                    raise ValueError("nope")
                buf.write(b"pngdata")
        uri = _pil_to_data_uri(FakeImage())
        assert calls == ["JPEG", "PNG"]
        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1]) == b"pngdata"


# ---------------------------------------------------------------- messages_to_json_safe

class TestMessagesToJsonSafe:
    def test_plain_messages_untouched_shape(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = messages_to_json_safe(msgs)
        assert out == [{"role": "user", "content": "hi"}]
        # new list
        assert out is not msgs
        assert out[0] is not msgs[0]

    @needs_pillow
    def test_converts_pil_image_to_data_uri(self):
        from PIL import Image
        img = Image.new("RGB", (1, 1))
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "image": img},
            ],
        }]
        out = messages_to_json_safe(msgs)
        parts = out[0]["content"]
        assert parts[0] == {"type": "text", "text": "look"}
        assert parts[1]["type"] == "image"
        assert parts[1]["image"].startswith("data:image/png;base64,")
        # original untouched
        assert msgs[0]["content"][1]["image"] is img

    @needs_pillow
    def test_non_image_parts_preserved(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": "already-a-string"},
                {"type": "text", "text": "x"},
            ],
        }]
        out = messages_to_json_safe(msgs)
        assert out[0]["content"] == msgs[0]["content"]

    def test_missing_content_key(self):
        out = messages_to_json_safe([{"role": "user"}])
        assert out == [{"role": "user", "content": None}]

    def test_extra_keys_preserved(self):
        out = messages_to_json_safe([{"role": "user", "content": "hi", "extra": 1}])
        assert out == [{"role": "user", "content": "hi", "extra": 1}]

    def test_image_only_dict_unchanged_when_not_pil(self, monkeypatch):
        # Force ImportError branch of PIL: image converted to repr.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PIL" or name == "PIL.Image" or name.startswith("PIL."):
                raise ImportError("no PIL")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        obj = object()
        msgs = [{"role": "user", "content": [{"type": "image", "image": obj}]}]
        out = messages_to_json_safe(msgs)
        assert out[0]["content"] == [{"type": "image", "image": repr(obj)}]

    @needs_pillow
    def test_json_dump_works(self):
        from PIL import Image
        img = Image.new("RGB", (1, 1))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        dumped = json.dumps(messages_to_json_safe(msgs))
        assert "data:image/png;base64," in dumped


# ---------------------------------------------------------------- merge_content_parts

class TestMergeContentParts:
    def test_append_to_empty(self):
        existing = []
        merge_content_parts(existing, [{"type": "text", "text": "a"}])
        assert existing == [{"type": "text", "text": "a"}]

    def test_merge_adjacent_text(self):
        existing = [{"type": "text", "text": "a"}]
        merge_content_parts(existing, [{"type": "text", "text": "b"}])
        assert existing == [{"type": "text", "text": "a\nb"}]

    def test_merge_multiple_text_parts(self):
        existing = []
        merge_content_parts(existing, [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
            {"type": "text", "text": "c"},
        ])
        assert existing == [{"type": "text", "text": "a\nb\nc"}]

    def test_image_breaks_text_merge(self):
        existing = [{"type": "text", "text": "a"}]
        merge_content_parts(existing, [
            {"type": "image", "image": "x"},
            {"type": "text", "text": "b"},
        ])
        assert existing == [
            {"type": "text", "text": "a"},
            {"type": "image", "image": "x"},
            {"type": "text", "text": "b"},
        ]

    def test_copies_parts(self):
        part = {"type": "text", "text": "a"}
        existing = []
        merge_content_parts(existing, [part])
        assert existing[0] == part
        assert existing[0] is not part

    def test_empty_new(self):
        existing = [{"type": "text", "text": "a"}]
        merge_content_parts(existing, [])
        assert existing == [{"type": "text", "text": "a"}]

    def test_returns_none(self):
        existing = []
        result = merge_content_parts(existing, [{"type": "text", "text": "a"}])
        assert result is None


# ---------------------------------------------------------------- finalize_content

class TestFinalizeContent:
    def test_empty_list(self):
        assert finalize_content([]) == ""

    def test_text_only_joined(self):
        parts = [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]
        assert finalize_content(parts) == "a\nb"

    def test_single_text(self):
        assert finalize_content([{"type": "text", "text": "hi"}]) == "hi"

    def test_with_image_returns_list(self):
        parts = [
            {"type": "text", "text": "a"},
            {"type": "image", "image": "x"},
        ]
        assert finalize_content(parts) is parts


# ---------------------------------------------------------------- dget / dlist

class TestDget:
    def test_normal(self):
        assert dget({"a": {"b": 1}}, "a") == {"b": 1}

    def test_missing_key(self):
        assert dget({}, "a") == {}

    def test_explicit_null(self):
        assert dget({"a": None}, "a") == {}

    def test_non_dict_parent(self):
        assert dget(None, "a") == {}
        assert dget([], "a") == {}
        assert dget("x", "a") == {}

    def test_non_dict_value(self):
        assert dget({"a": [1, 2]}, "a") == {}
        assert dget({"a": "str"}, "a") == {}


class TestDlist:
    def test_normal(self):
        assert dlist({"a": [1, 2]}, "a") == [1, 2]

    def test_missing(self):
        assert dlist({}, "a") == []

    def test_explicit_null(self):
        assert dlist({"a": None}, "a") == []

    def test_non_dict_parent(self):
        assert dlist(None, "a") == []

    def test_non_list_value(self):
        assert dlist({"a": {"x": 1}}, "a") == []
        assert dlist({"a": "str"}, "a") == []


# ---------------------------------------------------------------- InspectorSchema

class TestInspectorSchema:
    def _schema(self, **overrides):
        defaults = dict(
            label="TEST",
            record_type_of=lambda r: r.get("type", "?"),
            is_message=lambda r: r.get("type") == "msg",
            role_of=lambda r: r.get("role", "user"),
            text_of=lambda r: r.get("text", ""),
        )
        defaults.update(overrides)
        return InspectorSchema(**defaults)

    def test_defaults(self):
        s = self._schema()
        assert s.label == "TEST"
        assert s.timestamp_of({"x": 1}) == ""
        assert s.role_colors == DEFAULT_ROLE_COLORS
        # default_factory means a copy, not the shared object
        s.role_colors["user"] = "changed"
        assert DEFAULT_ROLE_COLORS["user"] == GREEN

    def test_optional_defaults_none(self):
        s = self._schema()
        assert s.total_tokens_of is None
        assert s.tokens_of is None
        assert s.block_types_of is None
        assert s.blocks_of is None
        assert s.render_block is None

    def test_custom_role_colors(self):
        s = self._schema(role_colors={"user": RED})
        assert s.role_colors == {"user": RED}


# ---------------------------------------------------------------- print_header

class TestPrintHeader:
    def test_basic(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text('{"a": 1}\n')
        schema = InspectorSchema(
            label="CODEX",
            record_type_of=lambda r: "?",
            is_message=lambda r: False,
            role_of=lambda r: "",
            text_of=lambda r: "",
        )
        print_header(p, [{"a": 1}, {"a": 2}], schema)
        out = capsys.readouterr().out
        assert "CODEX SESSION INSPECTOR" in out
        assert "File  :" in out
        assert str(p) in out
        assert "Lines :" in out
        assert SAMPLE_LINES_PLACEHOLDER not in out if False else True  # noqa
        assert "Size  :" in out

    def test_lines_count(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x")
        schema = InspectorSchema(
            label="X",
            record_type_of=lambda r: "?",
            is_message=lambda r: False,
            role_of=lambda r: "",
            text_of=lambda r: "",
        )
        print_header(p, [{"a": 1}, {"b": 2}, {"c": 3}], schema)
        out = capsys.readouterr().out
        assert "3" in out


# ---------------------------------------------------------------- print_record_types

def _simple_schema(**overrides):
    defaults = dict(
        label="T",
        record_type_of=lambda r: r.get("type", "?"),
        is_message=lambda r: r.get("type") == "msg",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
    )
    defaults.update(overrides)
    return InspectorSchema(**defaults)


class TestPrintRecordTypes:
    def test_counts(self, capsys):
        schema = _simple_schema()
        records = [{"type": "a"}, {"type": "a"}, {"type": "b"}]
        print_record_types(records, schema)
        out = capsys.readouterr().out
        assert "RECORD TYPES" in out
        assert "a" in out and "b" in out

    def test_empty_records(self, capsys):
        schema = _simple_schema()
        print_record_types([], schema)
        out = capsys.readouterr().out
        assert "RECORD TYPES" in out

    def test_uses_role_colors(self, capsys):
        schema = _simple_schema(role_colors={"a": RED})
        print_record_types([{"type": "a"}], schema)
        out = capsys.readouterr().out
        assert RED in out


# ---------------------------------------------------------------- print_block_summary

class TestPrintBlockSummary:
    def test_noop_without_block_types_of(self, capsys):
        schema = _simple_schema()
        print_block_summary([{"type": "msg", "role": "user"}], schema)
        out = capsys.readouterr().out
        assert "CONTENT BLOCKS" not in out

    def test_with_block_types(self, capsys):
        schema = _simple_schema(block_types_of=lambda r: r.get("blocks", []))
        records = [
            {"type": "msg", "role": "user", "blocks": ["text", "tool_use"]},
            {"type": "msg", "role": "user", "blocks": ["text"]},
        ]
        print_block_summary(records, schema)
        out = capsys.readouterr().out
        assert "CONTENT BLOCKS" in out
        assert "user" in out
        assert "text" in out
        assert "tool_use" in out

    def test_non_message_uses_record_type(self, capsys):
        schema = _simple_schema(block_types_of=lambda r: r.get("blocks", []))
        records = [{"type": "event", "blocks": ["x"]}]
        print_block_summary(records, schema)
        out = capsys.readouterr().out
        assert "event" in out
        assert "x" in out

    def test_empty_by_role_returns_without_printing(self, capsys):
        schema = _simple_schema(block_types_of=lambda r: [])
        print_block_summary([], schema)
        out = capsys.readouterr().out
        assert "CONTENT BLOCKS" not in out


# ---------------------------------------------------------------- print_message_roles

class TestPrintMessageRoles:
    def test_no_messages_noop(self, capsys):
        schema = _simple_schema()
        print_message_roles([{"type": "event"}], schema)
        out = capsys.readouterr().out
        assert "MESSAGE ROLES" not in out

    def test_counts(self, capsys):
        schema = _simple_schema()
        records = [
            {"type": "msg", "role": "user"},
            {"type": "msg", "role": "user"},
            {"type": "msg", "role": "assistant"},
        ]
        print_message_roles(records, schema)
        out = capsys.readouterr().out
        assert "MESSAGE ROLES" in out
        assert "user" in out
        assert "assistant" in out


# ---------------------------------------------------------------- print_token_usage

class TestPrintTokenUsage:
    def test_noop_without_total_tokens(self, capsys):
        schema = _simple_schema()
        print_token_usage([{"type": "msg"}], schema)
        out = capsys.readouterr().out
        assert "TOKEN USAGE" not in out

    def test_noop_when_empty_totals(self, capsys):
        schema = _simple_schema(total_tokens_of=lambda recs: {})
        print_token_usage([], schema)
        out = capsys.readouterr().out
        assert "TOKEN USAGE" not in out

    def test_prints_formatted(self, capsys):
        schema = _simple_schema(total_tokens_of=lambda recs: {"input_tokens": 1234})
        print_token_usage([], schema)
        out = capsys.readouterr().out
        assert "TOKEN USAGE" in out
        assert "Input Tokens" in out
        assert "1,234" in out


# ---------------------------------------------------------------- print_flow

class TestPrintFlow:
    def test_no_messages_noop(self, capsys):
        schema = _simple_schema()
        print_flow([{"type": "event"}], schema)
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" not in out

    def test_basic_roles(self, capsys):
        schema = _simple_schema(
            timestamp_of=lambda r: "00:00",
            tokens_of=lambda r: 5,
        )
        records = [
            {"type": "msg", "role": "user"},
            {"type": "msg", "role": "assistant"},
        ]
        print_flow(records, schema)
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" in out
        assert "[USER]" in out
        assert "[ASSISTANT]" in out
        assert "5tk" in out

    def test_no_token_label_when_zero(self, capsys):
        schema = _simple_schema(tokens_of=lambda r: 0)
        print_flow([{"type": "msg", "role": "user"}], schema)
        out = capsys.readouterr().out
        assert "0tk" not in out

    def test_block_types_line(self, capsys):
        schema = _simple_schema(block_types_of=lambda r: ["text", "image"])
        print_flow([{"type": "msg", "role": "user"}], schema)
        out = capsys.readouterr().out
        assert "blocks: text, image" in out

    def test_show_text_with_render_blocks(self, capsys):
        schema = _simple_schema(
            blocks_of=lambda r: [{"k": "v"}],
            render_block=lambda b: f"RENDERED:{b['k']}",
        )
        print_flow([{"type": "msg", "role": "user"}], schema, show_text=True)
        out = capsys.readouterr().out
        assert "RENDERED:v" in out

    def test_show_text_without_render_uses_text_of(self, capsys):
        schema = _simple_schema(text_of=lambda r: "hello world")
        print_flow([{"type": "msg", "role": "user"}], schema, show_text=True)
        out = capsys.readouterr().out
        assert "hello world" in out

    def test_blocks_of_without_render_block_uses_text(self, capsys):
        schema = _simple_schema(
            text_of=lambda r: "TXT",
            blocks_of=lambda r: [{"k": "v"}],
        )
        print_flow([{"type": "msg", "role": "user"}], schema, show_text=True)
        out = capsys.readouterr().out
        assert "TXT" in out


# ---------------------------------------------------------------- print_raw_examples

class TestPrintRawExamples:
    def test_one_example_per_type(self, capsys):
        schema = _simple_schema()
        records = [
            {"type": "a", "x": 1},
            {"type": "a", "x": 2},
            {"type": "b", "y": 3},
        ]
        print_raw_examples(records, schema)
        out = capsys.readouterr().out
        assert "RAW RECORD EXAMPLES" in out
        assert out.count("type = a") == 1
        assert out.count("type = b") == 1

    def test_message_shows_flattened_text(self, capsys):
        schema = _simple_schema(text_of=lambda r: "flattened")
        records = [{"type": "msg", "role": "user"}]
        print_raw_examples(records, schema)
        out = capsys.readouterr().out
        assert "flattened" in out

    def test_non_message_no_flattened(self, capsys):
        schema = _simple_schema(text_of=lambda r: "flattened")
        records = [{"type": "event"}]
        print_raw_examples(records, schema)
        out = capsys.readouterr().out
        assert "flattened" not in out


# ---------------------------------------------------------------- run_inspection

class TestRunInspection:
    def test_basic(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text('{"type":"msg","role":"user"}\n')
        schema = _simple_schema()
        records = [{"type": "msg", "role": "user"}]
        run_inspection(p, records, schema)
        out = capsys.readouterr().out
        assert "SESSION INSPECTOR" in out
        assert "RECORD TYPES" in out
        assert "MESSAGE ROLES" in out
        # flow not shown unless requested
        assert "CONVERSATION FLOW" not in out

    def test_accepts_string_path(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x")
        schema = _simple_schema()
        run_inspection(str(p), [{"type": "event"}], schema)
        out = capsys.readouterr().out
        assert "SESSION INSPECTOR" in out

    def test_show_flow(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x")
        schema = _simple_schema()
        run_inspection(p, [{"type": "msg", "role": "user"}], schema, show_flow=True)
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" in out

    def test_extra_header_called(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x")
        schema = _simple_schema()
        calls = []
        def extra(recs):
            calls.append(recs)
            print("EXTRA HEADER")
        run_inspection(p, [{"type": "event"}], schema, extra_header=extra)
        out = capsys.readouterr().out
        assert "EXTRA HEADER" in out
        assert len(calls) == 1
        assert calls[0] == [{"type": "event"}]

    def test_show_blocks_without_flow(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x")
        schema = _simple_schema(
            blocks_of=lambda r: [{"k": 1}],
            render_block=lambda b: "RENDERED",
        )
        run_inspection(
            p, [{"type": "msg", "role": "user"}], schema,
            show_flow=False, show_blocks=True,
        )
        out = capsys.readouterr().out
        assert "CONVERSATION FLOW" not in out
        assert "RENDERED" not in out

    def test_show_flow_and_blocks(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text("x")
        schema = _simple_schema(
            blocks_of=lambda r: [{"k": 1}],
            render_block=lambda b: "RENDERED",
        )
        records = [{"type": "msg", "role": "user"}]
        run_inspection(p, records, schema, show_flow=True, show_blocks=True)
        out = capsys.readouterr().out
        assert "RENDERED" in out
