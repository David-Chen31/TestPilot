from __future__ import annotations

import base64
import io
import json
import sys

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

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


# A tiny valid PNG (1x1) as raw bytes.
def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


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


# --------------------------------------------------------------------------
# Constants / simple helpers
# --------------------------------------------------------------------------

def test_constants():
    assert RESET == "\033[0m"
    assert BOLD == "\033[1m"
    assert DIM == "\033[2m"
    assert CYAN == "\033[36m"
    assert GREEN == "\033[32m"
    assert YELLOW == "\033[33m"
    assert DEFAULT_ROLE_COLORS == {
        "user": GREEN,
        "assistant": CYAN,
        "system": YELLOW,
    }


# --------------------------------------------------------------------------
# load_jsonl
# --------------------------------------------------------------------------

def test_load_jsonl_basic(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_str_path(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    assert load_jsonl(str(p)) == [{"a": 1}]


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('\n{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_skips_invalid_json_and_warns(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
    result = load_jsonl(p)
    assert result == [{"a": 1}, {"b": 2}]
    err = capsys.readouterr().err
    assert "Line 2 is invalid JSON" in err


def test_load_jsonl_json_extension_accepted(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}]


def test_load_jsonl_missing_file(tmp_path):
    p = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_jsonl(p)


def test_load_jsonl_bad_extension(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_jsonl(p)


def test_load_jsonl_empty_file(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or contains no valid records"):
        load_jsonl(p)


def test_load_jsonl_all_invalid(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("bad\nbad2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or contains no valid records"):
        load_jsonl(p)


# --------------------------------------------------------------------------
# convert_base64_to_pil_image
# --------------------------------------------------------------------------

def test_convert_base64_bare_string():
    b64 = base64.b64encode(_png_bytes()).decode()
    img = convert_base64_to_pil_image(b64)
    assert isinstance(img, Image.Image)
    assert img.format == "PNG"
    assert img.size == (2, 2)


def test_convert_base64_data_uri():
    b64 = base64.b64encode(_png_bytes()).decode()
    uri = f"data:image/png;base64,{b64}"
    img = convert_base64_to_pil_image(uri)
    assert isinstance(img, Image.Image)
    assert img.size == (2, 2)


def test_convert_base64_invalid():
    with pytest.raises(ValueError, match="Could not decode base64 image"):
        convert_base64_to_pil_image("not-valid-base64!!!")


# --------------------------------------------------------------------------
# messages_to_json_safe
# --------------------------------------------------------------------------

def test_messages_to_json_safe_text_unchanged():
    msgs = [{"role": "user", "content": "hello"}]
    out = messages_to_json_safe(msgs)
    assert out == [{"role": "user", "content": "hello"}]
    # returns copies of the message dicts
    assert out[0] is not msgs[0]


def test_messages_to_json_safe_converts_image_part():
    img = Image.new("RGB", (1, 1), (0, 0, 255))
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
    uri = parts[1]["image"]
    assert uri.startswith("data:image/")
    assert ";base64," in uri
    # ensure the produced string is JSON-serializable and decodes to same bytes
    json.dumps(out)
    b64 = uri.split(",", 1)[1]
    decoded = base64.b64decode(b64)
    assert decoded == _png_bytes_for(img)


def _png_bytes_for(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_messages_to_json_safe_non_image_part_kept():
    part = {"type": "tool_use", "id": "x"}
    msgs = [{"role": "assistant", "content": [part]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [part]


def test_messages_to_json_safe_missing_content():
    msgs = [{"role": "user"}]
    out = messages_to_json_safe(msgs)
    assert out == [{"role": "user", "content": None}]


def test_messages_to_json_safe_non_pil_image_object():
    # An "image" part whose value isn't a PIL image is left as-is.
    msgs = [{"role": "user", "content": [{"type": "image", "image": "rawstring"}]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [{"type": "image", "image": "rawstring"}]


# --------------------------------------------------------------------------
# merge_content_parts
# --------------------------------------------------------------------------

def test_merge_content_parts_joins_adjacent_text():
    existing = [{"type": "text", "text": "hello"}]
    new = [{"type": "text", "text": "world"}]
    merge_content_parts(existing, new)
    assert existing == [{"type": "text", "text": "hello\nworld"}]


def test_merge_content_parts_appends_image():
    existing = [{"type": "text", "text": "a"}]
    new = [{"type": "image", "image": "x"}]
    merge_content_parts(existing, new)
    assert existing == [
        {"type": "text", "text": "a"},
        {"type": "image", "image": "x"},
    ]


def test_merge_content_parts_empty_existing():
    existing: list = []
    new = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, new)
    assert existing == [{"type": "text", "text": "a"}]


def test_merge_content_parts_copies_new_parts():
    existing: list = []
    part = {"type": "image", "image": "x"}
    merge_content_parts(existing, [part])
    assert existing[0] == part
    assert existing[0] is not part


def test_merge_content_parts_text_after_image_appended():
    existing = [{"type": "image", "image": "x"}]
    new = [{"type": "text", "text": "hi"}]
    merge_content_parts(existing, new)
    assert existing == [
        {"type": "image", "image": "x"},
        {"type": "text", "text": "hi"},
    ]


def test_merge_content_parts_multiple_new():
    existing = [{"type": "text", "text": "a"}]
    new = [
        {"type": "text", "text": "b"},
        {"type": "text", "text": "c"},
    ]
    merge_content_parts(existing, new)
    assert existing == [{"type": "text", "text": "a\nb\nc"}]


# --------------------------------------------------------------------------
# finalize_content
# --------------------------------------------------------------------------

def test_finalize_content_text_only():
    parts = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
    assert finalize_content(parts) == "a\nb"


def test_finalize_content_with_image_returns_parts():
    parts = [
        {"type": "text", "text": "a"},
        {"type": "image", "image": "x"},
    ]
    assert finalize_content(parts) is parts


def test_finalize_content_empty_returns_empty_string():
    assert finalize_content([]) == ""


# --------------------------------------------------------------------------
# dget / dlist
# --------------------------------------------------------------------------

def test_dget_basic():
    assert dget({"a": {"b": 1}}, "a") == {"b": 1}


def test_dget_missing():
    assert dget({}, "a") == {}


def test_dget_explicit_null():
    assert dget({"a": None}, "a") == {}


def test_dget_non_dict_parent():
    assert dget(None, "a") == {}
    assert dget([1, 2], "a") == {}
    assert dget("str", "a") == {}


def test_dget_non_dict_value():
    assert dget({"a": 5}, "a") == {}


def test_dlist_basic():
    assert dlist({"a": [1, 2]}, "a") == [1, 2]


def test_dlist_missing():
    assert dlist({}, "a") == []


def test_dlist_explicit_null():
    assert dlist({"a": None}, "a") == []


def test_dlist_non_list_value():
    assert dlist({"a": {"b": 1}}, "a") == []


def test_dlist_non_dict_parent():
    assert dlist(None, "a") == []
    assert dlist([1], "a") == []


# --------------------------------------------------------------------------
# InspectorSchema
# --------------------------------------------------------------------------

def test_inspector_schema_defaults():
    schema = _make_schema()
    assert schema.label == "TEST"
    assert schema.timestamp_of({}) == ""
    assert schema.role_colors == DEFAULT_ROLE_COLORS
    assert schema.total_tokens_of is None
    assert schema.tokens_of is None
    assert schema.block_types_of is None
    assert schema.blocks_of is None
    assert schema.render_block is None


def test_inspector_schema_role_colors_independent():
    s1 = _make_schema()
    s2 = _make_schema()
    s1.role_colors["user"] = "x"
    assert s2.role_colors["user"] == GREEN


# --------------------------------------------------------------------------
# print_header
# --------------------------------------------------------------------------

def test_print_header(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    schema = _make_schema()
    print_header(p, [{"type": "a"}, {"type": "b"}], schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert str(p) in out
    assert "Lines :" in out
    assert "KB" in out


# --------------------------------------------------------------------------
# print_record_types
# --------------------------------------------------------------------------

def test_print_record_types(capsys):
    schema = _make_schema()
    records = [{"type": "a"}, {"type": "a"}, {"type": "b"}]
    print_record_types(records, schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out
    assert "a" in out and "b" in out


def test_print_record_types_empty(capsys):
    schema = _make_schema()
    print_record_types([], schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out


# --------------------------------------------------------------------------
# print_block_summary
# --------------------------------------------------------------------------

def test_print_block_summary_noop_without_block_types(capsys):
    schema = _make_schema()
    print_block_summary([{"type": "message", "role": "user"}], schema)
    assert capsys.readouterr().out == ""


def test_print_block_summary(capsys):
    schema = _make_schema(
        block_types_of=lambda r: r.get("blocks", []),
    )
    records = [
        {"type": "message", "role": "user", "blocks": ["text"]},
        {"type": "message", "role": "assistant", "blocks": ["text", "tool_use"]},
        {"type": "other", "blocks": ["text"]},
    ]
    print_block_summary(records, schema)
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" in out
    assert "assistant" in out
    assert "tool_use" in out


def test_print_block_summary_empty_records(capsys):
    schema = _make_schema(block_types_of=lambda r: [])
    print_block_summary([], schema)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# print_message_roles
# --------------------------------------------------------------------------

def test_print_message_roles(capsys):
    schema = _make_schema()
    records = [
        {"type": "message", "role": "user"},
        {"type": "message", "role": "assistant"},
        {"type": "message", "role": "user"},
        {"type": "other"},
    ]
    print_message_roles(records, schema)
    out = capsys.readouterr().out
    assert "MESSAGE ROLES" in out
    assert "user" in out
    assert "assistant" in out


def test_print_message_roles_none(capsys):
    schema = _make_schema()
    print_message_roles([{"type": "other"}], schema)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# print_token_usage
# --------------------------------------------------------------------------

def test_print_token_usage_none(capsys):
    schema = _make_schema()
    print_token_usage([], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_empty_totals(capsys):
    schema = _make_schema(total_tokens_of=lambda r: {})
    print_token_usage([], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage(capsys):
    schema = _make_schema(total_tokens_of=lambda r: {"total_tokens": 1234})
    print_token_usage([{}], schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" in out
    assert "Total Tokens" in out
    assert "1,234" in out


# --------------------------------------------------------------------------
# print_flow
# --------------------------------------------------------------------------

def test_print_flow_no_messages(capsys):
    schema = _make_schema()
    print_flow([{"type": "other"}], schema)
    assert capsys.readouterr().out == ""


def test_print_flow_basic(capsys):
    schema = _make_schema(
        timestamp_of=lambda r: "12:00",
        tokens_of=lambda r: r.get("tokens", 0),
    )
    records = [
        {"type": "message", "role": "user", "text": "hi", "tokens": 5},
        {"type": "message", "role": "assistant", "text": "yo", "tokens": 0},
    ]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "[USER]" in out
    assert "[ASSISTANT]" in out
    assert "5tk" in out
    assert "12:00" in out


def test_print_flow_show_text_truncated(capsys):
    schema = _make_schema()
    long_text = "x" * 300
    print_flow([{"type": "message", "role": "user", "text": long_text}], schema, show_text=True)
    out = capsys.readouterr().out
    assert "..." in out


def test_print_flow_show_blocks(capsys):
    schema = _make_schema(
        blocks_of=lambda r: [{"kind": "text"}],
        render_block=lambda b: "RENDERED-" + b["kind"],
        block_types_of=lambda r: ["text"],
    )
    print_flow([{"type": "message", "role": "user"}], schema, show_text=True)
    out = capsys.readouterr().out
    assert "RENDERED-text" in out
    assert "blocks: text" in out


def test_print_flow_block_types_only(capsys):
    schema = _make_schema(block_types_of=lambda r: ["text", "image"])
    print_flow([{"type": "message", "role": "user"}], schema)
    out = capsys.readouterr().out
    assert "blocks: text, image" in out


# --------------------------------------------------------------------------
# print_raw_examples
# --------------------------------------------------------------------------

def test_print_raw_examples(capsys):
    schema = _make_schema()
    records = [
        {"type": "message", "role": "user", "text": "hello", "extra": 1},
        {"type": "message", "role": "user", "text": "second"},
        {"type": "other", "x": 2},
    ]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    assert "type = message" in out
    assert "type = other" in out
    # Only one example of "message" shown
    assert out.count("type = message") == 1
    assert "flattened text:" in out
    assert "hello" in out


def test_print_raw_examples_non_message(capsys):
    schema = _make_schema()
    print_raw_examples([{"type": "other", "k": "v"}], schema)
    out = capsys.readouterr().out
    assert "type = other" in out
    assert "flattened text:" not in out


# --------------------------------------------------------------------------
# run_inspection
# --------------------------------------------------------------------------

def test_run_inspection_basic(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    schema = _make_schema()
    records = [{"type": "message", "role": "user", "text": "hi"}]
    run_inspection(p, records, schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert "RECORD TYPES" in out
    assert "MESSAGE ROLES" in out


def test_run_inspection_with_extra_header(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    schema = _make_schema()
    calls = []

    def extra(records):
        calls.append(records)
        print("EXTRA-HEADER-MARKER")

    run_inspection(p, [], schema, extra_header=extra)
    out = capsys.readouterr().out
    assert "EXTRA-HEADER-MARKER" in out
    assert calls == [[]]


def test_run_inspection_show_flow(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    schema = _make_schema()
    records = [{"type": "message", "role": "user", "text": "hi"}]
    run_inspection(p, records, schema, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out


def test_run_inspection_no_flow_by_default(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    schema = _make_schema()
    records = [{"type": "message", "role": "user", "text": "hi"}]
    run_inspection(p, records, schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" not in out
