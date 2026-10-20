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


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

def test_load_jsonl_basic(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_accepts_str_path(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"x": 1}\n', encoding="utf-8")
    assert load_jsonl(str(p)) == [{"x": 1}]


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('\n{"a": 1}\n\n   \n{"b": 2}\n\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_invalid_lines_warn_to_stderr(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
    result = load_jsonl(p)
    assert result == [{"a": 1}, {"b": 2}]
    err = capsys.readouterr().err
    assert "Line 2 is invalid JSON" in err
    assert "X" in err


def test_load_jsonl_json_extension(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_missing_file(tmp_path):
    p = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_jsonl(p)


def test_load_jsonl_bad_extension(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_jsonl(p)


def test_load_jsonl_empty_file(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or contains no valid records"):
        load_jsonl(p)


def test_load_jsonl_only_invalid_lines(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("garbage\nalso garbage\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or contains no valid records"):
        load_jsonl(p)


# ---------------------------------------------------------------------------
# convert_base64_to_pil_image
# ---------------------------------------------------------------------------

PIL = pytest.importorskip("PIL.Image", reason="Pillow required")


def _make_png_bytes(color=(255, 0, 0)):
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (2, 2), color).save(buf, format="PNG")
    return buf.getvalue()


def test_convert_base64_bare_string():
    png = _make_png_bytes()
    b64 = base64.b64encode(png).decode()
    img = convert_base64_to_pil_image(b64)
    assert img.size == (2, 2)
    assert img.format == "PNG"


def test_convert_base64_data_uri():
    png = _make_png_bytes()
    b64 = base64.b64encode(png).decode()
    uri = f"data:image/png;base64,{b64}"
    img = convert_base64_to_pil_image(uri)
    assert img.size == (2, 2)


def test_convert_base64_invalid_payload_raises_value_error():
    with pytest.raises(ValueError, match="Could not decode base64 image"):
        convert_base64_to_pil_image("not-valid-base64-###")


def test_convert_base64_valid_b64_but_not_image():
    b64 = base64.b64encode(b"hello world, not an image").decode()
    with pytest.raises(ValueError, match="Could not decode base64 image"):
        convert_base64_to_pil_image(b64)


def test_convert_base64_importerror_scenario(monkeypatch):
    """Simulate Pillow being unavailable."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Pillow is required"):
        convert_base64_to_pil_image("whatever")


# ---------------------------------------------------------------------------
# messages_to_json_safe
# ---------------------------------------------------------------------------

def test_messages_to_json_safe_plain_text_unchanged():
    msgs = [{"role": "user", "content": "hi"}]
    out = messages_to_json_safe(msgs)
    assert out == [{"role": "user", "content": "hi"}]
    # original unmutated & distinct objects
    assert out[0] is not msgs[0]


def test_messages_to_json_safe_non_list_content_unchanged():
    msgs = [{"role": "user", "content": None}]
    assert messages_to_json_safe(msgs) == [{"role": "user", "content": None}]


def test_messages_to_json_safe_converts_pil_image():
    from PIL import Image

    img = Image.new("RGB", (3, 3), (1, 2, 3))
    img.format = "PNG"
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image", "image": img},
    ]}]
    out = messages_to_json_safe(msgs)
    content = out[0]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"
    assert content[1]["image"].startswith("data:image/png;base64,")
    assert isinstance(content[1]["image"], str)
    # json-serializable
    json.dumps(out)


def test_messages_to_json_safe_preserves_non_image_parts():
    msgs = [{"role": "assistant", "content": [
        {"type": "text", "text": "x"},
        {"type": "other", "value": 1},
    ]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [
        {"type": "text", "text": "x"},
        {"type": "other", "value": 1},
    ]


def test_messages_to_json_safe_leaves_non_pil_image_value():
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": "already-a-string"},
    ]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [{"type": "image", "image": "already-a-string"}]


def test_messages_to_json_safe_image_key_required():
    # type==image but no "image" key -> passed through unchanged
    part = {"type": "image"}
    msgs = [{"role": "user", "content": [part]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [part]


def test_messages_to_json_safe_missing_content_key():
    msgs = [{"role": "user"}]
    out = messages_to_json_safe(msgs)
    assert out == [{"role": "user", "content": None}]


def test_messages_to_json_safe_import_error_fallback(monkeypatch):
    from PIL import Image

    img = Image.new("RGB", (2, 2))
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"][0]["type"] == "image"
    assert isinstance(out[0]["content"][0]["image"], str)
    assert "PngImageFile" in out[0]["content"][0]["image"] or "Image" in out[0]["content"][0]["image"]


# ---------------------------------------------------------------------------
# merge_content_parts
# ---------------------------------------------------------------------------

def test_merge_content_parts_empty_existing():
    existing = []
    merge_content_parts(existing, [{"type": "text", "text": "a"}])
    assert existing == [{"type": "text", "text": "a"}]


def test_merge_content_parts_joins_adjacent_text():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [{"type": "text", "text": "b"}])
    assert existing == [{"type": "text", "text": "a\nb"}]


def test_merge_content_parts_distinct_text_parts_joined_chain():
    existing = []
    merge_content_parts(existing, [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
        {"type": "text", "text": "c"},
    ])
    assert existing == [{"type": "text", "text": "a\nb\nc"}]


def test_merge_content_parts_image_breaks_text():
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


def test_merge_content_parts_new_parts_are_copied():
    existing = []
    part = {"type": "text", "text": "a"}
    merge_content_parts(existing, [part])
    existing[0]["text"] = "changed"
    assert part["text"] == "a"


def test_merge_content_parts_returns_none():
    existing = []
    result = merge_content_parts(existing, [{"type": "text", "text": "x"}])
    assert result is None


# ---------------------------------------------------------------------------
# finalize_content
# ---------------------------------------------------------------------------

def test_finalize_content_text_only_returns_string():
    parts = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
    assert finalize_content(parts) == "a\nb"


def test_finalize_content_empty_list():
    # all() over empty is True -> "\n".join([]) == ""
    assert finalize_content([]) == ""


def test_finalize_content_with_image_returns_list():
    parts = [
        {"type": "text", "text": "a"},
        {"type": "image", "image": "x"},
    ]
    result = finalize_content(parts)
    assert result == parts
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# dget / dlist
# ---------------------------------------------------------------------------

def test_dget_basic():
    assert dget({"a": {"b": 1}}, "a") == {"b": 1}


def test_dget_missing_returns_empty():
    assert dget({"a": 1}, "b") == {}


def test_dget_null_returns_empty():
    assert dget({"a": None}, "a") == {}


def test_dget_value_not_dict_returns_empty():
    assert dget({"a": [1, 2]}, "a") == {}
    assert dget({"a": "str"}, "a") == {}
    assert dget({"a": 5}, "a") == {}


def test_dget_non_dict_parent():
    assert dget(None, "a") == {}
    assert dget([1, 2, 3], "a") == {}
    assert dget("string", "a") == {}


def test_dlist_basic():
    assert dlist({"a": [1, 2]}, "a") == [1, 2]


def test_dlist_missing_returns_empty_list():
    assert dlist({"a": 1}, "b") == []


def test_dlist_null_returns_empty_list():
    assert dlist({"a": None}, "a") == []


def test_dlist_non_list_returns_empty_list():
    assert dlist({"a": {"b": 1}}, "a") == []
    assert dlist({"a": "string"}, "a") == []


def test_dlist_non_dict_parent():
    assert dlist(None, "a") == []
    assert dlist([1, 2], "a") == []


# ---------------------------------------------------------------------------
# InspectorSchema
# ---------------------------------------------------------------------------

def _make_schema(**kwargs) -> InspectorSchema:
    defaults = dict(
        label="TEST",
        record_type_of=lambda r: r.get("type", "?"),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: str(r.get("text", "")),
    )
    defaults.update(kwargs)
    return InspectorSchema(**defaults)


def test_inspector_schema_defaults():
    schema = _make_schema()
    assert schema.label == "TEST"
    assert schema.timestamp_of({"a": 1}) == ""
    assert schema.role_colors == DEFAULT_ROLE_COLORS
    assert schema.role_colors is not DEFAULT_ROLE_COLORS
    assert schema.total_tokens_of is None
    assert schema.tokens_of is None
    assert schema.block_types_of is None
    assert schema.blocks_of is None
    assert schema.render_block is None


def test_inspector_schema_role_colors_default_isolation():
    s1 = _make_schema()
    s2 = _make_schema()
    s1.role_colors["user"] = "X"
    assert s2.role_colors["user"] == GREEN


# ---------------------------------------------------------------------------
# print functions
# ---------------------------------------------------------------------------

def _records_for_printing():
    return [
        {"type": "meta", "role": "system", "text": "info"},
        {"type": "message", "role": "user", "text": "hello there\nworld"},
        {"type": "message", "role": "assistant", "text": "hi back"},
        {"type": "message", "role": "assistant", "text": "another"},
    ]


def test_print_header(tmp_path, capsys):
    p = tmp_path / "session.jsonl"
    p.write_text("{}\n{}\n", encoding="utf-8")
    schema = _make_schema()
    records = [{"type": "x"}, {"type": "y"}]
    print_header(p, records, schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert str(p) in out
    assert "Lines :" in out
    assert "Size  :" in out
    assert "KB" in out


def test_print_record_types(capsys):
    schema = _make_schema()
    records = [
        {"type": "message"},
        {"type": "message"},
        {"type": "meta"},
    ]
    print_record_types(records, schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out
    assert "message" in out
    assert "meta" in out
    assert "█" in out  # bar rendering


def test_print_record_types_empty(capsys):
    schema = _make_schema()
    print_record_types([], schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out


def test_print_block_summary_noop_when_none(capsys):
    schema = _make_schema()
    print_block_summary([{"type": "message"}], schema)
    assert capsys.readouterr().out == ""


def test_print_block_summary_counts(capsys):
    schema = _make_schema(
        block_types_of=lambda r: r.get("blocks", []),
    )
    records = [
        {"type": "message", "role": "user", "blocks": ["text"]},
        {"type": "message", "role": "assistant", "blocks": ["text", "tool_use"]},
        {"type": "meta", "role": "system", "blocks": ["meta_block"]},
    ]
    print_block_summary(records, schema)
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" in out
    assert "text" in out
    assert "tool_use" in out
    assert "meta_block" in out
    assert "assistant" in out
    assert "user" in out
    assert "meta" in out


def test_print_block_summary_empty_records(capsys):
    schema = _make_schema(block_types_of=lambda r: [])
    print_block_summary([], schema)
    assert capsys.readouterr().out == ""


def test_print_message_roles(capsys):
    schema = _make_schema()
    records = _records_for_printing()
    print_message_roles(records, schema)
    out = capsys.readouterr().out
    assert "MESSAGE ROLES" in out
    assert "user" in out
    assert "assistant" in out


def test_print_message_roles_no_messages(capsys):
    schema = _make_schema()
    print_message_roles([{"type": "meta"}], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_none_schema(capsys):
    schema = _make_schema()
    print_token_usage([{"type": "m"}], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_empty_totals(capsys):
    schema = _make_schema(total_tokens_of=lambda rs: {})
    print_token_usage([{"type": "m"}], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_formats(capsys):
    schema = _make_schema(total_tokens_of=lambda rs: {"input_tokens": 1234, "output_tokens": 5})
    print_token_usage([], schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" in out
    assert "Input Tokens" in out
    assert "1,234" in out
    assert "Output Tokens" in out


def test_print_flow_no_messages(capsys):
    schema = _make_schema()
    print_flow([{"type": "meta"}], schema)
    assert capsys.readouterr().out == ""


def test_print_flow_basic(capsys):
    schema = _make_schema()
    records = _records_for_printing()
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "[USER]" in out
    assert "[ASSISTANT]" in out


def test_print_flow_show_text_truncated(capsys):
    schema = _make_schema()
    long_text = "x" * 300
    records = [{"type": "message", "role": "user", "text": long_text}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "x" * 200 in out
    assert "..." in out


def test_print_flow_show_text_short(capsys):
    schema = _make_schema()
    records = [{"type": "message", "role": "user", "text": "short"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "short" in out


def test_print_flow_tokens_label(capsys):
    schema = _make_schema(tokens_of=lambda r: 42)
    records = [{"type": "message", "role": "user", "text": "x"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "42tk" in out


def test_print_flow_tokens_zero_no_label(capsys):
    schema = _make_schema(tokens_of=lambda r: 0)
    records = [{"type": "message", "role": "user", "text": "x"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "0tk" not in out


def test_print_flow_block_types_line(capsys):
    schema = _make_schema(block_types_of=lambda r: ["text", "tool_use"])
    records = [{"type": "message", "role": "user", "text": "x"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "blocks: text, tool_use" in out


def test_print_flow_render_blocks(capsys):
    schema = _make_schema(
        blocks_of=lambda r: [{"kind": "text"}, {"kind": "image"}],
        render_block=lambda b: f"RENDER[{b['kind']}]",
    )
    records = [{"type": "message", "role": "user", "text": "ignored"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "RENDER[text]" in out
    assert "RENDER[image]" in out


def test_print_raw_examples(capsys):
    schema = _make_schema()
    records = [
        {"type": "message", "role": "user", "text": "one"},
        {"type": "message", "role": "user", "text": "two"},
        {"type": "meta", "role": "system", "text": "three"},
    ]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    # one per type: "message" and "meta" only
    assert out.count("type = message") == 1
    assert out.count("type = meta") == 1


def test_print_raw_examples_truncates_text(capsys):
    schema = _make_schema()
    long_text = "y" * 300
    records = [{"type": "message", "role": "user", "text": long_text}]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "y" * 150 in out
    assert "..." in out


# ---------------------------------------------------------------------------
# run_inspection
# ---------------------------------------------------------------------------

def test_run_inspection_full(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text('{"type":"message"}\n', encoding="utf-8")
    schema = _make_schema(
        total_tokens_of=lambda rs: {"total": 10},
        block_types_of=lambda r: ["text"],
    )
    records = _records_for_printing()
    run_inspection(p, records, schema, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert "RECORD TYPES" in out
    assert "CONTENT BLOCKS" in out
    assert "MESSAGE ROLES" in out
    assert "TOKEN USAGE" in out
    assert "CONVERSATION FLOW" in out


def test_run_inspection_accepts_str_path(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    schema = _make_schema()
    run_inspection(str(p), [{"type": "message", "role": "user"}], schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out


def test_run_inspection_extra_header(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    schema = _make_schema()
    calls = []

    def extra(records):
        calls.append(list(records))
        print("EXTRA-HEADER")

    records = [{"type": "message", "role": "user"}]
    run_inspection(p, records, schema, extra_header=extra)
    out = capsys.readouterr().out
    assert len(calls) == 1
    assert calls[0] == records
    assert "EXTRA-HEADER" in out


def test_run_inspection_no_flow(capsys, tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    schema = _make_schema()
    run_inspection(p, [{"type": "message", "role": "user"}], schema, show_flow=False)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" not in out
