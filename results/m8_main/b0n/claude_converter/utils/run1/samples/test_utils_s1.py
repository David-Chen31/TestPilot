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
    CYAN,
    GREEN,
    YELLOW,
    DEFAULT_ROLE_COLORS,
    _c,
    _hr,
    _truncate,
    _scaled_bar,
    _pil_to_data_uri,
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
)

PIL = pytest.importorskip("PIL")
from PIL import Image as PILImage


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_png_bytes(color=(255, 0, 0), size=(2, 2)) -> bytes:
    buf = BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _make_png_image(color=(0, 255, 0), size=(3, 3)):
    return PILImage.open(BytesIO(_make_png_bytes(color, size)))


# --------------------------------------------------------------------------
# _c / _hr / _truncate / _scaled_bar
# --------------------------------------------------------------------------

def test_c_wraps_with_codes_and_reset():
    assert _c("hi") == "hi" + RESET
    assert _c("hi", BOLD, CYAN) == BOLD + CYAN + "hi" + RESET
    assert _c(123, BOLD) == BOLD + "123" + RESET


def test_hr_default():
    assert _hr() == DIM + "-" * 70 + RESET


def test_hr_custom():
    assert _hr("=", width=5) == DIM + "=====" + RESET


def test_truncate_short_unchanged():
    assert _truncate("abc") == "abc"


def test_truncate_newlines_escaped():
    assert _truncate("a\nb") == "a\\nb"


def test_truncate_long_adds_ellipsis():
    long = "x" * 130
    result = _truncate(long, max_len=120)
    assert result == "x" * 120 + DIM + "..." + RESET


def test_truncate_custom_max_len():
    assert _truncate("abcdef", max_len=3) == "abc" + DIM + "..." + RESET


def test_scaled_bar_zero_max():
    assert _scaled_bar(5, 0) == ""


def test_scaled_bar_basic():
    assert _scaled_bar(1, 2, width=10) == "█" * 5
    assert _scaled_bar(2, 2, width=10) == "█" * 10


def test_scaled_bar_minimum_one():
    # Tiny ratio still yields at least one block.
    assert _scaled_bar(1, 1000, width=10) == "█"


# --------------------------------------------------------------------------
# load_jsonl
# --------------------------------------------------------------------------

def test_load_jsonl_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_jsonl(tmp_path / "nope.jsonl")


def test_load_jsonl_bad_extension(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("{}\n")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_jsonl(p)


def test_load_jsonl_valid(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n')
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_json_extension(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a": 1}\n{"b": 2}\n')
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_string_path(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n')
    assert load_jsonl(str(p)) == [{"a": 1}]


def test_load_jsonl_invalid_lines_skipped(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
    result = load_jsonl(p)
    assert result == [{"a": 1}, {"b": 2}]
    err = capsys.readouterr().err
    assert "Line 2 is invalid JSON" in err


def test_load_jsonl_all_invalid_raises(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text("garbage\nmore garbage\n")
    with pytest.raises(ValueError, match="no valid records"):
        load_jsonl(p)
    capsys.readouterr()


def test_load_jsonl_empty_file_raises(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="no valid records"):
        load_jsonl(p)


def test_load_jsonl_blank_lines_ignored(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n\n   \n")
    with pytest.raises(ValueError, match="no valid records"):
        load_jsonl(p)


# --------------------------------------------------------------------------
# convert_base64_to_pil_image
# --------------------------------------------------------------------------

def test_convert_base64_bare_string():
    b64 = base64.b64encode(_make_png_bytes()).decode()
    img = convert_base64_to_pil_image(b64)
    assert isinstance(img, PILImage.Image)
    assert img.size == (2, 2)


def test_convert_base64_data_uri():
    b64 = base64.b64encode(_make_png_bytes()).decode()
    uri = f"data:image/png;base64,{b64}"
    img = convert_base64_to_pil_image(uri)
    assert isinstance(img, PILImage.Image)


def test_convert_base64_invalid_raises():
    with pytest.raises(ValueError, match="Could not decode base64 image"):
        convert_base64_to_pil_image("!!!notvalid!!!")


def test_convert_base64_missing_pillow(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Pillow is required"):
        convert_base64_to_pil_image("abc")


# --------------------------------------------------------------------------
# _pil_to_data_uri
# --------------------------------------------------------------------------

def test_pil_to_data_uri_png():
    img = _make_png_image()
    uri = _pil_to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")
    payload = uri.split(",", 1)[1]
    decoded = base64.b64decode(payload)
    reopened = PILImage.open(BytesIO(decoded))
    assert reopened.size == img.size


def test_pil_to_data_uri_no_format_falls_back_to_png():
    # Freshly constructed Image has no format; saving as its default PNG works.
    img = PILImage.new("RGB", (2, 2), (1, 2, 3))
    uri = _pil_to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")


# --------------------------------------------------------------------------
# messages_to_json_safe
# --------------------------------------------------------------------------

def test_messages_to_json_safe_non_list_content_passthrough():
    msgs = [{"role": "user", "content": "hello"}]
    assert messages_to_json_safe(msgs) == [{"role": "user", "content": "hello"}]


def test_messages_to_json_safe_converts_image():
    img = _make_png_image()
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image", "image": img},
        ]}
    ]
    out = messages_to_json_safe(msgs)
    content = out[0]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"
    assert content[1]["image"].startswith("data:image/")
    # Original object untouched.
    assert isinstance(msgs[0]["content"][1]["image"], PILImage.Image)


def test_messages_to_json_safe_non_image_objects_kept():
    part = {"type": "image", "image": "already-a-string"}
    msgs = [{"role": "user", "content": [part]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"][0] == {"type": "image", "image": "already-a-string"}


def test_messages_to_json_safe_other_parts_untouched():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "x"},
        {"type": "tool_result", "content": "y"},
    ]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"][1] == {"type": "tool_result", "content": "y"}


def test_messages_to_json_safe_produces_json_serializable():
    img = _make_png_image()
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
    safe = messages_to_json_safe(msgs)
    dumped = json.dumps(safe)
    assert "data:image/png;base64," in dumped


def test_messages_to_json_safe_missing_content_key():
    out = messages_to_json_safe([{"role": "user"}])
    assert out[0]["role"] == "user"
    assert "content" in out[0]


def test_messages_to_json_safe_image_without_pillow(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no pillow")
        return real_import(name, *args, **kwargs)

    part = {"type": "image", "image": "someobj"}
    msgs = [{"role": "user", "content": [part]}]
    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = messages_to_json_safe(msgs)
    assert out[0]["content"][0]["type"] == "image"
    assert isinstance(out[0]["content"][0]["image"], str)


# --------------------------------------------------------------------------
# merge_content_parts
# --------------------------------------------------------------------------

def test_merge_content_parts_empty():
    existing = []
    merge_content_parts(existing, [{"type": "text", "text": "a"}])
    assert existing == [{"type": "text", "text": "a"}]


def test_merge_content_parts_joins_adjacent_text():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [{"type": "text", "text": "b"}])
    assert existing == [{"type": "text", "text": "a\nb"}]


def test_merge_content_parts_non_text_not_joined():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [{"type": "image", "image": "x"}])
    assert existing == [
        {"type": "text", "text": "a"},
        {"type": "image", "image": "x"},
    ]


def test_merge_content_parts_text_after_image_new_part():
    existing = [{"type": "image", "image": "x"}]
    merge_content_parts(existing, [{"type": "text", "text": "a"}])
    assert existing == [
        {"type": "image", "image": "x"},
        {"type": "text", "text": "a"},
    ]


def test_merge_content_parts_copies_new_parts():
    new_part = {"type": "text", "text": "a"}
    existing = []
    merge_content_parts(existing, [new_part])
    assert existing[0] is not new_part


def test_merge_content_parts_multiple():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [
        {"type": "text", "text": "b"},
        {"type": "image", "image": "i"},
        {"type": "text", "text": "c"},
    ])
    assert existing == [
        {"type": "text", "text": "a\nb"},
        {"type": "image", "image": "i"},
        {"type": "text", "text": "c"},
    ]


# --------------------------------------------------------------------------
# finalize_content
# --------------------------------------------------------------------------

def test_finalize_content_text_only():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert finalize_content(parts) == "a\nb"


def test_finalize_content_single_text():
    assert finalize_content([{"type": "text", "text": "a"}]) == "a"


def test_finalize_content_with_image_returns_list():
    parts = [
        {"type": "text", "text": "a"},
        {"type": "image", "image": "x"},
    ]
    assert finalize_content(parts) == parts


# --------------------------------------------------------------------------
# dget / dlist
# --------------------------------------------------------------------------

def test_dget_basic():
    assert dget({"a": {"b": 1}}, "a") == {"b": 1}


def test_dget_missing():
    assert dget({}, "a") == {}


def test_dget_null_value():
    assert dget({"a": None}, "a") == {}


def test_dget_non_dict_value():
    assert dget({"a": [1, 2]}, "a") == {}


def test_dget_non_dict_parent():
    assert dget(None, "a") == {}
    assert dget("string", "a") == {}
    assert dget([1, 2], "a") == {}


def test_dlist_basic():
    assert dlist({"a": [1, 2]}, "a") == [1, 2]


def test_dlist_missing():
    assert dlist({}, "a") == []


def test_dlist_null():
    assert dlist({"a": None}, "a") == []


def test_dlist_non_list():
    assert dlist({"a": {"b": 1}}, "a") == []


def test_dlist_non_dict_parent():
    assert dlist(None, "a") == []
    assert dlist("str", "a") == []


# --------------------------------------------------------------------------
# InspectorSchema
# --------------------------------------------------------------------------

def _basic_schema(**overrides) -> InspectorSchema:
    kwargs = dict(
        label="TEST",
        record_type_of=lambda r: r.get("type", "?"),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "?"),
        text_of=lambda r: str(r.get("text", "")),
    )
    kwargs.update(overrides)
    return InspectorSchema(**kwargs)


def test_inspector_schema_defaults():
    s = _basic_schema()
    assert s.label == "TEST"
    assert s.timestamp_of({"x": 1}) == ""
    assert s.role_colors == DEFAULT_ROLE_COLORS
    # role_colors default should be a distinct copy.
    assert s.role_colors is not DEFAULT_ROLE_COLORS
    assert s.total_tokens_of is None
    assert s.tokens_of is None
    assert s.block_types_of is None
    assert s.blocks_of is None
    assert s.render_block is None


def test_inspector_schema_custom_role_colors():
    colors = {"user": GREEN}
    s = _basic_schema(role_colors=colors)
    assert s.role_colors == colors


# --------------------------------------------------------------------------
# print_header
# --------------------------------------------------------------------------

def test_print_header(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n')
    schema = _basic_schema()
    print_header(p, [{}, {}], schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert "File  :" in out
    assert "Lines : " in out
    assert "KB" in out


# --------------------------------------------------------------------------
# print_record_types
# --------------------------------------------------------------------------

def test_print_record_types(capsys):
    schema = _basic_schema()
    records = [{"type": "message"}, {"type": "message"}, {"type": "event"}]
    print_record_types(records, schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out
    assert "message" in out
    assert "event" in out
    assert "█" in out


def test_print_record_types_empty(capsys):
    schema = _basic_schema()
    print_record_types([], schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out


# --------------------------------------------------------------------------
# print_block_summary
# --------------------------------------------------------------------------

def test_print_block_summary_no_block_types_is_noop(capsys):
    schema = _basic_schema()
    print_block_summary([{"type": "message", "role": "user"}], schema)
    assert capsys.readouterr().out == ""


def test_print_block_summary_prints(capsys):
    schema = _basic_schema(
        block_types_of=lambda r: r.get("blocks", []),
    )
    records = [
        {"type": "message", "role": "user", "blocks": ["text", "text"]},
        {"type": "message", "role": "assistant", "blocks": ["tool_use"]},
        {"type": "event", "blocks": ["x"]},
    ]
    print_block_summary(records, schema)
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" in out
    assert "user" in out
    assert "assistant" in out
    assert "event" in out
    assert "text" in out
    assert "tool_use" in out


def test_print_block_summary_empty_records(capsys):
    schema = _basic_schema(block_types_of=lambda r: [])
    print_block_summary([], schema)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# print_message_roles
# --------------------------------------------------------------------------

def test_print_message_roles(capsys):
    schema = _basic_schema()
    records = [
        {"type": "message", "role": "user"},
        {"type": "message", "role": "user"},
        {"type": "message", "role": "assistant"},
        {"type": "event"},
    ]
    print_message_roles(records, schema)
    out = capsys.readouterr().out
    assert "MESSAGE ROLES" in out
    assert "user" in out
    assert "assistant" in out


def test_print_message_roles_no_messages(capsys):
    schema = _basic_schema()
    print_message_roles([{"type": "event"}], schema)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# print_token_usage
# --------------------------------------------------------------------------

def test_print_token_usage_none_is_noop(capsys):
    schema = _basic_schema()
    print_token_usage([{}, {}], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_empty_totals_noop(capsys):
    schema = _basic_schema(total_tokens_of=lambda rs: {})
    print_token_usage([{}], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_prints(capsys):
    schema = _basic_schema(
        total_tokens_of=lambda rs: {"input_tokens": 1234, "output_tokens": 5}
    )
    print_token_usage([{}], schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" in out
    assert "Input Tokens" in out
    assert "1,234" in out
    assert "5" in out


# --------------------------------------------------------------------------
# print_flow
# --------------------------------------------------------------------------

def test_print_flow_no_messages_noop(capsys):
    schema = _basic_schema()
    print_flow([{"type": "event"}], schema)
    assert capsys.readouterr().out == ""


def test_print_flow_basic(capsys):
    schema = _basic_schema()
    records = [
        {"type": "message", "role": "user", "text": "hello"},
        {"type": "message", "role": "assistant", "text": "hi"},
    ]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "[USER]" in out
    assert "[ASSISTANT]" in out


def test_print_flow_show_text_uses_text_of(capsys):
    schema = _basic_schema()
    records = [{"type": "message", "role": "user", "text": "hello world"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "hello world" in out


def test_print_flow_show_text_no_render_uses_truncate(capsys):
    schema = _basic_schema()
    long = "z" * 300
    records = [{"type": "message", "role": "user", "text": long}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "z" * 200 in out


def test_print_flow_show_text_renders_blocks(capsys):
    schema = _basic_schema(
        blocks_of=lambda r: r.get("blocks", []),
        render_block=lambda b: f"<BLK:{b.get('t')}>",
    )
    records = [{
        "type": "message", "role": "user",
        "blocks": [{"t": "a"}, {"t": "b"}],
    }]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "<BLK:a>" in out
    assert "<BLK:b>" in out


def test_print_flow_show_text_blocks_requires_both(capsys):
    # Only blocks_of without render_block -> falls back to text_of.
    schema = _basic_schema(
        text_of=lambda r: "FALLBACK-TEXT",
        blocks_of=lambda r: [{"t": "a"}],
    )
    records = [{"type": "message", "role": "user"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "FALLBACK-TEXT" in out


def test_print_flow_block_types_shown(capsys):
    schema = _basic_schema(block_types_of=lambda r: ["text", "tool_use"])
    records = [{"type": "message", "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "blocks: text, tool_use" in out


def test_print_flow_tokens_label(capsys):
    schema = _basic_schema(tokens_of=lambda r: 42)
    records = [{"type": "message", "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "42tk" in out


def test_print_flow_tokens_zero_no_label(capsys):
    schema = _basic_schema(tokens_of=lambda r: 0)
    records = [{"type": "message", "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "tk" not in out.lower().replace("token", "")


# --------------------------------------------------------------------------
# print_raw_examples
# --------------------------------------------------------------------------

def test_print_raw_examples(capsys):
    schema = _basic_schema()
    records = [
        {"type": "message", "role": "user", "text": "hi", "k": 1},
        {"type": "message", "role": "user", "text": "again"},
        {"type": "event", "foo": "bar"},
    ]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    assert "type = message" in out
    assert "type = event" in out
    # Only one message example printed despite two messages.
    assert out.count("type = message") == 1
    assert "flattened text" in out


# --------------------------------------------------------------------------
# run_inspection
# --------------------------------------------------------------------------

def test_run_inspection_basic(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text('{"type": "message", "role": "user", "text": "hi"}\n')
    schema = _basic_schema()
    records = [{"type": "message", "role": "user", "text": "hi"}]
    run_inspection(p, records, schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert "RECORD TYPES" in out
    assert "MESSAGE ROLES" in out
    assert "CONVERSATION FLOW" not in out


def test_run_inspection_extra_header_called(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text("{}\n")
    schema = _basic_schema()
    calls = []

    def extra(records):
        calls.append(records)
        print("EXTRA-HEADER-MARKER")

    records = [{"type": "message", "role": "user"}]
    run_inspection(p, records, schema, extra_header=extra)
    out = capsys.readouterr().out
    assert calls == [records]
    assert "EXTRA-HEADER-MARKER" in out


def test_run_inspection_show_flow(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text("{}\n")
    schema = _basic_schema()
    records = [{"type": "message", "role": "user", "text": "hi there"}]
    run_inspection(p, records, schema, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "hi there" in out


def test_run_inspection_accepts_string_path(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text("{}\n")
    schema = _basic_schema()
    run_inspection(str(p), [{"type": "message", "role": "user"}], schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
