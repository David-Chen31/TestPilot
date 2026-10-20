from __future__ import annotations

import base64
import json
import sys
import types
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
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_accepts_str_path(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    assert load_jsonl(str(p)) == [{"a": 1}]


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('\n{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_reports_invalid_json_to_stderr(tmp_path, capsys):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]
    err = capsys.readouterr().err
    assert "Line 2 is invalid JSON" in err


def test_load_jsonl_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_jsonl(tmp_path / "nope.jsonl")


def test_load_jsonl_bad_extension(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_jsonl(p)


def test_load_jsonl_accepts_json_extension(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}]


def test_load_jsonl_empty_file_raises(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or contains no valid records"):
        load_jsonl(p)


def test_load_jsonl_all_invalid_raises(tmp_path, capsys):
    p = tmp_path / "data.jsonl"
    p.write_text("garbage\nalso bad\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(p)
    # each invalid line reported
    assert capsys.readouterr().err.count("invalid JSON") == 2


def test_load_jsonl_non_dict_records_kept(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("[1, 2]\n", encoding="utf-8")
    assert load_jsonl(p) == [[1, 2]]


# ---------------------------------------------------------------------------
# convert_base64_to_pil_image
# ---------------------------------------------------------------------------

try:
    from PIL import Image  # noqa: E402
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    _HAS_PIL = False

needs_pil = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@needs_pil
def test_convert_base64_bare_string():
    raw = _make_png_bytes()
    b64 = base64.b64encode(raw).decode()
    img = convert_base64_to_pil_image(b64)
    assert isinstance(img, Image.Image)
    assert img.size == (2, 2)


@needs_pil
def test_convert_base64_data_uri():
    raw = _make_png_bytes()
    uri = "data:image/png;base64," + base64.b64encode(raw).decode()
    img = convert_base64_to_pil_image(uri)
    assert isinstance(img, Image.Image)
    assert img.size == (2, 2)


@needs_pil
def test_convert_base64_invalid_raises_value_error():
    with pytest.raises(ValueError, match="Could not decode"):
        convert_base64_to_pil_image("not-valid-base64!!!")


@needs_pil
def test_convert_base64_valid_b64_but_not_image():
    b64 = base64.b64encode(b"hello world").decode()
    with pytest.raises(ValueError, match="Could not decode"):
        convert_base64_to_pil_image(b64)


def test_convert_base64_missing_pillow_raises_import_error(monkeypatch):
    """When Pillow is unavailable, an ImportError with a helpful message is raised."""
    # Force `from PIL import Image` inside the function to fail by blocking the module.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("simulated missing PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Pillow is required"):
        convert_base64_to_pil_image("abc")


# ---------------------------------------------------------------------------
# messages_to_json_safe
# ---------------------------------------------------------------------------


def test_messages_to_json_safe_passthrough_text():
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    out = messages_to_json_safe(msgs)
    assert out == msgs
    # returns new list objects
    assert out is not msgs
    assert out[0] is not msgs[0]


def test_messages_to_json_safe_string_content_unchanged():
    msgs = [{"role": "user", "content": "text only"}]
    out = messages_to_json_safe(msgs)
    assert out == [{"role": "user", "content": "text only"}]


@needs_pil
def test_messages_to_json_safe_encodes_pil_image():
    img = Image.open(BytesIO(_make_png_bytes()))
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
    # decodable back
    b64 = parts[1]["image"].split(",", 1)[1]
    decoded = base64.b64decode(b64)
    assert decoded == _make_png_bytes()
    # original untouched
    assert msgs[0]["content"][1]["image"] is img


def test_messages_to_json_safe_non_pil_image_kept_when_pil_available():
    """When Pillow is importable, a non-PIL image value is left untouched."""
    if not _HAS_PIL:
        pytest.skip("Pillow not installed")
    msgs = [{"role": "user", "content": [{"type": "image", "image": "already-a-string"}]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [{"type": "image", "image": "already-a-string"}]


def test_messages_to_json_safe_non_pil_image_repr_when_pil_missing(monkeypatch):
    """When Pillow is NOT importable, any image value falls back to repr()."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("simulated missing PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    msgs = [{"role": "user", "content": [{"type": "image", "image": "already-a-string"}]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [{"type": "image", "image": repr("already-a-string")}]


def test_messages_to_json_safe_non_image_parts_kept():
    parts = [{"type": "text", "text": "x"}, {"type": "tool_use", "id": "1"}]
    msgs = [{"role": "user", "content": parts}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == parts


def test_messages_to_json_safe_image_without_image_key_kept():
    content = [{"type": "image"}]
    out = messages_to_json_safe([{"role": "user", "content": content}])
    assert out[0]["content"] == [{"type": "image"}]


@needs_pil
def test_messages_to_json_safe_is_json_dumpable():
    img = Image.open(BytesIO(_make_png_bytes()))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
    safe = messages_to_json_safe(msgs)
    s = json.dumps(safe)
    assert "data:image/png;base64," in s


def test_messages_to_json_safe_no_content_key():
    out = messages_to_json_safe([{"role": "user"}])
    assert out == [{"role": "user", "content": None}]


def test_messages_to_json_safe_preserves_other_keys():
    msgs = [{"role": "user", "content": "hi", "name": "bob"}]
    out = messages_to_json_safe(msgs)
    assert out == [{"role": "user", "content": "hi", "name": "bob"}]


def test_messages_to_json_safe_pil_missing_falls_back_to_repr(monkeypatch):
    """If PIL cannot be imported in the converter, the image is replaced by repr()."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("simulated missing PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = messages_to_json_safe([
        {"role": "user", "content": [{"type": "image", "image": "OBJ"}]},
    ])
    assert out[0]["content"] == [{"type": "image", "image": repr("OBJ")}]


# ---------------------------------------------------------------------------
# merge_content_parts
# ---------------------------------------------------------------------------


def test_merge_content_parts_appends_new():
    existing: list[dict] = []
    merge_content_parts(existing, [{"type": "text", "text": "hello"}])
    assert existing == [{"type": "text", "text": "hello"}]


def test_merge_content_parts_joins_adjacent_text():
    existing = [{"type": "text", "text": "hello"}]
    merge_content_parts(existing, [{"type": "text", "text": "world"}])
    assert existing == [{"type": "text", "text": "hello\nworld"}]


def test_merge_content_parts_does_not_join_text_with_image():
    existing = [{"type": "text", "text": "hello"}]
    merge_content_parts(existing, [{"type": "image", "image": "x"}])
    assert existing == [
        {"type": "text", "text": "hello"},
        {"type": "image", "image": "x"},
    ]


def test_merge_content_parts_does_not_join_image_then_text():
    existing = [{"type": "image", "image": "x"}]
    merge_content_parts(existing, [{"type": "text", "text": "hello"}])
    assert existing == [
        {"type": "image", "image": "x"},
        {"type": "text", "text": "hello"},
    ]


def test_merge_content_parts_copies_part_dicts():
    existing: list[dict] = []
    part = {"type": "text", "text": "hi"}
    merge_content_parts(existing, [part])
    assert existing[0] == part
    assert existing[0] is not part


def test_merge_content_parts_multiple_parts():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [
        {"type": "text", "text": "b"},
        {"type": "image", "image": "img"},
        {"type": "text", "text": "c"},
    ])
    assert existing == [
        {"type": "text", "text": "a\nb"},
        {"type": "image", "image": "img"},
        {"type": "text", "text": "c"},
    ]


# ---------------------------------------------------------------------------
# finalize_content
# ---------------------------------------------------------------------------


def test_finalize_content_text_only_returns_string():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert finalize_content(parts) == "a\nb"


def test_finalize_content_single_text_returns_string():
    assert finalize_content([{"type": "text", "text": "only"}]) == "only"


def test_finalize_content_empty_returns_empty_string():
    assert finalize_content([]) == ""


def test_finalize_content_with_image_returns_list():
    parts = [
        {"type": "text", "text": "a"},
        {"type": "image", "image": "x"},
    ]
    result = finalize_content(parts)
    assert isinstance(result, list)
    assert result == parts


# ---------------------------------------------------------------------------
# dget / dlist
# ---------------------------------------------------------------------------


def test_dget_returns_dict():
    assert dget({"a": {"b": 1}}, "a") == {"b": 1}


def test_dget_missing_key():
    assert dget({"a": 1}, "b") == {}


def test_dget_none_value():
    assert dget({"a": None}, "a") == {}


def test_dget_non_dict_value():
    assert dget({"a": [1, 2]}, "a") == {}


def test_dget_non_dict_parent():
    for bad in (None, [], "string", 5):
        assert dget(bad, "a") == {}


def test_dlist_returns_list():
    assert dlist({"a": [1, 2]}, "a") == [1, 2]


def test_dlist_missing():
    assert dlist({"a": 1}, "b") == []


def test_dlist_none_value():
    assert dlist({"a": None}, "a") == []


def test_dlist_non_list_value():
    assert dlist({"a": {"b": 1}}, "a") == []


def test_dlist_non_dict_parent():
    for bad in (None, [], "string", 5):
        assert dlist(bad, "a") == []


# ---------------------------------------------------------------------------
# InspectorSchema
# ---------------------------------------------------------------------------


def test_inspector_schema_defaults():
    schema = InspectorSchema(
        label="TEST",
        record_type_of=lambda r: r.get("type", "?"),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
    )
    assert schema.label == "TEST"
    assert schema.timestamp_of({}) == ""
    assert schema.role_colors == DEFAULT_ROLE_COLORS
    assert schema.role_colors is not DEFAULT_ROLE_COLORS
    assert schema.total_tokens_of is None
    assert schema.tokens_of is None
    assert schema.block_types_of is None
    assert schema.blocks_of is None
    assert schema.render_block is None


def test_inspector_schema_role_colors_independent():
    def make():
        return InspectorSchema(
            label="X",
            record_type_of=lambda r: "",
            is_message=lambda r: False,
            role_of=lambda r: "",
            text_of=lambda r: "",
        )
    a = make()
    b = make()
    a.role_colors["user"] = "red"
    assert b.role_colors["user"] == GREEN


# Common helpers for the print functions


def _simple_schema():
    return InspectorSchema(
        label="DEMO",
        record_type_of=lambda r: r.get("type", "unknown"),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
        timestamp_of=lambda r: r.get("ts", ""),
    )


# ---------------------------------------------------------------------------
# print_header
# ---------------------------------------------------------------------------


def test_print_header(capsys, tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("x", encoding="utf-8")
    schema = _simple_schema()
    print_header(p, [{"a": 1}, {"b": 2}], schema)
    out = capsys.readouterr().out
    assert "DEMO SESSION INSPECTOR" in out
    assert str(p) in out
    assert "Lines" in out
    assert "2" in out


# ---------------------------------------------------------------------------
# print_record_types
# ---------------------------------------------------------------------------


def test_print_record_types(capsys):
    schema = _simple_schema()
    records = [
        {"type": "message", "role": "user"},
        {"type": "message", "role": "assistant"},
        {"type": "message", "role": "user"},
    ]
    print_record_types(records, schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out
    assert "message" in out
    # scaled bar rendered (some blocks)
    assert "█" in out


def test_print_record_types_empty(capsys):
    schema = _simple_schema()
    print_record_types([], schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out
    assert "█" not in out


# ---------------------------------------------------------------------------
# print_block_summary
# ---------------------------------------------------------------------------


def test_print_block_summary_noop_without_block_types(capsys):
    schema = _simple_schema()
    records = [{"type": "message", "role": "user"}]
    print_block_summary(records, schema)
    assert capsys.readouterr().out == ""


def test_print_block_summary_renders(capsys):
    schema = InspectorSchema(
        label="X",
        record_type_of=lambda r: r.get("type", ""),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: "",
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


def test_print_block_summary_empty_records_noop(capsys):
    schema = InspectorSchema(
        label="X",
        record_type_of=lambda r: "",
        is_message=lambda r: True,
        role_of=lambda r: "user",
        text_of=lambda r: "",
        block_types_of=lambda r: [],
    )
    print_block_summary([], schema)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# print_message_roles
# ---------------------------------------------------------------------------


def test_print_message_roles(capsys):
    schema = _simple_schema()
    records = [
        {"type": "message", "role": "user"},
        {"type": "message", "role": "user"},
        {"type": "message", "role": "assistant"},
        {"type": "other"},
    ]
    print_message_roles(records, schema)
    out = capsys.readouterr().out
    assert "MESSAGE ROLES" in out
    assert "user" in out
    assert "assistant" in out


def test_print_message_roles_no_messages(capsys):
    schema = _simple_schema()
    print_message_roles([{"type": "other"}], schema)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# print_token_usage
# ---------------------------------------------------------------------------


def test_print_token_usage_noop_without_callback(capsys):
    schema = _simple_schema()
    print_token_usage([{"type": "message"}], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_empty_totals_noop(capsys):
    schema = InspectorSchema(
        label="X",
        record_type_of=lambda r: "",
        is_message=lambda r: True,
        role_of=lambda r: "",
        text_of=lambda r: "",
        total_tokens_of=lambda recs: {},
    )
    print_token_usage([], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_renders(capsys):
    schema = InspectorSchema(
        label="X",
        record_type_of=lambda r: "",
        is_message=lambda r: True,
        role_of=lambda r: "",
        text_of=lambda r: "",
        total_tokens_of=lambda recs: {"input_tokens": 1234, "output_tokens": 5},
    )
    print_token_usage([], schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" in out
    assert "Input Tokens" in out
    assert "1,234" in out
    assert "Output Tokens" in out


# ---------------------------------------------------------------------------
# print_flow
# ---------------------------------------------------------------------------


def test_print_flow_no_messages(capsys):
    schema = _simple_schema()
    print_flow([{"type": "other"}], schema)
    assert capsys.readouterr().out == ""


def test_print_flow_basic(capsys):
    schema = _simple_schema()
    records = [
        {"type": "message", "role": "user", "text": "hi", "ts": "00:00"},
        {"type": "message", "role": "assistant", "text": "hello", "ts": "00:01"},
    ]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "[USER]" in out
    assert "[ASSISTANT]" in out
    assert "00:00" in out


def test_print_flow_show_text_truncation(capsys):
    schema = _simple_schema()
    long_text = "x" * 500
    records = [{"type": "message", "role": "user", "text": long_text}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    # truncated to 200 chars plus truncation marker
    assert "x" * 200 in out
    assert "..." in out


def test_print_flow_with_blocks_and_render(capsys):
    schema = InspectorSchema(
        label="X",
        record_type_of=lambda r: r.get("type", ""),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: "",
        block_types_of=lambda r: [b["kind"] for b in r.get("blocks", [])],
        blocks_of=lambda r: r.get("blocks", []),
        render_block=lambda b: f"RENDERED:{b['kind']}",
    )
    records = [{"type": "message", "role": "user",
                "blocks": [{"kind": "text"}, {"kind": "tool_use"}]}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "blocks: text, tool_use" in out
    assert "RENDERED:text" in out
    assert "RENDERED:tool_use" in out


def test_print_flow_with_tokens(capsys):
    schema = InspectorSchema(
        label="X",
        record_type_of=lambda r: r.get("type", ""),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: "",
        tokens_of=lambda r: r.get("tokens", 0),
    )
    records = [
        {"type": "message", "role": "user", "tokens": 42},
        {"type": "message", "role": "user", "tokens": 0},
    ]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "42tk" in out


# ---------------------------------------------------------------------------
# print_raw_examples
# ---------------------------------------------------------------------------


def test_print_raw_examples(capsys):
    schema = _simple_schema()
    records = [
        {"type": "message", "role": "user", "text": "hello"},
        {"type": "message", "role": "assistant", "text": "hi"},
        {"type": "session_meta", "id": "x"},
    ]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    assert "type = message" in out
    assert "type = session_meta" in out
    # Only one example per type
    assert out.count("type = message") == 1
    assert "flattened text" in out


def test_print_raw_examples_non_message_no_text(capsys):
    schema = _simple_schema()
    print_raw_examples([{"type": "session_meta", "id": "x"}], schema)
    out = capsys.readouterr().out
    assert "type = session_meta" in out
    assert "flattened text" not in out


# ---------------------------------------------------------------------------
# run_inspection
# ---------------------------------------------------------------------------


def test_run_inspection_smoke(capsys, tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("x", encoding="utf-8")
    schema = _simple_schema()
    records = [
        {"type": "message", "role": "user", "text": "hi"},
        {"type": "message", "role": "assistant", "text": "hello"},
    ]
    run_inspection(p, records, schema)
    out = capsys.readouterr().out
    assert "DEMO SESSION INSPECTOR" in out
    assert "RECORD TYPES" in out
    assert "MESSAGE ROLES" in out
    assert "CONVERSATION FLOW" not in out  # show_flow default False
    assert "RAW RECORD EXAMPLES" not in out


def test_run_inspection_with_flow_and_blocks(capsys, tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("x", encoding="utf-8")
    schema = _simple_schema()
    records = [{"type": "message", "role": "user", "text": "hi"}]
    run_inspection(p, records, schema, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out


def test_run_inspection_extra_header(capsys, tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("x", encoding="utf-8")
    schema = _simple_schema()
    captured: list[list[dict]] = []

    def extra(recs):
        captured.append(recs)
        print("EXTRA HEADER CALLED")

    records = [{"type": "message", "role": "user"}]
    run_inspection(p, records, schema, extra_header=extra)
    out = capsys.readouterr().out
    assert "EXTRA HEADER CALLED" in out
    assert captured == [records]


def test_run_inspection_show_blocks_without_flow(capsys, tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("x", encoding="utf-8")
    schema = _simple_schema()
    records = [{"type": "message", "role": "user", "text": "hi"}]
    run_inspection(p, records, schema, show_flow=False, show_blocks=True)
    out = capsys.readouterr().out
    # show_blocks alone has no effect
    assert "CONVERSATION FLOW" not in out
