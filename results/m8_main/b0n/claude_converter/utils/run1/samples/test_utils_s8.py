from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import pytest

from claude_converter import utils
from claude_converter.utils import (
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
# Color / helper constants
# ---------------------------------------------------------------------------

def test_ansi_constants():
    assert utils.RESET == "\033[0m"
    assert utils.BOLD == "\033[1m"
    assert utils.DIM == "\033[2m"
    assert utils.CYAN == "\033[36m"
    assert utils.GREEN == "\033[32m"
    assert utils.YELLOW == "\033[33m"
    assert utils.RED == "\033[31m"
    assert utils.MAGENTA == "\033[35m"


def test_default_role_colors():
    assert utils.DEFAULT_ROLE_COLORS == {
        "user": utils.GREEN,
        "assistant": utils.CYAN,
        "system": utils.YELLOW,
    }


def test__c_wraps_text():
    assert utils._c("hi", utils.BOLD) == utils.BOLD + "hi" + utils.RESET
    assert utils._c(42) == "42" + utils.RESET
    assert utils._c("x", utils.BOLD, utils.CYAN) == utils.BOLD + utils.CYAN + "x" + utils.RESET


def test__hr():
    result = utils._hr("=", 5)
    assert result == utils.DIM + "=====" + utils.RESET
    assert utils._hr() == utils.DIM + "-" * 70 + utils.RESET


def test__truncate_short():
    assert utils._truncate("hello", 10) == "hello"


def test__truncate_long():
    result = utils._truncate("abcdefghij", 5)
    assert result == "abcde" + utils.DIM + "..." + utils.RESET


def test__truncate_newlines():
    assert utils._truncate("a\nb", 10) == "a\\nb"


def test__truncate_exact_boundary():
    assert utils._truncate("abcde", 5) == "abcde"


def test__truncate_non_str():
    assert utils._truncate(12345, 10) == "12345"


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

def test_load_jsonl_basic(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_accepts_str(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    assert load_jsonl(str(p)) == [{"a": 1}]


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('\n\n{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_json_extension(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}]


def test_load_jsonl_invalid_line_prints_stderr(tmp_path, capsys):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\nnot-json\n{"b": 2}\n', encoding="utf-8")
    result = load_jsonl(p)
    assert result == [{"a": 1}, {"b": 2}]
    captured = capsys.readouterr()
    assert "Line 2 is invalid JSON" in captured.err


def test_load_jsonl_missing_file(tmp_path):
    p = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_jsonl(p)


def test_load_jsonl_bad_extension(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_jsonl(p)


def test_load_jsonl_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no valid records"):
        load_jsonl(p)


def test_load_jsonl_all_invalid(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("bad\nworse\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no valid records"):
        load_jsonl(p)


# ---------------------------------------------------------------------------
# convert_base64_to_pil_image
# ---------------------------------------------------------------------------

def _make_png_bytes(color=(255, 0, 0)) -> bytes:
    PIL = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    PIL.new("RGB", (2, 2), color).save(buf, format="PNG")
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


def test_convert_base64_invalid_payload():
    with pytest.raises(ValueError, match="Could not decode base64 image"):
        convert_base64_to_pil_image("not valid base64 !!!")


def test_convert_base64_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Pillow is required"):
        convert_base64_to_pil_image("anything")


# ---------------------------------------------------------------------------
# _pil_to_data_uri
# ---------------------------------------------------------------------------

def test_pil_to_data_uri_png():
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new("RGB", (2, 2), (0, 0, 255))
    uri = utils._pil_to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")
    payload = uri.split(",", 1)[1]
    decoded = base64.b64decode(payload)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_pil_to_data_uri_fallback_to_png():
    PIL = pytest.importorskip("PIL.Image")

    class WeirdImage:
        format = "WEIRDFMT"

        def __init__(self):
            self._real = PIL.new("RGB", (1, 1), (0, 0, 0))
            self.calls = []

        def save(self, buf, format=None):
            self.calls.append(format)
            if format != "PNG":
                raise ValueError("unsupported format")
            self._real.save(buf, format="PNG")

    img = WeirdImage()
    uri = utils._pil_to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")
    assert img.calls == ["WEIRDFMT", "PNG"]


def test_pil_to_data_uri_no_format_defaults_png():
    PIL = pytest.importorskip("PIL.Image")

    class NoFormat:
        def __init__(self):
            self._real = PIL.new("RGB", (1, 1), (0, 0, 0))

        def save(self, buf, format=None):
            self._real.save(buf, format=format)

    uri = utils._pil_to_data_uri(NoFormat())
    assert uri.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# messages_to_json_safe
# ---------------------------------------------------------------------------

def test_messages_to_json_safe_passthrough():
    messages = [{"role": "user", "content": "hi"}]
    result = messages_to_json_safe(messages)
    assert result == [{"role": "user", "content": "hi"}]
    # copy, not same dict
    assert result[0] is not messages[0]


def test_messages_to_json_safe_list_of_text_parts():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    result = messages_to_json_safe(messages)
    assert result[0]["content"] == [{"type": "text", "text": "hello"}]


def test_messages_to_json_safe_converts_image():
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new("RGB", (2, 2), (10, 20, 30))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see"},
                {"type": "image", "image": img},
            ],
        }
    ]
    result = messages_to_json_safe(messages)
    converted = result[0]["content"]
    assert converted[0] == {"type": "text", "text": "see"}
    assert converted[1]["type"] == "image"
    assert converted[1]["image"].startswith("data:image/png;base64,")
    # Original untouched
    assert messages[0]["content"][1]["image"] is img


def test_messages_to_json_safe_is_json_dumpable():
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new("RGB", (1, 1), (0, 0, 0))
    messages = [
        {"role": "user", "content": [{"type": "image", "image": img}]}
    ]
    safe = messages_to_json_safe(messages)
    dumped = json.dumps(safe)
    assert "data:image/png;base64," in dumped


def test_messages_to_json_safe_image_without_image_key():
    messages = [
        {"role": "user", "content": [{"type": "image"}]}
    ]
    result = messages_to_json_safe(messages)
    assert result[0]["content"] == [{"type": "image"}]


def test_messages_to_json_safe_non_image_value_left_alone():
    messages = [
        {"role": "user", "content": [{"type": "image", "image": "already-a-string"}]}
    ]
    result = messages_to_json_safe(messages)
    assert result[0]["content"] == [{"type": "image", "image": "already-a-string"}]


def test_messages_to_json_safe_no_content_key():
    messages = [{"role": "user"}]
    result = messages_to_json_safe(messages)
    assert result == [{"role": "user", "content": None}]


def test_messages_to_json_safe_pil_import_error(monkeypatch):
    """Simulate Pillow being unavailable: image falls back to repr()."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    obj = {"not": "an image"}
    messages = [{"role": "user", "content": [{"type": "image", "image": obj}]}]
    result = messages_to_json_safe(messages)
    assert result[0]["content"] == [{"type": "image", "image": repr(obj)}]


# ---------------------------------------------------------------------------
# merge_content_parts
# ---------------------------------------------------------------------------

def test_merge_content_parts_appends():
    existing = [{"type": "text", "text": "a"}]
    new = [{"type": "image", "image": "img"}]
    merge_content_parts(existing, new)
    assert existing == [{"type": "text", "text": "a"}, {"type": "image", "image": "img"}]


def test_merge_content_parts_joins_text():
    existing = [{"type": "text", "text": "a"}]
    new = [{"type": "text", "text": "b"}, {"type": "text", "text": "c"}]
    merge_content_parts(existing, new)
    assert existing == [{"type": "text", "text": "a\nb\nc"}]


def test_merge_content_parts_empty_new():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [])
    assert existing == [{"type": "text", "text": "a"}]


def test_merge_content_parts_empty_existing():
    existing = []
    new = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, new)
    assert existing == [{"type": "text", "text": "a"}]


def test_merge_content_parts_copies_new_dict():
    existing = []
    part = {"type": "image", "image": "x"}
    merge_content_parts(existing, [part])
    assert existing[0] == part
    assert existing[0] is not part


def test_merge_content_parts_text_then_image_then_text():
    existing = [{"type": "text", "text": "a"}]
    new = [
        {"type": "image", "image": "i"},
        {"type": "text", "text": "b"},
    ]
    merge_content_parts(existing, new)
    assert existing == [
        {"type": "text", "text": "a"},
        {"type": "image", "image": "i"},
        {"type": "text", "text": "b"},
    ]


# ---------------------------------------------------------------------------
# finalize_content
# ---------------------------------------------------------------------------

def test_finalize_content_text_only():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert finalize_content(parts) == "a\nb"


def test_finalize_content_single_text():
    assert finalize_content([{"type": "text", "text": "hello"}]) == "hello"


def test_finalize_content_with_image_returns_list():
    parts = [{"type": "text", "text": "a"}, {"type": "image", "image": "i"}]
    assert finalize_content(parts) == parts
    assert finalize_content(parts) is parts


def test_finalize_content_empty():
    assert finalize_content([]) == ""


# ---------------------------------------------------------------------------
# dget / dlist
# ---------------------------------------------------------------------------

def test_dget_basic():
    assert dget({"a": {"b": 1}}, "a") == {"b": 1}


def test_dget_missing_key():
    assert dget({"a": 1}, "b") == {}


def test_dget_null_value():
    assert dget({"a": None}, "a") == {}


def test_dget_non_dict_value():
    assert dget({"a": 5}, "a") == {}
    assert dget({"a": [1, 2]}, "a") == {}


def test_dget_non_dict_parent():
    assert dget(None, "a") == {}
    assert dget([1, 2], "a") == {}
    assert dget("string", "a") == {}


def test_dlist_basic():
    assert dlist({"a": [1, 2]}, "a") == [1, 2]


def test_dlist_missing():
    assert dlist({"a": 1}, "b") == []


def test_dlist_null():
    assert dlist({"a": None}, "a") == []


def test_dlist_non_list():
    assert dlist({"a": {"b": 1}}, "a") == []
    assert dlist({"a": "str"}, "a") == []


def test_dlist_non_dict_parent():
    assert dlist(None, "a") == []
    assert dlist([1, 2], "a") == []


# ---------------------------------------------------------------------------
# InspectorSchema
# ---------------------------------------------------------------------------

def _simple_helpers():
    return {
        "record_type_of": lambda r: r.get("type", "unknown"),
        "is_message": lambda r: r.get("type") == "message",
        "role_of": lambda r: r.get("role", "user"),
        "text_of": lambda r: r.get("text", ""),
    }


def test_inspector_schema_defaults():
    helpers = _simple_helpers()
    schema = InspectorSchema(label="TEST", **helpers)
    assert schema.label == "TEST"
    assert schema.timestamp_of({"x": 1}) == ""
    assert schema.total_tokens_of is None
    assert schema.tokens_of is None
    assert schema.block_types_of is None
    assert schema.blocks_of is None
    assert schema.render_block is None


def test_inspector_schema_default_role_colors_are_copies():
    helpers = _simple_helpers()
    s1 = InspectorSchema(label="A", **helpers)
    s2 = InspectorSchema(label="B", **helpers)
    assert s1.role_colors == utils.DEFAULT_ROLE_COLORS
    s1.role_colors["user"] = "changed"
    assert s2.role_colors["user"] == utils.GREEN


def test_inspector_schema_custom_role_colors():
    helpers = _simple_helpers()
    schema = InspectorSchema(label="X", role_colors={"user": "PURPLE"}, **helpers)
    assert schema.role_colors == {"user": "PURPLE"}


# ---------------------------------------------------------------------------
# Printing functions
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return InspectorSchema(
        label="TEST",
        record_type_of=lambda r: r.get("type", "unknown"),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
        timestamp_of=lambda r: r.get("ts", ""),
    )


@pytest.fixture
def full_schema():
    def total_tokens(records):
        users = sum(1 for r in records if r.get("role") == "user")
        return {"input_tokens": users * 10, "output_tokens": users * 5}

    def tokens_of(r):
        return r.get("tokens", 0)

    def block_types_of(r):
        return r.get("block_types", [])

    def blocks_of(r):
        return r.get("blocks", [])

    def render_block(block):
        return "    " + block.get("text", "")

    return InspectorSchema(
        label="FULL",
        record_type_of=lambda r: r.get("type", "unknown"),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
        timestamp_of=lambda r: r.get("ts", ""),
        total_tokens_of=total_tokens,
        tokens_of=tokens_of,
        block_types_of=block_types_of,
        blocks_of=blocks_of,
        render_block=render_block,
    )


def test_print_header(tmp_path, schema, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("x" * 2048, encoding="utf-8")
    records = [{"type": "a"}, {"type": "b"}]
    print_header(p, records, schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert str(p) in out
    assert "2.0 KB" in out
    assert "Lines :" in out


def test_print_record_types(schema, capsys):
    records = [
        {"type": "message"},
        {"type": "message"},
        {"type": "summary"},
    ]
    print_record_types(records, schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out
    assert "message" in out
    assert "summary" in out


def test_print_record_types_empty(schema, capsys):
    print_record_types([], schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out


def test_print_block_summary_noop_without_callback(schema, capsys):
    records = [{"type": "message", "role": "user"}]
    print_block_summary(records, schema)
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" not in out


def test_print_block_summary(full_schema, capsys):
    records = [
        {"type": "message", "role": "user", "block_types": ["text", "image"]},
        {"type": "message", "role": "assistant", "block_types": ["text"]},
        {"type": "system", "block_types": ["meta"]},
    ]
    print_block_summary(records, full_schema)
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" in out
    assert "user" in out
    assert "assistant" in out
    assert "system" in out


def test_print_block_summary_empty(full_schema, capsys):
    print_block_summary([], full_schema)
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" not in out


def test_print_message_roles(schema, capsys):
    records = [
        {"type": "message", "role": "user"},
        {"type": "message", "role": "assistant"},
        {"type": "message", "role": "user"},
        {"type": "summary"},
    ]
    print_message_roles(records, schema)
    out = capsys.readouterr().out
    assert "MESSAGE ROLES" in out
    assert "user" in out
    assert "assistant" in out


def test_print_message_roles_no_messages(schema, capsys):
    print_message_roles([{"type": "summary"}], schema)
    out = capsys.readouterr().out
    assert "MESSAGE ROLES" not in out


def test_print_token_usage_noop(schema, capsys):
    print_token_usage([{"type": "message"}], schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" not in out


def test_print_token_usage_empty_totals(capsys):
    schema = InspectorSchema(
        label="X",
        record_type_of=lambda r: "t",
        is_message=lambda r: False,
        role_of=lambda r: "user",
        text_of=lambda r: "",
        total_tokens_of=lambda records: {},
    )
    print_token_usage([], schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" not in out


def test_print_token_usage(full_schema, capsys):
    records = [
        {"type": "message", "role": "user"},
        {"type": "message", "role": "assistant"},
    ]
    print_token_usage(records, full_schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" in out
    assert "Input Tokens" in out
    assert "10" in out
    assert "Output Tokens" in out


def test_print_flow_no_messages(schema, capsys):
    print_flow([{"type": "summary"}], schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" not in out


def test_print_flow_basic(schema, capsys):
    records = [
        {"type": "message", "role": "user", "text": "hello", "ts": "10:00"},
        {"type": "summary"},
        {"type": "message", "role": "assistant", "text": "hi", "ts": "10:01"},
    ]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "[USER]" in out
    assert "[ASSISTANT]" in out
    assert "10:00" in out
    assert "hello" not in out  # show_text defaults to False


def test_print_flow_show_text(schema, capsys):
    records = [{"type": "message", "role": "user", "text": "hello"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "hello" in out


def test_print_flow_show_text_with_blocks(full_schema, capsys):
    records = [
        {
            "type": "message",
            "role": "user",
            "text": "ignored",
            "blocks": [{"text": "block-text"}],
            "block_types": ["text"],
            "tokens": 5,
        }
    ]
    print_flow(records, full_schema, show_text=True)
    out = capsys.readouterr().out
    assert "block-text" in out
    assert "5tk" in out
    assert "blocks:" in out


def test_print_flow_with_block_types_only(schema, capsys):
    schema_with_blocks = InspectorSchema(
        label="B",
        record_type_of=lambda r: r.get("type", "unknown"),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
        block_types_of=lambda r: ["text", "image"],
    )
    records = [{"type": "message", "role": "user"}]
    print_flow(records, schema_with_blocks)
    out = capsys.readouterr().out
    assert "blocks: text, image" in out


def test_print_flow_tokens_zero(schema, capsys):
    schema_tok = InspectorSchema(
        label="T",
        record_type_of=lambda r: r.get("type", "unknown"),
        is_message=lambda r: r.get("type") == "message",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
        tokens_of=lambda r: 0,
    )
    print_flow([{"type": "message", "role": "user"}], schema_tok)
    out = capsys.readouterr().out
    assert "0tk" not in out


def test_print_raw_examples(schema, capsys):
    records = [
        {"type": "message", "role": "user", "text": "hello"},
        {"type": "message", "role": "user", "text": "again"},
        {"type": "summary"},
    ]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    # one example per type: message appears once in header but flattened text printed once
    assert out.count("type = message") == 1
    assert out.count("type = summary") == 1
    assert "flattened text:" in out


def test_print_raw_examples_truncates_text(schema, capsys):
    long_text = "x" * 200
    records = [{"type": "message", "role": "user", "text": long_text}]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "x" * 150 in out
    assert "..." in out


def test_print_raw_examples_empty(schema, capsys):
    print_raw_examples([], schema)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out


# ---------------------------------------------------------------------------
# run_inspection
# ---------------------------------------------------------------------------

def test_run_inspection_basic(tmp_path, schema, capsys):
    p = tmp_path / "session.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    records = [
        {"type": "message", "role": "user", "text": "hi"},
        {"type": "summary"},
    ]
    run_inspection(p, records, schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert "RECORD TYPES" in out
    assert "MESSAGE ROLES" in out
    assert "CONVERSATION FLOW" not in out


def test_run_inspection_with_flow_and_blocks(tmp_path, full_schema, capsys):
    p = tmp_path / "session.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    records = [
        {
            "type": "message",
            "role": "user",
            "text": "hi",
            "blocks": [{"text": "blocker"}],
            "block_types": ["text"],
        }
    ]
    run_inspection(
        p, records, full_schema, show_flow=True, show_blocks=True
    )
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "blocker" in out
    assert "TOKEN USAGE" in out


def test_run_inspection_show_blocks_without_flow(tmp_path, full_schema, capsys):
    p = tmp_path / "session.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    records = [
        {
            "type": "message",
            "role": "user",
            "text": "hi",
            "blocks": [{"text": "blocker"}],
        }
    ]
    run_inspection(p, records, full_schema, show_flow=False, show_blocks=True)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" not in out
    assert "blocker" not in out


def test_run_inspection_extra_header(tmp_path, schema, capsys):
    p = tmp_path / "session.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    calls = []

    def extra(records):
        calls.append(records)
        print("EXTRA HEADER METADATA")

    records = [{"type": "message", "role": "user"}]
    run_inspection(p, records, schema, extra_header=extra)
    out = capsys.readouterr().out
    assert "EXTRA HEADER METADATA" in out
    assert calls == [records]


def test_run_inspection_accepts_str_path(tmp_path, schema, capsys):
    p = tmp_path / "session.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    run_inspection(str(p), [{"type": "message", "role": "user"}], schema)
    out = capsys.readouterr().out
    assert str(p) in out
