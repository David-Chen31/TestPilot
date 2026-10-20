import base64
import json
from io import BytesIO
from pathlib import Path

import pytest

from claude_converter.utils import (
    BOLD,
    CYAN,
    DEFAULT_ROLE_COLORS,
    DIM,
    GREEN,
    RED,
    RESET,
    YELLOW,
    MAGENTA,
    InspectorSchema,
    _c,
    _hr,
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_ansi_constants():
    assert RESET == "\033[0m"
    assert BOLD == "\033[1m"
    assert DIM == "\033[2m"
    assert CYAN == "\033[36m"
    assert GREEN == "\033[32m"
    assert YELLOW == "\033[33m"
    assert RED == "\033[31m"
    assert MAGENTA == "\033[35m"


def test_default_role_colors():
    assert DEFAULT_ROLE_COLORS == {"user": GREEN, "assistant": CYAN, "system": YELLOW}


# ---------------------------------------------------------------------------
# _c, _hr, _truncate, _scaled_bar
# ---------------------------------------------------------------------------

def test__c_basic():
    assert _c("hi", BOLD) == BOLD + "hi" + RESET


def test__c_multiple_codes_and_non_str():
    assert _c(42, BOLD, CYAN) == BOLD + CYAN + "42" + RESET


def test__c_no_codes():
    assert _c("x") == "x" + RESET


def test__hr_default():
    assert _hr() == DIM + ("-" * 70) + RESET


def test__hr_custom():
    assert _hr("=", 5) == DIM + "=====" + RESET


def test__truncate_short():
    assert _truncate("hello", 10) == "hello"


def test__truncate_exact():
    assert _truncate("hello", 5) == "hello"


def test__truncate_long():
    result = _truncate("hello world", 5)
    assert result == "hello" + DIM + "..." + RESET


def test__truncate_newline_replaced():
    assert _truncate("a\nb", 10) == "a\\nb"


def test__truncate_non_str():
    assert _truncate(12345, 10) == "12345"


def test__scaled_bar_zero_max():
    assert _scaled_bar(5, 0) == ""


def test__scaled_bar_negative_max():
    assert _scaled_bar(5, -1) == ""


def test__scaled_bar_full():
    assert _scaled_bar(10, 10, width=40) == "█" * 40


def test__scaled_bar_half():
    assert _scaled_bar(5, 10, width=40) == "█" * 20


def test__scaled_bar_min_one():
    assert _scaled_bar(1, 1000, width=40) == "█"


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

def test_load_jsonl_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_jsonl(tmp_path / "nope.jsonl")


def test_load_jsonl_bad_extension(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("{}\n")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_jsonl(p)


def test_load_jsonl_no_extension(tmp_path):
    p = tmp_path / "file"
    p.write_text("{}\n")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_jsonl(p)


def test_load_jsonl_valid(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n')
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_json_extension(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"a": 1}\n')
    assert load_jsonl(p) == [{"a": 1}]


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"b": 2}\n')
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_invalid_line_skipped(tmp_path, capsys):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
    result = load_jsonl(p)
    assert result == [{"a": 1}, {"b": 2}]
    err = capsys.readouterr().err
    assert "Line 2 is invalid JSON" in err


def test_load_jsonl_empty_file(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="empty or contains no valid records"):
        load_jsonl(p)


def test_load_jsonl_only_invalid(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("bad\nworse\n")
    with pytest.raises(ValueError, match="no valid records"):
        load_jsonl(p)


def test_load_jsonl_accepts_str_path(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"x": 1}\n')
    assert load_jsonl(str(p)) == [{"x": 1}]


# ---------------------------------------------------------------------------
# convert_base64_to_pil_image / _pil_to_data_uri
# ---------------------------------------------------------------------------

PIL = pytest.importorskip("PIL")
from PIL import Image as PILImage  # noqa: E402


def _make_png_bytes(color=(255, 0, 0)) -> bytes:
    img = PILImage.new("RGB", (4, 4), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_convert_base64_bare_string():
    png = _make_png_bytes()
    b64 = base64.b64encode(png).decode()
    img = convert_base64_to_pil_image(b64)
    assert isinstance(img, PILImage.Image)
    assert img.format == "PNG"
    assert img.size == (4, 4)


def test_convert_base64_data_uri():
    png = _make_png_bytes()
    b64 = base64.b64encode(png).decode()
    uri = f"data:image/png;base64,{b64}"
    img = convert_base64_to_pil_image(uri)
    assert isinstance(img, PILImage.Image)


def test_convert_base64_invalid_payload():
    with pytest.raises(ValueError, match="Could not decode base64 image"):
        convert_base64_to_pil_image("not-a-real-base64-image-at-all")


def test_convert_base64_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL"):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Pillow is required"):
        convert_base64_to_pil_image("anything")


def test__pil_to_data_uri_png():
    img = PILImage.new("RGB", (2, 2), (0, 0, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img2 = PILImage.open(buf)
    uri = _pil_to_data_uri(img2)
    assert uri.startswith("data:image/png;base64,")
    b64 = uri.split(",", 1)[1]
    decoded = base64.b64decode(b64)
    assert decoded[:4] == b"\x89PNG"


def test__pil_to_data_uri_jpeg():
    img = PILImage.new("RGB", (2, 2), (0, 0, 255))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    img2 = PILImage.open(buf)
    uri = _pil_to_data_uri(img2)
    assert uri.startswith("data:image/jpeg;base64,")


def test__pil_to_data_uri_no_format_falls_back_png():
    img = PILImage.new("RGB", (2, 2))
    assert img.format is None
    uri = _pil_to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")


def test__pil_to_data_uri_falls_back_when_save_fails():
    class FakeImage:
        format = "BOGUS"

        def __init__(self):
            self.calls = []

        def save(self, buf, format=None):
            self.calls.append(format)
            if format == "BOGUS":
                raise RuntimeError("nope")
            buf.write(b"PNGDATA")

    fake = FakeImage()
    uri = _pil_to_data_uri(fake)
    assert uri.startswith("data:image/png;base64,")
    assert fake.calls == ["BOGUS", "PNG"]


# ---------------------------------------------------------------------------
# messages_to_json_safe
# ---------------------------------------------------------------------------

def test_messages_to_json_safe_str_content():
    msgs = [{"role": "user", "content": "hello"}]
    out = messages_to_json_safe(msgs)
    assert out == [{"role": "user", "content": "hello"}]
    assert out is not msgs
    assert out[0] is not msgs[0]


def test_messages_to_json_safe_no_content_key():
    msgs = [{"role": "user"}]
    out = messages_to_json_safe(msgs)
    assert out == [{"role": "user", "content": None}]


def test_messages_to_json_safe_text_parts_preserved():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    out = messages_to_json_safe(msgs)
    assert out == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_messages_to_json_safe_image_part_converted():
    img = PILImage.new("RGB", (2, 2))
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image", "image": img},
    ]}]
    out = messages_to_json_safe(msgs)
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image"
    assert parts[1]["image"].startswith("data:image/")
    # original image untouched
    assert msgs[0]["content"][1]["image"] is img


def test_messages_to_json_safe_image_non_pil_kept():
    msgs = [{"role": "user", "content": [{"type": "image", "image": "already-a-string"}]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [{"type": "image", "image": "already-a-string"}]


def test_messages_to_json_safe_json_dumpable():
    img = PILImage.new("RGB", (2, 2))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
    out = messages_to_json_safe(msgs)
    dumped = json.dumps(out)
    assert "data:image/png;base64," in dumped


def test_messages_to_json_safe_image_part_no_image_key():
    msgs = [{"role": "user", "content": [{"type": "image"}]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [{"type": "image"}]


def test_messages_to_json_safe_part_not_dict():
    msgs = [{"role": "user", "content": ["plain", 5]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == ["plain", 5]


def test_messages_to_json_safe_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL"):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    msgs = [{"role": "user", "content": [{"type": "image", "image": 123}]}]
    out = messages_to_json_safe(msgs)
    assert out[0]["content"] == [{"type": "image", "image": "123"}]


# ---------------------------------------------------------------------------
# merge_content_parts
# ---------------------------------------------------------------------------

def test_merge_content_parts_into_empty():
    existing = []
    merge_content_parts(existing, [{"type": "text", "text": "hi"}])
    assert existing == [{"type": "text", "text": "hi"}]


def test_merge_content_parts_appends_non_text():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [{"type": "image", "image": "x"}])
    assert existing == [
        {"type": "text", "text": "a"},
        {"type": "image", "image": "x"},
    ]


def test_merge_content_parts_joins_adjacent_text():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [{"type": "text", "text": "b"}])
    assert existing == [{"type": "text", "text": "a\nb"}]


def test_merge_content_parts_joins_multiple():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [
        {"type": "text", "text": "b"},
        {"type": "text", "text": "c"},
    ])
    assert existing == [{"type": "text", "text": "a\nb\nc"}]


def test_merge_content_parts_copies_parts():
    new = [{"type": "image", "image": "x"}]
    existing = []
    merge_content_parts(existing, new)
    existing[0]["image"] = "changed"
    assert new[0]["image"] == "x"


def test_merge_content_parts_text_then_image_then_text():
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


# ---------------------------------------------------------------------------
# finalize_content
# ---------------------------------------------------------------------------

def test_finalize_content_text_only():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert finalize_content(parts) == "a\nb"


def test_finalize_content_single_text():
    assert finalize_content([{"type": "text", "text": "only"}]) == "only"


def test_finalize_content_with_image():
    parts = [{"type": "text", "text": "a"}, {"type": "image", "image": "x"}]
    assert finalize_content(parts) == parts


def test_finalize_content_empty():
    assert finalize_content([]) == ""


# ---------------------------------------------------------------------------
# dget / dlist
# ---------------------------------------------------------------------------

def test_dget_present():
    assert dget({"a": {"b": 1}}, "a") == {"b": 1}


def test_dget_missing():
    assert dget({}, "a") == {}


def test_dget_null():
    assert dget({"a": None}, "a") == {}


def test_dget_non_dict_value():
    assert dget({"a": 5}, "a") == {}


def test_dget_non_dict_parent():
    assert dget(None, "a") == {}
    assert dget([], "a") == {}
    assert dget("string", "a") == {}


def test_dlist_present():
    assert dlist({"a": [1, 2]}, "a") == [1, 2]


def test_dlist_missing():
    assert dlist({}, "a") == []


def test_dlist_null():
    assert dlist({"a": None}, "a") == []


def test_dlist_non_list():
    assert dlist({"a": "str"}, "a") == []


def test_dlist_non_dict_parent():
    assert dlist(None, "a") == []
    assert dlist(42, "a") == []


# ---------------------------------------------------------------------------
# InspectorSchema
# ---------------------------------------------------------------------------

def test_inspector_schema_defaults():
    schema = InspectorSchema(
        label="TEST",
        record_type_of=lambda r: r.get("type", "?"),
        is_message=lambda r: r.get("type") == "msg",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: str(r.get("text", "")),
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
    a = InspectorSchema("A", lambda r: "", lambda r: False, lambda r: "", lambda r: "")
    b = InspectorSchema("B", lambda r: "", lambda r: False, lambda r: "", lambda r: "")
    a.role_colors["user"] = "X"
    assert b.role_colors["user"] == GREEN


def _simple_schema(**overrides):
    defaults = dict(
        label="TEST",
        record_type_of=lambda r: r.get("type", "?"),
        is_message=lambda r: r.get("type") == "msg",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: str(r.get("text", "")),
    )
    defaults.update(overrides)
    return InspectorSchema(**defaults)


# ---------------------------------------------------------------------------
# print_header
# ---------------------------------------------------------------------------

def test_print_header(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("x" * 2048)
    schema = _simple_schema()
    print_header(p, [{"a": 1}, {"b": 2}], schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert str(p) in out
    assert "2.0 KB" in out
    assert "Lines : " in out


# ---------------------------------------------------------------------------
# print_record_types
# ---------------------------------------------------------------------------

def test_print_record_types(capsys):
    schema = _simple_schema()
    records = [{"type": "a"}, {"type": "a"}, {"type": "b"}]
    print_record_types(records, schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out
    assert "a" in out
    assert "b" in out


# ---------------------------------------------------------------------------
# print_block_summary
# ---------------------------------------------------------------------------

def test_print_block_summary_noop_when_none(capsys):
    schema = _simple_schema()
    print_block_summary([{"type": "msg", "role": "user"}], schema)
    assert capsys.readouterr().out == ""


def test_print_block_summary(capsys):
    records = [
        {"type": "msg", "role": "user", "block_types": ["text"]},
        {"type": "msg", "role": "assistant", "block_types": ["text", "tool_use"]},
        {"type": "other", "block_types": ["x"]},
    ]
    schema = _simple_schema(block_types_of=lambda r: r.get("block_types", []))
    print_block_summary(records, schema)
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" in out
    assert "user" in out
    assert "assistant" in out
    assert "other" in out


def test_print_block_summary_empty_records(capsys):
    schema = _simple_schema(block_types_of=lambda r: [])
    print_block_summary([], schema)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# print_message_roles
# ---------------------------------------------------------------------------

def test_print_message_roles_no_messages(capsys):
    schema = _simple_schema()
    print_message_roles([{"type": "other"}], schema)
    assert capsys.readouterr().out == ""


def test_print_message_roles(capsys):
    schema = _simple_schema()
    records = [
        {"type": "msg", "role": "user"},
        {"type": "msg", "role": "user"},
        {"type": "msg", "role": "assistant"},
        {"type": "other"},
    ]
    print_message_roles(records, schema)
    out = capsys.readouterr().out
    assert "MESSAGE ROLES" in out
    assert "user" in out
    assert "assistant" in out


# ---------------------------------------------------------------------------
# print_token_usage
# ---------------------------------------------------------------------------

def test_print_token_usage_noop_when_none(capsys):
    schema = _simple_schema()
    print_token_usage([], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_noop_when_empty_totals(capsys):
    schema = _simple_schema(total_tokens_of=lambda r: {})
    print_token_usage([], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage(capsys):
    schema = _simple_schema(total_tokens_of=lambda r: {"input_tokens": 1234, "output_tokens": 5})
    print_token_usage([], schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" in out
    assert "1,234" in out
    assert "Input Tokens" in out
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
    records = [{"type": "msg", "role": "user", "text": "hi"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "USER" in out


def test_print_flow_with_tokens(capsys):
    schema = _simple_schema(tokens_of=lambda r: 42)
    records = [{"type": "msg", "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "42tk" in out


def test_print_flow_zero_tokens_no_label(capsys):
    schema = _simple_schema(tokens_of=lambda r: 0)
    records = [{"type": "msg", "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "tk" not in out


def test_print_flow_with_block_types(capsys):
    schema = _simple_schema(block_types_of=lambda r: ["text", "tool_use"])
    records = [{"type": "msg", "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "blocks: text, tool_use" in out


def test_print_flow_show_text_fallback(capsys):
    schema = _simple_schema()
    records = [{"type": "msg", "role": "user", "text": "hello there"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "hello there" in out


def test_print_flow_show_text_with_blocks(capsys):
    schema = _simple_schema(
        blocks_of=lambda r: [{"b": 1}, {"b": 2}],
        render_block=lambda b: f"RENDERED-{b['b']}",
    )
    records = [{"type": "msg", "role": "user"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "RENDERED-1" in out
    assert "RENDERED-2" in out


def test_print_flow_show_text_false_no_text(capsys):
    schema = _simple_schema()
    records = [{"type": "msg", "role": "user", "text": "shouldnotappear"}]
    print_flow(records, schema, show_text=False)
    out = capsys.readouterr().out
    assert "shouldnotappear" not in out


# ---------------------------------------------------------------------------
# print_raw_examples
# ---------------------------------------------------------------------------

def test_print_raw_examples(capsys):
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


def test_print_raw_examples_message_text(capsys):
    schema = _simple_schema()
    records = [{"type": "msg", "role": "user", "text": "flattened"}]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "flattened text" in out
    assert "flattened" in out


# ---------------------------------------------------------------------------
# run_inspection
# ---------------------------------------------------------------------------

def test_run_inspection_minimal(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("")
    schema = _simple_schema()
    records = [{"type": "msg", "role": "user", "text": "hi"}]
    run_inspection(p, records, schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert "RECORD TYPES" in out
    assert "MESSAGE ROLES" in out


def test_run_inspection_extra_header(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("")
    schema = _simple_schema()
    called = []

    def extra(recs):
        called.append(recs)
        print("EXTRA-HEADER")

    run_inspection(p, [{"type": "msg", "role": "user"}], schema, extra_header=extra)
    out = capsys.readouterr().out
    assert "EXTRA-HEADER" in out
    assert len(called) == 1


def test_run_inspection_show_flow(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("")
    schema = _simple_schema()
    records = [{"type": "msg", "role": "user", "text": "convo"}]
    run_inspection(p, records, schema, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out


def test_run_inspection_no_flow(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("")
    schema = _simple_schema()
    run_inspection(p, [{"type": "msg", "role": "user"}], schema, show_flow=False)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" not in out


def test_run_inspection_accepts_str_path(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text("")
    schema = _simple_schema()
    run_inspection(str(p), [{"type": "msg", "role": "user"}], schema)
    out = capsys.readouterr().out
    assert str(p) in out
