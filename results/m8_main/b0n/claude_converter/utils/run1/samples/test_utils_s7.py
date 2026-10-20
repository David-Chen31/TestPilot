import base64
import json
from io import BytesIO
from pathlib import Path

import pytest

from claude_converter.utils import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    YELLOW,
    DEFAULT_ROLE_COLORS,
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


# --------------------------------------------------------------------------- #
# Constants / helpers
# --------------------------------------------------------------------------- #

def test_color_constants():
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


# --------------------------------------------------------------------------- #
# load_jsonl
# --------------------------------------------------------------------------- #

def test_load_jsonl_basic(tmp_path):
    p = tmp_path / "session.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_accepts_str_path(tmp_path):
    p = tmp_path / "session.jsonl"
    p.write_text('{"x": 1}\n', encoding="utf-8")
    assert load_jsonl(str(p)) == [{"x": 1}]


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "session.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_skips_invalid_lines_and_warns(tmp_path, capsys):
    p = tmp_path / "session.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
    records = load_jsonl(p)
    assert records == [{"a": 1}, {"b": 2}]
    err = capsys.readouterr().err
    assert "Line 2 is invalid JSON" in err


def test_load_jsonl_json_extension(tmp_path):
    p = tmp_path / "session.json"
    p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_missing_file(tmp_path):
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        load_jsonl(missing)


def test_load_jsonl_bad_extension(tmp_path):
    p = tmp_path / "session.txt"
    p.write_text('{"a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_jsonl(p)
    assert "Unrecognized extension" in str(excinfo.value)


def test_load_jsonl_empty_file(tmp_path):
    p = tmp_path / "session.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_jsonl(p)
    assert "empty or contains no valid records" in str(excinfo.value)


def test_load_jsonl_all_invalid(tmp_path, capsys):
    p = tmp_path / "session.jsonl"
    p.write_text("garbage\nalso bad\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(p)
    err = capsys.readouterr().err
    assert "invalid JSON" in err


# --------------------------------------------------------------------------- #
# convert_base64_to_pil_image
# --------------------------------------------------------------------------- #

def _png_bytes() -> bytes:
    PIL = pytest.importorskip("PIL.Image")
    buf = BytesIO()
    PIL.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_convert_base64_bare_string():
    pytest.importorskip("PIL")
    raw = _png_bytes()
    uri = base64.b64encode(raw).decode()
    img = convert_base64_to_pil_image(uri)
    assert img.format == "PNG"
    assert img.size == (2, 2)


def test_convert_base64_data_uri():
    pytest.importorskip("PIL")
    raw = _png_bytes()
    uri = "data:image/png;base64," + base64.b64encode(raw).decode()
    img = convert_base64_to_pil_image(uri)
    assert img.format == "PNG"


def test_convert_base64_invalid_payload():
    pytest.importorskip("PIL")
    with pytest.raises(ValueError) as excinfo:
        convert_base64_to_pil_image("not-a-valid-image-payload")
    assert "Could not decode base64 image" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# _pil_to_data_uri / messages_to_json_safe
# --------------------------------------------------------------------------- #

def test_messages_to_json_safe_text_only_passthrough():
    messages = [{"role": "user", "content": "hi"}]
    out = messages_to_json_safe(messages)
    assert out == [{"role": "user", "content": "hi"}]


def test_messages_to_json_safe_does_not_mutate_original():
    messages = [{"role": "user", "content": "hi"}]
    out = messages_to_json_safe(messages)
    out[0]["content"] = "changed"
    assert messages[0]["content"] == "hi"


def test_messages_to_json_safe_converts_pil_image():
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new("RGB", (2, 2), (0, 0, 255))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "image": img},
            ],
        }
    ]
    out = messages_to_json_safe(messages)
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "hello"}
    img_part = parts[1]
    assert img_part["type"] == "image"
    assert isinstance(img_part["image"], str)
    assert img_part["image"].startswith("data:image/png;base64,")
    # Bytes must be valid base64 decodable
    b64 = img_part["image"].split(",", 1)[1]
    assert len(base64.b64decode(b64)) > 0
    # Original still holds real PIL image
    assert isinstance(messages[0]["content"][1]["image"], PIL.Image)


def test_messages_to_json_safe_non_image_parts_kept():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x"},
                {"type": "audio", "data": "abc"},
            ],
        }
    ]
    out = messages_to_json_safe(messages)
    assert out[0]["content"][1] == {"type": "audio", "data": "abc"}


def test_messages_to_json_safe_result_json_serializable():
    pytest.importorskip("PIL.Image")
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new("RGB", (1, 1))
    messages = [{"role": "user", "content": [{"type": "image", "image": img}]}]
    out = messages_to_json_safe(messages)
    json.dumps(out)  # must not raise


def test_messages_to_json_safe_missing_content():
    messages = [{"role": "user"}]
    out = messages_to_json_safe(messages)
    assert out == [{"role": "user", "content": None}]


# --------------------------------------------------------------------------- #
# merge_content_parts
# --------------------------------------------------------------------------- #

def test_merge_content_parts_appends():
    existing = [{"type": "text", "text": "a"}]
    new = [{"type": "image", "image": "img"}]
    merge_content_parts(existing, new)
    assert existing == [
        {"type": "text", "text": "a"},
        {"type": "image", "image": "img"},
    ]


def test_merge_content_parts_joins_adjacent_text():
    existing = [{"type": "text", "text": "a"}]
    new = [{"type": "text", "text": "b"}]
    merge_content_parts(existing, new)
    assert existing == [{"type": "text", "text": "a\nb"}]


def test_merge_content_parts_copies_dicts():
    existing = []
    part = {"type": "text", "text": "x"}
    merge_content_parts(existing, [part])
    part["text"] = "changed"
    assert existing[0]["text"] == "x"


def test_merge_content_parts_multiple_text_then_image():
    existing = []
    merge_content_parts(existing, [{"type": "text", "text": "a"}])
    merge_content_parts(existing, [{"type": "text", "text": "b"}])
    merge_content_parts(existing, [{"type": "image", "image": "i"}])
    merge_content_parts(existing, [{"type": "text", "text": "c"}])
    assert existing == [
        {"type": "text", "text": "a\nb"},
        {"type": "image", "image": "i"},
        {"type": "text", "text": "c"},
    ]


# --------------------------------------------------------------------------- #
# finalize_content
# --------------------------------------------------------------------------- #

def test_finalize_content_text_only_returns_string():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert finalize_content(parts) == "a\nb"


def test_finalize_content_single_text():
    assert finalize_content([{"type": "text", "text": "solo"}]) == "solo"


def test_finalize_content_with_image_returns_list():
    parts = [{"type": "text", "text": "a"}, {"type": "image", "image": "x"}]
    assert finalize_content(parts) == parts


def test_finalize_content_empty_list_returns_empty_string():
    assert finalize_content([]) == ""


# --------------------------------------------------------------------------- #
# dget / dlist
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("container", [None, [], "string", 42, 3.14])
def test_dget_non_dict_returns_empty(container):
    assert dget(container, "key") == {}


def test_dget_present_dict():
    assert dget({"k": {"a": 1}}, "k") == {"a": 1}


def test_dget_missing_returns_empty():
    assert dget({}, "k") == {}


def test_dget_explicit_none_returns_empty():
    assert dget({"k": None}, "k") == {}


def test_dget_non_dict_value_returns_empty():
    assert dget({"k": [1, 2, 3]}, "k") == {}


@pytest.mark.parametrize("container", [None, [], "s", 0])
def test_dlist_non_dict_returns_empty(container):
    assert dlist(container, "key") == []


def test_dlist_present():
    assert dlist({"k": [1, 2]}, "k") == [1, 2]


def test_dlist_missing_returns_empty():
    assert dlist({}, "k") == []


def test_dlist_explicit_none_returns_empty():
    assert dlist({"k": None}, "k") == []


def test_dlist_non_list_value_returns_empty():
    assert dlist({"k": {"a": 1}}, "k") == []


# --------------------------------------------------------------------------- #
# InspectorSchema
# --------------------------------------------------------------------------- #

def _make_schema(**overrides):
    defaults = dict(
        label="TEST",
        record_type_of=lambda r: r.get("type", "?"),
        is_message=lambda r: r.get("type") == "msg",
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
    )
    defaults.update(overrides)
    return InspectorSchema(**defaults)


def test_inspector_schema_defaults():
    schema = _make_schema()
    assert schema.label == "TEST"
    assert schema.timestamp_of({"anything": 1}) == ""
    assert schema.role_colors == dict(DEFAULT_ROLE_COLORS)
    assert schema.total_tokens_of is None
    assert schema.tokens_of is None
    assert schema.block_types_of is None
    assert schema.blocks_of is None
    assert schema.render_block is None


def test_inspector_schema_role_colors_independent():
    s1 = _make_schema()
    s2 = _make_schema()
    s1.role_colors["user"] = RED
    assert s2.role_colors["user"] == GREEN


# --------------------------------------------------------------------------- #
# Print functions
# --------------------------------------------------------------------------- #

def test_print_header(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    records = [{"a": 1}, {"b": 2}]
    print_header(p, records, _make_schema())
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert "File" in out
    assert "Size" in out
    assert "Lines" in out
    assert "2" in out


def test_print_record_types(capsys):
    schema = _make_schema(record_type_of=lambda r: r.get("type"))
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


def test_print_block_summary_noop_without_block_types(capsys):
    print_block_summary([{"type": "msg"}], _make_schema())
    assert capsys.readouterr().out == ""


def test_print_block_summary(capsys):
    schema = _make_schema(
        block_types_of=lambda r: r.get("blocks", []),
    )
    records = [
        {"type": "msg", "role": "user", "blocks": ["text"]},
        {"type": "msg", "role": "user", "blocks": ["text", "image"]},
        {"type": "msg", "role": "assistant", "blocks": ["tool_use"]},
        {"type": "raw", "blocks": ["meta"]},
    ]
    print_block_summary(records, schema)
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" in out
    assert "user" in out
    assert "assistant" in out
    assert "raw" in out


def test_print_message_roles(capsys):
    schema = _make_schema()
    records = [
        {"type": "msg", "role": "user"},
        {"type": "msg", "role": "assistant"},
        {"type": "msg", "role": "assistant"},
    ]
    print_message_roles(records, schema)
    out = capsys.readouterr().out
    assert "MESSAGE ROLES" in out
    assert "user" in out
    assert "assistant" in out


def test_print_message_roles_empty_noop(capsys):
    print_message_roles([], _make_schema())
    assert capsys.readouterr().out == ""


def test_print_token_usage_noop_without_callable(capsys):
    print_token_usage([], _make_schema())
    assert capsys.readouterr().out == ""


def test_print_token_usage_noop_empty_totals(capsys):
    schema = _make_schema(total_tokens_of=lambda r: {})
    print_token_usage([{"type": "msg"}], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage(capsys):
    schema = _make_schema(total_tokens_of=lambda r: {"input_tokens": 1234, "output_tokens": 5})
    print_token_usage([{"type": "msg"}], schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" in out
    assert "Input Tokens" in out
    assert "1,234" in out
    assert "Output Tokens" in out


def test_print_flow_empty(capsys):
    print_flow([], _make_schema())
    assert capsys.readouterr().out == ""


def test_print_flow_basic(capsys):
    schema = _make_schema(
        timestamp_of=lambda r: r.get("ts", ""),
        tokens_of=lambda r: r.get("tokens", 0),
    )
    records = [{"type": "msg", "role": "user", "text": "hello", "ts": "00:00", "tokens": 12}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "[USER]" in out
    assert "00:00" in out
    assert "12tk" in out


def test_print_flow_with_blocks(capsys):
    schema = _make_schema(
        block_types_of=lambda r: ["text"],
        blocks_of=lambda r: [{"type": "text", "text": "hi"}],
        render_block=lambda b: "RENDERED:" + b["text"],
    )
    records = [{"type": "msg", "role": "user"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "blocks: text" in out
    assert "RENDERED:hi" in out


def test_print_flow_show_text_fallback(capsys):
    schema = _make_schema(text_of=lambda r: "a long text body")
    records = [{"type": "msg", "role": "user"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "a long text body" in out


def test_print_raw_examples(capsys):
    schema = _make_schema()
    records = [{"type": "a", "k": 1}, {"type": "a", "k": 2}, {"type": "b"}]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    assert "type = a" in out
    assert "type = b" in out
    # Only one example per type
    assert out.count("type = a") == 1
    assert out.count("type = b") == 1


# --------------------------------------------------------------------------- #
# run_inspection
# --------------------------------------------------------------------------- #

def test_run_inspection_no_flow(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    records = [{"type": "msg", "role": "user", "text": "hi"}]
    p.write_text('{"type": "msg"}\n', encoding="utf-8")
    schema = _make_schema()
    run_inspection(p, records, schema)
    out = capsys.readouterr().out
    assert "SESSION INSPECTOR" in out
    assert "RECORD TYPES" in out
    # flow only when show_flow
    assert "CONVERSATION FLOW" not in out


def test_run_inspection_with_flow(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    records = [{"type": "msg", "role": "user", "text": "hi"}]
    p.write_text('{}\n', encoding="utf-8")
    schema = _make_schema()
    run_inspection(p, records, schema, show_flow=True)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out


def test_run_inspection_extra_header_called(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    records = [{"type": "msg", "role": "user", "text": "hi"}]
    p.write_text('{}\n', encoding="utf-8")
    schema = _make_schema()
    calls = []

    def extra(recs):
        calls.append(recs)
        print("EXTRA-HEADER")

    run_inspection(p, records, schema, extra_header=extra)
    out = capsys.readouterr().out
    assert calls == [records]
    assert "EXTRA-HEADER" in out


def test_run_inspection_accepts_str_path(tmp_path, capsys):
    p = tmp_path / "f.jsonl"
    records = []
    p.write_text("", encoding="utf-8")
    schema = _make_schema()
    run_inspection(str(p), records, schema)
    out = capsys.readouterr().out
    assert "SESSION INSPECTOR" in out
