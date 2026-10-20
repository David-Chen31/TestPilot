from __future__ import annotations

import base64
import json
import sys
import types

import pytest

import claude_converter.utils as utils
from claude_converter.utils import (
    DEFAULT_ROLE_COLORS,
    RESET,
    BOLD,
    DIM,
    CYAN,
    GREEN,
    YELLOW,
    RED,
    MAGENTA,
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
    _c,
    _hr,
    _truncate,
    _scaled_bar,
    _pil_to_data_uri,
)


# ---------------------------------------------------------------------------
# Constants / small helpers
# ---------------------------------------------------------------------------

def test_constants_present():
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


def test_c_wraps_text():
    assert _c("hi", BOLD) == BOLD + "hi" + RESET
    assert _c(42) == "42" + RESET
    assert _c("x", BOLD, CYAN) == BOLD + CYAN + "x" + RESET


def test_hr_default_and_custom():
    assert _hr() == _c("-" * 70, DIM)
    assert _hr("=", 3) == _c("===", DIM)


def test_truncate_short_no_truncation():
    assert _truncate("hello", 10) == "hello"


def test_truncate_escapes_newlines():
    assert _truncate("a\nb", 10) == "a\\nb"


def test_truncate_long_appends_dim_ellipsis():
    out = _truncate("a" * 20, 5)
    assert out == "aaaaa" + DIM + "..." + RESET


def test_truncate_coerces_non_str():
    assert _truncate(123, 10) == "123"


def test_truncate_default_max_len_is_120():
    # Exactly 120 chars -> no truncation; 121 chars -> truncated.
    assert _truncate("a" * 120) == "a" * 120
    out = _truncate("a" * 121)
    assert out == "a" * 120 + DIM + "..." + RESET


def test_truncate_exactly_max_len_not_truncated():
    # Boundary: len == max_len must NOT append ellipsis (uses strict >).
    assert _truncate("abcde", 5) == "abcde"
    # len == max_len + 1 must truncate.
    out = _truncate("abcdef", 5)
    assert out == "abcde" + DIM + "..." + RESET


def test_scaled_bar_zero_max():
    assert _scaled_bar(5, 0) == ""
    assert _scaled_bar(0, -1) == ""


def test_scaled_bar_proportional():
    # 10/10 * 40 == 40
    assert _scaled_bar(10, 10) == "█" * 40
    # 5/10 * 40 == 20
    assert _scaled_bar(5, 10) == "█" * 20


def test_scaled_bar_minimum_one():
    # tiny count still gives at least one block
    assert _scaled_bar(1, 1000) == "█"


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

def test_load_jsonl_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_jsonl(tmp_path / "nope.jsonl")


def test_load_jsonl_bad_extension(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("{}\n")
    with pytest.raises(ValueError, match="Unrecognized extension"):
        load_jsonl(p)


def test_load_jsonl_bad_extension_message(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("{}\n")
    with pytest.raises(ValueError) as exc:
        load_jsonl(p)
    assert str(exc.value) == "Unrecognized extension: .txt. Expected .jsonl or .json"


def test_load_jsonl_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="no valid records"):
        load_jsonl(p)


def test_load_jsonl_empty_file_exact_message(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError) as exc:
        load_jsonl(p)
    assert str(exc.value) == "File is empty or contains no valid records."


def test_load_jsonl_only_blank_lines(tmp_path):
    p = tmp_path / "blank.jsonl"
    p.write_text("\n\n   \n")
    with pytest.raises(ValueError, match="no valid records"):
        load_jsonl(p)


def test_load_jsonl_valid_records(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n')
    records = load_jsonl(p)
    assert records == [{"a": 1}, {"b": 2}]


def test_load_jsonl_json_suffix_accepted(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"a": 1}\n')
    assert load_jsonl(p) == [{"a": 1}]


def test_load_jsonl_reads_utf8_encoded_content(tmp_path):
    # Non-ASCII written as utf-8; load must decode as utf-8.
    p = tmp_path / "data.jsonl"
    p.write_bytes('{"msg": "café ☕"}\n'.encode("utf-8"))
    records = load_jsonl(p)
    assert records == [{"msg": "café ☕"}]


def test_load_jsonl_skips_invalid_lines_and_warns(tmp_path, capsys):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
    records = load_jsonl(p)
    assert records == [{"a": 1}, {"b": 2}]
    err = capsys.readouterr().err
    assert "Line 2 is invalid JSON" in err
    assert err.startswith("X Line 2 is invalid JSON:")


def test_load_jsonl_invalid_line_number(tmp_path, capsys):
    # Line numbers count blank lines too (enumerate on raw file).
    p = tmp_path / "data.jsonl"
    p.write_text("\n\nbad\n")
    with pytest.raises(ValueError):
        load_jsonl(p)
    err = capsys.readouterr().err
    assert "Line 3 is invalid JSON" in err


def test_load_jsonl_invalid_then_none_valid(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("garbage\n")
    with pytest.raises(ValueError, match="no valid records"):
        load_jsonl(p)


# ---------------------------------------------------------------------------
# convert_base64_to_pil_image
# ---------------------------------------------------------------------------

PNG_BYTES_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def test_convert_base64_raises_import_error_without_pillow():
    # With no Pillow installed, always raises the helpful ImportError.
    b64 = base64.b64encode(PNG_BYTES_1x1).decode()
    with pytest.raises(ImportError, match="Pillow is required"):
        convert_base64_to_pil_image(b64)


def test_convert_base64_import_error_exact_message():
    b64 = base64.b64encode(PNG_BYTES_1x1).decode()
    with pytest.raises(ImportError) as exc:
        convert_base64_to_pil_image(b64)
    assert str(exc.value) == (
        "Pillow is required to decode embedded images. "
        "Install it with: pip install claude-converter[images]"
    )


def test_convert_base64_data_uri_raises_import_error_without_pillow():
    b64 = base64.b64encode(PNG_BYTES_1x1).decode()
    with pytest.raises(ImportError, match="Pillow is required"):
        convert_base64_to_pil_image(f"data:image/png;base64,{b64}")


def test_convert_base64_missing_pillow(monkeypatch):
    # Force the lazy `from PIL import Image` inside the function to fail.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="Pillow is required"):
        convert_base64_to_pil_image("abc")


class _FakePILImage:
    """Stand-in for PIL.Image.Image returned by a fake PIL module."""

    format = "PNG"

    def __init__(self, data):
        self.data = data
        self.size = (1, 1)

    def save(self, buf, format=None):
        buf.write(self.data)

    @classmethod
    def open(cls, fp):
        return cls(fp.read())


def _install_fake_pil(monkeypatch):
    """Install a fake `PIL` package so `from PIL import Image` succeeds."""
    pil_mod = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.Image = _FakePILImage
    image_mod.open = _FakePILImage.open
    pil_mod.Image = image_mod
    monkeypatch.setitem(sys.modules, "PIL", pil_mod)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)


def test_convert_base64_bare_string_with_fake_pil(monkeypatch):
    _install_fake_pil(monkeypatch)
    b64 = base64.b64encode(PNG_BYTES_1x1).decode()
    img = convert_base64_to_pil_image(b64)
    assert isinstance(img, _FakePILImage)
    assert img.data == PNG_BYTES_1x1


def test_convert_base64_data_uri_with_fake_pil(monkeypatch):
    _install_fake_pil(monkeypatch)
    b64 = base64.b64encode(PNG_BYTES_1x1).decode()
    img = convert_base64_to_pil_image(f"data:image/png;base64,{b64}")
    assert isinstance(img, _FakePILImage)
    assert img.data == PNG_BYTES_1x1


def test_convert_base64_data_uri_splits_on_first_comma(monkeypatch):
    # Only the first comma should be treated as the prefix separator.
    _install_fake_pil(monkeypatch)
    payload = base64.b64encode(PNG_BYTES_1x1).decode()
    img = convert_base64_to_pil_image(f"prefix,{payload}")
    assert img.data == PNG_BYTES_1x1


def test_convert_base64_invalid_raises_value_error_with_fake_pil(monkeypatch):
    _install_fake_pil(monkeypatch)

    # Make Image.open raise so the wrapper raises ValueError.
    def bad_open(fp):
        raise OSError("bad image")

    monkeypatch.setattr(sys.modules["PIL.Image"], "open", bad_open)
    monkeypatch.setattr(_FakePILImage, "open", staticmethod(bad_open))
    with pytest.raises(ValueError, match="Could not decode base64 image"):
        convert_base64_to_pil_image(base64.b64encode(b"not an image").decode())


def test_convert_base64_value_error_message_includes_cause(monkeypatch):
    _install_fake_pil(monkeypatch)

    def bad_open(fp):
        raise OSError("bad image")

    monkeypatch.setattr(sys.modules["PIL.Image"], "open", bad_open)
    monkeypatch.setattr(_FakePILImage, "open", staticmethod(bad_open))
    with pytest.raises(ValueError) as exc:
        convert_base64_to_pil_image(base64.b64encode(b"x").decode())
    assert str(exc.value) == "Could not decode base64 image: bad image"


# ---------------------------------------------------------------------------
# _pil_to_data_uri
# ---------------------------------------------------------------------------

def test_pil_to_data_uri_png_with_fake_pil(monkeypatch):
    _install_fake_pil(monkeypatch)
    img = _FakePILImage(PNG_BYTES_1x1)
    uri = _pil_to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")
    payload = uri.split(",", 1)[1]
    assert base64.b64decode(payload) == PNG_BYTES_1x1


def test_pil_to_data_uri_lowercases_format(monkeypatch):
    _install_fake_pil(monkeypatch)
    img = _FakePILImage(PNG_BYTES_1x1)
    uri = _pil_to_data_uri(img)
    # Entire URI prefix must be exactly this (format lowercased)
    head = uri.split(",", 1)[0]
    assert head == "data:image/png;base64"


def test_pil_to_data_uri_fallback_on_save_error():
    class FakeImage:
        format = "BOGUSFMT"

        def __init__(self):
            self.calls = []

        def save(self, buf, format=None):
            self.calls.append(format)
            if format == "BOGUSFMT":
                raise ValueError("cannot save")
            buf.write(PNG_BYTES_1x1)

    fake = FakeImage()
    uri = _pil_to_data_uri(fake)
    assert uri.startswith("data:image/png;base64,")
    assert fake.calls == ["BOGUSFMT", "PNG"]


def test_pil_to_data_uri_fallback_does_not_pollute_first_buffer():
    # Second save must write to a fresh buffer (fallback resets buf).
    class FakeImage:
        format = "BOGUSFMT"

        def __init__(self):
            self.first_buf = None

        def save(self, buf, format=None):
            if format == "BOGUSFMT":
                self.first_buf = buf
                buf.write(b"garbage-prefix")
                raise ValueError("cannot save")
            # On fallback save, the buffer must be empty (fresh BytesIO).
            assert buf.getvalue() == b""
            buf.write(PNG_BYTES_1x1)

    uri = _pil_to_data_uri(FakeImage())
    assert base64.b64decode(uri.split(",", 1)[1]) == PNG_BYTES_1x1


def test_pil_to_data_uri_no_format_defaults_png():
    class FakeImage:
        format = None

        def save(self, buf, format=None):
            assert format == "PNG"
            buf.write(PNG_BYTES_1x1)

    uri = _pil_to_data_uri(FakeImage())
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == PNG_BYTES_1x1


def test_pil_to_data_uri_empty_format_defaults_png():
    class FakeImage:
        format = ""

        def save(self, buf, format=None):
            assert format == "PNG"
            buf.write(PNG_BYTES_1x1)

    uri = _pil_to_data_uri(FakeImage())
    assert uri.startswith("data:image/png;base64,")


def test_pil_to_data_uri_jpeg_format(monkeypatch):
    _install_fake_pil(monkeypatch)

    class JpegImage(_FakePILImage):
        format = "JPEG"

    uri = _pil_to_data_uri(JpegImage(PNG_BYTES_1x1))
    assert uri.split(",", 1)[0] == "data:image/jpeg;base64"


# ---------------------------------------------------------------------------
# messages_to_json_safe
# ---------------------------------------------------------------------------

def test_messages_to_json_safe_non_list_content_unchanged():
    messages = [{"role": "user", "content": "hello"}]
    out = messages_to_json_safe(messages)
    assert out == [{"role": "user", "content": "hello"}]
    # new dict, not the same object
    assert out[0] is not messages[0]


def test_messages_to_json_safe_text_parts_unchanged():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    out = messages_to_json_safe(messages)
    assert out == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_messages_to_json_safe_image_part_with_pil_absent_uses_repr():
    # No PIL: image value replaced by repr(image).
    obj = object()
    messages = [{"role": "user", "content": [{"type": "image", "image": obj}]}]
    out = messages_to_json_safe(messages)
    assert out[0]["content"][0] == {"type": "image", "image": repr(obj)}


def test_messages_to_json_safe_image_part_without_image_key_kept():
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    out = messages_to_json_safe(messages)
    assert out[0]["content"] == [{"type": "image"}]


def test_messages_to_json_safe_non_pil_image_value_uses_repr():
    # Without PIL installed, a plain string in "image" gets repr()'d too,
    # because the isinstance check cannot run.
    messages = [{"role": "user", "content": [{"type": "image", "image": "already-a-uri"}]}]
    out = messages_to_json_safe(messages)
    assert out[0]["content"] == [
        {"type": "image", "image": repr("already-a-uri")}
    ]


def test_messages_to_json_safe_pillow_missing_uses_repr(monkeypatch):
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    obj = object()
    messages = [{"role": "user", "content": [{"type": "image", "image": obj}]}]
    out = messages_to_json_safe(messages)
    assert out[0]["content"][0] == {"type": "image", "image": repr(obj)}


def test_messages_to_json_safe_converts_pil_with_fake_pil(monkeypatch):
    _install_fake_pil(monkeypatch)
    img = _FakePILImage(PNG_BYTES_1x1)
    messages = [{"role": "user", "content": [{"type": "image", "image": img}]}]
    out = messages_to_json_safe(messages)
    # In-memory object untouched
    assert messages[0]["content"][0]["image"] is img
    part = out[0]["content"][0]
    assert part["type"] == "image"
    assert part["image"].startswith("data:image/png;base64,")
    assert base64.b64decode(part["image"].split(",", 1)[1]) == PNG_BYTES_1x1


def test_messages_to_json_safe_multiple_parts_continues_after_pil(monkeypatch):
    # With PIL present, a converted image part must NOT break out of the loop
    # and subsequent parts must still be preserved.
    _install_fake_pil(monkeypatch)
    img = _FakePILImage(PNG_BYTES_1x1)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "after"},
        ],
    }]
    out = messages_to_json_safe(messages)
    parts = out[0]["content"]
    assert len(parts) == 2
    assert parts[0]["image"].startswith("data:image/png;base64,")
    assert parts[1] == {"type": "text", "text": "after"}


def test_messages_to_json_safe_continues_after_repr_fallback(monkeypatch):
    # Without PIL, after the repr fallback the loop must continue.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": object()},
            {"type": "text", "text": "after"},
        ],
    }]
    out = messages_to_json_safe(messages)
    assert out[0]["content"][1] == {"type": "text", "text": "after"}
    assert len(out[0]["content"]) == 2


def test_messages_to_json_safe_missing_content_key():
    messages = [{"role": "user"}]
    out = messages_to_json_safe(messages)
    assert out == [{"role": "user", "content": None}]


def test_messages_to_json_safe_preserves_other_keys():
    messages = [{"role": "user", "content": "hi", "extra": {"k": [1, 2]}}]
    out = messages_to_json_safe(messages)
    assert out == [{"role": "user", "content": "hi", "extra": {"k": [1, 2]}}]


def test_messages_to_json_safe_json_serializable():
    # With a plain-object image value replaced by repr, output is JSON-safe.
    obj = object()
    messages = [
        {"role": "user", "content": [{"type": "image", "image": obj}]},
        {"role": "assistant", "content": "hello"},
    ]
    out = messages_to_json_safe(messages)
    json.dumps(out)


# ---------------------------------------------------------------------------
# merge_content_parts
# ---------------------------------------------------------------------------

def test_merge_content_parts_text_joined_with_newline():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [{"type": "text", "text": "b"}])
    assert existing == [{"type": "text", "text": "a\nb"}]


def test_merge_content_parts_appends_non_text():
    existing = [{"type": "text", "text": "a"}]
    part = {"type": "image", "image": "x"}
    merge_content_parts(existing, [part])
    assert existing == [{"type": "text", "text": "a"}, {"type": "image", "image": "x"}]
    # copy, not same object
    assert existing[1] is not part


def test_merge_content_parts_empty_existing():
    existing = []
    merge_content_parts(existing, [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert existing == [{"type": "text", "text": "a\nb"}]


def test_merge_content_parts_does_not_merge_nonadjacent_text():
    existing = [{"type": "image", "image": "x"}]
    merge_content_parts(existing, [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert existing == [
        {"type": "image", "image": "x"},
        {"type": "text", "text": "a\nb"},
    ]


def test_merge_content_parts_text_then_image_then_text():
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


def test_merge_content_parts_empty_new_noop():
    existing = [{"type": "text", "text": "a"}]
    merge_content_parts(existing, [])
    assert existing == [{"type": "text", "text": "a"}]


# ---------------------------------------------------------------------------
# finalize_content
# ---------------------------------------------------------------------------

def test_finalize_content_text_only_returns_string():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert finalize_content(parts) == "a\nb"


def test_finalize_content_single_text_returns_string():
    assert finalize_content([{"type": "text", "text": "only"}]) == "only"


def test_finalize_content_with_image_returns_list():
    parts = [
        {"type": "text", "text": "hi"},
        {"type": "image", "image": "x"},
    ]
    assert finalize_content(parts) is parts


def test_finalize_content_empty_list():
    assert finalize_content([]) == ""


# ---------------------------------------------------------------------------
# dget / dlist
# ---------------------------------------------------------------------------

def test_dget_normal():
    assert dget({"a": {"b": 1}}, "a") == {"b": 1}


def test_dget_missing_key():
    assert dget({}, "a") == {}


def test_dget_explicit_null():
    assert dget({"a": None}, "a") == {}


def test_dget_non_dict_parent():
    assert dget(None, "a") == {}
    assert dget(["list"], "a") == {}
    assert dget("str", "a") == {}


def test_dget_non_dict_value():
    assert dget({"a": 5}, "a") == {}
    assert dget({"a": [1, 2]}, "a") == {}


def test_dlist_normal():
    assert dlist({"a": [1, 2]}, "a") == [1, 2]


def test_dlist_missing():
    assert dlist({}, "a") == []


def test_dlist_explicit_null():
    assert dlist({"a": None}, "a") == []


def test_dlist_non_list_value():
    assert dlist({"a": "string"}, "a") == []
    assert dlist({"a": {"k": 1}}, "a") == []


def test_dlist_non_dict_parent():
    assert dlist(None, "a") == []
    assert dlist([1, 2], "a") == []


# ---------------------------------------------------------------------------
# InspectorSchema
# ---------------------------------------------------------------------------

def _make_schema(**overrides):
    defaults = dict(
        label="TEST",
        record_type_of=lambda r: r.get("type", "unknown"),
        is_message=lambda r: bool(r.get("is_msg", False)),
        role_of=lambda r: r.get("role", "user"),
        text_of=lambda r: r.get("text", ""),
    )
    defaults.update(overrides)
    return InspectorSchema(**defaults)


def test_inspector_schema_defaults():
    schema = _make_schema()
    assert schema.label == "TEST"
    assert schema.timestamp_of({}) == ""
    assert schema.role_colors == DEFAULT_ROLE_COLORS
    # default_factory -> fresh dict, not the shared constant
    assert schema.role_colors is not DEFAULT_ROLE_COLORS
    assert schema.total_tokens_of is None
    assert schema.tokens_of is None
    assert schema.block_types_of is None
    assert schema.blocks_of is None
    assert schema.render_block is None


def test_inspector_schema_role_colors_are_per_instance():
    s1 = _make_schema()
    s2 = _make_schema()
    s1.role_colors["user"] = "custom"
    assert s2.role_colors["user"] == GREEN
    assert DEFAULT_ROLE_COLORS["user"] == GREEN


def test_inspector_schema_stores_callables():
    rt = lambda r: "T"
    im = lambda r: True
    schema = InspectorSchema(
        label="L",
        record_type_of=rt,
        is_message=im,
        role_of=lambda r: "user",
        text_of=lambda r: "",
        total_tokens_of=lambda rs: {"x": 1},
        tokens_of=lambda r: 5,
        block_types_of=lambda r: ["text"],
        blocks_of=lambda r: [],
        render_block=lambda b: "x",
    )
    assert schema.record_type_of is rt
    assert schema.is_message is im
    assert schema.total_tokens_of([{}]) == {"x": 1}
    assert schema.tokens_of({}) == 5
    assert schema.block_types_of({}) == ["text"]
    assert schema.blocks_of({}) == []
    assert schema.render_block({}) == "x"


# ---------------------------------------------------------------------------
# print_header
# ---------------------------------------------------------------------------

def test_print_header(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("x\n")
    schema = _make_schema()
    print_header(p, [{"a": 1}, {"a": 2}], schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert "s.jsonl" in out
    assert "Lines : " in out
    assert "2" in out
    assert "KB" in out


def test_print_header_size_value(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_bytes(b"x" * 2048)
    schema = _make_schema()
    print_header(p, [], schema)
    out = capsys.readouterr().out
    assert "2.0 KB" in out


def test_print_header_size_scales_in_kb(tmp_path, capsys):
    # 1024 bytes -> 1.0 KB (division, not multiplication)
    p = tmp_path / "s.jsonl"
    p.write_bytes(b"x" * 1024)
    print_header(p, [], _make_schema())
    out = capsys.readouterr().out
    assert "1.0 KB" in out


def test_print_header_exact_formatting(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_bytes(b"x" * 1536)  # 1.5 KB
    print_header(p, [{}, {}, {}], _make_schema())
    out = capsys.readouterr().out
    lines = out.splitlines()
    # Header: blank, "=" hr, title, "=" hr, File, Size, Lines
    assert lines[0] == ""
    assert lines[1] == _hr("=")
    assert lines[2] == _c("  TEST SESSION INSPECTOR", BOLD + CYAN)
    assert lines[3] == _hr("=")
    assert lines[4] == f"  File  : {_c(str(p), BOLD)}"
    assert lines[5] == f"  Size  : {_c('1.5 KB', BOLD)}"
    assert lines[6] == f"  Lines : {_c(3, BOLD)}"


# ---------------------------------------------------------------------------
# print_record_types
# ---------------------------------------------------------------------------

def test_print_record_types(capsys):
    schema = _make_schema()
    records = [{"type": "a"}, {"type": "a"}, {"type": "b"}]
    print_record_types(records, schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out
    assert "a" in out and "b" in out


def test_print_record_types_counts(capsys):
    schema = _make_schema()
    records = [{"type": "a"}, {"type": "a"}, {"type": "b"}]
    print_record_types(records, schema)
    out = capsys.readouterr().out
    # "a" appears before "b" (most_common), with count 2
    assert BOLD + "2" in out
    assert BOLD + "1" in out


def test_print_record_types_bar_scaled_to_max(capsys):
    # "a" (count 4) is max -> full 40 blocks; "b" (count 2) -> 20 blocks.
    schema = _make_schema()
    records = ([{"type": "a"}] * 4) + ([{"type": "b"}] * 2)
    print_record_types(records, schema)
    out = capsys.readouterr().out
    assert _c("█" * 40, DIM) in out
    assert _c("█" * 20, DIM) in out


def test_print_record_types_uses_role_colors(capsys):
    schema = _make_schema()
    # record type named "user" -> colored GREEN
    records = [{"type": "user"}]
    print_record_types(records, schema)
    out = capsys.readouterr().out
    assert _c("user".ljust(28), GREEN) in out


def test_print_record_types_empty(capsys):
    schema = _make_schema()
    print_record_types([], schema)
    out = capsys.readouterr().out
    assert "RECORD TYPES" in out


def test_print_record_types_empty_no_crash(capsys):
    # max_count stays 0 for empty -> no bar
    print_record_types([], _make_schema())
    out = capsys.readouterr().out
    assert "█" not in out


# ---------------------------------------------------------------------------
# print_block_summary
# ---------------------------------------------------------------------------

def test_print_block_summary_noop_without_block_types(capsys):
    schema = _make_schema()
    print_block_summary([{"is_msg": True}], schema)
    assert capsys.readouterr().out == ""


def test_print_block_summary_groups_by_role(capsys):
    schema = _make_schema(
        block_types_of=lambda r: r.get("blocks", []),
    )
    records = [
        {"is_msg": True, "role": "user", "blocks": ["text", "tool_use"]},
        {"is_msg": True, "role": "user", "blocks": ["text"]},
        {"is_msg": False, "type": "meta", "blocks": ["summary"]},
    ]
    print_block_summary(records, schema)
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" in out
    assert "user" in out and "meta" in out
    assert "text" in out and "tool_use" in out and "summary" in out


def test_print_block_summary_counts(capsys):
    schema = _make_schema(block_types_of=lambda r: r.get("blocks", []))
    records = [
        {"is_msg": True, "role": "user", "blocks": ["text", "tool_use"]},
        {"is_msg": True, "role": "user", "blocks": ["text"]},
    ]
    print_block_summary(records, schema)
    out = capsys.readouterr().out
    # text count 2, tool_use count 1
    assert f"{_c('user', GREEN + BOLD)}" in out
    assert f"{'text'.ljust(20)} {_c(2, BOLD)}" in out
    assert f"{'tool_use'.ljust(20)} {_c(1, BOLD)}" in out


def test_print_block_summary_empty_records(capsys):
    schema = _make_schema(block_types_of=lambda r: [])
    print_block_summary([], schema)
    assert capsys.readouterr().out == ""


def test_print_block_summary_role_with_no_blocks_skipped(capsys):
    schema = _make_schema(block_types_of=lambda r: r.get("blocks", []))
    records = [{"is_msg": True, "role": "user", "blocks": []}]
    print_block_summary(records, schema)
    # Header still printed but role line skipped
    out = capsys.readouterr().out
    assert "CONTENT BLOCKS" in out
    assert "·" not in out


# ---------------------------------------------------------------------------
# print_message_roles
# ---------------------------------------------------------------------------

def test_print_message_roles_no_messages(capsys):
    schema = _make_schema()
    print_message_roles([{"is_msg": False}], schema)
    assert capsys.readouterr().out == ""


def test_print_message_roles_counts(capsys):
    schema = _make_schema()
    records = [
        {"is_msg": True, "role": "user"},
        {"is_msg": True, "role": "user"},
        {"is_msg": True, "role": "assistant"},
        {"is_msg": False, "role": "system"},
    ]
    print_message_roles(records, schema)
    out = capsys.readouterr().out
    assert "MESSAGE ROLES" in out
    assert "user" in out
    assert "assistant" in out
    # non-message system role should be absent
    assert "system" not in out


def test_print_message_roles_bar_scaling(capsys):
    schema = _make_schema()
    records = ([{"is_msg": True, "role": "user"}] * 4) + (
        [{"is_msg": True, "role": "assistant"}] * 2
    )
    print_message_roles(records, schema)
    out = capsys.readouterr().out
    assert _c("█" * 40, GREEN) in out
    assert _c("█" * 20, CYAN) in out


# ---------------------------------------------------------------------------
# print_token_usage
# ---------------------------------------------------------------------------

def test_print_token_usage_noop_without_total(capsys):
    schema = _make_schema()
    print_token_usage([{"a": 1}], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_noop_on_empty_totals(capsys):
    schema = _make_schema(total_tokens_of=lambda r: {})
    print_token_usage([{"a": 1}], schema)
    assert capsys.readouterr().out == ""


def test_print_token_usage_prints_totals(capsys):
    schema = _make_schema(total_tokens_of=lambda r: {"input_tokens": 1000, "output_tokens": 5})
    print_token_usage([{"a": 1}], schema)
    out = capsys.readouterr().out
    assert "TOKEN USAGE" in out
    assert "Input Tokens" in out
    assert "1,000" in out
    assert "Output Tokens" in out
    assert "5" in out


def test_print_token_usage_exact_lines(capsys):
    schema = _make_schema(total_tokens_of=lambda r: {"input_tokens": 1000, "output_tokens": 5})
    print_token_usage([{"a": 1}], schema)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == ""
    assert lines[1] == _c("  TOKEN USAGE", BOLD)
    assert lines[2] == _hr()
    assert lines[3] == f"  {'Input Tokens'.ljust(20)} {_c('1,000', BOLD)}"
    assert lines[4] == f"  {'Output Tokens'.ljust(20)} {_c('5', BOLD)}"


# ---------------------------------------------------------------------------
# print_flow
# ---------------------------------------------------------------------------

def test_print_flow_no_messages(capsys):
    schema = _make_schema()
    print_flow([{"is_msg": False}], schema)
    assert capsys.readouterr().out == ""


def test_print_flow_basic(capsys):
    schema = _make_schema()
    records = [
        {"is_msg": True, "role": "user", "text": "hi"},
        {"is_msg": True, "role": "assistant", "text": "hello"},
    ]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out
    assert "[USER]" in out
    assert "[ASSISTANT]" in out


def test_print_flow_numbering(capsys):
    schema = _make_schema()
    records = [
        {"is_msg": True, "role": "user", "text": "hi"},
        {"is_msg": True, "role": "assistant", "text": "hello"},
    ]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "  1." in out
    assert "  2." in out


def test_print_flow_role_label_padded(capsys):
    schema = _make_schema()
    records = [{"is_msg": True, "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert _c("[USER]".ljust(14), GREEN + BOLD) in out


def test_print_flow_show_text_truncates(capsys):
    schema = _make_schema()
    records = [{"is_msg": True, "role": "user", "text": "hello world"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "hello world" in out


def test_print_flow_show_text_uses_max_200(capsys):
    schema = _make_schema()
    long_text = "a" * 250
    records = [{"is_msg": True, "role": "user", "text": long_text}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    # _truncate(..., 200) -> 200 a's + ellipsis
    assert "a" * 200 + DIM + "..." + RESET in out
    assert "a" * 201 not in out


def test_print_flow_with_block_types(capsys):
    schema = _make_schema(block_types_of=lambda r: ["text", "tool_use"])
    records = [{"is_msg": True, "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "blocks: text, tool_use" in out


def test_print_flow_with_tokens(capsys):
    schema = _make_schema(tokens_of=lambda r: 42)
    records = [{"is_msg": True, "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "42tk" in out


def test_print_flow_zero_tokens_label_omitted(capsys):
    schema = _make_schema(tokens_of=lambda r: 0)
    records = [{"is_msg": True, "role": "user"}]
    print_flow(records, schema)
    out = capsys.readouterr().out
    assert "0tk" not in out


def test_print_flow_render_blocks(capsys):
    schema = _make_schema(
        blocks_of=lambda r: [{"b": 1}, {"b": 2}],
        render_block=lambda b: f"BLOCK-{b['b']}",
    )
    records = [{"is_msg": True, "role": "user", "text": "ignored"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "BLOCK-1" in out
    assert "BLOCK-2" in out
    # text fallback should not be used when blocks rendered
    assert "ignored" not in out


def test_print_flow_render_requires_both_callbacks(capsys):
    # Only blocks_of, no render_block -> falls back to text
    schema = _make_schema(blocks_of=lambda r: [{"b": 1}])
    records = [{"is_msg": True, "role": "user", "text": "shown-text"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "shown-text" in out


def test_print_flow_render_requires_both_callbacks_inverse(capsys):
    # Only render_block, no blocks_of -> falls back to text
    schema = _make_schema(render_block=lambda b: "RENDERED")
    records = [{"is_msg": True, "role": "user", "text": "shown-text"}]
    print_flow(records, schema, show_text=True)
    out = capsys.readouterr().out
    assert "shown-text" in out
    assert "RENDERED" not in out


def test_print_flow_show_text_false_does_not_print_text(capsys):
    schema = _make_schema()
    records = [{"is_msg": True, "role": "user", "text": "hidden-text"}]
    print_flow(records, schema, show_text=False)
    out = capsys.readouterr().out
    assert "hidden-text" not in out


# ---------------------------------------------------------------------------
# print_raw_examples
# ---------------------------------------------------------------------------

def test_print_raw_examples_one_per_type(capsys):
    schema = _make_schema()
    records = [
        {"type": "a", "is_msg": True, "text": "hi", "x": 1},
        {"type": "a", "is_msg": True, "text": "hi2"},
        {"type": "b", "is_msg": False},
    ]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "RAW RECORD EXAMPLES" in out
    assert out.count("type = a") == 1
    assert out.count("type = b") == 1
    # flattened text shown for messages
    assert "hi" in out
    # non-message does not get flattened text line
    assert "flattened text:" in out


def test_print_raw_examples_no_messages(capsys):
    schema = _make_schema()
    records = [{"type": "meta", "is_msg": False}]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "flattened text:" not in out


def test_print_raw_examples_top_level_keys(capsys):
    schema = _make_schema()
    records = [{"type": "a", "is_msg": False, "key1": 1, "key2": 2}]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "top-level keys:" in out
    # Keys rendered as a list repr
    assert str(list(records[0].keys())) in out


def test_print_raw_examples_text_truncated_150(capsys):
    schema = _make_schema()
    records = [{"type": "a", "is_msg": True, "text": "b" * 200}]
    print_raw_examples(records, schema)
    out = capsys.readouterr().out
    assert "b" * 150 + DIM + "..." + RESET in out


# ---------------------------------------------------------------------------
# run_inspection
# ---------------------------------------------------------------------------

def test_run_inspection_full(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("x\n")
    calls = []
    schema = _make_schema(
        total_tokens_of=lambda r: {"input_tokens": 3},
        block_types_of=lambda r: ["text"],
    )
    records = [
        {"is_msg": True, "role": "user", "type": "u", "text": "hello"},
        {"is_msg": True, "role": "assistant", "type": "a", "text": "world"},
    ]
    run_inspection(p, records, schema, extra_header=lambda r: calls.append(list(r)))
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out
    assert "RECORD TYPES" in out
    assert "CONTENT BLOCKS" in out
    assert "MESSAGE ROLES" in out
    assert "TOKEN USAGE" in out
    # show_flow defaults False -> flow not printed
    assert "CONVERSATION FLOW" not in out
    assert calls == [records]


def test_run_inspection_show_flow(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("x\n")
    schema = _make_schema()
    records = [{"is_msg": True, "role": "user", "type": "u", "text": "hello"}]
    run_inspection(p, records, schema, show_flow=True)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" in out


def test_run_inspection_show_blocks_without_flow_ignored(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("x\n")
    schema = _make_schema()
    records = [{"is_msg": True, "role": "user", "type": "u", "text": "hello"}]
    run_inspection(p, records, schema, show_flow=False, show_blocks=True)
    out = capsys.readouterr().out
    assert "CONVERSATION FLOW" not in out


def test_run_inspection_no_extra_header(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("x\n")
    schema = _make_schema()
    run_inspection(p, [], schema)
    out = capsys.readouterr().out
    assert "TEST SESSION INSPECTOR" in out


def test_run_inspection_accepts_str_path(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("x\n")
    schema = _make_schema()
    run_inspection(str(p), [], schema)
    out = capsys.readouterr().out
    assert "s.jsonl" in out


def test_run_inspection_order_of_sections(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("x\n")
    schema = _make_schema(
        total_tokens_of=lambda r: {"input_tokens": 3},
        block_types_of=lambda r: ["text"],
    )
    records = [{"is_msg": True, "role": "user", "type": "u", "text": "hello"}]
    run_inspection(p, records, schema, show_flow=True, show_blocks=True)
    out = capsys.readouterr().out
    i_header = out.index("TEST SESSION INSPECTOR")
    i_types = out.index("RECORD TYPES")
    i_blocks = out.index("CONTENT BLOCKS")
    i_roles = out.index("MESSAGE ROLES")
    i_tokens = out.index("TOKEN USAGE")
    i_flow = out.index("CONVERSATION FLOW")
    assert i_header < i_types < i_blocks < i_roles < i_tokens < i_flow


def test_run_inspection_extra_header_called_after_header(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    p.write_text("x\n")
    order = []
    schema = _make_schema()

    def extra(rs):
        out = capsys.readouterr().out
        order.append(out)

    run_inspection(p, [], schema, extra_header=extra)
    assert len(order) == 1
    assert "TEST SESSION INSPECTOR" in order[0]
