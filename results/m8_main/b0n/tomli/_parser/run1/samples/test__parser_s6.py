from __future__ import annotations

import io
import sys
import warnings
from datetime import date, datetime, time, timedelta, timezone

import pytest

from tomli._parser import (
    ASCII_CTRL,
    BASIC_STR_ESCAPE_REPLACEMENTS,
    BARE_KEY_CHARS,
    DEPRECATED_DEFAULT,
    HEXDIGIT_CHARS,
    ILLEGAL_BASIC_STR_CHARS,
    ILLEGAL_COMMENT_CHARS,
    ILLEGAL_LITERAL_STR_CHARS,
    ILLEGAL_MULTILINE_BASIC_STR_CHARS,
    ILLEGAL_MULTILINE_LITERAL_STR_CHARS,
    KEY_INITIAL_CHARS,
    MAX_INLINE_NESTING,
    MAX_KEY_PARTS,
    TOML_WS,
    TOML_WS_AND_NEWLINE,
    Flags,
    NestedDict,
    Output,
    TOMLDecodeError,
    create_dict_rule,
    create_list_rule,
    is_unicode_scalar_value,
    key_value_rule,
    load,
    loads,
    make_safe_parse_float,
    parse_array,
    parse_basic_str,
    parse_basic_str_escape,
    parse_basic_str_escape_multiline,
    parse_hex_char,
    parse_inline_table,
    parse_key,
    parse_key_part,
    parse_key_value_pair,
    parse_literal_str,
    parse_multiline_str,
    parse_one_line_basic_str,
    parse_value,
    skip_chars,
    skip_comment,
    skip_comments_and_array_ws,
    skip_until,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants_control_chars():
    assert ASCII_CTRL == frozenset(chr(i) for i in range(32)) | {"\x7f"}
    assert ILLEGAL_BASIC_STR_CHARS == ASCII_CTRL - {"\t"}
    assert ILLEGAL_MULTILINE_BASIC_STR_CHARS == ASCII_CTRL - {"\t", "\n"}
    assert ILLEGAL_LITERAL_STR_CHARS == ILLEGAL_BASIC_STR_CHARS
    assert ILLEGAL_MULTILINE_LITERAL_STR_CHARS == ILLEGAL_MULTILINE_BASIC_STR_CHARS
    assert ILLEGAL_COMMENT_CHARS == ILLEGAL_BASIC_STR_CHARS


def test_constants_ws_and_chars():
    assert TOML_WS == frozenset(" \t")
    assert TOML_WS_AND_NEWLINE == frozenset(" \t\n")
    assert BARE_KEY_CHARS == frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    )
    assert KEY_INITIAL_CHARS == BARE_KEY_CHARS | {"'", '"'}
    assert HEXDIGIT_CHARS == frozenset("abcdefABCDEF0123456789")


def test_escape_replacements():
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\b"] == "\u0008"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\t"] == "\u0009"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\n"] == "\u000a"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\f"] == "\u000c"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\r"] == "\u000d"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\e"] == "\u001b"
    assert BASIC_STR_ESCAPE_REPLACEMENTS['\\"'] == "\u0022"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\\\"] == "\u005c"


def test_max_inline_nesting():
    assert MAX_INLINE_NESTING == 400


def test_max_key_parts():
    assert MAX_KEY_PARTS == sys.getrecursionlimit()


# ---------------------------------------------------------------------------
# TOMLDecodeError
# ---------------------------------------------------------------------------


def test_toml_decode_error_basic_position():
    err = TOMLDecodeError("bad", "abc\ndef", 5)
    assert err.msg == "bad"
    assert err.doc == "abc\ndef"
    assert err.pos == 5
    assert err.lineno == 2
    assert err.colno == 2
    assert str(err) == "bad (at line 2, column 2)"


def test_toml_decode_error_first_line():
    err = TOMLDecodeError("msg", "hello", 2)
    assert err.lineno == 1
    assert err.colno == 3
    assert str(err) == "msg (at line 1, column 3)"


def test_toml_decode_error_end_of_document():
    err = TOMLDecodeError("oops", "abc", 3)
    assert err.lineno == 1
    assert err.colno == 4
    assert str(err) == "oops (at end of document)"


def test_toml_decode_error_past_end_of_document():
    err = TOMLDecodeError("oops", "abc", 100)
    assert err.lineno == 1
    assert str(err) == "oops (at end of document)"


def test_toml_decode_error_deprecated_free_form():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        err = TOMLDecodeError("free form", "more", 3, "extra")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert isinstance(err, ValueError)
    assert str(err) == "('free form', 'more', 3, 'extra')"


def test_toml_decode_error_deprecated_with_defaults():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError()
    assert isinstance(err, ValueError)
    assert str(err) == "()"


def test_toml_decode_error_deprecated_msg_only():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("just message")
    assert str(err) == "('just message',)"


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------


def test_load_basic():
    fp = io.BytesIO(b'foo = "bar"\nnum = 42\n')
    assert load(fp) == {"foo": "bar", "num": 42}


def test_load_requires_binary():
    with pytest.raises(TypeError, match="binary mode"):
        load(io.StringIO("a = 1"))


def test_load_parse_float_kwarg():
    fp = io.BytesIO(b"x = 1.5")
    result = load(fp, parse_float=str)
    assert result == {"x": "1.5"}


def test_loads_empty():
    assert loads("") == {}


def test_loads_simple():
    assert loads("a = 1\nb = 2") == {"a": 1, "b": 2}


def test_loads_non_str_raises():
    with pytest.raises(TypeError, match="Expected str object"):
        loads(123)


def test_loads_bytes_raises():
    with pytest.raises(TypeError, match="Expected str object"):
        loads(b"a = 1")


def test_loads_crlf_normalization():
    assert loads("a = 1\r\nb = 2\r\n") == {"a": 1, "b": 2}


def test_loads_crlf_in_multiline_string():
    assert loads('a = """line1\r\nline2"""') == {"a": "line1\nline2"}


def test_loads_parse_float():
    assert loads("a = 1.25", parse_float=str) == {"a": "1.25"}


def test_loads_comments_and_blank_lines():
    assert loads("# comment\n\na = 1  # trailing\n") == {"a": 1}


def test_loads_invalid_statement():
    with pytest.raises(TOMLDecodeError, match="Invalid statement"):
        loads("!")


def test_loads_expected_newline():
    with pytest.raises(TOMLDecodeError, match="Expected newline"):
        loads("a = 1 b = 2")


def test_loads_tables_and_arrays():
    src = "[a.b]\nx = 1\n[[c]]\ny = 2\n[[c]]\ny = 3\n"
    assert loads(src) == {"a": {"b": {"x": 1}}, "c": [{"y": 2}, {"y": 3}]}


def test_loads_parse_float_illegal_type():
    with pytest.raises(ValueError, match="parse_float must not return dicts or lists"):
        loads("a = 1.0", parse_float=lambda s: {})


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_flags_empty_key():
    flags = Flags()
    assert flags.is_((), Flags.EXPLICIT_NEST) is False


def test_flags_set_and_is():
    flags = Flags()
    flags.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
    assert flags.is_(("a", "b"), Flags.EXPLICIT_NEST) is True
    assert flags.is_(("a",), Flags.EXPLICIT_NEST) is False
    assert flags.is_(("a", "b", "c"), Flags.EXPLICIT_NEST) is True


def test_flags_set_recursive():
    flags = Flags()
    flags.set(("a",), Flags.FROZEN, recursive=True)
    assert flags.is_(("a",), Flags.FROZEN) is True
    assert flags.is_(("a", "b"), Flags.FROZEN) is True
    assert flags.is_(("a", "b", "c"), Flags.FROZEN) is True


def test_flags_unset_all():
    flags = Flags()
    flags.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
    flags.unset_all(("a", "b"))
    assert flags.is_(("a", "b"), Flags.EXPLICIT_NEST) is False


def test_flags_unset_all_missing_prefix():
    flags = Flags()
    flags.unset_all(("nope", "x"))  # should not raise
    assert flags.is_(("nope",), Flags.EXPLICIT_NEST) is False


def test_flags_pending():
    flags = Flags()
    flags.add_pending(("x",), Flags.EXPLICIT_NEST)
    # Nothing applied until finalized.
    assert flags.is_(("x",), Flags.EXPLICIT_NEST) is False
    flags.finalize_pending()
    assert flags.is_(("x",), Flags.EXPLICIT_NEST) is True
    # Pending cleared.
    assert flags._pending_flags == set()


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------


def test_nested_dict_get_or_create_nest():
    nd = NestedDict()
    nest = nd.get_or_create_nest(("a", "b"))
    assert nest == {}
    nest["c"] = 1
    assert nd.dict == {"a": {"b": {"c": 1}}}


def test_nested_dict_get_or_create_nest_no_key():
    nd = NestedDict()
    assert nd.get_or_create_nest(()) is nd.dict


def test_nested_dict_access_lists():
    nd = NestedDict()
    nd.dict["a"] = [{"x": 1}]
    nest = nd.get_or_create_nest(("a", "y"))
    assert nest == {}
    assert nd.dict["a"] == [{"x": 1, "y": {}}]


def test_nested_dict_access_lists_false():
    nd = NestedDict()
    nd.dict["a"] = [{"x": 1}]
    with pytest.raises(KeyError):
        nd.get_or_create_nest(("a", "y"), access_lists=False)


def test_nested_dict_non_dict_middle():
    nd = NestedDict()
    nd.dict["a"] = 5
    with pytest.raises(KeyError):
        nd.get_or_create_nest(("a", "b"))


def test_nested_dict_append_nest_to_list_new():
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    assert nd.dict == {"a": [{}]}
    nd.append_nest_to_list(("a",))
    assert nd.dict == {"a": [{}, {}]}


def test_nested_dict_append_nest_to_list_non_list():
    nd = NestedDict()
    nd.dict["a"] = {}
    with pytest.raises(KeyError, match="other than list"):
        nd.append_nest_to_list(("a",))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_output_default_fields():
    out = Output()
    assert out.data.dict == {}
    assert isinstance(out.flags, Flags)


# ---------------------------------------------------------------------------
# skip_chars
# ---------------------------------------------------------------------------


def test_skip_chars_basic():
    src = "   abc"
    assert skip_chars(src, 0, TOML_WS) == 3
    assert skip_chars(src, 3, TOML_WS) == 3


def test_skip_chars_to_end():
    assert skip_chars("   ", 0, TOML_WS) == 3


# ---------------------------------------------------------------------------
# skip_until
# ---------------------------------------------------------------------------


def test_skip_until_found():
    assert skip_until(
        "abcdef", 0, "d", error_on=frozenset(), error_on_eof=False
    ) == 3


def test_skip_until_eof_no_error():
    assert skip_until(
        "abc", 0, "x", error_on=frozenset(), error_on_eof=False
    ) == 3


def test_skip_until_eof_error():
    with pytest.raises(TOMLDecodeError, match="Expected 'x'"):
        skip_until("abc", 0, "x", error_on=frozenset(), error_on_eof=True)


def test_skip_until_error_on_char():
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        skip_until(
            "ab!cd", 0, "d", error_on=frozenset("!"), error_on_eof=False
        )


# ---------------------------------------------------------------------------
# skip_comment / skip_comments_and_array_ws
# ---------------------------------------------------------------------------


def test_skip_comment_none():
    assert skip_comment("abc", 0) == 0


def test_skip_comment_to_newline():
    assert skip_comment("#hello\nabc", 0) == 6


def test_skip_comment_to_eof():
    assert skip_comment("#hello", 0) == 6


def test_skip_comment_eof_index():
    assert skip_comment("", 0) == 0


def test_skip_comment_illegal_char():
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        skip_comment("#hel\x01lo\n", 0)


def test_skip_comments_and_array_ws():
    src = "  \n\t# comment\n  x"
    assert skip_comments_and_array_ws(src, 0) == 13


def test_skip_comments_and_array_ws_no_move():
    assert skip_comments_and_array_ws("x", 0) == 0


# ---------------------------------------------------------------------------
# create_dict_rule / create_list_rule
# ---------------------------------------------------------------------------


def test_create_dict_rule_basic():
    out = Output()
    pos, key = create_dict_rule("[a.b]", 0, out)
    assert key == ("a", "b")
    assert pos == 5
    assert out.data.dict == {"a": {"b": {}}}
    assert out.flags.is_(("a", "b"), Flags.EXPLICIT_NEST) is True


def test_create_dict_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']'"):
        create_dict_rule("[a", 0, out)


def test_create_dict_rule_declared_twice():
    out = Output()
    create_dict_rule("[a]", 0, out)
    with pytest.raises(TOMLDecodeError, match="Cannot declare"):
        create_dict_rule("[a]", 0, out)


def test_create_dict_rule_frozen_namespace():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="Cannot declare"):
        create_dict_rule("[a]", 0, out)


def test_create_dict_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        create_dict_rule("[a]", 0, out)


def test_create_list_rule_basic():
    out = Output()
    pos, key = create_list_rule("[[a.b]]", 0, out)
    assert key == ("a", "b")
    assert pos == 7
    assert out.data.dict == {"a": {"b": [{}]}}


def test_create_list_rule_missing_brackets():
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']]'"):
        create_list_rule("[[a]", 0, out)


def test_create_list_rule_frozen():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable namespace"):
        create_list_rule("[[a]]", 0, out)


def test_create_list_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 5
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        create_list_rule("[[a]]", 0, out)


def test_create_list_rule_twice():
    out = Output()
    create_list_rule("[[a]]", 0, out)
    create_list_rule("[[a]]", 0, out)
    assert out.data.dict == {"a": [{}, {}]}


# ---------------------------------------------------------------------------
# key_value_rule
# ---------------------------------------------------------------------------


def test_key_value_rule_simple():
    out = Output()
    pos = key_value_rule("a = 1", 0, out, (), float)
    assert pos == 5
    assert out.data.dict == {"a": 1}


def test_key_value_rule_with_header():
    out = Output()
    key_value_rule("a = 1", 0, out, ("header",), float)
    assert out.data.dict == {"header": {"a": 1}}


def test_key_value_rule_dotted_key():
    out = Output()
    key_value_rule("a.b = 1", 0, out, (), float)
    assert out.data.dict == {"a": {"b": 1}}


def test_key_value_rule_freezes_dict_value():
    out = Output()
    key_value_rule("a = { b = 1 }", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True
    assert out.flags.is_(("a", "b"), Flags.FROZEN) is True


def test_key_value_rule_freezes_list_value():
    out = Output()
    key_value_rule("a = [1]", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True


def test_key_value_rule_redefine_namespace():
    out = Output()
    key_value_rule("a.b = 1", 0, out, (), float)
    # Pending finalized; declaring the same namespace now errors.
    out.flags.finalize_pending()
    with pytest.raises(TOMLDecodeError):
        key_value_rule("a.c = 2", 0, out, (), float)


def test_key_value_rule_mutate_frozen():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable namespace"):
        key_value_rule("a.b = 1", 0, out, (), float)


def test_key_value_rule_overwrite():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        key_value_rule("a = 2", 0, out, (), float)


def test_key_value_rule_overwrite_value_in_header():
    out = Output()
    out.data.dict["t"] = {"a": 1}
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        key_value_rule("a = 2", 0, out, ("t",), float)


# ---------------------------------------------------------------------------
# parse_key_value_pair
# ---------------------------------------------------------------------------


def test_parse_key_value_pair():
    pos, key, value = parse_key_value_pair("a = 1", 0, float, 0)
    assert pos == 5
    assert key == ("a",)
    assert value == 1


def test_parse_key_value_pair_ws():
    pos, key, value = parse_key_value_pair("a  =   1  ", 0, float, 0)
    assert pos == 9
    assert key == ("a",)
    assert value == 1


def test_parse_key_value_pair_missing_eq():
    with pytest.raises(TOMLDecodeError, match="Expected '='"):
        parse_key_value_pair("a 1", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_key / parse_key_part
# ---------------------------------------------------------------------------


def test_parse_key_bare():
    assert parse_key("abc", 0) == (3, ("abc",))


def test_parse_key_dotted():
    pos, key = parse_key("a.b.c", 0)
    assert pos == 5
    assert key == ("a", "b", "c")


def test_parse_key_quoted_parts():
    pos, key = parse_key('"a b"."c.d"', 0)
    assert pos == 9
    assert key == ("a b", "c.d")


def test_parse_key_too_many_parts():
    n = MAX_KEY_PARTS + 1
    src = ".".join(["a"] * n)
    with pytest.raises(RecursionError):
        parse_key(src, 0)


def test_parse_key_part_bare():
    pos, part = parse_key_part("abc-def_1 =", 0)
    assert part == "abc-def_1"
    assert pos == 9


def test_parse_key_part_literal():
    pos, part = parse_key_part("'hello'", 0)
    assert part == "hello"
    assert pos == 7


def test_parse_key_part_basic():
    pos, part = parse_key_part('"hello"', 0)
    assert part == "hello"
    assert pos == 7


def test_parse_key_part_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key_part("!", 0)


def test_parse_key_part_eof():
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key_part("", 0)


# ---------------------------------------------------------------------------
# parse_one_line_basic_str
# ---------------------------------------------------------------------------


def test_parse_one_line_basic_str():
    pos, s = parse_one_line_basic_str('"hello"', 0)
    assert s == "hello"
    assert pos == 7


def test_parse_one_line_basic_str_escape():
    pos, s = parse_one_line_basic_str('"a\\tb"', 0)
    assert s == "a\tb"
    assert pos == 6


# ---------------------------------------------------------------------------
# parse_array
# ---------------------------------------------------------------------------


def test_parse_array_empty():
    pos, arr = parse_array("[]", 0, float, 0)
    assert pos == 2
    assert arr == []


def test_parse_array_values():
    pos, arr = parse_array("[1, 2, 3]", 0, float, 0)
    assert pos == 9
    assert arr == [1, 2, 3]


def test_parse_array_trailing_comma():
    pos, arr = parse_array("[1, 2, ]", 0, float, 0)
    assert pos == 9
    assert arr == [1, 2]


def test_parse_array_newlines_and_comments():
    src = "[\n  1,  # one\n  2,\n]"
    pos, arr = parse_array(src, 0, float, 0)
    assert arr == [1, 2]
    assert pos == len(src)


def test_parse_array_nested():
    pos, arr = parse_array("[[1], [2]]", 0, float, 0)
    assert arr == [[1], [2]]


def test_parse_array_unclosed():
    with pytest.raises(TOMLDecodeError, match="Unclosed array"):
        parse_array("[1 2]", 0, float, 0)


def test_parse_array_unclosed_after_value():
    with pytest.raises(TOMLDecodeError, match="Unclosed array"):
        parse_array("[1", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_inline_table
# ---------------------------------------------------------------------------


def test_parse_inline_table_empty():
    pos, tbl = parse_inline_table("{}", 0, float, 0)
    assert pos == 2
    assert tbl == {}


def test_parse_inline_table_values():
    pos, tbl = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
    assert pos == 14
    assert tbl == {"a": 1, "b": 2}


def test_parse_inline_table_trailing_comma():
    pos, tbl = parse_inline_table("{a = 1, }", 0, float, 0)
    assert pos == 10
    assert tbl == {"a": 1}


def test_parse_inline_table_nested():
    pos, tbl = parse_inline_table("{a = {b = 1}}", 0, float, 0)
    assert tbl == {"a": {"b": 1}}


def test_parse_inline_table_dotted():
    pos, tbl = parse_inline_table("{a.b = 1}", 0, float, 0)
    assert tbl == {"a": {"b": 1}}


def test_parse_inline_table_duplicate_key():
    with pytest.raises(TOMLDecodeError, match="Duplicate inline table key"):
        parse_inline_table("{a = 1, a = 2}", 0, float, 0)


def test_parse_inline_table_unclosed():
    with pytest.raises(TOMLDecodeError, match="Unclosed inline table"):
        parse_inline_table("{a = 1", 0, float, 0)


def test_parse_inline_table_overwrite_value():
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        parse_inline_table("{a.b = 1, a = 2}", 0, float, 0)


def test_parse_inline_table_frozen_on_next_iteration():
    with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable namespace"):
        parse_inline_table("{a = {b = 1}, a.c = 2}", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_basic_str_escape
# ---------------------------------------------------------------------------


def test_parse_basic_str_escape_b():
    pos, s = parse_basic_str_escape("\\b", 0)
    assert s == "\b"
    assert pos == 2


def test_parse_basic_str_escape_hex():
    pos, s = parse_basic_str_escape("\\x41", 0)
    assert s == "A"
    assert pos == 4


def test_parse_basic_str_escape_u():
    pos, s = parse_basic_str_escape("\\u00e9", 0)
    assert s == "\u00e9"
    assert pos == 6


def test_parse_basic_str_escape_U():
    pos, s = parse_basic_str_escape("\\U0001F600", 0)
    assert s == "\U0001f600"
    assert pos == 10


def test_parse_basic_str_escape_invalid():
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape("\\q", 0)


def test_parse_basic_str_escape_multiline_backslash_newline():
    pos, s = parse_basic_str_escape_multiline("\\\n   xyz", 0)
    assert s == ""
    assert pos == 5


def test_parse_basic_str_escape_multiline_space_then_newline():
    pos, s = parse_basic_str_escape_multiline("\\   \n  rest", 0)
    assert s == ""
    assert pos == 5


def test_parse_basic_str_escape_multiline_space_then_char():
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape_multiline("\\  x", 0)


def test_parse_basic_str_escape_multiline_space_at_eof():
    pos, s = parse_basic_str_escape_multiline("\\  ", 0)
    assert s == ""
    assert pos == 3


# ---------------------------------------------------------------------------
# parse_hex_char
# ---------------------------------------------------------------------------


def test_parse_hex_char_2():
    pos, s = parse_hex_char("41abc", 0, 2)
    assert s == "A"
    assert pos == 2


def test_parse_hex_char_4():
    pos, s = parse_hex_char("00e9rest", 0, 4)
    assert s == "\u00e9"
    assert pos == 4


def test_parse_hex_char_invalid_digit():
    with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
        parse_hex_char("zz", 0, 2)


def test_parse_hex_char_short():
    with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
        parse_hex_char("4", 0, 2)


def test_parse_hex_char_surrogate():
    with pytest.raises(TOMLDecodeError, match="not a Unicode scalar value"):
        parse_hex_char("d800", 0, 4)


# ---------------------------------------------------------------------------
# parse_literal_str
# ---------------------------------------------------------------------------


def test_parse_literal_str():
    pos, s = parse_literal_str("'hello'", 0)
    assert s == "hello"
    assert pos == 7


def test_parse_literal_str_empty():
    pos, s = parse_literal_str("''", 0)
    assert s == ""
    assert pos == 2


def test_parse_literal_str_unterminated():
    with pytest.raises(TOMLDecodeError, match="Expected"):
        parse_literal_str("'hello", 0)


def test_parse_literal_str_illegal_char():
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        parse_literal_str("'he\x01llo'", 0)


# ---------------------------------------------------------------------------
# parse_multiline_str
# ---------------------------------------------------------------------------


def test_parse_multiline_literal():
    pos, s = parse_multiline_str("'''hello\nworld'''", 0, literal=True)
    assert s == "hello\nworld"
    assert pos == 18


def test_parse_multiline_literal_leading_newline_skipped():
    pos, s = parse_multiline_str("'''\nhello'''", 0, literal=True)
    assert s == "hello"


def test_parse_multiline_basic():
    pos, s = parse_multiline_str('"""hello\nworld"""', 0, literal=False)
    assert s == "hello\nworld"
    assert pos == 18


def test_parse_multiline_basic_escape():
    pos, s = parse_multiline_str('"""a\\tb"""', 0, literal=False)
    assert s == "a\tb"


def test_parse_multiline_extra_one_quote():
    pos, s = parse_multiline_str('"""hello""""', 0, literal=False)
    assert s == 'hello"'
    assert pos == 13


def test_parse_multiline_extra_two_quotes():
    pos, s = parse_multiline_str('"""hello"""""', 0, literal=False)
    assert s == 'hello""'
    assert pos == 14


def test_parse_multiline_literal_extra_quotes():
    pos, s = parse_multiline_str("'''hello'''''", 0, literal=True)
    assert s == 'hello""'
    assert pos == 14


def test_parse_multiline_unterminated():
    with pytest.raises(TOMLDecodeError):
        parse_multiline_str('"""hello', 0, literal=False)


# ---------------------------------------------------------------------------
# parse_basic_str
# ---------------------------------------------------------------------------


def test_parse_basic_str_oneline():
    pos, s = parse_basic_str("hello\"", 0, multiline=False)
    assert s == "hello"
    assert pos == 6


def test_parse_basic_str_oneline_escape():
    pos, s = parse_basic_str("a\\nb\"", 0, multiline=False)
    assert s == "a\nb"


def test_parse_basic_str_oneline_illegal():
    with pytest.raises(TOMLDecodeError, match="Illegal character"):
        parse_basic_str("a\x01\"", 0, multiline=False)


def test_parse_basic_str_oneline_unterminated():
    with pytest.raises(TOMLDecodeError, match="Unterminated string"):
        parse_basic_str("hello", 0, multiline=False)


def test_parse_basic_str_multiline():
    pos, s = parse_basic_str('hello\n"""', 0, multiline=True)
    assert s == "hello\n"
    assert pos == 9


def test_parse_basic_str_multiline_single_quote():
    pos, s = parse_basic_str('a"b"""', 0, multiline=True)
    assert s == 'a"b'
    assert pos == 7


def test_parse_basic_str_multiline_no_end_escape():
    # A lone backslash-like handling: actual escape resolves.
    pos, s = parse_basic_str('a\\tb"""', 0, multiline=True)
    assert s == "a\tb"


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------


def test_parse_value_bool_true():
    assert parse_value("true", 0, float, 0) == (4, True)


def test_parse_value_bool_false():
    assert parse_value("false", 0, float, 0) == (5, False)


def test_parse_value_basic_string():
    assert parse_value('"x"', 0, float, 0) == (3, "x")


def test_parse_value_multiline_basic_string():
    pos, val = parse_value('"""x"""', 0, float, 0)
    assert val == "x"
    assert pos == 7


def test_parse_value_literal_string():
    assert parse_value("'x'", 0, float, 0) == (3, "x")


def test_parse_value_multiline_literal():
    pos, val = parse_value("'''x'''", 0, float, 0)
    assert val == "x"
    assert pos == 7


def test_parse_value_array():
    assert parse_value("[1,2]", 0, float, 0) == (5, [1, 2])


def test_parse_value_inline_table():
    assert parse_value("{a=1}", 0, float, 0) == (5, {"a": 1})


def test_parse_value_integer():
    assert parse_value("42", 0, float, 0) == (2, 42)


def test_parse_value_hex_integer():
    assert parse_value("0xff", 0, float, 0) == (4, 255)


def test_parse_value_octal_integer():
    assert parse_value("0o17", 0, float, 0) == (4, 15)


def test_parse_value_binary_integer():
    assert parse_value("0b101", 0, float, 0) == (5, 5)


def test_parse_value_float():
    assert parse_value("3.14", 0, float, 0) == (4, 3.14)


def test_parse_value_float_underscore():
    assert parse_value("1_000.5", 0, float, 0)[1] == 1000.5


def test_parse_value_float_parse_float():
    assert parse_value("1.5", 0, str, 0) == (3, "1.5")


def test_parse_value_inf():
    assert parse_value("inf", 0, float, 0) == (3, float("inf"))


def test_parse_value_nan():
    pos, val = parse_value("nan", 0, float, 0)
    assert pos == 3
    assert val != val  # NaN


def test_parse_value_positive_inf():
    assert parse_value("+inf", 0, float, 0) == (4, float("inf"))


def test_parse_value_negative_inf():
    assert parse_value("-inf", 0, float, 0) == (4, float("-inf"))


def test_parse_value_positive_nan():
    pos, val = parse_value("+nan", 0, float, 0)
    assert pos == 4
    assert val != val


def test_parse_value_negative_nan():
    pos, val = parse_value("-nan", 0, float, 0)
    assert pos == 4
    assert val != val


def test_parse_value_date():
    assert parse_value("2021-01-01", 0, float, 0) == (10, date(2021, 1, 1))


def test_parse_value_datetime():
    assert parse_value(
        "2021-01-01T00:00:00", 0, float, 0
    ) == (19, datetime(2021, 1, 1, 0, 0, 0))


def test_parse_value_datetime_zulu():
    assert parse_value(
        "2021-01-01T00:00:00Z", 0, float, 0
    ) == (20, datetime(2021, 1, 1, tzinfo=timezone.utc))


def test_parse_value_datetime_offset():
    pos, val = parse_value("2021-01-01T00:00:00+01:00", 0, float, 0)
    assert val == datetime(
        2021, 1, 1, tzinfo=timezone(timedelta(hours=1))
    )


def test_parse_value_localtime():
    assert parse_value("12:30:45", 0, float, 0) == (8, time(12, 30, 45))


def test_parse_value_invalid_datetime():
    # Syntactically matches but semantically invalid date (Feb 30).
    with pytest.raises(TOMLDecodeError, match="Invalid date or datetime"):
        parse_value("2021-02-30", 0, float, 0)


def test_parse_value_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("!", 0, float, 0)


def test_parse_value_empty():
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("", 0, float, 0)


def test_parse_value_nesting_limit():
    src = "[" * (MAX_INLINE_NESTING + 2)
    with pytest.raises(RecursionError):
        parse_value(src, 0, float, MAX_INLINE_NESTING + 1)


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "codepoint,expected",
    [
        (0, True),
        (0x41, True),
        (0xD7FF, True),
        (0xD800, False),
        (0xDFFF, False),
        (0xE000, True),
        (0x10FFFF, True),
        (0x110000, False),
        (-1, False),
    ],
)
def test_is_unicode_scalar_value(codepoint, expected):
    assert is_unicode_scalar_value(codepoint) is expected


# ---------------------------------------------------------------------------
# make_safe_parse_float
# ---------------------------------------------------------------------------


def test_make_safe_parse_float_default_identity():
    assert make_safe_parse_float(float) is float


def test_make_safe_parse_float_ok():
    wrapped = make_safe_parse_float(str)
    assert wrapped("1.5") == "1.5"


def test_make_safe_parse_float_rejects_dict():
    wrapped = make_safe_parse_float(lambda s: {})
    with pytest.raises(ValueError, match="must not return dicts or lists"):
        wrapped("1.5")


def test_make_safe_parse_float_rejects_list():
    wrapped = make_safe_parse_float(lambda s: [])
    with pytest.raises(ValueError, match="must not return dicts or lists"):
        wrapped("1.5")


def test_deprecated_default_sentinel_class():
    assert isinstance(DEPRECATED_DEFAULT, type)
