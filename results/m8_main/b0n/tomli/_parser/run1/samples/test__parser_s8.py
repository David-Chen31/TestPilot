# SPDX-License-Identifier: MIT
"""Tests for tomli._parser."""

from __future__ import annotations

import datetime
import io
import sys

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


def test_constants_basic() -> None:
    assert ASCII_CTRL == frozenset(chr(i) for i in range(32)) | frozenset(chr(127))
    assert "\t" not in ILLEGAL_BASIC_STR_CHARS
    assert "\n" in ILLEGAL_BASIC_STR_CHARS
    assert "\t" not in ILLEGAL_MULTILINE_BASIC_STR_CHARS
    assert "\n" not in ILLEGAL_MULTILINE_BASIC_STR_CHARS
    assert ILLEGAL_LITERAL_STR_CHARS is ILLEGAL_BASIC_STR_CHARS
    assert ILLEGAL_MULTILINE_LITERAL_STR_CHARS is ILLEGAL_MULTILINE_BASIC_STR_CHARS
    assert ILLEGAL_COMMENT_CHARS is ILLEGAL_BASIC_STR_CHARS
    assert TOML_WS == frozenset(" \t")
    assert TOML_WS_AND_NEWLINE == frozenset(" \t\n")
    assert BARE_KEY_CHARS == frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    )
    assert KEY_INITIAL_CHARS == BARE_KEY_CHARS | frozenset("\"'")
    assert HEXDIGIT_CHARS == frozenset("abcdefABCDEF0123456789")
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\n"] == "\u000a"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\\\"] == "\\"
    assert MAX_INLINE_NESTING == 400
    assert MAX_KEY_PARTS == sys.getrecursionlimit()


# ---------------------------------------------------------------------------
# DEPRECATED_DEFAULT & TOMLDecodeError
# ---------------------------------------------------------------------------


def test_deprecated_default_sentinel() -> None:
    assert isinstance(DEPRECATED_DEFAULT, type)
    assert DEPRECATED_DEFAULT is not str


def test_toml_decode_error_attributes() -> None:
    doc = "a = 1\nb = 2"
    err = TOMLDecodeError("bad thing", doc, 4)
    assert isinstance(err, ValueError)
    assert err.msg == "bad thing"
    assert err.doc == doc
    assert err.pos == 4
    assert err.lineno == 1
    assert err.colno == 5
    assert str(err) == "bad thing (at line 1, column 5)"


def test_toml_decode_error_multiline_coords() -> None:
    doc = "abc\ndef"
    err = TOMLDecodeError("boom", doc, 4)
    assert err.lineno == 2
    assert err.colno == 1


def test_toml_decode_error_end_of_document() -> None:
    doc = "abc"
    err = TOMLDecodeError("eof", doc, len(doc))
    assert err.lineno == 1
    assert err.colno == 4
    assert "(at end of document)" in str(err)
    assert "line 1" not in str(err)


def test_toml_decode_error_end_of_document_later_line() -> None:
    doc = "a\nb"
    err = TOMLDecodeError("eof", doc, 3)
    assert "end of document" in str(err)
    assert err.lineno == 2


def test_toml_decode_error_free_form_deprecated() -> None:
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("a", "b", "c", "extra")
    assert isinstance(err, ValueError)


def test_toml_decode_error_free_form_no_args() -> None:
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError()
    assert str(err) == ""


def test_toml_decode_error_free_form_with_positional() -> None:
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("m", "d", 3)
    assert str(err) == "m d 3"


def test_toml_decode_error_non_str_msg() -> None:
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError(5, "doc", 0)  # type: ignore[arg-type]
    assert "5" in str(err)


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------


def test_load_basic() -> None:
    fp = io.BytesIO(b"x = 1\ny = 'foo'")
    assert load(fp) == {"x": 1, "y": "foo"}


def test_load_type_error_for_text_file() -> None:
    fp = io.StringIO("x = 1")
    with pytest.raises(TypeError, match="binary mode"):
        load(fp)  # type: ignore[arg-type]


def test_load_parse_float() -> None:
    from decimal import Decimal

    fp = io.BytesIO(b"x = 1.5")
    assert load(fp, parse_float=Decimal) == {"x": Decimal("1.5")}


def test_loads_basic() -> None:
    assert loads("a = true\nb = [1, 2, 3]") == {"a": True, "b": [1, 2, 3]}


def test_loads_empty() -> None:
    assert loads("") == {}


def test_loads_crlf_normalization() -> None:
    assert loads("a = 1\r\nb = 2\r\n") == {"a": 1, "b": 2}


def test_loads_crlf_in_string() -> None:
    assert loads('a = "x\r\ny"') == {"a": "x\ny"}


def test_loads_type_error_bytes() -> None:
    with pytest.raises(TypeError, match="Expected str object"):
        loads(b"x = 1")  # type: ignore[arg-type]


def test_loads_type_error_none() -> None:
    with pytest.raises(TypeError, match="Expected str object"):
        loads(None)  # type: ignore[arg-type]


def test_loads_tables_and_arrays() -> None:
    src = (
        "[table]\n"
        "key = 'value'\n"
        "[[arr]]\n"
        "x = 1\n"
        "[[arr]]\n"
        "x = 2\n"
    )
    assert loads(src) == {
        "table": {"key": "value"},
        "arr": [{"x": 1}, {"x": 2}],
    }


def test_loads_comment_only() -> None:
    assert loads("# just a comment") == {}


def test_loads_invalid_statement() -> None:
    with pytest.raises(TOMLDecodeError, match="Invalid statement"):
        loads("!!! = 1")


def test_loads_expected_newline_after_statement() -> None:
    with pytest.raises(TOMLDecodeError, match="Expected newline"):
        loads("a = 1 b = 2")


def test_loads_parse_float_decimal() -> None:
    from decimal import Decimal

    assert loads("x = 1.5", parse_float=Decimal) == {"x": Decimal("1.5")}


def test_loads_parse_float_illegal_return() -> None:
    def bad(_s: str) -> dict:  # type: ignore[return-value]
        return {}

    with pytest.raises(ValueError, match="parse_float must not return"):
        loads("x = 1.5", parse_float=bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_flags_basic_set_and_is() -> None:
    flags = Flags()
    assert flags.is_(("a",), Flags.FROZEN) is False
    flags.set(("a",), Flags.FROZEN, recursive=False)
    assert flags.is_(("a",), Flags.FROZEN) is True
    assert flags.is_(("a",), Flags.EXPLICIT_NEST) is False


def test_flags_empty_key_returns_false() -> None:
    flags = Flags()
    assert flags.is_((), Flags.FROZEN) is False


def test_flags_missing_key_returns_false() -> None:
    flags = Flags()
    flags.set(("a", "b"), Flags.FROZEN, recursive=False)
    assert flags.is_(("x",), Flags.FROZEN) is False
    assert flags.is_(("a", "z"), Flags.FROZEN) is False


def test_flags_recursive() -> None:
    flags = Flags()
    flags.set(("a",), Flags.FROZEN, recursive=True)
    assert flags.is_(("a",), Flags.FROZEN) is True
    assert flags.is_(("a", "b"), Flags.FROZEN) is True
    assert flags.is_(("a", "b", "c"), Flags.FROZEN) is True


def test_flags_non_recursive_does_not_propagate() -> None:
    flags = Flags()
    flags.set(("a",), Flags.FROZEN, recursive=False)
    assert flags.is_(("a", "b"), Flags.FROZEN) is False


def test_flags_pending() -> None:
    flags = Flags()
    flags.add_pending(("a",), Flags.EXPLICIT_NEST)
    assert flags.is_(("a",), Flags.EXPLICIT_NEST) is False
    flags.finalize_pending()
    assert flags.is_(("a",), Flags.EXPLICIT_NEST) is True
    # Pending cleared after finalize
    flags.finalize_pending()
    assert flags.is_(("a",), Flags.EXPLICIT_NEST) is True


def test_flags_unset_all() -> None:
    flags = Flags()
    flags.set(("a", "b"), Flags.FROZEN, recursive=False)
    flags.unset_all(("a", "b"))
    assert flags.is_(("a", "b"), Flags.FROZEN) is False


def test_flags_unset_all_missing_parent() -> None:
    flags = Flags()
    # Should not raise
    flags.unset_all(("x", "y", "z"))


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------


def test_nested_dict_get_or_create() -> None:
    nd = NestedDict()
    nest = nd.get_or_create_nest(("a", "b"))
    assert nest == {}
    nest["c"] = 1
    assert nd.dict == {"a": {"b": {"c": 1}}}


def test_nested_dict_access_lists() -> None:
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    inner = nd.get_or_create_nest(("a",))
    inner["x"] = 1
    assert nd.dict == {"a": [{"x": 1}]}


def test_nested_dict_access_lists_false() -> None:
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    with pytest.raises(KeyError):
        nd.get_or_create_nest(("a",), access_lists=False)


def test_nested_dict_no_nest_behind_key() -> None:
    nd = NestedDict()
    nd.dict["a"] = 1
    with pytest.raises(KeyError):
        nd.get_or_create_nest(("a", "b"))


def test_nested_dict_append_nest_to_list_new() -> None:
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    assert nd.dict == {"a": [{}]}


def test_nested_dict_append_nest_to_list_existing() -> None:
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    nd.append_nest_to_list(("a",))
    assert nd.dict == {"a": [{}, {}]}


def test_nested_dict_append_nest_to_list_not_list() -> None:
    nd = NestedDict()
    nd.dict["a"] = 1
    with pytest.raises(KeyError):
        nd.append_nest_to_list(("a",))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_output() -> None:
    out = Output()
    assert isinstance(out.data, NestedDict)
    assert isinstance(out.flags, Flags)
    assert out.data.dict == {}


# ---------------------------------------------------------------------------
# skip_chars / skip_until / skip_comment / skip_comments_and_array_ws
# ---------------------------------------------------------------------------


def test_skip_chars() -> None:
    assert skip_chars("   abc", 0, TOML_WS) == 3
    assert skip_chars("\t \t", 0, TOML_WS) == 3
    assert skip_chars("abc", 1, TOML_WS) == 1


def test_skip_chars_eof() -> None:
    assert skip_chars("   ", 0, TOML_WS) == 3


def test_skip_until_basic() -> None:
    assert skip_until("abc'def", 0, "'", error_on=frozenset(), error_on_eof=True) == 3


def test_skip_until_error_on() -> None:
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        skip_until("ab\x00c'", 0, "'", error_on=frozenset("\x00"), error_on_eof=True)


def test_skip_until_eof_error() -> None:
    with pytest.raises(TOMLDecodeError, match="Expected"):
        skip_until("abc", 0, "'", error_on=frozenset(), error_on_eof=True)


def test_skip_until_eof_no_error() -> None:
    assert skip_until("abc", 0, "'", error_on=frozenset(), error_on_eof=False) == 3


def test_skip_comment_none() -> None:
    assert skip_comment("abc", 0) == 0


def test_skip_comment_at_eof() -> None:
    assert skip_comment("", 0) == 0


def test_skip_comment_present() -> None:
    assert skip_comment("# hello\nx", 0) == 7


def test_skip_comment_eof() -> None:
    assert skip_comment("# hello", 0) == 7


def test_skip_comment_illegal_char() -> None:
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        skip_comment("# a\x00b\n", 0)


def test_skip_comments_and_array_ws() -> None:
    src = "  # comment\n  \n  x"
    assert skip_comments_and_array_ws(src, 0) == 15


def test_skip_comments_and_array_ws_nothing() -> None:
    assert skip_comments_and_array_ws("x", 0) == 0


# ---------------------------------------------------------------------------
# create_dict_rule / create_list_rule
# ---------------------------------------------------------------------------


def test_create_dict_rule() -> None:
    out = Output()
    pos, key = create_dict_rule("[a.b]", 0, out)
    assert key == ("a", "b")
    assert pos == 5
    assert out.data.dict == {"a": {"b": {}}}
    assert out.flags.is_(("a", "b"), Flags.EXPLICIT_NEST)


def test_create_dict_rule_whitespace() -> None:
    out = Output()
    pos, key = create_dict_rule("[ a . b ]", 0, out)
    assert key == ("a", "b")
    assert pos == 9


def test_create_dict_rule_twice() -> None:
    out = Output()
    create_dict_rule("[a]", 0, out)
    with pytest.raises(TOMLDecodeError, match="Cannot declare"):
        create_dict_rule("[a]", 0, out)


def test_create_dict_rule_frozen() -> None:
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="Cannot declare"):
        create_dict_rule("[a]", 0, out)


def test_create_dict_rule_overwrite_value() -> None:
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        create_dict_rule("[a.b]", 0, out)


def test_create_dict_rule_missing_bracket() -> None:
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']'"):
        create_dict_rule("[a", 0, out)


def test_create_list_rule() -> None:
    out = Output()
    pos, key = create_list_rule("[[a.b]]", 0, out)
    assert key == ("a", "b")
    assert pos == 7
    assert out.data.dict == {"a": {"b": [{}]}}


def test_create_list_rule_twice() -> None:
    out = Output()
    create_list_rule("[[a]]", 0, out)
    create_list_rule("[[a]]", 0, out)
    assert out.data.dict == {"a": [{}, {}]}


def test_create_list_rule_frozen() -> None:
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable namespace"):
        create_list_rule("[[a]]", 0, out)


def test_create_list_rule_overwrite() -> None:
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        create_list_rule("[[a]]", 0, out)


def test_create_list_rule_missing_bracket() -> None:
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']]'"):
        create_list_rule("[[a]", 0, out)


# ---------------------------------------------------------------------------
# key_value_rule / parse_key_value_pair
# ---------------------------------------------------------------------------


def test_key_value_rule_simple() -> None:
    out = Output()
    pos = key_value_rule("a = 1", 0, out, (), float)
    assert pos == 5
    assert out.data.dict == {"a": 1}


def test_key_value_rule_header() -> None:
    out = Output()
    out.data.get_or_create_nest(("t",))
    pos = key_value_rule("a = 1", 0, out, ("t",), float)
    assert out.data.dict == {"t": {"a": 1}}
    assert pos == 5


def test_key_value_rule_dotted_key() -> None:
    out = Output()
    pos = key_value_rule("a.b.c = 1", 0, out, (), float)
    assert out.data.dict == {"a": {"b": {"c": 1}}}
    assert out.flags.is_(("a",), Flags.EXPLICIT_NEST)


def test_key_value_rule_dotted_redefine() -> None:
    out = Output()
    out.flags.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    with pytest.raises(TOMLDecodeError, match="Cannot redefine namespace"):
        key_value_rule("a.b = 1", 0, out, (), float)


def test_key_value_rule_frozen_parent() -> None:
    out = Output()
    out.flags.set(("t",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable namespace"):
        key_value_rule("a = 1", 0, out, ("t",), float)


def test_key_value_rule_overwrite() -> None:
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        key_value_rule("a = 2", 0, out, (), float)


def test_key_value_rule_no_nest_behind_key() -> None:
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        key_value_rule("a.b = 1", 0, out, (), float)


def test_key_value_rule_freezes_inline_table() -> None:
    out = Output()
    key_value_rule("a = {x = 1}", 0, out, (), float)
    assert out.flags.is_(("a", "x"), Flags.FROZEN)


def test_parse_key_value_pair() -> None:
    pos, key, value = parse_key_value_pair("a = 1", 0, float, 0)
    assert key == ("a",)
    assert value == 1
    assert pos == 5


def test_parse_key_value_pair_whitespace() -> None:
    pos, key, value = parse_key_value_pair("a   =   1", 0, float, 0)
    assert key == ("a",)
    assert value == 1
    assert pos == 9


def test_parse_key_value_pair_missing_eq() -> None:
    with pytest.raises(TOMLDecodeError, match="Expected '='"):
        parse_key_value_pair("a 1", 0, float, 0)


def test_parse_key_value_pair_missing_value() -> None:
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_key_value_pair("a =", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_key / parse_key_part
# ---------------------------------------------------------------------------


def test_parse_key_bare() -> None:
    pos, key = parse_key("abc", 0)
    assert key == ("abc",)
    assert pos == 3


def test_parse_key_dotted() -> None:
    pos, key = parse_key("a.b.c", 0)
    assert key == ("a", "b", "c")
    assert pos == 5


def test_parse_key_quoted_parts() -> None:
    pos, key = parse_key('"a".\'b\'', 0)
    assert key == ("a", "b")


def test_parse_key_too_many_parts() -> None:
    src = ".".join(["a"] * (MAX_KEY_PARTS + 1))
    with pytest.raises(RecursionError):
        parse_key(src, 0)


def test_parse_key_part_bare() -> None:
    pos, part = parse_key_part("abc-def", 0)
    assert part == "abc-def"
    assert pos == 7


def test_parse_key_part_literal() -> None:
    pos, part = parse_key_part("'abc'", 0)
    assert part == "abc"
    assert pos == 5


def test_parse_key_part_basic() -> None:
    pos, part = parse_key_part('"abc"', 0)
    assert part == "abc"
    assert pos == 5


def test_parse_key_part_invalid() -> None:
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key_part("!abc", 0)


# ---------------------------------------------------------------------------
# parse_one_line_basic_str
# ---------------------------------------------------------------------------


def test_parse_one_line_basic_str() -> None:
    pos, s = parse_one_line_basic_str('"hello"', 0)
    assert s == "hello"
    assert pos == 7


def test_parse_one_line_basic_str_escape() -> None:
    pos, s = parse_one_line_basic_str(r'"a\tb"', 0)
    assert s == "a\tb"
    assert pos == 6


def test_parse_one_line_basic_str_unterminated() -> None:
    with pytest.raises(TOMLDecodeError, match="Unterminated string"):
        parse_one_line_basic_str('"abc', 0)


# ---------------------------------------------------------------------------
# parse_array
# ---------------------------------------------------------------------------


def test_parse_array_empty() -> None:
    pos, arr = parse_array("[]", 0, float, 0)
    assert arr == []
    assert pos == 2


def test_parse_array_basic() -> None:
    pos, arr = parse_array("[1, 2, 3]", 0, float, 0)
    assert arr == [1, 2, 3]
    assert pos == 9


def test_parse_array_trailing_comma() -> None:
    pos, arr = parse_array("[1, 2,]", 0, float, 0)
    assert arr == [1, 2]
    assert pos == 7


def test_parse_array_with_whitespace_comments() -> None:
    src = "[ 1, # c\n 2 ]"
    pos, arr = parse_array(src, 0, float, 0)
    assert arr == [1, 2]
    assert pos == len(src)


def test_parse_array_unclosed() -> None:
    with pytest.raises(TOMLDecodeError, match="Unclosed array"):
        parse_array("[1 2]", 0, float, 0)


def test_parse_array_nested() -> None:
    pos, arr = parse_array("[[1], [2]]", 0, float, 0)
    assert arr == [[1], [2]]
    assert pos == 10


# ---------------------------------------------------------------------------
# parse_inline_table
# ---------------------------------------------------------------------------


def test_parse_inline_table_empty() -> None:
    pos, d = parse_inline_table("{}", 0, float, 0)
    assert d == {}
    assert pos == 2


def test_parse_inline_table_basic() -> None:
    pos, d = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
    assert d == {"a": 1, "b": 2}
    assert pos == 14


def test_parse_inline_table_trailing_comma() -> None:
    pos, d = parse_inline_table("{a = 1,}", 0, float, 0)
    assert d == {"a": 1}
    assert pos == 9


def test_parse_inline_table_duplicate_key() -> None:
    with pytest.raises(TOMLDecodeError, match="Duplicate inline table key"):
        parse_inline_table("{a = 1, a = 2}", 0, float, 0)


def test_parse_inline_table_unclosed() -> None:
    with pytest.raises(TOMLDecodeError, match="Unclosed inline table"):
        parse_inline_table("{a = 1 b = 2}", 0, float, 0)


def test_parse_inline_table_frozen() -> None:
    with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable namespace"):
        parse_inline_table("{a = {x = 1}, a.y = 2}", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_basic_str_escape
# ---------------------------------------------------------------------------


def test_parse_basic_str_escape_simple() -> None:
    pos, s = parse_basic_str_escape(r"\n", 0)
    assert s == "\n"
    assert pos == 2


def test_parse_basic_str_escape_backslash() -> None:
    pos, s = parse_basic_str_escape("\\\\", 0)
    assert s == "\\"
    assert pos == 2


def test_parse_basic_str_escape_hex() -> None:
    pos, s = parse_basic_str_escape(r"\x41", 0)
    assert s == "A"
    assert pos == 4


def test_parse_basic_str_escape_unicode4() -> None:
    pos, s = parse_basic_str_escape(r"\u00e9", 0)
    assert s == "é"
    assert pos == 6


def test_parse_basic_str_escape_unicode8() -> None:
    pos, s = parse_basic_str_escape(r"\U0001F600", 0)
    assert s == "\U0001F600"
    assert pos == 10


def test_parse_basic_str_escape_invalid() -> None:
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape(r"\q", 0)


def test_parse_basic_str_escape_multiline_line_ending_backslash() -> None:
    src = "\\\n   abc"
    pos, s = parse_basic_str_escape_multiline(src, 0)
    assert s == ""
    assert pos == 5  # skips whitespace and newline


def test_parse_basic_str_escape_multiline_space_newline() -> None:
    src = "\\  \n  x"
    pos, s = parse_basic_str_escape_multiline(src, 0)
    assert s == ""
    assert pos == 5


def test_parse_basic_str_escape_multiline_eof_after_ws() -> None:
    src = "\\  "
    pos, s = parse_basic_str_escape_multiline(src, 0)
    assert s == ""
    assert pos == 3


def test_parse_basic_str_escape_multiline_unescaped() -> None:
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape_multiline("\\ x", 0)


# ---------------------------------------------------------------------------
# parse_hex_char
# ---------------------------------------------------------------------------


def test_parse_hex_char() -> None:
    pos, c = parse_hex_char("41", 0, 2)
    assert c == "A"
    assert pos == 2


def test_parse_hex_char_too_short() -> None:
    with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
        parse_hex_char("4", 0, 2)


def test_parse_hex_char_non_hex() -> None:
    with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
        parse_hex_char("zz", 0, 2)


def test_parse_hex_char_surrogate() -> None:
    with pytest.raises(TOMLDecodeError, match="not a Unicode scalar value"):
        parse_hex_char("D800", 0, 4)


# ---------------------------------------------------------------------------
# parse_literal_str
# ---------------------------------------------------------------------------


def test_parse_literal_str() -> None:
    pos, s = parse_literal_str("'hello'", 0)
    assert s == "hello"
    assert pos == 7


def test_parse_literal_str_eof() -> None:
    with pytest.raises(TOMLDecodeError, match="Expected"):
        parse_literal_str("'hello", 0)


def test_parse_literal_str_illegal_char() -> None:
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        parse_literal_str("'a\x00b'", 0)


# ---------------------------------------------------------------------------
# parse_multiline_str
# ---------------------------------------------------------------------------


def test_parse_multiline_str_literal() -> None:
    pos, s = parse_multiline_str("'''abc'''", 0, literal=True)
    assert s == "abc"
    assert pos == 9


def test_parse_multiline_str_literal_leading_newline() -> None:
    pos, s = parse_multiline_str("'''\nabc'''", 0, literal=True)
    assert s == "abc"
    assert pos == 10


def test_parse_multiline_str_literal_extra_quote() -> None:
    pos, s = parse_multiline_str("''''abc'''", 0, literal=True)
    # starts with 4 quotes
    assert s == "'abc"


def test_parse_multiline_str_literal_extra_two_quotes() -> None:
    pos, s = parse_multiline_str("'''''abc'''", 0, literal=True)
    assert s == "''abc"


def test_parse_multiline_str_basic() -> None:
    pos, s = parse_multiline_str('"""abc"""', 0, literal=False)
    assert s == "abc"
    assert pos == 9


def test_parse_multiline_str_basic_leading_newline() -> None:
    pos, s = parse_multiline_str('"""\nabc"""', 0, literal=False)
    assert s == "abc"


def test_parse_multiline_str_basic_escape() -> None:
    pos, s = parse_multiline_str('"""a\\tb"""', 0, literal=False)
    assert s == "a\tb"


def test_parse_multiline_str_basic_extra_quote() -> None:
    pos, s = parse_multiline_str('""""abc"""', 0, literal=False)
    assert s == '"abc'


def test_parse_multiline_str_basic_extra_two_quotes() -> None:
    pos, s = parse_multiline_str('"""""abc"""', 0, literal=False)
    assert s == '""abc'


def test_parse_multiline_str_literal_eof() -> None:
    with pytest.raises(TOMLDecodeError, match="Expected"):
        parse_multiline_str("'''abc", 0, literal=True)


# ---------------------------------------------------------------------------
# parse_basic_str
# ---------------------------------------------------------------------------


def test_parse_basic_str_one_line() -> None:
    pos, s = parse_basic_str('abc"', 0, multiline=False)
    assert s == "abc"
    assert pos == 4


def test_parse_basic_str_one_line_with_escapes() -> None:
    pos, s = parse_basic_str(r'a\tb"', 0, multiline=False)
    assert s == "a\tb"
    assert pos == 6


def test_parse_basic_str_multiline() -> None:
    pos, s = parse_basic_str('abc"""', 0, multiline=True)
    assert s == "abc"
    assert pos == 6


def test_parse_basic_str_multiline_single_quote() -> None:
    pos, s = parse_basic_str('a"b"""', 0, multiline=True)
    assert s == 'a"b'
    assert pos == 7


def test_parse_basic_str_multiline_two_quotes() -> None:
    pos, s = parse_basic_str('a""b"""', 0, multiline=True)
    assert s == 'a""b'


def test_parse_basic_str_unterminated() -> None:
    with pytest.raises(TOMLDecodeError, match="Unterminated string"):
        parse_basic_str("abc", 0, multiline=False)


def test_parse_basic_str_illegal_char() -> None:
    with pytest.raises(TOMLDecodeError, match="Illegal character"):
        parse_basic_str("a\x00b\"", 0, multiline=False)


def test_parse_basic_str_multiline_allows_newline() -> None:
    pos, s = parse_basic_str('a\nb"""', 0, multiline=True)
    assert s == "a\nb"


def test_parse_basic_str_multiline_line_ending_backslash() -> None:
    pos, s = parse_basic_str('a\\\n  b"""', 0, multiline=True)
    assert s == "ab"


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------


def test_parse_value_bool_true() -> None:
    pos, val = parse_value("true x", 0, float, 0)
    assert val is True
    assert pos == 4


def test_parse_value_bool_false() -> None:
    pos, val = parse_value("false x", 0, float, 0)
    assert val is False
    assert pos == 5


def test_parse_value_string() -> None:
    pos, val = parse_value('"hi"', 0, float, 0)
    assert val == "hi"


def test_parse_value_literal_string() -> None:
    pos, val = parse_value("'hi'", 0, float, 0)
    assert val == "hi"


def test_parse_value_multiline_string() -> None:
    pos, val = parse_value('"""hi"""', 0, float, 0)
    assert val == "hi"


def test_parse_value_multiline_literal() -> None:
    pos, val = parse_value("'''hi'''", 0, float, 0)
    assert val == "hi"


def test_parse_value_integer() -> None:
    pos, val = parse_value("42", 0, float, 0)
    assert val == 42
    assert pos == 2


def test_parse_value_float() -> None:
    pos, val = parse_value("1.5", 0, float, 0)
    assert val == 1.5
    assert isinstance(val, float)


def test_parse_value_inf() -> None:
    pos, val = parse_value("inf", 0, float, 0)
    assert val == float("inf")
    assert pos == 3


def test_parse_value_neg_inf() -> None:
    pos, val = parse_value("-inf", 0, float, 0)
    assert val == float("-inf")
    assert pos == 4


def test_parse_value_nan() -> None:
    import math

    pos, val = parse_value("nan", 0, float, 0)
    assert math.isnan(val)
    assert pos == 3


def test_parse_value_plus_nan() -> None:
    import math

    pos, val = parse_value("+nan", 0, float, 0)
    assert math.isnan(val)
    assert pos == 4


def test_parse_value_date() -> None:
    pos, val = parse_value("2021-01-01", 0, float, 0)
    assert val == datetime.date(2021, 1, 1)


def test_parse_value_datetime_zulu() -> None:
    pos, val = parse_value("2021-01-01T00:00:00Z", 0, float, 0)
    assert isinstance(val, datetime.datetime)
    assert val.tzinfo == datetime.timezone.utc


def test_parse_value_localtime() -> None:
    pos, val = parse_value("12:34:56", 0, float, 0)
    assert val == datetime.time(12, 34, 56)


def test_parse_value_array() -> None:
    pos, val = parse_value("[1, 2]", 0, float, 0)
    assert val == [1, 2]


def test_parse_value_inline_table() -> None:
    pos, val = parse_value("{a = 1}", 0, float, 0)
    assert val == {"a": 1}


def test_parse_value_invalid() -> None:
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("?", 0, float, 0)


def test_parse_value_invalid_date() -> None:
    with pytest.raises(TOMLDecodeError, match="Invalid date"):
        parse_value("2021-13-01", 0, float, 0)


def test_parse_value_eof() -> None:
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("", 0, float, 0)


def test_parse_value_recursion_limit() -> None:
    with pytest.raises(RecursionError):
        parse_value("[" * (MAX_INLINE_NESTING + 2), 0, float, 0)


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "codepoint, expected",
    [
        (0, True),
        (65, True),
        (55295, True),
        (55296, False),
        (57343, False),
        (57344, True),
        (1114111, True),
        (1114112, False),
        (-1, False),
    ],
)
def test_is_unicode_scalar_value(codepoint: int, expected: bool) -> None:
    assert is_unicode_scalar_value(codepoint) is expected


# ---------------------------------------------------------------------------
# make_safe_parse_float
# ---------------------------------------------------------------------------


def test_make_safe_parse_float_default() -> None:
    assert make_safe_parse_float(float) is float


def test_make_safe_parse_float_wraps() -> None:
    from decimal import Decimal

    wrapped = make_safe_parse_float(Decimal)
    assert wrapped("1.5") == Decimal("1.5")


def test_make_safe_parse_float_rejects_dict() -> None:
    wrapped = make_safe_parse_float(lambda s: {})  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError, match="must not return dicts or lists"):
        wrapped("1.5")


def test_make_safe_parse_float_rejects_list() -> None:
    wrapped = make_safe_parse_float(lambda s: [])  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError, match="must not return dicts or lists"):
        wrapped("1.5")


# ---------------------------------------------------------------------------
# Integration-ish extras
# ---------------------------------------------------------------------------


def test_loads_dotted_key_and_nested_table_conflict() -> None:
    with pytest.raises(TOMLDecodeError):
        loads("a.b = 1\na.b.c = 2")


def test_loads_inline_table_frozen() -> None:
    with pytest.raises(TOMLDecodeError):
        loads("a = {x = 1}\na.y = 2")


def test_loads_array_of_tables_with_nested() -> None:
    src = "[[a]]\n[[a.b]]\nx = 1\n"
    assert loads(src) == {"a": [{"b": [{"x": 1}]}]}


def test_loads_deep_intermediate_key_after_array() -> None:
    # Ensure list-access path in get_or_create_nest is exercised
    src = "[[a]]\nb = 1\n[a.c]\nd = 2\n"
    assert loads(src) == {"a": [{"b": 1, "c": {"d": 2}}]}
