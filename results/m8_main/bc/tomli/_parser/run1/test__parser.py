# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import math
import sys
from datetime import date, datetime, time, timedelta, timezone

import pytest

from tomli._parser import (
    DEPRECATED_DEFAULT,
    TOMLDecodeError,
    Flags,
    NestedDict,
    Output,
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
# DEPRECATED_DEFAULT and TOMLDecodeError
# ---------------------------------------------------------------------------


def test_deprecated_default_is_class_sentinel():
    assert isinstance(DEPRECATED_DEFAULT, type)


def test_toml_decode_error_attributes_first_line():
    doc = "abc def"
    err = TOMLDecodeError("Bad thing", doc, 4)
    assert isinstance(err, ValueError)
    assert err.msg == "Bad thing"
    assert err.doc == doc
    assert err.pos == 4
    assert err.lineno == 1
    assert err.colno == 5
    assert str(err) == "Bad thing (at line 1, column 5)"


def test_toml_decode_error_attributes_second_line():
    doc = "line1\nline2"
    err = TOMLDecodeError("oops", doc, 7)
    assert err.lineno == 2
    assert err.colno == 2
    assert str(err) == "oops (at line 2, column 2)"


def test_toml_decode_error_end_of_document():
    doc = "abc"
    err = TOMLDecodeError("done", doc, 3)
    assert err.pos == 3
    assert "end of document" in str(err)
    assert str(err) == "done (at end of document)"


def test_toml_decode_error_pos_beyond_len_is_end_of_document():
    doc = "abc"
    err = TOMLDecodeError("x", doc, 10)
    assert "end of document" in str(err)


def test_toml_decode_error_deprecated_free_form():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("a", "b", "c", "d")
    assert err.args == ("a", "b", "c", "d")


def test_toml_decode_error_deprecated_positional_legacy():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("custom message")
    assert err.args == ("custom message",)


def test_toml_decode_error_deprecated_multiple_freeform():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("a", "b", "c", "d")
    assert err.args == ("a", "b", "c", "d")


def test_toml_decode_error_deprecated_partial_args():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("msg", "doc")
    assert err.args == ("msg", "doc")


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------


def test_load_binary_file():
    fp = io.BytesIO(b'a = 1\nb = "x"\n')
    result = load(fp)
    assert result == {"a": 1, "b": "x"}


def test_load_rejects_text_file():
    fp = io.StringIO("a = 1\n")
    with pytest.raises(TypeError, match="File must be opened in binary mode"):
        load(fp)


def test_load_parse_float_hook():
    fp = io.BytesIO(b"a = 1.5\n")
    calls = []

    def parse_float(s):
        calls.append(s)
        return ("F", s)

    result = load(fp, parse_float=parse_float)
    assert result == {"a": ("F", "1.5")}
    assert calls == ["1.5"]


def test_loads_basic():
    assert loads('x = 1\ny = "two"\n') == {"x": 1, "y": "two"}


def test_loads_empty():
    assert loads("") == {}


def test_loads_crlf_normalization():
    result = loads('a = """x\r\ny"""\r\n')
    assert result == {"a": "x\ny"}


def test_loads_rejects_non_str():
    with pytest.raises(TypeError, match="Expected str object"):
        loads(b"a = 1")


def test_loads_none_rejected():
    with pytest.raises(TypeError, match="Expected str object"):
        loads(None)


def test_loads_rejects_non_int_not_error():
    with pytest.raises(TypeError, match="Expected str object"):
        loads(123)


def test_loads_invalid_statement():
    with pytest.raises(TOMLDecodeError, match="Invalid statement"):
        loads("%")


def test_loads_expected_newline_after_statement():
    with pytest.raises(
        TOMLDecodeError, match="Expected newline or end of document"
    ):
        loads("a = 1 b = 2")


def test_loads_comment_only():
    assert loads("# a comment\n") == {}


def test_loads_comment_eof_no_newline():
    assert loads("# comment") == {}


def test_loads_parse_float_hook():
    result = loads("a = 3.14", parse_float=lambda s: s)
    assert result == {"a": "3.14"}


def test_loads_trailing_whitespace_lines():
    assert loads("  \t\na = 1\n   ") == {"a": 1}


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_flags_is_empty_key_false():
    f = Flags()
    assert f.is_((), Flags.EXPLICIT_NEST) is False


def test_flags_set_and_is():
    f = Flags()
    f.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
    assert f.is_(("a",), Flags.FROZEN) is False
    assert f.is_(("b",), Flags.EXPLICIT_NEST) is False


def test_flags_nested_path_not_present():
    f = Flags()
    f.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is False


def test_flags_missing_intermediate():
    f = Flags()
    f.set(("a", "b"), Flags.FROZEN, recursive=False)
    assert f.is_(("a", "b"), Flags.FROZEN) is True
    assert f.is_(("a", "b", "c"), Flags.FROZEN) is False


def test_flags_missing_parent_returns_false():
    f = Flags()
    assert f.is_(("x", "y"), Flags.FROZEN) is False


def test_flags_recursive():
    f = Flags()
    f.set(("a",), Flags.FROZEN, recursive=True)
    assert f.is_(("a",), Flags.FROZEN) is True
    assert f.is_(("a", "b"), Flags.FROZEN) is True
    assert f.is_(("a", "b", "c"), Flags.FROZEN) is True


def test_flags_recursive_only_at_path():
    f = Flags()
    f.set(("a", "b"), Flags.FROZEN, recursive=True)
    assert f.is_(("a",), Flags.FROZEN) is False
    assert f.is_(("a", "b"), Flags.FROZEN) is True
    assert f.is_(("a", "b", "c"), Flags.FROZEN) is True


def test_flags_pending_finalize():
    f = Flags()
    f.add_pending(("a",), Flags.EXPLICIT_NEST)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
    f.finalize_pending()
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
    assert f._pending_flags == set()


def test_flags_unset_all():
    f = Flags()
    f.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
    f.unset_all(("a", "b"))
    assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is False


def test_flags_unset_all_missing_path():
    f = Flags()
    f.unset_all(("a", "b"))


def test_flags_unset_all_only_leaf():
    f = Flags()
    f.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    f.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
    f.unset_all(("a",))
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
    assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is False


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------


def test_nested_dict_get_or_create():
    nd = NestedDict()
    nest = nd.get_or_create_nest(("a", "b"))
    assert nest == {}
    assert nd.dict == {"a": {"b": {}}}
    assert nd.get_or_create_nest(("a", "b")) is nest


def test_nested_dict_get_or_create_empty():
    nd = NestedDict()
    assert nd.get_or_create_nest(()) is nd.dict


def test_nested_dict_access_list():
    nd = NestedDict()
    nd.dict["a"] = [{"b": 1}, {"c": 2}]
    nest = nd.get_or_create_nest(("a",))
    assert nest == {"c": 2}


def test_nested_dict_access_list_disabled():
    nd = NestedDict()
    nd.dict["a"] = [{"b": 1}]
    with pytest.raises(KeyError, match="no nest behind this key"):
        nd.get_or_create_nest(("a",), access_lists=False)


def test_nested_dict_no_nest_behind_key():
    nd = NestedDict()
    nd.dict["a"] = 5
    with pytest.raises(KeyError, match="no nest behind this key"):
        nd.get_or_create_nest(("a", "b"))


def test_nested_dict_append_nest_to_list_creates():
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    assert nd.dict == {"a": [{}]}


def test_nested_dict_append_nest_to_list_appends():
    nd = NestedDict()
    nd.dict["a"] = [{"x": 1}]
    nd.append_nest_to_list(("a",))
    assert nd.dict == {"a": [{"x": 1}, {}]}


def test_nested_dict_append_nest_to_list_not_list():
    nd = NestedDict()
    nd.dict["a"] = 5
    with pytest.raises(KeyError, match="other than list"):
        nd.append_nest_to_list(("a",))


def test_nested_dict_append_nest_to_list_nested_key():
    nd = NestedDict()
    nd.append_nest_to_list(("a", "b"))
    assert nd.dict == {"a": {"b": [{}]}}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_output_contains_data_and_flags():
    out = Output()
    assert isinstance(out.data, NestedDict)
    assert isinstance(out.flags, Flags)
    assert out.data.dict == {}


# ---------------------------------------------------------------------------
# skip_chars / skip_until / skip_comment / skip_comments_and_array_ws
# ---------------------------------------------------------------------------


def test_skip_chars_basic():
    assert skip_chars("aaab", 0, frozenset("a")) == 3


def test_skip_chars_none():
    assert skip_chars("bbbb", 0, frozenset("a")) == 0


def test_skip_chars_to_eof():
    assert skip_chars("aaa", 0, frozenset("a")) == 3


def test_skip_chars_start_pos():
    assert skip_chars("aaa", 2, frozenset("a")) == 3


def test_skip_until_found():
    assert (
        skip_until("abcXdef", 0, "X", error_on=frozenset(), error_on_eof=True)
        == 3
    )


def test_skip_until_not_found_ok():
    assert (
        skip_until("abcdef", 0, "X", error_on=frozenset(), error_on_eof=False)
        == 6
    )


def test_skip_until_not_found_error():
    with pytest.raises(TOMLDecodeError, match="Expected 'X'"):
        skip_until("abcdef", 0, "X", error_on=frozenset(), error_on_eof=True)


def test_skip_until_error_on_char():
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        skip_until("abZcdX", 0, "X", error_on=frozenset("Z"), error_on_eof=True)


def test_skip_comment_no_comment():
    assert skip_comment("abc", 0) == 0


def test_skip_comment_at_eof():
    assert skip_comment("abc", 3) == 3


def test_skip_comment_skips_to_newline():
    assert skip_comment("#hello\nworld", 0) == 6


def test_skip_comment_to_eof():
    assert skip_comment("#hello", 0) == 6


def test_skip_comment_illegal_char():
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        skip_comment("#he\x00llo\n", 0)


def test_skip_comments_and_array_ws_mixed():
    src = "  # c\n\t\n  x"
    pos = skip_comments_and_array_ws(src, 0)
    assert src[pos] == "x"


def test_skip_comments_and_array_ws_no_ws():
    assert skip_comments_and_array_ws("abc", 0) == 0


def test_skip_comments_and_array_ws_multiple_comments():
    src = " # a\n # b\nstuff"
    pos = skip_comments_and_array_ws(src, 0)
    assert src[pos:] == "stuff"


# ---------------------------------------------------------------------------
# create_dict_rule / create_list_rule / key_value_rule
# ---------------------------------------------------------------------------


def test_create_dict_rule_simple():
    out = Output()
    pos, key = create_dict_rule("[table]", 0, out)
    assert pos == 7
    assert key == ("table",)
    assert out.data.dict == {"table": {}}
    assert out.flags.is_(("table",), Flags.EXPLICIT_NEST) is True


def test_create_dict_rule_with_whitespace():
    out = Output()
    pos, key = create_dict_rule("[ a . b ]", 0, out)
    assert key == ("a", "b")
    assert pos == len("[ a . b ]")
    assert out.data.dict == {"a": {"b": {}}}


def test_create_dict_rule_twice_raises():
    out = Output()
    create_dict_rule("[t]", 0, out)
    with pytest.raises(TOMLDecodeError, match="Cannot declare"):
        create_dict_rule("[t]", 0, out)


def test_create_dict_rule_frozen_raises():
    out = Output()
    out.flags.set(("t",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="Cannot declare"):
        create_dict_rule("[t]", 0, out)


def test_create_dict_rule_overwrite_value():
    out = Output()
    out.data.dict["t"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        create_dict_rule("[t]", 0, out)


def test_create_dict_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']'"):
        create_dict_rule("[t", 0, out)


def test_create_list_rule_simple():
    out = Output()
    pos, key = create_list_rule("[[items]]", 0, out)
    assert pos == len("[[items]]")
    assert key == ("items",)
    assert out.data.dict == {"items": [{}]}


def test_create_list_rule_multiple():
    out = Output()
    create_list_rule("[[items]]", 0, out)
    create_list_rule("[[items]]", 0, out)
    assert out.data.dict == {"items": [{}, {}]}


def test_create_list_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']]'"):
        create_list_rule("[[x]", 0, out)


def test_create_list_rule_frozen():
    out = Output()
    out.flags.set(("x",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable"):
        create_list_rule("[[x]]", 0, out)


def test_create_list_rule_overwrite_scalar():
    out = Output()
    out.data.dict["x"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        create_list_rule("[[x]]", 0, out)


def test_key_value_rule_simple():
    out = Output()
    pos = key_value_rule("a = 1", 0, out, (), float)
    assert pos == len("a = 1")
    assert out.data.dict == {"a": 1}


def test_key_value_rule_dotted_key_pending_explicit_nest():
    out = Output()
    key_value_rule("a.b = 1", 0, out, (), float)
    assert out.data.dict == {"a": {"b": 1}}
    assert (("a",), Flags.EXPLICIT_NEST) in out.flags._pending_flags


def test_key_value_rule_redefine_namespace():
    out = Output()
    out.flags.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    with pytest.raises(TOMLDecodeError, match="Cannot redefine namespace"):
        key_value_rule("a.b = 1", 0, out, (), float)


def test_key_value_rule_frozen_parent():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable"):
        key_value_rule("a.b = 1", 0, out, (), float)


def test_key_value_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        key_value_rule("a.b = 1", 0, out, (), float)


def test_key_value_rule_duplicate_key():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        key_value_rule("a = 2", 0, out, (), float)


def test_key_value_rule_freezes_container():
    out = Output()
    key_value_rule("a = [1, 2]", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True
    assert out.flags.is_(("a", "anything"), Flags.FROZEN) is True


def test_key_value_rule_table_scalar_not_frozen():
    out = Output()
    key_value_rule("a = 1", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is False


def test_key_value_rule_parent_path_through_non_dict():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
        key_value_rule("a.b.c = 1", 0, out, (), float)


# ---------------------------------------------------------------------------
# parse_key_value_pair
# ---------------------------------------------------------------------------


def test_parse_key_value_pair():
    pos, key, value = parse_key_value_pair("abc = 42", 0, float, 0)
    assert pos == len("abc = 42")
    assert key == ("abc",)
    assert value == 42


def test_parse_key_value_pair_no_equals():
    with pytest.raises(TOMLDecodeError, match="Expected '=' after a key"):
        parse_key_value_pair("abc 42", 0, float, 0)


def test_parse_key_value_pair_eof_after_key():
    with pytest.raises(TOMLDecodeError, match="Expected '=' after a key"):
        parse_key_value_pair("abc", 0, float, 0)


def test_parse_key_value_pair_whitespace():
    pos, key, value = parse_key_value_pair('"a b" = "x"', 0, float, 0)
    assert key == ("a b",)
    assert value == "x"


# ---------------------------------------------------------------------------
# parse_key / parse_key_part
# ---------------------------------------------------------------------------


def test_parse_key_bare():
    assert parse_key("abc", 0) == (3, ("abc",))


def test_parse_key_dotted():
    pos, key = parse_key("a.b.c", 0)
    assert pos == 5
    assert key == ("a", "b", "c")


def test_parse_key_dotted_whitespace():
    pos, key = parse_key("a . b . c", 0)
    assert key == ("a", "b", "c")
    assert pos == len("a . b . c")


def test_parse_key_stops_at_non_dot():
    pos, key = parse_key("a=1", 0)
    assert pos == 1
    assert key == ("a",)


def test_parse_key_empty_fails():
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key("= 1", 0)


def test_parse_key_pointed_trailing():
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key("a.=", 0)


def test_parse_key_too_many_parts():
    n = sys.getrecursionlimit() + 5
    src = ".".join(["a"] * n)
    with pytest.raises(RecursionError):
        parse_key(src, 0)


def test_parse_key_part_bare():
    assert parse_key_part("abc", 0) == (3, "abc")


def test_parse_key_part_literal():
    pos, part = parse_key_part("'lit'", 0)
    assert pos == 5
    assert part == "lit"


def test_parse_key_part_basic():
    pos, part = parse_key_part('"basic"', 0)
    assert pos == 7
    assert part == "basic"


def test_parse_key_part_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key_part("@", 0)


def test_parse_key_part_eof():
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key_part("", 0)


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


def test_parse_one_line_basic_str():
    assert parse_one_line_basic_str('"hello"', 0) == (7, "hello")


def test_parse_one_line_basic_str_unterminated():
    with pytest.raises(TOMLDecodeError, match="Unterminated string"):
        parse_one_line_basic_str('"hello', 0)


def test_parse_literal_str():
    assert parse_literal_str("'hello'", 0) == (7, "hello")


def test_parse_literal_str_unterminated():
    with pytest.raises(TOMLDecodeError, match="Expected \"'\""):
        parse_literal_str("'hello", 0)


def test_parse_literal_str_illegal_char():
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        parse_literal_str("'he\x01llo'", 0)


def test_parse_multiline_str_literal():
    pos, s = parse_multiline_str("'''hello'''", 0, literal=True)
    assert pos == len("'''hello'''")
    assert s == "hello"


def test_parse_multiline_str_literal_leading_newline_removed():
    pos, s = parse_multiline_str("'''\nhello'''", 0, literal=True)
    assert s == "hello"


def test_parse_multiline_str_basic():
    pos, s = parse_multiline_str('"""hi"""', 0, literal=False)
    assert pos == len('"""hi"""')
    assert s == "hi"


def test_parse_multiline_str_extra_quotes_one():
    pos, s = parse_multiline_str('"""hi""""', 0, literal=False)
    assert pos == len('"""hi""""')
    assert s == 'hi"'


def test_parse_multiline_str_extra_quotes_two():
    pos, s = parse_multiline_str('"""hi"""""', 0, literal=False)
    assert pos == len('"""hi"""""')
    assert s == 'hi""'


def test_parse_multiline_str_literal_extra_quotes():
    pos, s = parse_multiline_str("'''hi''''", 0, literal=True)
    assert s == "hi'"


def test_parse_multiline_str_literal_unterminated():
    with pytest.raises(TOMLDecodeError, match="Expected \"'''\""):
        parse_multiline_str("'''hi", 0, literal=True)


def test_parse_basic_str():
    assert parse_basic_str('abc"', 0, multiline=False) == (4, "abc")


def test_parse_basic_str_escapes():
    pos, s = parse_basic_str('a\\nb"', 0, multiline=False)
    assert s == "a\nb"
    assert pos == len('a\\nb"')


def test_parse_basic_str_unterminated():
    with pytest.raises(TOMLDecodeError, match="Unterminated string"):
        parse_basic_str("abc", 0, multiline=False)


def test_parse_basic_str_illegal_char():
    with pytest.raises(TOMLDecodeError, match="Illegal character"):
        parse_basic_str("a\x01b", 0, multiline=False)


def test_parse_basic_str_multiline_quotes_single():
    pos, s = parse_basic_str('a"b"""', 0, multiline=True)
    assert s == 'a"b'


def test_parse_basic_str_non_multiline_stops_at_quote():
    pos, s = parse_basic_str('abc"def', 0, multiline=False)
    assert pos == 4
    assert s == "abc"


def test_parse_basic_str_escape_basic():
    pos, s = parse_basic_str_escape(r"\n", 0)
    assert pos == 2
    assert s == "\n"


def test_parse_basic_str_escape_all_known():
    repl = {
        "\\b": "\u0008",
        "\\t": "\u0009",
        "\\n": "\u000a",
        "\\f": "\u000c",
        "\\r": "\u000d",
        "\\e": "\u001b",
        '\\"': "\u0022",
        "\\\\": "\u005c",
    }
    for esc, expected in repl.items():
        pos, out = parse_basic_str_escape(esc, 0)
        assert pos == 2
        assert out == expected


def test_parse_basic_str_escape_invalid():
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape(r"\q", 0)


def test_parse_basic_str_escape_x():
    pos, s = parse_basic_str_escape(r"\x41", 0)
    assert pos == 4
    assert s == "A"


def test_parse_basic_str_escape_u():
    pos, s = parse_basic_str_escape(r"\u00e9", 0)
    assert pos == 6
    assert s == "\u00e9"


def test_parse_basic_str_escape_U():
    pos, s = parse_basic_str_escape(r"\U0001F600", 0)
    assert pos == 10
    assert s == "\U0001F600"


def test_parse_basic_str_escape_multiline_ws_newline():
    src = "\\\n   next"
    pos, s = parse_basic_str_escape_multiline(src, 0)
    assert s == ""
    assert src[pos:] == "next"


def test_parse_basic_str_escape_multiline_space():
    src = "\\   \nnext"
    pos, s = parse_basic_str_escape_multiline(src, 0)
    assert s == ""
    assert src[pos:] == "next"


def test_parse_basic_str_escape_multiline_tab():
    src = "\\\t\nnext"
    pos, s = parse_basic_str_escape_multiline(src, 0)
    assert s == ""
    assert src[pos:] == "next"


def test_parse_basic_str_escape_multiline_trailing_ws_eof():
    src = "\\   "
    pos, s = parse_basic_str_escape_multiline(src, 0)
    assert s == ""
    assert pos == len(src)


def test_parse_basic_str_escape_multiline_no_newline_raises():
    src = "\\  x"
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape_multiline(src, 0)


def test_parse_basic_str_escape_multiline_falls_back_to_non_ws():
    pos, s = parse_basic_str_escape_multiline(r"\n", 0)
    assert s == "\n"


# ---------------------------------------------------------------------------
# parse_hex_char
# ---------------------------------------------------------------------------


def test_parse_hex_char_2():
    assert parse_hex_char("41", 0, 2) == (2, "A")


def test_parse_hex_char_4():
    assert parse_hex_char("00e9", 0, 4) == (4, "\u00e9")


def test_parse_hex_char_8():
    assert parse_hex_char("0001F600", 0, 8) == (8, "\U0001F600")


def test_parse_hex_char_too_short():
    with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
        parse_hex_char("4", 0, 2)


def test_parse_hex_char_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
        parse_hex_char("ZZ", 0, 2)


def test_parse_hex_char_surrogate_not_scalar():
    with pytest.raises(TOMLDecodeError, match="not a Unicode scalar value"):
        parse_hex_char("D800", 0, 4)


def test_parse_hex_char_too_large():
    with pytest.raises(TOMLDecodeError, match="not a Unicode scalar value"):
        parse_hex_char("110000", 0, 6)


# ---------------------------------------------------------------------------
# parse_array / parse_inline_table
# ---------------------------------------------------------------------------


def test_parse_array_empty():
    pos, arr = parse_array("[]", 0, float, 0)
    assert pos == 2
    assert arr == []


def test_parse_array_simple():
    pos, arr = parse_array("[1, 2, 3]", 0, float, 0)
    assert pos == len("[1, 2, 3]")
    assert arr == [1, 2, 3]


def test_parse_array_trailing_comma():
    pos, arr = parse_array("[1, 2,]", 0, float, 0)
    assert arr == [1, 2]


def test_parse_array_with_whitespace_and_comments():
    src = "[ 1 , # c\n 2 ]"
    pos, arr = parse_array(src, 0, float, 0)
    assert arr == [1, 2]


def test_parse_array_nested():
    pos, arr = parse_array("[[1], [2]]", 0, float, 0)
    assert arr == [[1], [2]]


def test_parse_array_unclosed():
    with pytest.raises(TOMLDecodeError, match="Unclosed array"):
        parse_array("[1 2]", 0, float, 0)


def test_parse_array_unclosed_eof():
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_array("[1,", 0, float, 0)


def test_parse_inline_table_empty():
    pos, table = parse_inline_table("{}", 0, float, 0)
    assert pos == 2
    assert table == {}


def test_parse_inline_table_simple():
    pos, table = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
    assert pos == len("{a = 1, b = 2}")
    assert table == {"a": 1, "b": 2}


def test_parse_inline_table_trailing_comma():
    pos, table = parse_inline_table("{a = 1,}", 0, float, 0)
    assert table == {"a": 1}


def test_parse_inline_table_dotted_key():
    pos, table = parse_inline_table("{a.b = 1}", 0, float, 0)
    assert table == {"a": {"b": 1}}


def test_parse_inline_table_duplicate_key():
    with pytest.raises(TOMLDecodeError, match="Duplicate inline table key"):
        parse_inline_table("{a = 1, a = 2}", 0, float, 0)


def test_parse_inline_table_unclosed():
    with pytest.raises(TOMLDecodeError, match="Unclosed inline table"):
        parse_inline_table("{a = 1", 0, float, 0)


def test_parse_inline_table_frozen_nested():
    with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable"):
        parse_inline_table("{a = {x = 1}, a.y = 2}", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------


def test_parse_value_true():
    assert parse_value("true", 0, float, 0) == (4, True)


def test_parse_value_false():
    assert parse_value("false", 0, float, 0) == (5, False)


def test_parse_value_int():
    assert parse_value("42", 0, float, 0) == (2, 42)


def test_parse_value_negative_int():
    assert parse_value("-42", 0, float, 0) == (3, -42)


def test_parse_value_hex_int():
    assert parse_value("0x10", 0, float, 0) == (4, 16)


def test_parse_value_octal_int():
    assert parse_value("0o10", 0, float, 0) == (4, 8)


def test_parse_value_binary_int():
    assert parse_value("0b1010", 0, float, 0) == (6, 10)


def test_parse_value_float():
    pos, v = parse_value("1.5", 0, float, 0)
    assert v == 1.5
    assert pos == 3


def test_parse_value_float_custom_parse():
    pos, v = parse_value("1.5", 0, lambda s: ("F", s), 0)
    assert v == ("F", "1.5")


def test_parse_value_inf():
    assert parse_value("inf", 0, float, 0) == (3, float("inf"))


def test_parse_value_neg_inf():
    assert parse_value("-inf", 0, float, 0) == (4, float("-inf"))


def test_parse_value_plus_inf():
    assert parse_value("+inf", 0, float, 0) == (4, float("inf"))


def test_parse_value_nan():
    pos, v = parse_value("nan", 0, float, 0)
    assert pos == 3
    assert math.isnan(v)


def test_parse_value_neg_nan():
    pos, v = parse_value("-nan", 0, float, 0)
    assert pos == 4
    assert math.isnan(v)


def test_parse_value_basic_string():
    assert parse_value('"hello"', 0, float, 0) == (7, "hello")


def test_parse_value_multiline_basic_string():
    pos, v = parse_value('"""hi"""', 0, float, 0)
    assert v == "hi"


def test_parse_value_literal_string():
    assert parse_value("'hello'", 0, float, 0) == (7, "hello")


def test_parse_value_multiline_literal_string():
    pos, v = parse_value("'''hi'''", 0, float, 0)
    assert v == "hi"


def test_parse_value_array():
    pos, v = parse_value("[1, 2]", 0, float, 0)
    assert v == [1, 2]


def test_parse_value_inline_table():
    pos, v = parse_value("{a = 1}", 0, float, 0)
    assert v == {"a": 1}


def test_parse_value_date():
    pos, v = parse_value("2020-01-01", 0, float, 0)
    assert v == date(2020, 1, 1)
    assert pos == 10


def test_parse_value_datetime():
    pos, v = parse_value("2020-01-01T00:00:00", 0, float, 0)
    assert v == datetime(2020, 1, 1, 0, 0, 0)
    assert pos == len("2020-01-01T00:00:00")


def test_parse_value_datetime_offset():
    pos, v = parse_value("2020-01-01T00:00:00+01:00", 0, float, 0)
    expected = datetime(
        2020, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=1))
    )
    assert v == expected


def test_parse_value_localtime():
    pos, v = parse_value("12:00:00", 0, float, 0)
    assert v == time(12, 0, 0)
    assert pos == 8


def test_parse_value_invalid_date():
    # "2020-13-01" is not a valid date. The current implementation parses it
    # via RE_NUMBER as the integer 2020 (with a trailing remainder), so it
    # succeeds with the value 2020 and a position just past "2020".
    pos, v = parse_value("2020-13-01", 0, float, 0)
    assert v == 2020
    assert pos == 4


def test_parse_value_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("@", 0, float, 0)


def test_parse_value_eof():
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("", 0, float, 0)


def test_parse_value_nesting_limit():
    from tomli._parser import MAX_INLINE_NESTING

    with pytest.raises(RecursionError):
        parse_value("[", 0, float, MAX_INLINE_NESTING + 1)


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------


def test_is_unicode_scalar_value():
    assert is_unicode_scalar_value(0) is True
    assert is_unicode_scalar_value(0x41) is True
    assert is_unicode_scalar_value(0xD7FF) is True
    assert is_unicode_scalar_value(0xD800) is False
    assert is_unicode_scalar_value(0xDFFF) is False
    assert is_unicode_scalar_value(0xE000) is True
    assert is_unicode_scalar_value(0x10FFFF) is True
    assert is_unicode_scalar_value(0x110000) is False
    assert is_unicode_scalar_value(-1) is False


# ---------------------------------------------------------------------------
# make_safe_parse_float
# ---------------------------------------------------------------------------


def test_make_safe_parse_float_default_returns_float():
    assert make_safe_parse_float(float) is float


def test_make_safe_parse_float_wraps():
    calls = []

    def custom(s):
        calls.append(s)
        return 42

    wrapped = make_safe_parse_float(custom)
    assert wrapped is not custom
    assert wrapped("1.5") == 42
    assert calls == ["1.5"]


def test_make_safe_parse_float_rejects_list():
    wrapped = make_safe_parse_float(lambda s: [])
    with pytest.raises(ValueError, match="must not return dicts or lists"):
        wrapped("1")


def test_make_safe_parse_float_rejects_dict():
    wrapped = make_safe_parse_float(lambda s: {})
    with pytest.raises(ValueError, match="must not return dicts or lists"):
        wrapped("1")


def test_make_safe_parse_float_used_in_loads():
    with pytest.raises(ValueError, match="must not return dicts or lists"):
        loads("a = 1.5", parse_float=lambda s: [])
