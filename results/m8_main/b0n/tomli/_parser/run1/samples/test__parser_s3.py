# SPDX-License-Identifier: MIT
# Tests for tomli._parser

from __future__ import annotations

import datetime
import io
import sys
import warnings

import pytest

from tomli._parser import (
    BASIC_STR_ESCAPE_REPLACEMENTS,
    DEPRECATED_DEFAULT,
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


def test_constants():
    assert TOML_WS == frozenset(" \t")
    assert TOML_WS_AND_NEWLINE == frozenset(" \t\n")
    assert "\t" not in ILLEGAL_BASIC_STR_CHARS
    assert "\n" not in ILLEGAL_MULTILINE_BASIC_STR_CHARS
    assert "\t" not in ILLEGAL_MULTILINE_BASIC_STR_CHARS
    assert ILLEGAL_LITERAL_STR_CHARS == ILLEGAL_BASIC_STR_CHARS
    assert ILLEGAL_MULTILINE_LITERAL_STR_CHARS == ILLEGAL_MULTILINE_BASIC_STR_CHARS
    assert ILLEGAL_COMMENT_CHARS == ILLEGAL_BASIC_STR_CHARS
    assert "a" in KEY_INITIAL_CHARS and "'" in KEY_INITIAL_CHARS and '"' in KEY_INITIAL_CHARS
    assert MAX_INLINE_NESTING == 400
    assert MAX_KEY_PARTS == sys.getrecursionlimit()


def test_basic_str_escape_replacements():
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\b"] == "\u0008"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\t"] == "\u0009"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\n"] == "\u000a"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\f"] == "\u000c"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\r"] == "\u000d"
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\e"] == "\u001b"
    assert BASIC_STR_ESCAPE_REPLACEMENTS['\\"'] == '"'
    assert BASIC_STR_ESCAPE_REPLACEMENTS["\\\\"] == "\\"


# ---------------------------------------------------------------------------
# DEPRECATED_DEFAULT / TOMLDecodeError
# ---------------------------------------------------------------------------


def test_deprecated_default_sentinel():
    assert isinstance(DEPRECATED_DEFAULT, type)


def test_toml_decode_error_normal():
    doc = "abc\ndef"
    err = TOMLDecodeError("oops", doc, 5)
    assert err.msg == "oops"
    assert err.doc == doc
    assert err.pos == 5
    assert err.lineno == 2
    assert err.colno == 2
    assert str(err) == "oops (at line 2, column 2)"
    assert isinstance(err, ValueError)


def test_toml_decode_error_first_line():
    err = TOMLDecodeError("bad", "hello", 2)
    assert err.lineno == 1
    assert err.colno == 3
    assert "line 1, column 3" in str(err)


def test_toml_decode_error_end_of_document():
    doc = "abc"
    err = TOMLDecodeError("bad", doc, 3)
    assert err.pos == 3
    assert str(err) == "bad (at end of document)"


def test_toml_decode_error_pos_beyond_end():
    err = TOMLDecodeError("bad", "abc", 10)
    assert "end of document" in str(err)
    assert err.lineno == 1


def test_toml_decode_error_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        err = TOMLDecodeError("msg", "doc", 1)
    assert err.msg == "msg"
    assert err.doc == "doc"
    assert err.pos == 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        err2 = TOMLDecodeError()
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert isinstance(err2, ValueError)
    assert not hasattr(err2, "msg") or err2.msg == DEPRECATED_DEFAULT

    # Free-form args
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        err3 = TOMLDecodeError("just", "free", "form", "args")
    assert len(w) == 1
    assert str(err3) == "just"


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------


def test_load_basic():
    fp = io.BytesIO(b"a = 1\nb = 'x'\n")
    assert load(fp) == {"a": 1, "b": "x"}


def test_load_parse_float():
    fp = io.BytesIO(b"x = 1.5\n")
    assert load(fp, parse_float=lambda s: "F:" + s) == {"x": "F:1.5"}


def test_load_text_mode_raises():
    fp = io.StringIO("a = 1")
    with pytest.raises(TypeError, match="binary mode"):
        load(fp)


def test_load_crlf(tmp_path):
    p = tmp_path / "f.toml"
    p.write_bytes(b"a = 1\r\nb = 2\r\n")
    with open(p, "rb") as f:
        assert load(f) == {"a": 1, "b": 2}


def test_loads_basic():
    assert loads("a = 1") == {"a": 1}
    assert loads("") == {}
    assert loads("   \n\t\n") == {}


def test_loads_crlf_normalization():
    assert loads("a = '''x\r\ny'''") == {"a": "x\ny"}


def test_loads_wrong_type():
    with pytest.raises(TypeError, match="Expected str object"):
        loads(b"a = 1")
    with pytest.raises(TypeError):
        loads(None)


def test_loads_parse_float():
    assert loads("x = 1.0", parse_float=lambda s: "NUM") == {"x": "NUM"}


def test_loads_parse_float_illegal():
    def pf(s):
        return [1, 2]

    with pytest.raises(ValueError, match="parse_float"):
        loads("x = 1.0", parse_float=pf)


def test_loads_comment_and_blank_lines():
    src = "# comment\n\na = 1 # trailing\n# tail\n"
    assert loads(src) == {"a": 1}


def test_loads_invalid_statement():
    with pytest.raises(TOMLDecodeError, match="Invalid statement"):
        loads("= 1")


def test_loads_expected_newline():
    with pytest.raises(TOMLDecodeError, match="Expected newline"):
        loads("a = 1 b = 2")


def test_loads_tables_and_arrays():
    src = """
[table]
x = 1
[[arr]]
y = 2
[[arr]]
y = 3
"""
    assert loads(src) == {"table": {"x": 1}, "arr": [{"y": 2}, {"y": 3}]}


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_flags_is_empty_key():
    f = Flags()
    assert f.is_(("a",), Flags.FROZEN) is False
    # empty key
    assert f.is_((), Flags.FROZEN) is False


def test_flags_set_and_is():
    f = Flags()
    f.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
    assert f.is_(("a",), Flags.FROZEN) is False


def test_flags_recursive():
    f = Flags()
    f.set(("a", "b"), Flags.FROZEN, recursive=True)
    assert f.is_(("a",), Flags.FROZEN) is True
    assert f.is_(("a", "b"), Flags.FROZEN) is True
    assert f.is_(("a", "b", "c"), Flags.FROZEN) is False


def test_flags_non_recursive_not_in_parent():
    f = Flags()
    f.set(("a", "b"), Flags.FROZEN, recursive=False)
    assert f.is_(("a",), Flags.FROZEN) is False
    assert f.is_(("a", "b"), Flags.FROZEN) is True
    assert f.is_(("a", "b", "c"), Flags.FROZEN) is False


def test_flags_missing_key():
    f = Flags()
    assert f.is_(("x", "y"), Flags.FROZEN) is False


def test_flags_pending():
    f = Flags()
    f.add_pending(("a",), Flags.EXPLICIT_NEST)
    f.add_pending(("b", "c"), Flags.FROZEN)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
    f.finalize_pending()
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
    assert f.is_(("b", "c"), Flags.FROZEN) is True
    # pending cleared
    assert f._pending_flags == set()


def test_flags_unset_all():
    f = Flags()
    f.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
    f.set(("a", "b", "c"), Flags.EXPLICIT_NEST, recursive=False)
    f.unset_all(("a", "b"))
    assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is False
    assert f.is_(("a", "b", "c"), Flags.EXPLICIT_NEST) is False


def test_flags_unset_all_missing_parent():
    f = Flags()
    # no-op, should not raise
    f.unset_all(("nope", "x"))


def test_flags_unset_all_missing_stem():
    f = Flags()
    f.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    f.unset_all(("a", "missing"))


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------


def test_nested_dict_get_or_create():
    nd = NestedDict()
    nest = nd.get_or_create_nest(("a", "b"))
    assert nest == {}
    assert nd.dict == {"a": {"b": {}}}
    nest["x"] = 1
    assert nd.dict == {"a": {"b": {"x": 1}}}


def test_nested_dict_get_or_create_empty_key():
    nd = NestedDict()
    assert nd.get_or_create_nest(()) is nd.dict


def test_nested_dict_get_or_create_over_non_dict():
    nd = NestedDict()
    nd.dict["a"] = 1
    with pytest.raises(KeyError):
        nd.get_or_create_nest(("a", "b"))


def test_nested_dict_access_lists():
    nd = NestedDict()
    nd.dict["a"] = [{"x": 1}]
    nest = nd.get_or_create_nest(("a", "b"))
    assert nest is nd.dict["a"][-1]["b"]


def test_nested_dict_no_access_lists():
    nd = NestedDict()
    nd.dict["a"] = [{"x": 1}]
    with pytest.raises(KeyError):
        nd.get_or_create_nest(("a", "b"), access_lists=False)


def test_nested_dict_append_nest_to_list():
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


def test_nested_dict_append_nest_to_list_non_list_parent():
    nd = NestedDict()
    nd.dict["a"] = 1
    with pytest.raises(KeyError):
        nd.append_nest_to_list(("a", "b"))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_output():
    out = Output()
    assert isinstance(out.data, NestedDict)
    assert isinstance(out.flags, Flags)
    assert out.data.dict == {}


# ---------------------------------------------------------------------------
# skip_chars
# ---------------------------------------------------------------------------


def test_skip_chars():
    assert skip_chars(" \t\nabc", 0, TOML_WS) == 2
    assert skip_chars("abc", 0, TOML_WS) == 0
    assert skip_chars("   ", 0, TOML_WS) == 3


def test_skip_chars_past_end():
    assert skip_chars("", 0, TOML_WS) == 0


# ---------------------------------------------------------------------------
# skip_until
# ---------------------------------------------------------------------------


def test_skip_until_found():
    assert skip_until(
        "abc'def", 0, "'", error_on=frozenset(), error_on_eof=True
    ) == 3


def test_skip_until_eof_ok():
    assert skip_until(
        "abc", 0, "'", error_on=frozenset(), error_on_eof=False
    ) == 3


def test_skip_until_eof_error():
    with pytest.raises(TOMLDecodeError, match="Expected"):
        skip_until("abc", 0, "'", error_on=frozenset(), error_on_eof=True)


def test_skip_until_error_on_char():
    with pytest.raises(TOMLDecodeError, match="invalid character"):
        skip_until("ab\x01c'", 0, "'", error_on=frozenset("\x01"), error_on_eof=True)


# ---------------------------------------------------------------------------
# skip_comment
# ---------------------------------------------------------------------------


def test_skip_comment_none():
    assert skip_comment("abc", 0) == 0


def test_skip_comment_at_eof():
    assert skip_comment("", 0) == 0


def test_skip_comment_with_comment():
    assert skip_comment("# hello\nnext", 0) == 7


def test_skip_comment_no_newline():
    assert skip_comment("# hello", 0) == 7


def test_skip_comment_illegal_char():
    with pytest.raises(TOMLDecodeError, match="invalid character"):
        skip_comment("# \x01\n", 0)


# ---------------------------------------------------------------------------
# skip_comments_and_array_ws
# ---------------------------------------------------------------------------


def test_skip_comments_and_array_ws():
    src = "  \n\t # comment\n  x"
    assert skip_comments_and_array_ws(src, 0) == 17
    assert src[17] == "x"


def test_skip_comments_and_array_ws_nothing():
    assert skip_comments_and_array_ws("x", 0) == 0


def test_skip_comments_and_array_ws_multiple_comments():
    src = "# a\n# b\n  x"
    assert skip_comments_and_array_ws(src, 0) == 9


# ---------------------------------------------------------------------------
# create_dict_rule / create_list_rule
# ---------------------------------------------------------------------------


def test_create_dict_rule():
    out = Output()
    pos, key = create_dict_rule("[a.b]", 0, out)
    assert key == ("a", "b")
    assert pos == 5
    assert out.data.dict == {"a": {"b": {}}}
    assert out.flags.is_(("a", "b"), Flags.EXPLICIT_NEST) is True


def test_create_dict_rule_duplicate():
    out = Output()
    create_dict_rule("[a]", 0, out)
    with pytest.raises(TOMLDecodeError, match="twice"):
        create_dict_rule("[a]", 0, out)


def test_create_dict_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']'"):
        create_dict_rule("[a", 0, out)


def test_create_dict_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="overwrite"):
        create_dict_rule("[a.b]", 0, out)


def test_create_dict_rule_frozen():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="twice"):
        create_dict_rule("[a]", 0, out)


def test_create_list_rule():
    out = Output()
    pos, key = create_list_rule("[[a.b]]", 0, out)
    assert key == ("a", "b")
    assert pos == 7
    assert out.data.dict == {"a": {"b": [{}]}}


def test_create_list_rule_twice():
    out = Output()
    create_list_rule("[[a]]", 0, out)
    create_list_rule("[[a]]", 0, out)
    assert out.data.dict == {"a": [{}, {}]}


def test_create_list_rule_frozen():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="immutable"):
        create_list_rule("[[a]]", 0, out)


def test_create_list_rule_missing_brackets():
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']]'"):
        create_list_rule("[[a]", 0, out)


def test_create_list_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="overwrite"):
        create_list_rule("[[a]]", 0, out)


# ---------------------------------------------------------------------------
# key_value_rule
# ---------------------------------------------------------------------------


def test_key_value_rule_simple():
    out = Output()
    pos = key_value_rule("a = 1", 0, out, (), float)
    assert pos == 5
    assert out.data.dict == {"a": 1}


def test_key_value_rule_nested():
    out = Output()
    key_value_rule("a.b = 1", 0, out, (), float)
    assert out.data.dict == {"a": {"b": 1}}
    assert out.flags.is_(("a",), Flags.EXPLICIT_NEST) is False  # pending, not finalized


def test_key_value_rule_pending_finalized_by_table():
    out = Output()
    key_value_rule("a.b = 1", 0, out, (), float)
    create_dict_rule("[a]", 0, out)
    assert out.flags.is_(("a",), Flags.EXPLICIT_NEST) is True


def test_key_value_rule_redefine_namespace():
    out = Output()
    key_value_rule("a.b = 1", 0, out, (), float)
    out.flags.finalize_pending()
    with pytest.raises(TOMLDecodeError, match="Cannot redefine namespace"):
        key_value_rule("a.c.d = 1", 0, out, (), float)


def test_key_value_rule_overwrite():
    out = Output()
    key_value_rule("a = 1", 0, out, (), float)
    with pytest.raises(TOMLDecodeError, match="overwrite"):
        key_value_rule("a = 2", 0, out, (), float)


def test_key_value_rule_frozen_parent():
    # inline table makes namespace immutable
    out = Output()
    key_value_rule("a = {b = 1}", 0, out, (), float)
    with pytest.raises(TOMLDecodeError, match="immutable"):
        key_value_rule("a.c = 2", 0, out, (), float)


def test_key_value_rule_freeze_dict():
    out = Output()
    key_value_rule("a = {b = 1}", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True
    assert out.flags.is_(("a", "b"), Flags.FROZEN) is True


def test_key_value_rule_freeze_list():
    out = Output()
    key_value_rule("a = [1]", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True


def test_key_value_rule_header():
    out = Output()
    key_value_rule("x = 1", 0, out, ("h",), float)
    assert out.data.dict == {"h": {"x": 1}}


def test_key_value_rule_parent_not_dict():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="overwrite"):
        key_value_rule("a.b = 1", 0, out, (), float)


# ---------------------------------------------------------------------------
# parse_key_value_pair
# ---------------------------------------------------------------------------


def test_parse_key_value_pair():
    pos, key, value = parse_key_value_pair("a = 42", 0, float, 0)
    assert key == ("a",)
    assert value == 42
    assert pos == 6


def test_parse_key_value_pair_nested_key():
    pos, key, value = parse_key_value_pair('a . b = "x"', 0, float, 0)
    assert key == ("a", "b")
    assert value == "x"


def test_parse_key_value_pair_missing_eq():
    with pytest.raises(TOMLDecodeError, match="Expected '='"):
        parse_key_value_pair("a 1", 0, float, 0)


def test_parse_key_value_pair_eof_before_eq():
    with pytest.raises(TOMLDecodeError, match="Expected '='"):
        parse_key_value_pair("a", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_key / parse_key_part
# ---------------------------------------------------------------------------


def test_parse_key_bare():
    pos, key = parse_key("abc", 0)
    assert key == ("abc",)
    assert pos == 3


def test_parse_key_dotted():
    pos, key = parse_key("a.b.c", 0)
    assert key == ("a", "b", "c")
    assert pos == 5


def test_parse_key_dotted_ws():
    pos, key = parse_key(" a . b ", 0)
    assert key == ("a", "b")


def test_parse_key_quoted():
    pos, key = parse_key('"a b".c', 0)
    assert key == ("a b", "c")


def test_parse_key_too_many_parts():
    n = MAX_KEY_PARTS + 1
    src = ".".join(["a"] * n)
    with pytest.raises(RecursionError):
        parse_key(src, 0)


def test_parse_key_part_bare():
    pos, part = parse_key_part("abc-def_123", 0)
    assert part == "abc-def_123"
    assert pos == 11


def test_parse_key_part_literal():
    pos, part = parse_key_part("'abc'", 0)
    assert part == "abc"
    assert pos == 5


def test_parse_key_part_basic():
    pos, part = parse_key_part('"a\\nb"', 0)
    assert part == "a\nb"
    assert pos == 7


def test_parse_key_part_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key_part("!bad", 0)


def test_parse_key_part_eof():
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key_part("", 0)


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


def test_parse_one_line_basic_str():
    pos, s = parse_one_line_basic_str('"hello"', 0)
    assert s == "hello"
    assert pos == 7


def test_parse_one_line_basic_str_escapes():
    pos, s = parse_one_line_basic_str('"a\\tb\\u00e9"', 0)
    assert s == "a\tb\u00e9"


def test_parse_basic_str():
    pos, s = parse_basic_str('abc"', 0, multiline=False)
    assert s == "abc"
    assert pos == 4


def test_parse_basic_str_illegal_char():
    with pytest.raises(TOMLDecodeError, match="Illegal character"):
        parse_basic_str("\x01a\"", 0, multiline=False)


def test_parse_basic_str_unterminated():
    with pytest.raises(TOMLDecodeError, match="Unterminated"):
        parse_basic_str("abc", 0, multiline=False)


def test_parse_basic_str_multiline():
    src = 'abc"""'
    pos, s = parse_basic_str(src, 0, multiline=True)
    assert s == "abc"
    assert pos == 6


def test_parse_basic_str_multiline_single_quote_inside():
    src = 'a"b"""'
    pos, s = parse_basic_str(src, 0, multiline=True)
    assert s == 'a"b'


def test_parse_multiline_str_literal():
    pos, s = parse_multiline_str("'''abc'''", 0, literal=True)
    assert s == "abc"
    assert pos == 9


def test_parse_multiline_str_literal_leading_newline():
    pos, s = parse_multiline_str("'''\nabc'''", 0, literal=True)
    assert s == "abc"
    assert pos == 10


def test_parse_multiline_str_literal_extra_quotes():
    pos, s = parse_multiline_str("'''''''", 0, literal=True)
    # ''' + '' + '''? Actually: opening ''' then content then 3+ quotes
    assert s in ("'", "''", "")


def test_parse_multiline_str_basic():
    pos, s = parse_multiline_str('"""abc"""', 0, literal=False)
    assert s == "abc"
    assert pos == 9


def test_parse_multiline_str_basic_extra_quote():
    pos, s = parse_multiline_str('"""abc""""', 0, literal=False)
    assert s == 'abc"'


def test_parse_multiline_str_basic_extra_two_quotes():
    pos, s = parse_multiline_str('"""abc"""""', 0, literal=False)
    assert s == 'abc""'


def test_parse_literal_str():
    pos, s = parse_literal_str("'hello'", 0)
    assert s == "hello"
    assert pos == 7


def test_parse_literal_str_illegal_char():
    with pytest.raises(TOMLDecodeError, match="invalid character"):
        parse_literal_str("'a\x01b'", 0)


def test_parse_literal_str_unterminated():
    with pytest.raises(TOMLDecodeError, match="Expected"):
        parse_literal_str("'abc", 0)


# ---------------------------------------------------------------------------
# Escapes & hex
# ---------------------------------------------------------------------------


def test_parse_basic_str_escape_standard():
    for esc, expected in BASIC_STR_ESCAPE_REPLACEMENTS.items():
        pos, result = parse_basic_str_escape(esc, 0)
        assert result == expected
        assert pos == 2


def test_parse_basic_str_escape_hex():
    pos, result = parse_basic_str_escape("\\x41", 0)
    assert result == "A"
    assert pos == 4


def test_parse_basic_str_escape_unicode():
    pos, result = parse_basic_str_escape("\\u00e9", 0)
    assert result == "\u00e9"
    assert pos == 6


def test_parse_basic_str_escape_unicode8():
    pos, result = parse_basic_str_escape("\\U0001F600", 0)
    assert result == "\U0001F600"
    assert pos == 10


def test_parse_basic_str_escape_invalid():
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape("\\q", 0)


def test_parse_basic_str_escape_multiline_line_continuation():
    pos, result = parse_basic_str_escape("\\\n   abc", 0, multiline=True)
    assert result == ""
    assert pos == 6


def test_parse_basic_str_escape_multiline_space_continuation():
    pos, result = parse_basic_str_escape("\\   \nx", 0, multiline=True)
    assert result == ""
    assert pos == 6


def test_parse_basic_str_escape_multiline_bad_continuation():
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape("\\  x", 0, multiline=True)


def test_parse_basic_str_escape_multiline_space_at_eof():
    pos, result = parse_basic_str_escape("\\  ", 0, multiline=True)
    assert result == ""
    assert pos == 3


def test_parse_basic_str_escape_multiline_wrapper():
    pos, result = parse_basic_str_escape_multiline("\\n", 0)
    assert result == "\n"
    assert pos == 2


def test_parse_hex_char():
    pos, result = parse_hex_char("41", 0, 2)
    assert result == "A"
    assert pos == 2


def test_parse_hex_char_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
        parse_hex_char("ZZ", 0, 2)


def test_parse_hex_char_too_short():
    with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
        parse_hex_char("4", 0, 2)


def test_parse_hex_char_surrogate():
    with pytest.raises(TOMLDecodeError, match="not a Unicode scalar"):
        parse_hex_char("D800", 0, 4)


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


def test_parse_array_empty():
    pos, arr = parse_array("[]", 0, float, 0)
    assert arr == []
    assert pos == 2


def test_parse_array_values():
    pos, arr = parse_array("[1, 2, 3]", 0, float, 0)
    assert arr == [1, 2, 3]
    assert pos == 9


def test_parse_array_trailing_comma():
    pos, arr = parse_array("[1, 2,]", 0, float, 0)
    assert arr == [1, 2]
    assert pos == 8


def test_parse_array_multiline_comments():
    src = "[\n  1, # comment\n  2,\n]"
    pos, arr = parse_array(src, 0, float, 0)
    assert arr == [1, 2]
    assert pos == len(src)


def test_parse_array_unclosed():
    with pytest.raises(TOMLDecodeError, match="Unclosed array"):
        parse_array("[1 2]", 0, float, 0)


def test_parse_array_nested():
    pos, arr = parse_array("[[1], [2, 3]]", 0, float, 0)
    assert arr == [[1], [2, 3]]


def test_parse_array_with_parse_float():
    pos, arr = parse_array("[1.5]", 0, lambda s: "F" + s, 0)
    assert arr == ["F1.5"]


# ---------------------------------------------------------------------------
# Inline tables
# ---------------------------------------------------------------------------


def test_parse_inline_table_empty():
    pos, table = parse_inline_table("{}", 0, float, 0)
    assert table == {}
    assert pos == 2


def test_parse_inline_table_simple():
    pos, table = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
    assert table == {"a": 1, "b": 2}
    assert pos == 14


def test_parse_inline_table_trailing_comma():
    pos, table = parse_inline_table("{a = 1,}", 0, float, 0)
    assert table == {"a": 1}
    assert pos == 9


def test_parse_inline_table_dotted():
    pos, table = parse_inline_table("{a.b = 1}", 0, float, 0)
    assert table == {"a": {"b": 1}}


def test_parse_inline_table_duplicate():
    with pytest.raises(TOMLDecodeError, match="Duplicate inline table key"):
        parse_inline_table("{a = 1, a = 2}", 0, float, 0)


def test_parse_inline_table_unclosed():
    with pytest.raises(TOMLDecodeError, match="Unclosed inline table"):
        parse_inline_table("{a = 1", 0, float, 0)


def test_parse_inline_table_overwrite():
    with pytest.raises(TOMLDecodeError, match="overwrite"):
        parse_inline_table("{a = 1, a.b = 2}", 0, float, 0)


def test_parse_inline_table_frozen_namespace():
    # {a = {b = 1}, a.c = 2} -> a becomes frozen due to value being a dict
    with pytest.raises(TOMLDecodeError, match="immutable"):
        parse_inline_table("{a = {b = 1}, a.c = 2}", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------


def test_parse_value_basic_string():
    pos, val = parse_value('"hi"', 0, float, 0)
    assert val == "hi"
    assert pos == 4


def test_parse_value_multiline_basic():
    pos, val = parse_value('"""hi"""', 0, float, 0)
    assert val == "hi"
    assert pos == 9


def test_parse_value_literal_string():
    pos, val = parse_value("'hi'", 0, float, 0)
    assert val == "hi"
    assert pos == 4


def test_parse_value_multiline_literal():
    pos, val = parse_value("'''hi'''", 0, float, 0)
    assert val == "hi"
    assert pos == 9


def test_parse_value_true():
    pos, val = parse_value("true", 0, float, 0)
    assert val is True
    assert pos == 4


def test_parse_value_false():
    pos, val = parse_value("false", 0, float, 0)
    assert val is False
    assert pos == 5


def test_parse_value_array():
    pos, val = parse_value("[1, 2]", 0, float, 0)
    assert val == [1, 2]
    assert pos == 6


def test_parse_value_inline_table():
    pos, val = parse_value("{a = 1}", 0, float, 0)
    assert val == {"a": 1}
    assert pos == 7


def test_parse_value_date():
    pos, val = parse_value("2021-01-01", 0, float, 0)
    assert val == datetime.date(2021, 1, 1)
    assert pos == 10


def test_parse_value_datetime():
    pos, val = parse_value("2021-01-01T00:00:00Z", 0, float, 0)
    assert val == datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)


def test_parse_value_invalid_date():
    with pytest.raises(TOMLDecodeError, match="Invalid date"):
        parse_value("2021-02-30", 0, float, 0)


def test_parse_value_localtime():
    pos, val = parse_value("12:34:56", 0, float, 0)
    assert val == datetime.time(12, 34, 56)
    assert pos == 8


def test_parse_value_integer():
    pos, val = parse_value("42", 0, float, 0)
    assert val == 42
    assert pos == 2


def test_parse_value_hex_integer():
    pos, val = parse_value("0x10", 0, float, 0)
    assert val == 16


def test_parse_value_float():
    pos, val = parse_value("1.5", 0, float, 0)
    assert val == 1.5


def test_parse_value_float_custom():
    pos, val = parse_value("1.5", 0, lambda s: ("f", s), 0)
    assert val == ("f", "1.5")


def test_parse_value_inf():
    pos, val = parse_value("inf", 0, float, 0)
    assert val == float("inf")
    assert pos == 3


def test_parse_value_nan():
    pos, val = parse_value("nan", 0, float, 0)
    assert val != val


def test_parse_value_neg_inf():
    pos, val = parse_value("-inf", 0, float, 0)
    assert val == float("-inf")
    assert pos == 4


def test_parse_value_pos_nan():
    pos, val = parse_value("+nan", 0, float, 0)
    assert val != val
    assert pos == 4


def test_parse_value_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("!?", 0, float, 0)


def test_parse_value_eof():
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("", 0, float, 0)


def test_parse_value_too_deep():
    with pytest.raises(RecursionError):
        parse_value("1", 0, float, MAX_INLINE_NESTING + 1)


# ---------------------------------------------------------------------------
# Unicode scalar values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cp,expected",
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
def test_is_unicode_scalar_value(cp, expected):
    assert is_unicode_scalar_value(cp) is expected


# ---------------------------------------------------------------------------
# make_safe_parse_float
# ---------------------------------------------------------------------------


def test_make_safe_parse_float_default():
    assert make_safe_parse_float(float) is float


def test_make_safe_parse_float_custom():
    def pf(s):
        return ("custom", s)

    safe = make_safe_parse_float(pf)
    assert safe("1.5") == ("custom", "1.5")


def test_make_safe_parse_float_dict():
    def pf(s):
        return {}

    safe = make_safe_parse_float(pf)
    with pytest.raises(ValueError, match="parse_float"):
        safe("1.5")


def test_make_safe_parse_float_list():
    def pf(s):
        return []

    safe = make_safe_parse_float(pf)
    with pytest.raises(ValueError, match="parse_float"):
        safe("1.5")
