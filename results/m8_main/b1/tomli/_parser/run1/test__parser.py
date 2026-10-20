from __future__ import annotations

import datetime
import math
import sys
import warnings

import pytest

from tomli._parser import (
    BASIC_STR_ESCAPE_REPLACEMENTS,
    DEPRECATED_DEFAULT,
    MAX_INLINE_NESTING,
    MAX_KEY_PARTS,
    NestedDict,
    Output,
    TOMLDecodeError,
    Flags,
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


# --------------------------------------------------------------------------
# load / loads
# --------------------------------------------------------------------------


def test_load_basic(tmp_path):
    p = tmp_path / "x.toml"
    p.write_bytes(b"a = 1\nb = 'two'\n")
    with open(p, "rb") as f:
        assert load(f) == {"a": 1, "b": "two"}


def test_load_text_file_raises_typeerror(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text("a = 1\n")
    with open(p, "r") as f:
        with pytest.raises(TypeError, match="binary mode"):
            load(f)


def test_load_respects_parse_float(tmp_path):
    p = tmp_path / "x.toml"
    p.write_bytes(b"a = 1.5\n")
    with open(p, "rb") as f:
        result = load(f, parse_float=lambda s: ("F", s))
    assert result == {"a": ("F", "1.5")}


def test_loads_empty():
    assert loads("") == {}


def test_loads_basic():
    assert loads('x = 1\ny = "s"\n') == {"x": 1, "y": "s"}


def test_loads_crlf_normalized():
    assert loads("a = 1\r\nb = 2\r\n") == {"a": 1, "b": 2}


def test_loads_crlf_in_multiline_basic_string():
    # \r\n -> \n normalization also happens inside basic string literals;
    # a one-line basic string does not permit a raw newline
    with pytest.raises(TOMLDecodeError, match="Illegal character"):
        loads('s = "a\r\nb"')


def test_loads_rejects_non_str():
    with pytest.raises(TypeError, match="Expected str object"):
        loads(b"a = 1")
    with pytest.raises(TypeError, match="Expected str object"):
        loads(123)  # type: ignore[arg-type]


def test_loads_comment_only():
    assert loads("# just a comment") == {}


def test_loads_invalid_statement():
    with pytest.raises(TOMLDecodeError, match="Invalid statement"):
        loads("!nope")


def test_loads_expect_newline_after_statement():
    with pytest.raises(TOMLDecodeError, match="Expected newline"):
        loads("a = 1 b = 2")


def test_loads_table():
    assert loads("[a]\nb = 1\n") == {"a": {"b": 1}}


def test_loads_array_of_tables():
    assert loads("[[a]]\nb = 1\n[[a]]\nb = 2\n") == {"a": [{"b": 1}, {"b": 2}]}


# --------------------------------------------------------------------------
# TOMLDecodeError
# --------------------------------------------------------------------------


def test_decode_error_basic_attrs():
    doc = "abc\ndef"
    e = TOMLDecodeError("boom", doc, 5)
    assert e.msg == "boom"
    assert e.doc == doc
    assert e.pos == 5
    assert e.lineno == 2
    assert e.colno == 2
    assert str(e) == "boom (at line 2, column 2)"


def test_decode_error_first_line_column():
    e = TOMLDecodeError("bad", "hello", 3)
    assert e.lineno == 1
    assert e.colno == 4
    assert str(e) == "bad (at line 1, column 4)"


def test_decode_error_end_of_document():
    doc = "hello"
    e = TOMLDecodeError("bad", doc, len(doc))
    assert "end of document" in str(e)


def test_decode_error_free_form_args_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e = TOMLDecodeError("some message")
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert isinstance(e, TOMLDecodeError)
    assert str(e) == "some message"


def test_decode_error_free_form_with_pos():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        e = TOMLDecodeError("m", "doc", 1)
    # this is not deprecated because all three are proper types
    assert e.lineno == 1
    assert e.colno == 2


def test_decode_error_extra_args_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e = TOMLDecodeError("m", "d", 0, "extra")
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert isinstance(e, TOMLDecodeError)


def test_deprecated_default_sentinel_is_a_class():
    assert isinstance(DEPRECATED_DEFAULT, type)


# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------


def test_flags_set_and_is_non_recursive():
    f = Flags()
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
    f.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True


def test_flags_is_empty_key_always_false():
    f = Flags()
    assert f.is_((), Flags.FROZEN) is False
    assert f.is_((), Flags.EXPLICIT_NEST) is False


def test_flags_set_empty_key_raises_index_error():
    f = Flags()
    with pytest.raises(IndexError):
        f.set((), Flags.FROZEN, recursive=True)


def test_flags_recursive_propagates_to_children():
    f = Flags()
    f.set(("a",), Flags.FROZEN, recursive=True)
    f.set(("a", "b", "c"), Flags.EXPLICIT_NEST, recursive=False)
    assert f.is_(("a", "b"), Flags.FROZEN) is True
    assert f.is_(("a", "b", "c"), Flags.FROZEN) is True


def test_flags_non_recursive_does_not_propagate():
    f = Flags()
    f.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    f.set(("a", "b"), Flags.FROZEN, recursive=False)
    assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is False


def test_flags_unset_all_removes():
    f = Flags()
    f.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
    f.unset_all(("a", "b"))
    assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is False


def test_flags_unset_all_missing_path_noop():
    f = Flags()
    f.unset_all(("x", "y"))


def test_flags_pending():
    f = Flags()
    f.add_pending(("a",), Flags.EXPLICIT_NEST)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
    f.finalize_pending()
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True


# --------------------------------------------------------------------------
# NestedDict
# --------------------------------------------------------------------------


def test_nesteddict_get_or_create_nest():
    n = NestedDict()
    d = n.get_or_create_nest(("a", "b"))
    assert d == {}
    n.dict["a"]["b"]["x"] = 1
    assert n.get_or_create_nest(("a", "b")) is n.dict["a"]["b"]


def test_nesteddict_access_lists():
    n = NestedDict()
    n.dict["a"] = [{"v": 1}]
    assert n.get_or_create_nest(("a",)) is n.dict["a"][-1]


def test_nesteddict_access_lists_false():
    n = NestedDict()
    n.dict["a"] = [{"v": 1}]
    with pytest.raises(KeyError):
        n.get_or_create_nest(("a",), access_lists=False)


def test_nesteddict_no_nest_behind_key():
    n = NestedDict()
    n.dict["a"] = 1
    with pytest.raises(KeyError):
        n.get_or_create_nest(("a", "b"))


def test_nesteddict_append_nest_to_list_create():
    n = NestedDict()
    n.append_nest_to_list(("a",))
    assert n.dict == {"a": [{}]}


def test_nesteddict_append_nest_to_list_append():
    n = NestedDict()
    n.append_nest_to_list(("a",))
    n.append_nest_to_list(("a",))
    assert n.dict == {"a": [{}, {}]}


def test_nesteddict_append_not_a_list():
    n = NestedDict()
    n.dict["a"] = 1
    with pytest.raises(KeyError, match="other than list"):
        n.append_nest_to_list(("a",))


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def test_output():
    o = Output()
    assert o.data.dict == {}
    assert isinstance(o.flags, Flags)


# --------------------------------------------------------------------------
# skip_chars / skip_until / skip_comment / skip_comments_and_array_ws
# --------------------------------------------------------------------------


def test_skip_chars():
    assert skip_chars("   abc", 0, " ") == 3
    assert skip_chars("abc", 0, " ") == 0
    assert skip_chars("   ", 0, " ") == 3


def test_skip_until_found():
    assert skip_until("abc,def", 0, ",", error_on=frozenset(), error_on_eof=False) == 3


def test_skip_until_eof_ok():
    assert skip_until("abc", 0, ",", error_on=frozenset(), error_on_eof=False) == 3


def test_skip_until_eof_error():
    with pytest.raises(TOMLDecodeError, match="Expected"):
        skip_until("abc", 0, ",", error_on=frozenset(), error_on_eof=True)


def test_skip_until_error_on_char():
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        skip_until("aXb,", 0, ",", error_on=frozenset("X"), error_on_eof=False)


def test_skip_comment_none():
    assert skip_comment("abc", 0) == 0


def test_skip_comment_with_comment():
    assert skip_comment("# hello\nrest", 0) == 7


def test_skip_comment_at_eof():
    assert skip_comment("# hello", 0) == 7


def test_skip_comment_index_error():
    assert skip_comment("", 0) == 0


def test_skip_comment_illegal_char():
    with pytest.raises(TOMLDecodeError, match="Found invalid character"):
        skip_comment("# a\x00b\n", 0)


def test_skip_comments_and_array_ws():
    src = "  # c\n  x"
    assert skip_comments_and_array_ws(src, 0) == 8


def test_skip_comments_and_array_ws_stops():
    assert skip_comments_and_array_ws("abc", 0) == 0


# --------------------------------------------------------------------------
# create_dict_rule / create_list_rule
# --------------------------------------------------------------------------


def test_create_dict_rule():
    out = Output()
    pos, key = create_dict_rule("[a.b]\nx=1", 0, out)
    assert key == ("a", "b")
    assert out.data.dict == {"a": {"b": {}}}
    assert pos == 5


def test_create_dict_rule_twice_raises():
    out = Output()
    create_dict_rule("[a]\n", 0, out)
    with pytest.raises(TOMLDecodeError, match="Cannot declare"):
        create_dict_rule("[a]\n", 0, out)


def test_create_dict_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']'"):
        create_dict_rule("[a\n", 0, out)


def test_create_dict_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite"):
        create_dict_rule("[a]\n", 0, out)


def test_create_list_rule():
    out = Output()
    pos, key = create_list_rule("[[a]]\n", 0, out)
    assert key == ("a",)
    assert out.data.dict == {"a": [{}]}
    assert pos == 5


def test_create_list_rule_append():
    out = Output()
    create_list_rule("[[a]]\n", 0, out)
    create_list_rule("[[a]]\n", 0, out)
    assert out.data.dict == {"a": [{}, {}]}


def test_create_list_rule_missing_brackets():
    out = Output()
    with pytest.raises(TOMLDecodeError, match="Expected ']]'"):
        create_list_rule("[[a]\n", 0, out)


def test_create_list_rule_frozen_raises():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="immutable"):
        create_list_rule("[[a]]\n", 0, out)


def test_create_list_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite"):
        create_list_rule("[[a]]\n", 0, out)


# --------------------------------------------------------------------------
# key_value_rule
# --------------------------------------------------------------------------


def test_key_value_rule_simple():
    out = Output()
    pos = key_value_rule("a = 1\n", 0, out, (), float)
    assert out.data.dict == {"a": 1}
    assert pos == 5


def test_key_value_rule_duplicate():
    out = Output()
    key_value_rule("a = 1\n", 0, out, (), float)
    with pytest.raises(TOMLDecodeError, match="Cannot overwrite"):
        key_value_rule("a = 2\n", 0, out, (), float)


def test_key_value_rule_with_header():
    out = Output()
    out.data.dict.setdefault("h", {})
    key_value_rule("a = 1\n", 0, out, ("h",), float)
    assert out.data.dict == {"h": {"a": 1}}


def test_key_value_rule_frozen_parent():
    out = Output()
    out.flags.set(("h",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError, match="immutable"):
        key_value_rule("a = 1\n", 0, out, ("h",), float)


def test_key_value_rule_dotted_redefine():
    out = Output()
    out.flags.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    with pytest.raises(TOMLDecodeError, match="Cannot redefine"):
        key_value_rule("a.b = 1\n", 0, out, (), float)


def test_key_value_rule_freeze_dict_value():
    out = Output()
    key_value_rule("a = {x = 1}\n", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True


def test_key_value_rule_freeze_list_value():
    out = Output()
    key_value_rule("a = [1]\n", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True


# --------------------------------------------------------------------------
# parse_key_value_pair / parse_key / parse_key_part
# --------------------------------------------------------------------------


def test_parse_key_value_pair():
    pos, key, value = parse_key_value_pair("abc = 42 rest", 0, float, 0)
    assert key == ("abc",)
    assert value == 42
    assert pos == 8


def test_parse_key_value_pair_missing_equals():
    with pytest.raises(TOMLDecodeError, match="Expected '='"):
        parse_key_value_pair("abc 42", 0, float, 0)


def test_parse_key_simple():
    pos, key = parse_key("abc = 1", 0)
    assert key == ("abc",)
    assert pos == 4


def test_parse_key_dotted():
    pos, key = parse_key("a . b . c = 1", 0)
    assert key == ("a", "b", "c")


def test_parse_key_quoted():
    pos, key = parse_key('"a b" = 1', 0)
    assert key == ("a b",)


def test_parse_key_literal():
    pos, key = parse_key("'a.b' = 1", 0)
    assert key == ("a.b",)


def test_parse_key_too_many_parts():
    src = ".".join(["a"] * (MAX_KEY_PARTS + 1)) + " = 1"
    with pytest.raises(RecursionError):
        parse_key(src, 0)


def test_parse_key_part_bare():
    pos, part = parse_key_part("abc-def_1 x", 0)
    assert part == "abc-def_1"
    assert pos == 9


def test_parse_key_part_basic_str():
    pos, part = parse_key_part('"x"', 0)
    assert part == "x"


def test_parse_key_part_literal_str():
    pos, part = parse_key_part("'x'", 0)
    assert part == "x"


def test_parse_key_part_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
        parse_key_part(".x", 0)


# --------------------------------------------------------------------------
# String parsing
# --------------------------------------------------------------------------


def test_parse_one_line_basic_str():
    pos, s = parse_one_line_basic_str('"hello" x', 0)
    assert s == "hello"
    assert pos == 7


def test_parse_basic_str_simple():
    pos, s = parse_basic_str("abc\"rest", 0, multiline=False)
    assert s == "abc"
    assert pos == 4


def test_parse_basic_str_unterminated():
    with pytest.raises(TOMLDecodeError, match="Unterminated string"):
        parse_basic_str("abc", 0, multiline=False)


def test_parse_basic_str_illegal_char():
    with pytest.raises(TOMLDecodeError, match="Illegal character"):
        parse_basic_str("a\x00b", 0, multiline=False)


def test_parse_basic_str_escape():
    pos, s = parse_basic_str('a\\tb"', 0, multiline=False)
    assert s == "a\tb"
    assert pos == 5


def test_parse_basic_str_multiline():
    pos, s = parse_basic_str('abc"""rest', 0, multiline=True)
    assert s == "abc"
    assert pos == 6


def test_parse_basic_str_multiline_quotes_inside():
    src = 'a""b"""rest'
    pos, s = parse_basic_str(src, 0, multiline=True)
    assert s == 'a""b'
    assert pos == 7


def test_parse_basic_str_escape_simple():
    assert parse_basic_str_escape(r"\n", 0) == (2, "\n")
    assert parse_basic_str_escape(r"\t", 0) == (2, "\t")
    assert parse_basic_str_escape(r"\e", 0) == (2, "\x1b")
    assert parse_basic_str_escape(r"\\", 0) == (2, "\\")
    assert parse_basic_str_escape(r"\"", 0) == (2, '"')


def test_parse_basic_str_escape_hex():
    assert parse_basic_str_escape(r"\x41", 0) == (4, "A")
    assert parse_basic_str_escape(r"\u00e9", 0) == (6, "é")
    assert parse_basic_str_escape(r"\U0001F600", 0) == (10, "\U0001F600")


def test_parse_basic_str_escape_invalid():
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape(r"\q", 0)


def test_parse_basic_str_escape_multiline_line_ending():
    pos, s = parse_basic_str_escape_multiline("\\\n   x", 0)
    assert s == ""
    assert pos == 5


def test_parse_basic_str_escape_multiline_space():
    pos, s = parse_basic_str_escape_multiline("\\  \n x", 0)
    assert s == ""
    assert pos == 5


def test_parse_basic_str_escape_multiline_space_tab():
    pos, s = parse_basic_str_escape_multiline("\\\t \nx", 0)
    assert s == ""
    assert pos == 4


def test_parse_basic_str_escape_multiline_space_eof():
    pos, s = parse_basic_str_escape_multiline("\\  ", 0)
    assert s == ""
    assert pos == 3


def test_parse_basic_str_escape_multiline_space_bad():
    with pytest.raises(TOMLDecodeError, match="Unescaped"):
        parse_basic_str_escape_multiline("\\  x", 0)


def test_parse_hex_char_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid hex"):
        parse_hex_char("zz", 0, 2)


def test_parse_hex_char_too_short():
    with pytest.raises(TOMLDecodeError, match="Invalid hex"):
        parse_hex_char("4", 0, 2)


def test_parse_hex_char_not_scalar():
    with pytest.raises(TOMLDecodeError, match="scalar"):
        parse_hex_char("D800", 0, 4)


def test_parse_literal_str():
    pos, s = parse_literal_str("'abc' rest", 0)
    assert s == "abc"
    assert pos == 5


def test_parse_literal_str_unterminated():
    with pytest.raises(TOMLDecodeError):
        parse_literal_str("'abc", 0)


def test_parse_multiline_str_basic():
    pos, s = parse_multiline_str('"""\nhello"""', 0, literal=False)
    assert s == "hello"
    assert pos == 12


def test_parse_multiline_str_literal():
    pos, s = parse_multiline_str("'''\nhello'''", 0, literal=True)
    assert s == "hello"
    assert pos == 12


def test_parse_multiline_str_extra_quote():
    pos, s = parse_multiline_str('"""ab""""', 0, literal=False)
    assert s == 'ab"'
    assert pos == 9


def test_parse_multiline_str_extra_two_quotes():
    pos, s = parse_multiline_str('"""ab"""""', 0, literal=False)
    assert s == 'ab""'
    assert pos == 10


def test_parse_multiline_str_literal_extra_quote():
    pos, s = parse_multiline_str("'''ab''''", 0, literal=True)
    assert s == "ab'"
    assert pos == 9


# --------------------------------------------------------------------------
# parse_array / parse_inline_table
# --------------------------------------------------------------------------


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


def test_parse_array_with_comments_and_ws():
    src = "[\n  1, # c\n  2\n]"
    pos, arr = parse_array(src, 0, float, 0)
    assert arr == [1, 2]


def test_parse_array_unclosed():
    with pytest.raises(TOMLDecodeError, match="Unclosed array"):
        parse_array("[1 2]", 0, float, 0)


def test_parse_inline_table_empty():
    pos, d = parse_inline_table("{}", 0, float, 0)
    assert d == {}
    assert pos == 2


def test_parse_inline_table_values():
    pos, d = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
    assert d == {"a": 1, "b": 2}


def test_parse_inline_table_trailing_comma():
    pos, d = parse_inline_table("{a = 1,}", 0, float, 0)
    assert d == {"a": 1}


def test_parse_inline_table_dotted():
    pos, d = parse_inline_table("{a.b = 1}", 0, float, 0)
    assert d == {"a": {"b": 1}}


def test_parse_inline_table_duplicate_key():
    with pytest.raises(TOMLDecodeError, match="Duplicate inline table key"):
        parse_inline_table("{a = 1, a = 2}", 0, float, 0)


def test_parse_inline_table_unclosed():
    with pytest.raises(TOMLDecodeError, match="Unclosed inline table"):
        parse_inline_table("{a = 1 b = 2}", 0, float, 0)


def test_parse_inline_table_frozen_namespace():
    with pytest.raises(TOMLDecodeError, match="immutable"):
        parse_inline_table("{a = {b = 1}, a.c = 2}", 0, float, 0)


def test_parse_inline_table_overwrite_value():
    with pytest.raises(TOMLDecodeError, match="Duplicate inline table key"):
        parse_inline_table("{a.b = 1, a = 2}", 0, float, 0)


# --------------------------------------------------------------------------
# parse_value
# --------------------------------------------------------------------------


def test_parse_value_basic_str():
    assert parse_value('"abc"', 0, float, 0) == (5, "abc")


def test_parse_value_multiline_basic_str():
    pos, s = parse_value('"""abc"""', 0, float, 0)
    assert s == "abc"


def test_parse_value_literal_str():
    assert parse_value("'abc'", 0, float, 0) == (5, "abc")


def test_parse_value_multiline_literal_str():
    pos, s = parse_value("'''abc'''", 0, float, 0)
    assert s == "abc"


def test_parse_value_booleans():
    assert parse_value("true", 0, float, 0) == (4, True)
    assert parse_value("false", 0, float, 0) == (5, False)


def test_parse_value_array():
    assert parse_value("[1, 2]", 0, float, 0) == (6, [1, 2])


def test_parse_value_inline_table():
    assert parse_value("{a = 1}", 0, float, 0) == (7, {"a": 1})


def test_parse_value_date():
    pos, d = parse_value("2020-01-02", 0, float, 0)
    assert d == datetime.date(2020, 1, 2)


def test_parse_value_datetime():
    pos, d = parse_value("2020-01-02T03:04:05Z", 0, float, 0)
    assert d == datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)


def test_parse_value_out_of_range_date():
    # "2020-02-30" matches RE_DATETIME syntactically (valid month),
    # but February 30 is invalid so match_to_datetime raises ValueError,
    # which the parser converts into a TOMLDecodeError.
    with pytest.raises(TOMLDecodeError, match="Invalid date or datetime"):
        parse_value("2020-02-30", 0, float, 0)


def test_parse_value_invalid_month_not_matched_as_date():
    # "2020-13-02" does not match RE_DATETIME (month 13), so it falls
    # through to the number parser which matches just "2020".
    assert parse_value("2020-13-02", 0, float, 0) == (4, 2020)


def test_parse_value_localtime():
    pos, t = parse_value("03:04:05", 0, float, 0)
    assert t == datetime.time(3, 4, 5)


def test_parse_value_integer():
    assert parse_value("42", 0, float, 0) == (2, 42)


def test_parse_value_float():
    pos, v = parse_value("3.14", 0, float, 0)
    assert v == 3.14


def test_parse_value_special_floats():
    assert math.isinf(parse_value("inf", 0, float, 0)[1])
    assert math.isinf(parse_value("+inf", 0, float, 0)[1])
    assert math.isinf(parse_value("-inf", 0, float, 0)[1])
    assert math.isnan(parse_value("nan", 0, float, 0)[1])
    assert math.isnan(parse_value("+nan", 0, float, 0)[1])
    assert math.isnan(parse_value("-nan", 0, float, 0)[1])


def test_parse_value_invalid():
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("nope", 0, float, 0)


def test_parse_value_eof():
    with pytest.raises(TOMLDecodeError, match="Invalid value"):
        parse_value("", 0, float, 0)


def test_parse_value_nesting_limit():
    with pytest.raises(RecursionError):
        parse_value("", 0, float, MAX_INLINE_NESTING + 1)


# --------------------------------------------------------------------------
# is_unicode_scalar_value
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# make_safe_parse_float
# --------------------------------------------------------------------------


def test_make_safe_parse_float_default_identity():
    assert make_safe_parse_float(float) is float


def test_make_safe_parse_float_ok():
    def custom(s):
        return "F:" + s

    safe = make_safe_parse_float(custom)
    assert safe("1.5") == "F:1.5"


def test_make_safe_parse_float_rejects_dict():
    def custom(s):
        return {}

    safe = make_safe_parse_float(custom)
    with pytest.raises(ValueError, match="dicts or lists"):
        safe("1.0")


def test_make_safe_parse_float_rejects_list():
    def custom(s):
        return []

    safe = make_safe_parse_float(custom)
    with pytest.raises(ValueError, match="dicts or lists"):
        safe("1.0")


def test_make_safe_parse_float_used_by_loads():
    def custom(s):
        return {}

    with pytest.raises(ValueError, match="dicts or lists"):
        loads("a = 1.0", parse_float=custom)


# --------------------------------------------------------------------------
# BASIC_STR_ESCAPE_REPLACEMENTS sanity
# --------------------------------------------------------------------------


def test_escape_replacements_content():
    assert BASIC_STR_ESCAPE_REPLACEMENTS[r"\n"] == "\n"
    assert BASIC_STR_ESCAPE_REPLACEMENTS[r"\t"] == "\t"
    assert BASIC_STR_ESCAPE_REPLACEMENTS[r"\\"] == "\\"
    assert BASIC_STR_ESCAPE_REPLACEMENTS[r"\""] == '"'
    assert len(BASIC_STR_ESCAPE_REPLACEMENTS) == 8


# --------------------------------------------------------------------------
# Misc constants
# --------------------------------------------------------------------------


def test_max_key_parts_matches_recursion_limit():
    assert MAX_KEY_PARTS == sys.getrecursionlimit()


def test_max_inline_nesting_value():
    assert MAX_INLINE_NESTING == 400
