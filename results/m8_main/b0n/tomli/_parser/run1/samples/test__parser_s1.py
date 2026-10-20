import io
import math
import sys
import warnings
from datetime import date, datetime, time, timezone, timedelta

import pytest

from tomli._parser import (
    DEPRECATED_DEFAULT,
    TOMLDecodeError,
    load,
    loads,
    Flags,
    NestedDict,
    Output,
    skip_chars,
    skip_until,
    skip_comment,
    skip_comments_and_array_ws,
    create_dict_rule,
    create_list_rule,
    key_value_rule,
    parse_key_value_pair,
    parse_key,
    parse_key_part,
    parse_one_line_basic_str,
    parse_array,
    parse_inline_table,
    parse_basic_str_escape,
    parse_basic_str_escape_multiline,
    parse_hex_char,
    parse_literal_str,
    parse_multiline_str,
    parse_basic_str,
    parse_value,
    is_unicode_scalar_value,
    make_safe_parse_float,
    ILLEGAL_COMMENT_CHARS,
    MAX_INLINE_NESTING,
)


# ---------------------------------------------------------------------------
# TOMLDecodeError
# ---------------------------------------------------------------------------

def test_toml_decode_error_basic():
    doc = "a = 1\nb = 2\n"
    exc = TOMLDecodeError("msg", doc, 6)
    assert isinstance(exc, ValueError)
    assert exc.msg == "msg"
    assert exc.doc == doc
    assert exc.pos == 6
    assert exc.lineno == 2
    assert exc.colno == 1
    assert "msg" in str(exc)
    assert "line 2, column 1" in str(exc)


def test_toml_decode_error_first_line():
    exc = TOMLDecodeError("oops", "abcdef", 3)
    assert exc.lineno == 1
    assert exc.colno == 4
    assert "line 1, column 4" in str(exc)


def test_toml_decode_error_end_of_document():
    doc = "abc"
    exc = TOMLDecodeError("boom", doc, 3)
    assert exc.lineno == 1
    assert exc.colno == 4
    assert "end of document" in str(exc)


def test_toml_decode_error_deprecated_default():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exc = TOMLDecodeError()
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
    assert isinstance(exc, TOMLDecodeError)


def test_toml_decode_error_deprecated_freeform():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exc = TOMLDecodeError("a", "b", 1, "extra")
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "extra" in str(exc)


def test_toml_decode_error_deprecated_positional_args():
    with pytest.warns(DeprecationWarning):
        exc = TOMLDecodeError("msg", "doc")
    assert "msg" in str(exc)


def test_deprecated_default_sentinel_exists():
    assert isinstance(DEPRECATED_DEFAULT, type)


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------

def test_load_basic():
    fp = io.BytesIO(b"a = 1\nb = 'hello'\n")
    assert load(fp) == {"a": 1, "b": "hello"}


def test_load_text_mode_raises():
    with pytest.raises(TypeError):
        load(io.StringIO("a = 1"))


def test_load_parse_float():
    fp = io.BytesIO(b"x = 1.5\n")
    from decimal import Decimal
    assert load(fp, parse_float=Decimal) == {"x": Decimal("1.5")}


def test_loads_basic():
    assert loads("a = 1\nb = 'hello'\n") == {"a": 1, "b": "hello"}


def test_loads_empty():
    assert loads("") == {}


def test_loads_crlf_normalized():
    assert loads("a = 1\r\nb = 2\r\n") == {"a": 1, "b": 2}


def test_loads_bytes_raises():
    with pytest.raises(TypeError):
        loads(b"a = 1")


def test_loads_invalid_statement():
    with pytest.raises(TOMLDecodeError):
        loads("? = 1\n")


def test_loads_expected_newline():
    with pytest.raises(TOMLDecodeError):
        loads("a = 1 b = 2\n")


def test_loads_comment():
    assert loads("# a comment\na = 1 # trailing\n") == {"a": 1}


def test_loads_table():
    assert loads("[t]\nx = 1\n") == {"t": {"x": 1}}


def test_loads_array_of_tables():
    assert loads("[[t]]\nx = 1\n[[t]]\nx = 2\n") == {"t": [{"x": 1}, {"x": 2}]}


def test_loads_parse_float_custom():
    from decimal import Decimal
    result = loads("x = 3.14\n", parse_float=Decimal)
    assert result == {"x": Decimal("3.14")}


def test_loads_safe_parse_float_rejects_dict():
    def bad(s):
        return {}
    with pytest.raises(ValueError):
        loads("x = 1.0\n", parse_float=bad)


def test_loads_safe_parse_float_rejects_list():
    def bad(s):
        return []
    with pytest.raises(ValueError):
        loads("x = 1.0\n", parse_float=bad)


def test_loads_nested_tables():
    src = "[a]\n[a.b]\nc = 1\n"
    assert loads(src) == {"a": {"b": {"c": 1}}}


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def test_flags_set_and_is():
    f = Flags()
    f.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
    assert f.is_(("a",), Flags.FROZEN) is False


def test_flags_empty_key():
    f = Flags()
    assert f.is_((), Flags.FROZEN) is False


def test_flags_missing_key():
    f = Flags()
    assert f.is_(("x", "y"), Flags.FROZEN) is False


def test_flags_recursive():
    f = Flags()
    f.set(("a",), Flags.FROZEN, recursive=True)
    assert f.is_(("a",), Flags.FROZEN) is True
    assert f.is_(("a", "b"), Flags.FROZEN) is True
    assert f.is_(("a", "b", "c"), Flags.FROZEN) is True


def test_flags_pending_and_finalize():
    f = Flags()
    f.add_pending(("a",), Flags.EXPLICIT_NEST)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
    f.finalize_pending()
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True


def test_flags_unset_all():
    f = Flags()
    f.set(("a",), Flags.FROZEN, recursive=False)
    f.set(("a", "b"), Flags.FROZEN, recursive=False)
    f.unset_all(("a",))
    assert f.is_(("a",), Flags.FROZEN) is False
    assert f.is_(("a", "b"), Flags.FROZEN) is False


def test_flags_unset_all_missing_parent():
    f = Flags()
    f.unset_all(("nonexistent", "child"))
    # no error


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------

def test_nested_dict_get_or_create():
    nd = NestedDict()
    d = nd.get_or_create_nest(("a", "b"))
    assert d == {}
    nd.get_or_create_nest(("a", "b"))["x"] = 1
    assert nd.dict == {"a": {"b": {"x": 1}}}


def test_nested_dict_no_nest():
    nd = NestedDict()
    nd.dict["a"] = 1
    with pytest.raises(KeyError):
        nd.get_or_create_nest(("a", "b"))


def test_nested_dict_access_lists():
    nd = NestedDict()
    nd.dict["a"] = [{"x": 1}, {"x": 2}]
    d = nd.get_or_create_nest(("a", "y"))
    assert d == {}
    assert nd.dict["a"][-1] == {"x": 2, "y": {}}


def test_nested_dict_append_nest_to_list():
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    assert nd.dict == {"a": [{}]}
    nd.append_nest_to_list(("a",))
    assert nd.dict == {"a": [{}, {}]}


def test_nested_dict_append_nest_to_list_not_list():
    nd = NestedDict()
    nd.dict["a"] = 5
    with pytest.raises(KeyError):
        nd.append_nest_to_list(("a",))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def test_output_init():
    out = Output()
    assert isinstance(out.data, NestedDict)
    assert isinstance(out.flags, Flags)
    assert out.data.dict == {}


# ---------------------------------------------------------------------------
# skip functions
# ---------------------------------------------------------------------------

def test_skip_chars():
    assert skip_chars("  abc", 0, " ") == 2
    assert skip_chars("abc", 0, " ") == 0
    assert skip_chars("   ", 0, " ") == 3


def test_skip_until_basic():
    assert skip_until("abcXdef", 0, "X", error_on=frozenset(), error_on_eof=True) == 3


def test_skip_until_eof_no_error():
    assert skip_until("abc", 0, "X", error_on=frozenset(), error_on_eof=False) == 3


def test_skip_until_eof_error():
    with pytest.raises(TOMLDecodeError):
        skip_until("abc", 0, "X", error_on=frozenset(), error_on_eof=True)


def test_skip_until_error_on():
    with pytest.raises(TOMLDecodeError):
        skip_until("abXc", 0, "Z", error_on=frozenset("X"), error_on_eof=False)


def test_skip_comment():
    assert skip_comment("# hello\nrest", 0) == 7
    assert skip_comment("\n", 0) == 0


def test_skip_comment_eof():
    assert skip_comment("# hello", 0) == 7


def test_skip_comment_illegal_char():
    with pytest.raises(TOMLDecodeError):
        skip_comment("# a\x00b\n", 0)


def test_skip_comments_and_array_ws():
    assert skip_comments_and_array_ws("  \n# c\n  x", 0) == 9


def test_skip_comments_and_array_ws_stop():
    assert skip_comments_and_array_ws("x", 0) == 0


# ---------------------------------------------------------------------------
# create_dict_rule / create_list_rule
# ---------------------------------------------------------------------------

def test_create_dict_rule():
    out = Output()
    pos, key = create_dict_rule("[a.b]\n", 0, out)
    assert key == ("a", "b")
    assert pos == 6
    assert out.data.dict == {"a": {"b": {}}}


def test_create_dict_rule_twice_raises():
    out = Output()
    create_dict_rule("[a]\n", 0, out)
    with pytest.raises(TOMLDecodeError):
        create_dict_rule("[a]\n", 0, out)


def test_create_dict_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError):
        create_dict_rule("[a\n", 0, out)


def test_create_dict_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 5
    with pytest.raises(TOMLDecodeError):
        create_dict_rule("[a]\n", 0, out)


def test_create_list_rule():
    out = Output()
    pos, key = create_list_rule("[[a]]\n", 0, out)
    assert key == ("a",)
    assert pos == 5
    assert out.data.dict == {"a": [{}]}


def test_create_list_rule_twice():
    out = Output()
    create_list_rule("[[a]]\n", 0, out)
    pos, key = create_list_rule("[[a]]\n", 0, out)
    assert out.data.dict == {"a": [{}, {}]}


def test_create_list_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError):
        create_list_rule("[[a]\n", 0, out)


def test_create_list_rule_not_list():
    out = Output()
    out.data.dict["a"] = 5
    with pytest.raises(TOMLDecodeError):
        create_list_rule("[[a]]\n", 0, out)


def test_create_list_rule_frozen():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=False)
    with pytest.raises(TOMLDecodeError):
        create_list_rule("[[a]]\n", 0, out)


# ---------------------------------------------------------------------------
# parse_key / parse_key_part
# ---------------------------------------------------------------------------

def test_parse_key_bare():
    pos, key = parse_key("abc = 1", 0)
    assert key == ("abc",)
    assert pos == 4


def test_parse_key_dotted():
    pos, key = parse_key("a.b.c = 1", 0)
    assert key == ("a", "b", "c")
    assert pos == 6


def test_parse_key_dotted_ws():
    pos, key = parse_key("a . b . c = 1", 0)
    assert key == ("a", "b", "c")
    assert pos == 10


def test_parse_key_quoted():
    pos, key = parse_key('"a b" = 1', 0)
    assert key == ("a b",)
    assert pos == 6


def test_parse_key_part_bare():
    pos, part = parse_key_part("abc-def", 0)
    assert part == "abc-def"
    assert pos == 7


def test_parse_key_part_literal():
    pos, part = parse_key_part("'lit'", 0)
    assert part == "lit"
    assert pos == 5


def test_parse_key_part_basic():
    pos, part = parse_key_part('"basic"', 0)
    assert part == "basic"
    assert pos == 7


def test_parse_key_part_invalid():
    with pytest.raises(TOMLDecodeError):
        parse_key_part("!bad", 0)


def test_parse_key_part_index_error():
    with pytest.raises(TOMLDecodeError):
        parse_key_part("", 0)


def test_parse_key_too_many_parts():
    src = ".".join(["a"] * (sys.getrecursionlimit() + 1))
    with pytest.raises(RecursionError):
        parse_key(src, 0)


# ---------------------------------------------------------------------------
# parse_one_line_basic_str
# ---------------------------------------------------------------------------

def test_parse_one_line_basic_str():
    pos, s = parse_one_line_basic_str('"hello"', 0)
    assert s == "hello"
    assert pos == 7


def test_parse_one_line_basic_str_escapes():
    pos, s = parse_one_line_basic_str('"a\\nb"', 0)
    assert s == "a\nb"


# ---------------------------------------------------------------------------
# parse_array
# ---------------------------------------------------------------------------

def test_parse_array_empty():
    pos, arr = parse_array("[]", 0, float, 0)
    assert arr == []
    assert pos == 2


def test_parse_array_values():
    pos, arr = parse_array("[1, 2, 3]", 0, float, 0)
    assert arr == [1, 2, 3]
    assert pos == 9


def test_parse_array_nested():
    pos, arr = parse_array("[[1], [2, 3]]", 0, float, 0)
    assert arr == [[1], [2, 3]]
    assert pos == 13


def test_parse_array_multiline_with_comments():
    src = "[\n  1, # one\n  2,\n]"
    pos, arr = parse_array(src, 0, float, 0)
    assert arr == [1, 2]
    assert pos == len(src)


def test_parse_array_trailing_comma():
    pos, arr = parse_array("[1, 2,]", 0, float, 0)
    assert arr == [1, 2]
    assert pos == 7


def test_parse_array_unclosed():
    with pytest.raises(TOMLDecodeError):
        parse_array("[1 2]", 0, float, 0)


def test_parse_array_mixed_types():
    pos, arr = parse_array("[1, 'a', true]", 0, float, 0)
    assert arr == [1, "a", True]


# ---------------------------------------------------------------------------
# parse_inline_table
# ---------------------------------------------------------------------------

def test_parse_inline_table_empty():
    pos, tbl = parse_inline_table("{}", 0, float, 0)
    assert tbl == {}
    assert pos == 2


def test_parse_inline_table_basic():
    pos, tbl = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
    assert tbl == {"a": 1, "b": 2}
    assert pos == 14


def test_parse_inline_table_nested():
    pos, tbl = parse_inline_table("{a = {b = 1}}", 0, float, 0)
    assert tbl == {"a": {"b": 1}}


def test_parse_inline_table_duplicate_key():
    with pytest.raises(TOMLDecodeError):
        parse_inline_table("{a = 1, a = 2}", 0, float, 0)


def test_parse_inline_table_unclosed():
    with pytest.raises(TOMLDecodeError):
        parse_inline_table("{a = 1", 0, float, 0)


def test_parse_inline_table_dotted():
    pos, tbl = parse_inline_table("{a.b = 1}", 0, float, 0)
    assert tbl == {"a": {"b": 1}}


def test_parse_inline_table_frozen_nested():
    pos, tbl = parse_inline_table("{a = {b = 1}, c = 2}", 0, float, 0)
    assert tbl == {"a": {"b": 1}, "c": 2}


# ---------------------------------------------------------------------------
# parse_basic_str_escape / multiline / hex
# ---------------------------------------------------------------------------

def test_parse_basic_str_escape_simple():
    pos, s = parse_basic_str_escape("\\n", 0)
    assert s == "\n"
    assert pos == 2


def test_parse_basic_str_escape_tab():
    pos, s = parse_basic_str_escape("\\t", 0)
    assert s == "\t"


def test_parse_basic_str_escape_quote():
    pos, s = parse_basic_str_escape('\\"', 0)
    assert s == '"'


def test_parse_basic_str_escape_backslash():
    pos, s = parse_basic_str_escape("\\\\", 0)
    assert s == "\\"


def test_parse_basic_str_escape_invalid():
    with pytest.raises(TOMLDecodeError):
        parse_basic_str_escape("\\q", 0)


def test_parse_basic_str_escape_hex():
    pos, s = parse_basic_str_escape("\\x41", 0)
    assert s == "A"
    assert pos == 4


def test_parse_basic_str_escape_unicode():
    pos, s = parse_basic_str_escape("\\u00e9", 0)
    assert s == "\u00e9"


def test_parse_basic_str_escape_unicode_long():
    pos, s = parse_basic_str_escape("\\U0001F600", 0)
    assert s == "\U0001F600"


def test_parse_basic_str_escape_multiline_line_continuation():
    pos, s = parse_basic_str_escape_multiline("\\\n   x", 0)
    assert s == ""
    assert pos == 5


def test_parse_basic_str_escape_multiline_line_continuation_ws():
    pos, s = parse_basic_str_escape_multiline("\\   \n  x", 0)
    assert s == ""
    assert pos == 8


def test_parse_basic_str_escape_multiline_eof():
    pos, s = parse_basic_str_escape_multiline("\\ ", 0)
    assert s == ""
    assert pos == 2


def test_parse_basic_str_escape_multiline_invalid():
    with pytest.raises(TOMLDecodeError):
        parse_basic_str_escape_multiline("\\ x", 0)


def test_parse_hex_char_valid():
    pos, s = parse_hex_char("41", 0, 2)
    assert s == "A"
    assert pos == 2


def test_parse_hex_char_invalid_hex():
    with pytest.raises(TOMLDecodeError):
        parse_hex_char("ZZ", 0, 2)


def test_parse_hex_char_too_short():
    with pytest.raises(TOMLDecodeError):
        parse_hex_char("4", 0, 2)


def test_parse_hex_char_surrogate():
    with pytest.raises(TOMLDecodeError):
        parse_hex_char("D800", 0, 4)


def test_parse_hex_char_too_large():
    with pytest.raises(TOMLDecodeError):
        parse_hex_char("110000", 0, 8)


# ---------------------------------------------------------------------------
# parse_literal_str
# ---------------------------------------------------------------------------

def test_parse_literal_str():
    pos, s = parse_literal_str("'hello'", 0)
    assert s == "hello"
    assert pos == 7


def test_parse_literal_str_unterminated():
    with pytest.raises(TOMLDecodeError):
        parse_literal_str("'hello", 0)


def test_parse_literal_str_illegal_char():
    with pytest.raises(TOMLDecodeError):
        parse_literal_str("'a\x00b'", 0)


# ---------------------------------------------------------------------------
# parse_multiline_str
# ---------------------------------------------------------------------------

def test_parse_multiline_str_basic():
    pos, s = parse_multiline_str('"""hello"""', 0, literal=False)
    assert s == "hello"
    assert pos == 11


def test_parse_multiline_str_literal():
    pos, s = parse_multiline_str("'''hello'''", 0, literal=True)
    assert s == "hello"
    assert pos == 11


def test_parse_multiline_str_leading_newline():
    pos, s = parse_multiline_str('"""\nhello"""', 0, literal=False)
    assert s == "hello"


def test_parse_multiline_str_leading_newline_literal():
    pos, s = parse_multiline_str("'''\nhello'''", 0, literal=True)
    assert s == "hello"


def test_parse_multiline_str_extra_quote():
    pos, s = parse_multiline_str('"""hello""""', 0, literal=False)
    assert s == 'hello"'


def test_parse_multiline_str_extra_two_quotes():
    pos, s = parse_multiline_str('"""hello"""""', 0, literal=False)
    assert s == 'hello""'


def test_parse_multiline_str_literal_extra_quote():
    pos, s = parse_multiline_str("'''hello''''", 0, literal=True)
    assert s == "hello'"


def test_parse_multiline_str_unterminated():
    with pytest.raises(TOMLDecodeError):
        parse_multiline_str('"""hello', 0, literal=False)


def test_parse_multiline_str_unterminated_literal():
    with pytest.raises(TOMLDecodeError):
        parse_multiline_str("'''hello", 0, literal=True)


# ---------------------------------------------------------------------------
# parse_basic_str
# ---------------------------------------------------------------------------

def test_parse_basic_str_simple():
    pos, s = parse_basic_str('"hello"', 0, multiline=False)
    assert s == "hello"
    assert pos == 7


def test_parse_basic_str_unterminated():
    with pytest.raises(TOMLDecodeError):
        parse_basic_str('"hello', 0, multiline=False)


def test_parse_basic_str_illegal_char():
    with pytest.raises(TOMLDecodeError):
        parse_basic_str('"a\x00b"', 0, multiline=False)


def test_parse_basic_str_multiline():
    pos, s = parse_basic_str('"hello"', 0, multiline=True)
    assert s == "hello"
    assert pos == 7


def test_parse_basic_str_multiline_escapes():
    pos, s = parse_basic_str('"a\\nb"', 0, multiline=True)
    assert s == "a\nb"


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------

def test_parse_value_string():
    pos, v = parse_value('"hi"', 0, float, 0)
    assert v == "hi"
    assert pos == 4


def test_parse_value_multiline_string():
    pos, v = parse_value('"""hi"""', 0, float, 0)
    assert v == "hi"


def test_parse_value_literal_string():
    pos, v = parse_value("'hi'", 0, float, 0)
    assert v == "hi"


def test_parse_value_multiline_literal():
    pos, v = parse_value("'''hi'''", 0, float, 0)
    assert v == "hi"


def test_parse_value_true():
    pos, v = parse_value("true", 0, float, 0)
    assert v is True
    assert pos == 4


def test_parse_value_false():
    pos, v = parse_value("false", 0, float, 0)
    assert v is False
    assert pos == 5


def test_parse_value_integer():
    pos, v = parse_value("42", 0, float, 0)
    assert v == 42
    assert pos == 2


def test_parse_value_negative_integer():
    pos, v = parse_value("-42", 0, float, 0)
    assert v == -42


def test_parse_value_float():
    pos, v = parse_value("3.14", 0, float, 0)
    assert v == 3.14
    assert pos == 4


def test_parse_value_inf():
    pos, v = parse_value("inf", 0, float, 0)
    assert v == math.inf


def test_parse_value_nan():
    pos, v = parse_value("nan", 0, float, 0)
    assert math.isnan(v)


def test_parse_value_posinf():
    pos, v = parse_value("+inf", 0, float, 0)
    assert v == math.inf


def test_parse_value_neginf():
    pos, v = parse_value("-inf", 0, float, 0)
    assert v == -math.inf


def test_parse_value_posnan():
    pos, v = parse_value("+nan", 0, float, 0)
    assert math.isnan(v)


def test_parse_value_negNan():
    pos, v = parse_value("-nan", 0, float, 0)
    assert math.isnan(v)


def test_parse_value_array():
    pos, v = parse_value("[1, 2]", 0, float, 0)
    assert v == [1, 2]


def test_parse_value_inline_table():
    pos, v = parse_value("{a = 1}", 0, float, 0)
    assert v == {"a": 1}


def test_parse_value_date():
    pos, v = parse_value("2021-01-01", 0, float, 0)
    assert v == date(2021, 1, 1)


def test_parse_value_datetime():
    pos, v = parse_value("2021-01-01T12:00:00Z", 0, float, 0)
    assert v == datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_value_localtime():
    pos, v = parse_value("12:00:00", 0, float, 0)
    assert v == time(12, 0, 0)


def test_parse_value_invalid_date():
    with pytest.raises(TOMLDecodeError):
        parse_value("2021-13-01", 0, float, 0)


def test_parse_value_invalid():
    with pytest.raises(TOMLDecodeError):
        parse_value("@", 0, float, 0)


def test_parse_value_too_nested():
    with pytest.raises(RecursionError):
        parse_value("[", 0, float, MAX_INLINE_NESTING + 1)


# ---------------------------------------------------------------------------
# parse_key_value_pair / key_value_rule
# ---------------------------------------------------------------------------

def test_parse_key_value_pair():
    pos, key, value = parse_key_value_pair("a = 1", 0, float, 0)
    assert key == ("a",)
    assert value == 1
    assert pos == 5


def test_parse_key_value_pair_missing_eq():
    with pytest.raises(TOMLDecodeError):
        parse_key_value_pair("a 1", 0, float, 0)


def test_parse_key_value_pair_eof():
    with pytest.raises(TOMLDecodeError):
        parse_key_value_pair("a", 0, float, 0)


def test_key_value_rule_basic():
    out = Output()
    pos = key_value_rule("a = 1", 0, out, (), float)
    assert out.data.dict == {"a": 1}
    assert pos == 5


def test_key_value_rule_dotted():
    out = Output()
    pos = key_value_rule("a.b = 1", 0, out, (), float)
    assert out.data.dict == {"a": {"b": 1}}


def test_key_value_rule_duplicate():
    out = Output()
    key_value_rule("a = 1", 0, out, (), float)
    with pytest.raises(TOMLDecodeError):
        key_value_rule("a = 2", 0, out, (), float)


def test_key_value_rule_frozen_parent():
    out = Output()
    key_value_rule("a = {b = 1}", 0, out, (), float)
    with pytest.raises(TOMLDecodeError):
        key_value_rule("c = 2", 0, out, ("a",), float)


def test_key_value_rule_inline_table_freezes():
    out = Output()
    key_value_rule("a = {b = 1}", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN)
    assert out.flags.is_(("a", "b"), Flags.FROZEN)


def test_key_value_rule_inline_array_freezes():
    out = Output()
    key_value_rule("a = [1, 2]", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN)


def test_key_value_rule_with_header():
    out = Output()
    key_value_rule("a = 1", 0, out, ("t",), float)
    assert out.data.dict == {"t": {"a": 1}}


def test_key_value_rule_redefine_namespace():
    out = Output()
    out.flags.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
    with pytest.raises(TOMLDecodeError):
        key_value_rule("a.b = 1", 0, out, (), float)


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cp,expected", [
    (0, True),
    (0x41, True),
    (0xD7FF, True),
    (0xD800, False),
    (0xDFFF, False),
    (0xE000, True),
    (0x10FFFF, True),
    (0x110000, False),
    (-1, False),
])
def test_is_unicode_scalar_value(cp, expected):
    assert is_unicode_scalar_value(cp) is expected


# ---------------------------------------------------------------------------
# make_safe_parse_float
# ---------------------------------------------------------------------------

def test_make_safe_parse_float_identity():
    assert make_safe_parse_float(float) is float


def test_make_safe_parse_float_custom():
    from decimal import Decimal
    f = make_safe_parse_float(Decimal)
    assert f("1.5") == Decimal("1.5")


def test_make_safe_parse_float_rejects_dict():
    def bad(s):
        return {}
    f = make_safe_parse_float(bad)
    with pytest.raises(ValueError):
        f("1.5")


def test_make_safe_parse_float_rejects_list():
    def bad(s):
        return []
    f = make_safe_parse_float(bad)
    with pytest.raises(ValueError):
        f("1.5")


def test_make_safe_parse_float_allows_other():
    def custom(s):
        return "X" + s
    f = make_safe_parse_float(custom)
    assert f("1.5") == "X1.5"
