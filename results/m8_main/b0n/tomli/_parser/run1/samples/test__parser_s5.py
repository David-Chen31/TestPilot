import io
import math
import sys
import warnings
from datetime import date, datetime, time, timezone

import pytest

from tomli._parser import (
    BASIC_STR_ESCAPE_REPLACEMENTS,
    DEPRECATED_DEFAULT,
    MAX_INLINE_NESTING,
    MAX_KEY_PARTS,
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
# Constants / sentinel
# ---------------------------------------------------------------------------


def test_deprecated_default_is_class():
    assert isinstance(DEPRECATED_DEFAULT, type)


def test_max_inline_nesting_value():
    assert MAX_INLINE_NESTING == 400


def test_max_key_parts_default():
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
# TOMLDecodeError
# ---------------------------------------------------------------------------


def test_decode_error_basic():
    err = TOMLDecodeError("Bad", "abc", 1)
    assert isinstance(err, ValueError)
    assert err.msg == "Bad"
    assert err.doc == "abc"
    assert err.pos == 1
    assert err.lineno == 1
    assert err.colno == 2
    assert str(err) == "Bad (at line 1, column 2)"


def test_decode_error_multiline():
    err = TOMLDecodeError("Bad", "a\nbc", 3)
    assert err.lineno == 2
    assert err.colno == 2
    assert str(err) == "Bad (at line 2, column 2)"


def test_decode_error_second_line_start():
    err = TOMLDecodeError("Bad", "ab\ncd", 3)
    assert err.lineno == 2
    assert err.colno == 1


def test_decode_error_end_of_doc():
    err = TOMLDecodeError("Bad", "abc", 3)
    assert str(err) == "Bad (at end of document)"
    assert err.pos == 3


def test_decode_error_pos_past_doc():
    err = TOMLDecodeError("Bad", "abc", 10)
    assert str(err) == "Bad (at end of document)"
    assert err.lineno == 1
    assert err.colno == 11


def test_decode_error_deprecated_free_form():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("some message")
    assert isinstance(err, TOMLDecodeError)
    assert isinstance(err, ValueError)
    assert str(err) == "some message"


def test_decode_error_deprecated_no_args():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError()
    assert isinstance(err, TOMLDecodeError)
    assert str(err) == ""


def test_decode_error_deprecated_extra_args():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("msg", "doc", 1, "extra")
    assert isinstance(err, TOMLDecodeError)


def test_decode_error_deprecated_bad_types():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError(123, "doc", 1)
    assert isinstance(err, TOMLDecodeError)


def test_decode_error_deprecated_joins_in_order():
    with pytest.warns(DeprecationWarning):
        err = TOMLDecodeError("a", "b", 3)
    assert "a" in str(err)
    assert "b" in str(err)


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------


def test_load_basic():
    fp = io.BytesIO(b"x = 1\n")
    assert load(fp) == {"x": 1}


def test_load_parse_float():
    fp = io.BytesIO(b"x = 1.5\n")
    result = load(fp, parse_float=lambda s: ("F", s))
    assert result == {"x": ("F", "1.5")}


def test_load_requires_binary():
    with pytest.raises(TypeError) as excinfo:
        load(io.StringIO("x = 1\n"))
    msg = str(excinfo.value)
    assert "binary mode" in msg


def test_loads_empty():
    assert loads("") == {}


def test_loads_simple():
    assert loads("a = 1\nb = 'x'\n") == {"a": 1, "b": "x"}


def test_loads_crlf_normalized():
    assert loads("a = 1\r\nb = 2\r\n") == {"a": 1, "b": 2}


def test_loads_crlf_in_basic_string():
    assert loads('a = "x\r\ny"\n') == {"a": "x\ny"}


def test_loads_non_str():
    with pytest.raises(TypeError) as excinfo:
        loads(b"a = 1")
    assert "Expected str object" in str(excinfo.value)


def test_loads_none():
    with pytest.raises(TypeError):
        loads(None)


def test_loads_parse_float():
    result = loads("x = 3.14\n", parse_float=lambda s: ("F", s))
    assert result == {"x": ("F", "3.14")}


def test_loads_table_and_array_of_tables():
    src = "[a]\nx = 1\n[[b]]\ny = 2\n[[b]]\ny = 3\n"
    result = loads(src)
    assert result == {"a": {"x": 1}, "b": [{"y": 2}, {"y": 3}]}


def test_loads_comments_and_blank_lines():
    src = "# comment\n\na = 1 # trailing\n\n"
    assert loads(src) == {"a": 1}


def test_loads_invalid_statement():
    with pytest.raises(TOMLDecodeError) as excinfo:
        loads("= 1\n")
    assert "Invalid statement" in str(excinfo.value)


def test_loads_after_statement_garbage():
    with pytest.raises(TOMLDecodeError) as excinfo:
        loads("a = 1 b\n")
    assert "Expected newline" in str(excinfo.value)


def test_loads_leading_ws():
    assert loads("   a = 1\n") == {"a": 1}


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_flags_empty_key():
    f = Flags()
    assert f.is_((), Flags.FROZEN) is False


def test_flags_set_and_is():
    f = Flags()
    f.set(("a",), Flags.FROZEN, recursive=False)
    assert f.is_(("a",), Flags.FROZEN) is True
    assert f.is_(("b",), Flags.FROZEN) is False


def test_flags_set_nested():
    f = Flags()
    f.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
    assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is True
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False


def test_flags_recursive():
    f = Flags()
    f.set(("a",), Flags.FROZEN, recursive=True)
    assert f.is_(("a",), Flags.FROZEN) is True
    assert f.is_(("a", "b"), Flags.FROZEN) is True
    assert f.is_(("a", "b", "c"), Flags.FROZEN) is True


def test_flags_recursive_nonrecursive():
    f = Flags()
    f.set(("a",), Flags.FROZEN, recursive=False)
    assert f.is_(("a", "b"), Flags.FROZEN) is False


def test_flags_pending():
    f = Flags()
    f.add_pending(("a",), Flags.EXPLICIT_NEST)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
    f.finalize_pending()
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
    # pending cleared
    assert f._pending_flags == set()


def test_flags_unset_all():
    f = Flags()
    f.set(("a", "b"), Flags.FROZEN, recursive=False)
    f.unset_all(("a", "b"))
    assert f.is_(("a", "b"), Flags.FROZEN) is False


def test_flags_unset_all_missing():
    f = Flags()
    f.unset_all(("x", "y"))  # no error


def test_flags_unset_all_deep_missing():
    f = Flags()
    f.set(("a", "b"), Flags.FROZEN, recursive=False)
    f.unset_all(("a", "b", "c"))  # returns early


def test_flags_constants():
    assert Flags.FROZEN == 0
    assert Flags.EXPLICIT_NEST == 1


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------


def test_nesteddict_get_or_create():
    nd = NestedDict()
    nest = nd.get_or_create_nest(("a", "b"))
    assert nest == {}
    assert nd.dict == {"a": {"b": {}}}


def test_nesteddict_get_or_create_empty():
    nd = NestedDict()
    assert nd.get_or_create_nest(()) is nd.dict


def test_nesteddict_get_or_create_existing():
    nd = NestedDict()
    nd.dict["a"] = {"b": 1}
    nest = nd.get_or_create_nest(("a",))
    assert nest == {"b": 1}


def test_nesteddict_access_lists():
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    nest = nd.get_or_create_nest(("a",))
    assert nest == {}
    assert nd.dict == {"a": [{}]}


def test_nesteddict_no_access_lists():
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    with pytest.raises(KeyError):
        nd.get_or_create_nest(("a",), access_lists=False)


def test_nesteddict_key_error_non_dict():
    nd = NestedDict()
    nd.dict["a"] = 5
    with pytest.raises(KeyError):
        nd.get_or_create_nest(("a", "b"))


def test_nesteddict_append_nest_to_list():
    nd = NestedDict()
    nd.append_nest_to_list(("a",))
    nd.append_nest_to_list(("a",))
    assert nd.dict == {"a": [{}, {}]}


def test_nesteddict_append_non_list():
    nd = NestedDict()
    nd.dict["a"] = 5
    with pytest.raises(KeyError):
        nd.append_nest_to_list(("a",))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_output_defaults():
    out = Output()
    assert isinstance(out.data, NestedDict)
    assert isinstance(out.flags, Flags)
    assert out.data.dict == {}


# ---------------------------------------------------------------------------
# skip_chars
# ---------------------------------------------------------------------------


def test_skip_chars_basic():
    assert skip_chars("   abc", 0, frozenset(" ")) == 3


def test_skip_chars_to_eof():
    assert skip_chars("   ", 0, frozenset(" ")) == 3


def test_skip_chars_no_match():
    assert skip_chars("abc", 0, frozenset(" ")) == 0


def test_skip_chars_start_past_end():
    assert skip_chars("abc", 10, frozenset(" ")) == 10


# ---------------------------------------------------------------------------
# skip_until
# ---------------------------------------------------------------------------


def test_skip_until_basic():
    assert (
        skip_until("abc'd", 0, "'", error_on=frozenset(), error_on_eof=True) == 3
    )


def test_skip_until_eof_allowed():
    assert (
        skip_until("abc", 0, "'", error_on=frozenset(), error_on_eof=False) == 3
    )


def test_skip_until_eof_error():
    with pytest.raises(TOMLDecodeError) as excinfo:
        skip_until("abc", 0, "'", error_on=frozenset(), error_on_eof=True)
    assert "Expected" in str(excinfo.value)


def test_skip_until_illegal_char():
    with pytest.raises(TOMLDecodeError) as excinfo:
        skip_until("a\x01b'", 0, "'", error_on=frozenset("\x01"), error_on_eof=True)
    assert "invalid character" in str(excinfo.value)


# ---------------------------------------------------------------------------
# skip_comment
# ---------------------------------------------------------------------------


def test_skip_comment_none():
    assert skip_comment("abc", 0) == 0


def test_skip_comment_eof():
    assert skip_comment("", 0) == 0


def test_skip_comment_basic():
    assert skip_comment("# hi\nx", 0) == 4


def test_skip_comment_eof_no_newline():
    assert skip_comment("# hi", 0) == 4


def test_skip_comment_illegal_char():
    with pytest.raises(TOMLDecodeError):
        skip_comment("# a\x01b\n", 0)


# ---------------------------------------------------------------------------
# skip_comments_and_array_ws
# ---------------------------------------------------------------------------


def test_skip_comments_and_array_ws_ws():
    assert skip_comments_and_array_ws("   x", 0) == 3


def test_skip_comments_and_array_ws_newlines():
    assert skip_comments_and_array_ws("\n\n  x", 0) == 4


def test_skip_comments_and_array_ws_comment():
    assert skip_comments_and_array_ws("# c\nx", 0) == 4


def test_skip_comments_and_array_ws_combined():
    assert skip_comments_and_array_ws("  # c\n\n  x", 0) == 8


def test_skip_comments_and_array_ws_stop():
    assert skip_comments_and_array_ws("x", 0) == 0


# ---------------------------------------------------------------------------
# create_dict_rule
# ---------------------------------------------------------------------------


def test_create_dict_rule_basic():
    out = Output()
    pos, key = create_dict_rule("[a.b]\n", 0, out)
    assert key == ("a", "b")
    assert pos == 5
    assert out.data.dict == {"a": {"b": {}}}
    assert out.flags.is_(("a", "b"), Flags.EXPLICIT_NEST) is True


def test_create_dict_rule_spaces():
    out = Output()
    pos, key = create_dict_rule("[ a ]", 0, out)
    assert key == ("a",)
    assert pos == 5


def test_create_dict_rule_duplicate():
    out = Output()
    create_dict_rule("[a]", 0, out)
    with pytest.raises(TOMLDecodeError) as excinfo:
        create_dict_rule("[a]", 0, out)
    assert "Cannot declare" in str(excinfo.value)


def test_create_dict_rule_frozen():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError):
        create_dict_rule("[a]", 0, out)


def test_create_dict_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError) as excinfo:
        create_dict_rule("[a]", 0, out)
    assert "Cannot overwrite" in str(excinfo.value)


def test_create_dict_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError) as excinfo:
        create_dict_rule("[a\n", 0, out)
    assert "Expected ']'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# create_list_rule
# ---------------------------------------------------------------------------


def test_create_list_rule_basic():
    out = Output()
    pos, key = create_list_rule("[[a]]\n", 0, out)
    assert key == ("a",)
    assert pos == 5
    assert out.data.dict == {"a": [{}]}
    assert out.flags.is_(("a",), Flags.EXPLICIT_NEST) is True


def test_create_list_rule_twice():
    out = Output()
    create_list_rule("[[a]]", 0, out)
    pos, key = create_list_rule("[[a]]", 0, out)
    assert out.data.dict == {"a": [{}, {}]}


def test_create_list_rule_frozen():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError) as excinfo:
        create_list_rule("[[a]]", 0, out)
    assert "immutable" in str(excinfo.value)


def test_create_list_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 5
    with pytest.raises(TOMLDecodeError) as excinfo:
        create_list_rule("[[a]]", 0, out)
    assert "Cannot overwrite" in str(excinfo.value)


def test_create_list_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError) as excinfo:
        create_list_rule("[[a]\n", 0, out)
    assert "Expected ']]'" in str(excinfo.value)


def test_create_list_rule_unset_all():
    out = Output()
    create_dict_rule("[a.b]", 0, out)
    create_list_rule("[[a.b.c]]", 0, out)
    # flags for a.b.c reset before set
    assert out.flags.is_(("a", "b", "c"), Flags.EXPLICIT_NEST) is True


# ---------------------------------------------------------------------------
# key_value_rule
# ---------------------------------------------------------------------------


def test_key_value_rule_basic():
    out = Output()
    pos = key_value_rule("a = 1", 0, out, (), float)
    assert pos == 5
    assert out.data.dict == {"a": 1}


def test_key_value_rule_with_header():
    out = Output()
    key_value_rule("x = 1", 0, out, ("t",), float)
    assert out.data.dict == {"t": {"x": 1}}


def test_key_value_rule_dotted():
    out = Output()
    key_value_rule("a.b = 1", 0, out, (), float)
    assert out.data.dict == {"a": {"b": 1}}
    assert out.flags.is_(("a",), Flags.EXPLICIT_NEST) is True


def test_key_value_rule_redefine_table():
    out = Output()
    create_dict_rule("[a]", 0, out)
    with pytest.raises(TOMLDecodeError) as excinfo:
        key_value_rule("a.b = 1", 0, out, (), float)
    assert "Cannot redefine namespace" in str(excinfo.value)


def test_key_value_rule_frozen_parent():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=True)
    with pytest.raises(TOMLDecodeError) as excinfo:
        key_value_rule("a.b = 1", 0, out, (), float)
    assert "immutable" in str(excinfo.value)


def test_key_value_rule_overwrite_non_dict():
    out = Output()
    out.data.dict["a"] = 5
    with pytest.raises(TOMLDecodeError) as excinfo:
        key_value_rule("a.b = 1", 0, out, (), float)
    assert "Cannot overwrite" in str(excinfo.value)


def test_key_value_rule_duplicate_key():
    out = Output()
    key_value_rule("a = 1", 0, out, (), float)
    with pytest.raises(TOMLDecodeError) as excinfo:
        key_value_rule("a = 2", 0, out, (), float)
    assert "Cannot overwrite" in str(excinfo.value)


def test_key_value_rule_inline_table_frozen():
    out = Output()
    key_value_rule("a = {b = 1}", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True
    assert out.flags.is_(("a", "b"), Flags.FROZEN) is True


def test_key_value_rule_inline_array_frozen():
    out = Output()
    key_value_rule("a = [1]", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True


# ---------------------------------------------------------------------------
# parse_key_value_pair
# ---------------------------------------------------------------------------


def test_parse_key_value_pair():
    pos, key, value = parse_key_value_pair("a = 1", 0, float, 0)
    assert pos == 5
    assert key == ("a",)
    assert value == 1


def test_parse_key_value_pair_dotted():
    pos, key, value = parse_key_value_pair(" a.b = 2", 1, float, 0)
    assert key == ("a", "b")
    assert value == 2


def test_parse_key_value_pair_missing_eq():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_key_value_pair("a 1", 0, float, 0)
    assert "Expected '='" in str(excinfo.value)


def test_parse_key_value_pair_eof_after_key():
    with pytest.raises(TOMLDecodeError):
        parse_key_value_pair("a", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_key
# ---------------------------------------------------------------------------


def test_parse_key_simple():
    assert parse_key("abc", 0) == (3, ("abc",))


def test_parse_key_dotted():
    pos, key = parse_key("a.b.c", 0)
    assert pos == 5
    assert key == ("a", "b", "c")


def test_parse_key_whitespace():
    pos, key = parse_key(" a . b ", 0)
    assert key == ("a", "b")
    assert pos == 8


def test_parse_key_quoted():
    pos, key = parse_key('"a b".c', 0)
    assert key == ("a b", "c")


def test_parse_key_too_many_parts():
    src = ".".join(["a"] * (MAX_KEY_PARTS + 1))
    with pytest.raises(RecursionError):
        parse_key(src, 0)


def test_parse_key_empty_fails():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_key("", 0)
    assert "Invalid initial character" in str(excinfo.value)


# ---------------------------------------------------------------------------
# parse_key_part
# ---------------------------------------------------------------------------


def test_parse_key_part_bare():
    assert parse_key_part("abc def", 0) == (3, "abc")


def test_parse_key_part_bare_with_dash():
    assert parse_key_part("a-b_c1", 0) == (6, "a-b_c1")


def test_parse_key_part_literal():
    assert parse_key_part("'x y'", 0) == (5, "x y")


def test_parse_key_part_basic():
    assert parse_key_part('"x y"', 0) == (5, "x y")


def test_parse_key_part_invalid():
    with pytest.raises(TOMLDecodeError):
        parse_key_part("!x", 0)


def test_parse_key_part_eof():
    with pytest.raises(TOMLDecodeError):
        parse_key_part("", 0)


# ---------------------------------------------------------------------------
# parse_one_line_basic_str
# ---------------------------------------------------------------------------


def test_parse_one_line_basic_str():
    pos, s = parse_one_line_basic_str('"hi"', 0)
    assert pos == 4
    assert s == "hi"


def test_parse_one_line_basic_str_escape():
    pos, s = parse_one_line_basic_str('"a\\tb"', 0)
    assert s == "a\tb"


def test_parse_one_line_basic_str_unterminated():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_one_line_basic_str('"abc', 0)
    assert "Unterminated" in str(excinfo.value)


# ---------------------------------------------------------------------------
# parse_array
# ---------------------------------------------------------------------------


def test_parse_array_empty():
    pos, arr = parse_array("[]", 0, float, 0)
    assert pos == 2
    assert arr == []


def test_parse_array_basic():
    pos, arr = parse_array("[1, 2, 3]", 0, float, 0)
    assert pos == 9
    assert arr == [1, 2, 3]


def test_parse_array_trailing_comma():
    pos, arr = parse_array("[1, 2,]", 0, float, 0)
    assert arr == [1, 2]


def test_parse_array_whitespace_newlines():
    pos, arr = parse_array("[\n 1,\n 2,\n]", 0, float, 0)
    assert arr == [1, 2]


def test_parse_array_comments():
    pos, arr = parse_array("[1, # c\n2]", 0, float, 0)
    assert arr == [1, 2]


def test_parse_array_nested():
    pos, arr = parse_array("[[1], [2, 3]]", 0, float, 0)
    assert arr == [[1], [2, 3]]


def test_parse_array_unclosed():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_array("[1 2", 0, float, 0)
    assert "Unclosed array" in str(excinfo.value)


# ---------------------------------------------------------------------------
# parse_inline_table
# ---------------------------------------------------------------------------


def test_parse_inline_table_empty():
    pos, d = parse_inline_table("{}", 0, float, 0)
    assert pos == 2
    assert d == {}


def test_parse_inline_table_basic():
    pos, d = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
    assert pos == 14
    assert d == {"a": 1, "b": 2}


def test_parse_inline_table_trailing_comma():
    pos, d = parse_inline_table("{a = 1,}", 0, float, 0)
    assert d == {"a": 1}


def test_parse_inline_table_dotted():
    pos, d = parse_inline_table("{a.b = 1}", 0, float, 0)
    assert d == {"a": {"b": 1}}


def test_parse_inline_table_nested():
    pos, d = parse_inline_table("{a = {b = 1}}", 0, float, 0)
    assert d == {"a": {"b": 1}}


def test_parse_inline_table_duplicate_key():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_inline_table("{a = 1, a = 2}", 0, float, 0)
    assert "Duplicate" in str(excinfo.value)


def test_parse_inline_table_frozen_check():
    with pytest.raises(TOMLDecodeError):
        parse_inline_table("{a = {}, a.b = 1}", 0, float, 0)


def test_parse_inline_table_overwrite_value():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_inline_table("{a = 1, a.b = 2}", 0, float, 0)
    assert "Cannot overwrite" in str(excinfo.value)


def test_parse_inline_table_unclosed():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_inline_table("{a = 1", 0, float, 0)
    assert "Unclosed inline table" in str(excinfo.value)


# ---------------------------------------------------------------------------
# parse_basic_str_escape
# ---------------------------------------------------------------------------


def test_parse_basic_str_escape_known():
    pos, s = parse_basic_str_escape("\\n", 0)
    assert pos == 2
    assert s == "\n"


def test_parse_basic_str_escape_unknown():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_basic_str_escape("\\q", 0)
    assert "Unescaped" in str(excinfo.value)


def test_parse_basic_str_escape_hex():
    pos, s = parse_basic_str_escape("\\x41", 0)
    assert s == "A"
    assert pos == 4


def test_parse_basic_str_escape_unicode():
    pos, s = parse_basic_str_escape("\\u0041", 0)
    assert s == "A"
    assert pos == 6


def test_parse_basic_str_escape_unicode_big():
    pos, s = parse_basic_str_escape("\\U00000041", 0)
    assert s == "A"
    assert pos == 10


def test_parse_basic_str_escape_multiline_space():
    pos, s = parse_basic_str_escape("\\   \n  x", 0, multiline=True)
    assert s == ""
    assert pos == 4


def test_parse_basic_str_escape_multiline_newline():
    pos, s = parse_basic_str_escape("\\\n   x", 0, multiline=True)
    assert s == ""
    assert pos == 5


def test_parse_basic_str_escape_multiline_backslash_space_no_newline():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_basic_str_escape("\\  x", 0, multiline=True)
    assert "Unescaped" in str(excinfo.value)


def test_parse_basic_str_escape_multiline_backslash_space_eof():
    pos, s = parse_basic_str_escape("\\  ", 0, multiline=True)
    assert s == ""
    assert pos == 3


def test_parse_basic_str_escape_multiline_tab():
    pos, s = parse_basic_str_escape("\\\t\nx", 0, multiline=True)
    assert s == ""
    assert pos == 3


def test_parse_basic_str_escape_multiline_not_active_for_single_line():
    # "\ " is only special in multiline
    with pytest.raises(TOMLDecodeError):
        parse_basic_str_escape("\\ x", 0, multiline=False)


# ---------------------------------------------------------------------------
# parse_basic_str_escape_multiline
# ---------------------------------------------------------------------------


def test_parse_basic_str_escape_multiline_wrapper():
    pos, s = parse_basic_str_escape_multiline("\\\n  x", 0)
    assert s == ""
    assert pos == 4


def test_parse_basic_str_escape_multiline_wrapper_known():
    pos, s = parse_basic_str_escape_multiline("\\n", 0)
    assert s == "\n"


# ---------------------------------------------------------------------------
# parse_hex_char
# ---------------------------------------------------------------------------


def test_parse_hex_char_basic():
    pos, s = parse_hex_char("41", 0, 2)
    assert pos == 2
    assert s == "A"


def test_parse_hex_char_too_short():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_hex_char("4", 0, 2)
    assert "Invalid hex" in str(excinfo.value)


def test_parse_hex_char_invalid():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_hex_char("4G", 0, 2)
    assert "Invalid hex" in str(excinfo.value)


def test_parse_hex_char_surrogate():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_hex_char("D800", 0, 4)
    assert "Unicode scalar" in str(excinfo.value)


def test_parse_hex_char_too_large():
    with pytest.raises(TOMLDecodeError):
        parse_hex_char("110000", 0, 8)


# ---------------------------------------------------------------------------
# parse_literal_str
# ---------------------------------------------------------------------------


def test_parse_literal_str():
    pos, s = parse_literal_str("'hi'", 0)
    assert pos == 4
    assert s == "hi"


def test_parse_literal_str_empty():
    pos, s = parse_literal_str("''", 0)
    assert s == ""
    assert pos == 2


def test_parse_literal_str_illegal_char():
    with pytest.raises(TOMLDecodeError):
        parse_literal_str("'a\x01b'", 0)


def test_parse_literal_str_unterminated():
    with pytest.raises(TOMLDecodeError):
        parse_literal_str("'abc", 0)


# ---------------------------------------------------------------------------
# parse_multiline_str
# ---------------------------------------------------------------------------


def test_parse_multiline_literal():
    src = "'''hi'''"
    pos, s = parse_multiline_str(src, 0, literal=True)
    assert pos == 9
    assert s == "hi"


def test_parse_multiline_literal_leading_newline():
    src = "'''\nhi'''"
    pos, s = parse_multiline_str(src, 0, literal=True)
    assert s == "hi"
    assert pos == 10


def test_parse_multiline_literal_extra_quote():
    src = "''''hi''''"
    pos, s = parse_multiline_str(src, 0, literal=True)
    assert s == "hi'"


def test_parse_multiline_literal_extra_two_quotes():
    src = "'''''hi'''''"
    pos, s = parse_multiline_str(src, 0, literal=True)
    assert s == "hi''"


def test_parse_multiline_basic():
    src = '"""hi"""'
    pos, s = parse_multiline_str(src, 0, literal=False)
    assert pos == 9
    assert s == "hi"


def test_parse_multiline_basic_extra_quote():
    src = '""""hi""""'
    pos, s = parse_multiline_str(src, 0, literal=False)
    assert s == "hi'"


# ---------------------------------------------------------------------------
# parse_basic_str
# ---------------------------------------------------------------------------


def test_parse_basic_str_one_line():
    pos, s = parse_basic_str("abc", 0, multiline=False)
    assert pos == 3
    assert s == "abc"


def test_parse_basic_str_escape():
    pos, s = parse_basic_str("a\\tb", 0, multiline=False)
    assert s == "a\tb"


def test_parse_basic_str_unterminated():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_basic_str("abc", 0, multiline=False)
    assert "Unterminated" in str(excinfo.value)


def test_parse_basic_str_illegal_char():
    with pytest.raises(TOMLDecodeError):
        parse_basic_str("a\x01\"", 0, multiline=False)


def test_parse_basic_str_multiline_quote():
    pos, s = parse_basic_str('a"b"""', 0, multiline=True)
    assert s == 'a"b'


def test_parse_basic_str_multiline_newline_ok():
    pos, s = parse_basic_str('a\nb"""', 0, multiline=True)
    assert s == 'a\nb'


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------


def test_parse_value_int():
    assert parse_value("42", 0, float, 0) == (2, 42)


def test_parse_value_negative_int():
    assert parse_value("-7", 0, float, 0) == (2, -7)


def test_parse_value_float():
    pos, v = parse_value("1.5", 0, float, 0)
    assert pos == 3
    assert v == 1.5


def test_parse_value_true():
    assert parse_value("true", 0, float, 0) == (4, True)


def test_parse_value_false():
    assert parse_value("false", 0, float, 0) == (5, False)


def test_parse_value_string():
    pos, v = parse_value('"hi"', 0, float, 0)
    assert v == "hi"


def test_parse_value_literal_string():
    pos, v = parse_value("'hi'", 0, float, 0)
    assert v == "hi"


def test_parse_value_multiline_string():
    pos, v = parse_value('"""hi"""', 0, float, 0)
    assert v == "hi"


def test_parse_value_multiline_literal():
    pos, v = parse_value("'''hi'''", 0, float, 0)
    assert v == "hi"


def test_parse_value_array():
    pos, v = parse_value("[1, 2]", 0, float, 0)
    assert v == [1, 2]


def test_parse_value_inline_table():
    pos, v = parse_value("{a = 1}", 0, float, 0)
    assert v == {"a": 1}


def test_parse_value_date():
    pos, v = parse_value("2021-01-01", 0, float, 0)
    assert v == date(2021, 1, 1)
    assert pos == 10


def test_parse_value_datetime():
    pos, v = parse_value("2021-01-01T00:00:00Z", 0, float, 0)
    assert v == datetime(2021, 1, 1, tzinfo=timezone.utc)


def test_parse_value_localtime():
    pos, v = parse_value("12:34:56", 0, float, 0)
    assert v == time(12, 34, 56)


def test_parse_value_invalid_date():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_value("2021-02-30", 0, float, 0)
    assert "Invalid date" in str(excinfo.value)


def test_parse_value_inf():
    pos, v = parse_value("inf", 0, float, 0)
    assert math.isinf(v)
    assert v > 0


def test_parse_value_nan():
    pos, v = parse_value("nan", 0, float, 0)
    assert math.isnan(v)


def test_parse_value_neg_inf():
    pos, v = parse_value("-inf", 0, float, 0)
    assert math.isinf(v)
    assert v < 0


def test_parse_value_plus_nan():
    pos, v = parse_value("+nan", 0, float, 0)
    assert math.isnan(v)


def test_parse_value_hex_int():
    assert parse_value("0x10", 0, float, 0) == (4, 16)


def test_parse_value_invalid():
    with pytest.raises(TOMLDecodeError) as excinfo:
        parse_value("?", 0, float, 0)
    assert "Invalid value" in str(excinfo.value)


def test_parse_value_eof():
    with pytest.raises(TOMLDecodeError):
        parse_value("", 0, float, 0)


def test_parse_value_custom_float():
    pos, v = parse_value("1.5", 0, lambda s: ("F", s), 0)
    assert v == ("F", "1.5")


def test_parse_value_nesting_too_deep():
    with pytest.raises(RecursionError):
        parse_value("[", 0, float, MAX_INLINE_NESTING + 1)


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cp,expected",
    [
        (0, True),
        (0x41, True),
        (55295, True),
        (55296, False),
        (57343, False),
        (57344, True),
        (1114111, True),
        (1114112, False),
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


def test_make_safe_parse_float_wraps():
    def custom(s):
        return ("F", s)

    wrapped = make_safe_parse_float(custom)
    assert wrapped is not custom
    assert wrapped("1.5") == ("F", "1.5")


def test_make_safe_parse_float_rejects_dict():
    wrapped = make_safe_parse_float(lambda s: {})
    with pytest.raises(ValueError) as excinfo:
        wrapped("1.0")
    assert "must not return dicts or lists" in str(excinfo.value)


def test_make_safe_parse_float_rejects_list():
    wrapped = make_safe_parse_float(lambda s: [])
    with pytest.raises(ValueError):
        wrapped("1.0")


def test_make_safe_parse_float_used_in_loads():
    with pytest.raises(ValueError):
        loads("x = 1.0\n", parse_float=lambda s: {})


# ---------------------------------------------------------------------------
# Integration-ish cases through loads
# ---------------------------------------------------------------------------


def test_loads_nested_arrays_and_tables():
    src = """
[[a]]
b = [{c = [1, 2]}, {c = [3]}]
"""
    assert loads(src) == {"a": [{"b": [{"c": [1, 2]}, {"c": [3]}]}]}


def test_loads_multiline_string_content():
    src = 'a = """\nline1\nline2"""\n'
    assert loads(src) == {"a": "line1\nline2"}


def test_loads_literal_multiline():
    src = "a = '''\nraw\\n'''\n"
    assert loads(src) == {"a": "raw\\n"}


def test_loads_escapes():
    src = r'a = "\b\t\n\f\r\e\"\\"' + "\n"
    result = loads(src)
    assert result["a"] == "\b\t\n\f\r\x1b\"\\"


def test_loads_hex_escape():
    assert loads(r'a = "\x41"') == {"a": "A"}
    assert loads(r'a = "\u0041"') == {"a": "A"}
    assert loads(r'a = "\U00000041"') == {"a": "A"}


def test_loads_dotted_keys_and_tables():
    src = "a.b.c = 1\n[t]\n[d]\nx = 2\n"
    result = loads(src)
    assert result["a"]["b"]["c"] == 1
    assert result["t"]["d"]["x"] == 2


def test_loads_array_of_tables_nested():
    src = "[[a]]\n[[a.b]]\nx = 1\n"
    result = loads(src)
    assert result == {"a": [{"b": [{"x": 1}]}]}
