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
    MAX_KEY_PARTS,
)


# ---------------------------------------------------------------------------
# DEPRECATED_DEFAULT / TOMLDecodeError
# ---------------------------------------------------------------------------


def test_deprecated_default_is_a_class_sentinel():
    assert isinstance(DEPRECATED_DEFAULT, type)


def test_decode_error_formatted_message():
    doc = "a = 1\nb = 2\n"
    err = TOMLDecodeError("Invalid stmt", doc, 0)
    assert err.msg == "Invalid stmt"
    assert err.doc is doc
    assert err.pos == 0
    assert err.lineno == 1
    assert err.colno == 1
    assert str(err) == "Invalid stmt (at line 1, column 1)"


def test_decode_error_second_line_column():
    doc = "abc\ndefg"
    err = TOMLDecodeError("oops", doc, 6)  # 'e' on line 2
    assert err.lineno == 2
    assert err.colno == 3
    assert "line 2, column 3" in str(err)


def test_decode_error_end_of_document():
    doc = "abcdef"
    err = TOMLDecodeError("oops", doc, len(doc))
    assert "end of document" in str(err)
    assert err.pos == len(doc)


def test_decode_error_freeform_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        err = TOMLDecodeError("free")
    assert isinstance(err, ValueError)
    assert str(err) == "free"
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_decode_error_freeform_multiple_args():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        err = TOMLDecodeError("a", "b", "c")
    assert "a b c" == str(err)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_decode_error_wrong_types_trigger_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        err = TOMLDecodeError(msg=1)  # not str
    assert isinstance(err, ValueError)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------


def test_load_basic():
    fp = io.BytesIO(b'a = 1\nb = "x"\n')
    assert load(fp) == {"a": 1, "b": "x"}


def test_load_requires_binary_mode():
    with pytest.raises(TypeError):
        load(io.StringIO("a = 1"))


def test_load_parse_float():
    fp = io.BytesIO(b"x = 1.5")
    result = load(fp, parse_float=lambda s: ("f", s))
    assert result == {"x": ("f", "1.5")}


def test_loads_basic():
    assert loads("a = 1") == {"a": 1}


def test_loads_crlf_normalized():
    assert loads("a = 1\r\nb = 2") == {"a": 1, "b": 2}


def test_loads_type_error():
    with pytest.raises(TypeError) as ei:
        loads(b"a = 1")
    assert "Expected str object" in str(ei.value)


def test_loads_type_error_none():
    with pytest.raises(TypeError):
        loads(None)


def test_loads_empty():
    assert loads("") == {}


def test_loads_only_whitespace_and_comments():
    assert loads("   \n\n# comment\n") == {}


def test_loads_invalid_statement():
    with pytest.raises(TOMLDecodeError) as ei:
        loads("& = 1")
    assert "Invalid statement" in str(ei.value)


def test_loads_expected_newline_after_statement():
    with pytest.raises(TOMLDecodeError) as ei:
        loads("a = 1 b = 2")
    assert "Expected newline or end of document" in str(ei.value)


def test_loads_parse_float_plumbed():
    assert loads("x = 3.5", parse_float=lambda s: ("pf", s)) == {"x": ("pf", "3.5")}


def test_loads_tables_and_arrays():
    src = """
[a]
b = 1
[[c]]
d = 2
[[c]]
d = 3
"""
    assert loads(src) == {"a": {"b": 1}, "c": [{"d": 2}, {"d": 3}]}


def test_loads_datetime():
    assert loads("d = 2021-01-01") == {"d": date(2021, 1, 1)}
    assert loads("d = 2021-01-01T00:00:00Z") == {
        "d": datetime(2021, 1, 1, tzinfo=timezone.utc)
    }


def test_loads_special_floats():
    assert loads("a = inf")["a"] == math.inf
    assert loads("a = -inf")["a"] == -math.inf
    assert math.isnan(loads("a = nan")["a"])


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_flags_empty_key_is_false():
    f = Flags()
    assert f.is_((), Flags.FROZEN) is False


def test_flags_set_and_is():
    f = Flags()
    f.set(("a",), Flags.FROZEN, recursive=False)
    assert f.is_(("a",), Flags.FROZEN) is True
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False


def test_flags_missing_key():
    f = Flags()
    assert f.is_(("x",), Flags.FROZEN) is False


def test_flags_recursive():
    f = Flags()
    f.set(("a",), Flags.FROZEN, recursive=True)
    assert f.is_(("a",), Flags.FROZEN) is True
    assert f.is_(("a", "b"), Flags.FROZEN) is True
    assert f.is_(("a", "b", "c"), Flags.FROZEN) is True


def test_flags_nested_paths():
    f = Flags()
    f.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
    assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is True
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
    assert f.is_(("a", "b", "c"), Flags.EXPLICIT_NEST) is False


def test_flags_pending():
    f = Flags()
    f.add_pending(("a",), Flags.EXPLICIT_NEST)
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
    f.finalize_pending()
    assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
    assert not f._pending_flags


def test_flags_unset_all():
    f = Flags()
    f.set(("a", "b"), Flags.FROZEN, recursive=True)
    f.unset_all(("a", "b"))
    assert f.is_(("a", "b"), Flags.FROZEN) is False


def test_flags_unset_all_missing_path():
    f = Flags()
    f.unset_all(("x", "y"))  # must not raise


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------


def test_nesteddict_get_or_create_nest():
    d = NestedDict()
    nest = d.get_or_create_nest(("a", "b"))
    nest["c"] = 1
    assert d.dict == {"a": {"b": {"c": 1}}}


def test_nesteddict_access_lists():
    d = NestedDict()
    d.dict["a"] = [{"b": 1}, {"b": 2}]
    nest = d.get_or_create_nest(("a",))
    assert nest is d.dict["a"][-1]


def test_nesteddict_access_lists_disabled():
    d = NestedDict()
    d.dict["a"] = {"b": 1}
    nest = d.get_or_create_nest(("a",), access_lists=False)
    assert nest is d.dict["a"]


def test_nesteddict_key_error_not_dict():
    d = NestedDict()
    d.dict["a"] = 1
    with pytest.raises(KeyError):
        d.get_or_create_nest(("a",))


def test_nesteddict_append_nest_to_list_creates():
    d = NestedDict()
    d.append_nest_to_list(("a",))
    assert d.dict == {"a": [{}]}


def test_nesteddict_append_nest_to_list_appends():
    d = NestedDict()
    d.append_nest_to_list(("a",))
    d.append_nest_to_list(("a",))
    assert d.dict == {"a": [{}, {}]}


def test_nesteddict_append_nest_not_list():
    d = NestedDict()
    d.dict["a"] = 1
    with pytest.raises(KeyError):
        d.append_nest_to_list(("a",))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_output():
    out = Output()
    assert isinstance(out.data, NestedDict)
    assert isinstance(out.flags, Flags)


# ---------------------------------------------------------------------------
# skip_chars / skip_until / skip_comment / skip_comments_and_array_ws
# ---------------------------------------------------------------------------


def test_skip_chars():
    assert skip_chars("   abc", 0, {" ", "\t"}) == 3


def test_skip_chars_eof():
    assert skip_chars("   ", 0, {" "}) == 3


def test_skip_chars_none_matched():
    assert skip_chars("abc", 0, {" ", "\t"}) == 0


def test_skip_until():
    assert skip_until("abc#def", 0, "#", error_on=frozenset(), error_on_eof=True) == 3


def test_skip_until_eof_error():
    with pytest.raises(TOMLDecodeError) as ei:
        skip_until("abc", 0, "#", error_on=frozenset(), error_on_eof=True)
    assert "Expected '#'" in str(ei.value)


def test_skip_until_eof_no_error():
    assert skip_until("abc", 0, "#", error_on=frozenset(), error_on_eof=False) == 3


def test_skip_until_invalid_char():
    with pytest.raises(TOMLDecodeError) as ei:
        skip_until(
            "a\x01b#x", 0, "#", error_on=frozenset({"\x01"}), error_on_eof=True
        )
    assert "Found invalid character" in str(ei.value)


def test_skip_comment_no_comment():
    assert skip_comment("abc", 0) == 0


def test_skip_comment_at_eof():
    src = "abc"
    assert skip_comment(src, 3) == 3


def test_skip_comment_with_comment():
    assert skip_comment("# hello\nnext", 0) == 7


def test_skip_comment_comment_to_eof():
    assert skip_comment("# hello", 0) == 7


def test_skip_comments_and_array_ws():
    src = "  # c\n\n   x"
    assert skip_comments_and_array_ws(src, 0) == 11


def test_skip_comments_and_array_ws_no_progress():
    assert skip_comments_and_array_ws("x", 0) == 0


# ---------------------------------------------------------------------------
# create_dict_rule / create_list_rule
# ---------------------------------------------------------------------------


def test_create_dict_rule():
    out = Output()
    pos, key = create_dict_rule("[a.b]", 0, out)
    assert key == ("a", "b")
    assert pos == 5
    assert out.data.dict == {"a": {"b": {}}}


def test_create_dict_rule_whitespace():
    out = Output()
    pos, key = create_dict_rule("[  a  ] rest", 0, out)
    assert key == ("a",)


def test_create_dict_rule_duplicate():
    out = Output()
    create_dict_rule("[a]", 0, out)
    with pytest.raises(TOMLDecodeError) as ei:
        create_dict_rule("[a]", 0, out)
    assert "Cannot declare" in str(ei.value)


def test_create_dict_rule_missing_bracket():
    out = Output()
    with pytest.raises(TOMLDecodeError) as ei:
        create_dict_rule("[a", 0, out)
    assert "Expected ']'" in str(ei.value)


def test_create_dict_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError) as ei:
        create_dict_rule("[a]", 0, out)
    assert "Cannot overwrite a value" in str(ei.value)


def test_create_list_rule():
    out = Output()
    pos, key = create_list_rule("[[a]]", 0, out)
    assert key == ("a",)
    assert pos == 5
    assert out.data.dict == {"a": [{}]}


def test_create_list_rule_twice():
    out = Output()
    create_list_rule("[[a]]", 0, out)
    create_list_rule("[[a]]", 0, out)
    assert out.data.dict == {"a": [{}, {}]}


def test_create_list_rule_missing_brackets():
    out = Output()
    with pytest.raises(TOMLDecodeError) as ei:
        create_list_rule("[[a]", 0, out)
    assert "Expected ']]'" in str(ei.value)


def test_create_list_rule_frozen():
    out = Output()
    out.flags.set(("a",), Flags.FROZEN, recursive=False)
    with pytest.raises(TOMLDecodeError) as ei:
        create_list_rule("[[a]]", 0, out)
    assert "Cannot mutate immutable namespace" in str(ei.value)


def test_create_list_rule_overwrite_value():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError) as ei:
        create_list_rule("[[a]]", 0, out)
    assert "Cannot overwrite a value" in str(ei.value)


# ---------------------------------------------------------------------------
# key_value_rule / parse_key_value_pair
# ---------------------------------------------------------------------------


def test_key_value_rule_basic():
    out = Output()
    pos = key_value_rule("a = 1", 0, out, (), float)
    assert pos == 5
    assert out.data.dict == {"a": 1}


def test_key_value_rule_duplicate():
    out = Output()
    key_value_rule("a = 1", 0, out, (), float)
    with pytest.raises(TOMLDecodeError) as ei:
        key_value_rule("a = 2", 0, out, (), float)
    assert "Cannot overwrite a value" in str(ei.value)


def test_key_value_rule_dotted_pending_flags():
    out = Output()
    key_value_rule("a.b = 1", 0, out, (), float)
    out.flags.finalize_pending()
    assert out.flags.is_(("a",), Flags.EXPLICIT_NEST) is True


def test_key_value_rule_redefine_namespace():
    out = Output()
    create_dict_rule("[a]", 0, out)
    with pytest.raises(TOMLDecodeError) as ei:
        key_value_rule("a.b = 1", 0, out, (), float)
    assert "Cannot redefine namespace" in str(ei.value)


def test_key_value_rule_frozen_parent():
    out = Output()
    key_value_rule("a = []", 0, out, (), float)
    with pytest.raises(TOMLDecodeError) as ei:
        key_value_rule("a.b = 1", 0, out, (), float)
    assert "Cannot mutate immutable namespace" in str(ei.value)


def test_key_value_rule_freezes_nested():
    out = Output()
    key_value_rule("a = {b = 1}", 0, out, (), float)
    assert out.flags.is_(("a",), Flags.FROZEN) is True
    assert out.flags.is_(("a", "b"), Flags.FROZEN) is True


def test_key_value_rule_overwrite_value_keyerror():
    out = Output()
    out.data.dict["a"] = 1
    with pytest.raises(TOMLDecodeError) as ei:
        key_value_rule("a.b = 2", 0, out, (), float)
    assert "Cannot overwrite a value" in str(ei.value)


def test_parse_key_value_pair():
    pos, key, value = parse_key_value_pair("a = 1", 0, float, 0)
    assert key == ("a",)
    assert value == 1
    assert pos == 5


def test_parse_key_value_pair_whitespace():
    pos, key, value = parse_key_value_pair("  a  =   1", 2, float, 0)
    assert (key, value, pos) == (("a",), 1, 10)


def test_parse_key_value_pair_missing_equals():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_key_value_pair("a 1", 0, float, 0)
    assert "Expected '='" in str(ei.value)


# ---------------------------------------------------------------------------
# parse_key / parse_key_part
# ---------------------------------------------------------------------------


def test_parse_key_simple():
    assert parse_key("abc", 0) == (3, ("abc",))


def test_parse_key_dotted():
    pos, key = parse_key("a.b.c", 0)
    assert key == ("a", "b", "c")
    assert pos == 5


def test_parse_key_with_spaces():
    pos, key = parse_key(" a . b ", 0)
    assert key == ("a", "b")
    assert pos == 7


def test_parse_key_quoted():
    assert parse_key('"a b"', 0) == (5, ("a b",))


def test_parse_key_too_many_parts():
    src = ".".join(["a"] * (MAX_KEY_PARTS + 1))
    with pytest.raises(RecursionError):
        parse_key(src, 0)


def test_parse_key_part_bare():
    assert parse_key_part("abc", 0) == (3, "abc")


def test_parse_key_part_literal():
    assert parse_key_part("'x y'", 0) == (5, "x y")


def test_parse_key_part_basic():
    assert parse_key_part('"x"', 0) == (3, "x")


def test_parse_key_part_invalid():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_key_part("=x", 0)
    assert "Invalid initial character" in str(ei.value)


async def _test_parse_key_part_eof_is_placeholder():
    pass


def test_parse_key_part_eof():
    with pytest.raises(TOMLDecodeError):
        parse_key_part("", 0)


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


def test_parse_one_line_basic_str():
    assert parse_one_line_basic_str('"abc"', 0) == (5, "abc")


def test_parse_one_line_basic_str_escape():
    assert parse_one_line_basic_str('"a\\nb"', 0) == (6, "a\nb")


def test_parse_literal_str():
    assert parse_literal_str("'abc'", 0) == (5, "abc")


def test_parse_literal_str_illegal_char():
    with pytest.raises(TOMLDecodeError):
        parse_literal_str("'a\x01b'", 0)


def test_parse_literal_str_unterminated():
    with pytest.raises(TOMLDecodeError):
        parse_literal_str("'abc", 0)


def test_parse_basic_str_simple():
    assert parse_basic_str("abc\"", 0, multiline=False) == (4, "abc")


def test_parse_basic_str_unterminated():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_basic_str("abc", 0, multiline=False)
    assert "Unterminated string" in str(ei.value)


def test_parse_basic_str_illegal_char():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_basic_str("a\x01b\"", 0, multiline=False)
    assert "Illegal character" in str(ei.value)


def test_parse_basic_str_multiline():
    assert parse_basic_str('ab"""', 0, multiline=True) == (5, "ab")


def test_parse_basic_str_multiline_quote_char():
    # A single quote inside multiline is kept
    assert parse_basic_str('a"b"""', 0, multiline=True) == (6, 'a"b')


def test_parse_multiline_str_literal():
    pos, val = parse_multiline_str("'''ab'''", 0, literal=True)
    assert val == "ab"
    assert pos == 9


def test_parse_multiline_str_literal_initial_newline():
    pos, val = parse_multiline_str("'''\nab'''", 0, literal=True)
    assert val == "ab"


def test_parse_multiline_str_basic():
    pos, val = parse_multiline_str('"""ab"""', 0, literal=False)
    assert val == "ab"


def test_parse_multiline_str_adds_extra_delims():
    pos, val = parse_multiline_str('"""ab""""', 0, literal=False)
    assert val == 'ab"'
    pos, val = parse_multiline_str('"""ab"""""', 0, literal=False)
    assert val == 'ab""'


def test_parse_basic_str_escape_simple():
    assert parse_basic_str_escape("\\n", 0) == (2, "\n")


def test_parse_basic_str_escape_all_simple():
    for esc, expected in [
        ("\\b", "\b"),
        ("\\t", "\t"),
        ("\\n", "\n"),
        ("\\f", "\f"),
        ("\\r", "\r"),
        ("\\e", "\x1b"),
        ('\\"', '"'),
        ("\\\\", "\\"),
    ]:
        assert parse_basic_str_escape(esc, 0) == (2, expected)


def test_parse_basic_str_escape_hex_x():
    assert parse_basic_str_escape("\\x41", 0) == (4, "A")


def test_parse_basic_str_escape_hex_u():
    assert parse_basic_str_escape("\\u00e9", 0) == (6, "\xe9")


def test_parse_basic_str_escape_hex_U():
    assert parse_basic_str_escape("\\U0001F600", 0) == (10, "\U0001F600")


def test_parse_basic_str_escape_invalid():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_basic_str_escape("\\q", 0)
    assert "Unescaped '\\'" in str(ei.value)


def test_parse_basic_str_escape_multiline_line_continuation():
    assert parse_basic_str_escape_multiline("\\\n   abc", 0) == (6, "")


def test_parse_basic_str_escape_multiline_space_then_newline():
    assert parse_basic_str_escape_multiline("\\   \n   abc", 0) == (9, "")


def test_parse_basic_str_escape_multiline_space_not_newline():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_basic_str_escape_multiline("\\  abc", 0)
    assert "Unescaped '\\'" in str(ei.value)


def test_parse_basic_str_escape_multiline_eof():
    assert parse_basic_str_escape_multiline("\\   ", 0) == (4, "")


def test_parse_hex_char_valid():
    assert parse_hex_char("41ab", 0, 2) == (2, "A")


def test_parse_hex_char_invalid():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_hex_char("zz", 0, 2)
    assert "Invalid hex value" in str(ei.value)


def test_parse_hex_char_too_short():
    with pytest.raises(TOMLDecodeError):
        parse_hex_char("4", 0, 2)


def test_parse_hex_char_not_scalar():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_hex_char("d800", 0, 4)
    assert "not a Unicode scalar value" in str(ei.value)


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------


def test_is_unicode_scalar_value():
    assert is_unicode_scalar_value(0) is True
    assert is_unicode_scalar_value(0xD7FF) is True
    assert is_unicode_scalar_value(0xD800) is False
    assert is_unicode_scalar_value(0xDFFF) is False
    assert is_unicode_scalar_value(0xE000) is True
    assert is_unicode_scalar_value(0x10FFFF) is True
    assert is_unicode_scalar_value(-1) is False
    assert is_unicode_scalar_value(0x110000) is False


# ---------------------------------------------------------------------------
# parse_array / parse_inline_table / parse_value
# ---------------------------------------------------------------------------


def test_parse_array_empty():
    assert parse_array("[]", 0, float, 0) == (2, [])


def test_parse_array_basic():
    pos, arr = parse_array("[1, 2, 3]", 0, float, 0)
    assert arr == [1, 2, 3]
    assert pos == 9


def test_parse_array_trailing_comma():
    _, arr = parse_array("[1, 2,]", 0, float, 0)
    assert arr == [1, 2]


def test_parse_array_whitespace_and_comments():
    src = "[\n  1,  # c\n  2\n]"
    _, arr = parse_array(src, 0, float, 0)
    assert arr == [1, 2]


def test_parse_array_nested():
    _, arr = parse_array("[[1], [2, 3]]", 0, float, 0)
    assert arr == [[1], [2, 3]]


def test_parse_array_unclosed():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_array("[1 2]", 0, float, 0)
    assert "Unclosed array" in str(ei.value)


def test_parse_inline_table_empty():
    assert parse_inline_table("{}", 0, float, 0) == (2, {})


def test_parse_inline_table_basic():
    pos, tbl = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
    assert tbl == {"a": 1, "b": 2}
    assert pos == 13


def test_parse_inline_table_dotted_keys():
    _, tbl = parse_inline_table("{a.b = 1}", 0, float, 0)
    assert tbl == {"a": {"b": 1}}


def test_parse_inline_table_trailing_comma():
    _, tbl = parse_inline_table("{a = 1,}", 0, float, 0)
    assert tbl == {"a": 1}


def test_parse_inline_table_duplicate():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_inline_table("{a = 1, a = 2}", 0, float, 0)
    assert "Duplicate inline table key" in str(ei.value)


def test_parse_inline_table_unclosed():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_inline_table("{a = 1 b = 2}", 0, float, 0)
    assert "Unclosed inline table" in str(ei.value)


def test_parse_inline_table_frozen():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_inline_table("{a = {b = 1}, a.b = 2}", 0, float, 0)
    assert "Cannot mutate immutable namespace" in str(ei.value)


def test_parse_value_string():
    assert parse_value('"x"', 0, float, 0) == (3, "x")


def test_parse_value_multiline():
    assert parse_value('"""x"""', 0, float, 0) == (7, "x")


def test_parse_value_literal():
    assert parse_value("'x'", 0, float, 0) == (3, "x")


def test_parse_value_multiline_literal():
    assert parse_value("'''x'''", 0, float, 0) == (7, "x")


def test_parse_value_true():
    assert parse_value("true", 0, float, 0) == (4, True)


def test_parse_value_false():
    assert parse_value("false", 0, float, 0) == (5, False)


def test_parse_value_array():
    assert parse_value("[1, 2]", 0, float, 0) == (6, [1, 2])


def test_parse_value_inline_table():
    assert parse_value("{a = 1}", 0, float, 0) == (7, {"a": 1})


def test_parse_value_date():
    assert parse_value("2021-01-02", 0, float, 0) == (10, date(2021, 1, 2))


def test_parse_value_datetime():
    assert parse_value("2021-01-02T03:04:05Z", 0, float, 0) == (
        20,
        datetime(2021, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )


def test_parse_value_localtime():
    assert parse_value("03:04:05", 0, float, 0) == (8, time(3, 4, 5))


def test_parse_value_integer():
    assert parse_value("123", 0, float, 0) == (3, 123)


def test_parse_value_float():
    assert parse_value("1.5", 0, float, 0) == (3, 1.5)


def test_parse_value_special_floats():
    assert parse_value("inf", 0, float, 0) == (3, math.inf)
    assert parse_value("-inf", 0, float, 0) == (4, -math.inf)
    assert math.isnan(parse_value("nan", 0, float, 0)[1])
    assert math.isnan(parse_value("+nan", 0, float, 0)[1])


def test_parse_value_invalid():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_value("@", 0, float, 0)
    assert "Invalid value" in str(ei.value)


def test_parse_value_invalid_date():
    with pytest.raises(TOMLDecodeError) as ei:
        parse_value("2021-13-45", 0, float, 0)
    assert "Invalid date or datetime" in str(ei.value)


def test_parse_value_nesting_too_deep():
    # nest_lvl > MAX_INLINE_NESTING must raise RecursionError
    from tomli._parser import MAX_INLINE_NESTING

    with pytest.raises(RecursionError):
        parse_value("[1]", 0, float, MAX_INLINE_NESTING + 1)


# ---------------------------------------------------------------------------
# make_safe_parse_float
# ---------------------------------------------------------------------------


def test_make_safe_parse_float_returns_float_itself():
    assert make_safe_parse_float(float) is float


def test_make_safe_parse_float_wraps_custom():
    def pf(s):
        return ("x", s)

    wrapped = make_safe_parse_float(pf)
    assert wrapped("1.5") == ("x", "1.5")


def test_make_safe_parse_float_rejects_dict():
    wrapped = make_safe_parse_float(lambda s: {})
    with pytest.raises(ValueError) as ei:
        wrapped("1.0")
    assert "must not return dicts or lists" in str(ei.value)


def test_make_safe_parse_float_rejects_list():
    wrapped = make_safe_parse_float(lambda s: [])
    with pytest.raises(ValueError):
        wrapped("1.0")


def test_loads_parse_float_dict_rejected():
    with pytest.raises(ValueError):
        loads("a = 1.0", parse_float=lambda s: {})
