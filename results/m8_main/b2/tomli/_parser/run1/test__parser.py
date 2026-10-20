import io
import math
import datetime as dt

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
    MAX_INLINE_NESTING,
    MAX_KEY_PARTS,
)


# ---------------------------------------------------------------------------
# DEPRECATED_DEFAULT / TOMLDecodeError
# ---------------------------------------------------------------------------

class TestTOMLDecodeError:
    def test_modern_signature_message(self):
        err = TOMLDecodeError("bad", "abc", 1)
        assert isinstance(err, ValueError)
        assert err.msg == "bad"
        assert err.doc == "abc"
        assert err.pos == 1
        assert str(err) == "bad (at line 1, column 2)"

    def test_coordinates_first_line(self):
        err = TOMLDecodeError("x", "abcdef", 3)
        assert err.lineno == 1
        assert err.colno == 4

    def test_coordinates_multiline(self):
        doc = "abc\ndef\nghi"
        err = TOMLDecodeError("x", doc, 5)
        # pos 5 is 'e' on line 2
        assert err.lineno == 2
        assert err.colno == 2
        assert str(err) == "x (at line 2, column 2)"

    def test_end_of_document(self):
        doc = "abc"
        err = TOMLDecodeError("x", doc, 3)
        assert str(err) == "x (at end of document)"
        assert err.lineno == 1
        assert err.colno == 4

    def test_pos_beyond_len_is_end_of_document(self):
        err = TOMLDecodeError("x", "abc", 100)
        assert str(err) == "x (at end of document)"

    def test_attributes_set(self):
        err = TOMLDecodeError("msg", "doc", 0)
        assert err.msg == "msg"
        assert err.doc == "doc"
        assert err.pos == 0

    def test_deprecated_freeform_warns(self):
        with pytest.warns(DeprecationWarning):
            err = TOMLDecodeError("a", "b", 1, "extra")
        # free-form path delegates to ValueError with all args
        assert args_of(err) == ("a", "b", 1, "extra")

    def test_deprecated_defaults_only_warns(self):
        with pytest.warns(DeprecationWarning):
            err = TOMLDecodeError()
        assert str(err) == ""

    def test_kwargs_signature_no_warning(self):
        # All three kwargs are valid types, so the modern path is used and
        # no DeprecationWarning is emitted.
        err = TOMLDecodeError(pos=5, doc="d", msg="m")
        assert err.msg == "m"
        assert err.doc == "d"
        assert err.pos == 5

    def test_deprecated_msg_non_str(self):
        with pytest.warns(DeprecationWarning):
            err = TOMLDecodeError(123, "doc", 2)
        assert args_of(err) == (123, "doc", 2)


def args_of(err):
    return err.args


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_basic(self):
        fp = io.BytesIO(b'a = 1\nb = "x"\n')
        assert load(fp) == {"a": 1, "b": "x"}

    def test_load_non_binary_raises_stub(self):
        # object without decode
        class Bad:
            def read(self):
                return 5

        with pytest.raises(TypeError):
            load(Bad())

    def test_load_string_io_raises_typeerror(self):
        # BytesIO-like that returns str -> has .decode? no; str decode works.
        with pytest.raises(TypeError):
            load(io.StringIO("a = 1"))

    def test_load_parse_float(self):
        fp = io.BytesIO(b"x = 1.5\n")
        import decimal

        result = load(fp, parse_float=decimal.Decimal)
        assert result["x"] == decimal.Decimal("1.5")


class TestLoads:
    def test_loads_empty(self):
        assert loads("") == {}

    def test_loads_simple_table(self):
        assert loads("[a]\nb = 1") == {"a": {"b": 1}}

    def test_loads_nested_table(self):
        assert loads("[a.b]\nc = 2") == {"a": {"b": {"c": 2}}}

    def test_loads_array_of_tables(self):
        doc = "[[a]]\nx=1\n[[a]]\nx=2\n"
        assert loads(doc) == {"a": [{"x": 1}, {"x": 2}]}

    def test_loads_crlf_normalized(self):
        assert loads("a = 1\r\nb = 2\r\n") == {"a": 1, "b": 2}

    def test_loads_crlf_in_string_literal(self):
        # \r\n inside a multiline basic string is normalized to \n
        assert loads('a = """x\r\ny"""') == {"a": "x\ny"}

    def test_loads_non_str_typeerror(self):
        with pytest.raises(TypeError):
            loads(b"a = 1")

    def test_loads_comments_and_blank_lines(self):
        doc = "# comment\n\na = 1 # trailing\n\n"
        assert loads(doc) == {"a": 1}

    def test_loads_invalid_statement(self):
        with pytest.raises(TOMLDecodeError):
            loads("$")

    def test_loads_expected_newline_after_statement(self):
        with pytest.raises(TOMLDecodeError):
            loads("a = 1 b = 2")

    def test_loads_missing_value(self):
        with pytest.raises(TOMLDecodeError):
            loads("a =")

    def test_loads_key_no_equals(self):
        with pytest.raises(TOMLDecodeError):
            loads("a")

    def test_loads_parse_float_custom(self):
        result = loads("x = 1.25", parse_float=lambda s: ("F", s))
        assert result == {"x": ("F", "1.25")}


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

class TestFlags:
    def test_empty_key_is_never_set(self):
        f = Flags()
        assert f.is_((), Flags.FROZEN) is False

    def test_set_empty_key_raises(self):
        f = Flags()
        with pytest.raises(IndexError):
            f.set((), Flags.FROZEN, recursive=False)

    def test_set_and_is(self):
        f = Flags()
        f.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
        assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
        assert f.is_(("a",), Flags.FROZEN) is False

    def test_missing_key(self):
        f = Flags()
        assert f.is_(("x", "y"), Flags.FROZEN) is False

    def test_recursive_flag_applies_to_children(self):
        f = Flags()
        f.set(("a",), Flags.FROZEN, recursive=True)
        assert f.is_(("a",), Flags.FROZEN) is True
        assert f.is_(("a", "b"), Flags.FROZEN) is True
        assert f.is_(("a", "b", "c"), Flags.FROZEN) is True

    def test_nonrecursive_does_not_apply_to_children(self):
        f = Flags()
        f.set(("a",), Flags.FROZEN, recursive=False)
        assert f.is_(("a", "b"), Flags.FROZEN) is False

    def test_recursive_mid_path(self):
        f = Flags()
        f.set(("a",), Flags.FROZEN, recursive=True)
        # path where a recursive flag exists at an intermediate level
        assert f.is_(("a", "b", "c"), Flags.FROZEN) is True

    def test_pending_flags(self):
        f = Flags()
        f.add_pending(("a",), Flags.EXPLICIT_NEST)
        # not applied yet
        assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
        f.finalize_pending()
        assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
        # pending cleared
        f._pending_flags.add(("b", Flags.EXPLICIT_NEST))
        # second finalize applies only new one
        assert f.is_(("b",), Flags.EXPLICIT_NEST) is False

    def test_unset_all(self):
        f = Flags()
        f.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
        assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is True
        f.unset_all(("a", "b"))
        assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is False

    def test_unset_all_missing_path_noop(self):
        f = Flags()
        f.unset_all(("x", "y", "z"))  # no error

    def test_unset_all_keeps_other_flags(self):
        f = Flags()
        f.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
        f.set(("a", "b"), Flags.FROZEN, recursive=False)
        f.unset_all(("a", "b"))
        assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is False
        assert f.is_(("a", "b"), Flags.FROZEN) is False


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------

class TestNestedDict:
    def test_get_or_create_nest(self):
        nd = NestedDict()
        nest = nd.get_or_create_nest(("a", "b"))
        assert nest == {}
        nest["c"] = 1
        assert nd.dict == {"a": {"b": {"c": 1}}}

    def test_get_or_create_empty(self):
        nd = NestedDict()
        assert nd.get_or_create_nest(()) is nd.dict

    def test_get_or_create_access_lists(self):
        nd = NestedDict()
        nd.dict["a"] = [{"x": 1}]
        nest = nd.get_or_create_nest(("a",))
        assert nest == {"x": 1}

    def test_get_or_create_no_access_lists(self):
        nd = NestedDict()
        nd.dict["a"] = [{"x": 1}]
        with pytest.raises(KeyError):
            nd.get_or_create_nest(("a", "b"), access_lists=False)

    def test_get_or_create_non_dict_raises(self):
        nd = NestedDict()
        nd.dict["a"] = 5
        with pytest.raises(KeyError):
            nd.get_or_create_nest(("a", "b"))

    def test_append_nest_to_list_creates(self):
        nd = NestedDict()
        nd.append_nest_to_list(("a",))
        assert nd.dict == {"a": [{}]}

    def test_append_nest_to_list_appends(self):
        nd = NestedDict()
        nd.append_nest_to_list(("a",))
        nd.append_nest_to_list(("a",))
        assert nd.dict == {"a": [{}, {}]}

    def test_append_nest_to_list_non_list_raises(self):
        nd = NestedDict()
        nd.dict["a"] = 5
        with pytest.raises(KeyError):
            nd.append_nest_to_list(("a",))

    def test_append_nest_nested(self):
        nd = NestedDict()
        nd.append_nest_to_list(("a", "b"))
        assert nd.dict == {"a": {"b": [{}]}}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class TestOutput:
    def test_output(self):
        out = Output()
        assert isinstance(out.data, NestedDict)
        assert isinstance(out.flags, Flags)
        assert out.data.dict == {}


# ---------------------------------------------------------------------------
# skip_chars / skip_until / skip_comment / skip_comments_and_array_ws
# ---------------------------------------------------------------------------

class TestSkip:
    def test_skip_chars(self):
        assert skip_chars("   abc", 0, frozenset(" ")) == 3

    def test_skip_chars_none(self):
        assert skip_chars("abc", 0, frozenset(" ")) == 0

    def test_skip_chars_eof(self):
        assert skip_chars("   ", 0, frozenset(" ")) == 3

    def test_skip_until_basic(self):
        assert skip_until("abcXdef", 0, "X", error_on=frozenset(), error_on_eof=True) == 3

    def test_skip_until_start_offset(self):
        assert skip_until("aXbXc", 2, "X", error_on=frozenset(), error_on_eof=True) == 3

    def test_skip_until_eof_error(self):
        with pytest.raises(TOMLDecodeError):
            skip_until("abc", 0, "X", error_on=frozenset(), error_on_eof=True)

    def test_skip_until_eof_no_error(self):
        assert skip_until("abc", 0, "X", error_on=frozenset(), error_on_eof=False) == 3

    def test_skip_until_error_char(self):
        with pytest.raises(TOMLDecodeError):
            skip_until("ab$cX", 0, "X", error_on=frozenset("$"), error_on_eof=True)

    def test_skip_comment_none(self):
        assert skip_comment("abc", 0) == 0

    def test_skip_comment_with_comment(self):
        assert skip_comment("# hi\nnext", 0) == 4

    def test_skip_comment_to_eof(self):
        assert skip_comment("# hi", 0) == 4

    def test_skip_comment_invalid_char(self):
        with pytest.raises(TOMLDecodeError):
            skip_comment("# a\x00b", 0)

    def test_skip_comment_at_eof_index(self):
        assert skip_comment("", 0) == 0

    def test_skip_comments_and_array_ws(self):
        assert skip_comments_and_array_ws("  \n# c\n  x", 0) == 9

    def test_skip_comments_and_array_ws_nothing(self):
        assert skip_comments_and_array_ws("x", 0) == 0

    def test_skip_comments_and_array_ws_multiple_comments(self):
        doc = "\n # a\n # b\n z"
        assert skip_comments_and_array_ws(doc, 0) == 12


# ---------------------------------------------------------------------------
# create_dict_rule / create_list_rule / key_value_rule
# ---------------------------------------------------------------------------

class TestRules:
    def test_create_dict_rule(self):
        out = Output()
        pos, key = create_dict_rule("[a.b]", 0, out)
        assert pos == 5
        assert key == ("a", "b")
        assert out.data.dict == {"a": {"b": {}}}

    def test_create_dict_rule_with_ws(self):
        out = Output()
        pos, key = create_dict_rule("[ a ]", 0, out)
        assert pos == 5
        assert key == ("a",)

    def test_create_dict_rule_duplicate(self):
        out = Output()
        create_dict_rule("[a]", 0, out)
        with pytest.raises(TOMLDecodeError):
            create_dict_rule("[a]", 0, out)

    def test_create_dict_rule_frozen(self):
        out = Output()
        out.flags.set(("a",), Flags.FROZEN, recursive=False)
        with pytest.raises(TOMLDecodeError):
            create_dict_rule("[a]", 0, out)

    def test_create_dict_rule_overwrite_value(self):
        out = Output()
        out.data.dict["a"] = 5
        with pytest.raises(TOMLDecodeError):
            create_dict_rule("[a.b]", 0, out)

    def test_create_dict_rule_missing_bracket(self):
        out = Output()
        with pytest.raises(TOMLDecodeError):
            create_dict_rule("[a", 0, out)

    def test_create_list_rule(self):
        out = Output()
        pos, key = create_list_rule("[[a]]", 0, out)
        assert pos == 5
        assert key == ("a",)
        assert out.data.dict == {"a": [{}]}

    def test_create_list_rule_multiple(self):
        out = Output()
        create_list_rule("[[a]]", 0, out)
        pos, key = create_list_rule("[[a]]", 0, out)
        assert out.data.dict == {"a": [{}, {}]}

    def test_create_list_rule_frozen(self):
        out = Output()
        out.flags.set(("a",), Flags.FROZEN, recursive=False)
        with pytest.raises(TOMLDecodeError):
            create_list_rule("[[a]]", 0, out)

    def test_create_list_rule_overwrite_value(self):
        out = Output()
        out.data.dict["a"] = 5
        with pytest.raises(TOMLDecodeError):
            create_list_rule("[[a]]", 0, out)

    def test_create_list_rule_missing_brackets(self):
        out = Output()
        with pytest.raises(TOMLDecodeError):
            create_list_rule("[[a]", 0, out)

    def test_key_value_rule_simple(self):
        out = Output()
        pos = key_value_rule("a = 1", 0, out, (), float)
        assert pos == 5
        assert out.data.dict == {"a": 1}

    def test_key_value_rule_with_header(self):
        out = Output()
        pos = key_value_rule("a = 1", 0, out, ("t",), float)
        assert out.data.dict == {"t": {"a": 1}}

    def test_key_value_rule_duplicate(self):
        out = Output()
        key_value_rule("a = 1", 0, out, (), float)
        with pytest.raises(TOMLDecodeError):
            key_value_rule("a = 2", 0, out, (), float)

    def test_key_value_rule_frozen_parent(self):
        out = Output()
        out.flags.set(("a",), Flags.FROZEN, recursive=True)
        with pytest.raises(TOMLDecodeError):
            key_value_rule("a.b = 1", 0, out, (), float)

    def test_key_value_rule_redefine_namespace(self):
        out = Output()
        out.flags.set(("a",), Flags.EXPLICIT_NEST, recursive=False)
        with pytest.raises(TOMLDecodeError):
            key_value_rule("a.b = 1", 0, out, (), float)

    def test_key_value_rule_inline_table_frozen(self):
        out = Output()
        key_value_rule("a = {b = 1}", 0, out, (), float)
        assert out.flags.is_(("a", "b"), Flags.FROZEN) is True
        assert out.flags.is_(("a", "zzz"), Flags.FROZEN) is True

    def test_key_value_rule_marks_pending(self):
        out = Output()
        key_value_rule("a.b = 1", 0, out, (), float)
        assert ("a",) in [k for k, _ in out.flags._pending_flags]


# ---------------------------------------------------------------------------
# parse_key_value_pair / parse_key / parse_key_part
# ---------------------------------------------------------------------------

class TestKeyParsing:
    def test_parse_key_value_pair(self):
        pos, key, val = parse_key_value_pair("a = 1", 0, float, 0)
        assert (pos, key, val) == (5, ("a",), 1)

    def test_parse_key_value_pair_no_equals(self):
        with pytest.raises(TOMLDecodeError):
            parse_key_value_pair("a", 0, float, 0)

    def test_parse_key_value_pair_dotted(self):
        pos, key, val = parse_key_value_pair("a . b = 2", 0, float, 0)
        assert key == ("a", "b")
        assert val == 2

    def test_parse_key_simple(self):
        assert parse_key("abc", 0) == (3, ("abc",))

    def test_parse_key_dotted(self):
        assert parse_key("a.b.c", 0) == (5, ("a", "b", "c"))

    def test_parse_key_leading_ws_is_error(self):
        # parse_key itself does not skip leading whitespace
        with pytest.raises(TOMLDecodeError):
            parse_key(" a . b ", 0)

    def test_parse_key_with_ws_around_dot(self):
        assert parse_key("a . b ", 0) == (6, ("a", "b"))

    def test_parse_key_quoted(self):
        assert parse_key('"a b"', 0) == (5, ("a b",))

    def test_parse_key_literal(self):
        assert parse_key("'a.b'", 0) == (5, ("a.b",))

    def test_parse_key_too_many_parts(self):
        doc = ".".join(["a"] * (MAX_KEY_PARTS + 1))
        with pytest.raises(RecursionError):
            parse_key(doc, 0)

    def test_parse_key_part_bare(self):
        assert parse_key_part("abc-1_2.x", 0) == (7, "abc-1_2")

    def test_parse_key_part_basic_quoted(self):
        assert parse_key_part('"abc"', 0) == (5, "abc")

    def test_parse_key_part_literal_quoted(self):
        assert parse_key_part("'abc'", 0) == (5, "abc")

    def test_parse_key_part_invalid(self):
        with pytest.raises(TOMLDecodeError):
            parse_key_part("$", 0)

    def test_parse_key_part_eof(self):
        with pytest.raises(TOMLDecodeError):
            parse_key_part("", 0)

    def test_parse_one_line_basic_str(self):
        assert parse_one_line_basic_str('"abc"', 0) == (5, "abc")


# ---------------------------------------------------------------------------
# parse_array / parse_inline_table
# ---------------------------------------------------------------------------

class TestArraysAndInlineTables:
    def test_parse_array_empty(self):
        assert parse_array("[]", 0, float, 0) == (2, [])

    def test_parse_array_simple(self):
        assert parse_array("[1, 2, 3]", 0, float, 0) == (9, [1, 2, 3])

    def test_parse_array_trailing_comma(self):
        assert parse_array("[1, 2,]", 0, float, 0) == (7, [1, 2])

    def test_parse_array_multiline(self):
        doc = "[\n 1,\n 2,\n]"
        pos, arr = parse_array(doc, 0, float, 0)
        assert arr == [1, 2]

    def test_parse_array_with_comments(self):
        doc = "[1, # c\n 2]"
        pos, arr = parse_array(doc, 0, float, 0)
        assert arr == [1, 2]

    def test_parse_array_unclosed(self):
        with pytest.raises(TOMLDecodeError):
            parse_array("[1 2]", 0, float, 0)

    def test_parse_array_nested(self):
        assert parse_array("[[1], [2]]", 0, float, 0)[1] == [[1], [2]]

    def test_parse_array_invalid_value(self):
        with pytest.raises(TOMLDecodeError):
            parse_array("[zzz]", 0, float, 0)

    def test_parse_array_deep_nesting_recursion(self):
        depth = MAX_INLINE_NESTING + 5
        doc = "[" * depth + "]" * depth
        with pytest.raises(RecursionError):
            parse_array(doc, 0, float, MAX_INLINE_NESTING)

    def test_parse_inline_table_empty(self):
        assert parse_inline_table("{}", 0, float, 0) == (2, {})

    def test_parse_inline_table_simple(self):
        pos, d = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
        assert d == {"a": 1, "b": 2}

    def test_parse_inline_table_trailing_comma(self):
        pos, d = parse_inline_table("{a = 1,}", 0, float, 0)
        assert d == {"a": 1}

    def test_parse_inline_table_dotted_key(self):
        pos, d = parse_inline_table("{a.b = 1}", 0, float, 0)
        assert d == {"a": {"b": 1}}

    def test_parse_inline_table_duplicate(self):
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a = 1, a = 2}", 0, float, 0)

    def test_parse_inline_table_unclosed(self):
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a = 1 b = 2}", 0, float, 0)

    def test_parse_inline_table_overwrite_value(self):
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a.b = 1, a = 2}", 0, float, 0)

    def test_parse_inline_table_frozen_namespace(self):
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a = {b = 1}, a.c = 2}", 0, float, 0)


# ---------------------------------------------------------------------------
# Strings and escapes
# ---------------------------------------------------------------------------

class TestStrings:
    def test_parse_basic_str_escape_simple(self):
        assert parse_basic_str_escape("\\n", 0) == (2, "\n")

    def test_parse_basic_str_escape_all_simple(self):
        mapping = {
            "\\b": "\u0008",
            "\\t": "\u0009",
            "\\n": "\u000a",
            "\\f": "\u000c",
            "\\r": "\u000d",
            "\\e": "\u001b",
            '\\"': "\u0022",
            "\\\\": "\u005c",
        }
        for esc, expected in mapping.items():
            assert parse_basic_str_escape(esc, 0) == (2, expected)

    def test_parse_basic_str_escape_invalid(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str_escape("\\q", 0)

    def test_parse_basic_str_escape_hex_x(self):
        assert parse_basic_str_escape("\\x41", 0) == (4, "A")

    def test_parse_basic_str_escape_hex_u(self):
        assert parse_basic_str_escape("\\u0041", 0) == (6, "A")

    def test_parse_basic_str_escape_hex_U(self):
        assert parse_basic_str_escape("\\U0001F600", 0) == (10, "\U0001F600")

    def test_parse_basic_str_escape_multiline_line_continuation(self):
        assert parse_basic_str_escape_multiline("\\\n  x", 0) == (4, "")

    def test_parse_basic_str_escape_multiline_backslash_space(self):
        assert parse_basic_str_escape_multiline("\\   \n  y", 0) == (7, "")

    def test_parse_basic_str_escape_multiline_space_not_newline(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str_escape_multiline("\\ x", 0)

    def test_parse_basic_str_escape_multiline_eof_after_ws(self):
        assert parse_basic_str_escape_multiline("\\  ", 0) == (3, "")

    def test_parse_hex_char_ok(self):
        assert parse_hex_char("41", 0, 2) == (2, "A")

    def test_parse_hex_char_invalid_digit(self):
        with pytest.raises(TOMLDecodeError):
            parse_hex_char("4Z", 0, 2)

    def test_parse_hex_char_too_short(self):
        with pytest.raises(TOMLDecodeError):
            parse_hex_char("4", 0, 2)

    def test_parse_hex_char_surrogate(self):
        with pytest.raises(TOMLDecodeError):
            parse_hex_char("D800", 0, 4)

    def test_parse_hex_char_too_large(self):
        with pytest.raises(TOMLDecodeError):
            parse_hex_char("110000", 0, 6)

    def test_parse_literal_str(self):
        assert parse_literal_str("'abc'", 0) == (5, "abc")

    def test_parse_literal_str_empty(self):
        assert parse_literal_str("''", 0) == (2, "")

    def test_parse_literal_str_unterminated(self):
        with pytest.raises(TOMLDecodeError):
            parse_literal_str("'abc", 0)

    def test_parse_literal_str_illegal_char(self):
        with pytest.raises(TOMLDecodeError):
            parse_literal_str("'a\x01b'", 0)

    def test_parse_multiline_str_literal(self):
        assert parse_multiline_str("'''abc'''", 0, literal=True) == (9, "abc")

    def test_parse_multiline_str_literal_leading_newline(self):
        pos, val = parse_multiline_str("'''\nabc'''", 0, literal=True)
        assert val == "abc"
        assert pos == 10

    def test_parse_multiline_str_basic(self):
        assert parse_multiline_str('"""abc"""', 0, literal=False) == (9, "abc")

    def test_parse_multiline_str_basic_leading_newline(self):
        pos, val = parse_multiline_str('"""\nabc"""', 0, literal=False)
        assert val == "abc"
        assert pos == 10

    def test_parse_multiline_str_four_quotes(self):
        # ending with 4 quotes -> one quote becomes part of the string
        pos, val = parse_multiline_str('"""abc""""', 0, literal=False)
        assert val == 'abc"'
        assert pos == 10

    def test_parse_multiline_str_five_quotes(self):
        pos, val = parse_multiline_str('"""abc"""""', 0, literal=False)
        assert val == 'abc""'
        assert pos == 11

    def test_parse_multiline_str_literal_unterminated(self):
        with pytest.raises(TOMLDecodeError):
            parse_multiline_str("'''abc", 0, literal=True)

    def test_parse_basic_str_simple(self):
        assert parse_basic_str('abc"', 0, multiline=False) == (4, "abc")

    def test_parse_basic_str_escape(self):
        assert parse_basic_str('a\\nb"', 0, multiline=False) == (5, "a\nb")

    def test_parse_basic_str_unterminated(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str("abc", 0, multiline=False)

    def test_parse_basic_str_illegal_char(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str("a\x01b\"", 0, multiline=False)

    def test_parse_basic_str_multiline(self):
        assert parse_basic_str('abc"""', 0, multiline=True) == (6, "abc")

    def test_parse_basic_str_multiline_single_quotes(self):
        # single/double quotes inside are kept; """ closes at pos 3..6
        assert parse_basic_str('a"b"""', 0, multiline=True) == (6, 'a"b')

    def test_parse_basic_str_multiline_newlines_ok(self):
        assert parse_basic_str('a\nb"""', 0, multiline=True) == (6, "a\nb")

    def test_parse_basic_str_multiline_illegal(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str("a\x01b\"\"\"", 0, multiline=True)


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------

class TestParseValue:
    def test_basic_string(self):
        assert parse_value('"abc"', 0, float, 0) == (5, "abc")

    def test_multiline_basic_string(self):
        assert parse_value('"""abc"""', 0, float, 0) == (9, "abc")

    def test_literal_string(self):
        assert parse_value("'abc'", 0, float, 0) == (5, "abc")

    def test_multiline_literal_string(self):
        assert parse_value("'''abc'''", 0, float, 0) == (9, "abc")

    def test_true(self):
        assert parse_value("true", 0, float, 0) == (4, True)

    def test_false(self):
        assert parse_value("false", 0, float, 0) == (5, False)

    def test_invalid_bool_prefix(self):
        with pytest.raises(TOMLDecodeError):
            parse_value("tru", 0, float, 0)

    def test_array(self):
        assert parse_value("[1, 2]", 0, float, 0) == (6, [1, 2])

    def test_inline_table(self):
        assert parse_value("{a = 1}", 0, float, 0) == (7, {"a": 1})

    def test_integer(self):
        assert parse_value("42", 0, float, 0) == (2, 42)

    def test_integer_negative(self):
        assert parse_value("-17", 0, float, 0) == (3, -17)

    def test_integer_hex(self):
        assert parse_value("0x10", 0, float, 0) == (4, 16)

    def test_float(self):
        assert parse_value("1.5", 0, float, 0) == (3, 1.5)

    def test_special_float_inf(self):
        pos, val = parse_value("inf", 0, float, 0)
        assert val == math.inf

    def test_special_float_nan(self):
        pos, val = parse_value("nan", 0, float, 0)
        assert math.isnan(val)

    def test_special_float_plus_inf(self):
        pos, val = parse_value("+inf", 0, float, 0)
        assert val == math.inf
        assert pos == 4

    def test_special_float_minus_inf(self):
        pos, val = parse_value("-inf", 0, float, 0)
        assert val == -math.inf
        assert pos == 4

    def test_date(self):
        assert parse_value("2021-01-01", 0, float, 0) == (10, dt.date(2021, 1, 1))

    def test_datetime(self):
        val = parse_value("2021-01-01T00:00:00Z", 0, float, 0)[1]
        assert val == dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)

    def test_localtime(self):
        assert parse_value("12:34:56", 0, float, 0) == (8, dt.time(12, 34, 56))

    def test_invalid_date(self):
        # RE_DATETIME does not match an out-of-range month; falls through
        # to the number regex which then fails via match_to_number / or is
        # rejected as an invalid value.
        with pytest.raises(TOMLDecodeError):
            loads("a = 2021-13-01")

    def test_invalid_date_value_error(self):
        # A date that matches the regex but has an out-of-range field
        # (e.g. February 30th) triggers match_to_datetime's ValueError.
        with pytest.raises(TOMLDecodeError):
            parse_value("2021-02-30", 0, float, 0)

    def test_invalid_value(self):
        with pytest.raises(TOMLDecodeError):
            parse_value("zzz", 0, float, 0)

    def test_eof(self):
        with pytest.raises(TOMLDecodeError):
            parse_value("", 0, float, 0)

    def test_nest_level_too_deep(self):
        with pytest.raises(RecursionError):
            parse_value("[", 0, float, MAX_INLINE_NESTING + 1)

    def test_parse_float_used(self):
        assert parse_value("1.5", 0, lambda s: ("f", s), 0) == (3, ("f", "1.5"))


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------

class TestUnicodeScalar:
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
    def test_values(self, cp, expected):
        assert is_unicode_scalar_value(cp) is expected


# ---------------------------------------------------------------------------
# make_safe_parse_float
# ---------------------------------------------------------------------------

class TestMakeSafeParseFloat:
    def test_float_identity(self):
        assert make_safe_parse_float(float) is float

    def test_custom_ok(self):
        safe = make_safe_parse_float(lambda s: 42)
        assert safe("x") == 42

    def test_custom_returns_dict(self):
        safe = make_safe_parse_float(lambda s: {})
        with pytest.raises(ValueError):
            safe("x")

    def test_custom_returns_list(self):
        safe = make_safe_parse_float(lambda s: [])
        with pytest.raises(ValueError):
            safe("x")

    def test_wrapping_preserves_value(self):
        safe = make_safe_parse_float(lambda s: s + "!")
        assert safe("a") == "a!"


# ---------------------------------------------------------------------------
# Integration roundtrip cases
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_full_document(self):
        doc = """
title = "TOML Example"
[owner]
name = "Tom"
dob = 1979-05-27T07:32:00-08:00
[database]
ports = [8000, 8001, 8002]
enabled = true
temp_targets = { cpu = 79.5, case = 72.0 }
"""
        data = loads(doc)
        assert data["title"] == "TOML Example"
        assert data["owner"]["name"] == "Tom"
        assert data["owner"]["dob"] == dt.datetime(
            1979, 5, 27, 7, 32, tzinfo=dt.timezone(dt.timedelta(hours=-8))
        )
        assert data["database"]["ports"] == [8000, 8001, 8002]
        assert data["database"]["enabled"] is True
        assert data["database"]["temp_targets"] == {"cpu": 79.5, "case": 72.0}

    def test_duplicate_table_error(self):
        with pytest.raises(TOMLDecodeError):
            loads("[a]\n[a]\n")

    def test_cannot_overwrite_value(self):
        with pytest.raises(TOMLDecodeError):
            loads("a = 1\na = 2\n")

    def test_dotted_redefine_error(self):
        with pytest.raises(TOMLDecodeError):
            loads("[a]\nb.c = 1\nb.c = 2\n")

    def test_immutable_inline_table(self):
        with pytest.raises(TOMLDecodeError):
            loads("a = {b = 1}\n[a.c]\n")
