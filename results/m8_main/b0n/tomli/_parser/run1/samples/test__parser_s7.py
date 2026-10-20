# SPDX-License-Identifier: MIT
"""Tests for tomli._parser."""

from __future__ import annotations

import datetime as dt
import io
import sys
from types import MappingProxyType

import pytest

from tomli._parser import (
    BASIC_STR_ESCAPE_REPLACEMENTS,
    DEPRECATED_DEFAULT,
    ILLEGAL_BASIC_STR_CHARS,
    ILLEGAL_COMMENT_CHARS,
    ILLEGAL_LITERAL_STR_CHARS,
    ILLEGAL_MULTILINE_BASIC_STR_CHARS,
    ILLEGAL_MULTILINE_LITERAL_STR_CHARS,
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
# Sentinel + TOMLDecodeError
# ---------------------------------------------------------------------------


class TestDeprecatedDefault:
    def test_is_class_sentinel(self):
        assert isinstance(DEPRECATED_DEFAULT, type)
        assert DEPRECATED_DEFAULT is not DEPRECATED_DEFAULT()


class TestTOMLDecodeError:
    def test_structured_args(self):
        doc = "a = 1\nbad line"
        pos = doc.index("bad")
        err = TOMLDecodeError("Nice message", doc, pos)
        assert isinstance(err, ValueError)
        assert err.msg == "Nice message"
        assert err.doc == doc
        assert err.pos == pos
        assert err.lineno == 2
        assert err.colno == 1
        assert str(err) == "Nice message (at line 2, column 1)"

    def test_first_line_column(self):
        doc = "abcdef"
        err = TOMLDecodeError("msg", doc, 3)
        assert err.lineno == 1
        assert err.colno == 4
        assert str(err) == "msg (at line 1, column 4)"

    def test_multiline_second_line_col(self):
        doc = "abc\ndefgh"
        pos = doc.index("g")  # index 6
        err = TOMLDecodeError("msg", doc, pos)
        assert err.lineno == 2
        assert err.colno == 3

    def test_end_of_document(self):
        doc = "abc"
        err = TOMLDecodeError("msg", doc, 3)
        assert err.pos == 3
        assert str(err) == "msg (at end of document)"

    def test_end_of_document_empty(self):
        err = TOMLDecodeError("msg", "", 0)
        assert err.lineno == 1
        assert err.colno == 1
        assert str(err) == "msg (at end of document)"

    def test_no_args_missing_structured_warns(self):
        with pytest.warns(DeprecationWarning):
            err = TOMLDecodeError()
        # free-form path
        assert isinstance(err, ValueError)

    def test_freeform_args_deprecated(self):
        with pytest.warns(DeprecationWarning):
            err = TOMLDecodeError("foo", "bar", 3)
        assert not hasattr(err, "msg")

    def test_extra_positional_args_deprecated(self):
        with pytest.warns(DeprecationWarning):
            err = TOMLDecodeError("msg", "doc", 0, "extra")
        assert isinstance(err, ValueError)

    def test_msg_not_str_deprecated(self):
        with pytest.warns(DeprecationWarning):
            TOMLDecodeError(123, "doc", 0)

    def test_doc_not_str_deprecated(self):
        with pytest.warns(DeprecationWarning):
            TOMLDecodeError("msg", b"doc", 0)

    def test_pos_not_int_deprecated(self):
        with pytest.warns(DeprecationWarning):
            TOMLDecodeError("msg", "doc", "0")


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------


class TestLoad:
    def test_basic(self):
        fp = io.BytesIO(b"a = 1\nb = 'x'\n")
        assert load(fp) == {"a": 1, "b": "x"}

    def test_parse_float(self):
        fp = io.BytesIO(b"a = 1.5\n")
        from decimal import Decimal

        result = load(fp, parse_float=Decimal)
        assert result == {"a": Decimal("1.5")}
        assert isinstance(result["a"], Decimal)

    def test_non_binary_file_raises(self):
        fp = io.StringIO("a = 1")
        with pytest.raises(TypeError, match="binary mode"):
            load(fp)

    def test_crlf_normalized(self):
        fp = io.BytesIO(b"a = 1\r\n")
        assert load(fp) == {"a": 1}


class TestLoads:
    def test_empty(self):
        assert loads("") == {}

    def test_simple(self):
        assert loads("a = 1") == {"a": 1}

    def test_parse_float_kwarg(self):
        def my_float(s):
            return ("F", s)

        assert loads("x = 1.5", parse_float=my_float) == {"x": ("F", "1.5")}

    def test_crlf_to_lf(self):
        assert loads("a = 1\r\nb = 2\r\n") == {"a": 1, "b": 2}

    def test_non_str_raises_typeerror(self):
        with pytest.raises(TypeError, match="Expected str object"):
            loads(b"a = 1")

    def test_non_str_none_raises(self):
        with pytest.raises(TypeError, match="Expected str object"):
            loads(None)

    def test_invalid_statement(self):
        with pytest.raises(TOMLDecodeError) as exc:
            loads("= 1")
        assert exc.value.msg == "Invalid initial character for a key part"

    def test_invalid_statement_char(self):
        with pytest.raises(TOMLDecodeError, match="Invalid statement"):
            loads("?")

    def test_expected_newline_after_statement(self):
        with pytest.raises(TOMLDecodeError, match="Expected newline"):
            loads("a = 1 b")

    def test_comment_only(self):
        assert loads("# a comment\n") == {}

    def test_comment_at_eof(self):
        assert loads("#comment") == {}

    def test_comment_then_ws(self):
        assert loads("a = 1 # comment\n") == {"a": 1}

    def test_tables_and_arrays(self):
        doc = """
[t]
a = 1
[[items]]
b = 2
[[items]]
b = 3
"""
        assert loads(doc) == {"t": {"a": 1}, "items": [{"b": 2}, {"b": 3}]}


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


class TestFlags:
    def test_empty_key(self):
        f = Flags()
        assert f.is_((), Flags.FROZEN) is False

    def test_set_and_is(self):
        f = Flags()
        f.set(("a",), Flags.FROZEN, recursive=False)
        assert f.is_(("a",), Flags.FROZEN) is True
        assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
        assert f.is_(("b",), Flags.FROZEN) is False

    def test_is_missing_parent(self):
        f = Flags()
        f.set(("a", "b"), Flags.FROZEN, recursive=False)
        assert f.is_(("x", "b"), Flags.FROZEN) is False
        assert f.is_(("a", "b"), Flags.FROZEN) is True

    def test_recursive(self):
        f = Flags()
        f.set(("a",), Flags.FROZEN, recursive=True)
        assert f.is_(("a",), Flags.FROZEN) is True
        assert f.is_(("a", "b"), Flags.FROZEN) is True
        assert f.is_(("a", "b", "c"), Flags.FROZEN) is True

    def test_recursive_does_not_leak_to_siblings(self):
        f = Flags()
        f.set(("a",), Flags.FROZEN, recursive=True)
        assert f.is_(("z",), Flags.FROZEN) is False

    def test_pending(self):
        f = Flags()
        f.add_pending(("a",), Flags.EXPLICIT_NEST)
        assert f.is_(("a",), Flags.EXPLICIT_NEST) is False
        f.finalize_pending()
        assert f.is_(("a",), Flags.EXPLICIT_NEST) is True
        # cleared
        assert f._pending_flags == set()

    def test_unset_all(self):
        f = Flags()
        f.set(("a",), Flags.FROZEN, recursive=False)
        f.set(("a", "b"), Flags.EXPLICIT_NEST, recursive=False)
        f.unset_all(("a",))
        assert f.is_(("a",), Flags.FROZEN) is False
        assert f.is_(("a", "b"), Flags.EXPLICIT_NEST) is False

    def test_unset_all_missing(self):
        f = Flags()
        # Should not raise
        f.unset_all(("nonexistent",))

    def test_unset_all_missing_inner(self):
        f = Flags()
        f.unset_all(("nonexistent", "child"))


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------


class TestNestedDict:
    def test_get_or_create_nest_new(self):
        nd = NestedDict()
        result = nd.get_or_create_nest(("a", "b"))
        assert result == {}
        assert nd.dict == {"a": {"b": {}}}

    def test_get_or_create_nest_existing(self):
        nd = NestedDict()
        nd.dict["a"] = {"b": {"c": 1}}
        assert nd.get_or_create_nest(("a", "b")) == {"c": 1}

    def test_get_or_create_nest_empty_key(self):
        nd = NestedDict()
        assert nd.get_or_create_nest(()) == nd.dict

    def test_access_lists(self):
        nd = NestedDict()
        nd.dict["a"] = [{"b": 1}]
        assert nd.get_or_create_nest(("a",)) == {"b": 1}

    def test_no_access_lists(self):
        nd = NestedDict()
        nd.dict["a"] = [{}]
        with pytest.raises(KeyError):
            nd.get_or_create_nest(("a", "x"), access_lists=False)

    def test_non_dict_value_raises(self):
        nd = NestedDict()
        nd.dict["a"] = 5
        with pytest.raises(KeyError, match="no nest behind this key"):
            nd.get_or_create_nest(("a", "b"))

    def test_append_nest_to_list_new(self):
        nd = NestedDict()
        nd.append_nest_to_list(("a",))
        assert nd.dict == {"a": [{}]}

    def test_append_nest_to_list_append(self):
        nd = NestedDict()
        nd.append_nest_to_list(("a",))
        nd.append_nest_to_list(("a",))
        assert nd.dict == {"a": [{}, {}]}

    def test_append_nest_to_list_nested(self):
        nd = NestedDict()
        nd.append_nest_to_list(("x", "y"))
        assert nd.dict == {"x": {"y": [{}]}}

    def test_append_nest_to_list_conflict(self):
        nd = NestedDict()
        nd.dict["a"] = 5
        with pytest.raises(KeyError, match="An object other than list"):
            nd.append_nest_to_list(("a",))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class TestOutput:
    def test_default(self):
        out = Output()
        assert isinstance(out.data, NestedDict)
        assert isinstance(out.flags, Flags)
        assert out.data.dict == {}


# ---------------------------------------------------------------------------
# skip_chars
# ---------------------------------------------------------------------------


class TestSkipChars:
    def test_basic(self):
        assert skip_chars("aaabbc", 0, frozenset("ab")) == 5

    def test_none(self):
        assert skip_chars("abc", 0, frozenset("x")) == 0

    def test_eof(self):
        assert skip_chars("aaa", 0, frozenset("a")) == 3

    def test_start_past(self):
        assert skip_chars("abc", 10, frozenset("a")) == 10


# ---------------------------------------------------------------------------
# skip_until
# ---------------------------------------------------------------------------


class TestSkipUntil:
    def test_found(self):
        assert (
            skip_until("hello world", 0, " ", error_on=frozenset(), error_on_eof=True)
            == 5
        )

    def test_found_with_offset(self):
        assert (
            skip_until(
                "a b c", 2, " ", error_on=frozenset(), error_on_eof=True
            )
            == 3
        )

    def test_eof_no_error(self):
        assert (
            skip_until("abc", 0, "Z", error_on=frozenset(), error_on_eof=False) == 3
        )

    def test_eof_error(self):
        with pytest.raises(TOMLDecodeError, match="Expected 'Z'"):
            skip_until("abc", 0, "Z", error_on=frozenset(), error_on_eof=True)

    def test_error_on_char(self):
        with pytest.raises(TOMLDecodeError, match="Found invalid character"):
            skip_until(
                "ab\x01c", 0, "c", error_on=frozenset("\x01"), error_on_eof=False
            )


# ---------------------------------------------------------------------------
# skip_comment
# ---------------------------------------------------------------------------


class TestSkipComment:
    def test_no_comment(self):
        assert skip_comment("abc", 0) == 0

    def test_eof(self):
        assert skip_comment("", 0) == 0

    def test_comment_to_newline(self):
        assert skip_comment("# hello\nrest", 0) == 7

    def test_comment_to_eof(self):
        assert skip_comment("# hello", 0) == 7

    def test_comment_illegal_char(self):
        with pytest.raises(TOMLDecodeError, match="Found invalid character"):
            skip_comment("# ab\x00c\n", 0)


# ---------------------------------------------------------------------------
# skip_comments_and_array_ws
# ---------------------------------------------------------------------------


class TestSkipCommentsAndArrayWs:
    def test_whitespace_only(self):
        assert skip_comments_and_array_ws("  \n\t ", 0) == 5

    def test_with_comment(self):
        assert skip_comments_and_array_ws("  # hi\n  x", 0) == 9

    def test_multiple_comments(self):
        src = " # a\n # b\n x"
        pos = skip_comments_and_array_ws(src, 0)
        assert src[pos] == "x"

    def test_stops(self):
        assert skip_comments_and_array_ws("x", 0) == 0


# ---------------------------------------------------------------------------
# parse_key_part / parse_key
# ---------------------------------------------------------------------------


class TestParseKeyPart:
    def test_bare(self):
        assert parse_key_part("abc-def_1 ", 0) == (10, "abc-def_1")

    def test_bare_stops(self):
        assert parse_key_part("abc.def", 0) == (3, "abc")

    def test_literal(self):
        assert parse_key_part("'a b'", 0) == (5, "a b")

    def test_basic(self):
        assert parse_key_part('"a b"', 0) == (5, "a b")

    def test_empty(self):
        with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
            parse_key_part("", 0)


class TestParseKey:
    def test_simple(self):
        assert parse_key("abc = 1", 0) == (4, ("abc",))

    def test_dotted(self):
        assert parse_key("a.b.c = 1", 0) == (6, ("a", "b", "c"))

    def test_dotted_with_ws(self):
        assert parse_key("a . b = 1", 0) == (6, ("a", "b"))

    def test_quoted(self):
        assert parse_key("'a'.b = 1", 0) == (6, ("a", "b"))

    def test_eof(self):
        # single key parsed at eof
        assert parse_key("abc", 0) == (3, ("abc",))

    def test_too_many_parts(self):
        src = ".".join("a" for _ in range(MAX_KEY_PARTS + 2))
        with pytest.raises(RecursionError, match="more than the allowed"):
            parse_key(src, 0)


# ---------------------------------------------------------------------------
# create_dict_rule / create_list_rule
# ---------------------------------------------------------------------------


class TestCreateDictRule:
    def test_simple(self):
        src = "[foo]"
        out = Output()
        pos, key = create_dict_rule(src, 0, out)
        assert pos == 5
        assert key == ("foo",)
        assert out.data.dict == {"foo": {}}
        assert out.flags.is_(("foo",), Flags.EXPLICIT_NEST)

    def test_nested(self):
        out = Output()
        pos, key = create_dict_rule("[a.b]", 0, out)
        assert key == ("a", "b")
        assert out.data.dict == {"a": {"b": {}}}

    def test_explicit_twice(self):
        out = Output()
        create_dict_rule("[a]", 0, out)
        with pytest.raises(TOMLDecodeError, match="Cannot declare"):
            create_dict_rule("[a]", 0, out)

    def test_frozen(self):
        out = Output()
        out.flags.set(("a",), Flags.FROZEN, recursive=True)
        with pytest.raises(TOMLDecodeError, match="Cannot declare"):
            create_dict_rule("[a]", 0, out)

    def test_overwrite_value(self):
        out = Output()
        out.data.dict["a"] = 5
        with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
            create_dict_rule("[a]", 0, out)

    def test_missing_bracket(self):
        out = Output()
        with pytest.raises(TOMLDecodeError, match=r"Expected '\]'"):
            create_dict_rule("[a", 0, out)


class TestCreateListRule:
    def test_simple(self):
        out = Output()
        pos, key = create_list_rule("[[foo]]", 0, out)
        assert pos == 7
        assert key == ("foo",)
        assert out.data.dict == {"foo": [{}]}

    def test_multiple(self):
        out = Output()
        create_list_rule("[[a]]", 0, out)
        create_list_rule("[[a]]", 0, out)
        assert out.data.dict == {"a": [{}, {}]}

    def test_frozen(self):
        out = Output()
        out.flags.set(("a",), Flags.FROZEN, recursive=True)
        with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable namespace"):
            create_list_rule("[[a]]", 0, out)

    def test_overwrite_value(self):
        out = Output()
        out.data.dict["a"] = 5
        with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
            create_list_rule("[[a]]", 0, out)

    def test_missing_double_bracket(self):
        out = Output()
        with pytest.raises(TOMLDecodeError, match=r"Expected '\]\]'"):
            create_list_rule("[[a]", 0, out)

    def test_after_dict_is_explicit(self):
        out = Output()
        create_dict_rule("[a]", 0, out)
        out.flags.finalize_pending()
        # declaring list on same name -> EXPLICIT_NEST is not FROZEN, so allowed
        create_list_rule("[[a]]", 0, out)
        assert out.data.dict == {"a": [{}]}


# ---------------------------------------------------------------------------
# key_value_rule
# ---------------------------------------------------------------------------


class TestKeyValueRule:
    def test_simple(self):
        out = Output()
        pos = key_value_rule("a = 1", 0, out, (), float)
        assert pos == 5
        assert out.data.dict == {"a": 1}

    def test_parse_float(self):
        out = Output()
        key_value_rule("a = 1.5", 0, out, (), lambda s: ("F", s))
        assert out.data.dict == {"a": ("F", "1.5")}

    def test_nested_key(self):
        out = Output()
        key_value_rule("a.b = 1", 0, out, (), float)
        assert out.data.dict == {"a": {"b": 1}}

    def test_with_header(self):
        out = Output()
        out.data.dict["h"] = {}
        key_value_rule("a = 1", 0, out, ("h",), float)
        assert out.data.dict == {"h": {"a": 1}}

    def test_sets_frozen_for_dict_value(self):
        out = Output()
        key_value_rule("a = {}", 0, out, (), float)
        assert out.flags.is_(("a",), Flags.FROZEN)

    def test_redefine_namespace(self):
        out = Output()
        key_value_rule("a.b = 1", 0, out, (), float)
        out.flags.finalize_pending()
        with pytest.raises(TOMLDecodeError, match="Cannot redefine namespace"):
            key_value_rule("a.c = 2", 0, out, (), float)

    def test_frozen_parent(self):
        out = Output()
        key_value_rule("a = {}", 0, out, (), float)
        with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable namespace"):
            key_value_rule("a.b = 1", 0, out, (), float)

    def test_overwrite(self):
        out = Output()
        out.data.dict["a"] = 1
        with pytest.raises(TOMLDecodeError, match="Cannot overwrite a value"):
            key_value_rule("a = 2", 0, out, (), float)


# ---------------------------------------------------------------------------
# parse_key_value_pair
# ---------------------------------------------------------------------------


class TestParseKeyValuePair:
    def test_simple(self):
        assert parse_key_value_pair("a = 1", 0, float, 0) == (5, ("a",), 1)

    def test_ws(self):
        assert parse_key_value_pair("a  =  1", 0, float, 0) == (7, ("a",), 1)

    def test_no_equal(self):
        with pytest.raises(TOMLDecodeError, match="Expected '='"):
            parse_key_value_pair("a 1", 0, float, 0)

    def test_eof_after_key(self):
        with pytest.raises(TOMLDecodeError, match="Expected '='"):
            parse_key_value_pair("a", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_one_line_basic_str
# ---------------------------------------------------------------------------


class TestParseOneLineBasicStr:
    def test_simple(self):
        assert parse_one_line_basic_str('"abc"', 0) == (5, "abc")

    def test_with_escapes(self):
        assert parse_one_line_basic_str('"a\\nb"', 0) == (6, "a\nb")


# ---------------------------------------------------------------------------
# parse_literal_str
# ---------------------------------------------------------------------------


class TestParseLiteralStr:
    def test_simple(self):
        assert parse_literal_str("'abc'", 0) == (5, "abc")

    def test_empty(self):
        assert parse_literal_str("''", 0) == (2, "")

    def test_illegal_char(self):
        with pytest.raises(TOMLDecodeError, match="Found invalid character"):
            parse_literal_str("'a\tb'", 0)

    def test_unterminated(self):
        with pytest.raises(TOMLDecodeError, match="Expected"):
            parse_literal_str("'abc", 0)


# ---------------------------------------------------------------------------
# parse_multiline_str
# ---------------------------------------------------------------------------


class TestParseMultilineStr:
    def test_basic_simple(self):
        assert parse_multiline_str('"""abc"""', 0, literal=False) == (9, "abc")

    def test_basic_strips_initial_newline(self):
        assert parse_multiline_str('"""\nabc"""', 0, literal=False) == (10, "abc")

    def test_literal_simple(self):
        assert parse_multiline_str("'''abc'''", 0, literal=True) == (9, "abc")

    def test_literal_strips_initial_newline(self):
        assert parse_multiline_str("'''\nabc'''", 0, literal=True) == (10, "abc")

    def test_extra_quotes_one(self):
        assert parse_multiline_str('"""abc""""', 0, literal=False) == (11, 'abc"')

    def test_extra_quotes_two(self):
        assert parse_multiline_str('"""abc"""""', 0, literal=False) == (12, 'abc""')

    def test_extra_apostrophes_one(self):
        assert parse_multiline_str("'''abc''''", 0, literal=True) == (11, "abc'")

    def test_extra_apostrophes_two(self):
        assert parse_multiline_str("'''abc'''''", 0, literal=True) == (12, "abc''")

    def test_literal_illegal_char(self):
        with pytest.raises(TOMLDecodeError, match="Found invalid character"):
            parse_multiline_str("'''a\x01'''", 0, literal=True)

    def test_literal_unterminated(self):
        with pytest.raises(TOMLDecodeError, match="Expected"):
            parse_multiline_str("'''abc", 0, literal=True)


# ---------------------------------------------------------------------------
# parse_basic_str / parse_basic_str_escape
# ---------------------------------------------------------------------------


class TestParseBasicStr:
    def test_simple(self):
        assert parse_basic_str('abc"', 0, multiline=False) == (4, "abc")

    def test_escapes(self):
        assert parse_basic_str('a\\tb"', 0, multiline=False) == (5, "a\tb")

    def test_illegal_char(self):
        with pytest.raises(TOMLDecodeError, match="Illegal character"):
            parse_basic_str("\x01\"", 0, multiline=False)

    def test_unterminated(self):
        with pytest.raises(TOMLDecodeError, match="Unterminated string"):
            parse_basic_str("abc", 0, multiline=False)

    def test_multiline_newline_ok(self):
        assert parse_basic_str('ab\nc"""', 0, multiline=True) == (7, "ab\nc")

    def test_multiline_single_quote(self):
        assert parse_basic_str('a"b"""', 0, multiline=True) == (6, 'a"b')

    def test_multiline_illegal_char(self):
        with pytest.raises(TOMLDecodeError, match="Illegal character"):
            parse_basic_str("a\x01b\"\"\"", 0, multiline=True)


class TestParseBasicStrEscape:
    @pytest.mark.parametrize("key,val", sorted(BASIC_STR_ESCAPE_REPLACEMENTS.items()))
    def test_replacements(self, key, val):
        assert parse_basic_str_escape(key, 0) == (2, val)

    def test_unknown_escape(self):
        with pytest.raises(TOMLDecodeError, match="Unescaped"):
            parse_basic_str_escape("\\z", 0)

    def test_hex_x(self):
        assert parse_basic_str_escape("\\x41", 0) == (4, "A")

    def test_hex_u(self):
        assert parse_basic_str_escape("\\u0041", 0) == (6, "A")

    def test_hex_U(self):
        assert parse_basic_str_escape("\\U00000041", 0) == (10, "A")

    def test_multiline_line_continuation(self):
        assert parse_basic_str_escape("\\\n  abc", 0, multiline=True) == (5, "")

    def test_multiline_ws_then_newline(self):
        assert parse_basic_str_escape("\\  \nabc", 0, multiline=True) == (5, "")

    def test_multiline_ws_then_eof(self):
        assert parse_basic_str_escape("\\  ", 0, multiline=True) == (3, "")

    def test_multiline_ws_then_char_error(self):
        with pytest.raises(TOMLDecodeError, match="Unescaped"):
            parse_basic_str_escape("\\  x", 0, multiline=True)


class TestParseBasicStrEscapeMultiline:
    def test_basic(self):
        assert parse_basic_str_escape_multiline("\\\n  abc", 0) == (5, "")


# ---------------------------------------------------------------------------
# parse_hex_char
# ---------------------------------------------------------------------------


class TestParseHexChar:
    def test_simple(self):
        assert parse_hex_char("41", 0, 2) == (2, "A")

    def test_u(self):
        assert parse_hex_char("0041", 0, 4) == (4, "A")

    def test_invalid_length(self):
        with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
            parse_hex_char("4", 0, 2)

    def test_invalid_char(self):
        with pytest.raises(TOMLDecodeError, match="Invalid hex value"):
            parse_hex_char("4G", 0, 2)

    def test_surrogate_rejected(self):
        with pytest.raises(TOMLDecodeError, match="not a Unicode scalar value"):
            parse_hex_char("D800", 0, 4)

    def test_out_of_range_rejected(self):
        with pytest.raises(TOMLDecodeError, match="not a Unicode scalar value"):
            parse_hex_char("110000", 0, 6)

    def test_max_valid(self):
        assert parse_hex_char("10FFFF", 0, 6) == (6, "\U0010FFFF")


# ---------------------------------------------------------------------------
# parse_array
# ---------------------------------------------------------------------------


class TestParseArray:
    def test_empty(self):
        assert parse_array("[]", 0, float, 0) == (2, [])

    def test_empty_with_ws(self):
        assert parse_array("[  ]", 0, float, 0) == (4, [])

    def test_simple(self):
        assert parse_array("[1, 2, 3]", 0, float, 0) == (9, [1, 2, 3])

    def test_trailing_comma(self):
        assert parse_array("[1, 2, ]", 0, float, 0) == (9, [1, 2])

    def test_nested(self):
        assert parse_array("[[1], [2]]", 0, float, 0) == (10, [[1], [2]])

    def test_with_comments(self):
        src = "[\n  1, # one\n  2, # two\n]"
        assert parse_array(src, 0, float, 0) == (len(src), [1, 2])

    def test_unclosed(self):
        with pytest.raises(TOMLDecodeError, match="Unclosed array"):
            parse_array("[1 2]", 0, float, 0)

    def test_parse_float(self):
        assert parse_array("[1.5]", 0, lambda s: ("F", s), 0) == (5, [("F", "1.5")])


# ---------------------------------------------------------------------------
# parse_inline_table
# ---------------------------------------------------------------------------


class TestParseInlineTable:
    def test_empty(self):
        assert parse_inline_table("{}", 0, float, 0) == (2, {})

    def test_empty_with_ws(self):
        assert parse_inline_table("{ }", 0, float, 0) == (3, {})

    def test_simple(self):
        assert parse_inline_table("{a = 1}", 0, float, 0) == (7, {"a": 1})

    def test_multiple(self):
        assert parse_inline_table("{a = 1, b = 2}", 0, float, 0) == (
            14,
            {"a": 1, "b": 2},
        )

    def test_trailing_comma(self):
        assert parse_inline_table("{a = 1,}", 0, float, 0) == (8, {"a": 1})

    def test_nested_key(self):
        assert parse_inline_table("{a.b = 1}", 0, float, 0) == (9, {"a": {"b": 1}})

    def test_nested_table(self):
        assert parse_inline_table("{a = {b = 1}}", 0, float, 0) == (
            13,
            {"a": {"b": 1}},
        )

    def test_duplicate_key(self):
        with pytest.raises(TOMLDecodeError, match="Duplicate inline table key"):
            parse_inline_table("{a = 1, a = 2}", 0, float, 0)

    def test_unclosed(self):
        with pytest.raises(TOMLDecodeError, match="Unclosed inline table"):
            parse_inline_table("{a = 1 b = 2}", 0, float, 0)

    def test_frozen_nested(self):
        with pytest.raises(TOMLDecodeError, match="Cannot mutate immutable namespace"):
            parse_inline_table("{a = {b = 1}, a.c = 2}", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------


class TestParseValue:
    def test_basic_str(self):
        assert parse_value('"x"', 0, float, 0) == (3, "x")

    def test_multiline_basic_str(self):
        assert parse_value('"""x"""', 0, float, 0) == (7, "x")

    def test_literal_str(self):
        assert parse_value("'x'", 0, float, 0) == (3, "x")

    def test_multiline_literal_str(self):
        assert parse_value("'''x'''", 0, float, 0) == (7, "x")

    def test_true(self):
        assert parse_value("true", 0, float, 0) == (4, True)

    def test_false(self):
        assert parse_value("false", 0, float, 0) == (5, False)

    def test_integer(self):
        assert parse_value("42", 0, float, 0) == (2, 42)

    def test_hex_integer(self):
        assert parse_value("0xff", 0, float, 0) == (4, 255)

    def test_float(self):
        assert parse_value("1.5", 0, float, 0) == (3, 1.5)

    def test_parse_float_called(self):
        assert parse_value("1.5", 0, lambda s: ("F", s), 0) == (3, ("F", "1.5"))

    def test_inf(self):
        assert parse_value("inf", 0, float, 0) == (3, float("inf"))

    def test_nan(self):
        result = parse_value("nan", 0, float, 0)
        assert result[0] == 3
        assert result[1] != result[1]  # NaN

    def test_neg_inf(self):
        assert parse_value("-inf", 0, float, 0) == (4, float("-inf"))

    def test_pos_inf(self):
        assert parse_value("+inf", 0, float, 0) == (4, float("inf"))

    def test_pos_nan(self):
        result = parse_value("+nan", 0, float, 0)
        assert result[0] == 4
        assert result[1] != result[1]

    def test_array(self):
        assert parse_value("[1, 2]", 0, float, 0) == (6, [1, 2])

    def test_inline_table(self):
        assert parse_value("{a = 1}", 0, float, 0) == (7, {"a": 1})

    def test_datetime(self):
        result = parse_value("1979-05-27T07:32:00Z", 0, float, 0)
        assert result[0] == 20
        assert result[1] == dt.datetime(
            1979, 5, 27, 7, 32, tzinfo=dt.timezone.utc
        )

    def test_date(self):
        assert parse_value("1979-05-27", 0, float, 0) == (10, dt.date(1979, 5, 27))

    def test_localtime(self):
        assert parse_value("07:32:00", 0, float, 0) == (8, dt.time(7, 32))

    def test_invalid_datetime(self):
        with pytest.raises(TOMLDecodeError, match="Invalid date or datetime"):
            parse_value("1979-13-27", 0, float, 0)

    def test_invalid_value(self):
        with pytest.raises(TOMLDecodeError, match="Invalid value"):
            parse_value("?", 0, float, 0)

    def test_eof_invalid_value(self):
        with pytest.raises(TOMLDecodeError, match="Invalid value"):
            parse_value("", 0, float, 0)

    def test_nesting_limit(self):
        if MAX_INLINE_NESTING == 400:
            # this branch only hits in mypyc, but the check is still reachable
            pass
        with pytest.raises(RecursionError):
            parse_value("1", 0, float, MAX_INLINE_NESTING + 1)


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------


class TestIsUnicodeScalarValue:
    @pytest.mark.parametrize("cp", [0, 1, 0x41, 55295, 57344, 1114111])
    def test_valid(self, cp):
        assert is_unicode_scalar_value(cp) is True

    @pytest.mark.parametrize("cp", [55296, 55297, 57343, 1114112, -1])
    def test_invalid(self, cp):
        assert is_unicode_scalar_value(cp) is False


# ---------------------------------------------------------------------------
# make_safe_parse_float
# ---------------------------------------------------------------------------


class TestMakeSafeParseFloat:
    def test_float_identity(self):
        assert make_safe_parse_float(float) is float

    def test_wrapper_returns_value(self):
        wrapped = make_safe_parse_float(lambda s: ("F", s))
        assert wrapped("1.5") == ("F", "1.5")

    def test_wrapper_rejects_dict(self):
        wrapped = make_safe_parse_float(lambda s: {})
        with pytest.raises(ValueError, match="must not return dicts or lists"):
            wrapped("1.5")

    def test_wrapper_rejects_list(self):
        wrapped = make_safe_parse_float(lambda s: [])
        with pytest.raises(ValueError, match="must not return dicts or lists"):
            wrapped("1.5")

    def test_used_via_loads(self):
        with pytest.raises(ValueError):
            loads("a = 1.5", parse_float=lambda s: [])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_ws(self):
        assert TOML_WS == frozenset(" \t")
        assert TOML_WS_AND_NEWLINE == frozenset(" \t\n")

    def test_illegal_sets(self):
        assert "\t" not in ILLEGAL_BASIC_STR_CHARS
        assert "\n" not in ILLEGAL_BASIC_STR_CHARS
        assert "\t" not in ILLEGAL_MULTILINE_BASIC_STR_CHARS
        assert "\n" not in ILLEGAL_MULTILINE_BASIC_STR_CHARS
        assert ILLEGAL_LITERAL_STR_CHARS is ILLEGAL_BASIC_STR_CHARS
        assert (
            ILLEGAL_MULTILINE_LITERAL_STR_CHARS
            is ILLEGAL_MULTILINE_BASIC_STR_CHARS
        )
        assert ILLEGAL_COMMENT_CHARS is ILLEGAL_BASIC_STR_CHARS

    def test_escape_replacements_is_readonly(self):
        if sys.version_info < (3, 15):
            assert isinstance(BASIC_STR_ESCAPE_REPLACEMENTS, MappingProxyType)
        with pytest.raises(TypeError):
            BASIC_STR_ESCAPE_REPLACEMENTS["x"] = "y"  # type: ignore[index]

    def test_max_key_parts(self):
        assert MAX_KEY_PARTS == sys.getrecursionlimit()

    def test_max_inline_nesting(self):
        assert MAX_INLINE_NESTING == 400
