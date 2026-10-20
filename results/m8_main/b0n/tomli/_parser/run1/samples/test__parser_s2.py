# SPDX-License-Identifier: MIT
import io
import sys
import warnings
from datetime import date, datetime, time, timezone

import pytest

from tomli import _parser
from tomli._parser import (
    MAX_INLINE_NESTING,
    MAX_KEY_PARTS,
    TOMLDecodeError,
    DEPRECATED_DEFAULT,
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
# TOMLDecodeError
# ---------------------------------------------------------------------------

class TestTOMLDecodeError:
    def test_basic_attributes(self):
        err = TOMLDecodeError("msg", "a=1\nb=2", 4)
        assert err.msg == "msg"
        assert err.doc == "a=1\nb=2"
        assert err.pos == 4
        # position 4 is on line 2
        assert err.lineno == 2
        assert err.colno == 1
        assert str(err) == "msg (at line 2, column 1)"

    def test_first_line_col(self):
        err = TOMLDecodeError("bad", "abcdef", 3)
        assert err.lineno == 1
        assert err.colno == 4
        assert str(err) == "bad (at line 1, column 4)"

    def test_pos_at_end_of_doc(self):
        err = TOMLDecodeError("eof", "abc", 3)
        assert str(err) == "eof (at end of document)"
        assert err.lineno == 1
        assert err.colno == 4

    def test_pos_beyond_end(self):
        err = TOMLDecodeError("over", "ab", 10)
        assert "end of document" in str(err)

    def test_is_value_error(self):
        assert issubclass(TOMLDecodeError, ValueError)

    def test_free_form_args_deprecated(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            err = TOMLDecodeError("custom message")
        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)
        assert str(err) == "custom message"

    def test_free_form_args_all(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            err = TOMLDecodeError("m", "d", 0, "extra")
        # With extra positional args, they are passed along to ValueError
        assert "m" in str(err)
        assert "d" in str(err)


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------

class TestLoads:
    def test_empty(self):
        assert loads("") == {}

    def test_simple(self):
        assert loads('a = 1\nb = "x"') == {"a": 1, "b": "x"}

    def test_crlf_normalized(self):
        assert loads('a = "x"\r\nb = 2') == {"a": "x", "b": 2}

    def test_crlf_in_multiline(self):
        assert loads('a = """x\r\n y"""') == {"a": "x\n y"}

    def test_comments(self):
        assert loads("# hi\na = 1 # trailing\n# end") == {"a": 1}

    def test_tables(self):
        result = loads("[a]\nb = 1\n[a.c]\nd = 2")
        assert result == {"a": {"b": 1, "c": {"d": 2}}}

    def test_array_of_tables(self):
        result = loads("[[a]]\nx = 1\n[[a]]\nx = 2")
        assert result == {"a": [{"x": 1}, {"x": 2}]}

    def test_parse_float_custom(self):
        result = loads("a = 1.5\nb = 2", parse_float=lambda s: "F" + s)
        assert result == {"a": "F1.5", "b": 2}

    def test_parse_float_rejects_list(self):
        with pytest.raises(ValueError):
            loads("a = 1.5", parse_float=lambda s: [])

    def test_parse_float_rejects_dict(self):
        with pytest.raises(ValueError):
            loads("a = 1.5", parse_float=lambda s: {})

    def test_not_str(self):
        with pytest.raises(TypeError):
            loads(b"a=1")  # type: ignore[arg-type]

    def test_invalid_statement(self):
        with pytest.raises(TOMLDecodeError, match="Invalid statement"):
            loads("!foo")

    def test_missing_newline_after_statement(self):
        with pytest.raises(TOMLDecodeError):
            loads("a = 1 b = 2")

    def test_key_too_many_parts(self):
        key = ".".join(["a"] * (MAX_KEY_PARTS + 1))
        with pytest.raises(RecursionError):
            loads(f"{key} = 1")


class TestLoad:
    def test_load_from_bytes(self, tmp_path):
        p = tmp_path / "f.toml"
        p.write_bytes(b'a = 1\nb = "hi"\n')
        with open(p, "rb") as f:
            assert load(f) == {"a": 1, "b": "hi"}

    def test_load_text_file_raises(self, tmp_path):
        p = tmp_path / "f.toml"
        p.write_text("a = 1\n")
        with open(p) as f:
            with pytest.raises(TypeError, match="binary mode"):
                load(f)  # type: ignore[arg-type]

    def test_load_io_bytes(self):
        assert load(io.BytesIO(b"a = 1")) == {"a": 1}

    def test_load_parse_float(self):
        assert load(io.BytesIO(b"a = 1.5"), parse_float=str) == {"a": "1.5"}


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

class TestFlags:
    def test_empty_key_is_false(self):
        flags = Flags()
        assert flags.is_((), Flags.FROZEN) is False

    def test_set_and_is(self):
        flags = Flags()
        flags.set(("a",), Flags.FROZEN, recursive=False)
        assert flags.is_(("a",), Flags.FROZEN) is True
        assert flags.is_(("a",), Flags.EXPLICIT_NEST) is False

    def test_nested(self):
        flags = Flags()
        flags.set(("a", "b"), Flags.FROZEN, recursive=False)
        assert flags.is_(("a", "b"), Flags.FROZEN) is True
        assert flags.is_(("a",), Flags.FROZEN) is False

    def test_recursive(self):
        flags = Flags()
        flags.set(("a",), Flags.FROZEN, recursive=True)
        assert flags.is_(("a",), Flags.FROZEN) is True
        assert flags.is_(("a", "b"), Flags.FROZEN) is True
        assert flags.is_(("a", "b", "c"), Flags.FROZEN) is True

    def test_pending(self):
        flags = Flags()
        flags.add_pending(("a",), Flags.EXPLICIT_NEST)
        assert flags.is_(("a",), Flags.EXPLICIT_NEST) is False
        flags.finalize_pending()
        assert flags.is_(("a",), Flags.EXPLICIT_NEST) is True

    def test_finalize_pending_clears(self):
        flags = Flags()
        flags.add_pending(("a",), Flags.EXPLICIT_NEST)
        flags.finalize_pending()
        # no more pending entries
        assert flags._pending_flags == set()

    def test_unset_all(self):
        flags = Flags()
        flags.set(("a", "b"), Flags.FROZEN, recursive=False)
        flags.unset_all(("a", "b"))
        assert flags.is_(("a", "b"), Flags.FROZEN) is False

    def test_unset_all_missing_parent_returns(self):
        flags = Flags()
        flags.unset_all(("x", "y", "z"))  # should not raise

    def test_unset_all_unsets_subtree(self):
        flags = Flags()
        flags.set(("a", "b"), Flags.FROZEN, recursive=False)
        flags.unset_all(("a",))
        assert flags.is_(("a", "b"), Flags.FROZEN) is False


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------

class TestNestedDict:
    def test_get_or_create(self):
        nd = NestedDict()
        result = nd.get_or_create_nest(("a", "b"))
        assert result == {}
        assert nd.dict == {"a": {"b": {}}}

    def test_get_existing(self):
        nd = NestedDict()
        nd.get_or_create_nest(("a",))
        assert nd.get_or_create_nest(("a",)) is nd.dict["a"]

    def test_access_list(self):
        nd = NestedDict()
        nd.dict["a"] = [{"b": {}}]
        result = nd.get_or_create_nest(("a", "b"))
        assert result is nd.dict["a"][-1]["b"]

    def test_no_access_list(self):
        nd = NestedDict()
        nd.dict["a"] = [{}]
        with pytest.raises(KeyError):
            nd.get_or_create_nest(("a", "b"), access_lists=False)

    def test_non_dict_in_path(self):
        nd = NestedDict()
        nd.dict["a"] = 5
        with pytest.raises(KeyError):
            nd.get_or_create_nest(("a", "b"))

    def test_append_nest_to_list_new(self):
        nd = NestedDict()
        nd.append_nest_to_list(("a",))
        assert nd.dict == {"a": [{}]}

    def test_append_nest_to_list_append(self):
        nd = NestedDict()
        nd.dict["a"] = [{"x": 1}]
        nd.append_nest_to_list(("a",))
        assert nd.dict == {"a": [{"x": 1}, {}]}

    def test_append_nest_to_list_non_list(self):
        nd = NestedDict()
        nd.dict["a"] = 5
        with pytest.raises(KeyError):
            nd.append_nest_to_list(("a",))

    def test_append_nest_nested_key(self):
        nd = NestedDict()
        nd.append_nest_to_list(("a", "b"))
        assert nd.dict == {"a": {"b": [{}]}}


class TestOutput:
    def test_defaults(self):
        out = Output()
        assert isinstance(out.data, NestedDict)
        assert isinstance(out.flags, Flags)
        assert out.data.dict == {}


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

class TestSkipChars:
    def test_basic(self):
        assert skip_chars("   abc", 0, " ") == 3

    def test_no_skip(self):
        assert skip_chars("abc", 0, " ") == 0

    def test_eof(self):
        assert skip_chars("   ", 0, " ") == 3

    def test_set_of_chars(self):
        assert skip_chars(" \t \tx", 0, " \t") == 5


class TestSkipUntil:
    def test_basic(self):
        assert skip_until(
            "abc]d", 0, "]", error_on=frozenset(), error_on_eof=True
        ) == 3

    def test_error_on_eof(self):
        with pytest.raises(TOMLDecodeError, match="Expected"):
            skip_until("abc", 0, "]", error_on=frozenset(), error_on_eof=True)

    def test_no_error_on_eof(self):
        assert skip_until(
            "abc", 0, "]", error_on=frozenset(), error_on_eof=False
        ) == 3

    def test_error_on_char(self):
        with pytest.raises(TOMLDecodeError, match="invalid character"):
            skip_until(
                "a\x01b]", 0, "]", error_on=frozenset("\x01"), error_on_eof=True
            )

    def test_error_on_char_before_expect(self):
        # invalid char appears before the expect string
        with pytest.raises(TOMLDecodeError):
            skip_until(
                "a\x01]", 0, "]", error_on=frozenset("\x01"), error_on_eof=True
            )


class TestSkipComment:
    def test_not_comment(self):
        assert skip_comment("abc", 0) == 0

    def test_comment_to_newline(self):
        assert skip_comment("# hello\nx", 0) == 7

    def test_comment_to_eof(self):
        assert skip_comment("# hello", 0) == 7

    def test_empty_source(self):
        assert skip_comment("", 0) == 0

    def test_illegal_char_in_comment(self):
        with pytest.raises(TOMLDecodeError):
            skip_comment("# a\x01b\n", 0)


class TestSkipCommentsAndArrayWs:
    def test_ws_and_newlines(self):
        assert skip_comments_and_array_ws(" \n\t x", 0) == 4

    def test_comment(self):
        # skip whitespace, comment, then whitespace
        assert skip_comments_and_array_ws(" # c\n x", 0) == 6

    def test_no_movement(self):
        assert skip_comments_and_array_ws("x", 0) == 0


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

class TestParseKeyPart:
    def test_bare(self):
        assert parse_key_part("abc def", 0) == (3, "abc")

    def test_bare_with_symbols(self):
        assert parse_key_part("a-b_c1 2", 0) == (6, "a-b_c1")

    def test_literal(self):
        assert parse_key_part("'a b' x", 0) == (5, "a b")

    def test_basic(self):
        assert parse_key_part('"a b" x', 0) == (5, "a b")

    def test_invalid(self):
        with pytest.raises(TOMLDecodeError, match="Invalid initial character"):
            parse_key_part("= x", 0)

    def test_empty(self):
        with pytest.raises(TOMLDecodeError):
            parse_key_part("", 0)


class TestParseKey:
    def test_simple(self):
        assert parse_key("abc = 1", 0) == (4, ("abc",))

    def test_dotted(self):
        assert parse_key("a . b = 1", 0) == (6, ("a", "b"))

    def test_dotted_trailing_ws(self):
        assert parse_key("a.b ", 0) == (4, ("a", "b"))

    def test_quoted_parts(self):
        assert parse_key('"a b".\'c d\' =', 0) == (11, ("a b", "c d"))

    def test_too_many_parts(self):
        key = ".".join(["a"] * (MAX_KEY_PARTS + 1)) + " = 1"
        with pytest.raises(RecursionError):
            parse_key(key, 0)


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------

class TestParseBasicStr:
    def test_simple(self):
        assert parse_one_line_basic_str('"abc" x', 0) == (5, "abc")

    def test_escapes(self):
        assert parse_one_line_basic_str(r'"a\tb\nc"', 0) == (9, "a\tb\nc")

    def test_all_escapes(self):
        src = r'"\b\t\n\f\r\e\"\\"'
        _, result = parse_one_line_basic_str(src, 0)
        assert result == "\b\t\n\f\r\x1b\"\\"

    def test_hex_escapes(self):
        assert parse_one_line_basic_str(r'"\x41\u0042\U00000043"', 0) == (
            19,
            "ABC",
        )

    def test_unterminated(self):
        with pytest.raises(TOMLDecodeError, match="Unterminated"):
            parse_one_line_basic_str('"abc', 0)

    def test_illegal_char(self):
        with pytest.raises(TOMLDecodeError, match="Illegal character"):
            parse_one_line_basic_str('"a\x01b"', 0)

    def test_bad_escape(self):
        with pytest.raises(TOMLDecodeError, match="Unescaped"):
            parse_one_line_basic_str(r'"a\qb"', 0)

    def test_hex_escape_illegal_value_d800(self):
        with pytest.raises(TOMLDecodeError, match="Unicode scalar"):
            parse_one_line_basic_str(r'"\uD800"', 0)

    def test_multiline_basic(self):
        result = parse_basic_str('abc""" rest', 0, multiline=True)
        assert result == (6, "abc")

    def test_multiline_with_quote_runs(self):
        # 4 quotes: content ends with one quote
        assert parse_basic_str('a"""" x', 0, multiline=True) == (5, 'a"')

    def test_multiline_with_five_quotes(self):
        assert parse_basic_str('a""""" x', 0, multiline=True) == (6, 'a""')

    def test_multiline_lone_quotes_inside(self):
        assert parse_basic_str('a"b""" x', 0, multiline=True) == (6, 'a"b')

    def test_multiline_unterminated(self):
        with pytest.raises(TOMLDecodeError, match="Unterminated"):
            parse_basic_str("abc", 0, multiline=True)

    def test_multiline_illegal_control(self):
        with pytest.raises(TOMLDecodeError, match="Illegal character"):
            parse_basic_str("a\x00b", 0, multiline=True)


class TestParseBasicStrEscape:
    def test_simple_escapes(self):
        assert parse_basic_str_escape(r"\n", 0) == (2, "\n")
        assert parse_basic_str_escape(r"\t", 0) == (2, "\t")

    def test_hex_x(self):
        assert parse_basic_str_escape(r"\x41", 0) == (4, "A")

    def test_hex_u(self):
        assert parse_basic_str_escape(r"\u0041", 0) == (6, "A")

    def test_hex_U(self):
        assert parse_basic_str_escape(r"\U00000041", 0) == (10, "A")

    def test_invalid_escape(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str_escape(r"\Q", 0)

    def test_multiline_line_ending_backslash(self):
        assert parse_basic_str_escape("\\\n  x", 0) == (4, "")

    def test_multiline_space_backslash(self):
        # backslash-space followed by newline, then more whitespace newlines
        assert parse_basic_str_escape("\\ \n\n  x", 0) == (4, "")

    def test_multiline_space_backslash_eof(self):
        assert parse_basic_str_escape("\\  ", 0, multiline=True) == (3, "")

    def test_multiline_space_then_non_newline(self):
        with pytest.raises(TOMLDecodeError, match="Unescaped"):
            parse_basic_str_escape("\\ x", 0, multiline=True)

    def test_multiline_wrapper(self):
        assert parse_basic_str_escape_multiline("\\\n x", 0) == (2, "")


class TestParseHexChar:
    def test_valid(self):
        assert parse_hex_char("41 rest", 0, 2) == (2, "A")

    def test_short(self):
        with pytest.raises(TOMLDecodeError, match="Invalid hex"):
            parse_hex_char("4", 0, 2)

    def test_non_hex(self):
        with pytest.raises(TOMLDecodeError, match="Invalid hex"):
            parse_hex_char("zz", 0, 2)

    def test_surrogate(self):
        with pytest.raises(TOMLDecodeError, match="Unicode scalar"):
            parse_hex_char("D800", 0, 4)

    def test_too_large(self):
        with pytest.raises(TOMLDecodeError, match="Unicode scalar"):
            parse_hex_char("110000", 0, 6)


class TestParseLiteralStr:
    def test_basic(self):
        assert parse_literal_str("'abc' x", 0) == (5, "abc")

    def test_empty(self):
        assert parse_literal_str("'' x", 0) == (2, "")

    def test_unterminated(self):
        with pytest.raises(TOMLDecodeError):
            parse_literal_str("'abc", 0)

    def test_illegal_char(self):
        with pytest.raises(TOMLDecodeError):
            parse_literal_str("'a\x01b'", 0)


class TestParseMultilineStr:
    def test_literal(self):
        assert parse_multiline_str("'''abc''' x", 0, literal=True) == (9, "abc")

    def test_literal_newline_trimmed(self):
        assert parse_multiline_str("'''\nabc''' x", 0, literal=True) == (
            10,
            "abc",
        )

    def test_basic(self):
        assert parse_multiline_str('"""abc""" x', 0, literal=False) == (9, "abc")

    def test_literal_four_quotes(self):
        assert parse_multiline_str("''''''' x", 0, literal=True) == (7, "'")

    def test_literal_five_quotes(self):
        assert parse_multiline_str("'''''''' x", 0, literal=True) == (8, "''")


# ---------------------------------------------------------------------------
# Arrays and inline tables
# ---------------------------------------------------------------------------

class TestParseArray:
    def test_empty(self):
        assert parse_array("[]", 0, float, 0) == (2, [])

    def test_simple(self):
        assert parse_array("[1, 2, 3]", 0, float, 0) == (9, [1, 2, 3])

    def test_trailing_comma(self):
        assert parse_array("[1, 2,]", 0, float, 0) == (7, [1, 2])

    def test_with_whitespace_and_comments(self):
        src = "[\n  1,  # c\n  2,\n]"
        assert parse_array(src, 0, float, 0) == (len(src), [1, 2])

    def test_nested(self):
        assert parse_array("[[1], [2]]", 0, float, 0) == (10, [[1], [2]])

    def test_unclosed(self):
        with pytest.raises(TOMLDecodeError, match="Unclosed array"):
            parse_array("[1 2]", 0, float, 0)

    def test_unclosed_eof(self):
        with pytest.raises(TOMLDecodeError):
            parse_array("[1", 0, float, 0)

    def test_recursion_limit(self):
        with pytest.raises(RecursionError):
            parse_array("[" * (MAX_INLINE_NESTING + 5), 0, float, 0)


class TestParseInlineTable:
    def test_empty(self):
        assert parse_inline_table("{}", 0, float, 0) == (2, {})

    def test_simple(self):
        assert parse_inline_table("{a = 1, b = 2}", 0, float, 0) == (
            14,
            {"a": 1, "b": 2},
        )

    def test_trailing_comma(self):
        assert parse_inline_table("{a = 1,}", 0, float, 0) == (9, {"a": 1})

    def test_nested(self):
        assert parse_inline_table("{a = {b = 1}}", 0, float, 0) == (
            14,
            {"a": {"b": 1}},
        )

    def test_unclosed(self):
        with pytest.raises(TOMLDecodeError, match="Unclosed inline table"):
            parse_inline_table("{a = 1 b = 2}", 0, float, 0)

    def test_duplicate_key(self):
        with pytest.raises(TOMLDecodeError, match="Duplicate inline table key"):
            parse_inline_table("{a = 1, a = 2}", 0, float, 0)

    def test_immutable_namespace(self):
        # nested value frozen, then mutated by dotted key
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a = {b = 1}, a.c = 2}", 0, float, 0)

    def test_overwrite_value(self):
        with pytest.raises(TOMLDecodeError, match="Cannot overwrite"):
            parse_inline_table("{a = 1, a.b = 2}", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------

class TestParseValue:
    def test_basic_string(self):
        assert parse_value('"x"', 0, float, 0) == (3, "x")

    def test_multiline_basic_string(self):
        assert parse_value('"""x"""', 0, float, 0) == (7, "x")

    def test_literal_string(self):
        assert parse_value("'x'", 0, float, 0) == (3, "x")

    def test_multiline_literal_string(self):
        assert parse_value("'''x'''", 0, float, 0) == (7, "x")

    def test_true_false(self):
        assert parse_value("true", 0, float, 0) == (4, True)
        assert parse_value("false", 0, float, 0) == (5, False)

    def test_integer(self):
        assert parse_value("42", 0, float, 0) == (2, 42)

    def test_hex_integer(self):
        assert parse_value("0x1f", 0, float, 0) == (4, 31)

    def test_float(self):
        assert parse_value("1.5", 0, float, 0) == (3, 1.5)

    def test_inf_nan(self):
        assert parse_value("inf", 0, float, 0) == (3, float("inf"))
        assert parse_value("-inf", 0, float, 0) == (4, float("-inf"))
        assert parse_value("+nan", 0, float, 0)[1] != parse_value("+nan", 0, float, 0)[1]

    def test_date(self):
        assert parse_value("2021-01-02", 0, float, 0) == (10, date(2021, 1, 2))

    def test_datetime_zulu(self):
        result = parse_value("2021-01-02T03:04:05Z", 0, float, 0)
        assert result[0] == 20
        assert result[1] == datetime(2021, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    def test_datetime_offset(self):
        result = parse_value("2021-01-02T03:04:05+01:00", 0, float, 0)
        assert result[0] == 25
        assert result[1].utcoffset().total_seconds() == 3600

    def test_localtime(self):
        assert parse_value("03:04:05", 0, float, 0) == (8, time(3, 4, 5))

    def test_invalid_date(self):
        with pytest.raises(TOMLDecodeError, match="Invalid date or datetime"):
            parse_value("2021-13-01", 0, float, 0)

    def test_invalid_value(self):
        with pytest.raises(TOMLDecodeError, match="Invalid value"):
            parse_value("!", 0, float, 0)

    def test_eof(self):
        with pytest.raises(TOMLDecodeError, match="Invalid value"):
            parse_value("", 0, float, 0)

    def test_nest_lvl_ok(self):
        # Exactly MAX_INLINE_NESTING doesn't raise (guard is about exceeding)
        assert parse_value("1", 0, float, MAX_INLINE_NESTING) == (1, 1)

    def test_custom_parse_float(self):
        assert parse_value("1.5", 0, lambda s: "F" + s, 0) == (3, "F1.5")


class TestParseKeyValuePair:
    def test_basic(self):
        assert parse_key_value_pair("a = 1", 0, float, 0) == (5, ("a",), 1)

    def test_dotted(self):
        assert parse_key_value_pair("a.b = 1", 0, float, 0) == (7, ("a", "b"), 1)

    def test_missing_equals(self):
        with pytest.raises(TOMLDecodeError, match="Expected '='"):
            parse_key_value_pair("a 1", 0, float, 0)

    def test_eof_after_key(self):
        with pytest.raises(TOMLDecodeError, match="Expected '='"):
            parse_key_value_pair("a", 0, float, 0)


# ---------------------------------------------------------------------------
# Rules (create_dict_rule, create_list_rule, key_value_rule)
# ---------------------------------------------------------------------------

class TestCreateDictRule:
    def test_basic(self):
        out = Output()
        pos, key = create_dict_rule("[a]", 0, out)
        assert pos == 3
        assert key == ("a",)
        assert out.data.dict == {"a": {}}
        assert out.flags.is_(("a",), Flags.EXPLICIT_NEST)

    def test_nested(self):
        out = Output()
        pos, key = create_dict_rule("[a.b]", 0, out)
        assert pos == 5
        assert key == ("a", "b")
        assert out.data.dict == {"a": {"b": {}}}

    def test_whitespace(self):
        out = Output()
        pos, key = create_dict_rule("[ a . b ]", 0, out)
        assert key == ("a", "b")
        assert pos == 9

    def test_duplicate_declaration(self):
        out = Output()
        create_dict_rule("[a]", 0, out)
        with pytest.raises(TOMLDecodeError, match="Cannot declare"):
            create_dict_rule("[a]", 0, out)

    def test_missing_bracket(self):
        out = Output()
        with pytest.raises(TOMLDecodeError, match="Expected ']'"):
            create_dict_rule("[a", 0, out)

    def test_overwrite_value(self):
        out = Output()
        out.data.dict["a"] = 5
        with pytest.raises(TOMLDecodeError, match="Cannot overwrite"):
            create_dict_rule("[a.b]", 0, out)


class TestCreateListRule:
    def test_basic(self):
        out = Output()
        pos, key = create_list_rule("[[a]]", 0, out)
        assert pos == 5
        assert key == ("a",)
        assert out.data.dict == {"a": [{}]}

    def test_multiple(self):
        out = Output()
        create_list_rule("[[a]]", 0, out)
        create_list_rule("[[a]]", 0, out)
        assert out.data.dict == {"a": [{}, {}]}

    def test_missing_double_bracket(self):
        out = Output()
        with pytest.raises(TOMLDecodeError, match="Expected ']]'"):
            create_list_rule("[[a]", 0, out)

    def test_cannot_overwrite_value(self):
        out = Output()
        out.data.dict["a"] = 5
        with pytest.raises(TOMLDecodeError, match="Cannot overwrite"):
            create_list_rule("[[a]]", 0, out)

    def test_frozen_namespace(self):
        out = Output()
        out.flags.set(("a",), Flags.FROZEN, recursive=True)
        with pytest.raises(TOMLDecodeError, match="immutable namespace"):
            create_list_rule("[[a]]", 0, out)

    def test_whitespace(self):
        out = Output()
        pos, key = create_list_rule("[[ a . b ]]", 0, out)
        assert key == ("a", "b")
        assert pos == 11


class TestKeyValueRule:
    def test_basic(self):
        out = Output()
        pos = key_value_rule("a = 1", 0, out, (), float)
        assert pos == 5
        assert out.data.dict == {"a": 1}

    def test_header_appended(self):
        out = Output()
        pos = key_value_rule("b = 1", 0, out, ("a",), float)
        assert out.data.dict == {"a": {"b": 1}}

    def test_dotted_key(self):
        out = Output()
        pos = key_value_rule("a.b = 1", 0, out, (), float)
        assert out.data.dict == {"a": {"b": 1}}

    def test_frozen_namespace(self):
        out = Output()
        out.flags.set(("a",), Flags.FROZEN, recursive=True)
        with pytest.raises(TOMLDecodeError, match="immutable namespace"):
            key_value_rule("a.b = 1", 0, out, (), float)

    def test_overwrite_value(self):
        out = Output()
        out.data.dict["a"] = 1
        with pytest.raises(TOMLDecodeError, match="Cannot overwrite"):
            key_value_rule("a = 2", 0, out, (), float)

    def test_redefine_namespace(self):
        out = Output()
        create_dict_rule("[a]", 0, out)
        with pytest.raises(TOMLDecodeError, match="Cannot redefine"):
            key_value_rule("a.b = 1", 0, out, (), float)

    def test_immutable_recursively_frozen(self):
        out = Output()
        out.flags.set(("a",), Flags.FROZEN, recursive=True)
        with pytest.raises(TOMLDecodeError, match="immutable namespace"):
            key_value_rule("a.b.c = 1", 0, out, (), float)

    def test_inline_table_frozen(self):
        out = Output()
        key_value_rule("a = {b = 1}", 0, out, (), float)
        assert out.flags.is_(("a",), Flags.FROZEN)
        with pytest.raises(TOMLDecodeError):
            key_value_rule("a.b = 2", 0, out, (), float)


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------

class TestIsUnicodeScalarValue:
    @pytest.mark.parametrize(
        "cp,expected",
        [
            (0, True),
            (65, True),
            (0xD7FF, True),
            (0xD800, False),
            (0xDC00, False),
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
    def test_default_float_returned_as_is(self):
        assert make_safe_parse_float(float) is float

    def test_custom_ok(self):
        f = make_safe_parse_float(lambda s: s.upper())
        assert f("abc") == "ABC"

    def test_rejects_list(self):
        f = make_safe_parse_float(lambda s: [])
        with pytest.raises(ValueError):
            f("1")

    def test_rejects_dict(self):
        f = make_safe_parse_float(lambda s: {})
        with pytest.raises(ValueError):
            f("1")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_inline_nesting(self):
        assert MAX_INLINE_NESTING == 400

    def test_max_key_parts(self):
        assert MAX_KEY_PARTS == sys.getrecursionlimit()
