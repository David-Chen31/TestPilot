"""Tests for tomli._parser."""

from __future__ import annotations

import datetime
import sys
import warnings

import pytest

from tomli._parser import (
    ASCII_CTRL,
    BARE_KEY_CHARS,
    BASIC_STR_ESCAPE_REPLACEMENTS,
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


class TestConstants:
    def test_ascii_ctrl(self):
        assert ASCII_CTRL == frozenset(chr(i) for i in range(32)) | {"\x7f"}
        assert "\x00" in ASCII_CTRL
        assert "\x1f" in ASCII_CTRL
        assert "\x7f" in ASCII_CTRL
        assert "\t" in ASCII_CTRL
        assert " " not in ASCII_CTRL

    def test_illegal_str_chars(self):
        assert "\t" not in ILLEGAL_BASIC_STR_CHARS
        assert "\n" in ILLEGAL_BASIC_STR_CHARS
        assert "\t" not in ILLEGAL_MULTILINE_BASIC_STR_CHARS
        assert "\n" not in ILLEGAL_MULTILINE_BASIC_STR_CHARS
        assert ILLEGAL_LITERAL_STR_CHARS == ILLEGAL_BASIC_STR_CHARS
        assert ILLEGAL_MULTILINE_LITERAL_STR_CHARS == ILLEGAL_MULTILINE_BASIC_STR_CHARS
        assert ILLEGAL_COMMENT_CHARS == ILLEGAL_BASIC_STR_CHARS

    def test_ws_chars(self):
        assert TOML_WS == frozenset({" ", "\t"})
        assert TOML_WS_AND_NEWLINE == frozenset({" ", "\t", "\n"})

    def test_key_chars(self):
        assert "a" in BARE_KEY_CHARS
        assert "Z" in BARE_KEY_CHARS
        assert "5" in BARE_KEY_CHARS
        assert "-" in BARE_KEY_CHARS
        assert "_" in BARE_KEY_CHARS
        assert "." not in BARE_KEY_CHARS
        assert "'" in KEY_INITIAL_CHARS
        assert '"' in KEY_INITIAL_CHARS

    def test_hexdigits(self):
        assert HEXDIGIT_CHARS == frozenset("abcdef" "ABCDEF" "0123456789")

    def test_escape_replacements(self):
        assert BASIC_STR_ESCAPE_REPLACEMENTS["\\n"] == "\n"
        assert BASIC_STR_ESCAPE_REPLACEMENTS["\\\\"] == "\\"
        assert BASIC_STR_ESCAPE_REPLACEMENTS['\\"'] == '"'

    def test_max_inline_nesting(self):
        assert MAX_INLINE_NESTING == 400

    def test_max_key_parts(self):
        assert MAX_KEY_PARTS == sys.getrecursionlimit()


# ---------------------------------------------------------------------------
# DEPRECATED_DEFAULT / TOMLDecodeError
# ---------------------------------------------------------------------------


class TestTOMLDecodeError:
    def test_normal_construction(self):
        exc = TOMLDecodeError("bad", "line1\nline2", 7)
        assert exc.msg == "bad"
        assert exc.doc == "line1\nline2"
        assert exc.pos == 7
        assert exc.lineno == 2
        assert exc.colno == 2
        assert "bad (at line 2, column 2)" in str(exc)

    def test_first_line_coord(self):
        exc = TOMLDecodeError("oops", "abcdef", 3)
        assert exc.lineno == 1
        assert exc.colno == 4
        assert "line 1, column 4" in str(exc)

    def test_end_of_document(self):
        exc = TOMLDecodeError("oops", "abc", 3)
        assert exc.pos == 3
        assert "end of document" in str(exc)
        assert "line" not in str(exc)

    def test_second_line_col(self):
        exc = TOMLDecodeError("x", "ab\ncd\nef", 5)
        assert exc.lineno == 2
        # pos=5 is 'd' which is right after "\n" -> col 2
        assert exc.colno == 2

    def test_frozen_sentinel_class_exists(self):
        # DEPRECATED_DEFAULT is a class used as a sentinel
        assert isinstance(DEPRECATED_DEFAULT, type)

    def test_free_form_deprecated(self):
        with pytest.warns(DeprecationWarning):
            exc = TOMLDecodeError("hello")
        # Value goes through ValueError.__init__ path
        assert "hello" in str(exc)
        # Attributes shouldn't exist since free-form path returns early
        assert not hasattr(exc, "msg") or "hello" not in getattr(exc, "msg", "")

    def test_free_form_deprecated_with_args(self):
        with pytest.warns(DeprecationWarning):
            exc = TOMLDecodeError("a", "b", 1, "extra")
        assert not hasattr(exc, "msg")


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_binary(self, tmp_path):
        fp = tmp_path / "x.toml"
        fp.write_bytes(b"a = 1\n")
        with open(fp, "rb") as f:
            result = load(f)
        assert result == {"a": 1}

    def test_load_text_raises(self, tmp_path):
        fp = tmp_path / "x.toml"
        fp.write_text("a = 1\n")
        with open(fp, "r") as f:
            with pytest.raises(TypeError):
                load(f)

    def test_load_parse_float(self, tmp_path):
        fp = tmp_path / "x.toml"
        fp.write_bytes(b"a = 1.5\n")
        with open(fp, "rb") as f:
            result = load(f, parse_float=lambda s: "F" + s)
        assert result == {"a": "F1.5"}


class TestLoads:
    def test_simple(self):
        assert loads("a = 1\n") == {"a": 1}

    def test_empty(self):
        assert loads("") == {}

    def test_comment_only(self):
        assert loads("# hello\n") == {}

    def test_type_error_int(self):
        with pytest.raises(TypeError):
            loads(1)

    def test_type_error_none(self):
        with pytest.raises(TypeError):
            loads(None)

    def test_type_error_bytes(self):
        with pytest.raises(TypeError):
            loads(b"a = 1")

    def test_crlf_normalized(self):
        assert loads("a = 1\r\nb = 2\r\n") == {"a": 1, "b": 2}

    def test_crlf_in_string(self):
        assert loads('a = "x\r\ny"\n') == {"a": "x\ny"}

    def test_invalid_statement(self):
        with pytest.raises(TOMLDecodeError):
            loads("a = 1 b = 2\n")

    def test_leading_ws(self):
        assert loads("   a = 1\n") == {"a": 1}

    def test_missing_newline_after_statement(self):
        with pytest.raises(TOMLDecodeError):
            loads("a = 1 b = 2")

    def test_comments_between(self):
        result = loads("# c1\na = 1 # c2\n# c3\nb = 2\n")
        assert result == {"a": 1, "b": 2}

    def test_parse_float_custom(self):
        result = loads("x = 1.5\n", parse_float=lambda s: s)
        assert result == {"x": "1.5"}

    def test_parse_float_returning_list_raises(self):
        def bad(s):
            return []

        with pytest.raises(ValueError):
            loads("x = 1.5\n", parse_float=bad)

    def test_parse_float_returning_dict_raises(self):
        def bad(s):
            return {}

        with pytest.raises(ValueError):
            loads("x = 1.5\n", parse_float=bad)

    def test_inf_nan(self):
        result = loads("a = inf\nb = -inf\nc = nan\nd = -nan\n")
        import math

        assert result["a"] == math.inf
        assert result["b"] == -math.inf
        assert math.isnan(result["c"])
        assert math.isnan(result["d"])

    def test_table(self):
        assert loads("[t]\na = 1\n") == {"t": {"a": 1}}

    def test_array_of_tables(self):
        assert loads("[[t]]\na = 1\n[[t]]\na = 2\n") == {
            "t": [{"a": 1}, {"a": 2}]
        }

    def test_invalid_statement_char(self):
        with pytest.raises(TOMLDecodeError):
            loads("! bang\n")


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


class TestFlags:
    def test_set_and_is(self):
        f = Flags()
        f.set(("a",), 0, recursive=False)
        assert f.is_(("a",), 0)
        assert not f.is_(("a",), 1)

    def test_empty_key_false(self):
        f = Flags()
        assert f.is_((), 0) is False

    def test_missing_key(self):
        f = Flags()
        assert f.is_(("x",), 0) is False

    def test_recursive(self):
        f = Flags()
        f.set(("a",), 0, recursive=True)
        assert f.is_(("a", "b", "c"), 0)

    def test_nested_path(self):
        f = Flags()
        f.set(("a", "b"), 1, recursive=False)
        assert f.is_(("a", "b"), 1)
        assert not f.is_(("a", "b", "c"), 1)
        assert not f.is_(("a",), 1)

    def test_unset_all(self):
        f = Flags()
        f.set(("a", "b"), 0, recursive=False)
        f.unset_all(("a", "b"))
        assert not f.is_(("a", "b"), 0)

    def test_unset_all_missing_parent(self):
        f = Flags()
        # Should not raise
        f.unset_all(("x", "y"))

    def test_pending(self):
        f = Flags()
        f.add_pending(("a",), 1)
        # Not yet visible
        assert not f.is_(("a",), 1)
        f.finalize_pending()
        assert f.is_(("a",), 1)
        assert f._pending_flags == set()

    def test_set_recursive_on_intermediate(self):
        f = Flags()
        f.set(("a", "b"), 1, recursive=True)
        assert f.is_(("a", "b"), 1)
        assert not f.is_(("a", "b", "c"), 1)


# ---------------------------------------------------------------------------
# NestedDict
# ---------------------------------------------------------------------------


class TestNestedDict:
    def test_create_and_get(self):
        nd = NestedDict()
        assert nd.dict == {}
        nest = nd.get_or_create_nest(("a", "b"))
        nest["c"] = 1
        assert nd.dict == {"a": {"b": {"c": 1}}}

    def test_access_lists(self):
        nd = NestedDict()
        nd.dict["a"] = [{"x": 1}]
        result = nd.get_or_create_nest(("a",), access_lists=True)
        assert result == {"x": 1}

    def test_no_access_lists(self):
        nd = NestedDict()
        nd.dict["a"] = [{"x": 1}]
        with pytest.raises(KeyError):
            nd.get_or_create_nest(("a",), access_lists=False)

    def test_non_dict_raises(self):
        nd = NestedDict()
        nd.dict["a"] = 5
        with pytest.raises(KeyError):
            nd.get_or_create_nest(("a", "b"))

    def test_append_new_list(self):
        nd = NestedDict()
        nd.append_nest_to_list(("a",))
        assert nd.dict == {"a": [{}]}

    def test_append_existing_list(self):
        nd = NestedDict()
        nd.append_nest_to_list(("a",))
        nd.append_nest_to_list(("a",))
        assert nd.dict == {"a": [{}, {}]}

    def test_append_non_list_raises(self):
        nd = NestedDict()
        nd.dict["a"] = 5
        with pytest.raises(KeyError):
            nd.append_nest_to_list(("a",))


class TestOutput:
    def test_init(self):
        out = Output()
        assert isinstance(out.data, NestedDict)
        assert isinstance(out.flags, Flags)
        assert out.data.dict == {}


# ---------------------------------------------------------------------------
# skip_chars / skip_until / skip_comment / skip_comments_and_array_ws
# ---------------------------------------------------------------------------


class TestSkipChars:
    def test_basic(self):
        assert skip_chars("   abc", 0, frozenset(" ")) == 3

    def test_eof(self):
        assert skip_chars("   ", 0, frozenset(" ")) == 3

    def test_start_at_pos(self):
        assert skip_chars("a  b", 1, frozenset(" ")) == 3

    def test_none_skipped(self):
        assert skip_chars("abc", 0, frozenset(" ")) == 0


class TestSkipUntil:
    def test_find(self):
        assert (
            skip_until(
                "abc)def",
                0,
                ")",
                error_on=frozenset(),
                error_on_eof=False,
            )
            == 3
        )

    def test_eof_not_error(self):
        assert (
            skip_until(
                "abc",
                0,
                ")",
                error_on=frozenset(),
                error_on_eof=False,
            )
            == 3
        )

    def test_eof_error(self):
        with pytest.raises(TOMLDecodeError):
            skip_until("abc", 0, ")", error_on=frozenset(), error_on_eof=True)

    def test_invalid_char_before_expect(self):
        with pytest.raises(TOMLDecodeError):
            skip_until(
                "ab\x01c)",
                0,
                ")",
                error_on=frozenset(["\x01"]),
                error_on_eof=False,
            )

    def test_invalid_char_after_expect_ignored(self):
        # error_on only checked before expect
        assert (
            skip_until(
                "a)\x01",
                0,
                ")",
                error_on=frozenset(["\x01"]),
                error_on_eof=False,
            )
            == 1
        )


class TestSkipComment:
    def test_no_comment(self):
        assert skip_comment("abc", 0) == 0

    def test_comment(self):
        assert skip_comment("# hello\nnext", 0) == 7

    def test_comment_eof(self):
        assert skip_comment("# hello", 0) == 7

    def test_comment_at_eof_index(self):
        assert skip_comment("", 0) == 0

    def test_illegal_char(self):
        with pytest.raises(TOMLDecodeError):
            skip_comment("# a\x01b\n", 0)


class TestSkipCommentsAndArrayWs:
    def test_ws_and_comment(self):
        src = "  # c\n  x"
        assert skip_comments_and_array_ws(src, 0) == 8

    def test_no_progress(self):
        assert skip_comments_and_array_ws("abc", 0) == 0

    def test_multiple_comments(self):
        src = "# a\n# b\nx"
        assert skip_comments_and_array_ws(src, 0) == 8


# ---------------------------------------------------------------------------
# create_dict_rule / create_list_rule
# ---------------------------------------------------------------------------


class TestCreateDictRule:
    def test_basic(self):
        out = Output()
        pos, key = create_dict_rule("[abc]", 0, out)
        assert key == ("abc",)
        assert pos == 5
        assert out.data.dict == {"abc": {}}

    def test_ws(self):
        out = Output()
        pos, key = create_dict_rule("[ a . b ]", 0, out)
        assert key == ("a", "b")
        assert out.data.dict == {"a": {"b": {}}}

    def test_missing_bracket(self):
        out = Output()
        with pytest.raises(TOMLDecodeError):
            create_dict_rule("[abc", 0, out)

    def test_declared_twice(self):
        out = Output()
        create_dict_rule("[abc]", 0, out)
        with pytest.raises(TOMLDecodeError):
            create_dict_rule("[abc]", 0, out)

    def test_overwrite_value(self):
        out = Output()
        out.data.dict["abc"] = 5
        with pytest.raises(TOMLDecodeError):
            create_dict_rule("[abc]", 0, out)


class TestCreateListRule:
    def test_basic(self):
        out = Output()
        pos, key = create_list_rule("[[abc]]", 0, out)
        assert key == ("abc",)
        assert pos == 7
        assert out.data.dict == {"abc": [{}]}

    def test_nested_list(self):
        out = Output()
        create_list_rule("[[a]]", 0, out)
        _, key = create_list_rule("[[a.b]]", 0, out)
        assert key == ("a", "b")
        assert out.data.dict == {"a": [{"b": [{}]}]}

    def test_missing_bracket(self):
        out = Output()
        with pytest.raises(TOMLDecodeError):
            create_list_rule("[[abc]", 0, out)

    def test_overwrite_value(self):
        out = Output()
        out.data.dict["abc"] = 5
        with pytest.raises(TOMLDecodeError):
            create_list_rule("[[abc]]", 0, out)

    def test_frozen_namespace(self):
        out = Output()
        out.data.dict["abc"] = {"x": 1}
        out.flags.set(("abc",), Flags.FROZEN, recursive=False)
        with pytest.raises(TOMLDecodeError):
            create_list_rule("[[abc]]", 0, out)


# ---------------------------------------------------------------------------
# parse_key / parse_key_part
# ---------------------------------------------------------------------------


class TestParseKeyPart:
    def test_bare(self):
        assert parse_key_part("abc", 0) == (3, "abc")

    def test_bare_dashes(self):
        assert parse_key_part("a-b_c1", 0) == (6, "a-b_c1")

    def test_literal(self):
        assert parse_key_part("'x y'", 0) == (5, "x y")

    def test_basic(self):
        assert parse_key_part('"x y"', 0) == (5, "x y")

    def test_invalid(self):
        with pytest.raises(TOMLDecodeError):
            parse_key_part(".", 0)

    def test_eof(self):
        with pytest.raises(TOMLDecodeError):
            parse_key_part("", 0)


class TestParseKey:
    def test_single(self):
        assert parse_key("abc", 0) == (3, ("abc",))

    def test_dotted(self):
        assert parse_key("a.b.c", 0) == (5, ("a", "b", "c"))

    def test_dotted_ws(self):
        assert parse_key("a . b", 0) == (5, ("a", "b"))

    def test_trailing_ws(self):
        assert parse_key("a  ", 0) == (3, ("a",))

    def test_too_many_parts(self):
        # Build key with MAX_KEY_PARTS + 1 parts. MAX_KEY_PARTS default = 1000.
        key_str = ".".join(["a"] * (MAX_KEY_PARTS + 1))
        with pytest.raises(RecursionError):
            parse_key(key_str, 0)


# ---------------------------------------------------------------------------
# parse_one_line_basic_str / parse_literal_str / parse_basic_str
# ---------------------------------------------------------------------------


class TestParseOneLineBasicStr:
    def test_simple(self):
        assert parse_one_line_basic_str('"abc"', 0) == (5, "abc")

    def test_escape(self):
        assert parse_one_line_basic_str('"a\\nb"', 0) == (6, "a\nb")

    def test_unterminated(self):
        with pytest.raises(TOMLDecodeError):
            parse_one_line_basic_str('"abc', 0)


class TestParseLiteralStr:
    def test_simple(self):
        assert parse_literal_str("'abc'", 0) == (5, "abc")

    def test_backslashes_preserved(self):
        assert parse_literal_str("'a\\nb'", 0) == (6, "a\\nb")

    def test_unterminated(self):
        with pytest.raises(TOMLDecodeError):
            parse_literal_str("'abc", 0)


class TestParseBasicStr:
    def test_no_multiline(self):
        assert parse_basic_str("abc\"", 0, multiline=False) == (4, "abc")

    def test_empty(self):
        assert parse_basic_str('"', 0, multiline=False) == (1, "")

    def test_unterminated(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str("abc", 0, multiline=False)

    def test_illegal_char(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str("a\x01b\"", 0, multiline=False)

    def test_multiline(self):
        assert parse_basic_str('abc"""', 0, multiline=True) == (6, "abc")

    def test_multiline_quotes(self):
        # Single/double quotes inside multiline are allowed
        assert parse_basic_str('a"b"""', 0, multiline=True) == (6, 'a"b')

    def test_multiline_escape(self):
        result = parse_basic_str('a\\nb"""', 0, multiline=True)
        assert result == (7, "a\nb")


# ---------------------------------------------------------------------------
# parse_basic_str_escape / parse_hex_char
# ---------------------------------------------------------------------------


class TestParseBasicStrEscape:
    def test_basic_escape(self):
        assert parse_basic_str_escape("\\nxx", 0) == (2, "\n")

    def test_backslash(self):
        assert parse_basic_str_escape("\\\\xx", 0) == (2, "\\")

    def test_quote(self):
        assert parse_basic_str_escape('\\"xx', 0) == (2, '"')

    def test_hex_2(self):
        assert parse_basic_str_escape("\\x41xx", 0) == (4, "A")

    def test_hex_4(self):
        assert parse_basic_str_escape("\\u0041xx", 0) == (6, "A")

    def test_hex_8(self):
        assert parse_basic_str_escape("\\U00000041xx", 0) == (10, "A")

    def test_invalid_escape(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str_escape("\\z", 0)

    def test_multiline_line_continuation_ws(self):
        # "\\ " followed by newline should be line continuation
        result = parse_basic_str_escape("\\ \n  abc", 0, multiline=True)
        assert result == (6, "")

    def test_multiline_newline_continuation(self):
        result = parse_basic_str_escape("\\\nabc", 0, multiline=True)
        assert result == (5, "")

    def test_multiline_ws_then_eof(self):
        result = parse_basic_str_escape("\\ ", 0, multiline=True)
        assert result == (2, "")

    def test_multiline_ws_then_non_newline_error(self):
        with pytest.raises(TOMLDecodeError):
            parse_basic_str_escape("\\ x", 0, multiline=True)


class TestParseBasicStrEscapeMultiline:
    def test_delegates(self):
        assert parse_basic_str_escape_multiline("\\nxx", 0) == (2, "\n")


class TestParseHexChar:
    def test_2(self):
        assert parse_hex_char("41", 0, 2) == (2, "A")

    def test_4(self):
        assert parse_hex_char("0041", 0, 4) == (4, "A")

    def test_8(self):
        assert parse_hex_char("00000041", 0, 8) == (8, "A")

    def test_invalid_digit(self):
        with pytest.raises(TOMLDecodeError):
            parse_hex_char("zzzz", 0, 2)

    def test_too_short(self):
        with pytest.raises(TOMLDecodeError):
            parse_hex_char("4", 0, 2)

    def test_surrogate(self):
        with pytest.raises(TOMLDecodeError):
            parse_hex_char("D800", 0, 4)

    def test_out_of_range(self):
        with pytest.raises(TOMLDecodeError):
            parse_hex_char("110000", 0, 6)


# ---------------------------------------------------------------------------
# is_unicode_scalar_value
# ---------------------------------------------------------------------------


class TestIsUnicodeScalarValue:
    def test_range(self):
        assert is_unicode_scalar_value(0)
        assert is_unicode_scalar_value(0x41)
        assert is_unicode_scalar_value(0xD7FF)  # 55295
        assert not is_unicode_scalar_value(0xD800)  # 55296
        assert not is_unicode_scalar_value(0xDFFF)  # 57343
        assert is_unicode_scalar_value(0xE000)  # 57344
        assert is_unicode_scalar_value(0x10FFFF)  # 1114111
        assert not is_unicode_scalar_value(0x110000)
        assert not is_unicode_scalar_value(-1)


# ---------------------------------------------------------------------------
# parse_multiline_str
# ---------------------------------------------------------------------------


class TestParseMultilineStr:
    def test_literal(self):
        pos, s = parse_multiline_str("'''abc'''", 0, literal=True)
        assert pos == 9
        assert s == "abc"

    def test_literal_with_newline(self):
        pos, s = parse_multiline_str("'''\nabc'''", 0, literal=True)
        assert pos == 10
        assert s == "abc"

    def test_basic(self):
        pos, s = parse_multiline_str('"""abc"""', 0, literal=False)
        assert pos == 9
        assert s == "abc"

    def test_extra_quotes_one(self):
        pos, s = parse_multiline_str('"""abc""""', 0, literal=False)
        assert s == 'abc"'
        assert pos == 10

    def test_extra_quotes_two(self):
        pos, s = parse_multiline_str('"""abc"""""', 0, literal=False)
        assert s == 'abc""'
        assert pos == 11

    def test_empty(self):
        pos, s = parse_multiline_str('""""""', 0, literal=False)
        assert s == ""
        assert pos == 6

    def test_unterminated(self):
        with pytest.raises(TOMLDecodeError):
            parse_multiline_str('"""abc', 0, literal=False)


# ---------------------------------------------------------------------------
# parse_array
# ---------------------------------------------------------------------------


class TestParseArray:
    def test_empty(self):
        assert parse_array("[]", 0, float, 0) == (2, [])

    def test_one(self):
        assert parse_array("[1]", 0, float, 0) == (3, [1])

    def test_many(self):
        assert parse_array("[1, 2, 3]", 0, float, 0) == (9, [1, 2, 3])

    def test_trailing_comma(self):
        assert parse_array("[1, 2,]", 0, float, 0) == (7, [1, 2])

    def test_multiline(self):
        src = "[\n  1,\n  2\n]"
        pos, arr = parse_array(src, 0, float, 0)
        assert arr == [1, 2]
        assert pos == len(src)

    def test_comments(self):
        src = "[ # c\n 1, # c2\n 2 ]"
        pos, arr = parse_array(src, 0, float, 0)
        assert arr == [1, 2]

    def test_unclosed(self):
        with pytest.raises(TOMLDecodeError):
            parse_array("[1", 0, float, 0)

    def test_unclosed_no_comma(self):
        with pytest.raises(TOMLDecodeError):
            parse_array("[1 2]", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_inline_table
# ---------------------------------------------------------------------------


class TestParseInlineTable:
    def test_empty(self):
        assert parse_inline_table("{}", 0, float, 0) == (2, {})

    def test_one(self):
        assert parse_inline_table("{a = 1}", 0, float, 0) == (7, {"a": 1})

    def test_many(self):
        pos, d = parse_inline_table("{a = 1, b = 2}", 0, float, 0)
        assert d == {"a": 1, "b": 2}
        assert pos == 12

    def test_trailing_comma(self):
        pos, d = parse_inline_table("{a = 1,}", 0, float, 0)
        assert d == {"a": 1}
        assert pos == 8

    def test_nested(self):
        pos, d = parse_inline_table("{a.b = 1}", 0, float, 0)
        assert d == {"a": {"b": 1}}

    def test_duplicate_key(self):
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a = 1, a = 2}", 0, float, 0)

    def test_unclosed(self):
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a = 1", 0, float, 0)

    def test_unclosed_no_comma(self):
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a = 1 b = 2}", 0, float, 0)

    def test_frozen_after_nested(self):
        # a.b = {} then a.b.c = 1 should fail
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a = {}, a.b = 1}", 0, float, 0)

    def test_overwrite_value(self):
        with pytest.raises(TOMLDecodeError):
            parse_inline_table("{a.b = 1, a.b.c = 2}", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------


class TestParseValue:
    def test_basic_string(self):
        assert parse_value('"abc"', 0, float, 0) == (5, "abc")

    def test_multiline_basic(self):
        pos, v = parse_value('"""abc"""', 0, float, 0)
        assert v == "abc"

    def test_literal_string(self):
        assert parse_value("'abc'", 0, float, 0) == (5, "abc")

    def test_multiline_literal(self):
        pos, v = parse_value("'''abc'''", 0, float, 0)
        assert v == "abc"

    def test_true(self):
        assert parse_value("true", 0, float, 0) == (4, True)

    def test_false(self):
        assert parse_value("false", 0, float, 0) == (5, False)

    def test_array(self):
        assert parse_value("[1,2]", 0, float, 0) == (5, [1, 2])

    def test_inline_table(self):
        assert parse_value("{a=1}", 0, float, 0) == (5, {"a": 1})

    def test_integer(self):
        assert parse_value("42", 0, float, 0) == (2, 42)

    def test_integer_underscore(self):
        assert parse_value("4_2", 0, float, 0) == (3, 42)

    def test_hex_int(self):
        assert parse_value("0xFF", 0, float, 0) == (4, 255)

    def test_oct_int(self):
        assert parse_value("0o17", 0, float, 0) == (4, 15)

    def test_bin_int(self):
        assert parse_value("0b101", 0, float, 0) == (5, 5)

    def test_float(self):
        assert parse_value("1.5", 0, float, 0) == (3, 1.5)

    def test_date(self):
        pos, v = parse_value("2020-01-01", 0, float, 0)
        assert v == datetime.date(2020, 1, 1)
        assert pos == 10

    def test_datetime(self):
        pos, v = parse_value("2020-01-01T00:00:00Z", 0, float, 0)
        assert v == datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)

    def test_localtime(self):
        pos, v = parse_value("00:00:00", 0, float, 0)
        assert v == datetime.time(0, 0, 0)

    def test_inf(self):
        import math

        pos, v = parse_value("inf", 0, float, 0)
        assert v == math.inf
        assert pos == 3

    def test_neg_nan(self):
        import math

        pos, v = parse_value("-nan", 0, float, 0)
        assert math.isnan(v)
        assert pos == 4

    def test_parse_float_called(self):
        calls = []

        def pf(s):
            calls.append(s)
            return "F" + s

        pos, v = parse_value("1.5", 0, pf, 0)
        assert v == "F1.5"
        assert calls == ["1.5"]

    def test_invalid(self):
        with pytest.raises(TOMLDecodeError):
            parse_value("?", 0, float, 0)

    def test_empty(self):
        with pytest.raises(TOMLDecodeError):
            parse_value("", 0, float, 0)

    def test_max_nesting(self):
        with pytest.raises(RecursionError):
            parse_value("[", 0, float, MAX_INLINE_NESTING + 1)

    def test_invalid_datetime(self):
        with pytest.raises(TOMLDecodeError):
            parse_value("2020-13-01", 0, float, 0)


# ---------------------------------------------------------------------------
# parse_key_value_pair / key_value_rule
# ---------------------------------------------------------------------------


class TestParseKeyValuePair:
    def test_basic(self):
        pos, key, value = parse_key_value_pair("a = 1", 0, float, 0)
        assert key == ("a",)
        assert value == 1
        assert pos == 5

    def test_dotted(self):
        pos, key, value = parse_key_value_pair("a.b = 2", 0, float, 0)
        assert key == ("a", "b")
        assert value == 2

    def test_missing_equals(self):
        with pytest.raises(TOMLDecodeError):
            parse_key_value_pair("a 1", 0, float, 0)

    def test_eof(self):
        with pytest.raises(TOMLDecodeError):
            parse_key_value_pair("a", 0, float, 0)


class TestKeyValueRule:
    def test_basic(self):
        out = Output()
        pos = key_value_rule("a = 1", 0, out, (), float)
        assert pos == 5
        assert out.data.dict == {"a": 1}

    def test_dotted(self):
        out = Output()
        key_value_rule("a.b = 1", 0, out, (), float)
        assert out.data.dict == {"a": {"b": 1}}

    def test_dotted_creates_explicit_nest(self):
        out = Output()
        key_value_rule("a.b = 1", 0, out, (), float)
        assert out.flags.is_(("a",), Flags.EXPLICIT_NEST)

    def test_redefine_namespace(self):
        out = Output()
        with pytest.raises(TOMLDecodeError):
            key_value_rule("a.b = 1\n", 0, out, (), float)
        # Now setting a.b again in same doc via a new key should fail on "a"
        # Actually must add pending first by another parse.
        out2 = Output()
        key_value_rule("x.y = 1\n", 0, out2, (), float)
        with pytest.raises(TOMLDecodeError):
            key_value_rule("x.z = 2\n", 6, out2, (), float)

    def test_overwrite_value(self):
        out = Output()
        key_value_rule("a = 1", 0, out, (), float)
        with pytest.raises(TOMLDecodeError):
            key_value_rule("a = 2", 0, out, (), float)

    def test_frozen_parent(self):
        out = Output()
        out.flags.set(("a",), Flags.FROZEN, recursive=False)
        with pytest.raises(TOMLDecodeError):
            key_value_rule("a.b = 1", 0, out, (), float)

    def test_inline_table_marks_frozen(self):
        out = Output()
        key_value_rule("a = {}", 0, out, (), float)
        assert out.flags.is_(("a",), Flags.FROZEN)

    def test_inline_array_marks_frozen(self):
        out = Output()
        key_value_rule("a = []", 0, out, (), float)
        assert out.flags.is_(("a",), Flags.FROZEN)

    def test_with_header(self):
        out = Output()
        key_value_rule("b = 1", 0, out, ("a",), float)
        assert out.data.dict == {"a": {"b": 1}}

    def test_recursive_frozen_check(self):
        out = Output()
        key_value_rule("a = {}", 0, out, (), float)
        # a.x = 1 should fail because a is recursively frozen
        with pytest.raises(TOMLDecodeError):
            key_value_rule("a.x = 1", 0, out, (), float)


# ---------------------------------------------------------------------------
# make_safe_parse_float
# ---------------------------------------------------------------------------


class TestMakeSafeParseFloat:
    def test_float_identity(self):
        assert make_safe_parse_float(float) is float

    def test_custom_passthrough(self):
        def custom(s):
            return "X" + s

        safe = make_safe_parse_float(custom)
        assert safe("1.5") == "X1.5"

    def test_custom_list_raises(self):
        def bad(s):
            return []

        safe = make_safe_parse_float(bad)
        with pytest.raises(ValueError):
            safe("1.5")

    def test_custom_dict_raises(self):
        def bad(s):
            return {}

        safe = make_safe_parse_float(bad)
        with pytest.raises(ValueError):
            safe("1.5")

    def test_custom_tuple_ok(self):
        def ok(s):
            return (s,)

        safe = make_safe_parse_float(ok)
        assert safe("1.5") == ("1.5",)
