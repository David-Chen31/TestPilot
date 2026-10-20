"""Tests for sqlparse.filters.others."""

import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.others import (
    StripCommentsFilter,
    StripWhitespaceFilter,
    SpacesAroundOperatorsFilter,
    StripTrailingSemicolonFilter,
    SerializerUnicode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stmt(text):
    """Parse text into a statement via the public API to get realistic
    token structures."""
    import sqlparse
    return sqlparse.parse(text)[0]


# ---------------------------------------------------------------------------
# StripCommentsFilter
# ---------------------------------------------------------------------------

class TestStripCommentsFilter:
    def test_strips_single_line_comment(self):
        stmt = make_stmt("SELECT 1 -- foo\n")
        result = StripCommentsFilter().process(stmt)
        assert "foo" not in str(result)
        assert isinstance(result, sql.Statement)

    def test_strips_multiline_comment(self):
        stmt = make_stmt("SELECT /* comment */ 1")
        result = StripCommentsFilter().process(stmt)
        assert "comment" not in str(result)

    def test_strips_comment_middle_of_expression(self):
        # comment between two non-whitespace tokens, replaced by single space
        stmt = make_stmt("SELECT a/*x*/+b")
        result = StripCommentsFilter().process(stmt)
        assert "x" not in str(result)
        assert str(result) == "SELECT a +b"

    def test_preserves_sql_hint_multiline(self):
        stmt = make_stmt("SELECT /*+ HINT */ 1")
        result = StripCommentsFilter().process(stmt)
        assert "HINT" in str(result)

    def test_preserves_sql_hint_single(self):
        stmt = make_stmt("SELECT --+ HINT\n1")
        result = StripCommentsFilter().process(stmt)
        assert "HINT" in str(result)

    def test_comment_at_beginning_no_prev(self):
        stmt = make_stmt("-- leading\nSELECT 1")
        result = StripCommentsFilter().process(stmt)
        assert "leading" not in str(result)

    def test_comment_at_end(self):
        stmt = make_stmt("SELECT 1 -- trailing")
        result = StripCommentsFilter().process(stmt)
        assert "trailing" not in str(result)

    def test_comment_preserves_newline_in_token(self):
        # comment with trailing newline should preserve the newline.
        stmt = make_stmt("SELECT 1 -- c\n")
        result = StripCommentsFilter().process(stmt)
        # newline survives as whitespace newline token
        assert "\n" in str(result) or str(result).endswith("1")

    def test_comment_returns_statement(self):
        stmt = make_stmt("SELECT 1")
        assert StripCommentsFilter().process(stmt) is stmt

    def test_nested_group_comments(self):
        # Comment inside a parenthesis group is processed recursively
        stmt = make_stmt("SELECT (a /* inner */ + b) FROM t")
        result = StripCommentsFilter().process(stmt)
        assert "inner" not in str(result)

    def test_multiple_comments(self):
        stmt = make_stmt("SELECT /* a */ 1 /* b */")
        result = StripCommentsFilter().process(stmt)
        text = str(result)
        assert "a" not in text
        assert "b" not in text

    def test_comment_after_open_paren(self):
        # prev is '(' punctuation -> no whitespace inserted
        stmt = make_stmt("SELECT (/*x*/1)")
        result = StripCommentsFilter().process(stmt)
        assert "x" not in str(result)
        assert str(result) == "SELECT (1)"

    def test_comment_before_close_paren(self):
        # prev is non-whitespace (not '('), next is ')' -> whitespace inserted
        stmt = make_stmt("SELECT (1/*x*/)")
        result = StripCommentsFilter().process(stmt)
        assert "x" not in str(result)
        assert str(result) == "SELECT (1 )"


# ---------------------------------------------------------------------------
# StripWhitespaceFilter
# ---------------------------------------------------------------------------

class TestStripWhitespaceFilter:
    def test_returns_statement(self):
        stmt = make_stmt("SELECT 1")
        assert StripWhitespaceFilter().process(stmt) is stmt

    def test_collapses_leading_whitespace(self):
        stmt = make_stmt("   SELECT   1")
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_collapses_multiple_spaces(self):
        stmt = make_stmt("SELECT    1")
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_removes_trailing_whitespace_at_depth_zero(self):
        # The last token is popped, but since its value was already collapsed
        # to a single space earlier, str() shows a trailing single space.
        stmt = make_stmt("SELECT 1   ")
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == "SELECT 1 "

    def test_identifierlist_removes_newline_before_comma(self):
        stmt = make_stmt("SELECT a\n,b FROM t")
        result = StripWhitespaceFilter().process(stmt)
        assert "\n" not in str(result)

    def test_parenthesis_strips_inner_whitespace(self):
        stmt = make_stmt("SELECT (  1  )")
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == "SELECT (1)"

    def test_parenthesis_strips_group_trailing_whitespace(self):
        stmt = make_stmt("SELECT (a + b   )")
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == "SELECT (a + b)"

    def test_stripws_default_direct(self):
        tlist = sql.TokenList([
            sql.Token(T.Whitespace, " "),
            sql.Token(T.Whitespace, " "),
            sql.Token(T.Keyword, "SELECT"),
            sql.Token(T.Whitespace, " "),
        ])
        StripWhitespaceFilter._stripws_default(tlist)
        assert [t.value for t in tlist.tokens] == ["", "", "SELECT", " "]

    def test_process_depth_gt_zero_keeps_trailing_ws(self):
        # when called with depth > 0, trailing whitespace is not popped
        stmt = make_stmt("SELECT 1   ")
        StripWhitespaceFilter().process(stmt, depth=1)
        assert stmt.tokens[-1].is_whitespace


# ---------------------------------------------------------------------------
# SpacesAroundOperatorsFilter
# ---------------------------------------------------------------------------

class TestSpacesAroundOperatorsFilter:
    def test_returns_statement(self):
        stmt = make_stmt("SELECT a=1")
        assert SpacesAroundOperatorsFilter().process(stmt) is stmt

    def test_spaces_around_equals(self):
        stmt = make_stmt("SELECT a=1")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT a = 1"

    def test_spaces_around_comparison(self):
        stmt = make_stmt("SELECT a>b")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT a > b"

    def test_spaces_around_operator(self):
        stmt = make_stmt("SELECT a+1")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT a + 1"

    def test_already_spaced_unchanged(self):
        stmt = make_stmt("SELECT a = 1")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT a = 1"

    def test_multiple_operators(self):
        stmt = make_stmt("SELECT a=1+2")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT a = 1 + 2"

    def test_nested_parenthesis_operators(self):
        stmt = make_stmt("SELECT (a=1) FROM t")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT (a = 1) FROM t"

    def test_operator_at_start_no_prev_skip(self):
        # '-' here is tokenized as a negative-number operator but is preceded
        # by whitespace already; no space is inserted because prev is ws and
        # next (the number) is not whitespace... actually a space is only
        # inserted after the operator when next is not whitespace. The output
        # remains "SELECT -1" because '-' followed by '1' forms part of the
        # numeric/operator handling is skipped for unary.
        stmt = make_stmt("SELECT -1")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT -1"


# ---------------------------------------------------------------------------
# StripTrailingSemicolonFilter
# ---------------------------------------------------------------------------

class TestStripTrailingSemicolonFilter:
    def test_removes_trailing_semicolon(self):
        stmt = make_stmt("SELECT 1;")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_removes_trailing_semicolon_and_whitespace(self):
        # The while loop pops whitespace and semicolons until a non-ws,
        # non-semicolon token remains, so trailing space is fully removed.
        stmt = make_stmt("SELECT 1 ;   ")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_returns_statement(self):
        stmt = make_stmt("SELECT 1;")
        assert StripTrailingSemicolonFilter().process(stmt) is stmt

    def test_no_semicolon_unchanged(self):
        stmt = make_stmt("SELECT 1")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_multiple_trailing_semicolons(self):
        stmt = make_stmt("SELECT 1;;")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_empty_statement(self):
        stmt = sql.Statement([])
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result.tokens == []


# ---------------------------------------------------------------------------
# SerializerUnicode
# ---------------------------------------------------------------------------

class TestSerializerUnicode:
    def test_returns_string(self):
        stmt = make_stmt("SELECT 1")
        result = SerializerUnicode.process(stmt)
        assert isinstance(result, str)

    def test_strips_trailing_whitespace_per_line(self):
        stmt = make_stmt("SELECT 1   \nFROM t   ")
        result = SerializerUnicode.process(stmt)
        for line in result.split("\n"):
            assert line == line.rstrip()

    def test_simple_query(self):
        stmt = make_stmt("SELECT 1")
        assert SerializerUnicode.process(stmt) == "SELECT 1"

    def test_multiline_query(self):
        stmt = make_stmt("SELECT 1\nFROM t")
        result = SerializerUnicode.process(stmt)
        assert result == "SELECT 1\nFROM t"

    def test_preserves_newline_in_string_literal(self):
        stmt = make_stmt("SELECT 'a\nb'")
        result = SerializerUnicode.process(stmt)
        # newline inside quoted string is preserved (not split)
        assert "a\nb" in result

    def test_empty_statement(self):
        stmt = sql.Statement([])
        assert SerializerUnicode.process(stmt) == ""
