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


# ---------- helpers ----------

def parse_one(sql_text):
    from sqlparse import parse
    return parse(sql_text)[0]


def render(stmt):
    return str(stmt)


# ---------- StripCommentsFilter ----------

class TestStripCommentsFilter:
    def test_removes_single_line_comment(self):
        stmt = parse_one("SELECT a -- comment\nFROM t")
        result = StripCommentsFilter().process(stmt)
        assert "-- comment" not in render(result)
        assert "SELECT a" in render(result)
        assert "FROM t" in render(result)

    def test_removes_multiline_comment(self):
        stmt = parse_one("SELECT a /* hi there */ FROM t")
        result = StripCommentsFilter().process(stmt)
        assert "/* hi there */" not in render(result)

    def test_removes_comment_at_end(self):
        stmt = parse_one("SELECT 1 -- trailing")
        result = StripCommentsFilter().process(stmt)
        assert "-- trailing" not in render(result)

    def test_preserves_sql_hint_multiline(self):
        stmt = parse_one("SELECT /*+ INDEX(t) */ a FROM t")
        result = StripCommentsFilter().process(stmt)
        assert "/*+ INDEX(t) */" in render(result)

    def test_preserves_sql_hint_single(self):
        stmt = parse_one("SELECT a --+ hint\nFROM t")
        result = StripCommentsFilter().process(stmt)
        assert "--+ hint" in render(result)

    def test_returns_same_stmt(self):
        stmt = parse_one("SELECT a -- c\nFROM t")
        result = StripCommentsFilter().process(stmt)
        assert result is stmt

    def test_comment_removed_replaced_with_space_between_tokens(self):
        stmt = parse_one("SELECT a/*x*/FROM t")
        result = StripCommentsFilter().process(stmt)
        out = render(result)
        assert "/*x*/" not in out
        # whitespace replacement keeps tokens separated
        assert "a" in out and "FROM" in out

    def test_comment_in_subquery(self):
        stmt = parse_one("SELECT (SELECT a -- inner\n) FROM t")
        result = StripCommentsFilter().process(stmt)
        assert "-- inner" not in render(result)

    def test_no_comment_noop(self):
        stmt = parse_one("SELECT a FROM t")
        before = render(stmt)
        result = StripCommentsFilter().process(stmt)
        assert render(result) == before


# ---------- StripWhitespaceFilter ----------

class TestStripWhitespaceFilter:
    def test_leading_trailing_whitespace_stripped(self):
        stmt = parse_one("   SELECT a   ")
        result = StripWhitespaceFilter().process(stmt)
        out = render(result)
        assert out == "SELECT a"

    def test_collapses_multiple_whitespace(self):
        stmt = parse_one("SELECT     a    FROM    t")
        result = StripWhitespaceFilter().process(stmt)
        out = render(result)
        assert "  " not in out
        assert out == "SELECT a FROM t"

    def test_returns_same_stmt(self):
        stmt = parse_one("SELECT    a")
        result = StripWhitespaceFilter().process(stmt)
        assert result is stmt

    def test_removes_trailing_whitespace_at_depth_zero(self):
        stmt = parse_one("SELECT a FROM t   ")
        result = StripWhitespaceFilter().process(stmt)
        assert not stmt.tokens[-1].is_whitespace

    def test_newline_before_comma_in_identifier_list(self):
        stmt = parse_one("SELECT a\n, b FROM t")
        result = StripWhitespaceFilter().process(stmt)
        out = render(result)
        assert "\n," not in out
        assert "a, b" in out or "a , b" not in out

    def test_parenthesis_whitespace_removed(self):
        stmt = parse_one("SELECT (   a   ) FROM t")
        result = StripWhitespaceFilter().process(stmt)
        out = render(result)
        assert "(   " not in out
        assert "   )" not in out
        assert "a" in out

    def test_no_tokens(self):
        stmt = parse_one("")
        # should not raise
        result = StripWhitespaceFilter().process(stmt)
        assert render(result) == ""

    def test_default_stripws_static(self):
        tlist = sql.TokenList([
            sql.Token(T.Whitespace, '   '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, '   '),
            sql.Token(T.Name, 'b'),
            sql.Token(T.Whitespace, '   '),
        ])
        StripWhitespaceFilter._stripws_default(tlist)
        values = [t.value for t in tlist.tokens]
        assert values == ['', 'a', ' ', 'b', ' ']


# ---------- SpacesAroundOperatorsFilter ----------

class TestSpacesAroundOperatorsFilter:
    def test_spaces_around_operator(self):
        stmt = parse_one("SELECT a+b FROM t")
        result = SpacesAroundOperatorsFilter().process(stmt)
        out = render(result)
        assert "a + b" in out

    def test_spaces_around_comparison(self):
        stmt = parse_one("SELECT a=b FROM t")
        result = SpacesAroundOperatorsFilter().process(stmt)
        out = render(result)
        assert "a = b" in out

    def test_existing_spaces_unchanged(self):
        stmt = parse_one("SELECT a + b FROM t")
        result = SpacesAroundOperatorsFilter().process(stmt)
        out = render(result)
        assert "a + b" in out
        # no double spaces introduced
        assert "a  +" not in out

    def test_returns_same_stmt(self):
        stmt = parse_one("SELECT a+b")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert result is stmt

    def test_multiple_operators(self):
        stmt = parse_one("SELECT a+b*c FROM t")
        result = SpacesAroundOperatorsFilter().process(stmt)
        out = render(result)
        assert "a + b * c" in out

    def test_operator_in_subquery(self):
        stmt = parse_one("SELECT * FROM (SELECT a+b FROM t) x")
        result = SpacesAroundOperatorsFilter().process(stmt)
        out = render(result)
        assert "a + b" in out

    def test_no_operator_noop(self):
        stmt = parse_one("SELECT a FROM t")
        before = render(stmt)
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert render(result) == before

    def test_operator_at_end(self):
        # trailing operator: next_ is None; shouldn't crash
        tlist = sql.TokenList([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Operator, '+'),
        ])
        SpacesAroundOperatorsFilter._process(tlist)
        values = [t.value for t in tlist.tokens]
        # space inserted before operator only
        assert values[-1] == '+'
        assert ' ' in values


# ---------- StripTrailingSemicolonFilter ----------

class TestStripTrailingSemicolonFilter:
    def test_strips_trailing_semicolon(self):
        stmt = parse_one("SELECT a FROM t;")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert render(result) == "SELECT a FROM t"

    def test_strips_semicolon_and_whitespace(self):
        stmt = parse_one("SELECT a FROM t;   ")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert render(result) == "SELECT a FROM t"

    def test_no_semicolon_noop(self):
        stmt = parse_one("SELECT a FROM t")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert render(result) == "SELECT a FROM t"

    def test_strips_only_last_semicolon(self):
        # "SELECT 1;;" -> one semicolon remains in middle, last removed
        stmt = parse_one("SELECT 1;;")
        result = StripTrailingSemicolonFilter().process(stmt)
        out = render(result)
        assert not out.endswith(";;")

    def test_returns_same_stmt(self):
        stmt = parse_one("SELECT 1;")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result is stmt

    def test_empty_statement(self):
        stmt = parse_one("")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert render(result) == ""

    def test_only_semicolon(self):
        stmt = parse_one(";")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert render(result) == ""


# ---------- SerializerUnicode ----------

class TestSerializerUnicode:
    def test_basic_serialization(self):
        stmt = parse_one("SELECT a FROM t")
        out = SerializerUnicode.process(stmt)
        assert out == "SELECT a FROM t"

    def test_strips_trailing_whitespace_per_line(self):
        # Build a statement containing trailing whitespace on a line
        stmt = parse_one("SELECT a   \nFROM t   ")
        out = SerializerUnicode.process(stmt)
        for line in out.split("\n"):
            assert line == line.rstrip()

    def test_preserves_newlines_between_lines(self):
        stmt = parse_one("SELECT a\nFROM t")
        out = SerializerUnicode.process(stmt)
        assert "\n" in out
        assert out == "SELECT a\nFROM t"

    def test_quoted_newline_not_split(self):
        # A quoted string containing CR/LF should not be split.
        stmt = parse_one("SELECT 'a\nb' FROM t")
        out = SerializerUnicode.process(stmt)
        assert "'a\nb'" in out

    def test_output_is_str(self):
        stmt = parse_one("SELECT 1")
        out = SerializerUnicode.process(stmt)
        assert isinstance(out, str)

    def test_empty(self):
        stmt = parse_one("")
        out = SerializerUnicode.process(stmt)
        assert out == ""
