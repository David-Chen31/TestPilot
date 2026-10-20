import pytest

import sqlparse
from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.others import (
    StripCommentsFilter,
    StripWhitespaceFilter,
    SpacesAroundOperatorsFilter,
    StripTrailingSemicolonFilter,
    SerializerUnicode,
)


def parse_one(text):
    return sqlparse.parse(text)[0]


# ---------------------------------------------------------------------------
# StripCommentsFilter
# ---------------------------------------------------------------------------

class TestStripCommentsFilter:

    def test_single_line_comment_between_tokens(self):
        stmt = parse_one("SELECT 1 --comment\nFROM foo")
        result = StripCommentsFilter().process(stmt)
        assert "--comment" not in str(result)
        assert str(result) == "SELECT 1 \nFROM foo"

    def test_multiline_comment_between_tokens(self):
        stmt = parse_one("SELECT/* hint */1")
        result = StripCommentsFilter().process(stmt)
        assert "/* hint */" not in str(result)
        assert str(result) == "SELECT 1"

    def test_comment_after_token_creates_whitespace(self):
        stmt = parse_one("SELECT 1/*c*/2")
        result = StripCommentsFilter().process(stmt)
        assert str(result) == "SELECT 1 2"

    def test_comment_at_end_is_removed(self):
        stmt = parse_one("SELECT 1 -- trailing")
        result = StripCommentsFilter().process(stmt)
        assert str(result) == "SELECT 1 "

    def test_comment_at_start_removed_without_leading_space(self):
        stmt = parse_one("-- leading\nSELECT 1")
        result = StripCommentsFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_sql_hint_multiline_preserved(self):
        stmt = parse_one("SELECT /*+ INDEX(t) */ col FROM t")
        result = StripCommentsFilter().process(stmt)
        assert "/*+ INDEX(t) */" in str(result)

    def test_sql_hint_single_preserved(self):
        stmt = parse_one("SELECT 1 --+ hint\nFROM t")
        # ensure it's parsed as a hint comment
        comments = [t for t in stmt.flatten() if t.ttype in (
            T.Comment.Single, T.Comment.Single.Hint)]
        listed = [c for c in stmt.tokens if c.ttype is T.Comment.Single]
        assert listed, "expected a single-line comment token"
        result = StripCommentsFilter().process(stmt)
        # Hint should still be present
        assert "--+ hint" in str(result)

    def test_comment_group_preserved_for_hint(self):
        # A sql.Comment group whose first token is a hint should be kept.
        stmt = sql.Statement()
        comment = sql.Comment([
            sql.Token(T.Comment.Multiline.Hint, "/*+ x */"),
        ])
        stmt.tokens = [comment]
        result = StripCommentsFilter().process(stmt)
        assert len(result.tokens) == 1
        assert isinstance(result.tokens[0], sql.Comment)

    def test_comment_group_not_hint_removed(self):
        stmt = sql.Statement()
        word = sql.Token(T.Keyword, "SELECT")
        comment = sql.Comment([sql.Token(T.Comment.Multiline, "/*x*/")])
        word2 = sql.Token(T.Name, "foo")
        stmt.tokens = [word, comment, word2]
        result = StripCommentsFilter().process(stmt)
        assert "/*x*/" not in str(result)

    def test_comment_after_open_paren_no_space (self):
        stmt = parse_one("SELECT (/* c */1)")
        result = StripCommentsFilter().process(stmt)
        assert "/* c */" not in str(result)

    def test_comment_before_close_paren(self):
        stmt = parse_one("SELECT (1 /* c */)")
        result = StripCommentsFilter().process(stmt)
        assert "/* c */" not in str(result)

    def test_process_returns_same_stmt_object(self):
        stmt = parse_one("SELECT 1 --x\nFROM t")
        returned = StripCommentsFilter().process(stmt)
        assert returned is stmt

    def test_nested_comment_in_subquery(self):
        stmt = parse_one("SELECT * FROM (SELECT 1 --c\n) AS x")
        result = StripCommentsFilter().process(stmt)
        assert "--c" not in str(result)

    def test_no_comments_no_change(self):
        stmt = parse_one("SELECT 1 FROM foo")
        original = str(stmt)
        result = StripCommentsFilter().process(stmt)
        assert str(result) == original

    def test_get_insert_token_newline(self):
        # Directly exercise branch that preserves trailing newline
        token = sql.Token(T.Comment.Single, "--x\n")
        insert = StripCommentsFilter._process.__doc__  # noqa - silence linters
        # Access the internal helper by triggering its behavior via _process
        stmt = parse_one("SELECT 1 --x\nFROM t")
        result = StripCommentsFilter().process(stmt)
        assert "\n" in str(result)


# ---------------------------------------------------------------------------
# StripWhitespaceFilter
# ---------------------------------------------------------------------------

class TestStripWhitespaceFilter:

    def test_collapses_multiple_spaces(self):
        stmt = parse_one("SELECT    1")
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_removes_trailing_whitespace_at_depth_zero(self):
        stmt = parse_one("SELECT 1   ")
        result = StripWhitespaceFilter().process(stmt)
        assert not result.tokens[-1].is_whitespace
        assert str(result) == "SELECT 1"

    def test_no_change_single_spaces(self):
        stmt = parse_one("SELECT 1 FROM foo")
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == "SELECT 1 FROM foo"

    def test_process_returns_same_stmt(self):
        stmt = parse_one("SELECT  1")
        assert StripWhitespaceFilter().process(stmt) is stmt

    def test_identifier_list_newline_before_comma(self):
        stmt = parse_one("SELECT a\n, b FROM t")
        result = StripWhitespaceFilter().process(stmt)
        # newline before comma should be removed
        assert "\n," not in str(result)

    def test_parenthesis_strips_inner_whitespace(self):
        stmt = parse_one("SELECT ( 1 )")
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == "SELECT (1)"

    def test_parenthesis_with_subgroup_trailing_ws(self):
        stmt = parse_one("SELECT (a + b )")
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == "SELECT (a + b)"

    def test_stripws_default_via_helper(self):
        tlist = sql.TokenList()
        tlist.tokens = [
            sql.Token(T.Whitespace, "   "),
            sql.Token(T.Keyword, "SELECT"),
            sql.Token(T.Whitespace, "   "),
            sql.Token(T.Whitespace, "   "),
            sql.Token(T.Number.Integer, "1"),
        ]
        StripWhitespaceFilter._stripws_default(tlist)
        assert tlist.tokens[0].value == ""
        assert tlist.tokens[2].value == " "
        assert tlist.tokens[3].value == ""

    def test_stripws_identifierlist_removes_newline_before_comma(self):
        tlist = sql.IdentifierList()
        tlist.tokens = [
            sql.Token(T.Name, "a"),
            sql.Token(T.Whitespace.Newline, "\n"),
            sql.Token(T.Punctuation, ","),
            sql.Token(T.Name, " b"),
        ]
        flt = StripWhitespaceFilter()
        flt._stripws_identifierlist(tlist)
        values = [t.value for t in tlist.tokens]
        assert "\n" not in values


# ---------------------------------------------------------------------------
# SpacesAroundOperatorsFilter
# ---------------------------------------------------------------------------

class TestSpacesAroundOperatorsFilter:

    def test_adds_spaces_around_assignment(self):
        stmt = parse_one("SELECT 1=2")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT 1 = 2"

    def test_adds_spaces_around_comparison(self):
        stmt = parse_one("SELECT a<>b FROM t")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT a <> b FROM t"

    def test_no_change_when_spaces_present(self):
        stmt = parse_one("SELECT 1 = 2")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT 1 = 2"

    def test_process_returns_same_stmt(self):
        stmt = parse_one("SELECT 1=2")
        assert SpacesAroundOperatorsFilter().process(stmt) is stmt

    def test_multiple_operators(self):
        stmt = parse_one("SELECT a+b*c")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == "SELECT a + b * c"

    def test_operator_at_start_inserts_only_after(self):
        tlist = sql.TokenList()
        tlist.tokens = [
            sql.Token(T.Operator, "+"),
            sql.Token(T.Number.Integer, "1"),
        ]
        SpacesAroundOperatorsFilter._process(tlist)
        assert tlist.tokens[0].value == "+"
        assert tlist.tokens[1].ttype is T.Whitespace
        assert tlist.tokens[3].ttype is T.Number.Integer

    def test_operator_at_end_inserts_only_before(self):
        tlist = sql.TokenList()
        tlist.tokens = [
            sql.Token(T.Number.Integer, "1"),
            sql.Token(T.Operator, "+"),
        ]
        SpacesAroundOperatorsFilter._process(tlist)
        assert tlist.tokens[1].ttype is T.Whitespace
        assert tlist.tokens[-1].value == "+"


# ---------------------------------------------------------------------------
# StripTrailingSemicolonFilter
# ---------------------------------------------------------------------------

class TestStripTrailingSemicolonFilter:

    def test_removes_semicolon(self):
        stmt = parse_one("SELECT 1;")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_removes_semicolon_and_trailing_ws(self):
        stmt = parse_one("SELECT 1;   ")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_no_semicolon_same(self):
        stmt = parse_one("SELECT 1")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == "SELECT 1"

    def test_process_returns_same_stmt(self):
        stmt = parse_one("SELECT 1;")
        assert StripTrailingSemicolonFilter().process(stmt) is stmt

    def test_empty_statement(self):
        stmt = sql.Statement()
        stmt.tokens = []
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result.tokens == []

    def test_only_semicolon(self):
        stmt = sql.Statement()
        stmt.tokens = [sql.Token(T.Punctuation, ";")]
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result.tokens == []

    def test_multiple_trailing_semicolons(self):
        stmt = sql.Statement()
        stmt.tokens = [
            sql.Token(T.Keyword, "SELECT"),
            sql.Token(T.Punctuation, ";"),
            sql.Token(T.Punctuation, ";"),
        ]
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == "SELECT"


# ---------------------------------------------------------------------------
# SerializerUnicode
# ---------------------------------------------------------------------------

class TestSerializerUnicode:

    def test_basic_statement(self):
        stmt = parse_one("SELECT 1")
        assert SerializerUnicode.process(stmt) == "SELECT 1"

    def test_strips_trailing_whitespace_per_line(self):
        stmt = parse_one("SELECT 1   \nFROM t   ")
        assert SerializerUnicode.process(stmt) == "SELECT 1\nFROM t"

    def test_newline_within_string_preserved(self):
        stmt = parse_one("SELECT 'a\nb'")
        result = SerializerUnicode.process(stmt)
        # Newline inside the quoted string should not be split
        assert result == "SELECT 'a\nb'"

    def test_multiple_newlines_preserved(self):
        stmt = parse_one("SELECT 1\n\nFROM t")
        result = SerializerUnicode.process(stmt)
        # All quoted output should keep newlines; trailing ws trimmed
        assert result == "SELECT 1\n\nFROM t"

    def test_empty_statement(self):
        stmt = sql.Statement()
        stmt.tokens = []
        assert SerializerUnicode.process(stmt) == ""

    def test_only_whitespace_statement(self):
        stmt = sql.Statement()
        stmt.tokens = [sql.Token(T.Whitespace, "   ")]
        assert SerializerUnicode.process(stmt) == ""
