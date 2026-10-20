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
# StripCommentsFilter
# ---------------------------------------------------------------------------

def test_stripcomments_removes_simple_comment():
    stmt = sql.Statement(
        [sql.Token(T.Keyword, "SELECT"),
         sql.Token(T.Whitespace, " "),
         sql.Token(T.Comment.Single, "-- hi"),
         sql.Token(T.Whitespace, " "),
         sql.Token(T.Number.Integer, "1")]
    )
    result = StripCommentsFilter().process(stmt)
    assert result is stmt
    assert str(result) == "SELECT 1"


def test_stripcomments_replaces_with_space_between_tokens():
    stmt = sql.Statement(
        [sql.Token(T.Wildcard, "*"),
         sql.Token(T.Comment.Single, "-- c"),
         sql.Token(T.Wildcard, "*")]
    )
    StripCommentsFilter().process(stmt)
    # No whitespace around -> replaced by a single space
    assert str(stmt) == "* *"


def test_stripcomments_preserves_newline_in_comment():
    stmt = sql.Statement(
        [sql.Token(T.Keyword, "SELECT"),
         sql.Token(T.Whitespace, " "),
         sql.Token(T.Comment.Multiline, "/* foo\n*/"),
         sql.Token(T.Whitespace, " "),
         sql.Token(T.Number.Integer, "1")]
    )
    StripCommentsFilter().process(stmt)
    assert str(stmt) == "SELECT \n 1"


def test_stripcomments_skips_hint_token():
    stmt = sql.Statement(
        [sql.Token(T.Comment.Single.Hint, "--+ hint"),
         sql.Token(T.Whitespace, " "),
         sql.Token(T.Keyword, "SELECT")]
    )
    StripCommentsFilter().process(stmt)
    assert "--+ hint" in str(stmt)
    assert "SELECT" in str(stmt)


def test_stripcomments_skips_comment_group_with_hint():
    inner = sql.Token(T.Comment.Single.Hint, "--+ hint")
    comment_group = sql.Comment([inner])
    stmt = sql.Statement([comment_group, sql.Token(T.Whitespace, " "),
                          sql.Token(T.Keyword, "SELECT")])
    StripCommentsFilter().process(stmt)
    # The hint comment group must remain
    assert any(isinstance(t, sql.Comment) for t in stmt.tokens)


def test_stripcomments_handles_empty_comment_group():
    comment_group = sql.Comment([])
    stmt = sql.Statement([comment_group,
                          sql.Token(T.Whitespace, " "),
                          sql.Token(T.Keyword, "SELECT")])
    StripCommentsFilter().process(stmt)
    assert not any(isinstance(t, sql.Comment) for t in stmt.tokens)


def test_stripcomments_leader_comment_no_leading_ws_added():
    # comment first -> prev is None -> just removed
    stmt = sql.Statement(
        [sql.Token(T.Comment.Single, "-- leading"),
         sql.Token(T.Whitespace, " "),
         sql.Token(T.Keyword, "SELECT")]
    )
    StripCommentsFilter().process(stmt)
    assert str(stmt) == " SELECT"


def test_stripcomments_after_open_paren_no_ws_added():
    stmt = sql.Statement(
        [sql.Token(T.Punctuation, "("),
         sql.Token(T.Comment.Single, "-- c"),
         sql.Token(T.Number.Integer, "1"),
         sql.Token(T.Punctuation, ")")]
    )
    StripCommentsFilter().process(stmt)
    assert str(stmt) == "(1)"


def test_stripcomments_trailing_comment():
    stmt = sql.Statement(
        [sql.Token(T.Keyword, "SELECT"),
         sql.Token(T.Comment.Single, "-- trailing")]
    )
    StripCommentsFilter().process(stmt)
    assert str(stmt) == "SELECT"


def test_stripcomments_recurses_into_sublists():
    inner = sql.Identifier([
        sql.Token(T.Name, "a"),
        sql.Token(T.Comment.Single, "-- x"),
    ])
    stmt = sql.Statement([inner])
    StripCommentsFilter().process(stmt)
    assert str(stmt) == "a"


# ---------------------------------------------------------------------------
# StripWhitespaceFilter
# ---------------------------------------------------------------------------

def test_stripwhitespace_default_collapses_whitespace():
    stmt = sql.Statement(
        [sql.Token(T.Keyword, "SELECT"),
         sql.Token(T.Whitespace, "   "),
         sql.Token(T.Whitespace, "\t"),
         sql.Token(T.Name, "x")]
    )
    StripWhitespaceFilter().process(stmt)
    assert [t.value for t in stmt.tokens] == ["SELECT", "", "", "x"]


def test_stripwhitespace_leading_ws_removed():
    stmt = sql.Statement(
        [sql.Token(T.Whitespace, "   "),
         sql.Token(T.Keyword, "SELECT")]
    )
    StripWhitespaceFilter().process(stmt)
    # only leading ws -> becomes '' then trailing pop behavior
    assert str(stmt) == "SELECT"


def test_stripwhitespace_trailing_ws_popped_at_depth_0():
    stmt = sql.Statement(
        [sql.Token(T.Keyword, "SELECT"),
         sql.Token(T.Whitespace, "  ")]
    )
    StripWhitespaceFilter().process(stmt)
    assert len(stmt.tokens) == 1
    assert str(stmt) == "SELECT"


def test_stripwhitespace_walks_sublists():
    ident = sql.Identifier(
        [sql.Token(T.Name, "a"),
         sql.Token(T.Whitespace, "  "),
         sql.Token(T.Name, "b")]
    )
    stmt = sql.Statement([
        sql.Token(T.Keyword, "SELECT"),
        sql.Token(T.Whitespace, " "),
        ident,
    ])
    StripWhitespaceFilter().process(stmt)
    assert str(ident) == "a b"


def test_stripwhitespace_identifierlist_removes_newline_before_comma():
    il = sql.IdentifierList([
        sql.Token(T.Name, "a"),
        sql.Token(T.Whitespace.Newline, "\n"),
        sql.Token(T.Punctuation, ","),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Name, "b"),
    ])
    stmt = sql.Statement([
        sql.Token(T.Keyword, "SELECT"),
        sql.Token(T.Whitespace, " "),
        il,
    ])
    StripWhitespaceFilter().process(stmt)
    assert str(il) == "a, b"


def test_stripwhitespace_parenthesis_strips_inner_ws():
    paren = sql.Parenthesis([
        sql.Token(T.Punctuation, "("),
        sql.Token(T.Whitespace, "  "),
        sql.Token(T.Number.Integer, "1"),
        sql.Token(T.Whitespace, "  "),
        sql.Token(T.Punctuation, ")"),
    ])
    stmt = sql.Statement([paren])
    StripWhitespaceFilter().process(stmt)
    assert str(paren) == "(1)"


def test_stripwhitespace_parenthesis_strips_trailing_inside_group():
    inner = sql.Identifier([
        sql.Token(T.Name, "a"),
        sql.Token(T.Whitespace, "  "),
    ])
    paren = sql.Parenthesis([
        sql.Token(T.Punctuation, "("),
        inner,
        sql.Token(T.Punctuation, ")"),
    ])
    stmt = sql.Statement([paren])
    StripWhitespaceFilter().process(stmt)
    assert str(paren) == "(a)"


# ---------------------------------------------------------------------------
# SpacesAroundOperatorsFilter
# ---------------------------------------------------------------------------

def test_spacesaroundoperators_adds_spaces():
    stmt = sql.Statement([
        sql.Token(T.Name, "a"),
        sql.Token(T.Operator, "+"),
        sql.Token(T.Name, "b"),
    ])
    SpacesAroundOperatorsFilter().process(stmt)
    assert str(stmt) == "a + b"


def test_spacesaroundoperators_does_not_double_spaces():
    stmt = sql.Statement([
        sql.Token(T.Name, "a"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Comparison, "="),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Name, "b"),
    ])
    SpacesAroundOperatorsFilter().process(stmt)
    assert str(stmt) == "a = b"


def test_spacesaroundoperators_handles_missing_neighbors():
    # Operator at the very start / very end
    stmt = sql.Statement([
        sql.Token(T.Operator, "-"),
        sql.Token(T.Number.Integer, "1"),
    ])
    SpacesAroundOperatorsFilter().process(stmt)
    assert str(stmt) == "- 1"


def test_spacesaroundoperators_recurses():
    ident = sql.Identifier([
        sql.Token(T.Name, "a"),
        sql.Token(T.Operator, "+"),
        sql.Token(T.Name, "b"),
    ])
    stmt = sql.Statement([ident])
    SpacesAroundOperatorsFilter().process(stmt)
    assert str(ident) == "a + b"


# ---------------------------------------------------------------------------
# StripTrailingSemicolonFilter
# ---------------------------------------------------------------------------

def test_striptrailingsemicolon_removes_semicolon_and_ws():
    stmt = sql.Statement([
        sql.Token(T.Keyword, "SELECT"),
        sql.Token(T.Punctuation, ";"),
        sql.Token(T.Whitespace, "  "),
    ])
    result = StripTrailingSemicolonFilter().process(stmt)
    assert result is stmt
    assert str(stmt) == "SELECT"


def test_striptrailingsemicolon_multiple_semicolons():
    stmt = sql.Statement([
        sql.Token(T.Keyword, "SELECT"),
        sql.Token(T.Punctuation, ";"),
        sql.Token(T.Punctuation, ";"),
    ])
    StripTrailingSemicolonFilter().process(stmt)
    assert str(stmt) == "SELECT"


def test_striptrailingsemicolon_no_op_when_no_semicolon():
    stmt = sql.Statement([sql.Token(T.Keyword, "SELECT")])
    StripTrailingSemicolonFilter().process(stmt)
    assert str(stmt) == "SELECT"


def test_striptrailingsemicolon_empty_statement():
    stmt = sql.Statement([])
    StripTrailingSemicolonFilter().process(stmt)
    assert str(stmt) == ""


# ---------------------------------------------------------------------------
# SerializerUnicode
# ---------------------------------------------------------------------------

def test_serializerunicode_rstrips_trailing_per_line():
    stmt = sql.Statement([
        sql.Token(T.Keyword, "SELECT"),
        sql.Token(T.Whitespace, "   "),
        sql.Token(T.Whitespace.Newline, "\n"),
        sql.Token(T.Keyword, "FROM"),
        sql.Token(T.Whitespace, "  "),
    ])
    out = SerializerUnicode().process(stmt)
    assert out == "SELECT\nFROM"


def test_serializerunicode_simple():
    stmt = sql.Statement([sql.Token(T.Keyword, "SELECT")])
    assert SerializerUnicode().process(stmt) == "SELECT"


def test_serializerunicode_no_trailing_newline():
    stmt = sql.Statement([
        sql.Token(T.Keyword, "A"),
        sql.Token(T.Whitespace.Newline, "\n"),
        sql.Token(T.Keyword, "B"),
    ])
    out = SerializerUnicode().process(stmt)
    assert out == "A\nB"
    assert not out.endswith("\n")
