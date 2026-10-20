"""Unit tests for sqlparse.engine.grouping."""

import pytest

import sqlparse
from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.exceptions import SQLParseError
from sqlparse.engine import grouping
from sqlparse.engine.grouping import (
    group_brackets,
    group_parenthesis,
    group_case,
    group_if,
    group_for,
    group_begin,
    group_typecasts,
    group_tzcasts,
    group_typed_literal,
    group_period,
    group_as,
    group_assignment,
    group_comparison,
    group_identifier,
    group_over,
    group_arrays,
    group_operator,
    group_identifier_list,
    group_comments,
    group_where,
    group_aliased,
    group_functions,
    group_order,
    align_comments,
    group_values,
    group,
    _group,
    _group_matching,
)


def make_tlist(text):
    """Parse text into a flat-ish TokenList suitable for grouping tests."""
    return sql.TokenList(sqlparse.parse(text)[0].tokens)


def values(tlist):
    return [t.value for t in tlist.tokens if not t.is_whitespace]


# ---------------------------------------------------------------------------
# _group_matching and the simple matching group functions
# ---------------------------------------------------------------------------

def test_group_brackets_groups_square_brackets():
    tl = make_tlist("[foo]")
    group_brackets(tl)
    groups = [t for t in tl.tokens if isinstance(t, sql.SquareBrackets)]
    assert len(groups) == 1
    assert groups[0].value == "[foo]"


def test_group_brackets_unbalanced_ignored():
    tl = make_tlist("]foo]")
    group_brackets(tl)
    assert not any(isinstance(t, sql.SquareBrackets) for t in tl.tokens)


def test_group_parenthesis_groups():
    tl = make_tlist("(a + b)")
    group_parenthesis(tl)
    groups = [t for t in tl.tokens if isinstance(t, sql.Parenthesis)]
    assert len(groups) == 1
    assert groups[0].value == "(a + b)"


def test_group_parenthesis_nested():
    tl = make_tlist("((x))")
    group_parenthesis(tl)
    groups = [t for t in tl.tokens if isinstance(t, sql.Parenthesis)]
    assert len(groups) == 1
    outer = groups[0]
    inner = [t for t in outer.tokens if isinstance(t, sql.Parenthesis)]
    assert len(inner) == 1
    assert inner[0].value == "(x)"


def test_group_parenthesis_extra_close_ignored():
    tl = make_tlist("a) (b)")
    group_parenthesis(tl)
    groups = [t for t in tl.tokens if isinstance(t, sql.Parenthesis)]
    assert [g.value for g in groups] == ["(b)"]


def test_group_case():
    tl = make_tlist("CASE WHEN a THEN b END")
    group_case(tl)
    groups = [t for t in tl.tokens if isinstance(t, sql.Case)]
    assert len(groups) == 1
    assert groups[0].value == "CASE WHEN a THEN b END"


def test_group_if():
    tl = make_tlist("IF a THEN b END")
    group_if(tl)
    # sql.If tokens: "IF ... END"
    groups = [t for t in tl.tokens if isinstance(t, sql.If)]
    # At minimum, verify no crash and value preserved if grouped.
    assert "".join(t.value for t in tl.tokens) == "IF a THEN b END"


def test_group_for():
    tl = make_tlist("FOR x IN y")
    group_for(tl)
    # Should not raise; tokens preserved
    assert "".join(t.value for t in tl.tokens) == "FOR x IN y"


def test_group_begin():
    tl = make_tlist("BEGIN x END")
    group_begin(tl)
    groups = [t for t in tl.tokens if isinstance(t, sql.Begin)]
    assert len(groups) == 1


def test_group_matching_depth_limit(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", 0)
    tl = make_tlist("(a)")
    outer = sql.TokenList(
        [sql.Parenthesis(tokens=[sql.Token(T.Text, "("),
                                 sql.Token(T.Text, "x"),
                                 sql.Token(T.Text, ")")])]
    )
    with pytest.raises(SQLParseError):
        _group_matching(outer, sql.Parenthesis, depth=1)


def test_group_matching_token_limit(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 2)
    tl = make_tlist("a b c d")
    with pytest.raises(SQLParseError):
        _group_matching(tl, sql.Parenthesis)


# ---------------------------------------------------------------------------
# _group depth/token limits
# ---------------------------------------------------------------------------

def test_group_depth_limit(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", 0)
    tl = make_tlist("a :: b")
    with pytest.raises(SQLParseError):
        _group(tl, sql.Identifier, lambda t: False, depth=1)


def test_group_token_limit(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 2)
    tl = make_tlist("a b c d")
    with pytest.raises(SQLParseError):
        _group(tl, sql.Identifier, lambda t: False)


# ---------------------------------------------------------------------------
# group_typecasts
# ---------------------------------------------------------------------------

def test_group_typecasts():
    tl = make_tlist("1 :: int")
    group_typecasts(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert idents[0].value == "1 :: int"


def test_group_typecasts_no_cast():
    tl = make_tlist("foo bar")
    group_typecasts(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_tzcasts
# ---------------------------------------------------------------------------

def test_group_tzcasts():
    # Handcraft a token with TZCast ttype
    tokens = [
        sql.Token(T.Number.Integer, "1"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Keyword.TZCast, "AT TIME ZONE"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.String.Single, "'UTC'"),
    ]
    tl = sql.TokenList(tokens)
    group_tzcasts(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert "AT TIME ZONE" in idents[0].value


# ---------------------------------------------------------------------------
# group_typed_literal
# ---------------------------------------------------------------------------

def test_group_typed_literal():
    stmt = sqlparse.parse("DATE '2020-01-01'")[0]
    group_typed_literal(stmt)
    tls = [t for t in stmt.tokens if isinstance(t, sql.TypedLiteral)]
    assert len(tls) == 1
    assert tls[0].value == "DATE '2020-01-01'"


# ---------------------------------------------------------------------------
# group_period / group_arrays
# ---------------------------------------------------------------------------

def test_group_period():
    tl = sql.TokenList([
        sql.Token(T.Name, "a"),
        sql.Token(T.Punctuation, "."),
        sql.Token(T.Name, "b"),
    ])
    group_period(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert idents[0].value == "a.b"


def test_group_period_arrow():
    tl = sql.TokenList([
        sql.Token(T.Name, "a"),
        sql.Token(T.Operator, "->"),
        sql.Token(T.Name, "b"),
    ])
    group_period(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert idents[0].value == "a->b"


def test_group_arrays():
    tl = sql.TokenList([
        sql.Token(T.Name, "arr"),
        sql.SquareBrackets(tokens=[
            sql.Token(T.Punctuation, "["),
            sql.Token(T.Number.Integer, "1"),
            sql.Token(T.Punctuation, "]"),
        ]),
    ])
    group_arrays(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert idents[0].value == "arr[1]"


# ---------------------------------------------------------------------------
# group_as
# ---------------------------------------------------------------------------

def test_group_as():
    tl = sql.TokenList([
        sql.Token(T.Name, "foo"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Keyword, "AS"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Name, "bar"),
    ])
    group_as(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert idents[0].value == "foo AS bar"


def test_group_as_null_prev():
    tl = sql.TokenList([
        sql.Token(T.Keyword, "NULL"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Keyword, "AS"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Name, "bar"),
    ])
    group_as(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1


# ---------------------------------------------------------------------------
# group_assignment
# ---------------------------------------------------------------------------

def test_group_assignment():
    tl = sql.TokenList([
        sql.Token(T.Name, "x"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Assignment, ":="),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Number.Integer, "1"),
    ])
    group_assignment(tl)
    assigns = [t for t in tl.tokens if isinstance(t, sql.Assignment)]
    assert len(assigns) == 1
    assert assigns[0].value == "x := 1"


def test_group_assignment_semicolon_terminated():
    tl = sql.TokenList([
        sql.Token(T.Name, "x"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Assignment, ":="),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Name, "a"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Name, "b"),
        sql.Token(T.Punctuation, ";"),
    ])
    group_assignment(tl)
    assigns = [t for t in tl.tokens if isinstance(t, sql.Assignment)]
    assert len(assigns) == 1
    assert assigns[0].value.endswith(";")


# ---------------------------------------------------------------------------
# group_comparison
# ---------------------------------------------------------------------------

def test_group_comparison():
    tl = sql.TokenList([
        sql.Token(T.Name, "a"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Operator.Comparison, "="),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Number.Integer, "1"),
    ])
    group_comparison(tl)
    cmps = [t for t in tl.tokens if isinstance(t, sql.Comparison)]
    assert len(cmps) == 1
    assert cmps[0].value == "a = 1"


def test_group_comparison_null_keyword():
    tl = sql.TokenList([
        sql.Token(T.Name, "a"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Operator.Comparison, "="),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Keyword, "NULL"),
    ])
    group_comparison(tl)
    cmps = [t for t in tl.tokens if isinstance(t, sql.Comparison)]
    assert len(cmps) == 1


# ---------------------------------------------------------------------------
# group_operator
# ---------------------------------------------------------------------------

def test_group_operator():
    tl = sql.TokenList([
        sql.Token(T.Number.Integer, "1"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Operator, "+"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Number.Integer, "2"),
    ])
    group_operator(tl)
    ops = [t for t in tl.tokens if isinstance(t, sql.Operation)]
    # 1 + 2 might be grouped into an Operation.
    assert ops
    assert ops[0].value == "1 + 2"


# ---------------------------------------------------------------------------
# group_identifier
# ---------------------------------------------------------------------------

def test_group_identifier():
    tl = sql.TokenList([
        sql.Token(T.Name, "foo"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.String.Symbol, '"bar"'),
    ])
    group_identifier(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 2
    assert idents[0].value == "foo"
    assert idents[1].value == '"bar"'


# ---------------------------------------------------------------------------
# group_over
# ---------------------------------------------------------------------------

def test_group_over():
    over = sql.Over(
        parents=None,
        tokens=[
            sql.Token(T.Keyword, "OVER"),
            sql.Token(T.Whitespace, " "),
        ],
    )
    paren = sql.Parenthesis(tokens=[
        sql.Token(T.Punctuation, "("),
        sql.Token(T.Wildcard, "*"),
        sql.Token(T.Punctuation, ")"),
    ])
    tl = sql.TokenList([sql.Token(T.Name, "row_number"), over, paren])
    group_over(tl)
    overs = [t for t in tl.tokens if isinstance(t, sql.Over)]
    assert len(overs) == 1
    assert "(*)" in overs[0].value


# ---------------------------------------------------------------------------
# group_identifier_list
# ---------------------------------------------------------------------------

def test_group_identifier_list():
    tl = sql.TokenList([
        sql.Token(T.Name, "a"),
        sql.Token(T.Punctuation, ","),
        sql.Token(T.Name, "b"),
    ])
    group_identifier_list(tl)
    ilists = [t for t in tl.tokens if isinstance(t, sql.IdentifierList)]
    assert len(ilists) == 1
    assert ilists[0].value == "a,b"


# ---------------------------------------------------------------------------
# group_comments
# ---------------------------------------------------------------------------

def test_group_comments_basic():
    tl = sql.TokenList([
        sql.Token(T.Comment.Single, "-- hi"),
        sql.Token(T.Newline, "\n"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Name, "x"),
    ])
    group_comments(tl)
    comments = [t for t in tl.tokens if isinstance(t, sql.Comment)]
    assert len(comments) == 1
    assert "-- hi" in comments[0].value


def test_group_comments_only_trailing():
    # A lone trailing comment: token_not_matching returns None -> break.
    tl = sql.TokenList([sql.Token(T.Comment.Single, "-- hi")])
    group_comments(tl)
    assert not any(isinstance(t, sql.Comment) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_where
# ---------------------------------------------------------------------------

def test_group_where():
    stmt = sqlparse.parse("SELECT * FROM t WHERE a = 1")[0]
    # Manually build a Where-less list with keyword tokens.
    group_where(stmt)
    wheres = [t for t in stmt.tokens if isinstance(t, sql.Where)]
    assert len(wheres) == 1
    assert wheres[0].value.startswith("WHERE")


# ---------------------------------------------------------------------------
# group_aliased
# ---------------------------------------------------------------------------

def test_group_aliased():
    ident = sql.Identifier(tokens=[sql.Token(T.Name, "x")])
    tl = sql.TokenList([
        ident,
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Number.Integer, "1"),
    ])
    group_aliased(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert "1" in idents[0].value


# ---------------------------------------------------------------------------
# group_functions
# ---------------------------------------------------------------------------

def test_group_functions():
    tl = sql.TokenList([
        sql.Token(T.Name, "count"),
        sql.Parenthesis(tokens=[
            sql.Token(T.Punctuation, "("),
            sql.Token(T.Wildcard, "*"),
            sql.Token(T.Punctuation, ")"),
        ]),
    ])
    group_functions(tl)
    funcs = [t for t in tl.tokens if isinstance(t, sql.Function)]
    assert len(funcs) == 1
    assert funcs[0].value == "count(*)"


def test_group_functions_skips_create_table_no_as():
    tl = sql.TokenList([
        sql.Token(T.DDL, "CREATE"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Keyword, "TABLE"),
        sql.Token(T.Whitespace, " "),
        sql.Token(T.Name, "foo"),
        sql.Parenthesis(tokens=[
            sql.Token(T.Punctuation, "("),
            sql.Token(T.Punctuation, ")"),
        ]),
    ])
    group_functions(tl)
    assert not any(isinstance(t, sql.Function) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_order
# ---------------------------------------------------------------------------

def test_group_order():
    stmt = sqlparse.parse("SELECT a FROM t ORDER BY x DESC")[0]
    group_identifier(stmt)
    group_order(stmt)
    order_idents = [
        t for t in stmt.tokens
        if isinstance(t, sql.Identifier) and "DESC" in t.value
    ]
    assert order_idents


# ---------------------------------------------------------------------------
# align_comments
# ---------------------------------------------------------------------------

def test_align_comments():
    comment = sql.Comment(tokens=[sql.Token(T.Comment.Single, "-- hi")])
    ident = sql.Identifier(tokens=[sql.Token(T.Name, "x")])
    tl = sql.TokenList([ident, sql.Token(T.Whitespace, " "), comment])
    align_comments(tl)
    lists = [t for t in tl.tokens if isinstance(t, sql.TokenList)]
    assert lists
    assert any(isinstance(t, sql.Identifier) for t in lists[0].tokens)


# ---------------------------------------------------------------------------
# group_values
# ---------------------------------------------------------------------------

def test_group_values():
    tl = sql.TokenList([
        sql.Token(T.Keyword, "VALUES"),
        sql.Token(T.Whitespace, " "),
        sql.Parenthesis(tokens=[
            sql.Token(T.Punctuation, "("),
            sql.Token(T.Number.Integer, "1"),
            sql.Token(T.Punctuation, ")"),
        ]),
    ])
    group_values(tl)
    vals = [t for t in tl.tokens if isinstance(t, sql.Values)]
    assert len(vals) == 1
    assert vals[0].value.startswith("VALUES")


def test_group_values_no_parens():
    tl = sql.TokenList([sql.Token(T.Keyword, "VALUES")])
    group_values(tl)
    assert not any(isinstance(t, sql.Values) for t in tl.tokens)


# ---------------------------------------------------------------------------
# top-level group()
# ---------------------------------------------------------------------------

def test_group_returns_same_statement():
    stmt = sqlparse.parse("SELECT a, b FROM t WHERE a = 1")[0]
    result = group(stmt)
    assert result is stmt


def test_group_full_pipeline():
    stmt = sqlparse.parse("SELECT count(*) FROM t")[0]
    result = group(stmt)
    # Functions should be grouped
    assert any(isinstance(t, sql.Function) for t in result.flatten())
</root>
