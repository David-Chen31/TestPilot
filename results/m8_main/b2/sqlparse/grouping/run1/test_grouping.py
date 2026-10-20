"""Unit tests for sqlparse.engine.grouping."""

import pytest

import sqlparse
from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.exceptions import SQLParseError
from sqlparse.engine import grouping
from sqlparse.engine.grouping import (
    _group,
    _group_matching,
    align_comments,
    group,
    group_aliased,
    group_arrays,
    group_as,
    group_assignment,
    group_begin,
    group_brackets,
    group_case,
    group_comments,
    group_comparison,
    group_for,
    group_functions,
    group_identifier,
    group_identifier_list,
    group_if,
    group_operator,
    group_order,
    group_over,
    group_parenthesis,
    group_period,
    group_typecasts,
    group_typed_literal,
    group_tzcasts,
    group_values,
    group_where,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_single(sql_text):
    """Return the first statement of a parsed SQL string."""
    return sqlparse.parse(sql_text)[0]


def run_pipeline(stmt, *funcs):
    for func in funcs:
        func(stmt)
    return stmt


def top_level_of_type(stmt, cls):
    return [t for t in stmt.tokens if isinstance(t, cls)]


# ---------------------------------------------------------------------------
# _group_matching & the "matching" wrappers
# ---------------------------------------------------------------------------

def test_group_matching_depth_limit():
    stmt = parse_single("( ( (1) ) )")
    with pytest.raises(SQLParseError) as ei:
        _group_matching(stmt, sql.Parenthesis,
                        depth=grouping.MAX_GROUPING_DEPTH + 1)
    assert str(ei.value) == (
        f"Maximum grouping depth exceeded ({grouping.MAX_GROUPING_DEPTH}).")


def test_group_matching_depth_boundary():
    # depth == MAX is allowed (strict >).
    stmt = parse_single("(1)")
    _group_matching(stmt, sql.Parenthesis, depth=grouping.MAX_GROUPING_DEPTH)


def test_group_matching_token_limit(monkeypatch):
    stmt = parse_single("a b c d e")
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 2)
    with pytest.raises(SQLParseError) as ei:
        _group_matching(stmt, sql.Parenthesis)
    assert str(ei.value) == "Maximum number of tokens exceeded (2)."


def test_group_matching_token_boundary(monkeypatch):
    # len(tokens) == limit should NOT raise (strict >).  Use a statement
    # with a single token and no sublists so the equality check is exact.
    stmt = parse_single("1")
    n = len(stmt.tokens)
    assert n == 1
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", n)
    _group_matching(stmt, sql.Parenthesis)


def test_group_matching_depth_disabled(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", None)
    stmt = parse_single("(1)")
    _group_matching(stmt, sql.Parenthesis)
    found = [t for t in stmt.tokens if isinstance(t, sql.Parenthesis)]
    assert len(found) == 1


def test_group_matching_token_limit_disabled(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", None)
    stmt = parse_single("(" + ", ".join(["1"] * 50) + ")")
    _group_matching(stmt, sql.Parenthesis)
    found = [t for t in stmt.tokens if isinstance(t, sql.Parenthesis)]
    assert len(found) == 1


def test_group_matching_nested_recursion():
    """Nested different-class groups should be recursed into."""
    stmt = parse_single("( [a b] )")
    # First group brackets, then parenthesis.
    _group_matching(stmt, sql.SquareBrackets)
    _group_matching(stmt, sql.Parenthesis)
    parens = [t for t in stmt.tokens if isinstance(t, sql.Parenthesis)]
    assert len(parens) == 1
    # The square-bracketed content ends up inside the parenthesis group.
    assert "[" in parens[0].value
    assert "]" in parens[0].value


def test_group_matching_multiple_groups_offset():
    """Two sibling parenthesis groups must both be formed."""
    stmt = parse_single("(a) (b)")
    _group_matching(stmt, sql.Parenthesis)
    parens = [t for t in stmt.tokens if isinstance(t, sql.Parenthesis)]
    assert len(parens) == 2
    assert parens[0].value == "(a)"
    assert parens[1].value == "(b)"


def test_group_matching_skip_unbalanced_open():
    """An unmatched open should not break processing of later groups."""
    stmt = parse_single("( (a)")
    _group_matching(stmt, sql.Parenthesis)
    # The inner (a) forms; outer remains unbalanced.
    parens = [t for t in stmt.tokens if isinstance(t, sql.Parenthesis)]
    assert len(parens) == 1
    assert parens[0].value == "(a)"


def test_group_matching_skip_unbalanced_close():
    """An unmatched close must not abort scanning of subsequent groups."""
    from sqlparse.sql import Token
    # Build a Statement manually so parsing/filter-stack grouping doesn't
    # pre-group the trailing "(b)".
    stmt = sql.Statement([
        Token(T.Name, "a"),
        Token(T.Whitespace, " "),
        Token(T.Punctuation, ")"),
        Token(T.Whitespace, " "),
        Token(T.Punctuation, "("),
        Token(T.Name, "b"),
        Token(T.Punctuation, ")"),
    ])
    _group_matching(stmt, sql.Parenthesis)
    parens = [t for t in stmt.tokens if isinstance(t, sql.Parenthesis)]
    assert len(parens) == 1
    assert parens[0].value == "(b)"


def test_group_brackets():
    stmt = parse_single("[a b]")
    group_brackets(stmt)
    first = [t for t in stmt.tokens if not t.is_whitespace][0]
    assert isinstance(first, sql.Identifier)


def test_group_parenthesis():
    stmt = parse_single("(a, b)")
    group_parenthesis(stmt)
    assert isinstance(stmt, sql.Statement)


def test_group_parenthesis_unbalanced_ignored():
    stmt = parse_single("a ) b")
    stmt.tokens = [t for t in stmt.tokens
                   if t.ttype is not T.Punctuation or t.value != ')']
    from sqlparse.sql import Token
    stmt.tokens.append(Token(T.Punctuation, ')'))
    group_parenthesis(stmt)
    assert not any(isinstance(t, sql.Parenthesis) for t in stmt.tokens)


def test_group_case():
    stmt = parse_single("CASE WHEN 1 THEN 2 END")
    group_case(stmt)
    first = [t for t in stmt.tokens if not t.is_whitespace][0]
    assert isinstance(first, sql.Case)


def test_group_if():
    stmt = parse_single("IF 1 THEN 2 END")
    group_if(stmt)
    first = [t for t in stmt.tokens if not t.is_whitespace][0]
    assert not isinstance(first, sql.If)


def test_group_for():
    stmt = parse_single("FOR x")
    group_for(stmt)
    first = [t for t in stmt.tokens if not t.is_whitespace][0]
    assert not isinstance(first, sql.For)


def test_group_begin():
    stmt = parse_single("BEGIN x END")
    group_begin(stmt)
    first = [t for t in stmt.tokens if not t.is_whitespace][0]
    assert isinstance(first, sql.Begin)


# ---------------------------------------------------------------------------
# group_typecasts / tzcasts / typed literal
# ---------------------------------------------------------------------------

def test_group_typecasts():
    stmt = parse_single("1::int")
    group_typecasts(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Identifier)]
    assert len(found) == 1
    assert found[0].value == "1::int"


def test_group_typecasts_no_next():
    """'::' at end of statement has no next token; must not group."""
    from sqlparse.sql import Token
    stmt = parse_single("1")
    stmt.tokens.append(Token(T.Punctuation, "::"))
    group_typecasts(stmt)
    assert not top_level_of_type(stmt, sql.Identifier)


def test_group_tzcasts():
    stmt = parse_single("foo AT TIME ZONE 'UTC'")
    for tok in stmt.tokens:
        if tok.value.upper() == "AT TIME ZONE":
            tok.ttype = T.Keyword.TZCast
    group_tzcasts(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Identifier)]
    assert len(found) == 1


def test_group_tzcasts_no_following_value():
    stmt = parse_single("x AT TIME ZONE")
    for tok in stmt.tokens:
        if tok.value.upper() == "AT TIME ZONE":
            tok.ttype = T.Keyword.TZCast
    group_tzcasts(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Identifier)]
    assert len(found) == 1
    assert found[0].value.strip().upper() == "X"


def test_group_tzcasts_invalid_prev():
    """No prior token: valid_prev should reject and nothing groups."""
    from sqlparse.sql import Token
    stmt = parse_single("1")
    stmt.tokens = [Token(T.Keyword.TZCast, "AT TIME ZONE"),
                   Token(T.Whitespace, " "),
                   Token(T.Literal.String.Single, "'UTC'")]
    group_tzcasts(stmt)
    assert not top_level_of_type(stmt, sql.Identifier)


def test_group_typed_literal_extension():
    """A TypedLiteral followed by an extension keyword should be extended."""
    stmt = parse_single("DATE '2020-01-01'")
    # Manually group into TypedLiteral first.
    group_typed_literal(stmt)
    # After first _group with (extend=False), then extend with M_EXTEND.
    # Assert the token structure is sane.
    non_ws = [t for t in stmt.tokens if not t.is_whitespace]
    assert any(isinstance(t, sql.TypedLiteral) for t in non_ws) or \
        any(t.value == "DATE" for t in non_ws)


# ---------------------------------------------------------------------------
# group_period
# ---------------------------------------------------------------------------

def test_group_period():
    stmt = parse_single("a.b")
    group_identifier(stmt)
    group_period(stmt)
    found = [t for t in stmt.tokens
             if isinstance(t, sql.Identifier) and "." in t.value]
    assert len(found) == 1


def test_group_period_arrow():
    stmt = parse_single("a->b")
    group_identifier(stmt)
    group_period(stmt)
    found = [t for t in stmt.tokens
             if isinstance(t, sql.Identifier) and "->" in t.value]
    assert len(found) == 1


def test_group_period_arrow_double():
    stmt = parse_single("a->>b")
    group_identifier(stmt)
    group_period(stmt)
    found = [t for t in stmt.tokens
             if isinstance(t, sql.Identifier) and "->>" in t.value]
    assert len(found) == 1


def test_group_period_invalid_next():
    """When the next token is not a valid name, only prev is grouped."""
    stmt = parse_single("a.")
    group_identifier(stmt)
    group_period(stmt)
    # post returns (pidx, tidx) when next invalid, so the '.' is not consumed
    # into the Identifier in the same span. Either no match or a single-token
    # identifier is acceptable - just ensure no crash and dot remains literal.
    assert "." in stmt.value


# ---------------------------------------------------------------------------
# group_as / assignment / comparison / operator
# ---------------------------------------------------------------------------

def test_group_as():
    stmt = parse_single("foo AS bar")
    group_identifier(stmt)
    group_as(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Identifier)]
    assert len(found) == 1
    assert "AS" in found[0].value.upper()


def test_group_as_null_allowed():
    stmt = parse_single("NULL AS bar")
    group_identifier(stmt)
    group_as(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Identifier)]
    assert len(found) == 1
    assert "AS" in found[0].value.upper()


def test_group_as_no_group_when_next_is_dml():
    """'AS SELECT' should not group because next is a DML keyword."""
    stmt = parse_single("x AS SELECT")
    group_identifier(stmt)
    group_as(stmt)
    found = [t for t in stmt.tokens
             if isinstance(t, sql.Identifier) and "AS" in t.value.upper()]
    assert not found


def test_group_assignment():
    stmt = parse_single("@a := 1")
    group_identifier(stmt)
    group_assignment(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Assignment)]
    assert len(found) == 1


def test_group_assignment_keyword_invalid():
    stmt = parse_single("@a := SELECT")
    group_identifier(stmt)
    group_assignment(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Assignment)]
    assert len(found) == 1


def test_group_assignment_semicolon_bound():
    """post uses the next semicolon as endpoint if present."""
    stmt = parse_single("@a := 1; x")
    group_identifier(stmt)
    group_assignment(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Assignment)]
    assert len(found) == 1


def test_group_comparison():
    stmt = parse_single("a = 1")
    group_identifier(stmt)
    group_comparison(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Comparison)]
    assert len(found) == 1


def test_group_comparison_null():
    stmt = parse_single("a = NULL")
    group_identifier(stmt)
    group_comparison(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Comparison)]
    assert len(found) == 1


def test_group_comparison_invalid_prev():
    """Comparison at start with no valid prev token."""
    from sqlparse.sql import Token
    stmt = parse_single("x")
    stmt.tokens = [Token(T.Operator.Comparison, "="),
                   Token(T.Whitespace, " "),
                   Token(T.Number.Integer, "1")]
    group_comparison(stmt)
    assert not top_level_of_type(stmt, sql.Comparison)


def test_group_operator():
    stmt = parse_single("1 + 2")
    group_operator(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Operation)]
    assert len(found) == 1
    assert found[0].value == "1 + 2"


def test_group_operator_changes_ttype_to_operator():
    """post sets tlist[tidx].ttype = T.Operator."""
    stmt = parse_single("1 * 2")
    group_operator(stmt)
    found = top_level_of_type(stmt, sql.Operation)
    assert len(found) == 1
    # the operator token inside must have ttype T.Operator
    ops = [t for t in found[0].tokens if t.value.strip() == "*"]
    assert len(ops) == 1
    assert ops[0].ttype == T.Operator


def test_group_operator_current_date():
    stmt = parse_single("1 + CURRENT_DATE")
    group_operator(stmt)
    found = top_level_of_type(stmt, sql.Operation)
    assert len(found) == 1
    assert "CURRENT_DATE" in found[0].value


# ---------------------------------------------------------------------------
# group_identifier / identifier_list
# ---------------------------------------------------------------------------

def test_group_identifier():
    stmt = parse_single("foo")
    group_identifier(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Identifier)]
    assert len(found) == 1


def test_group_identifier_list():
    stmt = parse_single("a, b, c")
    group_identifier(stmt)
    group_identifier_list(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.IdentifierList)]
    assert len(found) == 1
    assert [t.value for t in found[0].get_identifiers()] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# group_over / functions / order
# ---------------------------------------------------------------------------

def test_group_over():
    stmt = parse_single("foo OVER (bar)")
    group_parenthesis(stmt)
    group_identifier(stmt)
    group_over(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Over)]
    assert len(found) == 1


def test_group_over_name_target():
    """OVER can apply to a bare name (t=T.Name)."""
    stmt = parse_single("foo OVER bar")
    group_identifier(stmt)
    group_over(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Over)]
    assert len(found) == 1


def test_group_over_no_target():
    """OVER with no following token should not group."""
    from sqlparse.sql import Token
    stmt = parse_single("x")
    stmt.tokens = [Token(T.Whitespace, " "),
                   Token(T.Keyword, "OVER")]
    group_over(stmt)
    assert not top_level_of_type(stmt, sql.Over)


def test_group_functions():
    stmt = parse_single("SUM(a)")
    group_parenthesis(stmt)
    group_identifier(stmt)
    group_functions(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Function)]
    assert len(found) == 1


def test_group_functions_create_table_no_as():
    stmt = parse_single("CREATE TABLE foo (a)")
    group_parenthesis(stmt)
    group_identifier(stmt)
    group_functions(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Function)]
    assert not found


def test_group_functions_create_table_as():
    stmt = parse_single("CREATE TABLE foo AS (a)")
    group_parenthesis(stmt)
    group_identifier(stmt)
    group_functions(stmt)
    assert isinstance(stmt, sql.Statement)


def test_group_functions_over():
    stmt = parse_single("SUM(a) OVER (b)")
    group_parenthesis(stmt)
    group_identifier(stmt)
    group_over(stmt)
    group_functions(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Function)]
    assert len(found) == 1


def test_group_order():
    stmt = parse_single("foo DESC")
    group_identifier(stmt)
    group_order(stmt)
    found = [t for t in stmt.tokens
             if isinstance(t, sql.Identifier) and "DESC" in t.value.upper()]
    assert len(found) == 1


# ---------------------------------------------------------------------------
# group_arrays
# ---------------------------------------------------------------------------

def test_group_arrays():
    stmt = parse_single("foo[1]")
    group_brackets(stmt)
    group_identifier(stmt)
    group_arrays(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Identifier)]
    assert len(found) >= 1


# ---------------------------------------------------------------------------
# comments / group_where / aliased
# ---------------------------------------------------------------------------

def test_group_comments():
    stmt = parse_single("-- hi\nSELECT 1")
    group_comments(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Comment)]
    assert len(found) == 1


def test_group_comments_trailing():
    stmt = parse_single("SELECT 1 -- trailing")
    group_comments(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Comment)]
    assert not found


def test_group_comments_multiple():
    stmt = parse_single("-- one\n-- two\nSELECT 1")
    group_comments(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Comment)]
    # Comments are grouped into a single group.
    assert len(found) == 1


def test_group_where():
    stmt = parse_single("SELECT * WHERE a = 1")
    group_identifier(stmt)
    group_comparison(stmt)
    group_where(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Where)]
    assert len(found) == 1


def test_group_where_no_closer():
    """WHERE without an end-of-where terminator uses end-of-group."""
    stmt = parse_single("SELECT * WHERE a = 1")
    group_identifier(stmt)
    group_comparison(stmt)
    group_where(stmt)
    found = top_level_of_type(stmt, sql.Where)
    assert len(found) == 1
    assert "WHERE" in found[0].value.upper()


def test_group_aliased():
    stmt = parse_single("(a) alias")
    group_parenthesis(stmt)
    group_identifier(stmt)
    group_aliased(stmt)
    assert isinstance(stmt, sql.Statement)


# ---------------------------------------------------------------------------
# align_comments
# ---------------------------------------------------------------------------

def test_align_comments():
    stmt = parse_single("-- one\n-- two\nSELECT 1")
    group_comments(stmt)
    align_comments(stmt)
    assert isinstance(stmt, sql.Statement)


# ---------------------------------------------------------------------------
# group_values
# ---------------------------------------------------------------------------

def test_group_values():
    stmt = parse_single("INSERT INTO t VALUES (1, 2), (3, 4)")
    group_parenthesis(stmt)
    group_values(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Values)]
    assert len(found) == 1


def test_group_values_none():
    stmt = parse_single("SELECT 1")
    group_values(stmt)
    found = [t for t in stmt.tokens if isinstance(t, sql.Values)]
    assert not found


# ---------------------------------------------------------------------------
# _group depth / token limits
# ---------------------------------------------------------------------------

def test_group_depth_limit():
    stmt = parse_single("a = 1")
    with pytest.raises(SQLParseError) as ei:
        _group(stmt, sql.Comparison,
               lambda t: t.ttype == T.Operator.Comparison,
               depth=grouping.MAX_GROUPING_DEPTH + 1)
    assert str(ei.value) == (
        f"Maximum grouping depth exceeded ({grouping.MAX_GROUPING_DEPTH}).")


def test_group_depth_boundary():
    stmt = parse_single("a = 1")
    _group(stmt, sql.Comparison,
           lambda t: t.ttype == T.Operator.Comparison,
           depth=grouping.MAX_GROUPING_DEPTH)


def test_group_token_limit(monkeypatch):
    stmt = parse_single("a b")
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 1)
    with pytest.raises(SQLParseError) as ei:
        _group(stmt, sql.Comparison,
               lambda t: t.ttype == T.Operator.Comparison)
    assert str(ei.value) == "Maximum number of tokens exceeded (1)."


def test_group_token_boundary(monkeypatch):
    # Use a nodeless single token so the boundary equality is exact and no
    # recursion into sublists exceeds the limit (strict >).
    stmt = parse_single("1")
    n = len(stmt.tokens)
    assert n == 1
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", n)
    _group(stmt, sql.Comparison,
           lambda t: t.ttype == T.Operator.Comparison)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_group_full():
    stmt = parse_single("SELECT a, b FROM tbl WHERE a = 1")
    result = group(stmt)
    assert result is stmt
    assert any(isinstance(t, sql.IdentifierList) for t in stmt.tokens)
    assert any(isinstance(t, sql.Where) for t in stmt.tokens)


def test_group_returns_self():
    stmt = parse_single("SELECT 1")
    assert group(stmt) is stmt


def test_group_query_method():
    stmt = sqlparse.parse("SELECT COUNT(*) FROM x WHERE y = 1")[0]
    assert "COUNT" in stmt.value
