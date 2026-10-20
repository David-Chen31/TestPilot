import pytest

import sqlparse
from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.exceptions import SQLParseError

from sqlparse.engine import grouping


def _flatten(tlist):
    """Helper: top-level token values."""
    return [str(t) for t in tlist.tokens]


# ---------------------------------------------------------------------------
# _group_matching based functions
# ---------------------------------------------------------------------------

def test_group_brackets():
    tlist = sqlparse.parse('foo[bar]')[0]
    grouping.group_brackets(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.SquareBrackets)]
    assert len(groups) == 1
    assert str(groups[0]) == '[bar]'


def test_group_brackets_unbalanced():
    tlist = sqlparse.parse('foo]')[0]
    grouping.group_brackets(tlist)
    assert not any(isinstance(t, sql.SquareBrackets) for t in tlist.tokens)


def test_group_parenthesis():
    tlist = sqlparse.parse('(a + b)')[0]
    grouping.group_parenthesis(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.Parenthesis)]
    assert len(groups) == 1
    assert str(groups[0]) == '(a + b)'


def test_group_parenthesis_nested():
    tlist = sqlparse.parse('((a))')[0]
    grouping.group_parenthesis(tlist)
    outer = [t for t in tlist.tokens if isinstance(t, sql.Parenthesis)]
    assert len(outer) == 1
    assert str(outer[0]) == '((a))'
    inner = [t for t in outer[0].tokens if isinstance(t, sql.Parenthesis)]
    assert len(inner) == 1
    assert str(inner[0]) == '(a)'


def test_group_parenthesis_unbalanced_close():
    tlist = sqlparse.parse('a)')[0]
    grouping.group_parenthesis(tlist)
    assert not any(isinstance(t, sql.Parenthesis) for t in tlist.tokens)


def test_group_case():
    tlist = sqlparse.parse('CASE WHEN a THEN b END')[0]
    grouping.group_case(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.Case)]
    assert len(groups) == 1
    assert str(groups[0]).startswith('CASE')


def test_group_if():
    tlist = sqlparse.parse('IF a THEN b END')[0]
    grouping.group_if(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.If)]
    assert len(groups) == 1
    assert str(groups[0]).startswith('IF')


def test_group_for():
    tlist = sqlparse.parse('FOR x IN y LOOP END LOOP')[0]
    grouping.group_for(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.For)]
    assert len(groups) == 1
    assert str(groups[0]).startswith('FOR')


def test_group_begin():
    tlist = sqlparse.parse('BEGIN END')[0]
    grouping.group_begin(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.Begin)]
    assert len(groups) == 1
    assert str(groups[0]).startswith('BEGIN')


# ---------------------------------------------------------------------------
# group_typecasts
# ---------------------------------------------------------------------------

def test_group_typecasts():
    tlist = sqlparse.parse('a::int')[0]
    grouping.group_typecasts(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert str(idents[0]) == 'a::int'


def test_group_typecasts_no_match():
    tlist = sqlparse.parse('a')[0]
    grouping.group_typecasts(tlist)
    assert not any(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_tzcasts
# ---------------------------------------------------------------------------

def test_group_tzcasts():
    # "AT TIME ZONE" gets TZCast token type
    tlist = sqlparse.parse("'2004-10-19 10:23:54' AT TIME ZONE 'UTC'")[0]
    grouping.group_tzcasts(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert any('AT TIME ZONE' in str(t) for t in idents)


def test_group_tzcasts_no_match():
    tlist = sqlparse.parse('a')[0]
    grouping.group_tzcasts(tlist)
    assert not any(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_typed_literal
# ---------------------------------------------------------------------------

def test_group_typed_literal_date():
    tlist = sqlparse.parse("DATE '2020-01-01'")[0]
    grouping.group_typed_literal(tlist)
    tls = [t for t in tlist.tokens if isinstance(t, sql.TypedLiteral)]
    assert len(tls) == 1
    assert str(tls[0]) == "DATE '2020-01-01'"


def test_group_typed_literal_no_match():
    tlist = sqlparse.parse('foo')[0]
    grouping.group_typed_literal(tlist)
    assert not any(isinstance(t, sql.TypedLiteral) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_period
# ---------------------------------------------------------------------------

def test_group_period():
    tlist = sqlparse.parse('a.b')[0]
    grouping.group_identifier(tlist)
    grouping.group_period(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert str(idents[0]) == 'a.b'


def test_group_period_invalid_next():
    # invalid next: '.' followed by nothing valid -> post returns (pidx, tidx)
    tlist = sqlparse.parse('a. ')[0]
    grouping.group_identifier(tlist)
    grouping.group_period(tlist)
    # No Identifier wrapping the period should occur since next is whitespace
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    # Only the single-name identifiers exist
    for ident in idents:
        assert '.' not in str(ident)


# ---------------------------------------------------------------------------
# group_as
# ---------------------------------------------------------------------------

def test_group_as():
    tlist = sqlparse.parse('a AS b')[0]
    grouping.group_as(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert str(idents[0]) == 'a AS b'


def test_group_as_null_prev():
    tlist = sqlparse.parse('NULL AS b')[0]
    grouping.group_as(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert any('AS' in str(t) for t in idents)


def test_group_as_invalid_next():
    # SELECT is DML token, should not group
    tlist = sqlparse.parse('a AS SELECT')[0]
    grouping.group_as(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert not any('AS' in str(t) for t in idents)


# ---------------------------------------------------------------------------
# group_assignment
# ---------------------------------------------------------------------------

def test_group_assignment():
    tlist = sqlparse.parse('@a := 1;')[0]
    grouping.group_assignment(tlist)
    assigns = [t for t in tlist.tokens if isinstance(t, sql.Assignment)]
    assert len(assigns) == 1
    assert '=' in str(assigns[0]) or ':=' in str(assigns[0])


def test_group_assignment_no_semicolon():
    tlist = sqlparse.parse('@a := 1')[0]
    grouping.group_assignment(tlist)
    assigns = [t for t in tlist.tokens if isinstance(t, sql.Assignment)]
    assert len(assigns) == 1


# ---------------------------------------------------------------------------
# group_comparison
# ---------------------------------------------------------------------------

def test_group_comparison():
    tlist = sqlparse.parse('a = 1')[0]
    grouping.group_identifier(tlist)
    grouping.group_comparison(tlist)
    cmps = [t for t in tlist.tokens if isinstance(t, sql.Comparison)]
    assert len(cmps) == 1
    assert '=' in str(cmps[0])


def test_group_comparison_null():
    tlist = sqlparse.parse('a IS NULL')[0]
    grouping.group_identifier(tlist)
    grouping.group_comparison(tlist)
    cmps = [t for t in tlist.tokens if isinstance(t, sql.Comparison)]
    assert len(cmps) == 1
    assert 'NULL' in str(cmps[0])


# ---------------------------------------------------------------------------
# group_identifier
# ---------------------------------------------------------------------------

def test_group_identifier_name():
    tlist = sqlparse.parse('foo')[0]
    grouping.group_identifier(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert str(idents[0]) == 'foo'


def test_group_identifier_none():
    tlist = sqlparse.parse('123')[0]
    grouping.group_identifier(tlist)
    assert not any(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_over
# ---------------------------------------------------------------------------

def test_group_over():
    tlist = sqlparse.parse('ROW_NUMBER() OVER (PARTITION BY a)')[0]
    grouping.group_parenthesis(tlist)
    grouping.group_over(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.Over)]
    assert len(groups) == 1
    assert 'OVER' in str(groups[0])


def test_group_over_no_parenthesis():
    tlist = sqlparse.parse('OVER foo')[0]
    grouping.group_over(tlist)
    assert not any(isinstance(t, sql.Over) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_arrays
# ---------------------------------------------------------------------------

def test_group_arrays():
    # a[1] - square brackets after identifier
    tlist = sqlparse.parse('a[1]')[0]
    grouping.group_brackets(tlist)
    grouping.group_identifier(tlist)
    grouping.group_arrays(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 1
    assert str(idents[0]) == 'a[1]'


# ---------------------------------------------------------------------------
# group_operator
# ---------------------------------------------------------------------------

def test_group_operator():
    tlist = sqlparse.parse('a + b')[0]
    grouping.group_identifier(tlist)
    grouping.group_operator(tlist)
    ops = [t for t in tlist.tokens if isinstance(t, sql.Operation)]
    assert len(ops) == 1
    assert '+' in str(ops[0])


def test_group_operator_current_date():
    tlist = sqlparse.parse('CURRENT_DATE + 1')[0]
    grouping.group_operator(tlist)
    ops = [t for t in tlist.tokens if isinstance(t, sql.Operation)]
    assert len(ops) == 1


# ---------------------------------------------------------------------------
# group_identifier_list
# ---------------------------------------------------------------------------

def test_group_identifier_list():
    tlist = sqlparse.parse('a, b')[0]
    grouping.group_identifier(tlist)
    grouping.group_identifier_list(tlist)
    lists = [t for t in tlist.tokens if isinstance(t, sql.IdentifierList)]
    assert len(lists) == 1
    assert ',' in str(lists[0])


def test_group_identifier_list_null_role():
    tlist = sqlparse.parse('null, role')[0]
    grouping.group_identifier(tlist)
    grouping.group_identifier_list(tlist)
    lists = [t for t in tlist.tokens if isinstance(t, sql.IdentifierList)]
    assert len(lists) == 1


# ---------------------------------------------------------------------------
# group_comments
# ---------------------------------------------------------------------------

def test_group_comments_multiline():
    tlist = sqlparse.parse('a\n-- line1\n-- line2\nb')[0]
    grouping.group_comments(tlist)
    comments = [t for t in tlist.tokens if isinstance(t, sql.Comment)]
    assert len(comments) == 1


def test_group_comments_trailing():
    # trailing comment with no terminator - should not loop forever
    tlist = sqlparse.parse('a\n-- trailing')[0]
    grouping.group_comments(tlist)
    comments = [t for t in tlist.tokens if isinstance(t, sql.Comment)]
    assert len(comments) == 1


# ---------------------------------------------------------------------------
# group_where
# ---------------------------------------------------------------------------

def test_group_where():
    tlist = sqlparse.parse('WHERE a = 1')[0]
    grouping.group_where(tlist)
    wheres = [t for t in tlist.tokens if isinstance(t, sql.Where)]
    assert len(wheres) == 1
    assert str(wheres[0]).startswith('WHERE')


def test_group_where_to_end():
    tlist = sqlparse.parse('WHERE a')[0]
    grouping.group_where(tlist)
    wheres = [t for t in tlist.tokens if isinstance(t, sql.Where)]
    assert len(wheres) == 1


# ---------------------------------------------------------------------------
# group_aliased
# ---------------------------------------------------------------------------

def test_group_aliased():
    tlist = sqlparse.parse('abc AS def')[0]
    grouping.group_identifier(tlist)
    grouping.group_as(tlist)
    # group_aliased expects identifier followed by another identifier
    grouping.group_aliased(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    # combined form
    combined = [t for t in idents if str(t) == 'abc AS def']
    assert len(combined) == 1


# ---------------------------------------------------------------------------
# group_functions
# ---------------------------------------------------------------------------

def test_group_functions():
    tlist = sqlparse.parse('foo(1)')[0]
    grouping.group_parenthesis(tlist)
    grouping.group_functions(tlist)
    funcs = [t for t in tlist.tokens if isinstance(t, sql.Function)]
    assert len(funcs) == 1
    assert str(funcs[0]) == 'foo(1)'


def test_group_functions_create_table_no_as():
    tlist = sqlparse.parse('CREATE TABLE foo (a int)')[0]
    grouping.group_parenthesis(tlist)
    grouping.group_functions(tlist)
    # Create-table w/o AS should be skipped
    funcs = [t for t in tlist.tokens if isinstance(t, sql.Function)]
    assert funcs == []


def test_group_functions_with_over():
    tlist = sqlparse.parse('ROW_NUMBER() OVER (PARTITION BY a)')[0]
    grouping.group_parenthesis(tlist)
    grouping.group_over(tlist)
    grouping.group_functions(tlist)
    funcs = [t for t in tlist.tokens if isinstance(t, sql.Function)]
    assert len(funcs) == 1
    assert 'OVER' in str(funcs[0])


# ---------------------------------------------------------------------------
# group_order
# ---------------------------------------------------------------------------

def test_group_order():
    tlist = sqlparse.parse('a DESC')[0]
    grouping.group_identifier(tlist)
    grouping.group_order(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert any('DESC' in str(t) for t in idents)


def test_group_order_number():
    tlist = sqlparse.parse('1 DESC')[0]
    grouping.group_order(tlist)
    idents = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert any('DESC' in str(t) for t in idents)


# ---------------------------------------------------------------------------
# align_comments
# ---------------------------------------------------------------------------

def test_align_comments():
    tlist = sqlparse.parse('SELECT 1 -- comment\nFROM foo')[0]
    # First group comments by token-level parse, then align
    grouping.group_comments(tlist)
    grouping.align_comments(tlist)
    tls = [t for t in tlist.tokens if isinstance(t, sql.TokenList)]
    assert len(tls) >= 1


# ---------------------------------------------------------------------------
# group_values
# ---------------------------------------------------------------------------

def test_group_values():
    tlist = sqlparse.parse('VALUES (1)')[0]
    grouping.group_parenthesis(tlist)
    grouping.group_values(tlist)
    vals = [t for t in tlist.tokens if isinstance(t, sql.Values)]
    assert len(vals) == 1
    assert str(vals[0]).startswith('VALUES')


def test_group_values_none():
    tlist = sqlparse.parse('foo')[0]
    grouping.group_values(tlist)
    assert not any(isinstance(t, sql.Values) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# Full pipeline via group() and filter_stack
# ---------------------------------------------------------------------------

def test_group_pipeline_simple():
    stmt = sqlparse.parse('SELECT a FROM b')[0]
    result = grouping.group(stmt)
    assert result is stmt
    assert 'SELECT' in result.value


def test_group_pipeline_complex():
    stmt = sqlparse.parse(
        "SELECT a.b, CASE WHEN x THEN 1 ELSE 2 END FROM t "
        "WHERE y = 1 ORDER BY z DESC"
    )[0]
    grouping.group(stmt)
    assert isinstance(stmt, sql.Statement)


# ---------------------------------------------------------------------------
# MAX_GROUPING_DEPTH / TOKENS limit protections
# ---------------------------------------------------------------------------

def test_max_depth_exceeded_group_matching(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', -1)
    tlist = sqlparse.parse('foo')[0]
    with pytest.raises(SQLParseError):
        grouping.group_parenthesis(tlist)


def test_max_depth_exceeded_group(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', -1)
    tlist = sqlparse.parse('foo')[0]
    with pytest.raises(SQLParseError):
        grouping.group_typecasts(tlist)


def test_max_tokens_exceeded_group_matching(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 0)
    tlist = sqlparse.parse('foo')[0]
    with pytest.raises(SQLParseError):
        grouping.group_parenthesis(tlist)


def test_max_tokens_exceeded_group(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 0)
    tlist = sqlparse.parse('foo')[0]
    with pytest.raises(SQLParseError):
        grouping.group_typecasts(tlist)
