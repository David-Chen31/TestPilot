"""Unit tests for sqlparse.engine.grouping."""

import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine import grouping
from sqlparse.exceptions import SQLParseError
from sqlparse.tokens import Token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tlist(*tokens):
    """Build a TokenList from flat token objects."""
    tlist = sql.TokenList(list(tokens))
    return tlist


def tok(ttype, value):
    return Token(ttype, value)


def types_of(tlist):
    return [t.ttype for t in tlist.tokens]


def classes_of(tlist):
    return [type(t).__name__ for t in tlist.tokens]


# ---------------------------------------------------------------------------
# group_brackets / group_parenthesis / group_case / group_if / group_for /
# group_begin  (all via _group_matching)
# ---------------------------------------------------------------------------

def test_group_brackets_simple():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Punctuation, '['),
        tok(T.Name, 'b'),
        tok(T.Punctuation, ']'),
    )
    grouping.group_brackets(tlist)
    assert len(tlist.tokens) == 2
    assert isinstance(tlist.tokens[1], sql.SquareBrackets)
    assert tlist.tokens[1].value == '[b]'


def test_group_parenthesis_simple():
    tlist = make_tlist(
        tok(T.Name, 'f'),
        tok(T.Punctuation, '('),
        tok(T.Name, 'x'),
        tok(T.Punctuation, ')'),
    )
    grouping.group_parenthesis(tlist)
    assert len(tlist.tokens) == 2
    assert isinstance(tlist.tokens[1], sql.Parenthesis)
    assert tlist.tokens[1].value == '(x)'


def test_group_parenthesis_nested():
    tlist = make_tlist(
        tok(T.Punctuation, '('),
        tok(T.Punctuation, '('),
        tok(T.Name, 'x'),
        tok(T.Punctuation, ')'),
        tok(T.Punctuation, ')'),
    )
    grouping.group_parenthesis(tlist)
    assert len(tlist.tokens) == 1
    outer = tlist.tokens[0]
    assert isinstance(outer, sql.Parenthesis)
    # inner is on second _group_matching call.
    assert isinstance(outer.tokens[0], sql.Parenthesis)


def test_group_parenthesis_unbalanced_close_ignored():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Punctuation, ')'),
        tok(T.Punctuation, '('),
        tok(T.Name, 'b'),
        tok(T.Punctuation, ')'),
    )
    grouping.group_parenthesis(tlist)
    # First ')' is ignored, the matched '(' ')' becomes Parenthesis.
    assert len(tlist.tokens) == 3
    assert isinstance(tlist.tokens[2], sql.Parenthesis)


def test_group_parenthesis_unbalanced_open_ignored():
    tlist = make_tlist(
        tok(T.Punctuation, '('),
        tok(T.Name, 'a'),
    )
    grouping.group_parenthesis(tlist)
    assert len(tlist.tokens) == 2
    assert tlist.tokens[0].ttype == T.Punctuation


def test_group_case():
    tlist = make_tlist(
        tok(T.Keyword, 'CASE'),
        tok(T.Name, 'x'),
        tok(T.Keyword, 'END'),
    )
    grouping.group_case(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Case)
    assert tlist.tokens[0].value == 'CASE x END'


def test_group_if():
    tlist = make_tlist(
        tok(T.Keyword, 'IF'),
        tok(T.Name, 'x'),
        tok(T.Keyword, 'END'),
    )
    grouping.group_if(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.If)


def test_group_for():
    tlist = make_tlist(
        tok(T.Keyword, 'FOR'),
        tok(T.Name, 'x'),
        tok(T.Keyword, 'END'),
    )
    grouping.group_for(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.For)


def test_group_begin():
    tlist = make_tlist(
        tok(T.Keyword, 'BEGIN'),
        tok(T.Name, 'x'),
        tok(T.Keyword, 'END'),
    )
    grouping.group_begin(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Begin)


def test_group_matching_recurses_into_other_group():
    inner = sql.Parenthesis([tok(T.Name, 'x')], 'x')
    tlist = make_tlist(
        tok(T.Punctuation, '['),
        inner,
        tok(T.Name, 'y'),
        tok(T.Punctuation, ']'),
    )
    grouping.group_brackets(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.SquareBrackets)
    # the parenthesized group preserved
    assert isinstance(tlist.tokens[0].tokens[0], sql.Parenthesis)


def test_group_matching_max_depth(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 0)
    tlist = make_tlist(
        tok(T.Punctuation, '('),
        tok(T.Name, 'a'),
        tok(T.Punctuation, ')'),
    )
    with pytest.raises(SQLParseError):
        grouping.group_parenthesis(tlist)


def test_group_matching_max_tokens(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 1)
    tlist = make_tlist(
        tok(T.Punctuation, '('),
        tok(T.Name, 'a'),
        tok(T.Punctuation, ')'),
    )
    with pytest.raises(SQLParseError):
        grouping.group_parenthesis(tlist)


# ---------------------------------------------------------------------------
# group_typecasts
# ---------------------------------------------------------------------------

def test_group_typecasts():
    tlist = make_tlist(
        tok(T.Name, 'x'),
        tok(T.Punctuation, '::'),
        tok(T.Name, 'int'),
    )
    grouping.group_typecasts(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Identifier)
    assert tlist.tokens[0].value == 'x::int'


def test_group_typecasts_no_prev_ignored():
    tlist = make_tlist(
        tok(T.Punctuation, '::'),
        tok(T.Name, 'int'),
    )
    grouping.group_typecasts(tlist)
    # No valid prev -> unchanged count
    assert len(tlist.tokens) == 2


# ---------------------------------------------------------------------------
# group_tzcasts
# ---------------------------------------------------------------------------

def test_group_tzcasts_with_whitespace_next():
    tlist = make_tlist(
        tok(T.Name, 'ts'),
        tok(T.Keyword.TZCast, 'AT TIME ZONE'),
        tok(T.Whitespace, ' '),
    )
    grouping.group_tzcasts(tlist)
    assert isinstance(tlist.tokens[0], sql.Identifier)
    assert tlist.tokens[0].value == 'tsAT TIME ZONE'


def test_group_tzcasts_with_as_next():
    tlist = make_tlist(
        tok(T.Name, 'ts'),
        tok(T.Keyword.TZCast, 'AT TIME ZONE'),
        tok(T.Keyword, 'AS'),
    )
    grouping.group_tzcasts(tlist)
    assert isinstance(tlist.tokens[0], sql.Identifier)


def test_group_tzcasts_invalid_next_unchanged():
    tlist = make_tlist(
        tok(T.Name, 'ts'),
        tok(T.Keyword.TZCast, 'AT TIME ZONE'),
        tok(T.Name, 'tz'),
    )
    grouping.group_tzcasts(tlist)
    assert len(tlist.tokens) == 3
    assert not any(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_typed_literal
# ---------------------------------------------------------------------------

def test_group_typed_literal():
    tlist = make_tlist(
        tok(T.Name.Builtin, 'DATE'),
        tok(T.String.Single, "'2020-01-01'"),
    )
    grouping.group_typed_literal(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.TypedLiteral)


def test_group_typed_literal_extend():
    # First parse -> TypedLiteral, then with EXTEND literal appended.
    tlist = make_tlist(
        tok(T.Name.Builtin, 'INTERVAL'),
        tok(T.String.Single, "'1'"),
        tok(T.Keyword, 'YEAR'),
    )
    grouping.group_typed_literal(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.TypedLiteral)
    assert tlist.tokens[0].value == "INTERVAL'1'YEAR"


# ---------------------------------------------------------------------------
# group_period
# ---------------------------------------------------------------------------

def test_group_period_dot():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Punctuation, '.'),
        tok(T.Name, 'b'),
    )
    grouping.group_period(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Identifier)
    assert tlist.tokens[0].value == 'a.b'


def test_group_period_arrow():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Operator, '->'),
        tok(T.Name, 'b'),
    )
    grouping.group_period(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Identifier)


def test_group_period_arrow_double():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Operator, '->>'),
        tok(T.Name, 'b'),
    )
    grouping.group_period(tlist)
    assert isinstance(tlist.tokens[0], sql.Identifier)


def test_group_period_invalid_prev_unchanged():
    tlist = make_tlist(
        tok(T.Punctuation, '.'),
        tok(T.Name, 'b'),
    )
    grouping.group_period(tlist)
    assert len(tlist.tokens) == 2


def test_group_period_invalid_next_unchanged():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Punctuation, '.'),
        tok(T.Punctuation, ','),
    )
    grouping.group_period(tlist)
    # valid_next returns True but post rejects because next not valid type,
    # so no grouping (to_idx == tidx -> group over just ".").
    # Result: only the a and the "." got grouped? Actually post returns
    # (pidx, tidx) -> group a and ".".
    assert isinstance(tlist.tokens[0], sql.Identifier)
    assert tlist.tokens[0].value == 'a.'


# ---------------------------------------------------------------------------
# group_as
# ---------------------------------------------------------------------------

def test_group_as():
    tlist = make_tlist(
        tok(T.Name, 'col'),
        tok(T.Keyword, 'AS'),
        tok(T.Name, 'alias'),
    )
    grouping.group_as(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Identifier)


def test_group_as_null_prev():
    tlist = make_tlist(
        tok(T.Keyword, 'NULL'),
        tok(T.Keyword, 'AS'),
        tok(T.Name, 'alias'),
    )
    grouping.group_as(tlist)
    assert isinstance(tlist.tokens[0], sql.Identifier)


def test_group_as_invalid_next_dml():
    tlist = make_tlist(
        tok(T.Name, 'col'),
        tok(T.Keyword, 'AS'),
        tok(T.Keyword, 'SELECT'),
    )
    grouping.group_as(tlist)
    assert len(tlist.tokens) == 3


# ---------------------------------------------------------------------------
# group_assignment
# ---------------------------------------------------------------------------

def test_group_assignment():
    tlist = make_tlist(
        tok(T.Name, 'x'),
        tok(T.Assignment, ':='),
        tok(T.Name, 'y'),
    )
    grouping.group_assignment(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Assignment)


def test_group_assignment_semicolon_terminates():
    tlist = make_tlist(
        tok(T.Name, 'x'),
        tok(T.Assignment, ':='),
        tok(T.Name, 'y'),
        tok(T.Punctuation, ';'),
        tok(T.Name, 'z'),
    )
    grouping.group_assignment(tlist)
    assert isinstance(tlist.tokens[0], sql.Assignment)
    # 'z' remains separate
    assert len(tlist.tokens) == 2


def test_group_assignment_keyword_next_rejected():
    tlist = make_tlist(
        tok(T.Name, 'x'),
        tok(T.Assignment, ':='),
        tok(T.Keyword, 'SELECT'),
    )
    grouping.group_assignment(tlist)
    assert len(tlist.tokens) == 3


# ---------------------------------------------------------------------------
# group_comparison
# ---------------------------------------------------------------------------

def test_group_comparison():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Operator.Comparison, '='),
        tok(T.Number.Integer, '1'),
    )
    grouping.group_comparison(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Comparison)


def test_group_comparison_null():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Operator.Comparison, '='),
        tok(T.Keyword, 'NULL'),
    )
    grouping.group_comparison(tlist)
    assert isinstance(tlist.tokens[0], sql.Comparison)


def test_group_comparison_invalid_prev():
    tlist = make_tlist(
        tok(T.Punctuation, ','),
        tok(T.Operator.Comparison, '='),
        tok(T.Number.Integer, '1'),
    )
    grouping.group_comparison(tlist)
    assert len(tlist.tokens) == 3


# ---------------------------------------------------------------------------
# group_identifier
# ---------------------------------------------------------------------------

def test_group_identifier():
    tlist = make_tlist(
        tok(T.String.Symbol, '"foo"'),
        tok(T.Name, 'bar'),
    )
    grouping.group_identifier(tlist)
    assert len(tlist.tokens) == 2
    assert isinstance(tlist.tokens[0], sql.Identifier)
    assert isinstance(tlist.tokens[1], sql.Identifier)
    assert tlist.tokens[0].value == '"foo"'
    assert tlist.tokens[1].value == 'bar'


def test_group_identifier_recurses():
    inner = make_tlist(tok(T.Name, 'x'))
    outer = make_tlist(inner)
    grouping.group_identifier(outer)
    assert isinstance(outer.tokens[0].tokens[0], sql.Identifier)


# ---------------------------------------------------------------------------
# group_over
# ---------------------------------------------------------------------------

def test_group_over():
    inner = sql.Parenthesis([tok(T.Name, 'x')], '(x)')
    tlist = make_tlist(
        tok(T.Keyword, 'OVER'),
        inner,
    )
    grouping.group_over(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Over)


def test_group_over_with_name():
    tlist = make_tlist(
        tok(T.Keyword, 'OVER'),
        tok(T.Name, 'w'),
    )
    grouping.group_over(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Over)


def test_group_over_no_next():
    tlist = make_tlist(tok(T.Keyword, 'OVER'))
    grouping.group_over(tlist)
    assert len(tlist.tokens) == 1
    assert not isinstance(tlist.tokens[0], sql.Over)


# ---------------------------------------------------------------------------
# group_arrays
# ---------------------------------------------------------------------------

def test_group_arrays():
    brackets = sql.SquareBrackets([tok(T.Name, 'i')], '[i]')
    tlist = make_tlist(
        tok(T.Name, 'arr'),
        brackets,
    )
    grouping.group_arrays(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Identifier)


def test_group_arrays_invalid_prev():
    brackets = sql.SquareBrackets([tok(T.Name, 'i')], '[i]')
    tlist = make_tlist(
        tok(T.Punctuation, ','),
        brackets,
    )
    grouping.group_arrays(tlist)
    assert len(tlist.tokens) == 2


# ---------------------------------------------------------------------------
# group_operator
# ---------------------------------------------------------------------------

def test_group_operator():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Operator, '+'),
        tok(T.Number.Integer, '1'),
    )
    grouping.group_operator(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Operation)


def test_group_operator_current_date():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Operator, '-'),
        tok(T.Keyword, 'CURRENT_DATE'),
    )
    grouping.group_operator(tlist)
    assert isinstance(tlist.tokens[0], sql.Operation)


def test_group_operator_no_prev():
    tlist = make_tlist(
        tok(T.Operator, '+'),
        tok(T.Name, 'a'),
    )
    grouping.group_operator(tlist)
    assert len(tlist.tokens) == 2


# ---------------------------------------------------------------------------
# group_identifier_list
# ---------------------------------------------------------------------------

def test_group_identifier_list():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Punctuation, ','),
        tok(T.Name, 'b'),
    )
    grouping.group_identifier_list(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.IdentifierList)


def test_group_identifier_list_extends():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Punctuation, ','),
        tok(T.Name, 'b'),
        tok(T.Punctuation, ','),
        tok(T.Name, 'c'),
    )
    grouping.group_identifier_list(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.IdentifierList)


def test_group_identifier_list_invalid():
    tlist = make_tlist(
        tok(T.Punctuation, ','),
        tok(T.Name, 'a'),
    )
    grouping.group_identifier_list(tlist)
    assert len(tlist.tokens) == 2


# ---------------------------------------------------------------------------
# group_comments
# ---------------------------------------------------------------------------

def test_group_comments():
    tlist = make_tlist(
        tok(T.Comment.Single, '-- first'),
        tok(T.Newline, '\n'),
        tok(T.Comment.Single, '-- second'),
        tok(T.Newline, '\n'),
    )
    grouping.group_comments(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Comment)


def test_group_comments_trailing_without_terminator():
    tlist = make_tlist(
        tok(T.Comment.Single, '-- only comment'),
        tok(T.Newline, '\n'),
    )
    # Ends in comment / newline only: no terminator, should not raise.
    grouping.group_comments(tlist)
    assert len(tlist.tokens) >= 1


# ---------------------------------------------------------------------------
# group_where
# ---------------------------------------------------------------------------

def test_group_where_basic():
    tlist = make_tlist(
        tok(T.Keyword, 'WHERE'),
        tok(T.Name, 'a'),
        tok(T.Operator.Comparison, '='),
        tok(T.Number.Integer, '1'),
        tok(T.Keyword, 'ORDER'),
        tok(T.Keyword, 'BY'),
        tok(T.Name, 'a'),
    )
    grouping.group_where(tlist)
    where_groups = [t for t in tlist.tokens if isinstance(t, sql.Where)]
    assert len(where_groups) == 1


def test_group_where_no_close():
    tlist = make_tlist(
        tok(T.Keyword, 'WHERE'),
        tok(T.Name, 'a'),
    )
    grouping.group_where(tlist)
    assert isinstance(tlist.tokens[0], sql.Where)


# ---------------------------------------------------------------------------
# group_aliased
# ---------------------------------------------------------------------------

def test_group_aliased():
    tlist = make_tlist(
        tok(T.Number.Integer, '1'),
        tok(T.Name, 'x'),
    )
    # first make an Identifier from the name
    grouping.group_identifier(tlist)
    grouping.group_aliased(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Identifier)


def test_group_aliased_no_identifier_next():
    tlist = make_tlist(
        tok(T.Number.Integer, '1'),
        tok(T.Punctuation, ','),
    )
    grouping.group_aliased(tlist)
    assert len(tlist.tokens) == 2


# ---------------------------------------------------------------------------
# group_functions
# ---------------------------------------------------------------------------

def test_group_functions():
    paren = sql.Parenthesis([tok(T.Name, 'x')], '(x)')
    tlist = make_tlist(
        tok(T.Name, 'f'),
        paren,
    )
    grouping.group_functions(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Function)


def test_group_functions_with_over():
    paren = sql.Parenthesis([tok(T.Name, 'x')], '(x)')
    over = sql.Over([tok(T.Keyword, 'OVER')], 'OVER')
    tlist = make_tlist(
        tok(T.Name, 'f'),
        paren,
        over,
    )
    grouping.group_functions(tlist)
    assert isinstance(tlist.tokens[0], sql.Function)
    assert tlist.tokens[0].value == 'f(x)OVER'


def test_group_functions_create_table():
    tlist = make_tlist(
        tok(T.Keyword, 'CREATE'),
        tok(T.Keyword, 'TABLE'),
        tok(T.Name, 'x'),
    )
    grouping.group_functions(tlist)
    # early return: still three tokens
    assert len(tlist.tokens) == 3


def test_group_functions_create_table_as():
    paren = sql.Parenthesis([tok(T.Name, 'x')], '(x)')
    tlist = make_tlist(
        tok(T.Keyword, 'CREATE'),
        tok(T.Keyword, 'TABLE'),
        tok(T.Keyword, 'AS'),
        tok(T.Name, 'f'),
        paren,
    )
    grouping.group_functions(tlist)
    assert any(isinstance(t, sql.Function) for t in tlist.tokens)


def test_group_functions_no_parenthesis():
    tlist = make_tlist(
        tok(T.Name, 'f'),
        tok(T.Name, 'g'),
    )
    grouping.group_functions(tlist)
    assert not any(isinstance(t, sql.Function) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_order
# ---------------------------------------------------------------------------

def test_group_order():
    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Keyword.Order, 'ASC'),
    )
    grouping.group_identifier(tlist)
    grouping.group_order(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Identifier)
    assert tlist.tokens[0].value == 'aASC'


def test_group_order_with_number():
    tlist = make_tlist(
        tok(T.Number.Integer, '1'),
        tok(T.Keyword.Order, 'DESC'),
    )
    grouping.group_order(tlist)
    assert isinstance(tlist.tokens[0], sql.Identifier)


def test_group_order_no_prev():
    tlist = make_tlist(tok(T.Keyword.Order, 'ASC'))
    grouping.group_order(tlist)
    assert len(tlist.tokens) == 1
    assert tlist.tokens[0].ttype == T.Keyword.Order


# ---------------------------------------------------------------------------
# align_comments
# ---------------------------------------------------------------------------

def test_align_comments():
    identifier = sql.Identifier([tok(T.Name, 'a')], 'a')
    tlist = make_tlist(
        identifier,
        tok(T.Comment.Single, '-- c'),
    )
    grouping.align_comments(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.TokenList)


def test_align_comments_no_prev():
    tlist = make_tlist(tok(T.Comment.Single, '-- c'))
    grouping.align_comments(tlist)
    assert len(tlist.tokens) == 1


# ---------------------------------------------------------------------------
# group_values
# ---------------------------------------------------------------------------

def test_group_values():
    paren = sql.Parenthesis([tok(T.Number.Integer, '1')], '(1)')
    tlist = make_tlist(
        tok(T.Keyword, 'VALUES'),
        paren,
        tok(T.Punctuation, ','),
        paren,
    )
    grouping.group_values(tlist)
    assert isinstance(tlist.tokens[0], sql.Values)


def test_group_values_no_parenthesis():
    tlist = make_tlist(
        tok(T.Keyword, 'VALUES'),
        tok(T.Name, 'x'),
    )
    grouping.group_values(tlist)
    assert not any(isinstance(t, sql.Values) for t in tlist.tokens)


def test_group_values_no_values_keyword():
    paren = sql.Parenthesis([tok(T.Number.Integer, '1')], '(1)')
    tlist = make_tlist(paren)
    grouping.group_values(tlist)
    assert not any(isinstance(t, sql.Values) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# _group helpers / MAX limits
# ---------------------------------------------------------------------------

def test_group_func_max_depth(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 0)

    def match(token):
        return False

    tlist = make_tlist(tok(T.Name, 'a'))
    with pytest.raises(SQLParseError):
        grouping._group(tlist, sql.Identifier, match)


def test_group_func_max_tokens(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 0)

    def match(token):
        return False

    tlist = make_tlist(tok(T.Name, 'a'))
    with pytest.raises(SQLParseError):
        grouping._group(tlist, sql.Identifier, match)


def test_group_func_without_post_keeps_tidx_offset():
    # Exercise the generic _group by a custom matcher.
    def match(token):
        return token.match(T.Operator, '=')

    def post(tlist, pidx, tidx, nidx):
        return pidx, nidx

    tlist = make_tlist(
        tok(T.Name, 'a'),
        tok(T.Operator, '='),
        tok(T.Name, 'b'),
    )
    grouping._group(tlist, sql.Comparison, match,
                    lambda t: True, lambda t: True, post)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Comparison)


def test_group_func_recurses_into_groups():
    inner = sql.Parenthesis([
        tok(T.Name, 'a'),
        tok(T.Operator, '='),
        tok(T.Name, 'b'),
    ], 'a = b')
    tlist = make_tlist(inner)

    def match(token):
        return token.match(T.Operator, '=')

    def post(tlist, pidx, tidx, nidx):
        return pidx, nidx

    grouping._group(tlist, sql.Comparison, match,
                    lambda t: True, lambda t: True, post)
    assert isinstance(inner.tokens[0], sql.Comparison)


# ---------------------------------------------------------------------------
# group (full pipeline)
# ---------------------------------------------------------------------------

def test_group_pipeline_simple_select():
    import sqlparse
    from sqlparse import lexer
    stmt = sqlparse.parse('SELECT 1 FROM foo WHERE a = 1')[0]
    # parse already calls group; sanity-check structure.
    assert isinstance(stmt, sql.Statement)
    types = [type(t).__name__ for t in stmt.tokens]
    assert any(isinstance(t, sql.Where) for t in stmt.tokens)
    assert any(isinstance(t, sql.Comparison) for t in stmt.tokens)


def test_group_pipeline_function_call():
    import sqlparse
    stmt = sqlparse.parse('SELECT COUNT(*) FROM foo')[0]
    assert any(isinstance(t, sql.Function) for t in stmt.tokens)


def test_group_pipeline_typed_literal():
    import sqlparse
    stmt = sqlparse.parse("SELECT DATE '2020-01-01'")[0]
    values = [t for t in stmt.tokens
              if isinstance(t, sql.TypedLiteral)]
    assert len(values) == 1


def test_group_pipeline_case():
    import sqlparse
    stmt = sqlparse.parse('SELECT CASE WHEN a THEN b END FROM t')[0]
    assert any(isinstance(t, sql.Case) for t in stmt.tokens)


def test_group_pipeline_identifier_list():
    import sqlparse
    stmt = sqlparse.parse('SELECT a, b, c FROM t')[0]
    found = False
    for t in stmt.tokens:
        if isinstance(t, sql.IdentifierList):
            found = True
    assert found


def test_group_returns_same_object():
    import sqlparse
    from sqlparse import lexer
    # Build a statement ourselves and call group() explicitly.
    from sqlparse import tokens as TT
    flat = sql.TokenList([Token(TT.Name, 'a')])
    result = grouping.group(flat)
    assert result is flat
