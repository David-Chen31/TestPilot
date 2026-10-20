import pytest

from sqlparse import sql, tokens as T
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
    _group_matching,
    _group,
)


def token_list(*tokens):
    return sql.TokenList(tokens)


def tok(value, ttype=T.Name):
    return sql.Token(ttype, value)


def ws():
    return sql.Token(T.Whitespace, ' ')


def kw(value):
    return sql.Token(T.Keyword, value)


def op(value, ttype=T.Operator):
    return sql.Token(ttype, value)


def punc(value):
    return sql.Token(T.Punctuation, value)


# ---------------------------------------------------------------------------
# _group_matching / group_brackets / parenthesis / case / if / for / begin
# ---------------------------------------------------------------------------

def test_group_brackets_basic():
    tl = token_list(punc('['), tok('foo'), punc(']'))
    group_brackets(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.SquareBrackets)


def test_group_brackets_unbalanced_skipped():
    tl = token_list(punc(']'), tok('foo'))
    group_brackets(tl)
    # nothing grouped since close appears with empty stack
    assert not any(isinstance(t, sql.SquareBrackets) for t in tl.tokens)


def test_group_parenthesis_basic():
    tl = token_list(punc('('), tok('foo'), punc(')'))
    group_parenthesis(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Parenthesis)


def test_group_parenthesis_nested():
    tl = token_list(
        punc('('), tok('a'), punc('('), tok('b'), punc(')'), punc(')')
    )
    group_parenthesis(tl)
    assert len(tl.tokens) == 1
    outer = tl.tokens[0]
    assert isinstance(outer, sql.Parenthesis)
    inner = [t for t in outer.tokens if isinstance(t, sql.Parenthesis)]
    assert len(inner) == 1


def test_group_matching_recurses_into_existing_groups():
    inner = token_list(punc('('), tok('x'), punc(')'))
    inner_group = sql.Parenthesis(inner.tokens)
    tl = token_list(
        punc('['), inner_group, punc(']')
    )
    group_brackets(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.SquareBrackets)


def test_group_matching_depth_limit(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 0)
    tl = token_list(punc('('), punc(')'))
    with pytest.raises(SQLParseError):
        group_parenthesis(tl)


def test_group_matching_token_limit(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 1)
    tl = token_list(punc('('), punc(')'))
    with pytest.raises(SQLParseError):
        group_parenthesis(tl)


def test_group_case_basic():
    tl = token_list(kw('CASE'), tok('x'), kw('END'))
    group_case(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Case)


def test_group_if_basic():
    tl = token_list(kw('IF'), tok('x'), kw('END'))
    group_if(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.If)


def test_group_for_basic():
    tl = token_list(kw('FOR'), tok('x'), kw('END'))
    group_for(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.For)


def test_group_begin_basic():
    tl = token_list(kw('BEGIN'), tok('x'), kw('END'))
    group_begin(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Begin)


# ---------------------------------------------------------------------------
# group_typecasts
# ---------------------------------------------------------------------------

def test_group_typecasts():
    tl = token_list(tok('foo'), punc('::'), tok('int'))
    group_typecasts(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Identifier)


def test_group_typecasts_no_cast():
    tl = token_list(tok('foo'), tok('bar'))
    group_typecasts(tl)
    assert len(tl.tokens) == 2


# ---------------------------------------------------------------------------
# group_tzcasts
# ---------------------------------------------------------------------------

def test_group_tzcasts_with_as():
    tl = token_list(
        tok('foo'), sql.Token(T.Keyword.TZCast, 'AT'), ws(),
        kw('AS'), tok('tz')
    )
    group_tzcasts(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


def test_group_tzcasts_invalid_next():
    tl = token_list(tok('foo'), sql.Token(T.Keyword.TZCast, '/'))
    group_tzcasts(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_typed_literal
# ---------------------------------------------------------------------------

def test_group_typed_literal_basic():
    tl = token_list(
        sql.Token(T.Name.Builtin, 'DATE'), ws(),
        sql.Token(T.String.Single, "'2020-01-01'")
    )
    group_typed_literal(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.TypedLiteral)


# ---------------------------------------------------------------------------
# group_period
# ---------------------------------------------------------------------------

def test_group_period():
    tl = token_list(tok('a'), punc('.'), tok('b'))
    group_period(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Identifier)


def test_group_period_json_operator():
    tl = token_list(tok('a'), op('->'), tok('b'))
    group_period(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Identifier)


def test_group_period_invalid_prev():
    tl = token_list(op(','), punc('.'), tok('b'))
    group_period(tl)
    # No grouping because prev is not a valid identifier head
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_as
# ---------------------------------------------------------------------------

def test_group_as():
    tl = token_list(tok('foo'), ws(), kw('AS'), ws(), tok('bar'))
    group_as(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


def test_group_as_keyword_prev_rejected():
    tl = token_list(kw('SELECT'), ws(), kw('AS'), ws(), tok('bar'))
    group_as(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_assignment
# ---------------------------------------------------------------------------

def test_group_assignment():
    tl = token_list(tok('x'), sql.Token(T.Assignment, ':='), tok('y'))
    group_assignment(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Assignment)


def test_group_assignment_with_semicolon():
    tl = token_list(
        tok('x'), ws(), sql.Token(T.Assignment, ':='), ws(),
        tok('y'), punc(';'), tok('z')
    )
    group_assignment(tl)
    grp = tl.tokens[0]
    assert isinstance(grp, sql.Assignment)
    assert ';' in grp.value


def test_group_assignment_keyword_rejected():
    tl = token_list(tok('x'), sql.Token(T.Assignment, ':='), kw('SELECT'))
    group_assignment(tl)
    assert not any(isinstance(t, sql.Assignment) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_comparison
# ---------------------------------------------------------------------------

def test_group_comparison():
    tl = token_list(tok('a'), ws(), op('=', T.Operator.Comparison), ws(),
                    tok('b'))
    group_comparison(tl)
    assert any(isinstance(t, sql.Comparison) for t in tl.tokens)


def test_group_comparison_null():
    tl = token_list(tok('a'), ws(), op('=', T.Operator.Comparison), ws(),
                    kw('NULL'))
    group_comparison(tl)
    assert any(isinstance(t, sql.Comparison) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_identifier
# ---------------------------------------------------------------------------

def test_group_identifier():
    tl = token_list(tok('foo'), ws(), tok('bar'))
    group_identifier(tl)
    idents = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(idents) == 2


# ---------------------------------------------------------------------------
# group_over
# ---------------------------------------------------------------------------

def test_group_over():
    paren = sql.Parenthesis([tok('x')])
    tl = token_list(
        sql.Token(T.Keyword, 'OVER'), ws(), paren
    )
    group_over(tl)
    assert any(isinstance(t, sql.Over) for t in tl.tokens)


def test_group_over_no_parenthesis():
    tl = token_list(sql.Token(T.Keyword, 'OVER'), ws(), tok('x'))
    group_over(tl)
    assert not any(isinstance(t, sql.Over) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_arrays
# ---------------------------------------------------------------------------

def test_group_arrays():
    brackets = sql.SquareBrackets([tok('1')])
    tl = token_list(tok('arr'), brackets)
    group_arrays(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_operator
# ---------------------------------------------------------------------------

def test_group_operator():
    tl = token_list(tok('a'), ws(), op('+'), ws(), tok('b'))
    group_operator(tl)
    assert any(isinstance(t, sql.Operation) for t in tl.tokens)


def test_group_operator_current_date():
    tl = token_list(kw('CURRENT_DATE'), ws(), op('+'), ws(), tok('b'))
    group_operator(tl)
    assert any(isinstance(t, sql.Operation) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_identifier_list
# ---------------------------------------------------------------------------

def test_group_identifier_list():
    tl = token_list(tok('a'), punc(','), tok('b'))
    group_identifier_list(tl)
    assert any(isinstance(t, sql.IdentifierList) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_comments
# ---------------------------------------------------------------------------

def test_group_comments_basic():
    tl = token_list(
        sql.Token(T.Comment.Single, '-- foo'), ws(), tok('x')
    )
    group_comments(tl)
    # trailing comment trimmed at newline/terminator -> still grouped
    assert any(isinstance(t, sql.Comment) for t in tl.tokens)


def test_group_comments_trailing_only_does_not_loop_forever():
    tl = token_list(
        sql.Token(T.Comment.Single, '-- foo'),
        sql.Token(T.Comment.Single, '-- bar'),
    )
    group_comments(tl)
    # All comment tokens -> end is None -> break, nothing grouped
    assert not any(isinstance(t, sql.Comment) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_where
# ---------------------------------------------------------------------------

def test_group_where():
    tl = token_list(kw('WHERE'), ws(), tok('x'), punc(';'))
    group_where(tl)
    assert any(isinstance(t, sql.Where) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_aliased
# ---------------------------------------------------------------------------

def test_group_aliased():
    fn = sql.Function([tok('f'), sql.Parenthesis([tok('x')])])
    tl = token_list(fn, ws(), tok('alias'))
    group_aliased(tl)
    assert any(isinstance(t, sql.Identifier) and
               any(isinstance(s, sql.Function) for s in t.tokens)
               for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_functions
# ---------------------------------------------------------------------------

def test_group_functions_basic():
    tl = token_list(tok('foo'), sql.Parenthesis([tok('x')]))
    group_functions(tl)
    assert isinstance(tl.tokens[0], sql.Function)


def test_group_functions_create_table_skipped():
    tl = token_list(
        kw('CREATE'), ws(), kw('TABLE'), ws(),
        tok('foo'), sql.Parenthesis([tok('x')])
    )
    group_functions(tl)
    assert not any(isinstance(t, sql.Function) for t in tl.tokens)


def test_group_functions_create_table_with_as():
    tl = token_list(
        kw('CREATE'), ws(), kw('TABLE'), ws(), tok('foo'), ws(),
        kw('AS'), ws(), tok('bar'), sql.Parenthesis([tok('x')])
    )
    group_functions(tl)
    # has_as True -> not skipped
    assert any(isinstance(t, sql.Function) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_order
# ---------------------------------------------------------------------------

def test_group_order():
    tl = token_list(tok('foo'), ws(), sql.Token(T.Keyword.Order, 'DESC'))
    group_order(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# align_comments
# ---------------------------------------------------------------------------

def test_align_comments():
    ident = sql.Identifier([tok('foo')])
    comment = sql.Comment([sql.Token(T.Comment.Single, '-- x')])
    tl = token_list(ident, ws(), comment)
    align_comments(tl)
    assert any(isinstance(t, sql.TokenList) and t is not tl
               for t in tl.tokens)


def test_align_comments_no_prev_token_list():
    comment = sql.Comment([sql.Token(T.Comment.Single, '-- x')])
    tl = token_list(ws(), comment)
    align_comments(tl)
    assert not any(isinstance(t, sql.TokenList) and t is not tl
                   for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_values
# ---------------------------------------------------------------------------

def test_group_values():
    paren1 = sql.Parenthesis([tok('1')])
    paren2 = sql.Parenthesis([tok('2')])
    tl = token_list(kw('VALUES'), ws(), paren1, punc(','), ws(), paren2)
    group_values(tl)
    assert any(isinstance(t, sql.Values) for t in tl.tokens)


def test_group_values_no_parenthesis():
    tl = token_list(kw('VALUES'), ws(), tok('x'))
    group_values(tl)
    assert not any(isinstance(t, sql.Values) for t in tl.tokens)


# ---------------------------------------------------------------------------
# _group
# ---------------------------------------------------------------------------

def test_group_token_limit(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 1)
    tl = token_list(tok('a'), op('+'), tok('b'))
    with pytest.raises(SQLParseError):
        _group(tl, sql.Operation,
               match=lambda t: t.match(T.Operator, '+'))


def test_group_depth_limit(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 0)
    tl = token_list(tok('a'), op('+'), tok('b'))
    with pytest.raises(SQLParseError):
        _group(tl, sql.Operation,
               match=lambda t: t.match(T.Operator, '+'))


# ---------------------------------------------------------------------------
# group() end-to-end
# ---------------------------------------------------------------------------

def test_group_end_to_end():
    from sqlparse import tokens as T
    tokens = [
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Wildcard, '*'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'FROM'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'foo'),
    ]
    stmt = sql.Statement(tokens)
    result = group(stmt)
    assert result is stmt
    # FROM should be grouped into an Identifier at end or stay; foo grouped too
    assert any(isinstance(t, sql.Identifier) for t in stmt.tokens)


def test_group_returns_same_object():
    stmt = sql.Statement([sql.Token(T.Name, 'foo')])
    assert group(stmt) is stmt
