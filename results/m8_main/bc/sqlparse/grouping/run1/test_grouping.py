"""Tests for sqlparse.engine.grouping module."""

import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.exceptions import SQLParseError
from sqlparse.engine import grouping
from sqlparse.lexer import tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tokenlist(*tokens):
    """Build a TokenList from a sequence of Token objects.

    ``sql.TokenList`` requires a mutable list because grouping operations
    assign into ``tlist.tokens``; passing a tuple raises a TypeError.
    """
    return sql.TokenList(list(tokens))


def tok(value, ttype=T.Text):
    return sql.Token(ttype, value)


def ws():
    return sql.Token(T.Whitespace, ' ')


def lex_tokenlist(text):
    """Build a TokenList using the real lexer.

    This guarantees the tokens have exactly the token types/value that the
    rest of sqlparse (and therefore the grouping patterns such as
    ``sql.TypedLiteral.M_OPEN``) expects.
    """
    return sql.TokenList([sql.Token(ttype, value)
                          for ttype, value in tokenize(text)])


# ---------------------------------------------------------------------------
# Constants / module-level state
# ---------------------------------------------------------------------------

def test_module_constants():
    assert grouping.MAX_GROUPING_DEPTH == 100
    assert grouping.MAX_GROUPING_TOKENS == 10000
    assert grouping.T_NUMERICAL == (T.Number, T.Number.Integer, T.Number.Float)
    assert grouping.T_STRING == (T.String, T.String.Single, T.String.Symbol)
    assert grouping.T_NAME == (T.Name, T.Name.Placeholder)


# ---------------------------------------------------------------------------
# group_brackets
# ---------------------------------------------------------------------------

def test_group_brackets():
    tl = make_tokenlist(
        tok('[', T.Punctuation),
        tok('a', T.Name),
        tok(']', T.Punctuation),
    )
    grouping.group_brackets(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.SquareBrackets)
    assert tl.tokens[0].value == '[a]'


def test_group_brackets_unbalanced():
    tl = make_tokenlist(tok(']', T.Punctuation), tok('a', T.Name))
    grouping.group_brackets(tl)
    assert len(tl.tokens) == 2
    assert not any(isinstance(t, sql.SquareBrackets) for t in tl.tokens)


def test_group_brackets_nested():
    tl = make_tokenlist(
        tok('[', T.Punctuation),
        tok('[', T.Punctuation),
        tok(']', T.Punctuation),
        tok(']', T.Punctuation),
    )
    grouping.group_brackets(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.SquareBrackets)


# ---------------------------------------------------------------------------
# group_parenthesis
# ---------------------------------------------------------------------------

def test_group_parenthesis():
    tl = make_tokenlist(
        tok('(', T.Punctuation),
        tok('a', T.Name),
        tok(')', T.Punctuation),
    )
    grouping.group_parenthesis(tl)
    assert len(tl.tokens) == 1
    g = tl.tokens[0]
    assert isinstance(g, sql.Parenthesis)
    assert g.value == '(a)'


def test_group_parenthesis_unbalanced_close():
    tl = make_tokenlist(tok(')', T.Punctuation))
    grouping.group_parenthesis(tl)
    assert len(tl.tokens) == 1
    assert not isinstance(tl.tokens[0], sql.Parenthesis)


def test_group_parenthesis_unbalanced_open():
    tl = make_tokenlist(tok('(', T.Punctuation), tok('a', T.Name))
    grouping.group_parenthesis(tl)
    assert len(tl.tokens) == 2
    assert not any(isinstance(t, sql.Parenthesis) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_case
# ---------------------------------------------------------------------------

def test_group_case():
    tl = make_tokenlist(
        tok('CASE', T.Keyword),
        ws(),
        tok('WHEN', T.Keyword),
        ws(),
        tok('a', T.Name),
        ws(),
        tok('THEN', T.Keyword),
        ws(),
        tok('1', T.Number.Integer),
        ws(),
        tok('END', T.Keyword),
    )
    grouping.group_case(tl)
    assert any(isinstance(t, sql.Case) for t in tl.tokens)


def test_group_case_unbalanced():
    tl = make_tokenlist(tok('CASE', T.Keyword), ws(), tok('x', T.Name))
    grouping.group_case(tl)
    assert not any(isinstance(t, sql.Case) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_if / group_for / group_begin
# ---------------------------------------------------------------------------

def test_group_if():
    tl = make_tokenlist(
        tok('IF', T.Keyword),
        ws(),
        tok('a', T.Name),
        ws(),
        tok('END IF', T.Keyword),
    )
    grouping.group_if(tl)
    assert any(isinstance(t, sql.If) for t in tl.tokens)


def test_group_for():
    # ``sql.For.M_CLOSE`` is not the plain END keyword; inspect the actual
    # close pattern so the test tracks the implementation.
    close_ttype, close_value = sql.For.M_CLOSE
    tl = make_tokenlist(
        tok('FOR', T.Keyword),
        ws(),
        tok('x', T.Name),
        ws(),
        tok(close_value, close_ttype),
    )
    grouping.group_for(tl)
    assert any(isinstance(t, sql.For) for t in tl.tokens)


def test_group_for_no_close():
    tl = make_tokenlist(tok('FOR', T.Keyword), ws(), tok('x', T.Name))
    grouping.group_for(tl)
    assert not any(isinstance(t, sql.For) for t in tl.tokens)


def test_group_begin():
    tl = make_tokenlist(
        tok('BEGIN', T.Keyword),
        ws(),
        tok('x', T.Name),
        ws(),
        tok('END', T.Keyword),
    )
    grouping.group_begin(tl)
    assert any(isinstance(t, sql.Begin) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_typecasts
# ---------------------------------------------------------------------------

def test_group_typecasts():
    tl = make_tokenlist(
        tok('1', T.Number.Integer),
        ws(),
        tok('::', T.Punctuation),
        ws(),
        tok('int', T.Name),
    )
    grouping.group_typecasts(tl)
    assert any(isinstance(t, sql.Identifier) and '::' in t.value
               for t in tl.tokens)


def test_group_typecasts_no_prev():
    tl = make_tokenlist(tok('::', T.Punctuation), tok('int', T.Name))
    grouping.group_typecasts(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_tzcasts
# ---------------------------------------------------------------------------

def test_group_tzcasts_with_as():
    tl = make_tokenlist(
        tok('foo', T.Name),
        ws(),
        tok('AT TIME ZONE', T.Keyword.TZCast),
        ws(),
        tok('AS', T.Keyword),
    )
    grouping.group_tzcasts(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


def test_group_tzcasts_invalid_next():
    tl = make_tokenlist(
        tok('foo', T.Name),
        ws(),
        tok('AT TIME ZONE', T.Keyword.TZCast),
        tok('bar', T.Name),
    )
    grouping.group_tzcasts(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_typed_literal
# ---------------------------------------------------------------------------

def test_group_typed_literal():
    # A typed literal looks like: DATE '2020-01-01'. The opener must match
    # sql.TypedLiteral.M_OPEN and the closer must match M_CLOSE. The exact
    # token type of the ``DATE`` keyword depends on the lexer (it is a
    # "soft" keyword), so build the input through the real lexer to get the
    # token types the M_OPEN/M_CLOSE patterns were written against.
    # A leading name supplies the non-None previous token required by _group.
    tl = lex_tokenlist("x DATE '2020-01-01'")
    grouping.group_typed_literal(tl)
    lit = [t for t in tl.tokens if isinstance(t, sql.TypedLiteral)]
    assert len(lit) == 1
    assert lit[0].value == "DATE '2020-01-01'"


def test_group_typed_literal_no_match():
    tl = make_tokenlist(
        tok('foo', T.Name),
        ws(),
        tok('bar', T.Name),
    )
    grouping.group_typed_literal(tl)
    assert not any(isinstance(t, sql.TypedLiteral) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_period
# ---------------------------------------------------------------------------

def test_group_period():
    tl = make_tokenlist(
        tok('a', T.Name),
        tok('.', T.Punctuation),
        tok('b', T.Name),
    )
    grouping.group_period(tl)
    assert any(isinstance(t, sql.Identifier) and '.' in t.value
               for t in tl.tokens)


def test_group_period_arrow():
    tl = make_tokenlist(
        tok('a', T.Name),
        tok('->', T.Operator),
        tok('b', T.Name),
    )
    grouping.group_period(tl)
    assert any('->' in t.value for t in tl.tokens)


def test_group_period_invalid_prev():
    tl = make_tokenlist(
        tok('1', T.Number.Integer),
        tok('.', T.Punctuation),
        tok('b', T.Name),
    )
    grouping.group_period(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_as
# ---------------------------------------------------------------------------

def test_group_as():
    tl = make_tokenlist(
        tok('foo', T.Name),
        ws(),
        tok('AS', T.Keyword),
        ws(),
        tok('bar', T.Name),
    )
    grouping.group_as(tl)
    assert any(isinstance(t, sql.Identifier) and 'AS' in t.value
               for t in tl.tokens)


def test_group_as_no_valid_prev():
    tl = make_tokenlist(
        tok('SELECT', T.Keyword),
        ws(),
        tok('AS', T.Keyword),
        ws(),
        tok('bar', T.Name),
    )
    grouping.group_as(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_assignment
# ---------------------------------------------------------------------------

def test_group_assignment():
    tl = make_tokenlist(
        tok('a', T.Name),
        ws(),
        tok(':=', T.Assignment),
        ws(),
        tok('1', T.Number.Integer),
    )
    grouping.group_assignment(tl)
    assert any(isinstance(t, sql.Assignment) for t in tl.tokens)


def test_group_assignment_with_semicolon():
    tl = make_tokenlist(
        tok('a', T.Name),
        ws(),
        tok(':=', T.Assignment),
        ws(),
        tok('1', T.Number.Integer),
        tok(';', T.Punctuation),
    )
    grouping.group_assignment(tl)
    grp = [t for t in tl.tokens if isinstance(t, sql.Assignment)]
    assert len(grp) == 1
    assert ';' in grp[0].value


# ---------------------------------------------------------------------------
# group_comparison
# ---------------------------------------------------------------------------

def test_group_comparison():
    tl = make_tokenlist(
        tok('a', T.Name),
        ws(),
        tok('=', T.Operator.Comparison),
        ws(),
        tok('1', T.Number.Integer),
    )
    grouping.group_comparison(tl)
    assert any(isinstance(t, sql.Comparison) for t in tl.tokens)


def test_group_comparison_null():
    tl = make_tokenlist(
        tok('a', T.Name),
        ws(),
        tok('=', T.Operator.Comparison),
        ws(),
        tok('NULL', T.Keyword),
    )
    grouping.group_comparison(tl)
    assert any(isinstance(t, sql.Comparison) for t in tl.tokens)


def test_group_comparison_invalid_operands():
    tl = make_tokenlist(
        tok(';', T.Punctuation),
        ws(),
        tok('=', T.Operator.Comparison),
        ws(),
        tok(';', T.Punctuation),
    )
    grouping.group_comparison(tl)
    assert not any(isinstance(t, sql.Comparison) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_identifier
# ---------------------------------------------------------------------------

def test_group_identifier():
    tl = make_tokenlist(tok('foo', T.Name))
    grouping.group_identifier(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Identifier)


def test_group_identifier_symbol():
    tl = make_tokenlist(tok('foo', T.String.Symbol))
    grouping.group_identifier(tl)
    assert isinstance(tl.tokens[0], sql.Identifier)


def test_group_identifier_number_unchanged():
    tl = make_tokenlist(tok('42', T.Number.Integer))
    grouping.group_identifier(tl)
    assert not isinstance(tl.tokens[0], sql.Identifier)


# ---------------------------------------------------------------------------
# group_over
# ---------------------------------------------------------------------------

def test_group_over():
    paren = sql.Parenthesis([tok('a', T.Name)])
    tl = make_tokenlist(
        tok('OVER', T.Keyword),
        ws(),
        paren,
    )
    grouping.group_over(tl)
    assert any(isinstance(t, sql.Over) for t in tl.tokens)


def test_group_over_name():
    tl = make_tokenlist(
        tok('OVER', T.Keyword),
        ws(),
        tok('win', T.Name),
    )
    grouping.group_over(tl)
    assert any(isinstance(t, sql.Over) for t in tl.tokens)


def test_group_over_no_next():
    tl = make_tokenlist(
        tok('OVER', T.Keyword),
        ws(),
        tok(';', T.Punctuation),
    )
    grouping.group_over(tl)
    assert not any(isinstance(t, sql.Over) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_arrays
# ---------------------------------------------------------------------------

def test_group_arrays():
    brackets = sql.SquareBrackets([tok('1', T.Number.Integer)])
    tl = make_tokenlist(tok('a', T.Name), brackets)
    grouping.group_arrays(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


def test_group_arrays_invalid_prev():
    brackets = sql.SquareBrackets([tok('1', T.Number.Integer)])
    tl = make_tokenlist(tok(';', T.Punctuation), brackets)
    grouping.group_arrays(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_operator
# ---------------------------------------------------------------------------

def test_group_operator():
    tl = make_tokenlist(
        tok('a', T.Name),
        ws(),
        tok('+', T.Operator),
        ws(),
        tok('1', T.Number.Integer),
    )
    grouping.group_operator(tl)
    ops = [t for t in tl.tokens if isinstance(t, sql.Operation)]
    assert len(ops) == 1


def test_group_operator_current_date():
    tl = make_tokenlist(
        tok('CURRENT_DATE', T.Keyword),
        ws(),
        tok('+', T.Operator),
        ws(),
        tok('1', T.Number.Integer),
    )
    grouping.group_operator(tl)
    assert any(isinstance(t, sql.Operation) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_identifier_list
# ---------------------------------------------------------------------------

def test_group_identifier_list():
    tl = make_tokenlist(
        tok('a', T.Name),
        tok(',', T.Punctuation),
        tok('b', T.Name),
    )
    grouping.group_identifier_list(tl)
    assert any(isinstance(t, sql.IdentifierList) for t in tl.tokens)


def test_group_identifier_list_no_comma():
    tl = make_tokenlist(tok('a', T.Name), tok('b', T.Name))
    grouping.group_identifier_list(tl)
    assert not any(isinstance(t, sql.IdentifierList) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_comments
# ---------------------------------------------------------------------------

def test_group_comments():
    tl = make_tokenlist(
        tok('-- comment', T.Comment.Single),
        tok('\n', T.Whitespace.Newline),
        tok('a', T.Name),
    )
    grouping.group_comments(tl)
    assert any(isinstance(t, sql.Comment) for t in tl.tokens)


def test_group_comments_all_comment():
    tl = make_tokenlist(
        tok('-- comment', T.Comment.Single),
        tok('\n', T.Whitespace.Newline),
    )
    grouping.group_comments(tl)
    assert not any(isinstance(t, sql.Comment) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_where
# ---------------------------------------------------------------------------

def test_group_where():
    tl = make_tokenlist(
        tok('WHERE', T.Keyword),
        ws(),
        tok('a', T.Name),
        ws(),
        tok('=', T.Operator.Comparison),
        ws(),
        tok('1', T.Number.Integer),
        ws(),
        tok(';', T.Punctuation),
    )
    grouping.group_where(tl)
    assert any(isinstance(t, sql.Where) for t in tl.tokens)


def test_group_where_no_close():
    tl = make_tokenlist(
        tok('WHERE', T.Keyword),
        ws(),
        tok('a', T.Name),
    )
    grouping.group_where(tl)
    assert any(isinstance(t, sql.Where) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_aliased
# ---------------------------------------------------------------------------

def test_group_aliased():
    paren = sql.Parenthesis([tok('1', T.Number.Integer)])
    alias_id = sql.Identifier([tok('alias', T.Name)])
    tl = make_tokenlist(paren, ws(), alias_id)
    grouping.group_aliased(tl)
    assert any(isinstance(t, sql.Identifier)
               and isinstance(t.tokens[0], sql.Parenthesis)
               for t in tl.tokens)


def test_group_aliased_number():
    tl = make_tokenlist(
        tok('1', T.Number.Integer),
        ws(),
        sql.Identifier([tok('x', T.Name)]),
    )
    grouping.group_aliased(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_functions
# ---------------------------------------------------------------------------

def test_group_functions():
    paren = sql.Parenthesis([tok('1', T.Number.Integer)])
    tl = make_tokenlist(tok('f', T.Name), paren)
    grouping.group_functions(tl)
    assert any(isinstance(t, sql.Function) for t in tl.tokens)


def test_group_functions_with_over():
    paren = sql.Parenthesis([tok('1', T.Number.Integer)])
    over = sql.Over([tok('OVER', T.Keyword)])
    tl = make_tokenlist(tok('f', T.Name), paren, over)
    grouping.group_functions(tl)
    fns = [t for t in tl.tokens if isinstance(t, sql.Function)]
    assert len(fns) == 1
    assert isinstance(fns[0].tokens[-1], sql.Over)


def test_group_functions_create_table_no_as():
    paren = sql.Parenthesis([tok('1', T.Number.Integer)])
    tl = make_tokenlist(
        tok('CREATE', T.Keyword),
        ws(),
        tok('TABLE', T.Keyword),
        ws(),
        tok('f', T.Name),
        paren,
    )
    grouping.group_functions(tl)
    assert not any(isinstance(t, sql.Function) for t in tl.tokens)


def test_group_functions_create_table_with_as():
    paren = sql.Parenthesis([tok('SELECT', T.Keyword)])
    tl = make_tokenlist(
        tok('CREATE', T.Keyword),
        ws(),
        tok('TABLE', T.Keyword),
        ws(),
        tok('f', T.Name),
        paren,
        ws(),
        tok('AS', T.Keyword),
    )
    grouping.group_functions(tl)
    assert any(isinstance(t, sql.Function) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_order
# ---------------------------------------------------------------------------

def test_group_order():
    tl = make_tokenlist(
        sql.Identifier([tok('a', T.Name)]),
        ws(),
        tok('ASC', T.Keyword.Order),
    )
    grouping.group_order(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert any('ASC' in t.value for t in ids)


def test_group_order_number():
    tl = make_tokenlist(
        tok('1', T.Number.Integer),
        ws(),
        tok('DESC', T.Keyword.Order),
    )
    grouping.group_order(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert any('DESC' in t.value for t in ids)


def test_group_order_invalid_prev():
    tl = make_tokenlist(
        tok(';', T.Punctuation),
        ws(),
        tok('ASC', T.Keyword.Order),
    )
    grouping.group_order(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# align_comments
# ---------------------------------------------------------------------------

def test_align_comments():
    inner = sql.Identifier([tok('a', T.Name)])
    comment = sql.Comment([tok('--c', T.Comment.Single)])
    tl = make_tokenlist(inner, ws(), comment)
    grouping.align_comments(tl)
    top = [t for t in tl.tokens if isinstance(t, sql.TokenList)]
    assert len(top) >= 1


def test_align_comments_no_prev_group():
    tl = make_tokenlist(tok('a', T.Name), ws(),
                        sql.Comment([tok('--c', T.Comment.Single)]))
    grouping.align_comments(tl)
    assert all(not isinstance(t, sql.TokenList) or isinstance(t, sql.Comment)
               for t in tl.tokens)


# ---------------------------------------------------------------------------
# group_values
# ---------------------------------------------------------------------------

def test_group_values():
    paren = sql.Parenthesis([tok('1', T.Number.Integer)])
    tl = make_tokenlist(
        tok('VALUES', T.Keyword),
        ws(),
        paren,
    )
    grouping.group_values(tl)
    assert any(isinstance(t, sql.Values) for t in tl.tokens)


def test_group_values_no_paren():
    tl = make_tokenlist(
        tok('VALUES', T.Keyword),
        ws(),
        tok('1', T.Number.Integer),
    )
    grouping.group_values(tl)
    assert not any(isinstance(t, sql.Values) for t in tl.tokens)


def test_group_values_no_values():
    tl = make_tokenlist(tok('a', T.Name))
    grouping.group_values(tl)
    assert not any(isinstance(t, sql.Values) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group (top level)
# ---------------------------------------------------------------------------

def test_group_returns_stmt():
    tl = make_tokenlist(tok('SELECT', T.Keyword), ws(),
                        tok('1', T.Number.Integer))
    result = grouping.group(tl)
    assert result is tl


def test_group_select_name():
    tl = make_tokenlist(tok('a', T.Name))
    grouping.group(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# _group_matching error handling (depth / token limits)
# ---------------------------------------------------------------------------

def test_group_matching_depth_limit(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 0)
    tl = make_tokenlist(tok('(', T.Punctuation), tok(')', T.Punctuation))
    with pytest.raises(SQLParseError):
        grouping._group_matching(tl, sql.Parenthesis, depth=1)


def test_group_matching_token_limit(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 1)
    tl = make_tokenlist(tok('(', T.Punctuation), tok(')', T.Punctuation))
    with pytest.raises(SQLParseError):
        grouping._group_matching(tl, sql.Parenthesis)


def test_group_matching_nested_group_recursion():
    inner = sql.Case([tok('CASE', T.Keyword)])
    tl = make_tokenlist(inner)
    grouping._group_matching(tl, sql.Parenthesis)


# ---------------------------------------------------------------------------
# _group error handling (depth / token limits)
# ---------------------------------------------------------------------------

def test_underscore_group_depth_limit(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 0)
    tl = make_tokenlist(tok('a', T.Name))
    with pytest.raises(SQLParseError):
        grouping._group(tl, sql.Identifier, lambda t: False, depth=1)


def test_underscore_group_token_limit(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 1)
    tl = make_tokenlist(tok('a', T.Name), tok('b', T.Name))
    with pytest.raises(SQLParseError):
        grouping._group(tl, sql.Identifier, lambda t: False)


def test_underscore_group_basic():
    tl = make_tokenlist(
        tok('a', T.Name),
        tok(',', T.Punctuation),
        tok('b', T.Name),
    )

    def match(t):
        return t.match(T.Punctuation, ',')

    def post(tlist, pidx, tidx, nidx):
        return pidx, nidx

    grouping._group(tl, sql.IdentifierList, match,
                    valid_prev=lambda t: True,
                    valid_next=lambda t: True,
                    post=post, extend=True)
    assert any(isinstance(t, sql.IdentifierList) for t in tl.tokens)


# ---------------------------------------------------------------------------
# Limits disabled (None) should not raise
# ---------------------------------------------------------------------------

def test_limits_disabled(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', None)
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', None)
    tl = make_tokenlist(tok('(', T.Punctuation), tok(')', T.Punctuation))
    grouping._group_matching(tl, sql.Parenthesis, depth=1000)
    assert any(isinstance(t, sql.Parenthesis) for t in tl.tokens)


def test_underscore_group_limits_disabled(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', None)
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', None)
    tl = make_tokenlist(tok('a', T.Name))
    grouping._group(tl, sql.Identifier, lambda t: False, depth=1000)
