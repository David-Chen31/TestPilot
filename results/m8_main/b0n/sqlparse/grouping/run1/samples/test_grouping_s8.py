import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine import grouping
from sqlparse.exceptions import SQLParseError


def _tokenize(lexer, text):
    return list(lexer.get_tokens(text))


def build(text):
    from sqlparse import lexer
    stream = lexer.tokenize(text)
    stmt = sql.Statement(stream)
    return stmt


def flat(text):
    from sqlparse import lexer
    return sql.TokenList(lexer.tokenize(text))


# ---------------------------------------------------------------------------
# Helper-based matching groups
# ---------------------------------------------------------------------------

def test_group_brackets():
    tl = flat('[a b]')
    grouping.group_brackets(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.SquareBrackets)


def test_group_parenthesis():
    tl = flat('(a b)')
    grouping.group_parenthesis(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Parenthesis)


def test_group_parenthesis_unbalanced_close():
    tl = flat('a ) b')
    # Unbalanced close should be left alone; no exception
    grouping.group_parenthesis(tl)
    assert not any(isinstance(t, sql.Parenthesis) for t in tl.tokens)


def test_group_parenthesis_unbalanced_open():
    tl = flat('a ( b')
    grouping.group_parenthesis(tl)
    assert not any(isinstance(t, sql.Parenthesis) for t in tl.tokens)


def test_group_parenthesis_nested():
    tl = flat('(a (b) c)')
    grouping.group_parenthesis(tl)
    assert len(tl.tokens) == 1
    outer = tl.tokens[0]
    assert isinstance(outer, sql.Parenthesis)
    inner = [t for t in outer.tokens if isinstance(t, sql.Parenthesis)]
    assert len(inner) == 1


def test_group_case():
    tl = flat('CASE WHEN a THEN b END')
    grouping.group_case(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Case)


def test_group_if():
    tl = flat('IF a THEN b')
    grouping.group_if(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.If)


def test_group_for():
    tl = flat('FOR i IN x')
    grouping.group_for(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.For)


def test_group_begin():
    tl = flat('BEGIN a b END')
    grouping.group_begin(tl)
    assert len(tl.tokens) == 1
    assert isinstance(tl.tokens[0], sql.Begin)


# ---------------------------------------------------------------------------
# _group based groupings
# ---------------------------------------------------------------------------

def test_group_typecasts():
    tl = flat('a::int')
    grouping.group_typecasts(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1


def test_group_typecasts_no_match():
    tl = flat('a b c')
    grouping.group_typecasts(tl)
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


def test_group_tzcasts():
    from sqlparse import lexer
    tokens = list(lexer.tokenize('a AS TZ'))
    # Manually set a TZCast token: replace token with ttype Keyword.TZCast
    # Actually easier: build tokens manually.
    a = sql.Token(T.Name, 'a')
    ws = sql.Token(T.Whitespace, ' ')
    tz = sql.Token(T.Keyword.TZCast, 'TZ')
    tl = sql.TokenList([a, ws, tz])
    grouping.group_tzcasts(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


def test_group_tzcasts_valid_next_whitespace():
    a = sql.Token(T.Name, 'a')
    tz = sql.Token(T.Keyword.TZCast, 'TZ')
    ws = sql.Token(T.Whitespace, ' ')
    tl = sql.TokenList([a, ws, tz, ws])
    grouping.group_tzcasts(tl)
    # prev exists, next exists and is whitespace -> grouped
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


def test_group_tzcasts_invalid_next():
    a = sql.Token(T.Name, 'a')
    tz = sql.Token(T.Keyword.TZCast, 'TZ')
    other = sql.Token(T.Name, 'b')
    tl = sql.TokenList([a, tz, other])
    grouping.group_tzcasts(tl)
    # next is 'b' Name, which is not whitespace/AS/M_CLOSE -> no grouping
    # Actually prev exists and valid_next fails -> no group.
    assert not any(isinstance(t, sql.Identifier) for t in tl.tokens)


def test_group_typed_literal():
    tl = flat("DATE '2020-01-01'")
    grouping.group_typed_literal(tl)
    assert any(isinstance(t, sql.TypedLiteral) for t in tl.tokens)


def test_group_typed_literal_extend():
    tl = flat("INTERVAL '1' DAY")
    grouping.group_typed_literal(tl)
    assert any(isinstance(t, sql.TypedLiteral) for t in tl.tokens)


def test_group_period():
    tl = flat('a.b')
    grouping.group_period(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1


def test_group_period_arrow():
    tl = flat('a -> b')
    grouping.group_period(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1


def test_group_as():
    tl = flat('a AS b')
    grouping.group_as(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1


def test_group_as_null_prev():
    tl = flat('NULL AS b')
    grouping.group_as(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1


def test_group_assignment():
    tl = flat('a := b')
    grouping.group_assignment(tl)
    assigns = [t for t in tl.tokens if isinstance(t, sql.Assignment)]
    assert len(assigns) == 1


def test_group_assignment_semicolon():
    tl = flat('a := b ; c')
    grouping.group_assignment(tl)
    assigns = [t for t in tl.tokens if isinstance(t, sql.Assignment)]
    assert len(assigns) == 1


def test_group_comparison():
    tl = flat('a = b')
    grouping.group_comparison(tl)
    comps = [t for t in tl.tokens if isinstance(t, sql.Comparison)]
    assert len(comps) == 1


def test_group_comparison_null():
    tl = flat('a IS NULL')
    # normalized comparison 'IS' is T.Keyword, not Operator.Comparison.
    # Build manually
    a = sql.Token(T.Name, 'a')
    ws = sql.Token(T.Whitespace, ' ')
    op = sql.Token(T.Operator.Comparison, '=')
    ws2 = sql.Token(T.Whitespace, ' ')
    null = sql.Token(T.Keyword, 'NULL')
    tl = sql.TokenList([a, ws, op, ws2, null])
    grouping.group_comparison(tl)
    assert any(isinstance(t, sql.Comparison) for t in tl.tokens)


def test_group_identifier():
    tl = flat('foo bar')
    grouping.group_identifier(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 2


def test_group_identifier_symbol():
    from sqlparse import lexer
    tl = sql.TokenList(lexer.tokenize('"foo"'))
    grouping.group_identifier(tl)
    assert any(isinstance(t, sql.Identifier) for t in tl.tokens)


def test_group_over():
    tl = flat('OVER (x)')
    grouping.group_over(tl)
    overs = [t for t in tl.tokens if isinstance(t, sql.Over)]
    assert len(overs) == 1


def test_group_over_name_next():
    # OVER name (no parenthesis) should not group
    o = sql.Token(T.Keyword, 'OVER')
    ws = sql.Token(T.Whitespace, ' ')
    name = sql.Token(T.Name, 'x')
    tl = sql.TokenList([o, ws, name])
    grouping.group_over(tl)
    assert not any(isinstance(t, sql.Over) for t in tl.tokens)


def test_group_arrays():
    tl = flat('a[1]')
    grouping.group_arrays(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1


def test_group_operator():
    tl = flat('a + b')
    grouping.group_operator(tl)
    ops = [t for t in tl.tokens if isinstance(t, sql.Operation)]
    assert len(ops) == 1


def test_group_operator_wildcard():
    tl = flat('a * b')
    grouping.group_operator(tl)
    ops = [t for t in tl.tokens if isinstance(t, sql.Operation)]
    assert len(ops) == 1


def test_group_identifier_list():
    tl = flat('a, b, c')
    grouping.group_identifier_list(tl)
    lists = [t for t in tl.tokens if isinstance(t, sql.IdentifierList)]
    assert len(lists) == 1


def test_group_identifier_list_no_match():
    tl = flat('a b c')
    grouping.group_identifier_list(tl)
    assert not any(isinstance(t, sql.IdentifierList) for t in tl.tokens)


def test_group_comments():
    from sqlparse import lexer
    tl = sql.TokenList(lexer.tokenize('-- hi\na'))
    grouping.group_comments(tl)
    comments = [t for t in tl.tokens if isinstance(t, sql.Comment)]
    assert len(comments) == 1


def test_group_comments_only_comments():
    from sqlparse import lexer
    tl = sql.TokenList(lexer.tokenize('-- hi'))
    # no terminator: should not crash
    grouping.group_comments(tl)
    assert isinstance(tl, sql.TokenList)


def test_group_where():
    tl = flat('WHERE a = 1')
    grouping.group_where(tl)
    wheres = [t for t in tl.tokens if isinstance(t, sql.Where)]
    assert len(wheres) == 1


def test_group_where_unterminated():
    tl = flat('WHERE a = 1')
    grouping.group_where(tl)
    assert any(isinstance(t, sql.Where) for t in tl.tokens)


def test_group_aliased():
    tl = flat('a 1')
    grouping.group_aliased(tl)
    # a is Identifier and next is Number -> alias
    # Actually Number alone won't be aliased unless previous Identifier.
    # Let's test function alias: f() 1
    tl2 = flat('f() 1')
    grouping.group_functions(tl2)
    grouping.group_aliased(tl2)
    assert any(isinstance(t, sql.Identifier) for t in tl2.tokens)


def test_group_functions():
    tl = flat('foo(a)')
    grouping.group_functions(tl)
    funcs = [t for t in tl.tokens if isinstance(t, sql.Function)]
    assert len(funcs) == 1


def test_group_functions_create_table_no_as():
    from sqlparse import lexer
    tl = sql.TokenList(lexer.tokenize('CREATE TABLE foo'))
    grouping.group_functions(tl)
    # should return early, no function grouping
    assert not any(isinstance(t, sql.Function) for t in tl.tokens)


def test_group_functions_with_over():
    tl = flat('foo(a) OVER (b)')
    grouping.group_over(tl)
    grouping.group_functions(tl)
    funcs = [t for t in tl.tokens if isinstance(t, sql.Function)]
    assert len(funcs) == 1
    assert isinstance(funcs[0].tokens[-1], sql.Over) or any(
        isinstance(t, sql.Over) for t in funcs[0].tokens
    )


def test_group_order():
    tl = flat('a DESC')
    grouping.group_identifier(tl)
    grouping.group_order(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1


def test_group_order_number():
    tl = flat('1 DESC')
    grouping.group_order(tl)
    ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1


def test_align_comments():
    from sqlparse import lexer
    tl = sql.TokenList(lexer.tokenize('foo -- c'))
    grouping.group_comments(tl)
    grouping.group_identifier(tl)
    grouping.align_comments(tl)
    # Should have aligned comment into TokenList
    assert any(isinstance(t, sql.TokenList) for t in tl.tokens)


def test_group_values():
    tl = flat('VALUES (1), (2)')
    grouping.group_parenthesis(tl)
    grouping.group_values(tl)
    vals = [t for t in tl.tokens if isinstance(t, sql.Values)]
    assert len(vals) == 1


def test_group_values_no_values():
    tl = flat('SELECT 1')
    grouping.group_values(tl)
    assert not any(isinstance(t, sql.Values) for t in tl.tokens)


# ---------------------------------------------------------------------------
# group() end-to-end
# ---------------------------------------------------------------------------

def test_group_main():
    stmt = build('SELECT a, b FROM t WHERE a = 1')
    result = grouping.group(stmt)
    assert result is stmt
    # should contain an identifier list somewhere
    found = False
    for tok in stmt.flatten():
        if isinstance(tok, sql.IdentifierList):
            found = True
            break
    assert found


def test_group_function():
    stmt = build('SELECT foo(a)')
    grouping.group(stmt)
    funcs = [t for t in stmt.flatten() if isinstance(t, sql.Function)]
    assert len(funcs) >= 1


# ---------------------------------------------------------------------------
# DoS limits
# ---------------------------------------------------------------------------

def test_max_grouping_depth_exceeded(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 0)
    tl = flat('(a)')
    with pytest.raises(SQLParseError, match='Maximum grouping depth'):
        grouping._group_matching(tl, sql.Parenthesis)


def test_max_grouping_depth_group(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 0)
    tl = flat('a = b')
    with pytest.raises(SQLParseError, match='Maximum grouping depth'):
        grouping._group(tl, sql.Comparison,
                        lambda t: t.ttype == T.Operator.Comparison)


def test_max_grouping_tokens_exceeded(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 1)
    tl = flat('(a b)')
    with pytest.raises(SQLParseError, match='Maximum number of tokens'):
        grouping._group_matching(tl, sql.Parenthesis)


def test_max_grouping_tokens_group(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 1)
    tl = flat('a = b')
    with pytest.raises(SQLParseError, match='Maximum number of tokens'):
        grouping._group(tl, sql.Comparison,
                        lambda t: t.ttype == T.Operator.Comparison)


def test_max_grouping_depth_disabled(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', None)
    tl = flat('(a)')
    grouping._group_matching(tl, sql.Parenthesis)
    assert any(isinstance(t, sql.Parenthesis) for t in tl.tokens)


def test_max_grouping_tokens_disabled(monkeypatch):
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', None)
    tl = flat('a = b')
    grouping._group(tl, sql.Comparison,
                    lambda t: t.ttype == T.Operator.Comparison)
    assert any(isinstance(t, sql.Comparison) for t in tl.tokens)


# ---------------------------------------------------------------------------
# _group_matching recursion into subgroups
# ---------------------------------------------------------------------------

def test_group_matching_recurses_into_group():
    # Build a token list where a Parenthesis is inside an existing group
    inner = sql.Token(T.Name, 'a')
    par = sql.Parenthesis([inner])
    # Wrap par inside a comparison, which is a different class
    comp = sql.Comparison([par])
    tl = sql.TokenList([comp])
    grouping._group_matching(tl, sql.Parenthesis)
    # Should not raise and should recurse into Comparison
    assert isinstance(tl.tokens[0], sql.Comparison)


def test_group_matching_skips_whitespace():
    ws = sql.Token(T.Whitespace, ' ')
    tl = sql.TokenList([ws])
    grouping._group_matching(tl, sql.Parenthesis)
    assert len(tl.tokens) == 1


# ---------------------------------------------------------------------------
# T_NUMERICAL / T_STRING / T_NAME constants
# ---------------------------------------------------------------------------

def test_token_type_constants():
    assert T.Number in grouping.T_NUMERICAL
    assert T.Number.Integer in grouping.T_NUMERICAL
    assert T.Number.Float in grouping.T_NUMERICAL
    assert T.String in grouping.T_STRING
    assert T.String.Single in grouping.T_STRING
    assert T.String.Symbol in grouping.T_STRING
    assert T.Name in grouping.T_NAME
    assert T.Name.Placeholder in grouping.T_NAME
