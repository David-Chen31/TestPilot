import pytest

from sqlparse import tokens as T
from sqlparse.sql import (
    NameAliasMixin, Token, TokenList, Statement, Identifier, IdentifierList,
    TypedLiteral, Parenthesis, SquareBrackets, Assignment, If, For, Comparison,
    Comment, Where, Over, Having, Case, Function, Begin, Operation, Values,
    Command,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tokens():
    """A small helper structure mirroring 'a.b AS c' token stream."""
    return [
        Token(T.Name, 'a'),
        Token(T.Punctuation, '.'),
        Token(T.Name, 'b'),
        Token(T.Whitespace, ' '),
        Token(T.Keyword, 'AS'),
        Token(T.Whitespace, ' '),
        Token(T.Name, 'c'),
    ]


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

def test_token_basic_attributes():
    tok = Token(T.Name, 'foo')
    assert tok.value == 'foo'
    assert tok.ttype == T.Name
    assert tok.parent is None
    assert tok.is_group is False
    assert tok.is_keyword is False
    assert tok.is_whitespace is False
    assert tok.is_newline is False
    assert tok.normalized == 'foo'


def test_token_keyword_normalized_uppercase():
    tok = Token(T.Keyword, 'select')
    assert tok.is_keyword is True
    assert tok.normalized == 'SELECT'


def test_token_whitespace_and_newline():
    ws = Token(T.Whitespace, ' ')
    assert ws.is_whitespace is True
    assert ws.is_newline is False

    nl = Token(T.Newline, '\n')
    assert nl.is_newline is True


def test_token_value_stringified():
    tok = Token(T.Number.Integer, 123)
    assert tok.value == '123'


def test_token_str_and_repr():
    tok = Token(T.Name, 'abc')
    assert str(tok) == 'abc'
    assert repr(tok).startswith('<Name ')
    assert "'abc'" in repr(tok)


def test_token_repr_long_value_truncated():
    tok = Token(T.String.Single, 'abcdefghijk')
    assert "'abcdef..." in repr(tok)


def test_token_repr_quote_wrap():
    tok = Token(T.String.Single, "'abc'")
    assert '"\'abc\'"' in repr(tok)


def test_token_flatten_yields_self():
    tok = Token(T.Name, 'x')
    assert list(tok.flatten()) == [tok]


def test_token_match_ttype_none_values():
    tok = Token(T.Name, 'abc')
    # values None -> just type match
    assert tok.match(T.Name, None) is True
    assert tok.match(T.Other, None) is False


def test_token_match_values_string():
    tok = Token(T.Name, 'abc')
    assert tok.match(T.Name, 'abc') is True
    assert tok.match(T.Name, 'xyz') is False


def test_token_match_keyword_case_insensitive():
    tok = Token(T.Keyword, 'select')
    assert tok.match(T.Keyword, 'SELECT') is True
    assert tok.match(T.Keyword, 'select') is True
    assert tok.match(T.Keyword, 'Insert') is False


def test_token_match_regex():
    tok = Token(T.Name, 'abcdef')
    assert tok.match(T.Name, 'bc', regex=True) is True
    assert tok.match(T.Name, '^abc', regex=True) is True
    assert tok.match(T.Name, 'xyz', regex=True) is False


def test_token_match_regex_keyword_ignore_case():
    tok = Token(T.Keyword, 'SELECT')
    assert tok.match(T.Keyword, 'sel', regex=True) is True


def test_token_within_and_child_of_and_ancestor():
    parent = TokenList([Token(T.Name, 'a')])
    child = parent.tokens[0]
    assert child.is_child_of(parent) is True
    assert child.has_ancestor(parent) is True

    grandparent = TokenList([parent])
    assert child.is_child_of(parent) is True
    assert child.is_child_of(grandparent) is False
    assert child.has_ancestor(grandparent) is True


def test_token_within():
    inner = TokenList([Token(T.Name, 'col')])
    outer = TokenList([inner])
    tok = inner.tokens[0]
    assert tok.within(TokenList) is True
    assert tok.within(Identifier) is False


# ---------------------------------------------------------------------------
# TokenList
# ---------------------------------------------------------------------------

def test_tokenlist_init_and_parents():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Name, 'b')
    tl = TokenList([t1, t2])
    assert tl.is_group is True
    assert tl.tokens == [t1, t2]
    assert t1.parent is tl
    assert t2.parent is tl
    assert tl.value == 'ab'


def test_tokenlist_empty_default():
    tl = TokenList()
    assert tl.tokens == []
    assert tl.value == ''
    assert tl.is_group is True


def test_tokenlist_str_flatten():
    inner = TokenList([Token(T.Name, 'a'), Token(T.Name, 'b')])
    outer = TokenList([inner, Token(T.Name, 'c')])
    assert str(outer) == 'abc'


def test_tokenlist_iter_and_getitem():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Name, 'b')
    tl = TokenList([t1, t2])
    assert list(tl) == [t1, t2]
    assert tl[0] is t1
    assert tl[-1] is t2


def test_tokenlist_flatten():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Name, 'b')
    t3 = Token(T.Name, 'c')
    inner = TokenList([t1, t2])
    outer = TokenList([inner, t3])
    assert list(outer.flatten()) == [t1, t2, t3]


def test_tokenlist_get_sublists():
    t1 = Token(T.Name, 'a')
    inner = TokenList([t1])
    outer = TokenList([inner, Token(T.Name, 'b')])
    assert list(outer.get_sublists()) == [inner]


def test_tokenlist_get_token_at_offset():
    t1 = Token(T.Name, 'abc')
    t2 = Token(T.Name, 'def')
    tl = TokenList([t1, t2])
    assert tl.get_token_at_offset(0) is t1
    assert tl.get_token_at_offset(2) is t1
    assert tl.get_token_at_offset(3) is t2
    assert tl.get_token_at_offset(5) is t2
    assert tl.get_token_at_offset(6) is None


def test_tokenlist_token_first():
    ws = Token(T.Whitespace, ' ')
    cm = Comment([Token(T.Comment.Single, '--x')])
    name = Token(T.Name, 'x')
    tl = TokenList([ws, cm, name])
    assert tl.token_first() is cm
    assert tl.token_first(skip_ws=True, skip_cm=True) is name


def test_tokenlist_token_first_empty():
    tl = TokenList([Token(T.Whitespace, ' ')])
    assert tl.token_first() is None


def test_tokenlist_token_next_by_match():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Punctuation, '.')
    tl = TokenList([t1, t2])
    idx, tok = tl.token_next_by(m=(T.Punctuation, '.'))
    assert idx == 1
    assert tok is t2


def test_tokenlist_token_next_by_type_none():
    tl = TokenList([Token(T.Name, 'a')])
    idx, tok = tl.token_next_by(t=T.Punctuation)
    assert idx is None
    assert tok is None


def test_tokenlist_token_next():
    ws = Token(T.Whitespace, ' ')
    name = Token(T.Name, 'a')
    tl = TokenList([Token(T.Name, 'start'), ws, name])
    idx, tok = tl.token_next(0)
    assert idx == 2
    assert tok is name

    idx, tok = tl.token_next(2)
    assert idx is None
    assert tok is None


def test_tokenlist_token_next_idx_none():
    tl = TokenList([Token(T.Name, 'a')])
    assert tl.token_next(None) == (None, None)


def test_tokenlist_token_prev():
    start = Token(T.Name, 'start')
    ws = Token(T.Whitespace, ' ')
    name = Token(T.Name, 'a')
    tl = TokenList([start, ws, name])
    idx, tok = tl.token_prev(2)
    assert idx == 0
    assert tok is start


def test_tokenlist_token_index():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Name, 'b')
    tl = TokenList([t1, t2])
    assert tl.token_index(t2) == 1
    assert tl.token_index(t2, start=1) == 1


def test_tokenlist_token_matching():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Name, 'b')
    tl = TokenList([t1, t2])
    assert tl.token_matching(lambda tk: tk.value == 'b', 0) is t2


def test_tokenlist_token_not_matching():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Name, 'b')
    tl = TokenList([t1, t2])
    idx, tok = tl.token_not_matching(lambda tk: tk.value == 'a', 0)
    assert idx == 1
    assert tok is t2


def test_tokenlist_group_tokens():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Whitespace, ' ')
    t3 = Token(T.Name, 'b')
    tl = TokenList([t1, t2, t3])
    grp = tl.group_tokens(Parenthesis, 0, 2)
    assert isinstance(grp, Parenthesis)
    assert tl.tokens == [grp]
    assert grp.parent is tl
    assert t1.parent is grp
    assert t3.parent is grp


def test_tokenlist_group_tokens_no_include_end():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Name, 'b')
    t3 = Token(T.Name, 'c')
    tl = TokenList([t1, t2, t3])
    grp = tl.group_tokens(Parenthesis, 0, 1, include_end=False)
    assert grp.tokens == [t1]
    assert tl.tokens == [grp, t2, t3]


def test_tokenlist_group_tokens_extend():
    inner_tok = Token(T.Name, 'a')
    grp = Parenthesis([inner_tok])
    extra = Token(T.Name, 'b')
    tl = TokenList([grp, extra])
    result = tl.group_tokens(Parenthesis, 0, 1, extend=True)
    assert result is grp
    assert grp.tokens == [inner_tok, extra]
    assert extra.parent is grp


def test_tokenlist_insert_before_int():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Name, 'b')
    tl = TokenList([t1, t2])
    new = Token(T.Name, 'x')
    tl.insert_before(1, new)
    assert tl.tokens == [t1, new, t2]
    assert new.parent is tl


def test_tokenlist_insert_before_token():
    t1 = Token(T.Name, 'a')
    t2 = Token(T.Name, 'b')
    tl = TokenList([t1, t2])
    new = Token(T.Name, 'x')
    tl.insert_before(t2, new)
    assert tl.tokens == [t1, new, t2]


def test_tokenlist_insert_after_int():
    t1 = Token(T.Name, 'a')
    ws = Token(T.Whitespace, ' ')
    t2 = Token(T.Name, 'b')
    tl = TokenList([t1, ws, t2])
    new = Token(T.Name, 'x')
    tl.insert_after(0, new)
    # skip_ws=True -> after t2
    assert tl.tokens == [t1, ws, t2, new]
    assert new.parent is tl


def test_tokenlist_insert_after_no_next():
    t1 = Token(T.Name, 'a')
    tl = TokenList([t1])
    new = Token(T.Name, 'x')
    tl.insert_after(0, new)
    assert tl.tokens == [t1, new]


def test_tokenlist_default_alias_and_names():
    tl = TokenList([Token(T.Name, 'a')])
    assert tl.get_alias() is None
    assert tl.has_alias() is False
    assert tl.get_real_name() is None
    assert tl.get_name() is None


def test_tokenlist_get_parent_name():
    t1 = Token(T.Name, 'schema')
    dot = Token(T.Punctuation, '.')
    t2 = Token(T.Name, 'table')
    tl = TokenList([t1, dot, t2])
    assert tl.get_parent_name() == 'schema'


def test_tokenlist_get_parent_name_none():
    tl = TokenList([Token(T.Name, 'table')])
    assert tl.get_parent_name() is None


def test_tokenlist_get_parent_name_quoted():
    t1 = Token(T.Name, '"schema"')
    dot = Token(T.Punctuation, '.')
    t2 = Token(T.Name, 'table')
    tl = TokenList([t1, dot, t2])
    assert tl.get_parent_name() == 'schema'


# ---------------------------------------------------------------------------
# NameAliasMixin / Identifier / Function alias handling
# ---------------------------------------------------------------------------

def test_identifier_get_real_name():
    ident = Identifier(make_tokens())
    # a.b AS c -> real name after last dot is b
    assert ident.get_real_name() == 'b'


def test_identifier_get_alias_as():
    ident = Identifier(make_tokens())
    assert ident.get_alias() == 'c'
    assert ident.has_alias() is True


def test_identifier_get_alias_without_as():
    tokens = [
        Token(T.Name, 'foo'),
        Token(T.Whitespace, ' '),
        Token(T.Name, 'bar'),
    ]
    ident = Identifier(tokens)
    assert ident.get_alias() == 'bar'


def test_identifier_get_alias_none():
    ident = Identifier([Token(T.Name, 'foo')])
    assert ident.get_alias() is None


def test_identifier_get_name_prefers_alias():
    ident = Identifier(make_tokens())
    assert ident.get_name() == 'c'


def test_identifier_is_wildcard():
    ident = Identifier([Token(T.Name, 'a'), Token(T.Wildcard, '*')])
    assert ident.is_wildcard() is True

    ident2 = Identifier([Token(T.Name, 'a')])
    assert ident2.is_wildcard() is False


def test_identifier_get_typecast():
    tokens = [
        Token(T.Name, 'a'),
        Token(T.Punctuation, '::'),
        Token(T.Name.Builtin, 'int'),
    ]
    ident = Identifier(tokens)
    assert ident.get_typecast() == 'int'


def test_identifier_get_typecast_none():
    ident = Identifier([Token(T.Name, 'a')])
    assert ident.get_typecast() is None


def test_identifier_get_ordering():
    tokens = [Token(T.Name, 'a'), Token(T.Keyword.Order, 'desc')]
    ident = Identifier(tokens)
    assert ident.get_ordering() == 'DESC'


def test_identifier_get_ordering_none():
    ident = Identifier([Token(T.Name, 'a')])
    assert ident.get_ordering() is None


def test_identifier_get_array_indices():
    sb = SquareBrackets([
        Token(T.Punctuation, '['),
        Token(T.Number.Integer, '1'),
        Token(T.Punctuation, ']'),
    ])
    ident = Identifier([Token(T.Name, 'a'), sb])
    result = list(ident.get_array_indices())
    assert len(result) == 1
    assert result[0] == sb.tokens[1:-1]


def test_identifier_get_array_indices_empty():
    ident = Identifier([Token(T.Name, 'a')])
    assert list(ident.get_array_indices()) == []


def test_get_real_name_quoted():
    tokens = [
        Token(T.Name, '"tbl"'),
        Token(T.Whitespace, ' '),
        Token(T.Keyword, 'AS'),
        Token(T.Whitespace, ' '),
        Token(T.Name, 'alias'),
    ]
    ident = Identifier(tokens)
    assert ident.get_real_name() == 'tbl'


# ---------------------------------------------------------------------------
# IdentifierList
# ---------------------------------------------------------------------------

def test_identifierlist_get_identifiers():
    id1 = Identifier([Token(T.Name, 'a')])
    id2 = Identifier([Token(T.Name, 'b')])
    tl = IdentifierList([
        id1,
        Token(T.Punctuation, ','),
        Token(T.Whitespace, ' '),
        id2,
    ])
    assert list(tl.get_identifiers()) == [id1, id2]


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

def test_statement_get_type_dml():
    stmt = Statement([Token(T.Keyword.DML, 'select'), Token(T.Name, 'x')])
    assert stmt.get_type() == 'SELECT'


def test_statement_get_type_ddl():
    stmt = Statement([Token(T.Keyword.DDL, 'create'), Token(T.Name, 'x')])
    assert stmt.get_type() == 'CREATE'


def test_statement_get_type_unknown():
    stmt = Statement([Token(T.Name, 'foo')])
    assert stmt.get_type() == 'UNKNOWN'


def test_statement_get_type_empty():
    stmt = Statement([])
    assert stmt.get_type() == 'UNKNOWN'


def test_statement_get_type_whitespace_only():
    stmt = Statement([Token(T.Whitespace, ' ')])
    assert stmt.get_type() == 'UNKNOWN'


def test_statement_get_type_skips_comment():
    cm = Comment([Token(T.Comment.Single, '--x')])
    stmt = Statement([cm, Token(T.Whitespace, ' '),
                      Token(T.Keyword.DML, 'insert'),
                      Token(T.Name, 'x')])
    assert stmt.get_type() == 'INSERT'


def test_statement_get_type_cte():
    cte_kw = Token(T.Keyword.CTE, 'WITH')
    ident = Identifier([Token(T.Name, 'cte')])
    dml = Token(T.Keyword.DML, 'select')
    stmt = Statement([
        cte_kw,
        Token(T.Whitespace, ' '),
        ident,
        Token(T.Whitespace, ' '),
        dml,
        Token(T.Name, 'x'),
    ])
    assert stmt.get_type() == 'SELECT'


def test_statement_get_type_cte_unknown():
    cte_kw = Token(T.Keyword.CTE, 'WITH')
    stmt = Statement([cte_kw, Token(T.Whitespace, ' '),
                      Token(T.Name, 'x')])
    assert stmt.get_type() == 'UNKNOWN'


# ---------------------------------------------------------------------------
# Parenthesis / SquareBrackets
# ---------------------------------------------------------------------------

def test_parenthesis_groupable_tokens():
    t_open = Token(T.Punctuation, '(')
    t_mid = Token(T.Name, 'a')
    t_close = Token(T.Punctuation, ')')
    p = Parenthesis([t_open, t_mid, t_close])
    assert p._groupable_tokens == [t_mid]


def test_squarebrackets_groupable_tokens():
    t_open = Token(T.Punctuation, '[')
    t_mid = Token(T.Name, 'a')
    t_close = Token(T.Punctuation, ']')
    sb = SquareBrackets([t_open, t_mid, t_close])
    assert sb._groupable_tokens == [t_mid]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def test_comparison_left_right():
    left = Token(T.Name, 'a')
    right = Token(T.Name, 'b')
    cmp = Comparison([left, Token(T.Operator, '='), right])
    assert cmp.left is left
    assert cmp.right is right


# ---------------------------------------------------------------------------
# Comment
# ---------------------------------------------------------------------------

def test_comment_is_multiline():
    cm = Comment([Token(T.Comment.Multiline, '/* x */')])
    assert cm.is_multiline() is True

    cm2 = Comment([Token(T.Comment.Single, '-- x')])
    assert cm2.is_multiline() is False


def test_comment_is_multiline_empty():
    cm = Comment([])
    assert not cm.is_multiline()


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

def test_case_get_cases():
    tokens = [
        Token(T.Keyword, 'CASE'),
        Token(T.Whitespace, ' '),
        Token(T.Keyword, 'WHEN'),
        Token(T.Name, 'a'),
        Token(T.Keyword, 'THEN'),
        Token(T.Name, 'b'),
        Token(T.Keyword, 'ELSE'),
        Token(T.Name, 'c'),
        Token(T.Keyword, 'END'),
    ]
    case = Case(tokens)
    cases = case.get_cases()
    assert len(cases) == 2
    # first WHEN
    cond, val = cases[0]
    assert any(t.value == 'a' for t in cond if not t.is_whitespace)
    assert any(t.value == 'b' for t in val if not t.is_whitespace)
    # ELSE
    cond2, val2 = cases[1]
    assert cond2 is None
    assert any(t.value == 'c' for t in val2 if not t.is_whitespace)


def test_case_get_cases_skip_ws():
    tokens = [
        Token(T.Keyword, 'CASE'),
        Token(T.Whitespace, ' '),
        Token(T.Keyword, 'WHEN'),
        Token(T.Name, 'a'),
        Token(T.Keyword, 'THEN'),
        Token(T.Name, 'b'),
        Token(T.Keyword, 'END'),
    ]
    case = Case(tokens)
    cases = case.get_cases(skip_ws=True)
    assert len(cases) == 1
    cond, val = cases[0]
    assert all(not t.is_whitespace for t in cond)
    assert all(not t.is_whitespace for t in val)


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

def test_function_get_parameters():
    lit = Token(T.Literal, '1')
    paren = Parenthesis([
        Token(T.Punctuation, '('),
        lit,
        Token(T.Punctuation, ')'),
    ])
    fn = Function([Token(T.Name, 'fn'), paren])
    assert fn.get_parameters() == [lit]


def test_function_get_parameters_identifierlist():
    id1 = Identifier([Token(T.Name, 'a')])
    id2 = Identifier([Token(T.Name, 'b')])
    idlist = IdentifierList([id1, Token(T.Punctuation, ','),
                             Token(T.Whitespace, ' '), id2])
    paren = Parenthesis([
        Token(T.Punctuation, '('),
        idlist,
        Token(T.Punctuation, ')'),
    ])
    fn = Function([Token(T.Name, 'fn'), paren])
    params = fn.get_parameters()
    assert list(params) == [id1, id2]


def test_function_get_window_none():
    paren = Parenthesis([Token(T.Punctuation, '('),
                         Token(T.Punctuation, ')')])
    fn = Function([Token(T.Name, 'fn'), paren])
    assert fn.get_window() is None


def test_function_get_window():
    window_tok = Token(T.Name, 'w')
    over = Over([Token(T.Keyword, 'OVER'), Token(T.Whitespace, ' '),
                 window_tok])
    paren = Parenthesis([Token(T.Punctuation, '('),
                         Token(T.Punctuation, ')')])
    fn = Function([Token(T.Name, 'fn'), paren, Token(T.Whitespace, ' '),
                   over])
    assert fn.get_window() is window_tok


def test_function_get_alias():
    paren = Parenthesis([Token(T.Punctuation, '('),
                         Token(T.Punctuation, ')')])
    tokens = [
        Token(T.Name, 'fn'),
        paren,
        Token(T.Whitespace, ' '),
        Token(T.Keyword, 'AS'),
        Token(T.Whitespace, ' '),
        Token(T.Name, 'alias'),
    ]
    fn = Function(tokens)
    assert fn.get_alias() == 'alias'


# ---------------------------------------------------------------------------
# Misc classes existence / M_OPEN etc.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    TypedLiteral, Assignment, If, For, Where, Over, Having, Begin,
    Operation, Values, Command,
])
def test_classes_are_tokenlists(cls):
    obj = cls([Token(T.Name, 'x')])
    assert isinstance(obj, TokenList)
    assert obj.is_group is True


def test_typed_literal_constants():
    assert TypedLiteral.M_CLOSE == (T.String.Single, None)
    assert TypedLiteral.M_OPEN == [(T.Name.Builtin, None),
                                   (T.Keyword, "TIMESTAMP")]


def test_if_constants():
    assert If.M_OPEN == (T.Keyword, 'IF')
    assert If.M_CLOSE == (T.Keyword, 'END IF')


def test_for_constants():
    assert For.M_OPEN == (T.Keyword, ('FOR', 'FOREACH'))
    assert For.M_CLOSE == (T.Keyword, 'END LOOP')


def test_where_constants():
    assert Where.M_OPEN == (T.Keyword, 'WHERE')
    assert 'ORDER BY' in Where.M_CLOSE[1]


def test_having_constants():
    assert Having.M_OPEN == (T.Keyword, 'HAVING')
    assert Having.M_CLOSE == (T.Keyword, ('ORDER BY', 'LIMIT'))


def test_case_constants():
    assert Case.M_OPEN == (T.Keyword, 'CASE')
    assert Case.M_CLOSE == (T.Keyword, 'END')


def test_begin_constants():
    assert Begin.M_OPEN == (T.Keyword, 'BEGIN')
    assert Begin.M_CLOSE == (T.Keyword, 'END')


def test_over_constants():
    assert Over.M_OPEN == (T.Keyword, 'OVER')


def test_parenthesis_constants():
    assert Parenthesis.M_OPEN == (T.Punctuation, '(')
    assert Parenthesis.M_CLOSE == (T.Punctuation, ')')


def test_squarebrackets_constants():
    assert SquareBrackets.M_OPEN == (T.Punctuation, '[')
    assert SquareBrackets.M_CLOSE == (T.Punctuation, ']')
