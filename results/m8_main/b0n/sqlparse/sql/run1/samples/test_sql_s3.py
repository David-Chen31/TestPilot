import pytest

from sqlparse import tokens as T
from sqlparse.sql import (
    NameAliasMixin,
    Token,
    TokenList,
    Statement,
    Identifier,
    IdentifierList,
    TypedLiteral,
    Parenthesis,
    SquareBrackets,
    Assignment,
    If,
    For,
    Comparison,
    Comment,
    Where,
    Over,
    Having,
    Case,
    Function,
    Begin,
    Operation,
    Values,
    Command,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tok(ttype, value):
    return Token(ttype, value)


def ident(*tokens):
    return Identifier(list(tokens))


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

class TestToken:
    def test_init_basic(self):
        t = Token(T.Name, 'foo')
        assert t.value == 'foo'
        assert t.ttype == T.Name
        assert t.parent is None
        assert t.is_group is False
        assert t.is_keyword is False
        assert t.is_whitespace is False
        assert t.is_newline is False
        assert t.normalized == 'foo'

    def test_init_value_coerced_to_str(self):
        t = Token(T.Name, 123)
        assert t.value == '123'
        assert isinstance(t.value, str)

    def test_init_keyword_normalized_upper(self):
        t = Token(T.Keyword, 'select')
        assert t.is_keyword is True
        assert t.normalized == 'SELECT'

    def test_init_whitespace(self):
        t = Token(T.Whitespace, '  ')
        assert t.is_whitespace is True
        assert t.normalized == '  '

    def test_init_newline(self):
        t = Token(T.Newline, '\n')
        assert t.is_newline is True

    def test_str(self):
        assert str(Token(T.Name, 'abc')) == 'abc'

    def test_repr_name(self):
        t = Token(T.Name, 'abc')
        assert t._get_repr_name() == 'Name'

    def test_repr_value_short(self):
        t = Token(T.Name, 'abc')
        assert t._get_repr_value() == 'abc'

    def test_repr_value_long(self):
        t = Token(T.Name, 'abcdefghijk')
        assert t._get_repr_value() == 'abcdef...'

    def test_repr_value_whitespace_collapsed(self):
        t = Token(T.Name, 'a   b')
        assert t._get_repr_value() == 'a b'

    def test_repr_contains_class_and_value(self):
        t = Token(T.Name, 'abc')
        r = repr(t)
        assert r.startswith('<Name ')
        assert "'abc'" in r

    def test_flatten_yields_self(self):
        t = Token(T.Name, 'x')
        assert list(t.flatten()) == [t]

    def test_match_ttype_only(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, None) is True
        assert t.match(T.Keyword, None) is False

    def test_match_values_none_returns_type_match(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, None) is True

    def test_match_value_in_list(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, ('foo', 'bar')) is True
        assert t.match(T.Name, ('baz',)) is False

    def test_match_value_single_string(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, 'foo') is True
        assert t.match(T.Name, 'bar') is False

    def test_match_keyword_case_insensitive(self):
        t = Token(T.Keyword, 'SeLeCt')
        assert t.match(T.Keyword, 'select') is True
        assert t.match(T.Keyword, ('SELECT',)) is True

    def test_match_non_keyword_case_sensitive(self):
        t = Token(T.Name, 'Foo')
        assert t.match(T.Name, 'foo') is False
        assert t.match(T.Name, 'Foo') is True

    def test_match_regex(self):
        t = Token(T.Name, 'foobar')
        assert t.match(T.Name, 'foo', regex=True) is True
        assert t.match(T.Name, 'baz', regex=True) is False

    def test_match_regex_keyword_ignorecase(self):
        t = Token(T.Keyword, 'SELECT')
        assert t.match(T.Keyword, 'sel', regex=True) is True

    def test_match_ttype_identity(self):
        # match uses `is` on ttype
        t = Token(T.Name, 'foo')
        class FakeType(str):
            pass
        assert t.match(FakeType('Name'), 'foo') is False

    def test_is_child_of(self):
        parent = TokenList()
        child = Token(T.Name, 'x')
        child.parent = parent
        assert child.is_child_of(parent) is True
        assert child.is_child_of(TokenList()) is False

    def test_has_ancestor(self):
        grand = TokenList()
        parent = TokenList()
        child = Token(T.Name, 'x')
        parent.parent = grand
        child.parent = parent
        assert child.has_ancestor(grand) is True
        assert child.has_ancestor(parent) is True
        assert child.has_ancestor(TokenList()) is False

    def test_within(self):
        grand = TokenList()
        parent = TokenList()
        child = Token(T.Name, 'x')
        parent.parent = grand
        child.parent = parent
        assert child.within(TokenList) is True
        assert child.within(Statement) is False


# ---------------------------------------------------------------------------
# TokenList
# ---------------------------------------------------------------------------

class TestTokenList:
    def test_init_empty(self):
        tl = TokenList()
        assert tl.tokens == []
        assert tl.is_group is True
        assert tl.value == ''
        assert tl.ttype is None

    def test_init_sets_parents(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert a.parent is tl
        assert b.parent is tl

    def test_init_value_concatenates(self):
        tl = TokenList([Token(T.Name, 'a'), Token(T.Name, 'b')])
        assert tl.value == 'ab'

    def test_str_flattens(self):
        inner = TokenList([Token(T.Name, 'x')])
        outer = TokenList([Token(T.Name, 'a'), inner])
        assert str(outer) == 'ax'

    def test_iter(self):
        a, b = Token(T.Name, 'a'), Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert list(tl) == [a, b]

    def test_getitem(self):
        a, b = Token(T.Name, 'a'), Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl[0] is a
        assert tl[-1] is b

    def test_get_repr_name(self):
        assert TokenList()._get_repr_name() == 'TokenList'
        assert Parenthesis()._get_repr_name() == 'Parenthesis'

    def test_flatten(self):
        a = Token(T.Name, 'a')
        inner_b = Token(T.Name, 'b')
        inner = TokenList([inner_b])
        outer = TokenList([a, inner])
        assert list(outer.flatten()) == [a, inner_b]

    def test_get_sublists(self):
        a = Token(T.Name, 'a')
        inner = TokenList([Token(T.Name, 'b')])
        outer = TokenList([a, inner])
        assert list(outer.get_sublists()) == [inner]

    def test_get_token_at_offset(self):
        a = Token(T.Name, 'abc')
        b = Token(T.Name, 'de')
        tl = TokenList([a, b])
        assert tl.get_token_at_offset(0) is a
        assert tl.get_token_at_offset(2) is a
        assert tl.get_token_at_offset(3) is b
        assert tl.get_token_at_offset(4) is b
        assert tl.get_token_at_offset(5) is None

    def test_get_token_at_offset_nested(self):
        inner_tok = Token(T.Name, 'x')
        inner = TokenList([inner_tok])
        outer = TokenList([Token(T.Name, 'a'), inner])
        assert outer.get_token_at_offset(0).value == 'a'
        assert outer.get_token_at_offset(1) is inner_tok

    def test_token_first_skip_ws_default(self):
        ws = Token(T.Whitespace, ' ')
        a = Token(T.Name, 'a')
        tl = TokenList([ws, a])
        assert tl.token_first() is a

    def test_token_first_no_skip_ws(self):
        ws = Token(T.Whitespace, ' ')
        a = Token(T.Name, 'a')
        tl = TokenList([ws, a])
        assert tl.token_first(skip_ws=False) is ws

    def test_token_first_skip_cm(self):
        cm = Comment([Token(T.Comment.Single, '--x')])
        ws = Token(T.Whitespace, ' ')
        a = Token(T.Name, 'a')
        tl = TokenList([cm, ws, a])
        assert tl.token_first(skip_cm=True) is a
        assert tl.token_first(skip_cm=False) is cm

    def test_token_first_empty(self):
        assert TokenList().token_first() is None

    def test_token_next_by_type(self):
        a = Token(T.Name, 'a')
        b = Token(T.Wildcard, '*')
        tl = TokenList([a, b])
        idx, token = tl.token_next_by(t=T.Wildcard)
        assert idx == 1
        assert token is b

    def test_token_next_by_match(self):
        a = Token(T.Punctuation, '.')
        b = Token(T.Punctuation, ',')
        tl = TokenList([a, b])
        idx, token = tl.token_next_by(m=(T.Punctuation, '.'))
        assert idx == 0
        assert token is a

    def test_token_next_by_instance(self):
        inner = Parenthesis()
        tl = TokenList([Token(T.Name, 'a'), inner])
        idx, token = tl.token_next_by(i=Parenthesis)
        assert idx == 1
        assert token is inner

    def test_token_next_by_not_found(self):
        tl = TokenList([Token(T.Name, 'a')])
        idx, token = tl.token_next_by(t=T.Wildcard)
        assert idx is None
        assert token is None

    def test_token_not_matching(self):
        ws = Token(T.Whitespace, ' ')
        a = Token(T.Name, 'a')
        tl = TokenList([ws, a])
        idx, token = tl.token_not_matching(lambda t: t.is_whitespace, 0)
        assert idx == 1
        assert token is a

    def test_token_matching(self):
        a = Token(T.Name, 'a')
        b = Token(T.Wildcard, '*')
        tl = TokenList([a, b])
        assert tl.token_matching(lambda t: t.ttype == T.Wildcard, 0) is b

    def test_token_next(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        idx, token = tl.token_next(0)
        assert idx == 1
        assert token is b

    def test_token_next_skip_ws(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, token = tl.token_next(0)
        assert idx == 2
        assert token is b

    def test_token_next_no_skip_ws(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, token = tl.token_next(0, skip_ws=False)
        assert idx == 1
        assert token is ws

    def test_token_next_idx_none(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.token_next(None) == (None, None)

    def test_token_next_end_of_list(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl.token_next(0) == (None, None)

    def test_token_prev(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        idx, token = tl.token_prev(1)
        assert idx == 0
        assert token is a

    def test_token_prev_skip_ws(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, token = tl.token_prev(2)
        assert idx == 0
        assert token is a

    def test_token_prev_start(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl.token_prev(0) == (None, None)

    def test_token_index(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_index(b) == 1

    def test_token_index_start(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        assert tl.token_index(b, a) == 1
        assert tl.token_index(c, b) == 2

    def test_token_index_error(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        with pytest.raises(ValueError):
            tl.token_index(Token(T.Name, 'c'))

    def test_group_tokens(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        grp = tl.group_tokens(TokenList, 0, 1)
        assert isinstance(grp, TokenList)
        assert grp.value == 'ab'
        assert tl.tokens == [grp, c]
        assert grp.parent is tl
        assert a.parent is grp
        assert b.parent is grp

    def test_group_tokens_no_include_end(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        grp = tl.group_tokens(TokenList, 0, 1, include_end=False)
        assert grp.value == 'a'
        assert tl.tokens == [grp, b, c]

    def test_group_tokens_extend(self):
        inner_a = Token(T.Name, 'a')
        inner_b = Token(T.Name, 'b')
        existing = TokenList([inner_a])
        c = Token(T.Name, 'c')
        tl = TokenList([existing, inner_b, c])
        grp = tl.group_tokens(TokenList, 0, 1, extend=True)
        assert grp is existing
        assert grp.value == 'ab'
        assert [t.value for t in grp.tokens] == ['a', 'b']
        assert tl.tokens == [grp, c]

    def test_insert_before_int(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        new = Token(T.Name, 'x')
        tl.insert_before(0, new)
        assert tl.tokens == [new, a]
        assert new.parent is tl

    def test_insert_before_token(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        new = Token(T.Name, 'x')
        tl.insert_before(b, new)
        assert tl.tokens == [a, new, b]
        assert new.parent is tl

    def test_insert_after_int(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        new = Token(T.Name, 'x')
        tl.insert_after(0, new)
        assert tl.tokens == [a, new, b]
        assert new.parent is tl

    def test_insert_after_token(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        new = Token(T.Name, 'x')
        tl.insert_after(a, new)
        assert tl.tokens == [a, new, b]

    def test_insert_after_end(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        new = Token(T.Name, 'x')
        tl.insert_after(a, new)
        assert tl.tokens == [a, new]
        assert new.parent is tl

    def test_insert_after_no_skip_ws(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        new = Token(T.Name, 'x')
        tl.insert_after(a, new)
        # default skip_ws True -> insert before b
        assert tl.tokens == [a, ws, new, b]

    def test_has_alias_default_false(self):
        assert TokenList().has_alias() is False

    def test_get_alias_default_none(self):
        assert TokenList().get_alias() is None

    def test_get_real_name_default_none(self):
        assert TokenList().get_real_name() is None

    def test_get_name_uses_real_name(self):
        tl = TokenList()
        assert tl.get_name() is None

    def test_get_parent_name(self):
        a = Token(T.Name, 'tbl')
        dot = Token(T.Punctuation, '.')
        b = Token(T.Name, 'col')
        tl = TokenList([a, dot, b])
        assert tl.get_parent_name() == 'tbl'

    def test_get_parent_name_none(self):
        tl = TokenList([Token(T.Name, 'col')])
        assert tl.get_parent_name() is None

    def test_get_parent_name_quoted(self):
        a = Token(T.Name, '"tbl"')
        dot = Token(T.Punctuation, '.')
        b = Token(T.Name, 'col')
        tl = TokenList([a, dot, b])
        assert tl.get_parent_name() == 'tbl'

    def test_get_first_name(self):
        a = Token(T.Name, 'first')
        b = Token(T.Name, 'second')
        tl = TokenList([a, b])
        assert tl._get_first_name() == 'first'

    def test_get_first_name_reverse(self):
        a = Token(T.Name, 'first')
        b = Token(T.Name, 'second')
        tl = TokenList([a, b])
        assert tl._get_first_name(reverse=True) == 'second'

    def test_get_first_name_no_match(self):
        tl = TokenList([Token(T.Whitespace, ' ')])
        assert tl._get_first_name() is None

    def test_get_first_name_keywords(self):
        a = Token(T.Keyword, 'DISTINCT')
        tl = TokenList([a])
        assert tl._get_first_name(keywords=True) == 'DISTINCT'
        assert tl._get_first_name() is None

    def test_get_first_name_removes_quotes(self):
        a = Token(T.Name, '"foo"')
        tl = TokenList([a])
        assert tl._get_first_name() == 'foo'

    def test_get_first_name_index(self):
        a = Token(T.Name, 'first')
        b = Token(T.Name, 'second')
        tl = TokenList([a, b])
        assert tl._get_first_name(1) == 'second'

    def test_get_first_name_string_symbol(self):
        a = Token(T.String.Symbol, 'sym')
        tl = TokenList([a])
        assert tl._get_first_name() == 'sym'

    def test_get_first_name_wildcard(self):
        a = Token(T.Wildcard, '*')
        tl = TokenList([a])
        assert tl._get_first_name() == '*'

    def test_get_first_name_real_name_identifier(self):
        inner = ident(Token(T.Name, 'tbl'), Token(T.Punctuation, '.'),
                      Token(T.Name, 'col'))
        inner.get_real_name = lambda: 'col'
        inner.get_name = lambda: 'col_alias'
        tl = TokenList([inner])
        assert tl._get_first_name(real_name=True) == 'col'
        assert tl._get_first_name() == 'col_alias'


# ---------------------------------------------------------------------------
# NameAliasMixin / Identifier
# ---------------------------------------------------------------------------

class TestIdentifier:
    def test_get_real_name_simple(self):
        i = ident(Token(T.Name, 'foo'))
        assert i.get_real_name() == 'foo'

    def test_get_real_name_dotted(self):
        i = ident(Token(T.Name, 'a'), Token(T.Punctuation, '.'),
                  Token(T.Name, 'b'))
        assert i.get_real_name() == 'b'

    def test_get_real_name_multi_dot(self):
        i = ident(Token(T.Name, 'a'), Token(T.Punctuation, '.'),
                  Token(T.Name, 'b'), Token(T.Punctuation, '.'),
                  Token(T.Name, 'c'))
        assert i.get_real_name() == 'c'

    def test_get_real_name_quoted(self):
        i = ident(Token(T.Name, '"foo"'))
        assert i.get_real_name() == 'foo'

    def test_get_alias_with_as(self):
        i = ident(Token(T.Name, 'foo'), Token(T.Whitespace, ' '),
                  Token(T.Keyword, 'AS'), Token(T.Whitespace, ' '),
                  Token(T.Name, 'bar'))
        assert i.get_alias() == 'bar'

    def test_get_alias_with_as_keyword_case(self):
        i = ident(Token(T.Name, 'foo'), Token(T.Whitespace, ' '),
                  Token(T.Keyword, 'as'), Token(T.Whitespace, ' '),
                  Token(T.Identifier, 'bar'))
        assert i.get_alias() == 'bar'

    def test_get_alias_without_as(self):
        i = ident(Token(T.Name, 'foo'), Token(T.Whitespace, ' '),
                  Token(T.Name, 'bar'))
        assert i.get_alias() == 'bar'

    def test_get_alias_none(self):
        i = ident(Token(T.Name, 'foo'))
        assert i.get_alias() is None

    def test_has_alias(self):
        i = ident(Token(T.Name, 'foo'), Token(T.Whitespace, ' '),
                  Token(T.Name, 'bar'))
        assert i.has_alias() is True

    def test_has_alias_false(self):
        i = ident(Token(T.Name, 'foo'))
        assert i.has_alias() is False

    def test_get_name_alias_takes_priority(self):
        i = ident(Token(T.Name, 'foo'), Token(T.Whitespace, ' '),
                  Token(T.Name, 'bar'))
        assert i.get_name() == 'bar'

    def test_get_name_real_name(self):
        i = ident(Token(T.Name, 'foo'))
        assert i.get_name() == 'foo'

    def test_is_wildcard_true(self):
        i = ident(Token(T.Name, 't'), Token(T.Punctuation, '.'),
                  Token(T.Wildcard, '*'))
        assert i.is_wildcard() is True

    def test_is_wildcard_false(self):
        i = ident(Token(T.Name, 'foo'))
        assert i.is_wildcard() is False

    def test_get_typecast(self):
        i = ident(Token(T.Name, 'foo'), Token(T.Punctuation, '::'),
                  Token(T.Name.Builtin, 'int'))
        assert i.get_typecast() == 'int'

    def test_get_typecast_none(self):
        i = ident(Token(T.Name, 'foo'))
        assert i.get_typecast() is None

    def test_get_ordering(self):
        i = ident(Token(T.Name, 'foo'), Token(T.Keyword.Order, 'desc'))
        assert i.get_ordering() == 'DESC'

    def test_get_ordering_none(self):
        i = ident(Token(T.Name, 'foo'))
        assert i.get_ordering() is None

    def test_get_array_indices(self):
        inner = Token(T.Name, '5')
        brackets = SquareBrackets([Token(T.Punctuation, '['),
                                   inner,
                                   Token(T.Punctuation, ']')])
        i = ident(Token(T.Name, 'arr'), brackets)
        indices = list(i.get_array_indices())
        assert len(indices) == 1
        assert indices[0] == [inner]

    def test_get_array_indices_empty(self):
        i = ident(Token(T.Name, 'arr'))
        assert list(i.get_array_indices()) == []


# ---------------------------------------------------------------------------
# IdentifierList
# ---------------------------------------------------------------------------

class TestIdentifierList:
    def test_get_identifiers(self):
        a = ident(Token(T.Name, 'a'))
        comma = Token(T.Punctuation, ',')
        ws = Token(T.Whitespace, ' ')
        b = ident(Token(T.Name, 'b'))
        il = IdentifierList([a, comma, ws, b])
        assert list(il.get_identifiers()) == [a, b]

    def test_get_identifiers_empty(self):
        assert list(IdentifierList().get_identifiers()) == []


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

class TestStatement:
    def test_get_type_empty(self):
        assert Statement().get_type() == 'UNKNOWN'

    def test_get_type_whitespace_only(self):
        s = Statement([Token(T.Whitespace, ' ')])
        assert s.get_type() == 'UNKNOWN'

    def test_get_type_dml(self):
        s = Statement([Token(T.Keyword.DML, 'select')])
        assert s.get_type() == 'SELECT'

    def test_get_type_ddl(self):
        s = Statement([Token(T.Keyword.DDL, 'create')])
        assert s.get_type() == 'CREATE'

    def test_get_type_skips_comments(self):
        cm = Comment([Token(T.Comment.Single, '--x')])
        ws = Token(T.Whitespace, ' ')
        dml = Token(T.Keyword.DML, 'insert')
        s = Statement([cm, ws, dml])
        assert s.get_type() == 'INSERT'

    def test_get_type_unknown(self):
        s = Statement([Token(T.Name, 'foo')])
        assert s.get_type() == 'UNKNOWN'

    def test_get_type_cte(self):
        cte = Token(T.Keyword.CTE, 'WITH')
        ws = Token(T.Whitespace, ' ')
        idn = ident(Token(T.Name, 'x'))
        ws2 = Token(T.Whitespace, ' ')
        dml = Token(T.Keyword.DML, 'select')
        s = Statement([cte, ws, idn, ws2, dml])
        assert s.get_type() == 'SELECT'


# ---------------------------------------------------------------------------
# Parenthesis / SquareBrackets
# ---------------------------------------------------------------------------

class TestParenthesis:
    def test_groupable_tokens(self):
        open_ = Token(T.Punctuation, '(')
        mid = Token(T.Name, 'x')
        close = Token(T.Punctuation, ')')
        p = Parenthesis([open_, mid, close])
        assert p._groupable_tokens == [mid]


class TestSquareBrackets:
    def test_groupable_tokens(self):
        open_ = Token(T.Punctuation, '[')
        mid = Token(T.Name, '1')
        close = Token(T.Punctuation, ']')
        sb = SquareBrackets([open_, mid, close])
        assert sb._groupable_tokens == [mid]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestComparison:
    def test_left_right(self):
        a = Token(T.Name, 'a')
        op = Token(T.Operator, '=')
        b = Token(T.Name, 'b')
        c = Comparison([a, op, b])
        assert c.left is a
        assert c.right is b


# ---------------------------------------------------------------------------
# Comment
# ---------------------------------------------------------------------------

class TestComment:
    def test_is_multiline_true(self):
        c = Comment([Token(T.Comment.Multiline, '/* x */')])
        assert c.is_multiline() is True

    def test_is_multiline_false(self):
        c = Comment([Token(T.Comment.Single, '-- x')])
        assert c.is_multiline() is False

    def test_is_multiline_empty(self):
        c = Comment()
        assert c.is_multiline() == []


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

class TestCase:
    def test_get_cases_when_then(self):
        tokens = [
            Token(T.Keyword, 'CASE'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'WHEN'),
            Token(T.Name, 'cond'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'THEN'),
            Token(T.Name, 'val'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'END'),
        ]
        c = Case(tokens)
        cases = c.get_cases()
        assert len(cases) == 1
        cond, val = cases[0]
        assert cond == [Token(T.Whitespace, ' '), Token(T.Name, 'cond'),
                        Token(T.Whitespace, ' ')]
        assert val == [Token(T.Whitespace, ' '), Token(T.Name, 'val'),
                       Token(T.Whitespace, ' ')]

    def test_get_cases_else(self):
        tokens = [
            Token(T.Keyword, 'CASE'),
            Token(T.Keyword, 'ELSE'),
            Token(T.Name, 'val'),
            Token(T.Keyword, 'END'),
        ]
        c = Case(tokens)
        cases = c.get_cases()
        assert len(cases) == 1
        cond, val = cases[0]
        assert cond is None
        assert val == [Token(T.Name, 'val')]

    def test_get_cases_skip_ws(self):
        tokens = [
            Token(T.Keyword, 'CASE'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'WHEN'),
            Token(T.Name, 'cond'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'THEN'),
            Token(T.Name, 'val'),
            Token(T.Keyword, 'END'),
        ]
        c = Case(tokens)
        cases = c.get_cases(skip_ws=True)
        assert len(cases) == 1
        cond, val = cases[0]
        assert cond == [Token(T.Name, 'cond')]
        assert val == [Token(T.Name, 'val')]


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

class TestFunction:
    def _make_function(self, params):
        fname = Token(T.Name, 'func')
        open_ = Token(T.Punctuation, '(')
        close = Token(T.Punctuation, ')')
        par = Parenthesis([open_] + params + [close])
        return Function([fname, par])

    def test_get_parameters_simple(self):
        lit = Token(T.Literal.Number.Integer, 1)
        f = self._make_function([lit])
        assert f.get_parameters() == [lit]

    def test_get_parameters_identifier(self):
        idn = ident(Token(T.Name, 'x'))
        f = self._make_function([idn])
        assert f.get_parameters() == [idn]

    def test_get_parameters_identifierlist(self):
        a = ident(Token(T.Name, 'a'))
        comma = Token(T.Punctuation, ',')
        b = ident(Token(T.Name, 'b'))
        il = IdentifierList([a, comma, b])
        f = self._make_function([il])
        params = f.get_parameters()
        assert params == [a, b]

    def test_get_window_none(self):
        idn = ident(Token(T.Name, 'x'))
        f = self._make_function([idn])
        assert f.get_window() is None

    def test_get_window(self):
        fname = Token(T.Name, 'func')
        par = Parenthesis([Token(T.Punctuation, '('),
                           ident(Token(T.Name, 'x')),
                           Token(T.Punctuation, ')')])
        inner = Token(T.Name, 'win')
        over = Over([Token(T.Keyword, 'OVER'), inner])
        f = Function([fname, par, over])
        assert f.get_window() is inner

    def test_function_alias(self):
        fname = Token(T.Name, 'func')
        par = Parenthesis([Token(T.Punctuation, '('),
                           Token(T.Punctuation, ')')])
        ws = Token(T.Whitespace, ' ')
        alias = Token(T.Name, 'aliased')
        f = Function([fname, par, ws, alias])
        assert f.get_alias() == 'aliased'

    def test_function_real_name(self):
        fname = Token(T.Name, 'func')
        par = Parenthesis([Token(T.Punctuation, '('),
                           Token(T.Punctuation, ')')])
        f = Function([fname, par])
        assert f.get_real_name() == 'func'


# ---------------------------------------------------------------------------
# Typed constructs (M_OPEN / M_CLOSE / M_EXTEND presence)
# ---------------------------------------------------------------------------

class TestTypedConstructs:
    def test_typed_literal_attrs(self):
        assert TypedLiteral.M_CLOSE == (T.String.Single, None)
        assert TypedLiteral.M_EXTEND[0] == T.Keyword

    def test_parenthesis_attrs(self):
        assert Parenthesis.M_OPEN == (T.Punctuation, '(')
        assert Parenthesis.M_CLOSE == (T.Punctuation, ')')

    def test_square_brackets_attrs(self):
        assert SquareBrackets.M_OPEN == (T.Punctuation, '[')
        assert SquareBrackets.M_CLOSE == (T.Punctuation, ']')

    def test_if_attrs(self):
        assert If.M_OPEN == (T.Keyword, 'IF')
        assert If.M_CLOSE == (T.Keyword, 'END IF')

    def test_for_attrs(self):
        assert For.M_OPEN == (T.Keyword, ('FOR', 'FOREACH'))
        assert For.M_CLOSE == (T.Keyword, 'END LOOP')

    def test_where_attrs(self):
        assert Where.M_OPEN == (T.Keyword, 'WHERE')
        assert 'LIMIT' in Where.M_CLOSE[1]

    def test_over_attrs(self):
        assert Over.M_OPEN == (T.Keyword, 'OVER')

    def test_having_attrs(self):
        assert Having.M_OPEN == (T.Keyword, 'HAVING')
        assert Having.M_CLOSE == (T.Keyword, ('ORDER BY', 'LIMIT'))

    def test_case_attrs(self):
        assert Case.M_OPEN == (T.Keyword, 'CASE')
        assert Case.M_CLOSE == (T.Keyword, 'END')

    def test_begin_attrs(self):
        assert Begin.M_OPEN == (T.Keyword, 'BEGIN')
        assert Begin.M_CLOSE == (T.Keyword, 'END')


# ---------------------------------------------------------------------------
# Simple subclass checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cls', [
    Assignment, If, For, Where, Over, Having, Begin, Operation, Values,
    Command, TypedLiteral,
])
def test_simple_tokenlist_subclasses(cls):
    tl = cls()
    assert isinstance(tl, TokenList)
    assert tl.tokens == []
    assert tl.is_group is True


# ---------------------------------------------------------------------------
# NameAliasMixin default behavior through a custom class
# ---------------------------------------------------------------------------

class TestNameAliasMixin:
    def test_get_alias_as_not_found(self):
        # No AS, single token -> no whitespace -> None
        class M(Identifier):
            pass
        i = M([Token(T.Name, 'foo')])
        assert i.get_alias() is None

    def test_alias_from_as_only(self):
        i = Identifier([Token(T.Name, 'foo'),
                        Token(T.Whitespace, ' '),
                        Token(T.Keyword, 'AS'),
                        Token(T.Whitespace, ' '),
                        Token(T.Name, 'bar')])
        assert i.get_alias() == 'bar'

    def test_additional_tokens_required_for_bare_alias(self):
        # len(self.tokens) > 2 and whitespace present
        i = Identifier([Token(T.Name, 'a'),
                        Token(T.Whitespace, ' '),
                        Token(T.Name, 'b')])
        assert i.get_alias() == 'b'

    def test_no_bare_alias_with_two_tokens(self):
        i = Identifier([Token(T.Name, 'a'),
                        Token(T.Whitespace, ' ')])
        assert i.get_alias() is None
