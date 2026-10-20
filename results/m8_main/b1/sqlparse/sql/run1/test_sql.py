"""Unit tests for sqlparse.sql."""

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

def make_tokens(*specs):
    """Build a list of plain Tokens from (ttype, value) tuples."""
    return [Token(ttype, value) for ttype, value in specs]


def make_identifier(*specs):
    return Identifier(make_tokens(*specs))


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

class TestToken:
    def test_init_basic(self):
        tok = Token(T.Name, 'foo')
        assert tok.value == 'foo'
        assert tok.ttype is T.Name
        assert tok.parent is None
        assert tok.is_group is False
        assert tok.is_keyword is False
        assert tok.is_whitespace is False
        assert tok.is_newline is False
        assert tok.normalized == 'foo'

    def test_init_coerces_value_to_str(self):
        tok = Token(T.Name, 123)
        assert tok.value == '123'
        assert tok.normalized == '123'

    def test_init_keyword_normalized_upper(self):
        tok = Token(T.Keyword, 'select')
        assert tok.is_keyword is True
        assert tok.normalized == 'SELECT'

    def test_init_whitespace(self):
        tok = Token(T.Whitespace, ' ')
        assert tok.is_whitespace is True
        assert tok.normalized == ' '

    def test_init_newline(self):
        tok = Token(T.Newline, '\n')
        assert tok.is_newline is True

    def test_str_returns_value(self):
        tok = Token(T.Name, 'bar')
        assert str(tok) == 'bar'

    def test_repr_simple(self):
        tok = Token(T.Name, 'abc')
        r = repr(tok)
        assert r.startswith('<Name abc at 0x') or r.startswith('<Name \'abc\' at 0x')

    def test_repr_truncates_long_value(self):
        tok = Token(T.Name, 'abcdefghij')
        r = repr(tok)
        assert 'abcdef...' in r

    def test_repr_collapses_whitespace(self):
        tok = Token(T.Name, 'a   b')
        r = repr(tok)
        assert 'a b' in r

    def test_repr_single_quoted_value(self):
        tok = Token(T.Name, "'abc'")
        r = repr(tok)
        # single quoted value -> double quotes wrapping
        assert '"\'abc\'"' in r

    def test_repr_name_empty_ttype(self):
        tok = Token(None, 'x')
        r = repr(tok)
        assert '<None' in r or '<' in r

    def test_flatten_yields_self(self):
        tok = Token(T.Name, 'z')
        assert list(tok.flatten()) == [tok]

    def test_match_type_only_true(self):
        tok = Token(T.Name, 'foo')
        assert tok.match(T.Name, None) is True

    def test_match_type_only_false(self):
        tok = Token(T.Name, 'foo')
        assert tok.match(T.Keyword, None) is False

    def test_match_wrong_type_returns_false(self):
        tok = Token(T.Name, 'foo')
        assert tok.match(T.Keyword, 'foo') is False

    def test_match_value_string(self):
        tok = Token(T.Name, 'foo')
        assert tok.match(T.Name, 'foo') is True
        assert tok.match(T.Name, 'bar') is False

    def test_match_value_tuple(self):
        tok = Token(T.Name, 'foo')
        assert tok.match(T.Name, ('foo', 'bar')) is True
        assert tok.match(T.Name, ('x', 'y')) is False

    def test_match_keyword_case_insensitive(self):
        tok = Token(T.Keyword, 'select')
        assert tok.match(T.Keyword, 'SELECT') is True
        assert tok.match(T.Keyword, 'select') is True

    def test_match_name_case_sensitive(self):
        tok = Token(T.Name, 'Foo')
        assert tok.match(T.Name, 'foo') is False
        assert tok.match(T.Name, 'Foo') is True

    def test_match_regex(self):
        tok = Token(T.Keyword, 'SELECT')
        assert tok.match(T.Keyword, 'SEL', regex=True) is True
        assert tok.match(T.Keyword, r'^SEL', regex=True) is True
        assert tok.match(T.Keyword, 'XYZ', regex=True) is False

    def test_match_regex_string_value(self):
        tok = Token(T.Keyword, 'select')
        assert tok.match(T.Keyword, 'le', regex=True) is True

    def test_has_ancestor(self):
        inner = Token(T.Name, 'a')
        grp = TokenList([inner])
        assert inner.has_ancestor(grp) is True
        assert grp.has_ancestor(inner) is False

    def test_within(self):
        inner = Token(T.Name, 'a')
        grp = Identifier([inner])
        assert inner.within(Identifier) is True
        assert inner.within(Function) is False

    def test_is_child_of(self):
        inner = Token(T.Name, 'a')
        grp = TokenList([inner])
        assert inner.is_child_of(grp) is True
        other = TokenList([])
        assert inner.is_child_of(other) is False


# ---------------------------------------------------------------------------
# TokenList
# ---------------------------------------------------------------------------

class TestTokenList:
    def test_init_empty(self):
        tl = TokenList()
        assert tl.tokens == []
        assert tl.is_group is True
        assert tl.value == ''

    def test_init_sets_parents(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert a.parent is tl
        assert b.parent is tl
        assert tl.value == 'ab'

    def test_str_concatenates(self):
        tl = TokenList(make_tokens((T.Name, 'a'), (T.Whitespace, ' '), (T.Name, 'b')))
        assert str(tl) == 'a b'

    def test_str_uses_flatten(self):
        inner = TokenList(make_tokens((T.Name, 'x'), (T.Name, 'y')))
        outer = TokenList([inner])
        assert str(outer) == 'xy'

    def test_iter(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert list(tl) == [a, b]

    def test_getitem(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl[0] is a
        assert tl[-1] is a

    def test_repr_name(self):
        tl = TokenList()
        assert tl._get_repr_name() == 'TokenList'
        assert Identifier([])._get_repr_name() == 'Identifier'

    def test_flatten_nested(self):
        inner = TokenList(make_tokens((T.Name, 'x'), (T.Name, 'y')))
        a = Token(T.Name, 'a')
        outer = TokenList([a, inner])
        flat = list(outer.flatten())
        assert [t.value for t in flat] == ['a', 'x', 'y']

    def test_get_sublists(self):
        inner = TokenList([])
        a = Token(T.Name, 'a')
        outer = TokenList([a, inner])
        assert list(outer.get_sublists()) == [inner]

    def test_groupable_tokens_default(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl._groupable_tokens == [a]

    def test_get_token_at_offset(self):
        tl = TokenList(make_tokens((T.Name, 'abc'), (T.Whitespace, ' '), (T.Name, 'de')))
        assert tl.get_token_at_offset(0).value == 'abc'
        assert tl.get_token_at_offset(2).value == 'abc'
        assert tl.get_token_at_offset(3).value == ' '
        assert tl.get_token_at_offset(4).value == 'de'
        assert tl.get_token_at_offset(100) is None

    def test_get_token_at_offset_nested(self):
        inner = TokenList(make_tokens((T.Name, 'xy')))
        outer = TokenList([inner])
        assert outer.get_token_at_offset(0).value == 'xy'

    def test_token_first_skips_ws(self):
        tl = TokenList(make_tokens((T.Whitespace, ' '), (T.Name, 'a')))
        assert tl.token_first().value == 'a'

    def test_token_first_include_ws(self):
        tl = TokenList(make_tokens((T.Whitespace, ' '), (T.Name, 'a')))
        assert tl.token_first(skip_ws=False).value == ' '

    def test_token_first_skip_cm(self):
        cm = Comment(make_tokens((T.Comment.Single, '-- x')))
        tl = TokenList([cm, Token(T.Name, 'a')])
        assert tl.token_first(skip_cm=True).value == 'a'

    def test_token_first_empty(self):
        tl = TokenList()
        assert tl.token_first() is None

    def test_token_next(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        idx, tok = tl.token_next(0)
        assert idx == 1 and tok is b

    def test_token_next_skip_ws(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, tok = tl.token_next(0)
        assert idx == 2 and tok is b

    def test_token_next_no_skip_ws(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, tok = tl.token_next(0, skip_ws=False)
        assert idx == 1 and tok is ws

    def test_token_next_none_idx(self):
        tl = TokenList([])
        assert tl.token_next(None) == (None, None)

    def test_token_next_end(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl.token_next(0) == (None, None)

    def test_token_prev(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        idx, tok = tl.token_prev(1)
        assert idx == 0 and tok is a

    def test_token_prev_none(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl.token_prev(0) == (None, None)

    def test_token_next_by_ttype(self):
        a = Token(T.Name, 'a')
        kw = Token(T.Keyword, 'SELECT')
        tl = TokenList([a, kw])
        idx, tok = tl.token_next_by(t=T.Keyword)
        assert idx == 1 and tok is kw

    def test_token_next_by_match(self):
        a = Token(T.Name, 'a')
        dot = Token(T.Punctuation, '.')
        tl = TokenList([a, dot])
        idx, tok = tl.token_next_by(m=(T.Punctuation, '.'))
        assert idx == 1 and tok is dot

    def test_token_next_by_instance(self):
        a = Token(T.Name, 'a')
        par = Parenthesis(make_tokens((T.Punctuation, '('), (T.Punctuation, ')')))
        tl = TokenList([a, par])
        idx, tok = tl.token_next_by(i=Parenthesis)
        assert idx == 1 and tok is par

    def test_token_next_by_none(self):
        tl = TokenList(make_tokens((T.Name, 'a')))
        assert tl.token_next_by(t=T.Keyword) == (None, None)

    def test_token_not_matching(self):
        # token_not_matching uses token matching functions starting at idx,
        # and the negation applies to each matcher, so the first token (which
        # is not whitespace) matches immediately and is returned at index 0.
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, tok = tl.token_not_matching(lambda t: t.is_whitespace, 0)
        assert idx == 0 and tok is a

    def test_token_matching(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_matching(lambda t: t.value == 'b', 0) is b
        assert tl.token_matching(lambda t: t.value == 'zzz', 0) is None

    def test_token_index_by_token(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_index(b) == 1

    def test_token_index_with_start_token(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        assert tl.token_index(c, start=b) == 2

    def test_token_index_with_start_int(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_index(b, start=0) == 1

    def test_group_tokens(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        grp = tl.group_tokens(TokenList, 0, 1)
        assert isinstance(grp, TokenList)
        assert len(tl.tokens) == 2
        assert tl.tokens[0] is grp
        assert tl.tokens[1] is c
        assert grp.parent is tl
        assert a.parent is grp
        assert b.parent is grp

    def test_group_tokens_no_include_end(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        grp = tl.group_tokens(TokenList, 0, 1, include_end=False)
        assert grp.tokens == [a]
        assert tl.tokens[1] is b

    def test_group_tokens_extend(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        grp = TokenList([a])
        tl = TokenList([grp, b, c])
        result = tl.group_tokens(TokenList, 0, 1, extend=True)
        assert result is grp
        assert grp.tokens == [a, b]
        assert grp.value == 'ab'

    def test_insert_before_index(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        x = Token(T.Name, 'x')
        tl.insert_before(0, x)
        assert tl.tokens[0] is x
        assert x.parent is tl

    def test_insert_before_token(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        x = Token(T.Name, 'x')
        tl.insert_before(a, x)
        assert tl.tokens == [x, a]

    def test_insert_after_index(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        x = Token(T.Name, 'x')
        tl.insert_after(0, x)
        assert tl.tokens[0] is a
        assert tl.tokens[1] is x
        assert x.parent is tl

    def test_insert_after_end_appends(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        x = Token(T.Name, 'x')
        tl.insert_after(a, x)
        assert tl.tokens[-1] is x

    def test_get_alias_default_none(self):
        assert TokenList().get_alias() is None

    def test_has_alias_default_false(self):
        assert TokenList().has_alias() is False

    def test_get_real_name_default_none(self):
        assert TokenList().get_real_name() is None

    def test_get_name_falls_back_to_real_name(self):
        assert TokenList().get_name() is None

    def test_get_parent_name_none(self):
        tl = TokenList(make_tokens((T.Name, 'a')))
        assert tl.get_parent_name() is None

    def test_get_parent_name_dot(self):
        tl = TokenList(make_tokens((T.Name, 'foo'), (T.Punctuation, '.'), (T.Name, 'bar')))
        assert tl.get_parent_name() == 'foo'

    def test_get_parent_name_removes_quotes(self):
        tl = TokenList(make_tokens((T.Name, '"foo"'), (T.Punctuation, '.'), (T.Name, 'bar')))
        assert tl.get_parent_name() == 'foo'

    def test_get_first_name_simple(self):
        tl = TokenList(make_tokens((T.Name, 'foo')))
        assert tl._get_first_name() == 'foo'

    def test_get_first_name_with_idx(self):
        tl = TokenList(make_tokens((T.Name, 'foo'), (T.Name, 'bar')))
        assert tl._get_first_name(1) == 'bar'

    def test_get_first_name_reverse(self):
        tl = TokenList(make_tokens((T.Name, 'foo'), (T.Name, 'bar')))
        assert tl._get_first_name(reverse=True) == 'bar'

    def test_get_first_name_keywords(self):
        tl = TokenList(make_tokens((T.Keyword, 'AS'), (T.Name, 'x')))
        assert tl._get_first_name(keywords=True) == 'AS'

    def test_get_first_name_removes_quotes(self):
        tl = TokenList(make_tokens((T.Name, "'foo'")))
        assert tl._get_first_name() == 'foo'

    def test_get_first_name_wildcard(self):
        tl = TokenList(make_tokens((T.Wildcard, '*')))
        assert tl._get_first_name() == '*'

    def test_get_first_name_none(self):
        tl = TokenList(make_tokens((T.Punctuation, '.')))
        assert tl._get_first_name() is None

    def test_pprint_tree(self, capsys):
        tl = TokenList(make_tokens((T.Name, 'a'), (T.Name, 'b')))
        tl._pprint_tree()
        out = capsys.readouterr().out
        assert '0 Name' in out
        assert '1 Name' in out

    def test_pprint_tree_nested(self, capsys):
        inner = TokenList(make_tokens((T.Name, 'x')))
        outer = TokenList([inner])
        outer._pprint_tree()
        out = capsys.readouterr().out
        assert 'TokenList' in out
        assert 'Name' in out

    def test_pprint_tree_max_depth(self, capsys):
        inner = TokenList(make_tokens((T.Name, 'x')))
        outer = TokenList([inner])
        outer._pprint_tree(max_depth=0)
        out = capsys.readouterr().out
        assert 'TokenList' in out
        assert 'Name' not in out


# ---------------------------------------------------------------------------
# NameAliasMixin / Identifier
# ---------------------------------------------------------------------------

class TestNameAliasMixin:
    def test_get_real_name_simple(self):
        ident = make_identifier((T.Name, 'foo'))
        assert ident.get_real_name() == 'foo'

    def test_get_real_name_dotted(self):
        ident = make_identifier(
            (T.Name, 'a'), (T.Punctuation, '.'),
            (T.Name, 'b'), (T.Punctuation, '.'),
            (T.Name, 'c'),
        )
        assert ident.get_real_name() == 'c'

    def test_get_real_name_quoted(self):
        ident = make_identifier((T.Name, '"foo"'))
        assert ident.get_real_name() == 'foo'

    def test_get_alias_none(self):
        ident = make_identifier((T.Name, 'foo'))
        assert ident.get_alias() is None

    def test_get_alias_as(self):
        ident = make_identifier(
            (T.Name, 'foo'), (T.Whitespace, ' '),
            (T.Keyword, 'AS'), (T.Whitespace, ' '),
            (T.Name, 'bar'),
        )
        assert ident.get_alias() == 'bar'

    def test_get_alias_implicit(self):
        ident = make_identifier(
            (T.Name, 'foo'), (T.Whitespace, ' '), (T.Name, 'bar'),
        )
        assert ident.get_alias() == 'bar'

    def test_get_alias_quoted(self):
        ident = make_identifier(
            (T.Name, 'foo'), (T.Whitespace, ' '), (T.Name, '"bar"'),
        )
        assert ident.get_alias() == 'bar'

    def test_get_name_prefers_alias(self):
        ident = make_identifier(
            (T.Name, 'foo'), (T.Whitespace, ' '), (T.Name, 'bar'),
        )
        assert ident.get_name() == 'bar'

    def test_get_name_no_alias(self):
        ident = make_identifier((T.Name, 'foo'))
        assert ident.get_name() == 'foo'

    def test_has_alias(self):
        ident = make_identifier(
            (T.Name, 'foo'), (T.Whitespace, ' '), (T.Name, 'bar'),
        )
        assert ident.has_alias() is True


class TestIdentifier:
    def test_is_wildcard_true(self):
        ident = make_identifier((T.Wildcard, '*'))
        assert ident.is_wildcard() is True

    def test_is_wildcard_false(self):
        ident = make_identifier((T.Name, 'foo'))
        assert ident.is_wildcard() is False

    def test_get_typecast(self):
        ident = make_identifier(
            (T.Name, 'x'), (T.Punctuation, '::'), (T.Name, 'int'),
        )
        assert ident.get_typecast() == 'int'

    def test_get_typecast_none(self):
        ident = make_identifier((T.Name, 'x'))
        assert ident.get_typecast() is None

    def test_get_ordering(self):
        ident = make_identifier((T.Name, 'x'), (T.Whitespace, ' '),
                                (T.Keyword.Order, 'desc'))
        assert ident.get_ordering() == 'DESC'

    def test_get_ordering_none(self):
        ident = make_identifier((T.Name, 'x'))
        assert ident.get_ordering() is None

    def test_get_array_indices(self):
        sq = SquareBrackets(make_tokens(
            (T.Punctuation, '['), (T.Literal.Number.Integer, '1'),
            (T.Punctuation, ']')))
        ident = Identifier([sq])
        indices = list(ident.get_array_indices())
        assert len(indices) == 1
        assert [t.value for t in indices[0]] == ['1']

    def test_get_array_indices_none(self):
        ident = make_identifier((T.Name, 'x'))
        assert list(ident.get_array_indices()) == []

    def test_get_real_name_nested_identifier(self):
        # identifier whose token is another Identifier
        inner = Identifier(make_tokens((T.Name, 'a'), (T.Punctuation, '.'),
                                       (T.Name, 'b')))
        outer = Identifier([inner])
        assert outer.get_real_name() == 'b'


# ---------------------------------------------------------------------------
# IdentifierList
# ---------------------------------------------------------------------------

class TestIdentifierList:
    def test_get_identifiers(self):
        a = Token(T.Name, 'a')
        comma = Token(T.Punctuation, ',')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        il = IdentifierList([a, comma, ws, b])
        assert list(il.get_identifiers()) == [a, b]

    def test_get_identifiers_empty(self):
        il = IdentifierList([])
        assert list(il.get_identifiers()) == []


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

class TestStatement:
    def test_get_type_dml(self):
        st = Statement(make_tokens((T.Keyword.DML, 'select'), (T.Whitespace, ' ')))
        assert st.get_type() == 'SELECT'

    def test_get_type_ddl(self):
        st = Statement(make_tokens((T.Keyword.DDL, 'create')))
        assert st.get_type() == 'CREATE'

    def test_get_type_unknown(self):
        st = Statement(make_tokens((T.Name, 'foo')))
        assert st.get_type() == 'UNKNOWN'

    def test_get_type_empty(self):
        st = Statement([])
        assert st.get_type() == 'UNKNOWN'

    def test_get_type_whitespace_only(self):
        st = Statement(make_tokens((T.Whitespace, ' ')))
        assert st.get_type() == 'UNKNOWN'

    def test_get_type_skips_leading_ws_and_comments(self):
        cm = Comment(make_tokens((T.Comment.Single, '-- x')))
        st = Statement([cm, Token(T.Whitespace, ' '),
                        Token(T.Keyword.DML, 'INSERT')])
        assert st.get_type() == 'INSERT'

    def test_get_type_with_cte(self):
        cte = Token(T.Keyword.CTE, 'WITH')
        ident = Identifier(make_tokens((T.Name, 'foo')))
        ws = Token(T.Whitespace, ' ')
        dml = Token(T.Keyword.DML, 'select')
        st = Statement([cte, ws, ident, ws, dml])
        assert st.get_type() == 'SELECT'

    def test_get_type_cte_no_dml(self):
        cte = Token(T.Keyword.CTE, 'WITH')
        ident = Identifier(make_tokens((T.Name, 'foo')))
        ws = Token(T.Whitespace, ' ')
        st = Statement([cte, ws, ident])
        assert st.get_type() == 'UNKNOWN'


# ---------------------------------------------------------------------------
# Simple group classes
# ---------------------------------------------------------------------------

class TestParenthesis:
    def test_groupable_tokens(self):
        p = Parenthesis(make_tokens(
            (T.Punctuation, '('), (T.Name, 'x'), (T.Punctuation, ')')))
        assert p._groupable_tokens == p.tokens[1:-1]

    def test_m_open_m_close(self):
        assert Parenthesis.M_OPEN == (T.Punctuation, '(')
        assert Parenthesis.M_CLOSE == (T.Punctuation, ')')


class TestSquareBrackets:
    def test_groupable_tokens(self):
        s = SquareBrackets(make_tokens(
            (T.Punctuation, '['), (T.Name, 'x'), (T.Punctuation, ']')))
        assert s._groupable_tokens == s.tokens[1:-1]

    def test_m_open_m_close(self):
        assert SquareBrackets.M_OPEN == (T.Punctuation, '[')
        assert SquareBrackets.M_CLOSE == (T.Punctuation, ']')


class TestComparison:
    def test_left_right(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Comparison([a, b])
        assert c.left is a
        assert c.right is b


class TestComment:
    def test_is_multiline_true(self):
        cm = Comment(make_tokens((T.Comment.Multiline, '/* x */')))
        assert cm.is_multiline() is True

    def test_is_multiline_false(self):
        cm = Comment(make_tokens((T.Comment.Single, '-- x')))
        assert cm.is_multiline() is False

    def test_is_multiline_empty(self):
        cm = Comment([])
        assert cm.is_multiline() == []


class TestClassAttributes:
    def test_typed_literal(self):
        assert TypedLiteral.M_OPEN == [(T.Name.Builtin, None), (T.Keyword, "TIMESTAMP")]
        assert TypedLiteral.M_CLOSE == (T.String.Single, None)
        assert TypedLiteral.M_EXTEND == (
            T.Keyword, ("DAY", "HOUR", "MINUTE", "MONTH", "SECOND", "YEAR"))

    def test_if(self):
        assert If.M_OPEN == (T.Keyword, 'IF')
        assert If.M_CLOSE == (T.Keyword, 'END IF')

    def test_for(self):
        assert For.M_OPEN == (T.Keyword, ('FOR', 'FOREACH'))
        assert For.M_CLOSE == (T.Keyword, 'END LOOP')

    def test_where(self):
        assert Where.M_OPEN == (T.Keyword, 'WHERE')
        assert 'ORDER BY' in Where.M_CLOSE[1]
        assert 'GROUP BY' in Where.M_CLOSE[1]

    def test_over(self):
        assert Over.M_OPEN == (T.Keyword, 'OVER')

    def test_having(self):
        assert Having.M_OPEN == (T.Keyword, 'HAVING')
        assert Having.M_CLOSE == (T.Keyword, ('ORDER BY', 'LIMIT'))

    def test_begin(self):
        assert Begin.M_OPEN == (T.Keyword, 'BEGIN')
        assert Begin.M_CLOSE == (T.Keyword, 'END')

    def test_group_subclasses(self):
        for cls in (Assignment, Operation, Values, Command):
            assert issubclass(cls, TokenList)


class TestSimpleGroupInstances:
    def test_assignment(self):
        a = Assignment(make_tokens((T.Name, 'x')))
        assert isinstance(a, TokenList)

    def test_operation(self):
        o = Operation(make_tokens((T.Name, 'x')))
        assert isinstance(o, TokenList)

    def test_values(self):
        v = Values(make_tokens((T.Name, 'x')))
        assert isinstance(v, TokenList)

    def test_command(self):
        c = Command(make_tokens((T.Name, 'x')))
        assert isinstance(c, TokenList)

    def test_if(self):
        i = If(make_tokens((T.Keyword, 'IF')))
        assert isinstance(i, TokenList)

    def test_for(self):
        f = For(make_tokens((T.Keyword, 'FOR')))
        assert isinstance(f, TokenList)

    def test_where(self):
        w = Where(make_tokens((T.Keyword, 'WHERE')))
        assert isinstance(w, TokenList)

    def test_over(self):
        o = Over(make_tokens((T.Keyword, 'OVER')))
        assert isinstance(o, TokenList)

    def test_having(self):
        h = Having(make_tokens((T.Keyword, 'HAVING')))
        assert isinstance(h, TokenList)

    def test_begin(self):
        b = Begin(make_tokens((T.Keyword, 'BEGIN')))
        assert isinstance(b, TokenList)


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

class TestCase:
    def test_get_cases_simple(self):
        tokens = make_tokens(
            (T.Keyword, 'CASE'),
            (T.Whitespace, ' '),
            (T.Keyword, 'WHEN'),
            (T.Whitespace, ' '),
            (T.Name, 'x'),
            (T.Whitespace, ' '),
            (T.Keyword, 'THEN'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '1'),
            (T.Whitespace, ' '),
            (T.Keyword, 'END'),
        )
        c = Case(tokens)
        cases = c.get_cases(skip_ws=True)
        assert len(cases) == 1
        cond, val = cases[0]
        # The WHEN keyword is appended to the condition list before the mode
        # check; the THEN keyword is likewise appended to the value list.
        assert [t.value for t in cond] == ['WHEN', 'x']
        assert [t.value for t in val] == ['THEN', '1']

    def test_get_cases_else(self):
        tokens = make_tokens(
            (T.Keyword, 'CASE'),
            (T.Keyword, 'WHEN'),
            (T.Name, 'x'),
            (T.Keyword, 'THEN'),
            (T.Literal.Number.Integer, '1'),
            (T.Keyword, 'ELSE'),
            (T.Literal.Number.Integer, '2'),
            (T.Keyword, 'END'),
        )
        c = Case(tokens)
        cases = c.get_cases()
        assert len(cases) == 2
        # First condition list begins with the 'WHEN' keyword token; the
        # value list begins with the 'THEN' keyword token.
        assert cases[0][0][0].value == 'WHEN'
        assert cases[0][0][1].value == 'x'
        assert cases[0][1][0].value == 'THEN'
        assert cases[0][1][1].value == '1'
        # The ELSE case has a None condition; its value list begins with
        # the 'ELSE' keyword token, followed by the else value.
        assert cases[1][0] is None
        assert cases[1][1][0].value == 'ELSE'
        assert cases[1][1][1].value == '2'

    def test_get_cases_multiple_when(self):
        tokens = make_tokens(
            (T.Keyword, 'CASE'),
            (T.Keyword, 'WHEN'),
            (T.Name, 'a'),
            (T.Keyword, 'THEN'),
            (T.Literal.Number.Integer, '1'),
            (T.Keyword, 'WHEN'),
            (T.Name, 'b'),
            (T.Keyword, 'THEN'),
            (T.Literal.Number.Integer, '2'),
            (T.Keyword, 'END'),
        )
        c = Case(tokens)
        cases = c.get_cases()
        assert len(cases) == 2
        assert cases[0][0][0].value == 'WHEN'
        assert cases[0][0][1].value == 'a'
        assert cases[1][0][0].value == 'WHEN'
        assert cases[1][0][1].value == 'b'

    def test_get_cases_empty(self):
        c = Case([])
        assert c.get_cases() == []


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

class TestFunction:
    def test_get_parameters_no_paren(self):
        f = Function(make_tokens((T.Name, 'foo')))
        with pytest.raises(AttributeError):
            f.get_parameters()

    def test_get_parameters_identifiers(self):
        a = Identifier(make_tokens((T.Name, 'a')))
        b = Identifier(make_tokens((T.Name, 'b')))
        inner = [a, Token(T.Punctuation, ','), Token(T.Whitespace, ' '), b]
        par = Parenthesis([Token(T.Punctuation, '(')] + inner +
                          [Token(T.Punctuation, ')')])
        f = Function([Token(T.Name, 'func'), par])
        params = f.get_parameters()
        assert params == [a, b]

    def test_get_parameters_identifier_list(self):
        a = Identifier(make_tokens((T.Name, 'a')))
        b = Identifier(make_tokens((T.Name, 'b')))
        il = IdentifierList([a, Token(T.Punctuation, ','), b])
        par = Parenthesis([Token(T.Punctuation, '('), il,
                           Token(T.Punctuation, ')')])
        f = Function([Token(T.Name, 'func'), par])
        params = list(f.get_parameters())
        assert params == [a, b]

    def test_get_parameters_literal(self):
        lit = Token(T.Literal, '1')
        par = Parenthesis([Token(T.Punctuation, '('), lit,
                           Token(T.Punctuation, ')')])
        f = Function([Token(T.Name, 'func'), par])
        assert f.get_parameters() == [lit]

    def test_get_window_none(self):
        f = Function(make_tokens((T.Name, 'foo')))
        with pytest.raises(AttributeError):
            f.get_window()

    def test_get_window(self):
        last = Token(T.Name, 'w')
        ov = Over([Token(T.Keyword, 'OVER'), last])
        f = Function([Token(T.Name, 'foo'), ov])
        assert f.get_window() is last


# ---------------------------------------------------------------------------
# NameAliasMixin on Function
# ---------------------------------------------------------------------------

class TestFunctionAlias:
    def test_function_get_alias(self):
        f = Function(make_tokens(
            (T.Name, 'func'),
            (T.Punctuation, '('), (T.Punctuation, ')'),
            (T.Whitespace, ' '),
            (T.Keyword, 'AS'),
            (T.Whitespace, ' '),
            (T.Name, 'alias'),
        ))
        assert f.get_alias() == 'alias'
