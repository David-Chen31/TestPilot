"""Unit tests for sqlparse.sql"""

import re
import pytest

from sqlparse import tokens as T
from sqlparse.sql import (
    NameAliasMixin, Token, TokenList, Statement, Identifier, IdentifierList,
    TypedLiteral, Parenthesis, SquareBrackets, Assignment, If, For,
    Comparison, Comment, Where, Over, Having, Case, Function, Begin,
    Operation, Values, Command,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_list(*tokens):
    return TokenList(list(tokens))


def tok(ttype, value):
    return Token(ttype, value)


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

    def test_init_coerces_value_to_str(self):
        t = Token(T.Name, 123)
        assert t.value == '123'
        assert isinstance(t.value, str)

    def test_init_keyword_normalized_upper(self):
        t = Token(T.Keyword, 'select')
        assert t.is_keyword is True
        assert t.normalized == 'SELECT'

    def test_init_whitespace(self):
        t = Token(T.Whitespace, ' ')
        assert t.is_whitespace is True
        assert t.is_newline is False

    def test_init_newline(self):
        t = Token(T.Newline, '\n')
        assert t.is_newline is True
        assert t.is_whitespace is True

    def test_str(self):
        assert str(Token(T.Name, 'abc')) == 'abc'

    def test_repr(self):
        t = Token(T.Name, 'foo')
        r = repr(t)
        assert r.startswith('<Name ')
        assert "'foo'" in r

    def test_repr_quoted_value_uses_double_quotes(self):
        t = Token(T.String.Single, "'abc'")
        r = repr(t)
        assert r.startswith("<String.Single ")
        assert '"\'abc\'"' in r

    def test_repr_truncates_long_value(self):
        t = Token(T.Name, 'abcdefghij')
        r = repr(t)
        assert "'abcdef...'" in r

    def test_repr_collapses_whitespace(self):
        t = Token(T.Name, 'a   b')
        r = repr(t)
        assert "'a b'" in r

    def test_flatten_yields_self(self):
        t = Token(T.Name, 'x')
        assert list(t.flatten()) == [t]

    def test_match_simple(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, 'foo') is True
        assert t.match(T.Name, 'bar') is False

    def test_match_wrong_type(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Keyword, 'foo') is False

    def test_match_none_values_returns_type_match(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, None) is True
        assert t.match(T.Keyword, None) is False

    def test_match_keyword_case_insensitive(self):
        t = Token(T.Keyword, 'Select')
        assert t.match(T.Keyword, 'select') is True
        assert t.match(T.Keyword, 'SELECT') is True

    def test_match_list_of_values(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, ['bar', 'foo']) is True
        assert t.match(T.Name, ['bar', 'baz']) is False

    def test_match_regex(self):
        t = Token(T.Name, 'foobar')
        assert t.match(T.Name, 'foo', regex=True) is True
        assert t.match(T.Name, '^bar', regex=True) is False

    def test_match_regex_keyword_ignorecase(self):
        t = Token(T.Keyword, 'SELECT')
        assert t.match(T.Keyword, 'sel', regex=True) is True

    def test_within(self):
        outer = TokenList()
        inner = TokenList()
        leaf = Token(T.Name, 'x')
        outer.tokens.append(inner)
        inner.parent = outer
        inner.tokens.append(leaf)
        leaf.parent = inner
        assert leaf.within(TokenList) is True
        assert leaf.within(Identifier) is False

    def test_is_child_of(self):
        parent = TokenList()
        child = Token(T.Name, 'x')
        parent.tokens.append(child)
        child.parent = parent
        assert child.is_child_of(parent) is True
        assert child.is_child_of(TokenList()) is False

    def test_has_ancestor(self):
        grand = TokenList()
        child = TokenList()
        leaf = Token(T.Name, 'x')
        grand.tokens.append(child)
        child.parent = grand
        child.tokens.append(leaf)
        leaf.parent = child
        assert leaf.has_ancestor(grand) is True
        assert child.has_ancestor(grand) is True
        assert grand.has_ancestor(child) is False


# ---------------------------------------------------------------------------
# TokenList
# ---------------------------------------------------------------------------

class TestTokenList:
    def test_init_empty(self):
        tl = TokenList()
        assert tl.tokens == []
        assert tl.value == ''
        assert tl.is_group is True

    def test_init_sets_parent(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert a.parent is tl
        assert b.parent is tl

    def test_init_value_concatenation(self):
        tl = TokenList([Token(T.Name, 'foo'), Token(T.Name, 'bar')])
        assert tl.value == 'foobar'

    def test_str_flattens(self):
        inner = TokenList([Token(T.Name, 'foo')])
        outer = TokenList([inner, Token(T.Name, 'bar')])
        assert str(outer) == 'foobar'

    def test_iter_and_getitem(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert list(tl) == [a, b]
        assert tl[0] is a
        assert tl[-1] is b

    def test_repr_name(self):
        assert TokenList()._get_repr_name() == 'TokenList'
        assert Statement()._get_repr_name() == 'Statement'

    def test_flatten_nested(self):
        leaf1 = Token(T.Name, 'a')
        leaf2 = Token(T.Name, 'b')
        inner = TokenList([leaf1, leaf2])
        outer = TokenList([inner, Token(T.Name, 'c')])
        assert list(outer.flatten()) == [leaf1, leaf2, outer.tokens[1]]

    def test_get_sublists(self):
        inner = TokenList()
        leaf = Token(T.Name, 'a')
        outer = TokenList([leaf, inner])
        assert list(outer.get_sublists()) == [inner]

    def test_groupable_tokens_default(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl._groupable_tokens is tl.tokens

    def test_parenthesis_groupable_tokens(self):
        lp = Token(T.Punctuation, '(')
        inner = Token(T.Name, 'a')
        rp = Token(T.Punctuation, ')')
        p = Parenthesis([lp, inner, rp])
        assert p._groupable_tokens == [inner]

    def test_square_brackets_groupable_tokens(self):
        lb = Token(T.Punctuation, '[')
        inner = Token(T.Name, 'a')
        rb = Token(T.Punctuation, ']')
        sb = SquareBrackets([lb, inner, rb])
        assert sb._groupable_tokens == [inner]

    def test_get_token_at_offset(self):
        t = Token(T.Name, 'hello')
        tl = TokenList([t])
        assert tl.get_token_at_offset(0) is t
        assert tl.get_token_at_offset(4) is t
        assert tl.get_token_at_offset(5) is None

    def test_get_token_at_offset_multiple(self):
        a = Token(T.Name, 'ab')
        b = Token(T.Name, 'cd')
        tl = TokenList([a, b])
        assert tl.get_token_at_offset(1) is a
        assert tl.get_token_at_offset(2) is b
        assert tl.get_token_at_offset(3) is b

    def test_token_first_skips_ws(self):
        ws = Token(T.Whitespace, ' ')
        name = Token(T.Name, 'x')
        tl = TokenList([ws, name])
        assert tl.token_first() is name
        assert tl.token_first(skip_ws=False) is ws

    def test_token_first_empty(self):
        assert TokenList().token_first() is None

    def test_token_first_skip_cm(self):
        cm = Comment([Token(T.Comment.Single, '-- hi')])
        name = Token(T.Name, 'x')
        tl = TokenList([cm, name])
        assert tl.token_first(skip_cm=True) is name
        assert tl.token_first(skip_cm=False) is cm

    def test_token_next_by_type(self):
        name = Token(T.Name, 'x')
        tl = TokenList([Token(T.Whitespace, ' '), name])
        idx, tok_ = tl.token_next_by(t=T.Name)
        assert idx == 1
        assert tok_ is name

    def test_token_next_by_none(self):
        tl = TokenList([Token(T.Whitespace, ' ')])
        assert tl.token_next_by(t=T.Name) == (None, None)

    def test_token_next_by_match(self):
        dot = Token(T.Punctuation, '.')
        tl = TokenList([Token(T.Name, 'a'), dot])
        idx, tok_ = tl.token_next_by(m=(T.Punctuation, '.'))
        assert idx == 1
        assert tok_ is dot

    def test_token_next_by_instance(self):
        p = Parenthesis([Token(T.Punctuation, '('), Token(T.Punctuation, ')')])
        tl = TokenList([Token(T.Name, 'x'), p])
        idx, tok_ = tl.token_next_by(i=Parenthesis)
        assert tok_ is p

    def test_token_next(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        assert tl.token_next(0) == (2, b)
        assert tl.token_next(0, skip_ws=False) == (1, ws)
        assert tl.token_next(2) == (None, None)
        assert tl.token_next(None) == (None, None)

    def test_token_prev(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        assert tl.token_prev(2) == (0, a)
        assert tl.token_prev(2, skip_ws=False) == (1, ws)

    def test_token_prev_first(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.token_prev(0) == (None, None)

    def test_token_matching(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])

        def matcher(tk):
            return tk.value == 'b'

        assert tl.token_matching(matcher, 0) is b

    def test_token_not_matching(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])

        def matcher(tk):
            return tk.value == 'a'

        idx, tok_ = tl.token_not_matching(matcher, 0)
        assert tok_ is b

    def test_token_index(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_index(b) == 1
        assert tl.token_index(b, start=1) == 1

    def test_token_index_by_token(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        assert tl.token_index(c, start=a) == 2

    def test_insert_before_int(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        n = Token(T.Name, 'n')
        tl.insert_before(1, n)
        assert tl.tokens == [a, n, b]
        assert n.parent is tl

    def test_insert_before_token(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        n = Token(T.Name, 'n')
        tl.insert_before(b, n)
        assert tl.tokens == [a, n, b]

    def test_insert_after_int(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        n = Token(T.Name, 'n')
        tl.insert_after(0, n)
        assert tl.tokens == [a, n, b]
        assert n.parent is tl

    def test_insert_after_appends(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        n = Token(T.Name, 'n')
        tl.insert_after(0, n)
        assert tl.tokens == [a, n]

    def test_insert_after_skips_ws(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        n = Token(T.Name, 'n')
        tl.insert_after(0, n)
        assert tl.tokens == [a, ws, n, b]

    def test_group_tokens(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        grp = tl.group_tokens(Parenthesis, 0, 1)
        assert isinstance(grp, Parenthesis)
        assert tl.tokens == [grp, c]
        assert grp.tokens == [a, b]
        assert grp.parent is tl
        assert a.parent is grp
        assert b.parent is grp

    def test_group_tokens_include_end_false(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        grp = tl.group_tokens(Parenthesis, 0, 1, include_end=False)
        assert grp.tokens == [a]
        assert tl.tokens == [grp, b, c]

    def test_group_tokens_extend(self):
        a = Token(T.Name, 'a')
        existing = Parenthesis([a])
        a.parent = existing
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([existing, b, c])
        grp = tl.group_tokens(Parenthesis, 0, 1, extend=True)
        assert grp is existing
        assert grp.tokens == [a, b]
        assert tl.tokens == [grp, c]
        assert grp.value == 'ab'
        assert b.parent is grp

    def test_has_alias_default(self):
        assert TokenList().has_alias() is False

    def test_get_alias_default(self):
        assert TokenList().get_alias() is None

    def test_get_name_uses_real_name(self):
        tl = TokenList([Token(T.Name, 'foo')])
        assert tl.get_name() == 'foo'

    def test_get_real_name_default(self):
        assert TokenList().get_real_name() is None

    def test_get_parent_name_no_dot(self):
        tl = TokenList([Token(T.Name, 'foo')])
        assert tl.get_parent_name() is None

    def test_get_parent_name_with_dot(self):
        a = Token(T.Name, 'a')
        dot = Token(T.Punctuation, '.')
        b = Token(T.Name, 'b')
        tl = TokenList([a, dot, b])
        assert tl.get_parent_name() == 'a'

    def test_get_parent_name_removes_quotes(self):
        a = Token(T.Name, '"a"')
        dot = Token(T.Punctuation, '.')
        tl = TokenList([a, dot, Token(T.Name, 'b')])
        assert tl.get_parent_name() == 'a'

    def test_pprint_tree(self, capsys):
        a = Token(T.Name, 'foo')
        tl = TokenList([a])
        tl._pprint_tree()
        out = capsys.readouterr().out
        assert 'TokenList' not in out
        assert '0' in out
        assert 'foo' in out

    def test_pprint_tree_nested(self, capsys):
        inner = TokenList([Token(T.Name, 'x')])
        outer = TokenList([inner])
        outer._pprint_tree()
        out = capsys.readouterr().out
        assert 'TokenList' in out
        assert "'x'" in out


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

class TestStatement:
    def test_empty(self):
        assert Statement().get_type() == 'UNKNOWN'

    def test_only_whitespace(self):
        st = Statement([Token(T.Whitespace, ' ')])
        assert st.get_type() == 'UNKNOWN'

    def test_dml(self):
        st = Statement([Token(T.Keyword.DML, 'select')])
        assert st.get_type() == 'SELECT'

    def test_ddl(self):
        st = Statement([Token(T.Keyword.DDL, 'create')])
        assert st.get_type() == 'CREATE'

    def test_skip_leading_comment(self):
        cm = Comment([Token(T.Comment.Single, '-- c')])
        st = Statement([cm, Token(T.Keyword.DML, 'select')])
        assert st.get_type() == 'SELECT'

    def test_unknown_keyword(self):
        st = Statement([Token(T.Keyword, 'FOO')])
        assert st.get_type() == 'UNKNOWN'

    def test_cte_with_dml(self):
        with_kw = Token(T.Keyword.CTE, 'WITH')
        ident = Identifier([Token(T.Name, 'cte')])
        dml = Token(T.Keyword.DML, 'select')
        st = Statement([with_kw, Token(T.Whitespace, ' '), ident,
                        Token(T.Whitespace, ' '), dml])
        assert st.get_type() == 'SELECT'

    def test_cte_no_dml(self):
        with_kw = Token(T.Keyword.CTE, 'WITH')
        ident = Identifier([Token(T.Name, 'cte')])
        st = Statement([with_kw, Token(T.Whitespace, ' '), ident])
        assert st.get_type() == 'UNKNOWN'

    def test_cte_with_identifier_list(self):
        with_kw = Token(T.Keyword.CTE, 'WITH')
        il = IdentifierList([Identifier([Token(T.Name, 'a')])])
        dml = Token(T.Keyword.DML, 'insert')
        st = Statement([with_kw, Token(T.Whitespace, ' '), il,
                        Token(T.Whitespace, ' '), dml])
        assert st.get_type() == 'INSERT'


# ---------------------------------------------------------------------------
# Identifier / NameAliasMixin
# ---------------------------------------------------------------------------

class TestIdentifier:
    def test_get_real_name_simple(self):
        ident = Identifier([Token(T.Name, 'foo')])
        assert ident.get_real_name() == 'foo'

    def test_get_real_name_dotted(self):
        ident = Identifier([
            Token(T.Name, 'a'),
            Token(T.Punctuation, '.'),
            Token(T.Name, 'b'),
            Token(T.Punctuation, '.'),
            Token(T.Name, 'c'),
        ])
        assert ident.get_real_name() == 'c'

    def test_get_real_name_removes_quotes(self):
        ident = Identifier([Token(T.Name, '"foo"')])
        assert ident.get_real_name() == 'foo'

    def test_get_real_name_nested_identifier(self):
        inner = Identifier([Token(T.Name, 'inner')])
        ident = Identifier([inner])
        assert ident.get_real_name() == 'inner'

    def test_get_real_name_wildcard(self):
        ident = Identifier([Token(T.Wildcard, '*')])
        assert ident.get_real_name() == '*'

    def test_get_alias_as(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'AS'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        assert ident.get_alias() == 'bar'

    def test_get_alias_as_quoted(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'AS'),
            Token(T.Whitespace, ' '),
            Token(T.Name, '"bar"'),
        ])
        assert ident.get_alias() == 'bar'

    def test_get_alias_implicit(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        assert ident.get_alias() == 'bar'

    def test_get_alias_none_single(self):
        ident = Identifier([Token(T.Name, 'foo')])
        assert ident.get_alias() is None

    def test_get_name_prefers_alias(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        assert ident.get_name() == 'bar'

    def test_has_alias_true(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        assert ident.has_alias() is True

    def test_is_wildcard(self):
        ident = Identifier([Token(T.Wildcard, '*')])
        assert ident.is_wildcard() is True

    def test_is_wildcard_false(self):
        ident = Identifier([Token(T.Name, 'foo')])
        assert ident.is_wildcard() is False

    def test_get_typecast(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Punctuation, '::'),
            Token(T.Name.Builtin, 'int'),
        ])
        assert ident.get_typecast() == 'int'

    def test_get_typecast_none(self):
        ident = Identifier([Token(T.Name, 'foo')])
        assert ident.get_typecast() is None

    def test_get_ordering(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Keyword.Order, 'desc'),
        ])
        assert ident.get_ordering() == 'DESC'

    def test_get_ordering_none(self):
        ident = Identifier([Token(T.Name, 'foo')])
        assert ident.get_ordering() is None

    def test_get_array_indices(self):
        idx = TokenList([Token(T.Number.Integer, '1')])
        sb = SquareBrackets([
            Token(T.Punctuation, '['),
            idx,
            Token(T.Punctuation, ']'),
        ])
        ident = Identifier([Token(T.Name, 'arr'), sb])
        result = list(ident.get_array_indices())
        assert len(result) == 1
        assert result[0].tokens == [idx]
        assert result[0] is sb.tokens[1:-1]


# ---------------------------------------------------------------------------
# IdentifierList
# ---------------------------------------------------------------------------

class TestIdentifierList:
    def test_get_identifiers_skips_ws_and_commas(self):
        i1 = Identifier([Token(T.Name, 'a')])
        i2 = Identifier([Token(T.Name, 'b')])
        il = IdentifierList([
            i1,
            Token(T.Punctuation, ','),
            Token(T.Whitespace, ' '),
            i2,
        ])
        assert list(il.get_identifiers()) == [i1, i2]

    def test_get_identifiers_empty(self):
        assert list(IdentifierList().get_identifiers()) == []


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

class TestCase:
    def test_get_cases_simple(self):
        case = Case([
            Token(T.Keyword, 'CASE'),
            Token(T.Name, 'x'),
            Token(T.Keyword, 'WHEN'),
            Token(T.Name, 'cond'),
            Token(T.Keyword, 'THEN'),
            Token(T.Name, 'val'),
            Token(T.Keyword, 'END'),
        ])
        cases = case.get_cases()
        # First bare segment is (x, []) then WHEN gives (cond, val)
        assert len(cases) == 2
        assert [t.value for t in cases[0][0]] == ['x']
        assert cases[0][1] == []
        assert [t.value for t in cases[1][0]] == ['cond']
        assert [t.value for t in cases[1][1]] == ['val']

    def test_get_cases_else(self):
        case = Case([
            Token(T.Keyword, 'CASE'),
            Token(T.Keyword, 'WHEN'),
            Token(T.Name, 'cond'),
            Token(T.Keyword, 'THEN'),
            Token(T.Name, 'val'),
            Token(T.Keyword, 'ELSE'),
            Token(T.Name, 'other'),
            Token(T.Keyword, 'END'),
        ])
        cases = case.get_cases()
        assert len(cases) == 2
        assert cases[0][0][0].value == 'cond'
        assert cases[0][1][0].value == 'val'
        assert cases[1][0] is None
        assert cases[1][1][0].value == 'other'

    def test_get_cases_skip_ws(self):
        case = Case([
            Token(T.Keyword, 'CASE'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'WHEN'),
            Token(T.Name, 'c'),
            Token(T.Keyword, 'THEN'),
            Token(T.Name, 'v'),
            Token(T.Keyword, 'END'),
        ])
        cases = case.get_cases(skip_ws=True)
        assert len(cases) == 1
        assert cases[0][0] == []
        # 'c' was collected in CONDITION mode
        assert cases[0][1][0].value == 'v'

    def test_get_cases_empty(self):
        assert Case().get_cases() == []


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

class TestFunction:
    def test_get_parameters_simple(self):
        p = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Literal, '1'),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), p])
        params = fn.get_parameters()
        assert len(params) == 1
        assert params[0].value == '1'

    def test_get_parameters_identifiers_in_list(self):
        il = IdentifierList([
            Identifier([Token(T.Name, 'a')]),
            Token(T.Punctuation, ','),
            Identifier([Token(T.Name, 'b')]),
        ])
        p = Parenthesis([
            Token(T.Punctuation, '('),
            il,
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), p])
        params = list(fn.get_parameters())
        assert len(params) == 2
        assert params[0].value == 'a'
        assert params[1].value == 'b'

    def test_get_parameters_empty(self):
        p = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), p])
        assert fn.get_parameters() == []

    def test_get_window_none(self):
        p = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), p])
        assert fn.get_window() is None

    def test_get_window_present(self):
        over = Over([
            Token(T.Keyword, 'OVER'),
            Token(T.Whitespace, ' '),
            Parenthesis([
                Token(T.Punctuation, '('),
                Token(T.Punctuation, ')'),
            ]),
        ])
        fn = Function([Token(T.Name, 'f'), over])
        win = fn.get_window()
        assert win is over.tokens[-1]

    def test_get_alias_as(self):
        fn = Function([
            Token(T.Name, 'f'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'AS'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'alias'),
        ])
        assert fn.get_alias() == 'alias'


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
        assert bool(Comment().is_multiline()) is False


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestComparison:
    def test_left_right(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        cmp = Comparison([a, Token(T.Operator, '='), b])
        assert cmp.left is a
        assert cmp.right is b


# ---------------------------------------------------------------------------
# Subclass constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_paren_constants(self):
        assert Parenthesis.M_OPEN == (T.Punctuation, '(')
        assert Parenthesis.M_CLOSE == (T.Punctuation, ')')

    def test_square_constants(self):
        assert SquareBrackets.M_OPEN == (T.Punctuation, '[')
        assert SquareBrackets.M_CLOSE == (T.Punctuation, ']')

    def test_if_constants(self):
        assert If.M_OPEN == (T.Keyword, 'IF')
        assert If.M_CLOSE == (T.Keyword, 'END IF')

    def test_for_constants(self):
        assert For.M_OPEN == (T.Keyword, ('FOR', 'FOREACH'))
        assert For.M_CLOSE == (T.Keyword, 'END LOOP')

    def test_where_constants(self):
        assert Where.M_OPEN == (T.Keyword, 'WHERE')
        assert 'LIMIT' in Where.M_CLOSE[1]

    def test_over_constants(self):
        assert Over.M_OPEN == (T.Keyword, 'OVER')

    def test_having_constants(self):
        assert Having.M_OPEN == (T.Keyword, 'HAVING')
        assert Having.M_CLOSE == (T.Keyword, ('ORDER BY', 'LIMIT'))

    def test_case_constants(self):
        assert Case.M_OPEN == (T.Keyword, 'CASE')
        assert Case.M_CLOSE == (T.Keyword, 'END')

    def test_begin_constants(self):
        assert Begin.M_OPEN == (T.Keyword, 'BEGIN')
        assert Begin.M_CLOSE == (T.Keyword, 'END')

    def test_typedliteral_constants(self):
        assert TypedLiteral.M_CLOSE == (T.String.Single, None)
        assert (T.Name.Builtin, None) in TypedLiteral.M_OPEN

    def test_simple_subclasses_are_tokenlist(self):
        for cls in (Assignment, If, For, Comparison, Where, Over, Having,
                    Case, Begin, Operation, Values, Command):
            assert issubclass(cls, TokenList)
            inst = cls()
            assert inst.is_group is True
