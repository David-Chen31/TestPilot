"""Unit tests for sqlparse.sql."""

import pytest

from sqlparse import tokens as T
from sqlparse.sql import (
    Assignment,
    Begin,
    Case,
    Command,
    Comment,
    Comparison,
    For,
    Function,
    Having,
    Identifier,
    IdentifierList,
    If,
    NameAliasMixin,
    Operation,
    Over,
    Parenthesis,
    SquareBrackets,
    Statement,
    Token,
    TokenList,
    TypedLiteral,
    Values,
    Where,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_token_list(values, ttype=T.Name):
    """Build a TokenList from a list of (ttype, value) pairs."""
    return TokenList([Token(t, v) for t, v in values])


def name(value):
    return Token(T.Name, value)


def ws(value=' '):
    return Token(T.Whitespace, value)


def kw(value):
    return Token(T.Keyword, value)


def punct(value):
    return Token(T.Punctuation, value)


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

class TestToken:
    def test_init_value_stringified(self):
        tok = Token(T.Name, 123)
        assert tok.value == '123'
        assert tok.ttype == T.Name
        assert tok.parent is None
        assert tok.is_group is False

    def test_keyword_flags(self):
        tok = Token(T.Keyword, 'select')
        assert tok.is_keyword is True
        assert tok.normalized == 'SELECT'

    def test_non_keyword_normalized(self):
        tok = Token(T.Name, 'Foo')
        assert tok.is_keyword is False
        assert tok.normalized == 'Foo'

    def test_whitespace_flag(self):
        assert Token(T.Whitespace, ' ').is_whitespace is True
        assert Token(T.Name, 'x').is_whitespace is False

    def test_newline_flag(self):
        assert Token(T.Whitespace.Newline, '\n').is_newline is True
        assert Token(T.Whitespace, ' ').is_newline is False

    def test_str_and_repr(self):
        tok = Token(T.Name, "abc")
        assert str(tok) == 'abc'
        r = repr(tok)
        assert r.startswith('<Name ')
        assert 'abc' in r

    def test_repr_truncation_and_quote(self):
        tok = Token(T.String.Single, "'abcdefgh'")
        r = repr(tok)
        # Value gets truncated to first 6 chars + '...', so it no longer
        # ends with a quote -> single-quote wrapper is used.
        assert r.startswith("<Single '")
        assert "'abcde..." in r

    def test_repr_short_value_no_truncation(self):
        tok = Token(T.Name, "abc")
        assert "'abc'" in repr(tok)

    def test_repr_whitespace_collapsed(self):
        tok = Token(T.Name, "a\n b\tc")
        assert 'a b c' in repr(tok)

    def test_flatten_yields_self(self):
        tok = Token(T.Name, 'x')
        assert list(tok.flatten()) == [tok]

    def test_match_type_only(self):
        tok = Token(T.Name, 'abc')
        assert tok.match(T.Name, None) is True
        assert tok.match(T.Keyword, None) is False

    def test_match_values_string(self):
        tok = Token(T.Name, 'abc')
        assert tok.match(T.Name, 'abc') is True
        assert tok.match(T.Name, 'abd') is False

    def test_match_keyword_case_insensitive(self):
        tok = Token(T.Keyword, 'select')
        assert tok.match(T.Keyword, 'SELECT') is True
        assert tok.match(T.Keyword, ('select', 'x')) is True

    def test_match_regex_keyword(self):
        tok = Token(T.Keyword, 'select')
        assert tok.match(T.Keyword, 'SEL', regex=True) is True
        assert tok.match(T.Keyword, 'xyz', regex=True) is False

    def test_match_regex_non_keyword(self):
        tok = Token(T.Name, 'FooBar')
        assert tok.match(T.Name, 'foo', regex=True) is False
        assert tok.match(T.Name, 'Foo', regex=True) is True

    def test_within(self):
        inner = Token(T.Name, 'x')
        outer = Parenthesis([Token(T.Punctuation, '('), inner,
                             Token(T.Punctuation, ')')])
        assert inner.within(Parenthesis) is True
        assert inner.within(Identifier) is False

    def test_is_child_of(self):
        child = Token(T.Name, 'x')
        parent = TokenList([child])
        assert child.is_child_of(parent) is True
        assert parent.is_child_of(child) is False

    def test_has_ancestor(self):
        inner = Token(T.Name, 'x')
        mid = Parenthesis([Token(T.Punctuation, '('), inner,
                           Token(T.Punctuation, ')')])
        outer = TokenList([mid])
        assert inner.has_ancestor(outer) is True
        assert mid.has_ancestor(outer) is True
        assert outer.has_ancestor(mid) is False


# ---------------------------------------------------------------------------
# TokenList
# ---------------------------------------------------------------------------

class TestTokenList:
    def test_init_assigns_parents(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, b])
        assert a.parent is tl
        assert b.parent is tl
        assert tl.is_group is True
        assert tl.ttype is None

    def test_value_concatenation(self):
        tl = TokenList([name('a'), ws(), name('b')])
        assert tl.value == 'a b'
        assert str(tl) == 'a b'

    def test_iter_and_getitem(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, b])
        assert list(tl) == [a, b]
        assert tl[0] is a
        assert tl[-1] is b

    def test_repr_name(self):
        assert TokenList([name('a')])._get_repr_name() == 'TokenList'

    def test_empty_init(self):
        tl = TokenList()
        assert tl.tokens == []
        assert str(tl) == ''

    def test_flatten_recursive(self):
        inner_tok = name('inner')
        inner = Parenthesis([Token(T.Punctuation, '('), inner_tok,
                             Token(T.Punctuation, ')')])
        outer = TokenList([name('a'), inner])
        flat = list(outer.flatten())
        assert inner not in flat
        assert inner_tok in flat
        # flattened excludes groups
        assert all(not t.is_group for t in flat)

    def test_get_sublists(self):
        inner = Parenthesis([Token(T.Punctuation, '('),
                             Token(T.Punctuation, ')')])
        outer = TokenList([name('a'), inner])
        assert list(outer.get_sublists()) == [inner]

    def test_get_token_at_offset(self):
        tl = TokenList([name('ab'), name('cd')])
        assert tl.get_token_at_offset(0) is tl[0]
        assert tl.get_token_at_offset(1) is tl[0]
        assert tl.get_token_at_offset(2) is tl[1]
        assert tl.get_token_at_offset(3) is tl[1]
        assert tl.get_token_at_offset(4) is None

    def test_token_first_skip_ws(self):
        a = name('a')
        tl = TokenList([ws(), a])
        assert tl.token_first() is a
        assert tl.token_first(skip_ws=False) is tl[0]

    def test_token_first_skip_cm(self):
        comment = Comment([Token(T.Comment.Single, '--x')])
        a = name('a')
        tl = TokenList([comment, ws(), a])
        assert tl.token_first(skip_cm=True) is a
        assert tl.token_first(skip_cm=False) is comment

    def test_token_first_empty(self):
        assert TokenList([]).token_first() is None

    def test_token_next_basic(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, ws(), b])
        idx, tok = tl.token_next(0)
        assert idx == 2
        assert tok is b

    def test_token_next_skip_ws_false(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, ws(), b])
        idx, tok = tl.token_next(0, skip_ws=False)
        assert idx == 1
        assert tok is tl[1]

    def test_token_next_none_end(self):
        tl = TokenList([name('a')])
        assert tl.token_next(0) == (None, None)

    def test_token_next_idx_none(self):
        assert TokenList([name('a')]).token_next(None) == (None, None)

    def test_token_prev(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, ws(), b])
        idx, tok = tl.token_prev(2)
        assert idx == 0
        assert tok is a

    def test_token_next_by(self):
        a = name('a')
        b = kw('AS')
        c = name('c')
        tl = TokenList([a, ws(), b, ws(), c])
        idx, tok = tl.token_next_by(m=(T.Keyword, 'AS'))
        assert idx == 2
        assert tok is b

    def test_token_next_by_no_match(self):
        tl = TokenList([name('a')])
        assert tl.token_next_by(m=(T.Keyword, 'AS')) == (None, None)

    def test_token_next_by_instance(self):
        inner = Parenthesis([Token(T.Punctuation, '('),
                             Token(T.Punctuation, ')')])
        tl = TokenList([name('a'), inner])
        idx, tok = tl.token_next_by(i=Parenthesis)
        assert idx == 1
        assert tok is inner

    def test_token_matching(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, b])
        assert tl.token_matching(lambda tk: tk is b, 0) is b

    def test_token_not_matching(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, b])
        idx, tok = tl.token_not_matching(lambda tk: tk is a, 0)
        assert idx == 1
        assert tok is b

    def test_token_index(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, b])
        assert tl.token_index(b) == 1

    def test_token_index_from_token(self):
        a, b, c = name('a'), name('b'), name('c')
        tl = TokenList([a, b, c])
        assert tl.token_index(c, start=b) == 2

    def test_group_tokens_new(self):
        tl = TokenList([name('a'), name('b'), name('c')])
        grp = tl.group_tokens(Parenthesis, 0, 1)
        assert isinstance(grp, Parenthesis)
        assert len(tl.tokens) == 2
        assert tl.tokens[0] is grp
        assert grp.parent is tl
        assert [t.value for t in grp.tokens] == ['a', 'b']
        assert all(t.parent is grp for t in grp.tokens)

    def test_group_tokens_no_include_end(self):
        tl = TokenList([name('a'), name('b'), name('c')])
        grp = tl.group_tokens(Parenthesis, 0, 1, include_end=False)
        assert len(grp.tokens) == 1
        assert len(tl.tokens) == 3

    def test_group_tokens_extend(self):
        tl = TokenList([name('a'), name('b'), name('c')])
        grp = tl.group_tokens(Parenthesis, 0, 0)
        assert len(grp.tokens) == 1
        grp2 = tl.group_tokens(Parenthesis, 0, 1, extend=True)
        assert grp2 is grp
        assert len(grp.tokens) == 2
        assert grp.value == 'ab'

    def test_insert_before_with_int(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, b])
        new = name('x')
        tl.insert_before(1, new)
        assert tl.tokens[1] is new
        assert new.parent is tl

    def test_insert_before_with_token(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, b])
        new = name('x')
        tl.insert_before(b, new)
        assert tl.tokens[1] is new

    def test_insert_after_with_int(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, b])
        new = name('x')
        tl.insert_after(0, new)
        # next non-ws after index 0 is found, insert before it
        assert tl.tokens[1] is new

    def test_insert_after_at_end(self):
        a = name('a')
        tl = TokenList([a])
        new = name('x')
        tl.insert_after(0, new)
        assert tl.tokens[-1] is new

    def test_insert_after_with_token(self):
        a, b = name('a'), name('b')
        tl = TokenList([a, b])
        new = name('x')
        tl.insert_after(a, new)
        assert tl.tokens[1] is new

    def test_has_alias_default_false(self):
        assert TokenList([name('a')]).has_alias() is False

    def test_get_alias_default_none(self):
        assert TokenList([name('a')]).get_alias() is None

    def test_get_real_name_default_none(self):
        assert TokenList([name('a')]).get_real_name() is None

    def test_get_name_falls_back(self):
        # TokenList has no alias, no real name -> None
        assert TokenList([name('a')]).get_name() is None

    def test_get_parent_name(self):
        tl = TokenList([name('a'), punct('.'), name('b')])
        assert tl.get_parent_name() == 'a'

    def test_get_parent_name_no_dot(self):
        tl = TokenList([name('a')])
        assert tl.get_parent_name() is None

    def test_get_parent_name_quoted(self):
        tl = TokenList([name('"a"'), punct('.'), name('b')])
        assert tl.get_parent_name() == 'a'


# ---------------------------------------------------------------------------
# NameAliasMixin / Identifier
# ---------------------------------------------------------------------------

class TestNameAliasMixin:
    def test_direct_mixin_usage(self):
        obj = NameAliasMixin()
        # no tokens attribute -> error
        with pytest.raises(AttributeError):
            obj.get_real_name()

    def test_get_real_name_simple(self):
        ident = Identifier([name('tbl')])
        assert ident.get_real_name() == 'tbl'

    def test_get_real_name_after_last_dot(self):
        ident = Identifier([name('a'), punct('.'), name('b'),
                            punct('.'), name('c')])
        assert ident.get_real_name() == 'c'

    def test_get_real_name_with_quotes(self):
        ident = Identifier([name('"tbl"')])
        assert ident.get_real_name() == 'tbl'

    def test_get_alias_as(self):
        ident = Identifier([name('col'), ws(), kw('AS'), ws(), name('alias')])
        assert ident.get_alias() == 'alias'

    def test_get_alias_no_as(self):
        ident = Identifier([name('col'), ws(), name('alias')])
        assert ident.get_alias() == 'alias'

    def test_get_alias_none(self):
        ident = Identifier([name('col')])
        assert ident.get_alias() is None

    def test_get_alias_only_two_tokens(self):
        ident = Identifier([name('col'), ws()])
        assert ident.get_alias() is None

    def test_get_alias_with_quotes(self):
        ident = Identifier([name('col'), ws(), name('"alias"')])
        assert ident.get_alias() == 'alias'

    def test_get_alias_keyword(self):
        ident = Identifier([name('col'), ws(), kw('AS'), ws(), kw('select')])
        # The raw token value is returned (not the normalized form).
        assert ident.get_alias() == 'select'


class TestIdentifier:
    def test_is_wildcard_true(self):
        ident = Identifier([name('t'), punct('.'), Token(T.Wildcard, '*')])
        assert ident.is_wildcard() is True

    def test_is_wildcard_false(self):
        assert Identifier([name('t')]).is_wildcard() is False

    def test_get_typecast(self):
        ident = Identifier([name('a'), punct('::'), Token(T.Name.Builtin, 'int')])
        assert ident.get_typecast() == 'int'

    def test_get_typecast_none(self):
        assert Identifier([name('a')]).get_typecast() is None

    def test_get_ordering(self):
        ident = Identifier([name('a'), ws(), Token(T.Keyword.Order, 'desc')])
        assert ident.get_ordering() == 'DESC'

    def test_get_ordering_none(self):
        assert Identifier([name('a')]).get_ordering() is None

    def test_get_array_indices(self):
        brackets = SquareBrackets([
            Token(T.Punctuation, '['),
            Token(T.Literal.Number.Integer, '1'),
            Token(T.Punctuation, ']'),
        ])
        ident = Identifier([name('a'), brackets])
        indices = list(ident.get_array_indices())
        assert len(indices) == 1
        # get_array_indices builds a fresh list slice each call.
        assert indices[0] == brackets.tokens[1:-1]
        assert [t.value for t in indices[0]] == ['1']

    def test_get_array_indices_none(self):
        assert list(Identifier([name('a')]).get_array_indices()) == []

    def test_get_name_uses_alias(self):
        ident = Identifier([name('col'), ws(), name('alias')])
        assert ident.get_name() == 'alias'

    def test_get_name_uses_real_name(self):
        ident = Identifier([name('col')])
        assert ident.get_name() == 'col'

    def test_has_alias_true(self):
        ident = Identifier([name('col'), ws(), name('alias')])
        assert ident.has_alias() is True


# ---------------------------------------------------------------------------
# IdentifierList
# ---------------------------------------------------------------------------

class TestIdentifierList:
    def test_get_identifiers_skips_ws_and_commas(self):
        a, b = name('a'), name('b')
        il = IdentifierList([a, punct(','), ws(), b])
        assert list(il.get_identifiers()) == [a, b]

    def test_get_identifiers_empty(self):
        assert list(IdentifierList([]).get_identifiers()) == []


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

class TestStatement:
    def test_get_type_dml(self):
        stmt = Statement([Token(T.Keyword.DML, 'select'), ws(), name('x')])
        assert stmt.get_type() == 'SELECT'

    def test_get_type_ddl(self):
        stmt = Statement([Token(T.Keyword.DDL, 'create')])
        assert stmt.get_type() == 'CREATE'

    def test_get_type_unknown_empty(self):
        assert Statement([]).get_type() == 'UNKNOWN'

    def test_get_type_unknown_only_ws(self):
        assert Statement([ws()]).get_type() == 'UNKNOWN'

    def test_get_type_unknown_non_dml(self):
        stmt = Statement([name('foo')])
        assert stmt.get_type() == 'UNKNOWN'

    def test_get_type_unknown_generic_keyword(self):
        stmt = Statement([kw('select')])
        assert stmt.get_type() == 'UNKNOWN'

    def test_get_type_skips_comment(self):
        comment = Comment([Token(T.Comment.Single, '--x')])
        stmt = Statement([comment, ws(), Token(T.Keyword.DML, 'select')])
        assert stmt.get_type() == 'SELECT'

    def test_get_type_cte(self):
        ident = Identifier([name('cte')])
        stmt = Statement([
            Token(T.Keyword.CTE, 'WITH'), ws(), ident, ws(),
            Token(T.Keyword.DML, 'select'),
        ])
        assert stmt.get_type() == 'SELECT'

    def test_get_type_cte_identifier_list(self):
        il = IdentifierList([name('a')])
        stmt = Statement([
            Token(T.Keyword.CTE, 'WITH'), ws(), il, ws(),
            Token(T.Keyword.DML, 'insert'),
        ])
        assert stmt.get_type() == 'INSERT'

    def test_get_type_cte_no_dml(self):
        ident = Identifier([name('cte')])
        stmt = Statement([Token(T.Keyword.CTE, 'WITH'), ws(), ident, ws(),
                          name('foo')])
        assert stmt.get_type() == 'UNKNOWN'


# ---------------------------------------------------------------------------
# Simple subclasses
# ---------------------------------------------------------------------------

class TestTypedLiteral:
    def test_constants(self):
        assert TypedLiteral.M_OPEN == [(T.Name.Builtin, None),
                                       (T.Keyword, "TIMESTAMP")]
        assert TypedLiteral.M_CLOSE == (T.String.Single, None)
        assert TypedLiteral.M_EXTEND == (
            T.Keyword, ("DAY", "HOUR", "MINUTE", "MONTH", "SECOND", "YEAR"))


class TestParenthesis:
    def test_groupable_tokens(self):
        p = Parenthesis([Token(T.Punctuation, '('), name('a'),
                         Token(T.Punctuation, ')')])
        assert p._groupable_tokens == [p.tokens[1]]

    def test_constants(self):
        assert Parenthesis.M_OPEN == (T.Punctuation, '(')
        assert Parenthesis.M_CLOSE == (T.Punctuation, ')')


class TestSquareBrackets:
    def test_groupable_tokens(self):
        b = SquareBrackets([Token(T.Punctuation, '['), name('a'),
                            Token(T.Punctuation, ']')])
        assert b._groupable_tokens == [b.tokens[1]]


class TestAssignment:
    def test_is_tokenlist(self):
        assert isinstance(Assignment([]), TokenList)


class TestIf:
    def test_constants(self):
        assert If.M_OPEN == (T.Keyword, 'IF')
        assert If.M_CLOSE == (T.Keyword, 'END IF')


class TestFor:
    def test_constants(self):
        assert For.M_OPEN == (T.Keyword, ('FOR', 'FOREACH'))
        assert For.M_CLOSE == (T.Keyword, 'END LOOP')


class TestComparison:
    def test_left_right(self):
        a, b = name('a'), name('b')
        cmp_ = Comparison([a, Token(T.Operator, '='), b])
        assert cmp_.left is a
        assert cmp_.right is b


class TestComment:
    def test_is_multiline_true(self):
        c = Comment([Token(T.Comment.Multiline, '/* x */')])
        assert c.is_multiline() is True

    def test_is_multiline_false(self):
        c = Comment([Token(T.Comment.Single, '-- x')])
        assert c.is_multiline() is False

    def test_is_multiline_empty(self):
        # Empty comment: `self.tokens and ...` returns the empty list.
        assert Comment([]).is_multiline() == []


class TestWhere:
    def test_constants(self):
        assert Where.M_OPEN == (T.Keyword, 'WHERE')
        assert 'ORDER BY' in Where.M_CLOSE[1]


class TestOver:
    def test_constants(self):
        assert Over.M_OPEN == (T.Keyword, 'OVER')


class TestHaving:
    def test_constants(self):
        assert Having.M_OPEN == (T.Keyword, 'HAVING')
        assert Having.M_CLOSE == (T.Keyword, ('ORDER BY', 'LIMIT'))


class TestBegin:
    def test_constants(self):
        assert Begin.M_OPEN == (T.Keyword, 'BEGIN')
        assert Begin.M_CLOSE == (T.Keyword, 'END')


class TestOperationValuesCommand:
    def test_subclasses(self):
        for cls in (Operation, Values, Command):
            assert issubclass(cls, TokenList)


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

class TestCase:
    def _build(self):
        return Case([
            kw('CASE'),
            ws(),
            kw('WHEN'),
            ws(),
            name('a'),
            ws(),
            kw('THEN'),
            ws(),
            name('x'),
            ws(),
            kw('ELSE'),
            ws(),
            name('y'),
            ws(),
            kw('END'),
        ])

    def test_get_cases_basic(self):
        cases = self._build().get_cases()
        # Because CASE keyword is skipped but the leading whitespace token
        # triggers a fresh "condition" entry before WHEN is processed, there
        # are 3 result tuples.
        assert len(cases) == 3
        # First entry: leading whitespace only, in the condition slot.
        assert all(t.is_whitespace for t in cases[0][0])
        assert cases[0][1] == []

        # Second entry: real WHEN condition and THEN value.
        # Note: the THEN keyword itself is appended to the value slot
        # (mode is switched to VALUE before appending), so the value text
        # contains "THEN" followed by the actual value and a trailing
        # whitespace token (value slot also gets the whitespace after 'x').
        cond, val = cases[1]
        text = ''.join(t.value for t in cond)
        assert 'a' in text
        val_text = ''.join(t.value for t in val)
        assert val_text == 'THEN x '

        # Third entry: ELSE -> condition is None.
        cond2, val2 = cases[2]
        assert cond2 is None
        # ELSE keyword also ends up in the value slot, followed by
        # whitespace and 'y' and trailing whitespace.
        val2_text = ''.join(t.value for t in val2)
        assert val2_text == 'ELSE y '

    def test_get_cases_skip_ws(self):
        cases = self._build().get_cases(skip_ws=True)
        # With skip_ws, whitespace tokens are skipped, so we avoid the
        # spurious leading entry -> exactly 2 tuples.
        assert len(cases) == 2
        cond, val = cases[0]
        assert all(not t.is_whitespace for t in cond)
        assert all(not t.is_whitespace for t in val)

    def test_get_cases_empty(self):
        assert Case([]).get_cases() == []

    def test_get_cases_only_case(self):
        assert Case([kw('CASE')]).get_cases() == []


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

class TestFunction:
    def _make_func(self, params):
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            *params,
            Token(T.Punctuation, ')'),
        ])
        return Function([name('f'), paren])

    def test_get_parameters_simple(self):
        # Without an IdentifierList, only direct Identifier/Function/
        # TypedLiteral/Literal children of the parenthesis are picked up.
        # A bare Name token (not wrapped in Identifier) is ignored.
        p1 = Token(T.Literal, '1')
        p2 = name('x')
        f = self._make_func([p1, Token(T.Punctuation, ','), ws(), p2])
        params = f.get_parameters()
        assert params == [p1]

    def test_get_parameters_identifier(self):
        ident = Identifier([name('x')])
        f = self._make_func([ident])
        assert f.get_parameters() == [ident]

    def test_get_parameters_identifier_list(self):
        il = IdentifierList([name('a'), punct(','), name('b')])
        f = self._make_func([il])
        params = f.get_parameters()
        assert list(params) == list(il.get_identifiers())

    def test_get_parameters_empty(self):
        f = self._make_func([])
        assert f.get_parameters() == []

    def test_get_window_none(self):
        # Implementation does not guard against a missing OVER clause:
        # token_next_by returns (None, None), which is truthy as a tuple,
        # so it indexes over_clause[1] (None) and raises AttributeError.
        f = self._make_func([])
        with pytest.raises(AttributeError):
            f.get_window()

    def test_get_window(self):
        window_tok = name('w')
        over = Over([kw('OVER'), ws(), window_tok])
        paren = Parenthesis([Token(T.Punctuation, '('),
                             Token(T.Punctuation, ')')])
        f = Function([name('f'), paren, ws(), over])
        assert f.get_window() is window_tok

    def test_get_real_name(self):
        paren = Parenthesis([Token(T.Punctuation, '('),
                             Token(T.Punctuation, ')')])
        f = Function([name('f'), paren])
        assert f.get_real_name() == 'f'

    def test_get_alias(self):
        paren = Parenthesis([Token(T.Punctuation, '('),
                             Token(T.Punctuation, ')')])
        f = Function([name('f'), paren, ws(), name('alias')])
        assert f.get_alias() == 'alias'
