import re
import pytest

from sqlparse import tokens as T
from sqlparse.sql import (
    NameAliasMixin, Token, TokenList, Statement, Identifier, IdentifierList,
    TypedLiteral, Parenthesis, SquareBrackets, Assignment, If, For,
    Comparison, Comment, Where, Over, Having, Case, Function, Begin,
    Operation, Values, Command,
)


# ----------------------------- helpers ----------------------------------
def tok(ttype, value):
    return Token(ttype, value)


# ----------------------------- Token ------------------------------------
class TestToken:
    def test_basic_init(self):
        t = Token(T.Keyword, 'select')
        assert t.value == 'select'
        assert t.ttype == T.Keyword
        assert t.parent is None
        assert t.is_group is False
        assert t.is_keyword is True
        assert t.is_whitespace is False
        assert t.is_newline is False
        assert t.normalized == 'SELECT'

    def test_value_coerced_to_str(self):
        t = Token(T.Name, 123)
        assert t.value == '123'
        assert isinstance(t.value, str)

    def test_normalized_non_keyword(self):
        t = Token(T.Name, 'Foo')
        assert t.normalized == 'Foo'

    def test_whitespace_flags(self):
        t = Token(T.Whitespace, ' ')
        assert t.is_whitespace is True
        t2 = Token(T.Whitespace.Newline, '\n')
        assert t2.is_newline is True
        assert t2.is_whitespace is True

    def test_str_and_repr(self):
        t = Token(T.Name, 'foo')
        assert str(t) == 'foo'
        r = repr(t)
        assert 'Name' in r
        assert 'foo' in r

    def test_repr_truncates_long_value(self):
        t = Token(T.Name, 'abcdefghijk')
        r = repr(t)
        assert 'abcdef...' in r

    def test_repr_collapses_whitespace(self):
        t = Token(T.Name, 'a   b')
        assert "'a b'" in repr(t)

    def test_repr_name_dotted(self):
        t = Token(T.Keyword, 'x')
        assert t._get_repr_name() == 'Keyword'

    def test_flatten_yields_self(self):
        t = Token(T.Name, 'x')
        assert list(t.flatten()) == [t]

    def test_match_ttype_only(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, None) is True
        assert t.match(T.Keyword, None) is False

    def test_match_type_mismatch_returns_false(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Keyword, 'foo') is False

    def test_match_values_string(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, 'foo') is True
        assert t.match(T.Name, 'bar') is False

    def test_match_values_tuple(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, ('foo', 'bar')) is True
        assert t.match(T.Name, ('baz', 'bar')) is False

    def test_match_keyword_case_insensitive(self):
        t = Token(T.Keyword, 'select')
        assert t.match(T.Keyword, 'SELECT') is True
        assert t.match(T.Keyword, 'select') is True

    def test_match_regex(self):
        t = Token(T.Name, 'foobar')
        assert t.match(T.Name, 'foo', regex=True) is True
        assert t.match(T.Name, 'bar$', regex=True) is True
        assert t.match(T.Name, '^baz', regex=True) is False

    def test_match_regex_keyword_ignorecase(self):
        t = Token(T.Keyword, 'SELECT')
        assert t.match(T.Keyword, 'sel', regex=True) is True

    def test_identity_type_match_not_subclass(self):
        t = Token(T.Name, 'x')
        # is comparison with tuple type; T.Name is a tuple of types
        assert t.match(T.Name, 'x', ) is True

    def test_within_no_parent(self):
        t = Token(T.Name, 'x')
        assert t.within(TokenList) is False

    def test_is_child_of(self):
        parent = TokenList([])
        t = Token(T.Name, 'x')
        assert t.is_child_of(parent) is False
        t.parent = parent
        assert t.is_child_of(parent) is True

    def test_has_ancestor(self):
        gp = TokenList([])
        parent = TokenList([])
        t = Token(T.Name, 'x')
        parent.parent = gp
        t.parent = parent
        assert t.has_ancestor(gp) is True
        assert t.has_ancestor(parent) is True
        assert t.has_ancestor(TokenList([])) is False

    def test_within(self):
        gp = TokenList([])
        t = Token(T.Name, 'x')
        gp.tokens = [t]
        t.parent = gp
        assert t.within(TokenList) is True


# ----------------------------- TokenList --------------------------------
class TestTokenList:
    def test_init_sets_parents(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.is_group is True
        assert a.parent is tl
        assert b.parent is tl
        assert tl.value == 'ab'

    def test_iter_and_getitem(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert list(tl) == [a]
        assert tl[0] is a

    def test_str_flattens(self):
        inner = TokenList([Token(T.Name, 'x')])
        tl = TokenList([inner, Token(T.Name, 'y')])
        assert str(tl) == 'xy'

    def test_repr_name(self):
        tl = TokenList([])
        assert tl._get_repr_name() == 'TokenList'

    def test_flatten(self):
        a = Token(T.Name, 'a')
        inner = TokenList([Token(T.Name, 'b')])
        tl = TokenList([a, inner, Token(T.Name, 'c')])
        assert list(tl.flatten()) == [a, inner.tokens[0], tl.tokens[2]]

    def test_get_sublists(self):
        a = Token(T.Name, 'a')
        inner = TokenList([Token(T.Name, 'b')])
        tl = TokenList([a, inner])
        assert list(tl.get_sublists()) == [inner]

    def test_groupable_tokens(self):
        tl = TokenList([])
        assert tl._groupable_tokens == tl.tokens

    def test_get_token_at_offset(self):
        a = Token(T.Name, 'ab')
        b = Token(T.Name, 'cd')
        tl = TokenList([a, b])
        assert tl.get_token_at_offset(0) is a
        assert tl.get_token_at_offset(1) is a
        assert tl.get_token_at_offset(2) is b
        assert tl.get_token_at_offset(3) is b
        assert tl.get_token_at_offset(4) is None

    def test_get_token_at_offset_nested(self):
        inner = TokenList([Token(T.Name, 'cd')])
        tl = TokenList([Token(T.Name, 'ab'), inner])
        assert tl.get_token_at_offset(2) is inner.tokens[0]

    def test_token_first(self):
        ws = tok(T.Whitespace, ' ')
        name = tok(T.Name, 'x')
        tl = TokenList([ws, name])
        assert tl.token_first() is name
        assert tl.token_first(skip_ws=False) is ws

    def test_token_first_empty(self):
        tl = TokenList([])
        assert tl.token_first() is None

    def test_token_first_skip_cm(self):
        cm = Comment([Token(T.Comment.Single, '-- hi')])
        name = tok(T.Name, 'x')
        tl = TokenList([cm, name])
        assert tl.token_first(skip_cm=True) is name
        assert tl.token_first(skip_cm=False) is cm

    def test_token_next_by(self):
        a = tok(T.Name, 'a')
        b = tok(T.Keyword, 'FROM')
        tl = TokenList([a, b])
        idx, t = tl.token_next_by(t=T.Keyword)
        assert idx == 1
        assert t is b

    def test_token_next_by_not_found(self):
        tl = TokenList([tok(T.Name, 'a')])
        idx, t = tl.token_next_by(t=T.Keyword)
        assert idx is None
        assert t is None

    def test_token_next_by_idx_and_end(self):
        a = tok(T.Name, 'a')
        b = tok(T.Keyword, 'x')
        c = tok(T.Keyword, 'y')
        tl = TokenList([a, b, c])
        idx, t = tl.token_next_by(t=T.Keyword, end=2)
        assert t is b
        idx, t = tl.token_next_by(t=T.Keyword, idx=1)
        assert t is c

    def test_token_not_matching(self):
        a = tok(T.Name, 'a')
        b = tok(T.Keyword, 'x')
        tl = TokenList([a, b])
        idx, t = tl.token_not_matching(lambda tk: tk.ttype is T.Name, -1)
        assert t is b

    def test_token_matching(self):
        a = tok(T.Name, 'a')
        b = tok(T.Keyword, 'x')
        tl = TokenList([a, b])
        assert tl.token_matching(lambda tk: tk.ttype is T.Keyword, 0) is b

    def test_token_prev(self):
        a = tok(T.Name, 'a')
        ws = tok(T.Whitespace, ' ')
        b = tok(T.Name, 'b')
        tl = TokenList([a, ws, b])
        assert tl.token_prev(2) is a
        assert tl.token_prev(2, skip_ws=False) is ws

    def test_token_prev_none(self):
        tl = TokenList([tok(T.Name, 'a')])
        assert tl.token_prev(0) == (None, None)

    def test_token_next(self):
        a = tok(T.Name, 'a')
        ws = tok(T.Whitespace, ' ')
        b = tok(T.Name, 'b')
        tl = TokenList([a, ws, b])
        assert tl.token_next(0) == (2, b)
        assert tl.token_next(0, skip_ws=False) == (1, ws)

    def test_token_next_idx_none(self):
        tl = TokenList([])
        assert tl.token_next(None) == (None, None)

    def test_token_next_reverse(self):
        a = tok(T.Name, 'a')
        b = tok(T.Name, 'b')
        c = tok(T.Name, 'c')
        tl = TokenList([a, b, c])
        idx, t = tl.token_next(2, _reverse=True)
        # reverse starts at start+1-2 = 1
        assert t is a

    def test_token_index(self):
        a = tok(T.Name, 'a')
        b = tok(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_index(b) == 1

    def test_token_index_with_start_token(self):
        a = tok(T.Name, 'a')
        b = tok(T.Name, 'b')
        c = tok(T.Name, 'c')
        tl = TokenList([a, b, c])
        assert tl.token_index(c, start=b) == 2

    def test_token_index_not_found(self):
        tl = TokenList([tok(T.Name, 'a')])
        with pytest.raises(ValueError):
            tl.token_index(tok(T.Name, 'z'))

    def test_group_tokens(self):
        a = tok(T.Name, 'a')
        b = tok(T.Name, 'b')
        c = tok(T.Name, 'c')
        tl = TokenList([a, b, c])
        g = tl.group_tokens(Parenthesis, 0, 1)
        assert isinstance(g, Parenthesis)
        assert tl.tokens[0] is g
        assert g.tokens == [a, b]
        assert a.parent is g
        assert b.parent is g
        assert tl.tokens[1] is c

    def test_group_tokens_no_include_end(self):
        a = tok(T.Name, 'a')
        b = tok(T.Name, 'b')
        c = tok(T.Name, 'c')
        tl = TokenList([a, b, c])
        g = tl.group_tokens(Parenthesis, 0, 1, include_end=False)
        assert g.tokens == [a]
        assert tl.tokens[1] is b

    def test_group_tokens_extend(self):
        a = Token(T.Keyword, 'CASE')
        g = Case([a])
        b = tok(T.Name, 'b')
        c = tok(T.Name, 'c')
        tl = TokenList([g, b, c])
        result = tl.group_tokens(Case, 0, 1, extend=True)
        assert result is g
        assert g.tokens == [a, b]
        assert g.value == 'CASEb'
        assert len(tl.tokens) == 2

    def test_insert_before(self):
        a = tok(T.Name, 'a')
        b = tok(T.Name, 'b')
        tl = TokenList([a, b])
        new = tok(T.Name, 'n')
        tl.insert_before(b, new)
        assert tl.tokens == [a, new, b]
        assert new.parent is tl

    def test_insert_before_int(self):
        a = tok(T.Name, 'a')
        b = tok(T.Name, 'b')
        tl = TokenList([a, b])
        new = tok(T.Name, 'n')
        tl.insert_before(0, new)
        assert tl.tokens[0] is new

    def test_insert_after(self):
        a = tok(T.Name, 'a')
        b = tok(T.Name, 'b')
        tl = TokenList([a, b])
        new = tok(T.Name, 'n')
        tl.insert_after(a, new)
        assert tl.tokens == [a, new, b]
        assert new.parent is tl

    def test_insert_after_end_appends(self):
        a = tok(T.Name, 'a')
        tl = TokenList([a])
        new = tok(T.Name, 'n')
        tl.insert_after(a, new)
        assert tl.tokens == [a, new]

    def test_has_alias_default_false(self):
        tl = TokenList([])
        assert tl.has_alias() is False

    def test_get_alias_none(self):
        tl = TokenList([])
        assert tl.get_alias() is None

    def test_get_name(self):
        tl = TokenList([])
        assert tl.get_name() is None

    def test_get_real_name_none(self):
        tl = TokenList([])
        assert tl.get_real_name() is None

    def test_get_parent_name(self):
        a = tok(T.Name, 'a')
        dot = tok(T.Punctuation, '.')
        b = tok(T.Name, 'b')
        tl = TokenList([a, dot, b])
        assert tl.get_parent_name() == 'a'

    def test_get_parent_name_none(self):
        tl = TokenList([tok(T.Name, 'a')])
        assert tl.get_parent_name() is None

    def test_get_parent_name_quotes(self):
        a = tok(T.Name, '"a"')
        dot = tok(T.Punctuation, '.')
        b = tok(T.Name, 'b')
        tl = TokenList([a, dot, b])
        assert tl.get_parent_name() == 'a'

    def test_get_first_name(self):
        tl = TokenList([tok(T.Name, 'a'), tok(T.Name, 'b')])
        assert tl._get_first_name() == 'a'

    def test_get_first_name_idx(self):
        tl = TokenList([tok(T.Name, 'a'), tok(T.Name, 'b')])
        assert tl._get_first_name(1) == 'b'

    def test_get_first_name_reverse(self):
        tl = TokenList([tok(T.Name, 'a'), tok(T.Name, 'b')])
        assert tl._get_first_name(reverse=True) == 'b'

    def test_get_first_name_wildcard(self):
        tl = TokenList([tok(T.Wildcard, '*')])
        assert tl._get_first_name() == '*'

    def test_get_first_name_keywords(self):
        tl = TokenList([tok(T.Keyword, 'x')])
        assert tl._get_first_name(keywords=True) == 'x'
        assert tl._get_first_name(keywords=False) is None

    def test_get_first_name_string_symbol(self):
        tl = TokenList([tok(T.String.Symbol, '"foo"')])
        assert tl._get_first_name() == 'foo'

    def test_get_first_name_identifier(self):
        ident = Identifier([tok(T.Name, 'a')])
        tl = TokenList([ident])
        assert tl._get_first_name() == 'a'

    def test_pprint_tree(self, capsys):
        a = tok(T.Name, 'a')
        tl = TokenList([a])
        tl._pprint_tree()
        out = capsys.readouterr().out
        assert 'Name' in out
        assert 'a' in out


# ----------------------------- Statement --------------------------------
class TestStatement:
    def test_get_type_dml(self):
        s = Statement([tok(T.Keyword.DML, 'select'), tok(T.Name, 'x')])
        assert s.get_type() == 'SELECT'

    def test_get_type_ddl(self):
        s = Statement([tok(T.Keyword.DDL, 'create')])
        assert s.get_type() == 'CREATE'

    def test_get_type_empty(self):
        s = Statement([])
        assert s.get_type() == 'UNKNOWN'

    def test_get_type_ws_only(self):
        s = Statement([tok(T.Whitespace, ' ')])
        assert s.get_type() == 'UNKNOWN'

    def test_get_type_skip_comment(self):
        cm = Comment([Token(T.Comment.Single, '-- c')])
        s = Statement([cm, tok(T.Keyword.DML, 'insert')])
        assert s.get_type() == 'INSERT'

    def test_get_type_cte(self):
        with_kw = tok(T.Keyword.CTE, 'WITH')
        ident = Identifier([tok(T.Name, 'cte')])
        select = tok(T.Keyword.DML, 'select')
        s = Statement([with_kw, tok(T.Whitespace, ' '), ident,
                       tok(T.Whitespace, ' '), select])
        assert s.get_type() == 'SELECT'

    def test_get_type_unknown(self):
        s = Statement([tok(T.Name, 'foo')])
        assert s.get_type() == 'UNKNOWN'


# ----------------------------- Identifier -------------------------------
class TestIdentifier:
    def test_get_real_name_simple(self):
        i = Identifier([tok(T.Name, 'foo')])
        assert i.get_real_name() == 'foo'

    def test_get_real_name_dotted(self):
        i = Identifier([
            tok(T.Name, 'a'), tok(T.Punctuation, '.'),
            tok(T.Name, 'b'), tok(T.Punctuation, '.'),
            tok(T.Name, 'c'),
        ])
        assert i.get_real_name() == 'c'

    def test_get_real_name_quoted(self):
        i = Identifier([tok(T.Name, '"foo"')])
        assert i.get_real_name() == 'foo'

    def test_get_alias_as(self):
        i = Identifier([
            tok(T.Name, 'foo'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'AS'),
            tok(T.Whitespace, ' '),
            tok(T.Name, 'bar'),
        ])
        assert i.get_alias() == 'bar'
        assert i.has_alias() is True
        assert i.get_name() == 'bar'

    def test_get_alias_implicit(self):
        i = Identifier([
            tok(T.Name, 'foo'),
            tok(T.Whitespace, ' '),
            tok(T.Name, 'bar'),
        ])
        assert i.get_alias() == 'bar'

    def test_get_alias_none_single(self):
        i = Identifier([tok(T.Name, 'foo')])
        assert i.get_alias() is None
        assert i.has_alias() is False
        assert i.get_name() == 'foo'

    def test_is_wildcard(self):
        i = Identifier([tok(T.Wildcard, '*')])
        assert i.is_wildcard() is True
        i2 = Identifier([tok(T.Name, 'x')])
        assert i2.is_wildcard() is False

    def test_get_typecast(self):
        i = Identifier([
            tok(T.Name, 'foo'),
            tok(T.Punctuation, '::'),
            tok(T.Name.Builtin, 'int'),
        ])
        assert i.get_typecast() == 'int'

    def test_get_typecast_none(self):
        i = Identifier([tok(T.Name, 'foo')])
        assert i.get_typecast() is None

    def test_get_ordering(self):
        i = Identifier([
            tok(T.Name, 'x'), tok(T.Whitespace, ' '),
            tok(T.Keyword.Order, 'desc'),
        ])
        assert i.get_ordering() == 'DESC'

    def test_get_ordering_none(self):
        i = Identifier([tok(T.Name, 'x')])
        assert i.get_ordering() is None

    def test_get_array_indices(self):
        sb = SquareBrackets([
            tok(T.Punctuation, '['),
            tok(T.Number.Integer, '1'),
            tok(T.Punctuation, ']'),
        ])
        i = Identifier([tok(T.Name, 'arr'), sb])
        indices = list(i.get_array_indices())
        assert len(indices) == 1
        assert indices[0] == [sb.tokens[1]]


# ----------------------------- IdentifierList ---------------------------
class TestIdentifierList:
    def test_get_identifiers(self):
        a = Identifier([tok(T.Name, 'a')])
        b = Identifier([tok(T.Name, 'b')])
        il = IdentifierList([
            a, tok(T.Punctuation, ','), tok(T.Whitespace, ' '), b,
        ])
        assert list(il.get_identifiers()) == [a, b]

    def test_get_identifiers_empty(self):
        il = IdentifierList([])
        assert list(il.get_identifiers()) == []


# ----------------------------- TypedLiteral -----------------------------
class TestTypedLiteral:
    def test_class_attributes(self):
        assert TypedLiteral.M_OPEN == [(T.Name.Builtin, None),
                                       (T.Keyword, "TIMESTAMP")]
        assert TypedLiteral.M_CLOSE == T.String.Single
        # M_CLOSE in source is a tuple T.String.Single, None
        assert TypedLiteral.M_CLOSE[0] == T.String.Single
        assert TypedLiteral.M_CLOSE[1] is None

    def test_m_extend(self):
        assert TypedLiteral.M_EXTEND[0] == T.Keyword
        assert "DAY" in TypedLiteral.M_EXTEND[1]


# ----------------------------- Parenthesis / SquareBrackets ------------
class TestParenthesis:
    def test_groupable_tokens(self):
        p = Parenthesis([
            tok(T.Punctuation, '('),
            tok(T.Name, 'x'),
            tok(T.Punctuation, ')'),
        ])
        assert p._groupable_tokens == [p.tokens[1]]

    def test_markers(self):
        assert Parenthesis.M_OPEN == (T.Punctuation, '(')
        assert Parenthesis.M_CLOSE == (T.Punctuation, ')')


class TestSquareBrackets:
    def test_groupable_tokens(self):
        s = SquareBrackets([
            tok(T.Punctuation, '['),
            tok(T.Name, 'x'),
            tok(T.Punctuation, ']'),
        ])
        assert s._groupable_tokens == [s.tokens[1]]

    def test_markers(self):
        assert SquareBrackets.M_OPEN == (T.Punctuation, '[')
        assert SquareBrackets.M_CLOSE == (T.Punctuation, ']')


# ----------------------------- Simple subclasses -----------------------
class TestSimpleSubclasses:
    def test_assignment_is_tokenlist(self):
        assert issubclass(Assignment, TokenList)

    def test_if_markers(self):
        assert If.M_OPEN == (T.Keyword, 'IF')
        assert If.M_CLOSE == (T.Keyword, 'END IF')

    def test_for_markers(self):
        assert For.M_OPEN == (T.Keyword, ('FOR', 'FOREACH'))
        assert For.M_CLOSE == (T.Keyword, 'END LOOP')

    def test_where_markers(self):
        assert Where.M_OPEN == (T.Keyword, 'WHERE')
        assert 'LIMIT' in Where.M_CLOSE[1]

    def test_over_markers(self):
        assert Over.M_OPEN == (T.Keyword, 'OVER')

    def test_having_markers(self):
        assert Having.M_OPEN == (T.Keyword, 'HAVING')

    def test_begin_markers(self):
        assert Begin.M_OPEN == (T.Keyword, 'BEGIN')
        assert Begin.M_CLOSE == (T.Keyword, 'END')

    def test_operation_values_command(self):
        for cls in (Operation, Values, Command):
            assert issubclass(cls, TokenList)


# ----------------------------- Comparison -------------------------------
class TestComparison:
    def test_left_right(self):
        a = tok(T.Name, 'a')
        b = tok(T.Name, 'b')
        c = Comparison([a, tok(T.Operator, '='), b])
        assert c.left is a
        assert c.right is b


# ----------------------------- Comment ----------------------------------
class TestComment:
    def test_is_multiline_true(self):
        c = Comment([tok(T.Comment.Multiline, '/* x */')])
        assert c.is_multiline() is True

    def test_is_multiline_false(self):
        c = Comment([tok(T.Comment.Single, '-- x')])
        assert c.is_multiline() is False

    def test_is_multiline_empty(self):
        c = Comment([])
        assert not c.is_multiline()


# ----------------------------- Case -------------------------------------
class TestCase:
    def _build_tokens(self):
        return [
            tok(T.Keyword, 'CASE'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'WHEN'),
            tok(T.Whitespace, ' '),
            tok(T.Name, 'cond1'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'THEN'),
            tok(T.Whitespace, ' '),
            tok(T.Name, 'val1'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'WHEN'),
            tok(T.Whitespace, ' '),
            tok(T.Name, 'cond2'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'THEN'),
            tok(T.Whitespace, ' '),
            tok(T.Name, 'val2'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'ELSE'),
            tok(T.Whitespace, ' '),
            tok(T.Name, 'val3'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'END'),
        ]

    def test_get_cases(self):
        c = Case(self._build_tokens())
        cases = c.get_cases()
        assert len(cases) == 3
        # first WHEN
        cond1, val1 = cases[0]
        assert [t.value for t in cond1] == [' ', 'cond1', ' ']
        assert [t.value for t in val1] == [' ', 'val1', ' ']
        # second WHEN
        cond2, val2 = cases[1]
        assert [t.value for t in cond2] == [' ', 'cond2', ' ']
        assert [t.value for t in val2] == [' ', 'val2', ' ']
        # ELSE
        cond3, val3 = cases[2]
        assert cond3 is None
        assert [t.value for t in val3] == [' ', 'val3', ' ']

    def test_get_cases_skip_ws(self):
        c = Case(self._build_tokens())
        cases = c.get_cases(skip_ws=True)
        assert [t.value for t in cases[0][0]] == ['cond1']
        assert [t.value for t in cases[0][1]] == ['val1']
        assert cases[2][0] is None
        assert [t.value for t in cases[2][1]] == ['val3']

    def test_get_cases_empty(self):
        c = Case([tok(T.Keyword, 'CASE'), tok(T.Keyword, 'END')])
        assert c.get_cases() == []


# ----------------------------- Function --------------------------------
class TestFunction:
    def test_get_parameters(self):
        p = Parenthesis([
            tok(T.Punctuation, '('),
            tok(T.Name, 'a'),
            tok(T.Punctuation, ',') if False else tok(T.Punctuation, ','),
            tok(T.Name, 'b'),
            tok(T.Punctuation, ')'),
        ])
        # simpler: no IdentifierList
        p = Parenthesis([
            tok(T.Punctuation, '('),
            tok(T.Literal.Number.Integer, '1'),
            tok(T.Punctuation, ')'),
        ])
        f = Function([tok(T.Name, 'fn'), p])
        params = f.get_parameters()
        assert len(params) == 1
        assert params[0].value == '1'

    def test_get_parameters_identifier_list(self):
        a = Identifier([tok(T.Name, 'a')])
        b = Identifier([tok(T.Name, 'b')])
        il = IdentifierList([
            a, tok(T.Punctuation, ','), tok(T.Whitespace, ' '), b,
        ])
        p = Parenthesis([
            tok(T.Punctuation, '('), il, tok(T.Punctuation, ')'),
        ])
        f = Function([tok(T.Name, 'fn'), p])
        params = f.get_parameters()
        assert list(params) == [a, b]

    def test_get_window_none(self):
        f = Function([tok(T.Name, 'fn')])
        assert f.get_window() is None

    def test_get_window(self):
        over = Over([
            tok(T.Keyword, 'OVER'),
            tok(T.Whitespace, ' '),
            tok(T.Name, 'w'),
        ])
        f = Function([tok(T.Name, 'fn'), tok(T.Whitespace, ' '), over])
        assert f.get_window() is over.tokens[-1]


# ----------------------------- NameAliasMixin --------------------------
class TestNameAliasMixin:
    def test_mixin_used_by_identifier(self):
        assert issubclass(Identifier, NameAliasMixin)
        assert issubclass(Function, NameAliasMixin)
