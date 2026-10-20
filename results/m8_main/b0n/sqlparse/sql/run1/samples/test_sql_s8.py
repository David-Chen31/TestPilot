import pytest

from sqlparse import tokens as T
from sqlparse.sql import (
    NameAliasMixin, Token, TokenList, Statement, Identifier, IdentifierList,
    TypedLiteral, Parenthesis, SquareBrackets, Assignment, If, For, Comparison,
    Comment, Where, Over, Having, Case, Function, Begin, Operation, Values,
    Command,
)


# ---------------- Helpers ----------------

def mk_tokens(*pairs):
    return [Token(t, v) for t, v in pairs]


def mk_token_list(*pairs):
    return TokenList(mk_tokens(*pairs))


# ---------------- Token ----------------

class TestToken:
    def test_basic_attributes(self):
        tok = Token(T.Name, 'foo')
        assert tok.value == 'foo'
        assert tok.ttype == T.Name
        assert tok.parent is None
        assert tok.is_group is False
        assert tok.is_keyword is False
        assert tok.is_whitespace is False
        assert tok.is_newline is False
        assert tok.normalized == 'foo'

    def test_keyword_normalized(self):
        tok = Token(T.Keyword, 'select')
        assert tok.is_keyword is True
        assert tok.normalized == 'SELECT'

    def test_whitespace_and_newline(self):
        ws = Token(T.Whitespace, ' ')
        assert ws.is_whitespace is True
        nl = Token(T.Newline, '\n')
        assert nl.is_newline is True

    def test_value_coerced_to_str(self):
        tok = Token(T.Name, 123)
        assert tok.value == '123'

    def test_str_and_repr(self):
        tok = Token(T.Name, 'foo')
        assert str(tok) == 'foo'
        r = repr(tok)
        assert r.startswith('<Name')
        assert "'foo'" in r

    def test_repr_quoted_value(self):
        tok = Token(T.String.Single, "'abc'")
        r = repr(tok)
        assert '"\'abc\'"' in r

    def test_repr_long_value_truncated(self):
        tok = Token(T.Name, 'abcdefghij')
        r = repr(tok)
        assert 'abcdef...' in r

    def test_repr_whitespace_collapsed(self):
        tok = Token(T.Name, 'a\n  b')
        r = repr(tok)
        assert 'a b' in r

    def test_flatten_yields_self(self):
        tok = Token(T.Name, 'x')
        assert list(tok.flatten()) == [tok]

    def test_match_type_only(self):
        tok = Token(T.Name, 'foo')
        assert tok.match(T.Name, None) is True
        assert tok.match(T.Name, 'foo') is True
        assert tok.match(T.Name, 'bar') is False
        assert tok.match(T.Name, ('foo', 'bar')) is True
        assert tok.match(T.Keyword, None) is False

    def test_match_keyword_case_insensitive(self):
        tok = Token(T.Keyword, 'select')
        assert tok.match(T.Keyword, 'SELECT') is True
        assert tok.match(T.Keyword, 'select') is True
        assert tok.match(T.Keyword, 'from') is False

    def test_match_regex(self):
        tok = Token(T.Name, 'foo123')
        assert tok.match(T.Name, r'\d+', regex=True) is True
        assert tok.match(T.Name, r'xyz', regex=True) is False

    def test_match_regex_keyword_ignorecase(self):
        tok = Token(T.Keyword, 'select')
        assert tok.match(T.Keyword, r'^sel', regex=True) is True

    def test_match_values_string_coerced(self):
        tok = Token(T.Name, 'abc')
        assert tok.match(T.Name, 'abc') is True

    def test_within(self):
        inner = Token(T.Name, 'x')
        group = TokenList([inner])
        outer = TokenList([group])
        assert inner.within(TokenList) is True
        assert inner.within(Parenthesis) is False
        assert inner.parent is group

    def test_is_child_of(self):
        inner = Token(T.Name, 'x')
        group = TokenList([inner])
        assert inner.is_child_of(group) is True
        assert group.is_child_of(inner) is False

    def test_has_ancestor(self):
        inner = Token(T.Name, 'x')
        group = TokenList([inner])
        assert inner.has_ancestor(group) is True
        assert group.has_ancestor(inner) is False


# ---------------- TokenList ----------------

class TestTokenList:
    def test_init_parents(self):
        t1 = Token(T.Name, 'a')
        t2 = Token(T.Name, 'b')
        tl = TokenList([t1, t2])
        assert t1.parent is tl
        assert t2.parent is tl
        assert tl.is_group is True
        assert tl.value == 'ab'
        assert tl.ttype is None

    def test_empty_default(self):
        tl = TokenList()
        assert tl.tokens == []
        assert tl.value == ''

    def test_iter_and_getitem(self):
        t1 = Token(T.Name, 'a')
        t2 = Token(T.Name, 'b')
        tl = TokenList([t1, t2])
        assert list(tl) == [t1, t2]
        assert tl[0] is t1
        assert tl[-1] is t2

    def test_str_flattens(self):
        inner = TokenList([Token(T.Name, 'x'), Token(T.Name, 'y')])
        outer = TokenList([inner, Token(T.Name, 'z')])
        assert str(outer) == 'xyz'

    def test_flatten(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        inner = TokenList([a, b])
        outer = TokenList([inner])
        assert list(outer.flatten()) == [a, b]

    def test_get_sublists(self):
        a = Token(T.Name, 'a')
        inner = TokenList([a])
        outertok = Token(T.Name, 'c')
        outer = TokenList([inner, outertok])
        assert list(outer.get_sublists()) == [inner]

    def test_repr_name(self):
        tl = TokenList([])
        assert tl._get_repr_name() == 'TokenList'

    def test_get_token_at_offset(self):
        a = Token(T.Name, 'aa')
        b = Token(T.Name, 'bb')
        tl = TokenList([a, b])
        assert tl.get_token_at_offset(0) is a
        assert tl.get_token_at_offset(1) is a
        assert tl.get_token_at_offset(2) is b
        assert tl.get_token_at_offset(3) is b
        assert tl.get_token_at_offset(4) is None

    def test_token_first(self):
        ws = Token(T.Whitespace, ' ')
        a = Token(T.Name, 'a')
        tl = TokenList([ws, a])
        assert tl.token_first() is a
        assert tl.token_first(skip_ws=False) is ws

    def test_token_first_skip_cm(self):
        cm = Comment([Token(T.Comment.Single, '-- x')])
        a = Token(T.Name, 'a')
        tl = TokenList([cm, a])
        assert tl.token_first(skip_cm=True) is a
        assert tl.token_first(skip_cm=False) is cm

    def test_token_first_none(self):
        tl = TokenList([])
        assert tl.token_first() is None

    def test_token_next(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        assert tl.token_next(0) == (2, b)
        assert tl.token_next(0, skip_ws=False) == (1, ws)
        assert tl.token_next(2) == (None, None)

    def test_token_next_idx_none(self):
        tl = TokenList([])
        assert tl.token_next(None) == (None, None)

    def test_token_prev(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        assert tl.token_prev(2) == (0, a)
        assert tl.token_prev(2, skip_ws=False) == (1, ws)

    def test_token_next_by(self):
        a = Token(T.Name, 'a')
        b = Token(T.Keyword, 'B')
        tl = TokenList([a, b])
        idx, tok = tl.token_next_by(t=T.Keyword)
        assert (idx, tok) == (1, b)

    def test_token_next_by_match(self):
        a = Token(T.Name, 'a')
        b = Token(T.Punctuation, '.')
        tl = TokenList([a, b])
        idx, tok = tl.token_next_by(m=(T.Punctuation, '.'))
        assert (idx, tok) == (1, b)

    def test_token_matching(self):
        a = Token(T.Name, 'a')
        b = Token(T.Keyword, 'B')
        tl = TokenList([a, b])
        assert tl.token_matching(lambda tk: tk.ttype in T.Keyword, 0) is b

    def test_token_not_matching(self):
        a = Token(T.Name, 'a')
        b = Token(T.Keyword, 'B')
        tl = TokenList([a, b])
        # funcs = is_keyword; returns first non-matching
        idx, tok = tl.token_not_matching(lambda tk: tk.ttype in T.Keyword, 0)
        assert tok is a

    def test_token_index(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_index(a) == 0
        assert tl.token_index(b) == 1
        assert tl.token_index(a, start=0) == 0

    def test_group_tokens(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        grp = tl.group_tokens(TokenList, 0, 1)
        assert isinstance(grp, TokenList)
        assert grp.tokens == [a, b]
        assert tl.tokens == [grp, c]
        assert grp.parent is tl
        assert a.parent is grp

    def test_group_tokens_exclude_end(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        grp = tl.group_tokens(TokenList, 0, 1, include_end=False)
        assert grp.tokens == [a]
        assert tl.tokens == [grp, b, c]

    def test_group_tokens_extend(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        grp = Parenthesis([a])
        c = Token(T.Name, 'c')
        tl = TokenList([grp, b, c])
        result = tl.group_tokens(Parenthesis, 0, 1, extend=True)
        assert result is grp
        assert grp.tokens == [a, b]
        assert tl.tokens == [grp, c]

    def test_insert_before(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        new = Token(T.Name, 'new')
        tl.insert_before(b, new)
        assert tl.tokens == [a, new, b]
        assert new.parent is tl

    def test_insert_before_by_token(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        new = Token(T.Name, 'new')
        tl.insert_before(a, new)
        assert tl.tokens == [new, a, b]

    def test_insert_after(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        new = Token(T.Name, 'new')
        tl.insert_after(a, new)
        assert tl.tokens == [a, new, ws, b]

    def test_insert_after_end(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        new = Token(T.Name, 'new')
        tl.insert_after(a, new)
        assert tl.tokens == [a, new]

    def test_has_alias_default(self):
        tl = TokenList([])
        assert tl.has_alias() is False

    def test_get_alias_default(self):
        tl = TokenList([])
        assert tl.get_alias() is None

    def test_get_name_default(self):
        tl = TokenList([])
        assert tl.get_name() is None

    def test_get_real_name_default(self):
        tl = TokenList([])
        assert tl.get_real_name() is None

    def test_get_parent_name(self):
        a = Token(T.Name, 'a')
        dot = Token(T.Punctuation, '.')
        b = Token(T.Name, 'b')
        tl = TokenList([a, dot, b])
        assert tl.get_parent_name() == 'a'

    def test_get_parent_name_none(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.get_parent_name() is None


# ---------------- NameAliasMixin / Identifier ----------------

class TestIdentifier:
    def test_real_name_simple(self):
        ident = Identifier([Token(T.Name, 'col')])
        assert ident.get_real_name() == 'col'

    def test_real_name_dotted(self):
        a = Token(T.Name, 'a')
        dot = Token(T.Punctuation, '.')
        b = Token(T.Name, 'b')
        ident = Identifier([a, dot, b])
        assert ident.get_real_name() == 'b'

    def test_real_name_quoted(self):
        ident = Identifier([Token(T.Name, '"col"')])
        assert ident.get_real_name() == 'col'

    def test_alias_with_as(self):
        name = Token(T.Name, 'col')
        ws = Token(T.Whitespace, ' ')
        as_kw = Token(T.Keyword, 'AS')
        ws2 = Token(T.Whitespace, ' ')
        alias = Token(T.Name, 'ali')
        ident = Identifier([name, ws, as_kw, ws2, alias])
        assert ident.get_alias() == 'ali'

    def test_alias_without_as(self):
        name = Token(T.Name, 'col')
        ws = Token(T.Whitespace, ' ')
        alias = Token(T.Name, 'ali')
        ident = Identifier([name, ws, alias])
        assert ident.get_alias() == 'ali'

    def test_alias_none(self):
        ident = Identifier([Token(T.Name, 'col')])
        assert ident.get_alias() is None

    def test_alias_none_two_tokens(self):
        ident = Identifier([Token(T.Name, 'col'), Token(T.Name, 'x')])
        assert ident.get_alias() is None

    def test_get_name_prefers_alias(self):
        name = Token(T.Name, 'col')
        ws = Token(T.Whitespace, ' ')
        as_kw = Token(T.Keyword, 'AS')
        ws2 = Token(T.Whitespace, ' ')
        alias = Token(T.Name, 'ali')
        ident = Identifier([name, ws, as_kw, ws2, alias])
        assert ident.get_name() == 'ali'

    def test_is_wildcard(self):
        ident = Identifier([Token(T.Wildcard, '*')])
        assert ident.is_wildcard() is True
        ident2 = Identifier([Token(T.Name, 'x')])
        assert ident2.is_wildcard() is False

    def test_get_typecast(self):
        a = Token(T.Name, 'col')
        mark = Token(T.Punctuation, '::')
        b = Token(T.Name, 'int')
        ident = Identifier([a, mark, b])
        assert ident.get_typecast() == 'int'

    def test_get_typecast_none(self):
        ident = Identifier([Token(T.Name, 'col')])
        assert ident.get_typecast() is None

    def test_get_ordering(self):
        ident = Identifier([
            Token(T.Name, 'col'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword.Order, 'desc'),
        ])
        assert ident.get_ordering() == 'DESC'

    def test_get_ordering_none(self):
        ident = Identifier([Token(T.Name, 'col')])
        assert ident.get_ordering() is None

    def test_get_array_indices(self):
        a = Token(T.Name, 'col')
        sq = SquareBrackets([
            Token(T.Punctuation, '['),
            Token(T.Number.Integer, '1'),
            Token(T.Punctuation, ']'),
        ])
        ident = Identifier([a, sq])
        indices = list(ident.get_array_indices())
        assert len(indices) == 1
        assert [t.value for t in indices[0]] == ['1']

    def test_get_array_indices_empty(self):
        ident = Identifier([Token(T.Name, 'col')])
        assert list(ident.get_array_indices()) == []


# ---------------- IdentifierList ----------------

class TestIdentifierList:
    def test_get_identifiers(self):
        a = Identifier([Token(T.Name, 'a')])
        ws = Token(T.Whitespace, ' ')
        comma = Token(T.Punctuation, ',')
        b = Identifier([Token(T.Name, 'b')])
        il = IdentifierList([a, ws, comma, ws, b])
        assert list(il.get_identifiers()) == [a, b]


# ---------------- Statement ----------------

class TestStatement:
    def test_get_type_empty(self):
        stmt = Statement([])
        assert stmt.get_type() == 'UNKNOWN'

    def test_get_type_whitespace_only(self):
        stmt = Statement([Token(T.Whitespace, ' ')])
        assert stmt.get_type() == 'UNKNOWN'

    def test_get_type_dml(self):
        stmt = Statement([Token(T.Keyword.DML, 'select')])
        assert stmt.get_type() == 'SELECT'

    def test_get_type_ddl(self):
        stmt = Statement([Token(T.Keyword.DDL, 'create')])
        assert stmt.get_type() == 'CREATE'

    def test_get_type_unknown(self):
        stmt = Statement([Token(T.Name, 'x')])
        assert stmt.get_type() == 'UNKNOWN'

    def test_get_type_skips_comments(self):
        cm = Comment([Token(T.Comment.Single, '-- hi')])
        stmt = Statement([cm, Token(T.Whitespace, ' '),
                          Token(T.Keyword.DML, 'insert')])
        assert stmt.get_type() == 'INSERT'

    def test_get_type_cte(self):
        with_kw = Token(T.Keyword.CTE, 'WITH')
        ws = Token(T.Whitespace, ' ')
        ident = Identifier([Token(T.Name, 'x')])
        ws2 = Token(T.Whitespace, ' ')
        dml = Token(T.Keyword.DML, 'select')
        stmt = Statement([with_kw, ws, ident, ws2, dml])
        assert stmt.get_type() == 'SELECT'


# ---------------- TypedLiteral / Parenthesis / SquareBrackets ----------------

class TestTypedLiteral:
    def test_constants(self):
        assert TypedLiteral.M_OPEN == [(T.Name.Builtin, None),
                                       (T.Keyword, "TIMESTAMP")]
        assert TypedLiteral.M_CLOSE == (T.String.Single, None)
        assert TypedLiteral.M_EXTEND[0] == T.Keyword


class TestParenthesis:
    def test_groupable_tokens(self):
        p = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Name, 'x'),
            Token(T.Punctuation, ')'),
        ])
        assert p._groupable_tokens == [
            p.tokens[1]
        ]


class TestSquareBrackets:
    def test_groupable_tokens(self):
        s = SquareBrackets([
            Token(T.Punctuation, '['),
            Token(T.Name, 'x'),
            Token(T.Punctuation, ']'),
        ])
        assert s._groupable_tokens == [s.tokens[1]]


# ---------------- Simple TokenList subclasses ----------------

class TestSimpleSubclasses:
    @pytest.mark.parametrize("cls", [
        Assignment, If, For, Where, Over, Having, Begin, Operation,
        Values, Command,
    ])
    def test_instantiate(self, cls):
        tl = cls()
        assert isinstance(tl, TokenList)
        assert tl.is_group is True
        assert tl.tokens == []

    def test_if_constants(self):
        assert If.M_OPEN == (T.Keyword, 'IF')
        assert If.M_CLOSE == (T.Keyword, 'END IF')

    def test_for_constants(self):
        assert For.M_OPEN == (T.Keyword, ('FOR', 'FOREACH'))

    def test_where_constants(self):
        assert Where.M_OPEN == (T.Keyword, 'WHERE')


# ---------------- Comparison ----------------

class TestComparison:
    def test_left_right(self):
        a = Token(T.Name, 'a')
        op = Token(T.Operator, '=')
        b = Token(T.Name, 'b')
        comp = Comparison([a, op, b])
        assert comp.left is a
        assert comp.right is b


# ---------------- Comment ----------------

class TestComment:
    def test_is_multiline_true(self):
        cm = Comment([Token(T.Comment.Multiline, '/* x */')])
        assert cm.is_multiline() is True

    def test_is_multiline_false(self):
        cm = Comment([Token(T.Comment.Single, '-- x')])
        assert cm.is_multiline() is False

    def test_is_multiline_empty(self):
        cm = Comment([])
        assert cm.is_multiline() == []


# ---------------- Case ----------------

class TestCase:
    def _make_case(self, tokens):
        return Case(tokens)

    def test_get_cases_simple(self):
        tokens = [
            Token(T.Keyword, 'CASE'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'WHEN'),
            Token(T.Name, 'x'),
            Token(T.Keyword, 'THEN'),
            Token(T.Name, 'y'),
            Token(T.Keyword, 'END'),
        ]
        case = Case(tokens)
        result = case.get_cases()
        assert len(result) == 1
        cond, val = result[0]
        assert isinstance(cond, list)
        assert isinstance(val, list)

    def test_get_cases_with_else(self):
        tokens = [
            Token(T.Keyword, 'CASE'),
            Token(T.Keyword, 'WHEN'),
            Token(T.Name, 'x'),
            Token(T.Keyword, 'THEN'),
            Token(T.Name, 'y'),
            Token(T.Keyword, 'ELSE'),
            Token(T.Name, 'z'),
            Token(T.Keyword, 'END'),
        ]
        case = Case(tokens)
        result = case.get_cases()
        # one WHEN plus one ELSE
        assert len(result) == 2
        assert result[0][0] is not None
        assert result[1][0] is None

    def test_get_cases_no_when(self):
        tokens = [
            Token(T.Keyword, 'CASE'),
            Token(T.Name, 'x'),
            Token(T.Keyword, 'END'),
        ]
        case = Case(tokens)
        result = case.get_cases()
        assert len(result) == 1
        assert result[0][0] == [tokens[1]]

    def test_get_cases_empty(self):
        case = Case([Token(T.Keyword, 'CASE'),
                     Token(T.Keyword, 'END')])
        assert case.get_cases() == []


# ---------------- Function ----------------

class TestFunction:
    def test_get_parameters_simple(self):
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            Identifier([Token(T.Name, 'a')]),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), paren])
        params = fn.get_parameters()
        assert len(params) == 1
        assert isinstance(params[0], Identifier)

    def test_get_parameters_identifier_list(self):
        idents = IdentifierList([
            Identifier([Token(T.Name, 'a')]),
            Token(T.Punctuation, ','),
            Identifier([Token(T.Name, 'b')]),
        ])
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            idents,
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), paren])
        params = fn.get_parameters()
        from itertools import chain
        assert list(params) == list(idents.get_identifiers())

    def test_get_parameters_literal(self):
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Literal.Number.Integer, '1'),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), paren])
        params = fn.get_parameters()
        assert len(params) == 1
        assert params[0].value == '1'

    def test_get_window_none(self):
        fn = Function([Token(T.Name, 'f')])
        assert fn.get_window() is None

    def test_get_window(self):
        over = Over([
            Token(T.Keyword, 'OVER'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'w'),
        ])
        fn = Function([Token(T.Name, 'f'), Token(T.Whitespace, ' '), over])
        assert fn.get_window() is over.tokens[-1]


# ---------------- NameAliasMixin in Function ----------------

class TestFunctionNameAlias:
    def test_function_real_name_and_alias(self):
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Punctuation, ')'),
        ])
        name = Token(T.Name, 'f')
        ws = Token(T.Whitespace, ' ')
        as_kw = Token(T.Keyword, 'AS')
        ws2 = Token(T.Whitespace, ' ')
        alias = Token(T.Name, 'g')
        fn = Function([name, paren, ws, as_kw, ws2, alias])
        assert fn.get_real_name() == 'f'
        assert fn.get_alias() == 'g'
        assert fn.get_name() == 'g'


# ---------------- _pprint_tree ----------------

class TestPprintTree:
    def test_pprint(self, capsys):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        tl._pprint_tree()
        out = capsys.readouterr().out
        assert 'Name' in out
        assert "'a'" in out

    def test_pprint_nested(self, capsys):
        a = Token(T.Name, 'a')
        inner = TokenList([a])
        outer = TokenList([inner])
        outer._pprint_tree()
        out = capsys.readouterr().out
        assert 'TokenList' in out
        assert 'Name' in out

    def test_pprint_max_depth(self, capsys):
        a = Token(T.Name, 'a')
        inner = TokenList([a])
        outer = TokenList([inner])
        outer._pprint_tree(max_depth=0)
        out = capsys.readouterr().out
        assert 'TokenList' in out
        # depth limits recursion; child name shouldn't be printed
        assert "'a'" not in out
