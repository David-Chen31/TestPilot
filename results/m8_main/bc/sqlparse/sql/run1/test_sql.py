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

def make_identifier(ttype, value):
    return Token(ttype, value)


def build_token_list(*tokens):
    return TokenList(list(tokens))


def ident_with_alias_as():
    # name AS alias
    return TokenList([
        Token(T.Name, 'foo'),
        Token(T.Whitespace, ' '),
        Token(T.Keyword, 'AS'),
        Token(T.Whitespace, ' '),
        Token(T.Name, 'bar'),
    ])


# ---------------------------------------------------------------------------
# Token tests
# ---------------------------------------------------------------------------

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

    def test_value_is_str_coerced(self):
        tok = Token(T.Name, 123)
        assert tok.value == '123'

    def test_keyword_normalized_upper(self):
        tok = Token(T.Keyword, 'select')
        assert tok.is_keyword is True
        assert tok.normalized == 'SELECT'

    def test_whitespace_and_newline(self):
        ws = Token(T.Whitespace, ' ')
        assert ws.is_whitespace is True
        nl = Token(T.Newline, '\n')
        assert nl.is_newline is True

    def test_str_and_repr(self):
        tok = Token(T.Name, 'abc')
        assert str(tok) == 'abc'
        r = repr(tok)
        assert r.startswith('<Name')
        assert 'abc' in r

    def test_repr_uses_double_quote_for_single_quoted(self):
        tok = Token(T.String.Single, "'abc'")
        r = repr(tok)
        assert '"' in r

    def test_repr_value_truncated(self):
        tok = Token(T.Name, 'abcdefghij')
        r = repr(tok)
        assert 'abcdef...' in r

    def test_repr_value_whitespace_collapsed(self):
        tok = Token(T.String.Single, "'a\n b'")
        r = repr(tok)
        assert '\n' not in r

    def test_flatten_yields_self(self):
        tok = Token(T.Name, 'x')
        assert list(tok.flatten()) == [tok]

    def test_match_no_values(self):
        tok = Token(T.Name, 'foo')
        assert tok.match(T.Name, None) is True
        assert tok.match(T.Keyword, None) is False

    def test_match_string_value(self):
        tok = Token(T.Name, 'foo')
        assert tok.match(T.Name, 'foo') is True
        assert tok.match(T.Name, 'bar') is False

    def test_match_keyword_case_insensitive(self):
        tok = Token(T.Keyword, 'SELECT')
        assert tok.match(T.Keyword, 'select') is True
        assert tok.match(T.Keyword, 'SELECT') is True

    def test_match_tuple_values(self):
        tok = Token(T.Name, 'foo')
        assert tok.match(T.Name, ('bar', 'foo')) is True
        assert tok.match(T.Name, ('bar', 'baz')) is False

    def test_match_regex(self):
        tok = Token(T.Name, 'foobar')
        assert tok.match(T.Name, 'foo', regex=True) is True
        assert tok.match(T.Name, 'xyz', regex=True) is False

    def test_match_regex_keyword_ignorecase(self):
        tok = Token(T.Keyword, 'SELECT')
        assert tok.match(T.Keyword, 'sel', regex=True) is True

    def test_is_child_of(self):
        parent = TokenList([Token(T.Name, 'a')])
        child = parent.tokens[0]
        assert child.is_child_of(parent) is True
        assert child.is_child_of(TokenList([])) is False

    def test_has_ancestor(self):
        inner = Token(T.Name, 'a')
        mid = TokenList([inner])
        outer = TokenList([mid])
        assert inner.has_ancestor(outer) is True
        assert inner.has_ancestor(mid) is True
        assert mid.has_ancestor(inner) is False

    def test_within(self):
        inner = Token(T.Name, 'a')
        mid = Function([inner])
        outer = TokenList([mid])
        assert inner.within(Function) is True
        assert inner.within(Parenthesis) is False


# ---------------------------------------------------------------------------
# TokenList tests
# ---------------------------------------------------------------------------

class TestTokenList:
    def test_init_sets_parents(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert a.parent is tl
        assert b.parent is tl
        assert tl.is_group is True

    def test_init_empty(self):
        tl = TokenList()
        assert tl.tokens == []
        assert tl.value == ''

    def test_str_concatenates(self):
        tl = TokenList([Token(T.Name, 'a'), Token(T.Name, 'b')])
        assert str(tl) == 'ab'

    def test_iter_and_getitem(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert list(tl) == [a, b]
        assert tl[0] is a
        assert tl[1] is b

    def test_flatten(self):
        a = Token(T.Name, 'a')
        inner = TokenList([Token(T.Name, 'b')])
        tl = TokenList([a, inner])
        flat = list(tl.flatten())
        assert flat[0] is a
        assert flat[1].value == 'b'
        assert all(not t.is_group for t in flat)

    def test_get_sublists(self):
        a = Token(T.Name, 'a')
        inner = TokenList([])
        tl = TokenList([a, inner])
        sublists = list(tl.get_sublists())
        assert sublists == [inner]

    def test_token_first_default(self):
        ws = Token(T.Whitespace, ' ')
        a = Token(T.Name, 'a')
        tl = TokenList([ws, a])
        assert tl.token_first() is a

    def test_token_first_skip_ws_false(self):
        ws = Token(T.Whitespace, ' ')
        a = Token(T.Name, 'a')
        tl = TokenList([ws, a])
        assert tl.token_first(skip_ws=False) is ws

    def test_token_first_empty(self):
        tl = TokenList([])
        assert tl.token_first() is None

    def test_token_first_skip_cm(self):
        cm = Comment([Token(T.Comment.Single, '--x')])
        a = Token(T.Name, 'a')
        tl = TokenList([cm, a])
        assert tl.token_first(skip_cm=True) is a
        assert tl.token_first() is cm

    def test_token_next(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, tok = tl.token_next(0)
        assert idx == 2
        assert tok is b

    def test_token_next_skip_ws_false(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, tok = tl.token_next(0, skip_ws=False)
        assert idx == 1
        assert tok is ws

    def test_token_next_none_idx(self):
        tl = TokenList([])
        assert tl.token_next(None) == (None, None)

    def test_token_next_no_next(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl.token_next(0) == (None, None)

    def test_token_prev(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, tok = tl.token_prev(2)
        assert idx == 0
        assert tok is a

    def test_token_prev_skip_ws_false(self):
        a = Token(T.Name, 'a')
        ws = Token(T.Whitespace, ' ')
        b = Token(T.Name, 'b')
        tl = TokenList([a, ws, b])
        idx, tok = tl.token_prev(2, skip_ws=False)
        assert idx == 1
        assert tok is ws

    def test_token_next_by(self):
        a = Token(T.Name, 'a')
        kw = Token(T.Keyword, 'AS')
        tl = TokenList([a, kw])
        idx, tok = tl.token_next_by(t=T.Keyword)
        assert idx == 1
        assert tok is kw

    def test_token_next_by_match(self):
        a = Token(T.Name, 'a')
        dot = Token(T.Punctuation, '.')
        tl = TokenList([a, dot])
        idx, tok = tl.token_next_by(m=(T.Punctuation, '.'))
        assert idx == 1
        assert tok is dot

    def test_token_next_by_not_found(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.token_next_by(t=T.Keyword) == (None, None)

    def test_token_not_matching(self):
        tl = TokenList([Token(T.Whitespace, ' '), Token(T.Name, 'a')])
        idx, tok = tl.token_not_matching(lambda tk: tk.is_whitespace, 0)
        assert idx == 1
        assert tok.ttype == T.Name

    def test_token_matching(self):
        tl = TokenList([Token(T.Whitespace, ' '), Token(T.Name, 'a')])
        tok = tl.token_matching(lambda tk: tk.ttype == T.Name, 0)
        assert tok.ttype == T.Name

    def test_token_index_int_start(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_index(b) == 1

    def test_token_index_token_start(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        assert tl.token_index(c, start=a) == 2

    def test_get_token_at_offset(self):
        a = Token(T.Name, 'ab')
        b = Token(T.Name, 'cd')
        tl = TokenList([a, b])
        assert tl.get_token_at_offset(0) is a
        assert tl.get_token_at_offset(1) is a
        assert tl.get_token_at_offset(2) is b
        assert tl.get_token_at_offset(10) is None

    def test_group_tokens(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        grp = tl.group_tokens(Parenthesis, 0, 1)
        assert isinstance(grp, Parenthesis)
        assert len(tl.tokens) == 1
        assert tl.tokens[0] is grp
        assert a.parent is grp
        assert b.parent is grp

    def test_group_tokens_extend(self):
        inner = Token(T.Name, 'b')
        grp = Parenthesis([inner])
        tl = TokenList([grp, Token(T.Name, 'c')])
        new = tl.group_tokens(Parenthesis, 0, 1, extend=True)
        assert new is grp
        assert len(new.tokens) == 2

    def test_insert_before(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        new = Token(T.Name, 'x')
        tl.insert_before(b, new)
        assert tl.tokens[1] is new
        assert new.parent is tl

    def test_insert_before_token(self):
        # 'where' is a token, not int -> exercises token_index path.
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        new = Token(T.Name, 'x')
        tl.insert_before(b, new)
        assert [t.value for t in tl.tokens] == ['a', 'x', 'b']
        assert tl.tokens[1] is new
        assert new.parent is tl

    def test_insert_after(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        new = Token(T.Name, 'x')
        tl.insert_after(a, new)
        assert tl.tokens[1] is new
        assert new.parent is tl

    def test_insert_after_token(self):
        # 'where' is a token, not int -> exercises token_index path.
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        new = Token(T.Name, 'x')
        tl.insert_after(a, new)
        assert [t.value for t in tl.tokens] == ['a', 'x', 'b']
        assert tl.tokens[1] is new
        assert new.parent is tl

    def test_insert_after_end(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        new = Token(T.Name, 'x')
        tl.insert_after(a, new)
        assert tl.tokens[-1] is new

    def test_has_alias_false_and_get_alias_none(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.get_alias() is None
        assert tl.has_alias() is False

    def test_get_name_returns_real_name(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.get_real_name() is None
        assert tl.get_name() is None

    def test_get_parent_name(self):
        tl = TokenList([
            Token(T.Name, 'a'),
            Token(T.Punctuation, '.'),
            Token(T.Name, 'b'),
        ])
        assert tl.get_parent_name() == 'a'

    def test_get_parent_name_quoted(self):
        tl = TokenList([
            Token(T.Name, '"a"'),
            Token(T.Punctuation, '.'),
            Token(T.Name, 'b'),
        ])
        assert tl.get_parent_name() == 'a'

    def test_get_parent_name_none(self):
        tl = TokenList([Token(T.Name, 'b')])
        assert tl.get_parent_name() is None

    def test_get_first_name_keyword(self):
        tl = TokenList([Token(T.Keyword, 'bar')])
        assert tl._get_first_name(keywords=True) == 'bar'
        assert tl._get_first_name(keywords=False) is None

    def test_get_first_name_nested_identifier_returns_name(self):
        # First token is an Identifier -> returns its get_name().
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'AS'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        tl = TokenList([ident])
        assert tl._get_first_name() == 'bar'

    def test_get_first_name_nested_identifier_real_name(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'AS'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        tl = TokenList([ident])
        assert tl._get_first_name(real_name=True) == 'foo'

    def test_get_first_name_nested_function(self):
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'myfunc'), paren])
        tl = TokenList([fn])
        # Function.get_name() defaults to get_real_name() (no alias) -> 'myfunc'
        assert tl._get_first_name() == 'myfunc'
        assert tl._get_first_name(real_name=True) == 'myfunc'

    def test_pprint_tree(self, capsys):
        tl = TokenList([Token(T.Name, 'a')])
        tl._pprint_tree()
        out = capsys.readouterr().out
        assert 'Name' in out
        assert "'a'" in out

    def test_pprint_tree_nested_group(self, capsys):
        inner = TokenList([Token(T.Name, 'b')])
        tl = TokenList([inner])
        tl._pprint_tree()
        out = capsys.readouterr().out
        assert 'TokenList' in out
        assert 'Name' in out
        assert "'b'" in out

    def test_repr_name_uses_class_name(self):
        # TokenList._get_repr_name returns the concrete class name.
        tl = TokenList([])
        assert tl._get_repr_name() == 'TokenList'
        st = Statement([])
        assert st._get_repr_name() == 'Statement'
        ident = Identifier([])
        assert ident._get_repr_name() == 'Identifier'

    def test_groupable_tokens_property(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl._groupable_tokens == [a]
        assert tl._groupable_tokens is tl.tokens


# ---------------------------------------------------------------------------
# NameAliasMixin / Identifier / Function
# ---------------------------------------------------------------------------

class TestNameAlias:
    def test_get_real_name_simple(self):
        ident = Identifier([Token(T.Name, 'foo')])
        assert ident.get_real_name() == 'foo'

    def test_get_real_name_after_last_dot(self):
        ident = Identifier([
            Token(T.Name, 'a'),
            Token(T.Punctuation, '.'),
            Token(T.Name, 'b'),
            Token(T.Punctuation, '.'),
            Token(T.Name, 'c'),
        ])
        assert ident.get_real_name() == 'c'

    def test_get_real_name_quoted(self):
        ident = Identifier([Token(T.Name, '"foo"')])
        assert ident.get_real_name() == 'foo'

    def test_get_alias_as(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'AS'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        assert ident.get_alias() == 'bar'

    def test_get_alias_without_as(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        assert ident.get_alias() == 'bar'

    def test_get_alias_none_short(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
        ])
        assert ident.get_alias() is None

    def test_get_alias_none_no_ws(self):
        ident = Identifier([Token(T.Name, 'foo')])
        assert ident.get_alias() is None

    def test_identifier_has_alias(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'AS'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        assert ident.has_alias() is True

    def test_identifier_get_name_prefers_alias(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'AS'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'bar'),
        ])
        assert ident.get_name() == 'bar'


class TestIdentifier:
    def test_is_wildcard(self):
        ident = Identifier([Token(T.Wildcard, '*')])
        assert ident.is_wildcard() is True
        ident2 = Identifier([Token(T.Name, 'foo')])
        assert ident2.is_wildcard() is False

    def test_get_typecast(self):
        ident = Identifier([
            Token(T.Name, 'foo'),
            Token(T.Punctuation, '::'),
            Token(T.Name, 'int'),
        ])
        assert ident.get_typecast() == 'int'

    def test_get_typecast_none(self):
        ident = Identifier([Token(T.Name, 'foo')])
        assert ident.get_typecast() is None

    def test_get_ordering(self):
        ident = Identifier([Token(T.Keyword.Order, 'desc')])
        assert ident.get_ordering() == 'DESC'

    def test_get_ordering_none(self):
        ident = Identifier([Token(T.Name, 'foo')])
        assert ident.get_ordering() is None

    def test_get_array_indices(self):
        inner_tok = Token(T.Number.Integer, '1')
        sb = SquareBrackets([
            Token(T.Punctuation, '['),
            inner_tok,
            Token(T.Punctuation, ']'),
        ])
        ident = Identifier([Token(T.Name, 'foo'), sb])
        indices = list(ident.get_array_indices())
        assert len(indices) == 1
        assert indices[0] == [inner_tok]
        assert indices[0][0] is inner_tok


# ---------------------------------------------------------------------------
# IdentifierList
# ---------------------------------------------------------------------------

class TestIdentifierList:
    def test_get_identifiers(self):
        a = Identifier([Token(T.Name, 'a')])
        b = Identifier([Token(T.Name, 'b')])
        il = IdentifierList([
            a,
            Token(T.Punctuation, ','),
            Token(T.Whitespace, ' '),
            b,
        ])
        result = list(il.get_identifiers())
        assert result == [a, b]


# ---------------------------------------------------------------------------
# TypedLiteral / Parenthesis / SquareBrackets
# ---------------------------------------------------------------------------

class TestTypedLiteral:
    def test_constants(self):
        assert TypedLiteral.M_CLOSE == (T.String.Single, None)
        assert (T.Name.Builtin, None) in TypedLiteral.M_OPEN


class TestParenthesis:
    def test_groupable_tokens(self):
        inside = Token(T.Name, 'a')
        p = Parenthesis([
            Token(T.Punctuation, '('),
            inside,
            Token(T.Punctuation, ')'),
        ])
        assert p._groupable_tokens == [inside]


class TestSquareBrackets:
    def test_groupable_tokens(self):
        inside = Token(T.Name, 'a')
        sb = SquareBrackets([
            Token(T.Punctuation, '['),
            inside,
            Token(T.Punctuation, ']'),
        ])
        assert sb._groupable_tokens == [inside]


# ---------------------------------------------------------------------------
# Simple subclasses
# ---------------------------------------------------------------------------

class TestSubclasses:
    def test_assignment(self):
        a = Assignment([Token(T.Name, 'x')])
        assert isinstance(a, TokenList)

    def test_if(self):
        assert If.M_OPEN == (T.Keyword, 'IF')
        assert If.M_CLOSE == (T.Keyword, 'END IF')

    def test_for(self):
        assert For.M_OPEN == (T.Keyword, ('FOR', 'FOREACH'))
        assert For.M_CLOSE == (T.Keyword, 'END LOOP')

    def test_comparison_left_right(self):
        left = Token(T.Name, 'a')
        right = Token(T.Name, 'b')
        c = Comparison([left, Token(T.Punctuation, '='), right])
        assert c.left is left
        assert c.right is right

    def test_comment_is_multiline(self):
        cm = Comment([Token(T.Comment.Multiline, '/* x */')])
        assert cm.is_multiline() is True
        cm2 = Comment([Token(T.Comment.Single, '-- x')])
        assert cm2.is_multiline() is False

    def test_comment_empty(self):
        cm = Comment([])
        assert cm.is_multiline() == []

    def test_where(self):
        assert 'WHERE' in Where.M_OPEN
        assert Where.M_CLOSE[0] == T.Keyword

    def test_over(self):
        assert Over.M_OPEN == (T.Keyword, 'OVER')

    def test_having(self):
        assert Having.M_OPEN == (T.Keyword, 'HAVING')

    def test_begin(self):
        assert Begin.M_OPEN == (T.Keyword, 'BEGIN')
        assert Begin.M_CLOSE == (T.Keyword, 'END')

    def test_operation_values_command(self):
        assert isinstance(Operation([]), TokenList)
        assert isinstance(Values([]), TokenList)
        assert isinstance(Command([]), TokenList)


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

class TestCase:
    def test_get_cases_basic(self):
        case = Case([
            Token(T.Keyword, 'CASE'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'WHEN'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'a'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'THEN'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'b'),
            Token(T.Whitespace, ' '),
            Token(T.Keyword, 'END'),
        ])
        cases = case.get_cases(skip_ws=True)
        assert len(cases) == 1
        cond, val = cases[0]
        # The WHEN keyword is included in the condition list.
        assert [t.value for t in cond] == ['WHEN', 'a']
        # The THEN keyword is included in the value list.
        assert [t.value for t in val] == ['THEN', 'b']

    def test_get_cases_with_else(self):
        case = Case([
            Token(T.Keyword, 'CASE'),
            Token(T.Keyword, 'WHEN'),
            Token(T.Name, 'a'),
            Token(T.Keyword, 'THEN'),
            Token(T.Name, 'b'),
            Token(T.Keyword, 'ELSE'),
            Token(T.Name, 'c'),
            Token(T.Keyword, 'END'),
        ])
        cases = case.get_cases(skip_ws=True)
        assert len(cases) == 2
        assert cases[1][0] is None
        # The ELSE keyword is included in the value list.
        assert [t.value for t in cases[1][1]] == ['ELSE', 'c']

    def test_get_cases_empty(self):
        case = Case([Token(T.Keyword, 'CASE'), Token(T.Keyword, 'END')])
        assert case.get_cases(skip_ws=True) == []

    def test_get_cases_no_when(self):
        # No WHEN / ELSE / THEN: the 'first condition without preceding WHEN'
        # branch fires, creating an empty (cond, val) pair that holds CASE.
        case = Case([
            Token(T.Keyword, 'CASE'),
            Token(T.Name, 'a'),
        ])
        cases = case.get_cases(skip_ws=False)
        # CASE is skipped; 'a' with mode==CONDITION and no ret -> appends
        # [( [], [] )] then adds 'a' to the condition list.
        assert len(cases) == 1
        cond, val = cases[0]
        assert [t.value for t in cond] == ['a']
        assert val == []


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

class TestFunction:
    def test_get_parameters_simple(self):
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Number.Integer, '1'),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), paren])
        params = fn.get_parameters()
        assert len(params) == 1
        assert params[0].value == '1'

    def test_get_parameters_identifier_list(self):
        a = Identifier([Token(T.Name, 'a')])
        b = Identifier([Token(T.Name, 'b')])
        idlist = IdentifierList([a, Token(T.Punctuation, ','), b])
        paren = Parenthesis([Token(T.Punctuation, '('), idlist, Token(T.Punctuation, ')')])
        fn = Function([Token(T.Name, 'f'), paren])
        params = list(fn.get_parameters())
        assert params == [a, b]

    def test_get_parameters_ignores_comma_token(self):
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Number.Integer, '1'),
            Token(T.Punctuation, ','),
            Token(T.Number.Integer, '2'),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), paren])
        params = fn.get_parameters()
        assert len(params) == 2

    def test_get_window_no_over_raises(self):
        # `get_window` uses `token_next_by`, which returns the tuple
        # (None, None) on no match, so `if not over_clause` is always False.
        # Without an Over clause, accessing `[1].tokens` raises AttributeError.
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Number.Integer, '1'),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), paren])
        with pytest.raises(AttributeError):
            fn.get_window()

    def test_get_window_present(self):
        over = Over([
            Token(T.Keyword, 'OVER'),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'w'),
        ])
        paren = Parenthesis([
            Token(T.Punctuation, '('),
            Token(T.Punctuation, ')'),
        ])
        fn = Function([Token(T.Name, 'f'), paren, Token(T.Whitespace, ' '), over])
        assert fn.get_window() is over.tokens[-1]


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

class TestStatement:
    def test_get_type_dml(self):
        st = Statement([Token(T.Keyword.DML, 'select'), Token(T.Whitespace, ' ')])
        assert st.get_type() == 'SELECT'

    def test_get_type_ddl(self):
        st = Statement([Token(T.Keyword.DDL, 'create')])
        assert st.get_type() == 'CREATE'

    def test_get_type_empty(self):
        assert Statement([]).get_type() == 'UNKNOWN'
        assert Statement([Token(T.Whitespace, ' ')]).get_type() == 'UNKNOWN'

    def test_get_type_unknown(self):
        st = Statement([Token(T.Name, 'foo')])
        assert st.get_type() == 'UNKNOWN'

    def test_get_type_with_cte(self):
        st = Statement([
            Token(T.Keyword.CTE, 'with'),
            Token(T.Whitespace, ' '),
            Identifier([Token(T.Name, 'x')]),
            Token(T.Whitespace, ' '),
            Token(T.Keyword.DML, 'select'),
        ])
        assert st.get_type() == 'SELECT'

    def test_get_type_with_cte_no_dml(self):
        st = Statement([
            Token(T.Keyword.CTE, 'with'),
            Token(T.Whitespace, ' '),
            Identifier([Token(T.Name, 'x')]),
            Token(T.Whitespace, ' '),
            Token(T.Name, 'foo'),
        ])
        assert st.get_type() == 'UNKNOWN'


# ---------------------------------------------------------------------------
# TokenList._token_matching internals
# ---------------------------------------------------------------------------

class TestTokenMatchingInternal:
    def test_start_none(self):
        tl = TokenList([])
        # start=None returns None (not a tuple).
        assert tl._token_matching(lambda t: True, start=None) is None

    def test_reverse_with_end_raises(self):
        tl = TokenList([Token(T.Name, 'a')])
        with pytest.raises(AssertionError):
            tl._token_matching(lambda t: True, start=1, end=1, reverse=True)

    def test_single_func(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl._token_matching(lambda t: True) == (0, a)

    def test_end_bound(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl._token_matching(lambda t: True, 0, end=1) == (0, a)
