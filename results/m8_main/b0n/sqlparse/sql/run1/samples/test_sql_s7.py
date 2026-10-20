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
# helpers
# ---------------------------------------------------------------------------

def ws():
    return Token(T.Whitespace, ' ')


def name(v):
    return Token(T.Name, v)


def punct(v):
    return Token(T.Punctuation, v)


def kw(v):
    return Token(T.Keyword, v)


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

class TestToken:
    def test_init(self):
        t = Token(T.Name, 'foo')
        assert t.value == 'foo'
        assert t.ttype is T.Name
        assert t.parent is None
        assert t.is_group is False
        assert t.is_keyword is False
        assert t.is_whitespace is False
        assert t.is_newline is False
        assert t.normalized == 'foo'

    def test_init_converts_value_to_str(self):
        t = Token(T.Name, 123)
        assert t.value == '123'

    def test_keyword_normalized_upper(self):
        t = Token(T.Keyword, 'select')
        assert t.is_keyword is True
        assert t.normalized == 'SELECT'

    def test_whitespace(self):
        t = Token(T.Whitespace, ' ')
        assert t.is_whitespace is True
        assert t.is_keyword is False

    def test_newline(self):
        t = Token(T.Newline, '\n')
        assert t.is_newline is True

    def test_str(self):
        assert str(Token(T.Name, 'abc')) == 'abc'

    def test_repr_name(self):
        t = Token(T.Keyword, 'SELECT')
        assert t._get_repr_name() == 'Keyword'

    def test_repr_value_truncation(self):
        t = Token(T.Name, 'abcdefghij')
        assert t._get_repr_value() == 'abcdef...'

    def test_repr_value_short(self):
        t = Token(T.Name, 'abc')
        assert t._get_repr_value() == 'abc'

    def test_repr_value_whitespace_collapse(self):
        t = Token(T.Name, 'a\n\nb')
        assert t._get_repr_value() == 'a b'

    def test_repr(self):
        t = Token(T.Name, 'abc')
        r = repr(t)
        assert r.startswith("<Name 'abc' at 0x")
        assert r.endswith('>')

    def test_repr_value_starts_ends_single_quote(self):
        t = Token(T.String.Single, "'abc'")
        r = repr(t)
        assert r.startswith('<String.Single "')

    def test_flatten_yields_self(self):
        t = Token(T.Name, 'x')
        assert list(t.flatten()) == [t]

    def test_match_type_only(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, None) is True
        assert t.match(T.Name, 'anything') is False
        assert t.match(T.Keyword, None) is False

    def test_match_string_value(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, 'foo') is True
        assert t.match(T.Name, 'bar') is False

    def test_match_keyword_case_insensitive(self):
        t = Token(T.Keyword, 'select')
        assert t.match(T.Keyword, 'SELECT') is True
        assert t.match(T.Keyword, 'select') is True

    def test_match_tuple_values(self):
        t = Token(T.Name, 'foo')
        assert t.match(T.Name, ('bar', 'foo')) is True

    def test_match_regex(self):
        t = Token(T.Name, 'foobar')
        assert t.match(T.Name, 'foo.*', regex=True) is True
        assert t.match(T.Name, '^bar', regex=True) is False

    def test_match_regex_keyword_ignorecase(self):
        t = Token(T.Keyword, 'select')
        assert t.match(T.Keyword, 'sel', regex=True) is True

    def test_within(self):
        parent = TokenList([])
        child = Token(T.Name, 'x')
        parent.tokens.append(child)
        child.parent = parent
        assert child.within(TokenList) is True
        assert child.within(Statement) is False

    def test_within_no_parent(self):
        t = Token(T.Name, 'x')
        assert t.within(TokenList) is False

    def test_is_child_of(self):
        parent = TokenList([])
        child = Token(T.Name, 'x')
        child.parent = parent
        assert child.is_child_of(parent) is True

    def test_has_ancestor(self):
        grand = TokenList([])
        parent = TokenList([])
        child = Token(T.Name, 'x')
        grand.tokens.append(parent)
        parent.parent = grand
        parent.tokens.append(child)
        child.parent = parent
        assert child.has_ancestor(grand) is True
        assert child.has_ancestor(parent) is True

    def test_has_ancestor_false(self):
        t = Token(T.Name, 'x')
        assert t.has_ancestor(TokenList([])) is False


# ---------------------------------------------------------------------------
# TokenList
# ---------------------------------------------------------------------------

class TestTokenList:
    def test_init_empty(self):
        tl = TokenList()
        assert tl.tokens == []
        assert tl.is_group is True
        assert tl.value == ''

    def test_init_sets_parent(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert a.parent is tl
        assert b.parent is tl

    def test_value_joined(self):
        tl = TokenList([Token(T.Name, 'a'), Token(T.Name, 'b')])
        assert tl.value == 'ab'

    def test_str_recurses(self):
        inner = TokenList([Token(T.Name, 'a')])
        outer = TokenList([inner, Token(T.Name, 'b')])
        assert str(outer) == 'ab'

    def test_iter(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert list(tl) == [a]

    def test_getitem(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        assert tl[0] is a

    def test_repr_name(self):
        assert TokenList([])._get_repr_name() == 'TokenList'

    def test_flatten(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert list(tl.flatten()) == [a, b]

    def test_flatten_nested(self):
        a = Token(T.Name, 'a')
        inner = TokenList([a])
        tl = TokenList([inner])
        assert list(tl.flatten()) == [a]

    def test_get_sublists(self):
        a = Token(T.Name, 'a')
        inner = TokenList([a])
        tl = TokenList([a, inner])
        assert list(tl.get_sublists()) == [inner]

    def test_get_token_at_offset(self):
        a = Token(T.Name, 'abc')
        b = Token(T.Name, 'de')
        tl = TokenList([a, b])
        assert tl.get_token_at_offset(0) is a
        assert tl.get_token_at_offset(2) is a
        assert tl.get_token_at_offset(3) is b
        assert tl.get_token_at_offset(4) is b
        assert tl.get_token_at_offset(5) is None

    def test_token_first_skips_whitespace(self):
        a = Token(T.Name, 'a')
        tl = TokenList([ws(), a])
        assert tl.token_first() is a

    def test_token_first_no_skip_ws(self):
        w = ws()
        a = Token(T.Name, 'a')
        tl = TokenList([w, a])
        assert tl.token_first(skip_ws=False) is w

    def test_token_first_empty(self):
        assert TokenList([]).token_first() is None

    def test_token_first_skip_cm(self):
        cm = Comment([Token(T.Comment.Single, '-- c')])
        a = Token(T.Name, 'a')
        tl = TokenList([cm, a])
        assert tl.token_first(skip_cm=True) is a
        assert tl.token_first() is cm

    def test_token_next_by_instance(self):
        cm = Comment([Token(T.Comment.Single, '-- c')])
        a = Token(T.Name, 'a')
        tl = TokenList([a, cm])
        idx, tok = tl.token_next_by(i=Comment)
        assert idx == 1
        assert tok is cm

    def test_token_next_by_match(self):
        a = Token(T.Name, 'a')
        b = Token(T.Punctuation, '.')
        tl = TokenList([a, b])
        idx, tok = tl.token_next_by(m=(T.Punctuation, '.'))
        assert idx == 1
        assert tok is b

    def test_token_next_by_type(self):
        a = Token(T.Name, 'a')
        w = ws()
        tl = TokenList([a, w])
        idx, tok = tl.token_next_by(t=T.Whitespace)
        assert idx == 1
        assert tok is w

    def test_token_next_by_none(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.token_next_by(t=T.Whitespace) == (None, None)

    def test_token_next(self):
        a = Token(T.Name, 'a')
        w = ws()
        b = Token(T.Name, 'b')
        tl = TokenList([a, w, b])
        idx, tok = tl.token_next(0)
        assert idx == 2
        assert tok is b

    def test_token_next_skip_ws_false(self):
        a = Token(T.Name, 'a')
        w = ws()
        tl = TokenList([a, w])
        idx, tok = tl.token_next(0, skip_ws=False)
        assert idx == 1
        assert tok is w

    def test_token_next_idx_none(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.token_next(None) == (None, None)

    def test_token_next_at_end(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.token_next(0) == (None, None)

    def test_token_prev(self):
        a = Token(T.Name, 'a')
        w = ws()
        b = Token(T.Name, 'b')
        tl = TokenList([a, w, b])
        idx, tok = tl.token_prev(2)
        assert idx == 0
        assert tok is a

    def test_token_prev_none(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.token_prev(0) == (None, None)

    def test_token_index(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_index(b) == 1

    def test_token_index_start_token(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        assert tl.token_index(b, start=a) == 1

    def test_get_alias_none(self):
        assert TokenList([]).get_alias() is None

    def test_has_alias_false(self):
        assert TokenList([]).has_alias() is False

    def test_get_name(self):
        assert TokenList([]).get_name() is None

    def test_get_real_name_none(self):
        assert TokenList([]).get_real_name() is None

    def test_get_parent_name(self):
        a = Token(T.Name, 'parent')
        d = Token(T.Punctuation, '.')
        b = Token(T.Name, 'child')
        tl = TokenList([a, d, b])
        assert tl.get_parent_name() == 'parent'

    def test_get_parent_name_quoted(self):
        a = Token(T.Name, '"parent"')
        d = Token(T.Punctuation, '.')
        b = Token(T.Name, 'child')
        tl = TokenList([a, d, b])
        assert tl.get_parent_name() == 'parent'

    def test_get_parent_name_none(self):
        tl = TokenList([Token(T.Name, 'a')])
        assert tl.get_parent_name() is None

    def test_insert_before_int(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        b = Token(T.Name, 'b')
        tl.insert_before(0, b)
        assert tl.tokens == [b, a]
        assert b.parent is tl

    def test_insert_before_token(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        b = Token(T.Name, 'b')
        tl.insert_before(a, b)
        assert tl.tokens == [b, a]

    def test_insert_after(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        c = Token(T.Name, 'c')
        tl.insert_after(a, c)
        assert tl.tokens == [a, c, b]
        assert c.parent is tl

    def test_insert_after_end(self):
        a = Token(T.Name, 'a')
        tl = TokenList([a])
        b = Token(T.Name, 'b')
        tl.insert_after(a, b)
        assert tl.tokens == [a, b]

    def test_group_tokens(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        c = Token(T.Name, 'c')
        tl = TokenList([a, b, c])
        grp = tl.group_tokens(TokenList, 0, 1)
        assert isinstance(grp, TokenList)
        assert grp.tokens == [a, b]
        assert tl.tokens == [grp, c]
        assert a.parent is grp
        assert b.parent is grp
        assert grp.parent is tl

    def test_group_tokens_no_include_end(self):
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
        c = Token(T.Name, 'c')
        existing = TokenList([a])
        tl = TokenList([existing, b, c])
        grp = tl.group_tokens(TokenList, 0, 1, extend=True)
        assert grp is existing
        assert existing.tokens == [a, b]
        assert existing.value == 'ab'

    def test_token_matching(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        result = tl.token_matching(lambda tk: tk.value == 'b', 0)
        assert result is b

    def test_token_not_matching(self):
        a = Token(T.Name, 'a')
        b = Token(T.Name, 'b')
        tl = TokenList([a, b])
        idx, tok = tl.token_not_matching(lambda tk: tk.value == 'a', 0)
        assert idx == 1
        assert tok is b


# ---------------------------------------------------------------------------
# NameAliasMixin / Identifier
# ---------------------------------------------------------------------------

class TestIdentifier:
    def test_is_wildcard_true(self):
        ident = Identifier([Token(T.Wildcard, '*')])
        assert ident.is_wildcard() is True

    def test_is_wildcard_false(self):
        ident = Identifier([name('foo')])
        assert ident.is_wildcard() is False

    def test_get_real_name_simple(self):
        ident = Identifier([name('foo')])
        assert ident.get_real_name() == 'foo'

    def test_get_real_name_dotted(self):
        ident = Identifier([name('a'), punct('.'), name('b')])
        assert ident.get_real_name() == 'b'

    def test_get_real_name_quoted(self):
        ident = Identifier([name('"foo"')])
        assert ident.get_real_name() == 'foo'

    def test_get_real_name_none(self):
        ident = Identifier([Token(T.Whitespace, ' ')])
        assert ident.get_real_name() is None

    def test_get_alias_as(self):
        ident = Identifier([name('foo'), ws(), kw('AS'), ws(), name('bar')])
        assert ident.get_alias() == 'bar'

    def test_get_alias_no_as(self):
        ident = Identifier([name('foo'), ws(), name('bar')])
        assert ident.get_alias() == 'bar'

    def test_get_alias_none(self):
        ident = Identifier([name('foo')])
        assert ident.get_alias() is None

    def test_get_alias_as_quoted(self):
        ident = Identifier([name('foo'), ws(), kw('AS'), ws(), name('"bar"')])
        assert ident.get_alias() == 'bar'

    def test_get_name_prefers_alias(self):
        ident = Identifier([name('foo'), ws(), name('bar')])
        assert ident.get_name() == 'bar'

    def test_get_name_real(self):
        ident = Identifier([name('foo')])
        assert ident.get_name() == 'foo'

    def test_has_alias_true(self):
        ident = Identifier([name('foo'), ws(), name('bar')])
        assert ident.has_alias() is True

    def test_get_parent_name(self):
        ident = Identifier([name('p'), punct('.'), name('c')])
        assert ident.get_parent_name() == 'p'

    def test_get_typecast(self):
        ident = Identifier([
            name('foo'), Token(T.Punctuation, '::'), name('int'),
        ])
        assert ident.get_typecast() == 'int'

    def test_get_typecast_none(self):
        ident = Identifier([name('foo')])
        assert ident.get_typecast() is None

    def test_get_ordering(self):
        ident = Identifier([
            name('foo'), ws(), Token(T.Keyword.Order, 'asc'),
        ])
        assert ident.get_ordering() == 'ASC'

    def test_get_ordering_none(self):
        ident = Identifier([name('foo')])
        assert ident.get_ordering() is None

    def test_get_array_indices(self):
        inner = TokenList([Token(T.Literal.Number.Integer, '1')])
        sb = SquareBrackets([punct('['), inner, punct(']')])
        ident = Identifier([name('a'), sb])
        result = list(ident.get_array_indices())
        assert len(result) == 1
        assert list(result[0]) == list(inner.tokens)


# ---------------------------------------------------------------------------
# IdentifierList
# ---------------------------------------------------------------------------

class TestIdentifierList:
    def test_get_identifiers(self):
        a = Identifier([name('a')])
        comma = punct(',')
        w = ws()
        b = Identifier([name('b')])
        il = IdentifierList([a, comma, w, b])
        assert list(il.get_identifiers()) == [a, b]


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

class TestStatement:
    def test_get_type_dml(self):
        st = Statement([kw('select'), ws(), name('foo')])
        assert st.get_type() == 'SELECT'

    def test_get_type_ddl(self):
        st = Statement([kw('create'), ws(), name('foo')])
        assert st.get_type() == 'CREATE'

    def test_get_type_empty(self):
        assert Statement([]).get_type() == 'UNKNOWN'

    def test_get_type_only_whitespace(self):
        assert Statement([ws()]).get_type() == 'UNKNOWN'

    def test_get_type_skips_comments(self):
        cm = Comment([Token(T.Comment.Single, '-- x')])
        st = Statement([cm, ws(), kw('select')])
        assert st.get_type() == 'SELECT'

    def test_get_type_unknown(self):
        st = Statement([name('foo')])
        assert st.get_type() == 'UNKNOWN'

    def test_get_type_cte(self):
        st = Statement([
            Token(T.Keyword.CTE, 'WITH'),
            ws(),
            Identifier([name('x')]),
            ws(),
            kw('select'),
            ws(),
            name('foo'),
        ])
        assert st.get_type() == 'SELECT'


# ---------------------------------------------------------------------------
# Misc classes
# ---------------------------------------------------------------------------

class TestMiscClasses:
    def test_typed_literal_markers(self):
        assert TypedLiteral.M_OPEN == [(T.Name.Builtin, None), (T.Keyword, "TIMESTAMP")]
        assert TypedLiteral.M_CLOSE == (T.String.Single, None)
        assert TypedLiteral.M_EXTEND == (
            T.Keyword, ("DAY", "HOUR", "MINUTE", "MONTH", "SECOND", "YEAR"))

    def test_parenthesis_groupable_tokens(self):
        a = Token(T.Name, 'a')
        p = Parenthesis([punct('('), a, punct(')')])
        assert p._groupable_tokens == [a]
        assert Parenthesis.M_OPEN == (T.Punctuation, '(')
        assert Parenthesis.M_CLOSE == (T.Punctuation, ')')

    def test_square_brackets_groupable_tokens(self):
        a = Token(T.Name, 'a')
        s = SquareBrackets([punct('['), a, punct(']')])
        assert s._groupable_tokens == [a]

    def test_assignment(self):
        assert issubclass(Assignment, TokenList)

    def test_if_markers(self):
        assert If.M_OPEN == (T.Keyword, 'IF')
        assert If.M_CLOSE == (T.Keyword, 'END IF')

    def test_for_markers(self):
        assert For.M_OPEN == (T.Keyword, ('FOR', 'FOREACH'))
        assert For.M_CLOSE == (T.Keyword, 'END LOOP')

    def test_comparison_left_right(self):
        a = Token(T.Name, 'a')
        op = Token(T.Operator, '=')
        b = Token(T.Name, 'b')
        c = Comparison([a, op, b])
        assert c.left is a
        assert c.right is b

    def test_comment_multiline_true(self):
        c = Comment([Token(T.Comment.Multiline, '/* x */')])
        assert c.is_multiline() is True

    def test_comment_multiline_false(self):
        c = Comment([Token(T.Comment.Single, '-- x')])
        assert c.is_multiline() is False

    def test_comment_multiline_empty(self):
        assert Comment([]).is_multiline() == []  # falsy empty list

    def test_where_markers(self):
        assert Where.M_OPEN == (T.Keyword, 'WHERE')
        assert 'ORDER BY' in Where.M_CLOSE[1]

    def test_over_markers(self):
        assert Over.M_OPEN == (T.Keyword, 'OVER')

    def test_having_markers(self):
        assert Having.M_OPEN == (T.Keyword, 'HAVING')
        assert Having.M_CLOSE == (T.Keyword, ('ORDER BY', 'LIMIT'))

    def test_begin_markers(self):
        assert Begin.M_OPEN == (T.Keyword, 'BEGIN')
        assert Begin.M_CLOSE == (T.Keyword, 'END')

    def test_marker_classes_are_tokenlists(self):
        for cls in (TypedLiteral, Parenthesis, SquareBrackets, Assignment,
                    If, For, Comparison, Comment, Where, Over, Having,
                    Case, Function, Begin, Operation, Values, Command):
            assert issubclass(cls, TokenList)


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

class TestCase:
    def test_get_cases_simple(self):
        case = Case([
            kw('CASE'),
            ws(),
            kw('WHEN'),
            ws(),
            name('a'),
            ws(),
            kw('THEN'),
            ws(),
            Token(T.Literal, '1'),
            ws(),
            kw('END'),
        ])
        cases = case.get_cases()
        assert len(cases) == 1
        cond, val = cases[0]
        assert cond == [Token(T.Whitespace, ' '), name('a'), Token(T.Whitespace, ' ')]
        # easier: check values string
        assert ''.join(str(t) for t in cond).strip() == 'a'
        assert ''.join(str(t) for t in val).strip() == '1'

    def test_get_cases_with_else(self):
        case = Case([
            kw('CASE'),
            kw('WHEN'),
            name('a'),
            kw('THEN'),
            Token(T.Literal, '1'),
            kw('ELSE'),
            Token(T.Literal, '2'),
            kw('END'),
        ])
        cases = case.get_cases()
        assert len(cases) == 2
        assert cases[0][0] is not None
        assert cases[1][0] is None

    def test_get_cases_skip_ws(self):
        case = Case([
            kw('CASE'),
            ws(),
            kw('WHEN'),
            ws(),
            name('a'),
            ws(),
            kw('THEN'),
            ws(),
            Token(T.Literal, '1'),
            kw('END'),
        ])
        cases = case.get_cases(skip_ws=True)
        assert len(cases) == 1
        cond, val = cases[0]
        # when skipping ws, whitespace between WHEN and then not appended
        assert [t.value for t in cond] == ['a']
        assert [t.value for t in val] == ['1']

    def test_get_cases_empty(self):
        assert Case([]).get_cases() == []


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

class TestFunction:
    def test_get_parameters_identifier_list(self):
        il = IdentifierList([
            Identifier([name('a')]),
            punct(','),
            ws(),
            Identifier([name('b')]),
        ])
        paren = Parenthesis([punct('('), il, punct(')')])
        func = Function([name('f'), paren])
        result = list(func.get_parameters())
        # IdentifierList.get_identifiers returns tokens
        assert len(result) == 2

    def test_get_parameters_simple(self):
        a = Identifier([name('a')])
        paren = Parenthesis([punct('('), a, punct(')')])
        func = Function([name('f'), paren])
        result = func.get_parameters()
        assert result == [a]

    def test_get_window_none(self):
        func = Function([name('f')])
        assert func.get_window() is None

    def test_get_window_present(self):
        inner = Token(T.Name, 'w')
        over = Over([kw('OVER'), ws(), inner])
        func = Function([name('f'), ws(), over])
        assert func.get_window() is inner

    def test_function_get_alias(self):
        func = Function([name('f'), ws(), kw('AS'), ws(), name('x')])
        assert func.get_alias() == 'x'
