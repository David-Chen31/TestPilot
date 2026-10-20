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
# Token
# ---------------------------------------------------------------------------

def test_token_basic_attributes():
    tok = Token(T.Keyword, "select")
    assert tok.value == "select"
    assert tok.ttype == T.Keyword
    assert tok.parent is None
    assert tok.is_group is False
    assert tok.is_keyword is True
    assert tok.is_whitespace is False
    assert tok.is_newline is False
    assert tok.normalized == "SELECT"


def test_token_non_keyword_normalized():
    tok = Token(T.Name, "Foo")
    assert tok.normalized == "Foo"
    assert tok.is_keyword is False


def test_token_value_coerced_to_str():
    tok = Token(T.Literal.Number.Integer, 123)
    assert tok.value == "123"
    assert isinstance(tok.value, str)


def test_token_whitespace_newline_flags():
    ws = Token(T.Whitespace, " ")
    assert ws.is_whitespace is True
    assert ws.is_newline is False
    nl = Token(T.Newline, "\n")
    assert nl.is_newline is True


def test_token_str_and_repr():
    tok = Token(T.String.Single, "'abc'")
    assert str(tok) == "'abc'"
    r = repr(tok)
    assert r.startswith("<Single ")
    assert r.startswith("<Single \"")


def test_token_repr_truncates_long_value():
    tok = Token(T.Name, "abcdefghijk")
    r = repr(tok)
    assert "abcdef..." in r


def test_token_repr_normalizes_whitespace():
    tok = Token(T.Name, "a  b")
    assert "a b" in repr(tok)


def test_token_flatten_yields_self():
    tok = Token(T.Name, "x")
    assert list(tok.flatten()) == [tok]


def test_token_match_ttype_only():
    tok = Token(T.Keyword, "SELECT")
    assert tok.match(T.Keyword, None) is True
    assert tok.match(T.Name, None) is False


def test_token_match_values_case_insensitive_for_keyword():
    tok = Token(T.Keyword, "select")
    assert tok.match(T.Keyword, "SELECT") is True
    assert tok.match(T.Keyword, ["select", "other"]) is True
    assert tok.match(T.Keyword, "insert") is False


def test_token_match_values_case_sensitive_for_non_keyword():
    tok = Token(T.Name, "Foo")
    assert tok.match(T.Name, "Foo") is True
    assert tok.match(T.Name, "foo") is False


def test_token_match_regex_keyword_ignorecase():
    tok = Token(T.Keyword, "SELECT")
    assert tok.match(T.Keyword, "sel.*", regex=True) is True
    assert tok.match(T.Keyword, "^sel$", regex=True) is False


def test_token_match_regex_non_keyword_case_sensitive():
    tok = Token(T.Name, "FooBar")
    assert tok.match(T.Name, "^Foo", regex=True) is True
    assert tok.match(T.Name, "^foo", regex=True) is False


def test_token_within():
    inner = Token(T.Name, "x")
    grp = TokenList([inner])
    assert inner.within(TokenList) is True
    assert inner.within(Function) is False
    assert inner.is_child_of(grp) is True
    assert inner.has_ancestor(grp) is True


def test_token_is_child_of_false():
    tok = Token(T.Name, "x")
    other = TokenList([Token(T.Name, "y")])
    assert tok.is_child_of(other) is False


def test_token_has_ancestor_nested():
    a = Token(T.Name, "a")
    mid = TokenList([a])
    top = TokenList([mid])
    assert a.has_ancestor(top) is True
    assert a.has_ancestor(Token(T.Name, "z")) is False


# ---------------------------------------------------------------------------
# TokenList
# ---------------------------------------------------------------------------

def test_tokenlist_init_and_value():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a, b])
    assert tl.value == "ab"
    assert tl.is_group is True
    assert a.parent is tl
    assert b.parent is tl
    assert list(tl) == [a, b]
    assert tl[0] is a


def test_tokenlist_empty():
    tl = TokenList()
    assert tl.tokens == []
    assert str(tl) == ""


def test_tokenlist_str_flattens():
    inner = Token(T.Name, "x")
    grp = TokenList([inner])
    outer = TokenList([grp])
    assert str(outer) == "x"


def test_tokenlist_flatten_recursive():
    a = Token(T.Name, "a")
    inner = TokenList([a])
    b = Token(T.Name, "b")
    top = TokenList([inner, b])
    assert list(top.flatten()) == [a, b]


def test_tokenlist_get_sublists():
    a = Token(T.Name, "a")
    grp = TokenList([a])
    b = Token(T.Name, "b")
    top = TokenList([grp, b])
    assert list(top.get_sublists()) == [grp]


def test_tokenlist_get_token_at_offset():
    a = Token(T.Name, "abc")
    b = Token(T.Name, "de")
    tl = TokenList([a, b])
    assert tl.get_token_at_offset(0) is a
    assert tl.get_token_at_offset(2) is a
    assert tl.get_token_at_offset(3) is b
    assert tl.get_token_at_offset(4) is b
    assert tl.get_token_at_offset(5) is None


def test_tokenlist_token_first_skip_ws():
    ws = Token(T.Whitespace, " ")
    a = Token(T.Name, "a")
    tl = TokenList([ws, a])
    assert tl.token_first() is a
    assert tl.token_first(skip_ws=False) is ws


def test_tokenlist_token_first_skip_cm():
    cm = Comment([Token(T.Comment.Single, "--x")])
    a = Token(T.Name, "a")
    tl = TokenList([cm, a])
    assert tl.token_first(skip_cm=True) is a


def test_tokenlist_token_first_empty():
    tl = TokenList()
    assert tl.token_first() is None


def test_tokenlist_token_next():
    a = Token(T.Name, "a")
    ws = Token(T.Whitespace, " ")
    b = Token(T.Name, "b")
    tl = TokenList([a, ws, b])
    idx, tok = tl.token_next(0)
    assert idx == 2
    assert tok is b
    idx, tok = tl.token_next(0, skip_ws=False)
    assert idx == 1
    assert tok is ws


def test_tokenlist_token_next_none_idx():
    tl = TokenList([Token(T.Name, "a")])
    assert tl.token_next(None) == (None, None)


def test_tokenlist_token_next_end():
    a = Token(T.Name, "a")
    tl = TokenList([a])
    assert tl.token_next(0) == (None, None)


def test_tokenlist_token_prev():
    a = Token(T.Name, "a")
    ws = Token(T.Whitespace, " ")
    b = Token(T.Name, "b")
    tl = TokenList([a, ws, b])
    idx, tok = tl.token_prev(2)
    assert idx == 0
    assert tok is a


def test_tokenlist_token_prev_none():
    a = Token(T.Name, "a")
    tl = TokenList([a])
    assert tl.token_prev(0) == (None, None)


def test_tokenlist_token_next_by():
    ws = Token(T.Whitespace, " ")
    kw = Token(T.Keyword, "SELECT")
    tl = TokenList([ws, kw])
    idx, tok = tl.token_next_by(t=T.Keyword)
    assert idx == 1
    assert tok is kw


def test_tokenlist_token_next_by_match():
    a = Token(T.Name, "a")
    dot = Token(T.Punctuation, ".")
    tl = TokenList([a, dot])
    idx, tok = tl.token_next_by(m=(T.Punctuation, '.'))
    assert idx == 1
    assert tok is dot


def test_tokenlist_token_next_by_instance():
    a = Token(T.Name, "a")
    grp = Parenthesis([Token(T.Punctuation, "("), Token(T.Punctuation, ")")])
    tl = TokenList([a, grp])
    idx, tok = tl.token_next_by(i=Parenthesis)
    assert idx == 1
    assert tok is grp


def test_tokenlist_token_next_by_none():
    tl = TokenList([Token(T.Name, "a")])
    assert tl.token_next_by(t=T.Keyword) == (None, None)


def test_tokenlist_token_index():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a, b])
    assert tl.token_index(b) == 1
    assert tl.token_index(a, start=0) == 0
    assert tl.token_index(b, start=a) == 1


def test_tokenlist_group_tokens():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    c = Token(T.Name, "c")
    tl = TokenList([a, b, c])
    grp = tl.group_tokens(TokenList, 0, 1)
    assert isinstance(grp, TokenList)
    assert grp.tokens == [a, b]
    assert tl.tokens == [grp, c]
    assert a.parent is grp
    assert b.parent is grp
    assert grp.parent is tl


def test_tokenlist_group_tokens_exclude_end():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a, b])
    grp = tl.group_tokens(TokenList, 0, 0)
    assert grp.tokens == [a]
    assert tl.tokens == [grp, b]


def test_tokenlist_group_tokens_extend():
    a = Token(T.Name, "a")
    existing = TokenList([a])
    b = Token(T.Name, "b")
    tl = TokenList([existing, b])
    grp = tl.group_tokens(TokenList, 0, 1, extend=True)
    assert grp is existing
    assert grp.tokens == [a, b]
    assert tl.tokens == [grp]
    assert b.parent is grp


def test_tokenlist_insert_before_by_token():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a])
    tl.insert_before(a, b)
    assert tl.tokens == [b, a]
    assert b.parent is tl


def test_tokenlist_insert_before_by_index():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a])
    tl.insert_before(1, b)
    assert tl.tokens == [a, b]


def test_tokenlist_insert_after():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a])
    tl.insert_after(0, b)
    assert tl.tokens == [a, b]


def test_tokenlist_insert_after_skip_ws():
    a = Token(T.Name, "a")
    ws = Token(T.Whitespace, " ")
    b = Token(T.Name, "b")
    c = Token(T.Name, "c")
    tl = TokenList([a, ws, b])
    tl.insert_after(0, c)
    assert tl.tokens == [a, c, ws, b]


def test_tokenlist_insert_after_append():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a])
    tl.insert_after(a, b)
    assert tl.tokens == [a, b]


def test_tokenlist_default_names():
    tl = TokenList()
    assert tl.get_alias() is None
    assert tl.get_real_name() is None
    assert tl.has_alias() is False
    assert tl.get_name() is None


def test_tokenlist_get_parent_name():
    a = Token(T.Name, '"tbl"')
    dot = Token(T.Punctuation, ".")
    b = Token(T.Name, "col")
    tl = TokenList([a, dot, b])
    assert tl.get_parent_name() == "tbl"


def test_tokenlist_get_parent_name_none():
    tl = TokenList([Token(T.Name, "col")])
    assert tl.get_parent_name() is None


def test_tokenlist_repr_name_value():
    tl = TokenList([Token(T.Name, "abc")])
    assert tl._get_repr_name() == "TokenList"
    assert tl._get_repr_value() == "abc"


def test_tokenlist_pprint_tree(capsys):
    a = Token(T.Name, "a")
    tl = TokenList([a])
    tl._pprint_tree()
    out = capsys.readouterr().out
    assert "0 Name 'a'" in out


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

def test_statement_get_type_dml():
    st = Statement([Token(T.Keyword.DML, "select")])
    assert st.get_type() == "SELECT"


def test_statement_get_type_ddl():
    st = Statement([Token(T.Keyword.DDL, "create")])
    assert st.get_type() == "CREATE"


def test_statement_get_type_skips_ws():
    st = Statement([
        Token(T.Whitespace, " "),
        Token(T.Keyword.DML, "insert"),
    ])
    assert st.get_type() == "INSERT"


def test_statement_get_type_unknown():
    st = Statement([Token(T.Name, "foo")])
    assert st.get_type() == "UNKNOWN"


def test_statement_get_type_empty():
    assert Statement().get_type() == "UNKNOWN"


def test_statement_get_type_cte():
    with_kw = Token(T.Keyword.CTE, "WITH")
    ident = Identifier([Token(T.Name, "cte")])
    select_kw = Token(T.Keyword.DML, "SELECT")
    st = Statement([with_kw, Token(T.Whitespace, " "), ident,
                    Token(T.Whitespace, " "), select_kw])
    assert st.get_type() == "SELECT"


# ---------------------------------------------------------------------------
# NameAliasMixin / Identifier
# ---------------------------------------------------------------------------

def test_identifier_real_name_simple():
    ident = Identifier([Token(T.Name, "col")])
    assert ident.get_real_name() == "col"


def test_identifier_real_name_dotted():
    ident = Identifier([
        Token(T.Name, "a"),
        Token(T.Punctuation, "."),
        Token(T.Name, "b"),
        Token(T.Punctuation, "."),
        Token(T.Name, "c"),
    ])
    assert ident.get_real_name() == "c"


def test_identifier_real_name_quoted():
    ident = Identifier([Token(T.Name, '"col"')])
    assert ident.get_real_name() == "col"


def test_identifier_get_alias_as():
    ident = Identifier([
        Token(T.Name, "col"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "AS"),
        Token(T.Whitespace, " "),
        Token(T.Name, "alias"),
    ])
    assert ident.get_alias() == "alias"
    assert ident.has_alias() is True


def test_identifier_get_alias_implicit():
    ident = Identifier([
        Token(T.Name, "col"),
        Token(T.Whitespace, " "),
        Token(T.Name, "alias"),
    ])
    assert ident.get_alias() == "alias"


def test_identifier_no_alias():
    ident = Identifier([Token(T.Name, "col")])
    assert ident.get_alias() is None
    assert ident.has_alias() is False


def test_identifier_get_name_prefers_alias():
    ident = Identifier([
        Token(T.Name, "col"),
        Token(T.Whitespace, " "),
        Token(T.Name, "alias"),
    ])
    assert ident.get_name() == "alias"
    assert ident.get_real_name() == "col"


def test_identifier_is_wildcard():
    ident = Identifier([
        Token(T.Name, "t"),
        Token(T.Punctuation, "."),
        Token(T.Wildcard, "*"),
    ])
    assert ident.is_wildcard() is True


def test_identifier_not_wildcard():
    ident = Identifier([Token(T.Name, "col")])
    assert ident.is_wildcard() is False


def test_identifier_get_typecast():
    ident = Identifier([
        Token(T.Name, "col"),
        Token(T.Punctuation, "::"),
        Token(T.Name.Builtin, "int"),
    ])
    assert ident.get_typecast() == "int"


def test_identifier_get_typecast_none():
    ident = Identifier([Token(T.Name, "col")])
    assert ident.get_typecast() is None


def test_identifier_get_ordering():
    ident = Identifier([
        Token(T.Name, "col"),
        Token(T.Whitespace, " "),
        Token(T.Keyword.Order, "desc"),
    ])
    assert ident.get_ordering() == "DESC"


def test_identifier_get_ordering_none():
    ident = Identifier([Token(T.Name, "col")])
    assert ident.get_ordering() is None


def test_identifier_get_array_indices():
    inner = [Token(T.Name, "1")]
    sq = SquareBrackets([
        Token(T.Punctuation, "["),
        inner[0],
        Token(T.Punctuation, "]"),
    ])
    ident = Identifier([Token(T.Name, "col"), sq])
    result = list(ident.get_array_indices())
    assert len(result) == 1
    assert result[0] == [inner[0]]


# ---------------------------------------------------------------------------
# IdentifierList
# ---------------------------------------------------------------------------

def test_identifierlist_get_identifiers():
    a = Identifier([Token(T.Name, "a")])
    ws = Token(T.Whitespace, " ")
    comma = Token(T.Punctuation, ",")
    b = Identifier([Token(T.Name, "b")])
    il = IdentifierList([a, ws, comma, ws, b])
    assert list(il.get_identifiers()) == [a, b]


# ---------------------------------------------------------------------------
# Simple token subclasses
# ---------------------------------------------------------------------------

def test_parenthesis_groupable_tokens():
    p_open = Token(T.Punctuation, "(")
    inner = Token(T.Name, "x")
    p_close = Token(T.Punctuation, ")")
    p = Parenthesis([p_open, inner, p_close])
    assert p._groupable_tokens == [inner]


def test_squarebrackets_groupable_tokens():
    b_open = Token(T.Punctuation, "[")
    inner = Token(T.Name, "x")
    b_close = Token(T.Punctuation, "]")
    sb = SquareBrackets([b_open, inner, b_close])
    assert sb._groupable_tokens == [inner]


def test_typed_literal_markers():
    assert TypedLiteral.M_OPEN == [(T.Name.Builtin, None), (T.Keyword, "TIMESTAMP")]
    assert TypedLiteral.M_CLOSE == (T.String.Single, None)


@pytest.mark.parametrize("cls", [
    Assignment, If, For, Where, Over, Having, Case, Begin, Operation,
    Values, Command,
])
def test_tokenlist_subclasses_are_groups(cls):
    tok = cls([Token(T.Name, "x")])
    assert isinstance(tok, TokenList)
    assert tok.is_group is True


def test_comparison_left_right():
    left = Token(T.Name, "a")
    op = Token(T.Operator, "=")
    right = Token(T.Name, "b")
    cmp = Comparison([left, op, right])
    assert cmp.left is left
    assert cmp.right is right


def test_comment_is_multiline_true():
    c = Comment([Token(T.Comment.Multiline, "/*x*/")])
    assert c.is_multiline() is True


def test_comment_is_multiline_false():
    c = Comment([Token(T.Comment.Single, "--x")])
    assert c.is_multiline() is False


def test_comment_is_multiline_empty():
    c = Comment()
    assert not c.is_multiline()


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

def test_case_get_cases_condition_value_else():
    tokens = [
        Token(T.Keyword, "CASE"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "WHEN"),
        Token(T.Whitespace, " "),
        Token(T.Name, "cond"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "THEN"),
        Token(T.Whitespace, " "),
        Token(T.Name, "val"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "ELSE"),
        Token(T.Whitespace, " "),
        Token(T.Name, "other"),
        Token(T.Keyword, "END"),
    ]
    c = Case(tokens)
    result = c.get_cases()
    # first tuple: condition / value, second: else (None, [..])
    assert result[0][0][0].value == "cond"
    assert result[0][1][-1].value == "val"
    assert result[1][0] is None
    assert result[1][1][-1].value == "other"


def test_case_get_cases_skip_ws():
    tokens = [
        Token(T.Keyword, "CASE"),
        Token(T.Keyword, "WHEN"),
        Token(T.Whitespace, " "),
        Token(T.Name, "cond"),
        Token(T.Keyword, "THEN"),
        Token(T.Name, "val"),
        Token(T.Keyword, "END"),
    ]
    c = Case(tokens)
    result = c.get_cases(skip_ws=True)
    assert len(result) == 1
    assert [t.value for t in result[0][0]] == ["cond"]
    assert [t.value for t in result[0][1]] == ["val"]


def test_case_get_cases_empty():
    c = Case()
    assert c.get_cases() == []


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

def test_function_get_parameters_identifiers():
    a = Identifier([Token(T.Name, "a")])
    comma = Token(T.Punctuation, ",")
    b = Identifier([Token(T.Name, "b")])
    il = IdentifierList([a, comma, b])
    p = Parenthesis([
        Token(T.Punctuation, "("),
        il,
        Token(T.Punctuation, ")"),
    ])
    fn = Function([Token(T.Name, "fn"), p])
    assert list(fn.get_parameters()) == [a, b]


def test_function_get_parameters_simple():
    ident = Identifier([Token(T.Name, "arg")])
    p = Parenthesis([
        Token(T.Punctuation, "("),
        ident,
        Token(T.Punctuation, ")"),
    ])
    fn = Function([Token(T.Name, "fn"), p])
    assert fn.get_parameters() == [ident]


def test_function_get_parameters_literal():
    lit = Token(T.Literal.Number.Integer, "1")
    p = Parenthesis([
        Token(T.Punctuation, "("),
        lit,
        Token(T.Punctuation, ")"),
    ])
    fn = Function([Token(T.Name, "fn"), p])
    assert fn.get_parameters() == [lit]


def test_function_get_window_none():
    fn = Function([Token(T.Name, "fn")])
    assert fn.get_window() is None


def test_function_get_window():
    over = Over([
        Token(T.Keyword, "OVER"),
        Token(T.Whitespace, " "),
        Identifier([Token(T.Name, "w")]),
    ])
    fn = Function([Token(T.Name, "fn"), Token(T.Whitespace, " "), over])
    assert fn.get_window() is over.tokens[-1]


def test_function_real_name_and_alias():
    fn = Function([
        Token(T.Name, "fn"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "AS"),
        Token(T.Whitespace, " "),
        Token(T.Name, "alias"),
    ])
    assert fn.get_real_name() == "fn"
    assert fn.get_alias() == "alias"


# ---------------------------------------------------------------------------
# NameAliasMixin direct usage
# ---------------------------------------------------------------------------

def test_namealiasmixin_real_name_none_when_no_name():
    class Dummy(NameAliasMixin, TokenList):
        pass

    d = Dummy([Token(T.Whitespace, " ")])
    assert d.get_real_name() is None


def test_namealiasmixin_alias_none():
    class Dummy(NameAliasMixin, TokenList):
        pass

    d = Dummy([Token(T.Name, "x")])
    assert d.get_alias() is None
