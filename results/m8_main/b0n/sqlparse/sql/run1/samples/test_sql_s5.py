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
    assert tok.ttype is T.Keyword
    assert tok.parent is None
    assert tok.is_group is False
    assert tok.is_keyword is True
    assert tok.is_whitespace is False
    assert tok.is_newline is False
    assert tok.normalized == "SELECT"


def test_token_non_keyword_normalized_unchanged():
    tok = Token(T.Name, "Foo")
    assert tok.is_keyword is False
    assert tok.normalized == "Foo"


def test_token_value_str_conversion():
    tok = Token(T.Number, 42)
    assert tok.value == "42"
    assert isinstance(tok.value, str)


def test_token_whitespace_and_newline_flags():
    ws = Token(T.Whitespace, " ")
    nl = Token(T.Newline, "\n")
    assert ws.is_whitespace is True
    assert nl.is_newline is True


def test_token_str():
    tok = Token(T.String, "abc")
    assert str(tok) == "abc"


def test_token_repr_name_value():
    tok = Token(T.Keyword, "SELECT")
    assert tok._get_repr_name() == "Keyword"
    assert tok._get_repr_value() == "SELECT"
    # long value truncation
    long_tok = Token(T.String, "abcdefghij")
    assert long_tok._get_repr_value() == "abcdef..."


def test_token_repr_quotes():
    tok = Token(T.String, "'abc'")
    r = repr(tok)
    assert r.startswith('<String "')
    tok2 = Token(T.String, "abc")
    assert repr(tok2).startswith("<String '")


def test_token_repr_whitespace_collapse():
    tok = Token(T.String, "a   b")
    assert tok._get_repr_value() == "a b"


def test_token_flatten_yields_self():
    tok = Token(T.Name, "x")
    assert list(tok.flatten()) == [tok]


def test_token_match_wrong_ttype():
    tok = Token(T.Name, "x")
    assert tok.match(T.Keyword, "x") is False


def test_token_match_none_values():
    tok = Token(T.Name, "x")
    assert tok.match(T.Name, None) is True


def test_token_match_string_value():
    tok = Token(T.Name, "abc")
    assert tok.match(T.Name, "abc") is True
    assert tok.match(T.Name, "xyz") is False


def test_token_match_keyword_case_insensitive():
    tok = Token(T.Keyword, "select")
    assert tok.match(T.Keyword, "SELECT") is True
    assert tok.match(T.Keyword, "select") is True


def test_token_match_regex():
    tok = Token(T.Name, "abcdef")
    assert tok.match(T.Name, "abc", regex=True) is True
    assert tok.match(T.Name, "xyz", regex=True) is False


def test_token_match_regex_keyword_ignorecase():
    tok = Token(T.Keyword, "SELECT")
    assert tok.match(T.Keyword, "sel", regex=True) is True


def test_token_identity_of_ttype():
    # match uses `is` for ttype
    tok = Token(T.Name, "x")
    assert tok.match(T.Name, "x") is True


def test_token_slots_no_dict():
    tok = Token(T.Name, "x")
    assert not hasattr(tok, "__dict__")


# ---------------------------------------------------------------------------
# TokenList
# ---------------------------------------------------------------------------

def test_tokenlist_init_sets_parents():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a, b])
    assert tl.is_group is True
    assert tl.ttype is None
    assert tl.value == "ab"
    assert a.parent is tl
    assert b.parent is tl


def test_tokenlist_empty():
    tl = TokenList()
    assert tl.tokens == []
    assert tl.value == ""
    assert tl.is_group is True


def test_tokenlist_none():
    tl = TokenList(None)
    assert tl.tokens == []


def test_tokenlist_iter_and_getitem():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a, b])
    assert list(tl) == [a, b]
    assert tl[0] is a
    assert tl[-1] is b


def test_tokenlist_str_flattens():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    inner = TokenList([a, b])
    outer = TokenList([inner, Token(T.Name, "c")])
    assert str(outer) == "abc"


def test_tokenlist_flatten():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    inner = TokenList([a, b])
    c = Token(T.Name, "c")
    outer = TokenList([inner, c])
    assert list(outer.flatten()) == [a, b, c]


def test_tokenlist_get_sublists():
    a = Token(T.Name, "a")
    inner = TokenList([a])
    c = Token(T.Name, "c")
    outer = TokenList([inner, c])
    assert list(outer.get_sublists()) == [inner]


def test_tokenlist_repr_name():
    tl = TokenList()
    assert tl._get_repr_name() == "TokenList"


def test_tokenlist_token_first():
    ws = Token(T.Whitespace, " ")
    a = Token(T.Name, "a")
    tl = TokenList([ws, a])
    assert tl.token_first() is a


def test_tokenlist_token_first_skip_ws_false():
    ws = Token(T.Whitespace, " ")
    a = Token(T.Name, "a")
    tl = TokenList([ws, a])
    assert tl.token_first(skip_ws=False) is ws


def test_tokenlist_token_first_skip_cm():
    cm_tok = Token(T.Comment.Single, "-- c")
    cm = Comment([cm_tok])
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


def test_tokenlist_token_next_none_idx():
    tl = TokenList([Token(T.Name, "a")])
    assert tl.token_next(None) == (None, None)


def test_tokenlist_token_next_no_match():
    tl = TokenList([Token(T.Name, "a")])
    assert tl.token_next(0) == (None, None)


def test_tokenlist_token_prev():
    a = Token(T.Name, "a")
    ws = Token(T.Whitespace, " ")
    b = Token(T.Name, "b")
    tl = TokenList([a, ws, b])
    idx, tok = tl.token_prev(2)
    assert idx == 0
    assert tok is a


def test_tokenlist_token_next_by():
    a = Token(T.Name, "a")
    b = Token(T.Keyword, "SELECT")
    tl = TokenList([a, b])
    idx, tok = tl.token_next_by(t=T.Keyword)
    assert idx == 1
    assert tok is b


def test_tokenlist_token_next_by_instance():
    a = Token(T.Name, "a")
    p = Parenthesis([Token(T.Punctuation, "("), Token(T.Punctuation, ")")])
    tl = TokenList([a, p])
    idx, tok = tl.token_next_by(i=Parenthesis)
    assert idx == 1
    assert tok is p


def test_tokenlist_token_next_by_match():
    a = Token(T.Name, "a")
    b = Token(T.Punctuation, ".")
    tl = TokenList([a, b])
    idx, tok = tl.token_next_by(m=(T.Punctuation, "."))
    assert idx == 1
    assert tok is b


def test_tokenlist_token_matching():
    a = Token(T.Name, "a")
    b = Token(T.Keyword, "SELECT")
    tl = TokenList([a, b])
    result = tl.token_matching(lambda tk: tk.is_keyword, 0)
    assert result is b


def test_tokenlist_token_not_matching():
    a = Token(T.Keyword, "SELECT")
    b = Token(T.Name, "a")
    tl = TokenList([a, b])
    idx, tok = tl.token_not_matching(lambda tk: tk.is_keyword, 0)
    assert idx == 1
    assert tok is b


def test_tokenlist_token_index():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    tl = TokenList([a, b])
    assert tl.token_index(b) == 1


def test_tokenlist_token_index_with_token_start():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    c = Token(T.Name, "c")
    tl = TokenList([a, b, c])
    assert tl.token_index(c, start=b) == 2


def test_tokenlist_get_token_at_offset():
    a = Token(T.Name, "ab")
    b = Token(T.Name, "cd")
    tl = TokenList([a, b])
    assert tl.get_token_at_offset(0) is a
    assert tl.get_token_at_offset(1) is a
    assert tl.get_token_at_offset(2) is b
    assert tl.get_token_at_offset(3) is b
    assert tl.get_token_at_offset(4) is None


def test_tokenlist_group_tokens():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    c = Token(T.Name, "c")
    tl = TokenList([a, b, c])
    grp = tl.group_tokens(TokenList, 0, 1)
    assert isinstance(grp, TokenList)
    assert grp.tokens == [a, b]
    assert tl.tokens == [grp, c]
    assert grp.parent is tl


def test_tokenlist_group_tokens_no_include_end():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    c = Token(T.Name, "c")
    tl = TokenList([a, b, c])
    grp = tl.group_tokens(TokenList, 0, 1, include_end=False)
    assert grp.tokens == [a]
    assert tl.tokens == [grp, b, c]


def test_tokenlist_group_tokens_extend():
    a = Token(T.Name, "a")
    b = Token(T.Name, "b")
    c = Token(T.Name, "c")
    start = TokenList([a])
    tl = TokenList([start, b, c])
    grp = tl.group_tokens(TokenList, 0, 1, extend=True)
    assert grp is start
    assert grp.tokens == [a, b]
    assert tl.tokens == [start, c]


def test_tokenlist_insert_before():
    a = Token(T.Name, "a")
    tl = TokenList([a])
    new = Token(T.Name, "new")
    tl.insert_before(a, new)
    assert tl.tokens == [new, a]
    assert new.parent is tl


def test_tokenlist_insert_before_int():
    a = Token(T.Name, "a")
    tl = TokenList([a])
    new = Token(T.Name, "new")
    tl.insert_before(0, new)
    assert tl.tokens == [new, a]


def test_tokenlist_insert_after():
    a = Token(T.Name, "a")
    tl = TokenList([a])
    new = Token(T.Name, "new")
    tl.insert_after(a, new)
    assert tl.tokens == [a, new]
    assert new.parent is tl


def test_tokenlist_insert_after_end():
    a = Token(T.Name, "a")
    tl = TokenList([a])
    new = Token(T.Name, "new")
    tl.insert_after(a, new, skip_ws=False)
    assert tl.tokens == [a, new]


def test_tokenlist_get_alias_real_name_none():
    tl = TokenList()
    assert tl.get_alias() is None
    assert tl.get_real_name() is None
    assert tl.get_name() is None
    assert tl.has_alias() is False


def test_tokenlist_get_parent_name():
    a = Token(T.Name, "tbl")
    dot = Token(T.Punctuation, ".")
    b = Token(T.Name, "col")
    tl = TokenList([a, dot, b])
    assert tl.get_parent_name() == "tbl"


def test_tokenlist_get_parent_name_none():
    tl = TokenList([Token(T.Name, "x")])
    assert tl.get_parent_name() is None


def test_tokenlist_get_parent_name_quoted():
    a = Token(T.String.Symbol, '"tbl"')
    dot = Token(T.Punctuation, ".")
    b = Token(T.Name, "col")
    tl = TokenList([a, dot, b])
    assert tl.get_parent_name() == "tbl"


def test_tokenlist_get_first_name():
    a = Token(T.Name, "abc")
    tl = TokenList([a])
    assert tl._get_first_name() == "abc"


def test_tokenlist_get_first_name_reverse():
    a = Token(T.Name, "abc")
    b = Token(T.Name, "xyz")
    tl = TokenList([a, b])
    assert tl._get_first_name(reverse=True) == "xyz"


def test_tokenlist_get_first_name_quoted():
    a = Token(T.Name, '"abc"')
    tl = TokenList([a])
    assert tl._get_first_name() == "abc"


def test_tokenlist_get_first_name_keywords():
    a = Token(T.Keyword, "AS")
    tl = TokenList([a])
    assert tl._get_first_name(keywords=True) == "AS"


def test_tokenlist_pprint_tree(capsys):
    a = Token(T.Name, "a")
    tl = TokenList([a])
    tl._pprint_tree()
    out = capsys.readouterr().out
    assert "Name" in out


def test_tokenlist_within():
    a = Token(T.Name, "a")
    inner = TokenList([a])
    outer = TokenList([inner])
    assert a.within(TokenList) is True


def test_tokenlist_is_child_of():
    a = Token(T.Name, "a")
    tl = TokenList([a])
    assert a.is_child_of(tl) is True
    assert a.is_child_of(TokenList()) is False


def test_tokenlist_has_ancestor():
    a = Token(T.Name, "a")
    inner = TokenList([a])
    outer = TokenList([inner])
    assert a.has_ancestor(outer) is True
    assert a.has_ancestor(TokenList()) is False


# ---------------------------------------------------------------------------
# NameAliasMixin / Identifier
# ---------------------------------------------------------------------------

def _ident(*tokens):
    return Identifier(list(tokens))


def test_identifier_get_real_name():
    a = Token(T.Name, "tbl1")
    dot = Token(T.Punctuation, ".")
    b = Token(T.Name, "col")
    ident = _ident(a, dot, b)
    assert ident.get_real_name() == "col"


def test_identifier_get_alias_as():
    a = Token(T.Name, "col")
    ws = Token(T.Whitespace, " ")
    as_kw = Token(T.Keyword, "AS")
    ws2 = Token(T.Whitespace, " ")
    alias = Token(T.Name, "al")
    ident = _ident(a, ws, as_kw, ws2, alias)
    assert ident.get_alias() == "al"
    assert ident.has_alias() is True


def test_identifier_get_alias_no_as():
    a = Token(T.Name, "col")
    ws = Token(T.Whitespace, " ")
    alias = Token(T.Name, "al")
    ident = _ident(a, ws, alias)
    assert ident.get_alias() == "al"


def test_identifier_get_alias_none():
    a = Token(T.Name, "col")
    ident = _ident(a)
    assert ident.get_alias() is None


def test_identifier_get_name_prefers_alias():
    a = Token(T.Name, "col")
    ws = Token(T.Whitespace, " ")
    alias = Token(T.Name, "al")
    ident = _ident(a, ws, alias)
    assert ident.get_name() == "al"


def test_identifier_is_wildcard():
    a = Token(T.Name, "tbl")
    dot = Token(T.Punctuation, ".")
    star = Token(T.Wildcard, "*")
    ident = _ident(a, dot, star)
    assert ident.is_wildcard() is True


def test_identifier_is_wildcard_false():
    ident = _ident(Token(T.Name, "tbl"))
    assert ident.is_wildcard() is False


def test_identifier_get_typecast():
    a = Token(T.Name, "col")
    marker = Token(T.Punctuation, "::")
    tc = Token(T.Name, "int")
    ident = _ident(a, marker, tc)
    assert ident.get_typecast() == "int"


def test_identifier_get_typecast_none():
    ident = _ident(Token(T.Name, "col"))
    assert ident.get_typecast() is None


def test_identifier_get_ordering():
    a = Token(T.Name, "col")
    ws = Token(T.Whitespace, " ")
    order = Token(T.Keyword.Order, "desc")
    ident = _ident(a, ws, order)
    assert ident.get_ordering() == "DESC"


def test_identifier_get_ordering_none():
    ident = _ident(Token(T.Name, "col"))
    assert ident.get_ordering() is None


def test_identifier_get_array_indices():
    a = Token(T.Name, "col")
    inner = [Token(T.Number.Integer, "1")]
    sq = SquareBrackets(
        [Token(T.Punctuation, "["), inner[0], Token(T.Punctuation, "]")]
    )
    ident = _ident(a, sq)
    indices = list(ident.get_array_indices())
    assert len(indices) == 1
    assert indices[0] == inner


def test_identifier_get_array_indices_empty():
    ident = _ident(Token(T.Name, "col"))
    assert list(ident.get_array_indices()) == []


# ---------------------------------------------------------------------------
# IdentifierList
# ---------------------------------------------------------------------------

def test_identifier_list_get_identifiers():
    a = Token(T.Name, "a")
    ws = Token(T.Whitespace, " ")
    comma = Token(T.Punctuation, ",")
    b = Token(T.Name, "b")
    il = IdentifierList([a, ws, comma, Token(T.Whitespace, " "), b])
    assert list(il.get_identifiers()) == [a, b]


# ---------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------

def test_statement_get_type_dml():
    kw = Token(T.Keyword.DML, "select")
    stmt = Statement([kw])
    assert stmt.get_type() == "SELECT"


def test_statement_get_type_ddl():
    kw = Token(T.Keyword.DDL, "create")
    stmt = Statement([kw])
    assert stmt.get_type() == "CREATE"


def test_statement_get_type_unknown_empty():
    stmt = Statement([])
    assert stmt.get_type() == "UNKNOWN"


def test_statement_get_type_unknown_ws():
    stmt = Statement([Token(T.Whitespace, " ")])
    assert stmt.get_type() == "UNKNOWN"


def test_statement_get_type_unknown_other():
    stmt = Statement([Token(T.Name, "foo")])
    assert stmt.get_type() == "UNKNOWN"


def test_statement_get_type_cte():
    with_kw = Token(T.Keyword.CTE, "with")
    ident = Identifier([Token(T.Name, "cte")])
    select = Token(T.Keyword.DML, "select")
    stmt = Statement([with_kw, Token(T.Whitespace, " "), ident,
                      Token(T.Whitespace, " "), select])
    assert stmt.get_type() == "SELECT"


def test_statement_get_type_cte_no_dml():
    with_kw = Token(T.Keyword.CTE, "with")
    ident = Identifier([Token(T.Name, "cte")])
    stmt = Statement([with_kw, Token(T.Whitespace, " "), ident])
    assert stmt.get_type() == "UNKNOWN"


def test_statement_get_type_skips_comments():
    cm = Comment([Token(T.Comment.Single, "-- hello")])
    select = Token(T.Keyword.DML, "select")
    stmt = Statement([cm, Token(T.Whitespace, " "), select])
    assert stmt.get_type() == "SELECT"


# ---------------------------------------------------------------------------
# TypedLiteral and subclasses with M_* attributes
# ---------------------------------------------------------------------------

def test_typed_literal_attrs():
    assert TypedLiteral.M_OPEN == [(T.Name.Builtin, None), (T.Keyword, "TIMESTAMP")]
    assert TypedLiteral.M_CLOSE == (T.String.Single, None)
    assert TypedLiteral.M_EXTEND == (
        T.Keyword, ("DAY", "HOUR", "MINUTE", "MONTH", "SECOND", "YEAR")
    )


def test_parenthesis_groupable_tokens():
    lp = Token(T.Punctuation, "(")
    a = Token(T.Name, "a")
    rp = Token(T.Punctuation, ")")
    p = Parenthesis([lp, a, rp])
    assert p._groupable_tokens == [a]


def test_square_brackets_groupable_tokens():
    lb = Token(T.Punctuation, "[")
    a = Token(T.Name, "a")
    rb = Token(T.Punctuation, "]")
    sq = SquareBrackets([lb, a, rb])
    assert sq._groupable_tokens == [a]


def test_comparison_properties():
    left = Token(T.Name, "a")
    op = Token(T.Comparison, "=")
    right = Token(T.Name, "b")
    comp = Comparison([left, op, right])
    assert comp.left is left
    assert comp.right is right


def test_comment_is_multiline_true():
    c = Comment([Token(T.Comment.Multiline, "/* x */")])
    assert c.is_multiline() is True


def test_comment_is_multiline_false():
    c = Comment([Token(T.Comment.Single, "-- x")])
    assert c.is_multiline() is False


def test_comment_is_multiline_empty():
    c = Comment([])
    assert c.is_multiline() == []


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

def test_case_get_cases():
    tokens = [
        Token(T.Keyword, "CASE"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "WHEN"),
        Token(T.Whitespace, " "),
        Token(T.Name, "cond"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "THEN"),
        Token(T.Whitespace, " "),
        Token(T.Number, "1"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "END"),
    ]
    case = Case(tokens)
    cases = case.get_cases()
    assert len(cases) == 1
    cond, val = cases[0]
    assert Token(T.Name, "cond") in cond
    assert Token(T.Number, "1") in val


def test_case_get_cases_else():
    tokens = [
        Token(T.Keyword, "CASE"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "WHEN"),
        Token(T.Whitespace, " "),
        Token(T.Name, "cond"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "THEN"),
        Token(T.Whitespace, " "),
        Token(T.Number, "1"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "ELSE"),
        Token(T.Whitespace, " "),
        Token(T.Number, "2"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "END"),
    ]
    case = Case(tokens)
    cases = case.get_cases()
    assert len(cases) == 2
    assert cases[1][0] is None
    assert cases[1][1][-1].value == "2"


def test_case_get_cases_skip_ws():
    tokens = [
        Token(T.Keyword, "CASE"),
        Token(T.Whitespace, " "),
        Token(T.Keyword, "WHEN"),
        Token(T.Whitespace, " "),
        Token(T.Name, "cond"),
    ]
    case = Case(tokens)
    cases = case.get_cases(skip_ws=True)
    assert len(cases) == 1


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

def test_function_get_parameters():
    fn_name = Token(T.Name, "fn")
    lp = Token(T.Punctuation, "(")
    a = Token(T.Name, "a")
    comma = Token(T.Punctuation, ",")
    b = Token(T.Name, "b")
    rp = Token(T.Punctuation, ")")
    fn = Function([fn_name, Parenthesis([lp, a, comma, b, rp])])
    params = fn.get_parameters()
    assert list(params) == [a, b]


def test_function_get_parameters_empty():
    fn_name = Token(T.Name, "fn")
    p = Parenthesis([Token(T.Punctuation, "("), Token(T.Punctuation, ")")])
    fn = Function([fn_name, p])
    assert fn.get_parameters() == []


def test_function_get_window_none():
    fn = Function([Token(T.Name, "fn")])
    assert fn.get_window() is None


def test_function_get_window():
    fn_name = Token(T.Name, "fn")
    p = Parenthesis([Token(T.Punctuation, "("), Token(T.Punctuation, ")")])
    over = Over([Token(T.Keyword, "OVER"),
                 Token(T.Whitespace, " "),
                 Token(T.Identifier, "win")])
    fn = Function([fn_name, p, Token(T.Whitespace, " "), over])
    assert fn.get_window() is over.tokens[-1]


# ---------------------------------------------------------------------------
# Class attributes / simple classes
# ---------------------------------------------------------------------------

def test_class_m_open_m_close():
    assert If.M_OPEN == (T.Keyword, "IF")
    assert If.M_CLOSE == (T.Keyword, "END IF")
    assert For.M_OPEN == (T.Keyword, ("FOR", "FOREACH"))
    assert For.M_CLOSE == (T.Keyword, "END LOOP")
    assert Where.M_OPEN == (T.Keyword, "WHERE")
    assert Where.M_CLOSE[1] == (
        "ORDER BY", "GROUP BY", "LIMIT", "UNION", "UNION ALL", "EXCEPT",
        "INTERSECT", "HAVING", "RETURNING", "INTO")
    assert Over.M_OPEN == (T.Keyword, "OVER")
    assert Having.M_OPEN == (T.Keyword, "HAVING")
    assert Having.M_CLOSE == (T.Keyword, ("ORDER BY", "LIMIT"))
    assert Case.M_OPEN == (T.Keyword, "CASE")
    assert Case.M_CLOSE == (T.Keyword, "END")
    assert Begin.M_OPEN == (T.Keyword, "BEGIN")
    assert Begin.M_CLOSE == (T.Keyword, "END")
    assert Parenthesis.M_OPEN == (T.Punctuation, "(")
    assert Parenthesis.M_CLOSE == (T.Punctuation, ")")
    assert SquareBrackets.M_OPEN == (T.Punctuation, "[")
    assert SquareBrackets.M_CLOSE == (T.Punctuation, "]")


def test_simple_subclasses_instantiate():
    for cls in (Assignment, Operation, Values, Command):
        obj = cls([])
        assert isinstance(obj, TokenList)
        assert obj.tokens == []
