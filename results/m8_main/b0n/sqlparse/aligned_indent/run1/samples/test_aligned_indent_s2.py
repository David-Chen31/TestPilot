import pytest

from sqlparse.filters.aligned_indent import AlignedIndentFilter
from sqlparse import sql, tokens as T
from sqlparse.utils import indent, offset


def make_tokenlist(*tokens):
    return sql.TokenList(tokens)


def tok(ttype, value):
    return sql.Token(ttype, value)


# ---------------------------------------------------------------------------
# Construction / attributes
# ---------------------------------------------------------------------------

def test_default_attributes():
    f = AlignedIndentFilter()
    assert f.n == '\n'
    assert f.char == ' '
    assert f.offset == 0
    assert f.indent == 0
    assert f._max_kwd_len == len('select') == 6


def test_custom_char_and_newline():
    f = AlignedIndentFilter(char='\t', n='\r\n')
    assert f.char == '\t'
    assert f.n == '\r\n'


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

def test_nl_default_offset():
    f = AlignedIndentFilter()
    token = f.nl()
    assert token.ttype is T.Whitespace
    # newline + max_kwd_len + offset(1) + indent(0) + offset(0)
    assert token.value == '\n' + ' ' * (6 + 1)


def test_nl_int_offset():
    f = AlignedIndentFilter()
    token = f.nl(3)
    assert token.value == '\n' + ' ' * (6 + 3)


def test_nl_string_offset():
    f = AlignedIndentFilter()
    token = f.nl('SELECT')  # len 6, so offset becomes -6
    assert token.value == '\n' + ' ' * (6 - 6)


def test_nl_respects_indent_and_offset():
    f = AlignedIndentFilter()
    f.indent = 2
    f.offset = 4
    token = f.nl(1)
    expected = '\n' + ' ' * (6 + 1 + 2 * (2 + 6) + 4)
    assert token.value == expected


def test_nl_custom_char():
    f = AlignedIndentFilter(char='.')
    token = f.nl(1)
    assert token.value == '\n' + '.' * (6 + 1)


def test_nl_custom_newline():
    f = AlignedIndentFilter(n='\r\n')
    token = f.nl(1)
    assert token.value.startswith('\r\n')


# ---------------------------------------------------------------------------
# indent / offset context managers (imported usage)
# ---------------------------------------------------------------------------

def test_indent_context_manager():
    f = AlignedIndentFilter()
    assert f.indent == 0
    with indent(f):
        assert f.indent == 1
        with indent(f, 2):
            assert f.indent == 3
        assert f.indent == 1
    assert f.indent == 0


def test_offset_context_manager():
    f = AlignedIndentFilter()
    assert f.offset == 0
    with offset(f, 5):
        assert f.offset == 5
        with offset(f, -2):
            assert f.offset == 3
        assert f.offset == 5
    assert f.offset == 0


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

def test_next_token_finds_keyword():
    f = AlignedIndentFilter()
    t1 = tok(T.Keyword, 'FROM')
    t2 = tok(T.Keyword, 'WHERE')
    tlist = make_tokenlist(t1, t2)
    idx, token = f._next_token(tlist)
    assert token is t1
    assert idx == 0


def test_next_token_from_index():
    f = AlignedIndentFilter()
    t1 = tok(T.Keyword, 'FROM')
    t2 = tok(T.Keyword, 'WHERE')
    tlist = make_tokenlist(t1, t2)
    idx, token = f._next_token(tlist, 0)
    assert token is t2
    assert idx == 1


def test_next_token_none_when_no_match():
    f = AlignedIndentFilter()
    tlist = make_tokenlist(tok(T.Name, 'x'))
    idx, token = f._next_token(tlist)
    assert idx == -1
    assert token is None


def test_next_token_skips_between_and():
    # BETWEEN x AND y should be skipped as a unit
    f = AlignedIndentFilter()
    between = tok(T.Keyword, 'BETWEEN')
    name = tok(T.Name, 'x')
    and_ = tok(T.Keyword, 'AND')
    where = tok(T.Keyword, 'WHERE')
    tlist = make_tokenlist(between, name, and_, where)
    idx, token = f._next_token(tlist)
    assert token is where
    assert idx == 3


# ---------------------------------------------------------------------------
# _split_kwds
# ---------------------------------------------------------------------------

def test_split_kwds_inserts_newlines():
    f = AlignedIndentFilter()
    kw = tok(T.Keyword, 'FROM')
    tlist = make_tokenlist(kw)
    f._split_kwds(tlist)
    # A whitespace newline token is inserted before FROM
    assert len(tlist.tokens) == 2
    first = tlist.tokens[0]
    assert first.is_whitespace
    assert '\n' in first.value
    assert tlist.tokens[1] is kw


def test_split_kwds_join_uses_first_word():
    f = AlignedIndentFilter()
    kw = tok(T.Keyword, 'LEFT JOIN')
    tlist = make_tokenlist(kw)
    f._split_kwds(tlist)
    ws = tlist.tokens[0]
    assert ws.is_whitespace
    # token_indent is "LEFT" (len 4)
    expected = '\n' + ' ' * (6 - 4)
    assert ws.value == expected


def test_split_kwds_group_by_first_word():
    f = AlignedIndentFilter()
    kw = tok(T.Keyword, 'GROUP BY')
    tlist = make_tokenlist(kw)
    f._split_kwds(tlist)
    ws = tlist.tokens[0]
    expected = '\n' + ' ' * (6 - 5)  # 'GROUP' len 5
    assert ws.value == expected


# ---------------------------------------------------------------------------
# _process_statement
# ---------------------------------------------------------------------------

def test_process_statement_strips_leading_whitespace():
    f = AlignedIndentFilter()
    ws = tok(T.Whitespace, '  ')
    name = tok(T.Name, 'x')
    stmt = make_tokenlist(ws, name)
    f._process_statement(stmt)
    # leading whitespace removed
    assert stmt.tokens[0] is name


def test_process_statement_keeps_whitespace_when_indent():
    f = AlignedIndentFilter()
    f.indent = 1
    ws = tok(T.Whitespace, '  ')
    name = tok(T.Name, 'x')
    stmt = make_tokenlist(ws, name)
    f._process_statement(stmt)
    assert stmt.tokens[0] is ws


def test_process_statement_empty():
    f = AlignedIndentFilter()
    stmt = make_tokenlist()
    # must not raise
    f._process_statement(stmt)
    assert len(stmt.tokens) == 0


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

def test_process_parenthesis_no_select_noop():
    f = AlignedIndentFilter()
    inner = tok(T.Name, 'x')
    paren = sql.Parenthesis([inner])
    before = len(paren.tokens)
    f._process_parenthesis(paren)
    assert len(paren.tokens) == before


def test_process_parenthesis_with_select():
    f = AlignedIndentFilter()
    sel = tok(T.DML, 'SELECT')
    name = tok(T.Name, 'x')
    paren = sql.Parenthesis([sel, name])
    f._process_parenthesis(paren)
    # A newline should have been inserted after the opening paren
    assert paren.tokens[0].is_whitespace
    assert '\n' in paren.tokens[0].value
    # De-indent newline inserted before the closing paren
    assert paren.tokens[-1].is_whitespace
    assert '\n' in paren.tokens[-1].value


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

def test_process_identifierlist_inserts_newlines_between():
    f = AlignedIndentFilter()
    a = tok(T.Name, 'a')
    comma = tok(T.Punctuation, ',')
    b = tok(T.Name, 'b')
    c = tok(T.Name, 'c')
    ids = sql.IdentifierList([a, comma, b, comma, c])
    f._process_identifierlist(ids)
    # tokens are: a, ws, b, ws, c (commas become part of identifiers)
    ws_tokens = [t for t in ids.tokens if t.is_whitespace]
    assert len(ws_tokens) >= 2


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

def test_process_case_aligns():
    f = AlignedIndentFilter()
    case_kw = tok(T.Keyword, 'CASE')
    when1 = tok(T.Keyword, 'WHEN')
    cond1 = tok(T.Name, 'a')
    then1 = tok(T.Keyword, 'THEN')
    val1 = tok(T.Number, '1')
    else_kw = tok(T.Keyword, 'ELSE')
    val2 = tok(T.Number, '2')
    end_kw = tok(T.Keyword, 'END')
    case = sql.Case([
        case_kw, when1, cond1, then1, val1, else_kw, val2, end_kw
    ])
    f._process_case(case)
    # At least one whitespace inserted
    assert any(t.is_whitespace for t in case.tokens)


# ---------------------------------------------------------------------------
# _process_default - offset hack for group/order by
# ---------------------------------------------------------------------------

def test_process_default_group_by_offset():
    f = AlignedIndentFilter()
    by = tok(T.Keyword, 'GROUP BY')
    inner = sql.TokenList([tok(T.Name, 'x')])
    tlist = make_tokenlist(by, inner)
    f._process_default(tlist)
    # offset should be restored to 0 after
    assert f.offset == 0


# ---------------------------------------------------------------------------
# process() entry point
# ---------------------------------------------------------------------------

def test_process_returns_same_object():
    f = AlignedIndentFilter()
    stmt = make_tokenlist(tok(T.Keyword, 'SELECT'),
                          tok(T.Name, 'x'),
                          tok(T.Keyword, 'FROM'),
                          tok(T.Name, 'table'))
    result = f.process(stmt)
    assert result is stmt
    # Both SELECT and FROM split; newline tokens inserted
    assert any(t.is_whitespace and '\n' in t.value for t in stmt.tokens)


def test_process_full_statement_string():
    from sqlparse import parse
    stmt = parse('SELECT a, b FROM t WHERE x = 1')[0]
    f = AlignedIndentFilter()
    result = f.process(stmt)
    assert result is stmt


# ---------------------------------------------------------------------------
# _process dispatch
# ---------------------------------------------------------------------------

def test_process_dispatch_to_default():
    f = AlignedIndentFilter()
    name = tok(T.Name, 'x')
    tlist = make_tokenlist(name)
    f._process(tlist)
    # No keywords, nothing inserted
    assert len(tlist.tokens) == 1


def test_join_words_regex_matches():
    import re
    pattern = AlignedIndentFilter.join_words
    assert re.search(pattern, 'JOIN')
    assert re.search(pattern, 'LEFT JOIN')
    assert re.search(pattern, 'CROSS JOIN')
    assert re.search(pattern, 'NATURAL JOIN')


def test_by_words_regex_matches():
    import re
    pattern = AlignedIndentFilter.by_words
    assert re.search(pattern, 'GROUP BY')
    assert re.search(pattern, 'ORDER BY')
