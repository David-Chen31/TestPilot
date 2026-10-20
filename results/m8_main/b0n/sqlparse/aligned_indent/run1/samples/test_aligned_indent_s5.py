import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.aligned_indent import AlignedIndentFilter
from sqlparse.utils import indent, offset


def make_ws(value):
    return sql.Token(T.Whitespace, value)


def make_kw(value):
    return sql.Token(T.Keyword, value)


def make_token(text, ttype=T.Whitespace):
    return sql.Token(ttype, text)


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

def test_defaults():
    f = AlignedIndentFilter()
    assert f.n == '\n'
    assert f.offset == 0
    assert f.indent == 0
    assert f.char == ' '
    assert f._max_kwd_len == len('select')


def test_custom_char_and_newline():
    f = AlignedIndentFilter(char='\t', n='\r\n')
    assert f.char == '\t'
    assert f.n == '\r\n'


def test_class_attributes():
    assert AlignedIndentFilter.by_words == r'(GROUP|ORDER)\s+BY\b'
    assert 'FROM' in AlignedIndentFilter.split_words
    assert 'WHERE' in AlignedIndentFilter.split_words
    assert 'ON' in AlignedIndentFilter.split_words
    assert 'VALUES' in AlignedIndentFilter.split_words


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

def test_nl_default_offset():
    f = AlignedIndentFilter()
    tok = f.nl()
    assert isinstance(tok, sql.Token)
    assert tok.ttype is T.Whitespace
    # _max_kwd_len + offset(1) + indent(0) + offset(0) = 6 + 1 = 7
    assert tok.value == '\n' + ' ' * 7


def test_nl_with_int_offset():
    f = AlignedIndentFilter()
    tok = f.nl(3)
    assert tok.value == '\n' + ' ' * (6 + 3)


def test_nl_with_negative_int_offset():
    f = AlignedIndentFilter()
    tok = f.nl(-2)
    assert tok.value == '\n' + ' ' * (6 - 2)


def test_nl_with_string_offset():
    f = AlignedIndentFilter()
    # offset = -len('SELECT') = -6 -> spaces = 6 + (-6) = 0
    tok = f.nl('SELECT')
    assert tok.value == '\n'


def test_nl_with_string_offset_shorter_than_max():
    f = AlignedIndentFilter()
    # offset = -len('WHERE') = -5 -> spaces = 6 - 5 = 1
    tok = f.nl('WHERE')
    assert tok.value == '\n '


def test_nl_respects_indent_and_offset():
    f = AlignedIndentFilter()
    f.indent = 1
    f.offset = 2
    tok = f.nl(1)
    # 6 + 1 + 1*(2+6) + 2 = 17
    assert tok.value == '\n' + ' ' * 17


def test_nl_custom_char():
    f = AlignedIndentFilter(char='-')
    tok = f.nl(0)
    assert tok.value == '\n' + '-' * 6


# ---------------------------------------------------------------------------
# _process_statement
# ---------------------------------------------------------------------------

def test_process_statement_strips_leading_whitespace():
    f = AlignedIndentFilter()
    stmt = sql.Statement([make_ws('   '), make_kw('SELECT'), make_ws(' ')])
    f._process_statement(stmt)
    assert stmt.tokens[0].value == 'SELECT'


def test_process_statement_keeps_leading_ws_when_indented():
    f = AlignedIndentFilter()
    f.indent = 1
    stmt = sql.Statement([make_ws('   '), make_kw('SELECT')])
    f._process_statement(stmt)
    assert stmt.tokens[0].value == '   '


def test_process_statement_empty():
    f = AlignedIndentFilter()
    stmt = sql.Statement([])
    assert f._process_statement(stmt) is None
    assert stmt.tokens == []


def test_process_statement_non_whitespace_first():
    f = AlignedIndentFilter()
    stmt = sql.Statement([make_kw('SELECT'), make_ws(' ')])
    f._process_statement(stmt)
    assert stmt.tokens[0].value == 'SELECT'


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

def test_process_parenthesis_subquery_indents():
    f = AlignedIndentFilter()
    inner = sql.TokenList([
        make_token('('),
        make_kw('SELECT'),
        make_ws(' '),
    ])
    f._process_parenthesis(inner)
    # insert_after tlist[0] adds ws after '('
    assert inner.tokens[0].value == '('
    assert inner.tokens[1].ttype is T.Whitespace


def test_process_parenthesis_no_select_is_noop():
    f = AlignedIndentFilter()
    inner = sql.TokenList([make_token('('), make_kw('WHERE'), make_token(')')])
    before = list(inner.tokens)
    f._process_parenthesis(inner)
    assert inner.tokens == before


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

def test_process_identifierlist_inserts_newlines():
    f = AlignedIndentFilter()
    idents = sql.IdentifierList([
        sql.Identifier([make_token('a', T.Name)]),
        make_token(',', T.Punctuation),
        make_ws(' '),
        sql.Identifier([make_token('b', T.Name)]),
        make_token(',', T.Punctuation),
        make_ws(' '),
        sql.Identifier([make_token('c', T.Name)]),
    ])
    f._process_identifierlist(idents)
    # At least whitespace tokens got inserted before b and c
    ws_values = [t.value for t in idents.tokens
                 if t.ttype is T.Whitespace and t.value.startswith('\n')]
    assert len(ws_values) == 2


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

def test_process_case_simple():
    f = AlignedIndentFilter()
    tokens = [
        make_kw('CASE'),
        make_ws(' '),
        make_kw('WHEN'),
        make_ws(' '),
        make_token('1', T.Literal.Number.Integer),
        make_ws(' '),
        make_kw('THEN'),
        make_ws(' '),
        make_token('2', T.Literal.Number.Integer),
        make_ws(' '),
        make_kw('ELSE'),
        make_ws(' '),
        make_token('3', T.Literal.Number.Integer),
        make_ws(' '),
        make_kw('END'),
    ]
    case = sql.Case(tokens)
    f._process_case(case)
    # Should not error and END gets aligned
    assert any(t.value == 'END' for t in case.tokens if t.ttype is T.Keyword)


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

def test_next_token_finds_from():
    f = AlignedIndentFilter()
    tl = sql.TokenList([make_kw('SELECT'), make_ws(' '), make_kw('FROM')])
    idx, tok = f._next_token(tl)
    assert tok is not None
    assert tok.normalized == 'FROM'


def test_next_token_between_skips_and():
    f = AlignedIndentFilter()
    tl = sql.TokenList([
        make_kw('BETWEEN'),
        make_ws(' '),
        make_token('1', T.Literal.Number.Integer),
        make_ws(' '),
        make_kw('AND'),
        make_ws(' '),
        make_token('2', T.Literal.Number.Integer),
        make_ws(' '),
        make_kw('WHERE'),
    ])
    idx, tok = f._next_token(tl)
    assert tok.normalized == 'WHERE'


def test_next_token_returns_none_when_no_match():
    f = AlignedIndentFilter()
    tl = sql.TokenList([make_token('foo', T.Name)])
    idx, tok = f._next_token(tl)
    assert tok is None


# ---------------------------------------------------------------------------
# _split_kwds
# ---------------------------------------------------------------------------

def test_split_kwds_simple():
    f = AlignedIndentFilter()
    tl = sql.TokenList([make_kw('SELECT'), make_ws(' '), make_kw('FROM')])
    f._split_kwds(tl)
    ws_values = [t.value for t in tl.tokens
                 if t.ttype is T.Whitespace and t.value.startswith('\n')]
    assert len(ws_values) == 1


def test_split_kwds_join_uses_first_word():
    f = AlignedIndentFilter()
    join_tok = make_kw('LEFT JOIN')
    tl = sql.TokenList([make_kw('SELECT'), make_ws(' '), join_tok])
    f._split_kwds(tl)
    ws_values = [t.value for t in tl.tokens
                 if t.ttype is T.Whitespace and t.value.startswith('\n')]
    assert len(ws_values) == 1
    # 'LEFT' is 4 chars -> len 6-4=2 base + 1 = 3 spaces
    assert ws_values[0] == '\n' + ' ' * 3


def test_split_kwds_group_by():
    f = AlignedIndentFilter()
    tl = sql.TokenList([make_kw('SELECT'), make_ws(' '), make_kw('GROUP BY')])
    f._split_kwds(tl)
    ws_values = [t.value for t in tl.tokens
                 if t.ttype is T.Whitespace and t.value.startswith('\n')]
    assert len(ws_values) == 1


# ---------------------------------------------------------------------------
# _process_default with offset hack for group/order by
# ---------------------------------------------------------------------------

def test_process_default_by_words_offset():
    f = AlignedIndentFilter()
    sub = sql.TokenList([make_token('x', T.Name)])
    tl = sql.TokenList([make_kw('GROUP BY'), make_ws(' '), sub])
    f._process_default(tl)
    # offset must be reset afterwards
    assert f.offset == 0


def test_process_default_no_by_words():
    f = AlignedIndentFilter()
    sub = sql.TokenList([make_token('x', T.Name)])
    tl = sql.TokenList([make_kw('FROM'), make_ws(' '), sub])
    f._process_default(tl)
    assert f.offset == 0


# ---------------------------------------------------------------------------
# _process dispatch
# ---------------------------------------------------------------------------

def test_process_dispatch_statement():
    f = AlignedIndentFilter()
    stmt = sql.Statement([make_ws(' '), make_kw('SELECT')])
    f._process(stmt)
    assert stmt.tokens[0].value == 'SELECT'


def test_process_dispatch_falls_back_to_default():
    f = AlignedIndentFilter()
    tl = sql.TokenList([make_kw('SELECT'), make_ws(' '), make_kw('FROM')])
    f._process(tl)
    ws_values = [t.value for t in tl.tokens
                 if t.ttype is T.Whitespace and t.value.startswith('\n')]
    assert len(ws_values) == 1


# ---------------------------------------------------------------------------
# process() public entry point
# ---------------------------------------------------------------------------

def test_process_returns_same_stmt():
    f = AlignedIndentFilter()
    stmt = sql.Statement([make_kw('SELECT'), make_ws(' '), make_kw('FROM')])
    result = f.process(stmt)
    assert result is stmt


def test_process_modifies_stmt():
    f = AlignedIndentFilter()
    stmt = sql.Statement([make_kw('SELECT'), make_ws(' '), make_kw('FROM')])
    f.process(stmt)
    values = [t.value for t in stmt.tokens]
    assert any(v.startswith('\n') for v in values)


# ---------------------------------------------------------------------------
# indent / offset helpers integration
# ---------------------------------------------------------------------------

def test_indent_helper():
    f = AlignedIndentFilter()
    with indent(f):
        assert f.indent == 1
    assert f.indent == 0


def test_offset_helper():
    f = AlignedIndentFilter()
    with offset(f, 5):
        assert f.offset == 5
    assert f.offset == 0
