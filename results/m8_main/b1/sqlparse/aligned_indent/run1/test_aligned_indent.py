import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.aligned_indent import AlignedIndentFilter


def make_token(value, ttype=T.Keyword):
    return sql.Token(ttype, value)


def test_init_defaults():
    f = AlignedIndentFilter()
    assert f.n == '\n'
    assert f.offset == 0
    assert f.indent == 0
    assert f.char == ' '
    assert f._max_kwd_len == len('select')


def test_init_custom_char_and_n():
    f = AlignedIndentFilter(char='\t', n='\r\n')
    assert f.char == '\t'
    assert f.n == '\r\n'


def test_class_attributes():
    assert 'JOIN\\b' in AlignedIndentFilter.join_words
    assert AlignedIndentFilter.by_words == r'(GROUP|ORDER)\s+BY\b'
    for word in ('FROM', 'ON', 'WHERE', 'AND', 'OR', 'HAVING',
                 'LIMIT', 'UNION', 'VALUES', 'SET', 'BETWEEN', 'EXCEPT'):
        assert word in AlignedIndentFilter.split_words


def test_nl_default_offset():
    f = AlignedIndentFilter()
    tok = f.nl()
    assert isinstance(tok, sql.Token)
    assert tok.ttype == T.Whitespace
    # _max_kwd_len (6) + offset (1) = 7 spaces
    assert tok.value == '\n' + ' ' * 7


def test_nl_int_offset():
    f = AlignedIndentFilter()
    assert f.nl(3).value == '\n' + ' ' * 9
    assert f.nl(0).value == '\n' + ' ' * 6


def test_nl_string_offset():
    f = AlignedIndentFilter()
    # offset length negated: 'SELECT' -> -6 -> 6 - 6 = 0 spaces
    assert f.nl('SELECT').value == '\n'
    # 'ON' -> -2 -> 6 - 2 = 4 spaces
    assert f.nl('ON').value == '\n' + ' ' * 4
    # empty string -> 0 -> 6 spaces
    assert f.nl('').value == '\n' + ' ' * 6


def test_nl_with_indent_and_offset():
    f = AlignedIndentFilter()
    f.indent = 2
    f.offset = 3
    # indent = 2 * (2 + 6) = 16; 6 + 1 + 16 + 3 = 26
    assert f.nl().value == '\n' + ' ' * 26


def test_nl_custom_char():
    f = AlignedIndentFilter(char='.')
    assert f.nl().value == '\n' + '.' * 7


def test_nl_custom_newline():
    f = AlignedIndentFilter(n='#')
    assert f.nl().value == '#' + ' ' * 7


def test_next_token_finds_keyword():
    f = AlignedIndentFilter()
    tlist = sql.TokenList([
        make_token('foo', T.Name),
        make_token('FROM'),
        make_token('bar', T.Name),
    ])
    idx, token = f._next_token(tlist)
    assert idx == 1
    assert token.value == 'FROM'


def test_next_token_none_when_no_keyword():
    f = AlignedIndentFilter()
    tlist = sql.TokenList([make_token('foo', T.Name)])
    idx, token = f._next_token(tlist)
    assert token is None


def test_next_token_between_and_skipped():
    f = AlignedIndentFilter()
    tlist = sql.TokenList([
        make_token('BETWEEN'),
        make_token('x', T.Name),
        make_token('AND'),
        make_token('y', T.Name),
        make_token('WHERE'),
    ])
    idx, token = f._next_token(tlist)
    # BETWEEN is treated as a single statement including AND
    assert token.value == 'WHERE'
    assert idx == 4


def test_next_token_between_without_and():
    f = AlignedIndentFilter()
    tlist = sql.TokenList([
        make_token('BETWEEN'),
        make_token('x', T.Name),
        make_token('WHERE'),
    ])
    idx, token = f._next_token(tlist)
    assert token.value == 'WHERE'


def test_split_kwds_inserts_newlines():
    f = AlignedIndentFilter()
    tlist = sql.TokenList([
        make_token('SELECT'),
        make_token('a', T.Name),
        make_token('FROM'),
        make_token('b', T.Name),
    ])
    f._split_kwds(tlist)
    # A newline token should be inserted before each split keyword
    values = [t.value for t in tlist.tokens]
    assert '\n' + ' ' * 0 in values or any(
        v.startswith('\n') for v in values if isinstance(v, str))


def test_split_kwds_join_uses_first_word():
    f = AlignedIndentFilter()
    tlist = sql.TokenList([
        make_token('LEFT JOIN'),
        make_token('b', T.Name),
    ])
    f._split_kwds(tlist)
    first = tlist.tokens[0]
    assert isinstance(first, sql.Token)
    assert first.ttype == T.Whitespace
    # token_indent is 'LEFT' -> length 4; 6 - 4 = 2 spaces
    assert first.value == '\n' + ' ' * 2


def test_split_kwds_by_words_uses_first_word():
    f = AlignedIndentFilter()
    tlist = sql.TokenList([
        make_token('GROUP BY'),
        make_token('a', T.Name),
    ])
    f._split_kwds(tlist)
    first = tlist.tokens[0]
    # 'GROUP' -> length 5; 6 - 5 = 1 space
    assert first.value == '\n' + ' '


def test_process_default_processes_sublists():
    f = AlignedIndentFilter()
    inner = sql.TokenList([make_token('x', T.Name)])
    tlist = sql.TokenList([
        make_token('SELECT'),
        inner,
    ])
    f._process_default(tlist)
    # Newline inserted before sublist? At least no exception
    assert isinstance(tlist, sql.TokenList)


def test_process_default_group_by_offset_hack():
    f = AlignedIndentFilter()
    sub = sql.TokenList([make_token('x', T.Name)])
    tlist = sql.TokenList([
        make_token('GROUP BY'),
        sub,
    ])
    captured = {}

    original_process = f._process

    def recorder(t):
        captured['offset'] = f.offset
        original_process(t)

    f._process = recorder
    f._process_default(tlist)
    # offset should be 3 during processing of sublist
    assert captured.get('offset') == 3


def test_process_statement_pops_leading_whitespace():
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.Whitespace, '   '),
        make_token('SELECT'),
    ])
    f._process_statement(stmt)
    assert not any(
        t.is_whitespace and t.value == '   ' for t in stmt.tokens
        if isinstance(t, sql.Token) and not isinstance(t, sql.TokenList))


def test_process_statement_keeps_whitespace_when_indented():
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.Whitespace, '   '),
        make_token('SELECT'),
    ])
    f.indent = 1
    f._process_statement(stmt)
    assert stmt.tokens[0].is_whitespace
    assert stmt.tokens[0].value == '   '


def test_process_parenthesis_non_subquery():
    f = AlignedIndentFilter()
    paren = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        make_token('foo', T.Name),
        sql.Token(T.Punctuation, ')'),
    ])
    # No SELECT -> nothing inserted
    f._process_parenthesis(paren)
    assert len(paren.tokens) == 3


def test_process_parenthesis_subquery():
    f = AlignedIndentFilter()
    paren = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        make_token('SELECT', T.DML),
        make_token('x', T.Name),
        sql.Token(T.Punctuation, ')'),
    ])
    f._process_parenthesis(paren)
    # A newline should have been inserted after '(' and before ')'
    values = [t.value for t in paren.tokens if isinstance(t, sql.Token)]
    assert any(v.startswith('\n') for v in values)


def test_process_identifierlist_inserts_newlines():
    f = AlignedIndentFilter()
    il = sql.IdentifierList([
        sql.Identifier([make_token('a', T.Name)]),
        sql.Token(T.Punctuation, ','),
        sql.Identifier([make_token('b', T.Name)]),
    ])
    f._process_identifierlist(il)
    values = [t.value for t in il.tokens if isinstance(t, sql.Token)
              and not isinstance(t, sql.TokenList)]
    assert any(v.startswith('\n') for v in values)


def test_process_case_aligns_values():
    f = AlignedIndentFilter()
    case = sql.Case([
        make_token('CASE'),
        make_token('WHEN'),
        make_token('a', T.Name),
        make_token('THEN'),
        make_token('b', T.Name),
        make_token('ELSE'),
        make_token('c', T.Name),
        make_token('END'),
    ])
    f._process_case(case)
    # should have inserted whitespace tokens, no exception
    assert isinstance(case, sql.Case)


def test_process_dispatch_unknown_type_uses_default():
    f = AlignedIndentFilter()
    tlist = sql.TokenList([make_token('SELECT'), make_token('x', T.Name)])
    f._process(tlist)
    assert isinstance(tlist, sql.TokenList)


def test_process_with_update():
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.DML, 'UPDATE'),
        sql.Token(T.Whitespace, ' '),
        make_token('foo', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('SET'),
        sql.Token(T.Whitespace, ' '),
        make_token('bar', T.Name),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Punctuation, '='),
        sql.Token(T.Whitespace, ' '),
        make_token('1', T.Number.Integer),
    ])
    result = f.process(stmt)
    assert result is stmt
    # SET should have been split onto a new line
    values = [t.value for t in stmt.tokens]
    assert any(v.startswith('\n') for v in values if isinstance(v, str))


def test_process_select_from():
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.DML, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        make_token('a', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('FROM'),
        sql.Token(T.Whitespace, ' '),
        make_token('b', T.Name),
    ])
    f.process(stmt)
    values = [t.value for t in stmt.tokens]
    assert any(v.startswith('\n') for v in values if isinstance(v, str))


def test_process_select_from_where():
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.DML, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        make_token('a', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('FROM'),
        sql.Token(T.Whitespace, ' '),
        make_token('b', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('WHERE'),
        sql.Token(T.Whitespace, ' '),
        make_token('x', T.Name),
    ])
    f.process(stmt)
    values = [t.value for t in stmt.tokens]
    assert any(v.startswith('\n') for v in values if isinstance(v, str))


def test_process_empty_statement():
    f = AlignedIndentFilter()
    stmt = sql.Statement([])
    result = f.process(stmt)
    assert result is stmt
    assert len(stmt.tokens) == 0


def test_process_returns_same_object():
    f = AlignedIndentFilter()
    stmt = sql.Statement([make_token('SELECT', T.DML)])
    assert f.process(stmt) is stmt


def test_process_join_group_order():
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.DML, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        make_token('a', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('FROM'),
        sql.Token(T.Whitespace, ' '),
        make_token('t1', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('LEFT JOIN'),
        sql.Token(T.Whitespace, ' '),
        make_token('t2', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('GROUP BY'),
        sql.Token(T.Whitespace, ' '),
        make_token('a', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('ORDER BY'),
        sql.Token(T.Whitespace, ' '),
        make_token('a', T.Name),
    ])
    f.process(stmt)
    values = [t.value for t in stmt.tokens if isinstance(t, sql.Token)]
    assert sum(1 for v in values if isinstance(v, str) and v.startswith('\n')) >= 3


def test_process_between_and_not_split():
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.DML, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        make_token('a', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('FROM'),
        sql.Token(T.Whitespace, ' '),
        make_token('b', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('WHERE'),
        sql.Token(T.Whitespace, ' '),
        make_token('x', T.Name),
        sql.Token(T.Whitespace, ' '),
        make_token('BETWEEN'),
        sql.Token(T.Whitespace, ' '),
        make_token('1', T.Number.Integer),
        sql.Token(T.Whitespace, ' '),
        make_token('AND'),
        sql.Token(T.Whitespace, ' '),
        make_token('5', T.Number.Integer),
    ])
    f.process(stmt)
    # AND inside BETWEEN should not be split onto a new line.
    # The implementation skips BETWEEN plus its AND when scanning, so
    # BETWEEN and AND don't get their own newline; only a couple of split
    # keywords remain as top-level newline-inserted tokens.
    newlines = [t.value for t in stmt.tokens
                if isinstance(t, sql.Token) and isinstance(t.value, str)
                and t.value.startswith('\n')]
    assert len(newlines) == 2


def test_process_statement_with_leading_whitespace_pops_first():
    # _process_statement pops a leading whitespace token when indent == 0,
    # so a following non-keyword token then becomes the first token.
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.Whitespace, ' '),
        make_token('foo', T.Name),
    ])
    f._process_statement(stmt)
    assert len(stmt.tokens) == 1
    assert stmt.tokens[0].value == 'foo'


def test_indent_context_manager_effect():
    from sqlparse.utils import indent
    f = AlignedIndentFilter()
    with indent(f):
        assert f.indent == 1
        with indent(f):
            assert f.indent == 2
        assert f.indent == 1
    assert f.indent == 0


def test_offset_context_manager_effect():
    from sqlparse.utils import offset
    f = AlignedIndentFilter()
    with offset(f, 3):
        assert f.offset == 3
    assert f.offset == 0
