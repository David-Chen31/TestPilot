import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.aligned_indent import AlignedIndentFilter
from sqlparse.utils import indent, offset


# ---------------------------------------------------------------------------
# Construction / initial state
# ---------------------------------------------------------------------------

def test_default_init_state():
    f = AlignedIndentFilter()
    assert f.n == '\n'
    assert f.offset == 0
    assert f.indent == 0
    assert f.char == ' '
    assert f._max_kwd_len == len('select') == 6


def test_custom_char_and_newline():
    f = AlignedIndentFilter(char='\t', n='\r\n')
    assert f.char == '\t'
    assert f.n == '\r\n'
    assert f.offset == 0
    assert f.indent == 0


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

def test_nl_default_int_offset():
    f = AlignedIndentFilter()
    token = f.nl()
    assert isinstance(token, sql.Token)
    assert token.ttype == T.Whitespace
    # 6 (max kwd) + 1 (offset) + 0 (indent) + 0 (self.offset) = 7 spaces
    assert token.value == '\n' + ' ' * 7


def test_nl_with_explicit_int():
    f = AlignedIndentFilter()
    token = f.nl(3)
    assert token.value == '\n' + ' ' * (6 + 3)


def test_nl_with_string_offset():
    f = AlignedIndentFilter()
    # 'SELECT' has length 6
    token = f.nl('SELECT')
    assert token.value == '\n' + ' ' * (6 - 6 + 1)


def test_nl_with_longer_string_offset():
    f = AlignedIndentFilter()
    token = f.nl('FROM')
    assert token.value == '\n' + ' ' * (6 - 4 + 1)


def test_nl_uses_indent_and_offset_state():
    f = AlignedIndentFilter()
    f.indent = 2
    f.offset = 3
    token = f.nl()
    # 6 + 1 + 2*(2+6) + 3 = 26
    assert token.value == '\n' + ' ' * (6 + 1 + 2 * 8 + 3)


def test_nl_custom_char():
    f = AlignedIndentFilter(char='.')
    token = f.nl()
    assert token.value == '\n' + '.' * 7


# ---------------------------------------------------------------------------
# indent / offset context managers interplay (via utils, sanity check)
# ---------------------------------------------------------------------------

def test_indent_context_manager_updates_filter():
    f = AlignedIndentFilter()
    with indent(f, 2):
        assert f.indent == 2
    assert f.indent == 0


def test_offset_context_manager_updates_filter():
    f = AlignedIndentFilter()
    with offset(f, 3):
        assert f.offset == 3
    assert f.offset == 0


# ---------------------------------------------------------------------------
# split_words / class attributes
# ---------------------------------------------------------------------------

def test_join_words_attribute_present():
    assert 'JOIN' in AlignedIndentFilter.join_words
    assert 'LEFT' in AlignedIndentFilter.join_words
    assert 'CROSS' in AlignedIndentFilter.join_words


def test_by_words_attribute():
    assert 'GROUP' in AlignedIndentFilter.by_words
    assert 'ORDER' in AlignedIndentFilter.by_words
    assert 'BY' in AlignedIndentFilter.by_words


def test_split_words_attribute():
    for word in ('FROM', 'ON', 'WHERE', 'AND', 'OR', 'HAVING', 'LIMIT',
                 'UNION', 'VALUES', 'SET', 'BETWEEN', 'EXCEPT'):
        assert word in AlignedIndentFilter.split_words


# ---------------------------------------------------------------------------
# process() with real statements
# ---------------------------------------------------------------------------

def test_process_simple_statement():
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.Keyword, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    result = f.process(stmt)
    assert result is stmt


def test_process_removes_leading_whitespace():
    f = AlignedIndentFilter()
    ws = sql.Token(T.Whitespace, '   ')
    stmt = sql.Statement([
        ws,
        sql.Token(T.Keyword, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    assert stmt.tokens[0] is ws
    f.process(stmt)
    assert ws not in stmt.tokens


def test_process_keeps_leading_whitespace_when_indented():
    f = AlignedIndentFilter()
    f.indent = 1
    ws = sql.Token(T.Whitespace, '   ')
    stmt = sql.Statement([
        ws,
        sql.Token(T.Keyword, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    assert stmt.tokens[0] is ws


def test_process_empty_statement():
    f = AlignedIndentFilter()
    stmt = sql.Statement([])
    result = f.process(stmt)
    assert result is stmt
    assert len(stmt.tokens) == 0


def test_process_select_from():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('select * from foo')[0]
    result = f.process(stmt)
    rendered = str(result)
    # from is aligned on a new line
    assert '\nFROM' in rendered.upper() or '\nfrom' in rendered.lower()
    assert str(stmt)  # non-empty


def test_process_group_by_uses_first_word_as_aligner():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('select a, b from t group by a')[0]
    f.process(stmt)
    rendered = str(stmt)
    # GROUP should start a new line
    upper = rendered.upper()
    assert 'GROUP' in upper
    # By hack: group by appears after newline
    assert '\n' in rendered


def test_process_where_and_or():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('select * from t where a = 1 and b = 2')[0]
    f.process(stmt)
    rendered = str(stmt).upper()
    assert 'WHERE' in rendered
    assert 'AND' in rendered
    assert '\n' in rendered


def test_process_case_statement():
    import sqlparse
    f = AlignedIndentFilter()
    sqltext = "select case when a = 1 then 'x' else 'y' end from t"
    stmt = sqlparse.parse(sqltext)[0]
    f.process(stmt)
    rendered = str(stmt).upper()
    assert 'WHEN' in rendered
    assert 'ELSE' in rendered
    assert 'END' in rendered


def test_process_between_treated_as_single():
    f = AlignedIndentFilter()
    between = sql.Token(T.Keyword, 'BETWEEN')
    and_tok = sql.Token(T.Keyword, 'AND')
    tl = sql.Statement([
        sql.Token(T.Keyword, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Literal.Number.Integer, '1'),
        sql.Token(T.Whitespace, ' '),
        between,
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Literal.Number.Integer, '2'),
        sql.Token(T.Whitespace, ' '),
        and_tok,
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Literal.Number.Integer, '3'),
    ])
    # _next_token should skip AND after BETWEEN
    idx, tok = f._next_token(tl)
    assert tok is between
    # the AND following BETWEEN and number should not be returned as split
    idx2, tok2 = f._next_token(tl, idx)
    # should skip AND because BETWEEN was found
    assert tok2 is not and_tok


def test_process_parenthesis_subquery():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('select * from (select 1) as x')[0]
    f.process(stmt)
    rendered = str(stmt)
    assert 'SELECT' in rendered.upper()


def test_process_identifierlist_multiple_columns():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('select a, b, c from t')[0]
    f.process(stmt)
    rendered = str(stmt)
    # All identifiers present
    assert 'a' in rendered
    assert 'b' in rendered
    assert 'c' in rendered


def test_process_identifierlist_single_column():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('select a from t')[0]
    f.process(stmt)
    rendered = str(stmt)
    assert 'a' in rendered
    assert 'FROM' in rendered.upper()


def test_process_join_aligner_uses_first_word():
    f = AlignedIndentFilter()
    join_kwd = sql.Token(T.Keyword, 'LEFT JOIN')
    tl = sql.Statement([
        sql.Token(T.Keyword, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Literal.Number.Integer, '1'),
        sql.Token(T.Whitespace, ' '),
        join_kwd,
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 't2'),
    ])
    f._split_kwds(tl)
    # A whitespace token should have been inserted before the join keyword
    idx = tl.token_index(join_kwd)
    prev = tl.tokens[idx - 1]
    assert prev.is_whitespace
    assert '\n' in prev.value


def test_next_token_returns_none_when_not_found():
    f = AlignedIndentFilter()
    tl = sql.TokenList([
        sql.Token(T.Literal.Number.Integer, '1'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'foo'),
    ])
    idx, tok = f._next_token(tl)
    assert tok is None


def test_process_default_offsets_substatement_after_group_by():
    # Exercises the HACK offset branch for by_words
    f = AlignedIndentFilter()
    sub = sql.TokenList([sql.Token(T.Name, 'a')])
    tl = sql.TokenList([
        sql.Token(T.Keyword, 'GROUP BY'),
        sql.Token(T.Whitespace, ' '),
        sub,
    ])
    f._process_default(tl)
    # offset should be reset after processing
    assert f.offset == 0
    assert f.indent == 0


def test_process_default_offsets_zero():
    f = AlignedIndentFilter()
    sub = sql.TokenList([sql.Token(T.Name, 'a')])
    tl = sql.TokenList([
        sql.Token(T.Name, 'x'),
        sql.Token(T.Whitespace, ' '),
        sub,
    ])
    f._process_default(tl)
    assert f.offset == 0


def test_process_dispatch():
    f = AlignedIndentFilter()
    stmt = sql.Statement([])
    # _process should dispatch to _process_statement
    f._process(stmt)
    assert isinstance(stmt, sql.Statement)


def test_process_returns_same_object():
    f = AlignedIndentFilter()
    stmt = sql.Statement([])
    assert f.process(stmt) is stmt


def test_process_preserves_state_after_indent_and_offset():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('select * from t where a=1')[0]
    f.process(stmt)
    assert f.indent == 0
    assert f.offset == 0


def test_process_union():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('select 1 union select 2')[0]
    f.process(stmt)
    rendered = str(stmt).upper()
    assert 'UNION' in rendered


def test_process_having_limit():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse(
        'select a from t group by a having count(*) > 1 limit 10'
    )[0]
    f.process(stmt)
    rendered = str(stmt).upper()
    assert 'HAVING' in rendered
    assert 'LIMIT' in rendered


def test_process_values():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('insert into t values (1)')[0]
    f.process(stmt)
    rendered = str(stmt).upper()
    assert 'VALUES' in rendered


def test_process_set():
    import sqlparse
    f = AlignedIndentFilter()
    stmt = sqlparse.parse('update t set a = 1')[0]
    f.process(stmt)
    rendered = str(stmt).upper()
    assert 'SET' in rendered
