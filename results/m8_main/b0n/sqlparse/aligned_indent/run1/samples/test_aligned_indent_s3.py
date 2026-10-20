import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.aligned_indent import AlignedIndentFilter
from sqlparse.utils import indent, offset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt(stmt, char=' ', n='\n'):
    return str(AlignedIndentFilter(char=char, n=n).process(stmt))


def parse(sql_text):
    import sqlparse
    return sqlparse.parse(sql_text)[0]


# ---------------------------------------------------------------------------
# __init__ defaults
# ---------------------------------------------------------------------------

def test_init_defaults():
    f = AlignedIndentFilter()
    assert f.n == '\n'
    assert f.offset == 0
    assert f.indent == 0
    assert f.char == ' '
    assert f._max_kwd_len == len('select')
    assert f._max_kwd_len == 6


def test_init_custom():
    f = AlignedIndentFilter(char='\t', n='\r\n')
    assert f.char == '\t'
    assert f.n == '\r\n'
    assert f.offset == 0
    assert f.indent == 0


# ---------------------------------------------------------------------------
# class-level constants
# ---------------------------------------------------------------------------

def test_split_words_contents():
    assert 'FROM' in AlignedIndentFilter.split_words
    assert 'ON' in AlignedIndentFilter.split_words
    assert 'WHERE' in AlignedIndentFilter.split_words
    assert 'AND' in AlignedIndentFilter.split_words
    assert 'OR' in AlignedIndentFilter.split_words
    assert 'HAVING' in AlignedIndentFilter.split_words
    assert 'LIMIT' in AlignedIndentFilter.split_words
    assert 'UNION' in AlignedIndentFilter.split_words
    assert 'VALUES' in AlignedIndentFilter.split_words
    assert 'SET' in AlignedIndentFilter.split_words
    assert 'BETWEEN' in AlignedIndentFilter.split_words
    assert 'EXCEPT' in AlignedIndentFilter.split_words
    assert AlignedIndentFilter.by_words == r'(GROUP|ORDER)\s+BY\b'


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

def test_nl_default():
    f = AlignedIndentFilter()
    tok = f.nl()
    assert isinstance(tok, sql.Token)
    assert tok.ttype is T.Whitespace
    # indent = 0, offset default 1, _max_kwd_len = 6
    # -> n + char * (6 + 1 + 0 + 0) = '\n' + ' ' * 7
    assert tok.value == '\n' + ' ' * 7


def test_nl_int_offset():
    f = AlignedIndentFilter()
    tok = f.nl(offset=3)
    assert tok.value == '\n' + ' ' * (6 + 3)


def test_nl_string_offset():
    f = AlignedIndentFilter()
    tok = f.nl('SELECT')  # 6 chars -> offset = -6
    assert tok.value == '\n' + ' ' * (6 - 6)


def test_nl_with_indent_and_offset_state():
    f = AlignedIndentFilter()
    f.indent = 2
    f.offset = 4
    tok = f.nl()
    expected = 6 + 1 + 2 * (2 + 6) + 4
    assert tok.value == '\n' + ' ' * expected


def test_nl_custom_char_n():
    f = AlignedIndentFilter(char='.', n='|')
    tok = f.nl()
    assert tok.value == '|' + '.' * 7


# ---------------------------------------------------------------------------
# indent / offset context managers used by filter
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
    with offset(f):
        assert f.offset == 0
        with offset(f, 3):
            assert f.offset == 3
        assert f.offset == 0
    assert f.offset == 0


# ---------------------------------------------------------------------------
# process - basic SELECT
# ---------------------------------------------------------------------------

def test_process_returns_same_stmt():
    stmt = parse('select 1')
    f = AlignedIndentFilter()
    result = f.process(stmt)
    assert result is stmt


def test_process_simple_select():
    stmt = parse('select a from b')
    result = fmt(stmt)
    assert 'select' in result
    assert 'from' in result
    # FROM should be preceded by a newline
    assert '\n' in result


def test_process_where():
    stmt = parse('select a from b where c = 1')
    result = fmt(stmt)
    assert '\n' in result
    # newline before WHERE
    lines = result.split('\n')
    assert any(line.strip().startswith('WHERE') or
               line.strip().startswith('where') for line in lines)


def test_process_and_or():
    stmt = parse('select a from b where c = 1 and d = 2 or e = 3')
    result = fmt(stmt)
    assert '\n' in result
    # all three keywords should be present
    lower = result.lower()
    assert 'and' in lower
    assert 'or' in lower


# ---------------------------------------------------------------------------
# _process_statement strips leading whitespace only at indent 0
# ---------------------------------------------------------------------------

def test_process_statement_strips_leading_whitespace():
    stmt = parse('  select 1')
    # normalise token list to just whitespace + rest
    f = AlignedIndentFilter()
    f.process(stmt)
    # first token should no longer be whitespace
    first = stmt.tokens[0]
    assert not first.is_whitespace


def test_process_when_first_token_not_whitespace():
    stmt = parse('select 1')
    f = AlignedIndentFilter()
    f.process(stmt)
    # still ok, first token is the select keyword
    assert stmt.tokens


# ---------------------------------------------------------------------------
# _process_parenthesis / subquery
# ---------------------------------------------------------------------------

def test_process_parenthesis_subquery():
    stmt = parse('select * from (select a from b) t')
    result = fmt(stmt)
    assert 'select' in result.lower()
    # A newline should be introduced inside the parenthesis
    assert result.count('\n') >= 1


def test_process_parenthesis_no_select():
    # parenthesis without SELECT should not be re-indented
    stmt = parse('select (1 + 2) from t')
    f = AlignedIndentFilter()
    result = f.process(stmt)
    assert result is stmt
    assert '(' in str(result)


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

def test_process_identifierlist():
    stmt = parse('select a, b, c from t')
    result = fmt(stmt)
    # Each column after the first should be on its own line
    assert result.count('\n') >= 2
    assert 'a' in result
    assert 'b' in result
    assert 'c' in result


def test_process_identifierlist_single_column():
    stmt = parse('select a from t')
    result = fmt(stmt)
    assert 'a' in result
    # only the FROM newline should exist
    assert result.count('\n') == 1


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

def test_process_case_statement():
    stmt = parse('select case when a = 1 then 2 else 3 end from t')
    result = fmt(stmt)
    assert 'case' in result.lower()
    assert 'end' in result.lower()
    assert result.count('\n') >= 1
    # END should be on its own line
    assert any(line.strip().lower() in ('end',) for line in result.split('\n'))


def test_process_case_multiple_conditions():
    stmt = parse(
        'select case '
        'when aaaaaaaaa = 1 then x '
        'when b = 2 then y '
        'else z end '
        'from t'
    )
    result = fmt(stmt)
    # alignment padding: lines should contain "then"
    assert 'then' in result.lower()
    # END line present
    assert any(line.strip().lower().startswith('end')
               for line in result.split('\n'))


# ---------------------------------------------------------------------------
# _next_token BETWEEN handling
# ---------------------------------------------------------------------------

def test_between_treated_as_single_statement():
    stmt = parse('select a from t where x between 1 and 5 and y = 3')
    result = fmt(stmt)
    # BETWEEN x and y -- the AND inside BETWEEN must not be split
    lines = [ln.strip() for ln in result.split('\n')]
    between_lines = [ln for ln in lines if 'between' in ln.lower()]
    assert between_lines
    # the "and y = 3" should be its own line
    and_lines = [ln for ln in lines if ln.lower().startswith('and')]
    assert len(and_lines) >= 1


# ---------------------------------------------------------------------------
# _split_kwds - JOIN handling
# ---------------------------------------------------------------------------

def test_process_join():
    stmt = parse('select a from t left join u on t.x = u.x')
    result = fmt(stmt)
    lines = [ln.strip() for ln in result.split('\n')]
    # JOIN-line should use first word only as aligner
    join_lines = [ln for ln in lines if ln.lower().startswith('left')]
    assert any('join' in ln.lower() for ln in join_lines)


def test_process_group_by():
    stmt = parse('select a from t group by a order by b')
    result = fmt(stmt)
    lines = [ln.strip().lower() for ln in result.split('\n')]
    assert any(ln.startswith('group') for ln in lines)
    assert any(ln.startswith('order') for ln in lines)


# ---------------------------------------------------------------------------
# _process_default offset hack for group/order by
# ---------------------------------------------------------------------------

def test_group_by_with_sublist():
    stmt = parse('select a from t group by (a + b)')
    result = fmt(stmt)
    assert 'group' in result.lower()


# ---------------------------------------------------------------------------
# custom char / newline propagate
# ---------------------------------------------------------------------------

def test_custom_char_in_output():
    stmt = parse('select a from t')
    result = fmt(stmt, char='.')
    # no spaces produced by the filter for newline indentation
    assert '\n.' in result


def test_custom_newline_in_output():
    stmt = parse('select a from t')
    result = fmt(stmt, n='|')
    assert '|' in result
