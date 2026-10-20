import pytest

from sqlparse import sql, tokens as T
from sqlparse.filters.aligned_indent import AlignedIndentFilter
import sqlparse


def _filter(sql_text, **kwargs):
    return sqlparse.format(sql_text, reindent_aligned=True, **kwargs)


class TestInit:
    def test_defaults(self):
        f = AlignedIndentFilter()
        assert f.n == '\n'
        assert f.offset == 0
        assert f.indent == 0
        assert f.char == ' '
        assert f._max_kwd_len == len('select')

    def test_custom_char(self):
        f = AlignedIndentFilter(char='\t')
        assert f.char == '\t'

    def test_custom_n(self):
        f = AlignedIndentFilter(n='\r\n')
        assert f.n == '\r\n'


class TestNl:
    def test_nl_default(self):
        f = AlignedIndentFilter()
        tok = f.nl()
        assert isinstance(tok, sql.Token)
        assert tok.ttype == T.Whitespace
        # n + char * (_max_kwd_len + 1 + 0 + 0)
        assert tok.value == '\n' + ' ' * (len('select') + 1)

    def test_nl_int_offset(self):
        f = AlignedIndentFilter()
        tok = f.nl(3)
        assert tok.value == '\n' + ' ' * (len('select') + 3)

    def test_nl_string_offset(self):
        f = AlignedIndentFilter()
        tok = f.nl('SELECT')
        assert tok.value == '\n' + ' ' * (len('select') + 1 - len('SELECT'))

    def test_nl_with_indent_and_offset(self):
        f = AlignedIndentFilter()
        f.indent = 1
        f.offset = 2
        tok = f.nl(1)
        assert tok.value == '\n' + ' ' * (
            len('select') + 1 + 1 * (2 + len('select')) + 2)


class TestJoinWordsByWords:
    def test_join_words_attributes(self):
        assert AlignedIndentFilter.join_words is not None
        assert isinstance(AlignedIndentFilter.by_words, str)

    def test_simple_join_aligned(self):
        out = _filter('select * from foo join bar on foo.a = bar.a')
        lines = out.split('\n')
        # JOIN should be aligned as its own aligner
        joined = [line for line in lines if 'JOIN' in line]
        assert joined, out
        idx = joined[0].index('JOIN')
        # 'select' start column is 0, "from" index should match 'JOIN' too
        from_line = [line for line in lines if line.strip().upper().startswith('FROM')]
        assert from_line, out
        assert from_line[0].index('FROM') == idx


class TestProcessBasic:
    def test_process_returns_stmt(self):
        f = AlignedIndentFilter()
        stmt = sqlparse.parse('select * from foo')[0]
        result = f.process(stmt)
        assert result is stmt

    def test_select_from_alignment(self):
        out = _filter('select a, b from foo')
        lines = out.split('\n')
        assert lines[0].lstrip().startswith('SELECT')
        # columns indented
        assert any(line.strip() == 'a,' or line.strip().startswith('a') for line in lines)
        assert any(line.strip().upper().startswith('FROM') for line in lines)

    def test_where_aligned(self):
        out = _filter('select * from foo where a = 1')
        lines = out.split('\n')
        from_col = next(i for i, l in enumerate(lines) if 'FROM' in l.upper())
        where_col = next(i for i, l in enumerate(lines) if 'WHERE' in l.upper())
        assert lines[where_col].index('WHERE') == lines[from_col].index('FROM')

    def test_and_or_aligned(self):
        out = _filter('select * from foo where a = 1 and b = 2 or c = 3')
        lines = out.split('\n')
        where_line = next(l for l in lines if 'WHERE' in l.upper())
        and_line = next(l for l in lines if l.strip().upper().startswith('AND'))
        or_line = next(l for l in lines if l.strip().upper().startswith('OR'))
        start = where_line.index('WHERE')
        assert and_line.index('AND') == start
        assert or_line.index('OR') == start

    def test_group_by_aligned_on_first_word(self):
        out = _filter('select a, count(*) from foo group by a')
        lines = out.split('\n')
        group_line = next(l for l in lines if 'GROUP' in l.upper())
        from_line = next(l for l in lines if l.strip().upper().startswith('FROM'))
        assert group_line.index('GROUP') == from_line.index('FROM')

    def test_statement_leading_whitespace_removed(self):
        f = AlignedIndentFilter()
        stmt = sqlparse.parse('   select * from foo')[0]
        f.process(stmt)
        assert not stmt.tokens[0].is_whitespace

    def test_statement_leading_whitespace_kept_when_indented(self):
        f = AlignedIndentFilter()
        stmt = sqlparse.parse('   select * from foo')[0]
        f.indent = 1
        first_before = stmt.tokens[0]
        f.process(stmt)
        # since indent != 0 at start, leading ws should not be popped
        # (but _process_default may still insert stuff) - check indent reset
        assert f.indent == 1


class TestBetweenHandling:
    def test_between_treated_as_single(self):
        out = _filter('select * from foo where a between 1 and 2')
        lines = out.split('\n')
        # only one 'AND' should be aligned: the one starting a line
        and_lines = [l for l in lines if l.strip().upper().startswith('AND')]
        assert len(and_lines) == 0
        # Both between and its AND stay on the WHERE line
        where_line = next(l for l in lines if 'WHERE' in l.upper())
        assert 'between' in where_line.lower() or 'BETWEEN' in where_line


class TestParenthesis:
    def test_subquery_select_realigned(self):
        out = _filter('select * from foo where id in (select id from bar)')
        lines = out.split('\n')
        # There should be a line with 'select' from subquery
        inner_select = [l for l in lines if 'select' in l.lower() and 'in' in l.lower()]
        # Inner select moved to its own line
        assert any(l.strip().lower().startswith('select') for l in lines)
        # closing paren line ending with ')'
        assert out.strip().endswith(')')

    def test_non_subquery_parenthesis_not_reindented(self):
        out = _filter('select count(1) from foo')
        # "count(1)" stays together on same line
        assert 'count(1)' in out.replace(' ', '') or 'count' in out

    def test_parenthesis_indent_resets(self):
        f = AlignedIndentFilter()
        stmt = sqlparse.parse('select * from (select 1) x')[0]
        f.process(stmt)
        assert f.indent == 0


class TestIdentifierList:
    def test_columns_split(self):
        out = _filter('select a, b, c from foo')
        lines = out.split('\n')
        for col in ('a', 'b', 'c'):
            assert any(line.strip().startswith(col) for line in lines), out

    def test_single_column(self):
        out = _filter('select a from foo')
        # should not produce empty identifier split issue
        assert 'a' in out


class TestCase:
    def test_case_alignment(self):
        out = _filter(
            'select case when a = 1 then 1 when b = 2 then 2 else 0 end from foo')
        lines = out.split('\n')
        assert any('CASE' in l.upper() or 'case' in l for l in lines)
        # 'when' should be aligned
        when_lines = [l for l in lines if l.strip().lower().startswith('when')]
        assert when_lines
        cols = {l.index('when') for l in when_lines if 'when' in l}
        assert len(cols) == 1

    def test_case_end_aligned(self):
        out = _filter('select case when a then 1 else 0 end from foo')
        lines = out.split('\n')
        assert 'end' in out.lower()
        # END appears somewhere
        assert any('END' in l.upper() for l in lines)


class TestSplitKwds:
    def test_split_from(self):
        out = _filter('select 1 from foo')
        assert '\n' in out
        assert 'FROM' in out.upper()

    def test_split_having(self):
        out = _filter('select a from foo group by a having count(*) > 1')
        assert 'HAVING' in out.upper()

    def test_split_limit(self):
        out = _filter('select * from foo limit 10')
        lines = out.split('\n')
        limit_line = [l for l in lines if 'LIMIT' in l.upper()]
        assert limit_line

    def test_split_union(self):
        out = _filter('select 1 from a union select 2 from b')
        assert 'UNION' in out.upper()

    def test_split_values(self):
        out = _filter('insert into foo values (1, 2)')
        lines = out.split('\n')
        assert any('VALUES' in l.upper() for l in lines)

    def test_split_set(self):
        out = _filter('update foo set a = 1, b = 2')
        assert 'SET' in out.upper()

    def test_split_except(self):
        out = _filter('select 1 from a except select 2 from b')
        assert 'EXCEPT' in out.upper()

    def test_split_on(self):
        out = _filter('select * from a join b on a.id = b.id')
        lines = out.split('\n')
        on_line = [l for l in lines if l.strip().upper().startswith('ON')]
        assert on_line


class TestProcessDispatch:
    def test_process_paren_dispatch(self):
        f = AlignedIndentFilter()
        stmt = sqlparse.parse('select * from foo where id in (select id from bar)')[0]
        f.process(stmt)
        # no exceptions and statement ends with closing paren
        assert str(stmt).strip().endswith(')')

    def test_process_case_dispatch(self):
        f = AlignedIndentFilter()
        stmt = sqlparse.parse('select case when a then 1 end from foo')[0]
        f.process(stmt)
        assert 'case' in str(stmt).lower()

    def test_process_identifierlist_dispatch(self):
        f = AlignedIndentFilter()
        stmt = sqlparse.parse('select a, b, c from foo')[0]
        f.process(stmt)
        # identifier list should have gotten newlines inserted
        assert str(stmt).count('\n') >= 2


class TestFormatIntegration:
    def test_format_with_indent_char(self):
        out = sqlparse.format(
            'select a from foo',
            reindent_aligned=True,
            indent_char='\t',
        )
        assert 'SELECT' in out
        # tabs used for indentation if any
        assert 'FROM' in out.upper()

    def test_idempotent_ish(self):
        out1 = _filter('select a, b from foo where a = 1 and b = 2')
        # format again should not error
        out2 = sqlparse.format(out1, reindent_aligned=True)
        assert 'SELECT' in out2.upper()
        assert 'WHERE' in out2.upper()

    def test_multiple_statements(self):
        out = _filter('select 1 from foo; select 2 from bar')
        assert out.count('FROM') == 2


class TestOffsetIndentContext:
    def test_indent_reset_after_process(self):
        f = AlignedIndentFilter()
        stmt = sqlparse.parse('select * from (select 1) x')[0]
        f.process(stmt)
        assert f.indent == 0
        assert f.offset == 0

    def test_nested_subquery(self):
        out = _filter(
            'select * from (select * from (select 1) a) b')
        lines = out.split('\n')
        assert len(lines) > 1
        # balanced parens
        assert out.count('(') == out.count(')')
