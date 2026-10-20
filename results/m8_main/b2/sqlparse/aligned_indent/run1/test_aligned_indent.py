import pytest

from sqlparse import sql, tokens as T
from sqlparse.filters.aligned_indent import AlignedIndentFilter
from sqlparse.utils import indent, offset


class TestInit:
    def test_defaults(self):
        f = AlignedIndentFilter()
        assert f.n == '\n'
        assert f.char == ' '
        assert f.offset == 0
        assert f.indent == 0
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
        assert tok.ttype == T.Whitespace
        # max_kwd_len (6) + offset (1) + indent (0) + self.offset (0)
        assert tok.value == '\n' + ' ' * 7

    def test_nl_int_offset(self):
        f = AlignedIndentFilter()
        tok = f.nl(3)
        assert tok.value == '\n' + ' ' * 9

    def test_nl_string_offset(self):
        f = AlignedIndentFilter()
        tok = f.nl('SELECT')
        # offset = -len('SELECT') = -6
        assert tok.value == '\n' + ' ' * (6 - 6 + 0 + 0)

    def test_nl_with_indent(self):
        f = AlignedIndentFilter()
        f.indent = 2
        tok = f.nl()
        # indent*(2 + max_kwd_len) = 2*8 = 16
        assert tok.value == '\n' + ' ' * (6 + 1 + 16)

    def test_nl_with_offset(self):
        f = AlignedIndentFilter()
        f.offset = 3
        tok = f.nl()
        assert tok.value == '\n' + ' ' * (6 + 1 + 3)

    def test_nl_custom_char(self):
        f = AlignedIndentFilter(char='*')
        tok = f.nl()
        assert tok.value == '\n' + '*' * 7

    def test_nl_custom_n(self):
        f = AlignedIndentFilter(n='\r\n')
        tok = f.nl()
        assert tok.value == '\r\n' + ' ' * 7


class TestProcessStatement:
    def _build_list(self):
        stmt = sql.Statement()
        stmt.tokens = [sql.Token(T.Keyword, 'SELECT'),
                       sql.Token(T.Whitespace, ' '),
                       sql.Token(T.Number, '1')]
        return stmt

    def test_pops_leading_whitespace(self):
        f = AlignedIndentFilter()
        stmt = self._build_list()
        stmt.tokens.insert(0, sql.Token(T.Whitespace, '\n'))
        res = f.process(stmt)
        assert res is stmt
        assert stmt.tokens[0].value == 'SELECT'

    def test_does_not_pop_when_indented(self):
        f = AlignedIndentFilter()
        f.indent = 1
        stmt = self._build_list()
        stmt.tokens.insert(0, sql.Token(T.Whitespace, '\n'))
        f.process(stmt)
        assert stmt.tokens[0].value == '\n'

    def test_empty_statement(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement()
        res = f.process(stmt)
        assert res is stmt
        assert stmt.tokens == []


class TestProcessParenthesis:
    def test_subquery_reindented(self):
        f = AlignedIndentFilter()
        # A Parenthesis-like TokenList has an opening token first, which
        # tlist[0] refers to. Construct with a leading '(' token so
        # insert_after(tlist[0], ...) works properly.
        inner = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
            sql.Token(T.Punctuation, ')'),
        ])
        res = f._process_parenthesis(inner)
        assert res is None
        # after first '(' token, a newline whitespace token is inserted
        assert inner.tokens[1].ttype == T.Whitespace
        assert inner.tokens[1].value.startswith('\n')

    def test_non_select_parenthesis_not_reindented(self):
        f = AlignedIndentFilter()
        inner = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            sql.Token(T.Number, '1'),
            sql.Token(T.Punctuation, ')'),
        ])
        before = list(inner.tokens)
        f._process_parenthesis(inner)
        assert inner.tokens == before


class TestProcessIdentifierlist:
    def test_identifiers_split_on_newline(self):
        f = AlignedIndentFilter()
        idlist = sql.IdentifierList([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Punctuation, ','),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'b'),
            sql.Token(T.Punctuation, ','),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'c'),
        ])
        f._process_identifierlist(idlist)
        values = [t.value for t in idlist.tokens]
        # two newline whitespace tokens inserted for 'b' and 'c'
        newline_ws = [t for t in idlist.tokens
                      if t.ttype == T.Whitespace and t.value.startswith('\n')]
        assert len(newline_ws) == 2
        assert len(values) == 9


class TestProcessCase:
    def test_case_alignment(self):
        f = AlignedIndentFilter()
        tokens = [
            sql.Token(T.Keyword, 'CASE'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'WHEN'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'THEN'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'WHEN'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'b'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'THEN'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '2'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'END'),
        ]
        case = sql.Case(tokens)
        f._process_case(case)
        # A newline whitespace token inserted before 'WHEN b' and before 'END'
        ws_tokens = [t for t in case.tokens if t.ttype == T.Whitespace
                     and t.value.startswith('\n')]
        assert len(ws_tokens) >= 2


class TestNextToken:
    def test_finds_select(self):
        # _next_token matches against T.Keyword and split_words, not T.DML,
        # so a DML 'SELECT' token won't match; use a Keyword FROM instead.
        f = AlignedIndentFilter()
        tl = sql.TokenList([sql.Token(T.Keyword, 'FROM'),
                            sql.Token(T.Whitespace, ' '),
                            sql.Token(T.Name, 'x')])
        idx, tok = f._next_token(tl)
        assert tok is not None
        assert tok.normalized == 'FROM'
        assert idx == 0

    def test_between_skips_to_and(self):
        f = AlignedIndentFilter()
        tl = sql.TokenList([
            sql.Token(T.Keyword, 'BETWEEN'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'AND'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'x'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'FROM'),
        ])
        idx, tok = f._next_token(tl)
        # BETWEEN consumes the nested AND, so the next match is 'FROM'.
        assert tok is not None
        assert tok.normalized == 'FROM'

    def test_no_keyword_returns_none(self):
        f = AlignedIndentFilter()
        tl = sql.TokenList([sql.Token(T.Name, 'foo'),
                            sql.Token(T.Whitespace, ' '),
                            sql.Token(T.Name, 'bar')])
        idx, tok = f._next_token(tl)
        assert tok is None


class TestSplitKwds:
    def test_split_from_where(self):
        f = AlignedIndentFilter()
        tl = sql.TokenList([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'x'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'FROM'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 't'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'WHERE'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'y'),
        ])
        f._split_kwds(tl)
        newline_ws = [t for t in tl.tokens
                      if t.ttype == T.Whitespace and t.value.startswith('\n')]
        # one for FROM, one for WHERE
        assert len(newline_ws) == 2

    def test_group_by_uses_first_word(self):
        f = AlignedIndentFilter()
        tl = sql.TokenList([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'x'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'GROUP BY'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'x'),
        ])
        f._split_kwds(tl)
        newline_ws = [t for t in tl.tokens
                      if t.ttype == T.Whitespace and t.value.startswith('\n')]
        assert len(newline_ws) == 1
        # Group-by newline should be computed from 'GROUP' (len 5), so the
        # resulting padding is max_kwd_len + (-5) + indent + offset
        assert len(newline_ws[0].value) - 1 == 6 - 5 + 0 + 0


class TestProcessDefault:
    def test_processes_sublists(self):
        f = AlignedIndentFilter()
        inner = sql.TokenList([sql.Token(T.Keyword, 'WHERE'),
                               sql.Token(T.Whitespace, ' '),
                               sql.Token(T.Name, 'x')])
        tl = sql.TokenList([sql.Token(T.DML, 'SELECT'),
                            sql.Token(T.Whitespace, ' '),
                            inner])
        f._process_default(tl)
        # inner has WHERE -> newline inserted inside
        assert any(t.ttype == T.Whitespace and t.value.startswith('\n')
                   for t in inner.tokens)

    def test_by_words_offset_hack(self):
        f = AlignedIndentFilter()
        by_kw = sql.Token(T.Keyword, 'GROUP BY')
        inner = sql.TokenList([sql.Token(T.Keyword, 'ON'),
                               sql.Token(T.Whitespace, ' '),
                               sql.Token(T.Name, 'x')])
        tl = sql.TokenList([by_kw, sql.Token(T.Whitespace, ' '), inner])
        offset_before = f.offset
        f._process_default(tl)
        assert f.offset == offset_before


class TestProcessDispatch:
    def test_dispatch_default(self):
        f = AlignedIndentFilter()
        tl = sql.TokenList([sql.Token(T.Keyword, 'WHERE'),
                            sql.Token(T.Whitespace, ' '),
                            sql.Token(T.Name, 'x')])
        # TokenList has no _process_tokenlist method -> _process_default
        f._process(tl)
        assert any(t.ttype == T.Whitespace and t.value.startswith('\n')
                   for t in tl.tokens)

    def test_dispatch_statement(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement()
        stmt.tokens = [sql.Token(T.Whitespace, '\n'),
                       sql.Token(T.Keyword, 'SELECT'),
                       sql.Token(T.Whitespace, ' '),
                       sql.Token(T.Number, '1')]
        f._process(stmt)
        assert stmt.tokens[0].value == 'SELECT'


class TestProcessPublic:
    def test_returns_same_object(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement()
        stmt.tokens = [sql.Token(T.Keyword, 'SELECT'),
                       sql.Token(T.Whitespace, ' '),
                       sql.Token(T.Number, '1')]
        assert f.process(stmt) is stmt

    def test_does_not_mutate_indent_offset_after(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement()
        stmt.tokens = [sql.Token(T.Keyword, 'SELECT')]
        f.process(stmt)
        assert f.indent == 0
        assert f.offset == 0


class TestContextManagers:
    def test_indent_context(self):
        f = AlignedIndentFilter()
        with indent(f):
            assert f.indent == 1
        assert f.indent == 0

    def test_offset_context(self):
        f = AlignedIndentFilter()
        with offset(f, 3):
            assert f.offset == 3
        assert f.offset == 0


class TestSplitWords:
    def test_join_words_constant(self):
        assert 'JOIN' in AlignedIndentFilter.join_words
        assert 'LEFT' in AlignedIndentFilter.join_words

    def test_by_words_constant(self):
        assert 'GROUP' in AlignedIndentFilter.by_words
        assert 'ORDER' in AlignedIndentFilter.by_words

    def test_split_words_contains_keywords(self):
        for kw in ('FROM', 'ON', 'WHERE', 'AND', 'OR', 'HAVING',
                   'LIMIT', 'UNION', 'VALUES', 'SET', 'BETWEEN',
                   'EXCEPT'):
            assert kw in AlignedIndentFilter.split_words
