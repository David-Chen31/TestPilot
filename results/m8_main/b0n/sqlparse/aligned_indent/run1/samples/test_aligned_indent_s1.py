import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.aligned_indent import AlignedIndentFilter


def make_filter(**kwargs):
    return AlignedIndentFilter(**kwargs)


# ---------------------------------------------------------------------------
# __init__ / attributes
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        f = make_filter()
        assert f.char == ' '
        assert f.n == '\n'
        assert f.offset == 0
        assert f.indent == 0
        assert f._max_kwd_len == len('select')

    def test_custom_char(self):
        f = make_filter(char='\t')
        assert f.char == '\t'

    def test_custom_newline(self):
        f = make_filter(n='\r\n')
        assert f.n == '\r\n'

    def test_class_attributes(self):
        assert AlignedIndentFilter.by_words == r'(GROUP|ORDER)\s+BY\b'
        assert 'FROM' in AlignedIndentFilter.split_words
        assert 'WHERE' in AlignedIndentFilter.split_words
        assert 'AND' in AlignedIndentFilter.split_words
        assert AlignedIndentFilter.join_words.startswith(r'((LEFT\s+')


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

class TestNl:
    def test_int_offset(self):
        f = make_filter()
        tok = f.nl(1)
        assert isinstance(tok, sql.Token)
        assert tok.ttype == T.Whitespace
        # indent 0, offset 0: '\n' + 6 spaces ('select' -> 6) + 1
        assert tok.value == '\n' + ' ' * (6 + 1)

    def test_zero_offset_int(self):
        f = make_filter()
        assert f.nl(0).value == '\n' + ' ' * 6

    def test_string_offset(self):
        f = make_filter()
        # offset = -len('SELECT') = -6
        assert f.nl('SELECT').value == '\n' + ' ' * (6 - 6)

    def test_string_offset_negative(self):
        f = make_filter()
        # -len('WHERE') = -5 -> 6 - 5 = 1
        assert f.nl('WHERE').value == '\n' + ' ' * 1

    def test_offset_with_indent(self):
        f = make_filter()
        f.indent = 2
        # indent = 2 * (2 + 6) = 16
        assert f.nl(1).value == '\n' + ' ' * (6 + 1 + 16)

    def test_offset_with_offset_attr(self):
        f = make_filter()
        f.offset = 3
        assert f.nl(0).value == '\n' + ' ' * (6 + 0 + 0 + 3)

    def test_custom_char_used(self):
        f = make_filter(char='.')
        assert f.nl(0).value == '\n' + '.' * 6

    def test_custom_newline_used(self):
        f = make_filter(n='<NL>')
        assert f.nl(0).value == '<NL>' + ' ' * 6


# ---------------------------------------------------------------------------
# process / _process dispatch
# ---------------------------------------------------------------------------

class TestProcess:
    def test_process_returns_stmt(self):
        f = make_filter()
        stmt = sql.TokenList([])
        result = f.process(stmt)
        assert result is stmt

    def test_process_statement_pop_leading_whitespace(self):
        f = make_filter()
        stmt = sql.Statement([
            sql.Token(T.Whitespace, '   '),
            sql.Keyword('SELECT', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
        ])
        f.process(stmt)
        # leading whitespace removed since indent == 0
        assert stmt.tokens[0].value == 'SELECT'

    def test_process_statement_keeps_leading_whitespace_when_indented(self):
        f = make_filter()
        f.indent = 1
        stmt = sql.Statement([
            sql.Token(T.Whitespace, '   '),
            sql.Keyword('SELECT', T.Keyword),
        ])
        f.process(stmt)
        assert stmt.tokens[0].value == '   '

    def test_process_statement_empty(self):
        f = make_filter()
        stmt = sql.Statement([])
        # Should not raise
        f.process(stmt)
        assert len(stmt.tokens) == 0


# ---------------------------------------------------------------------------
# _split_kwds / _next_token (via _process_default)
# ---------------------------------------------------------------------------

class TestSplitKeywords:
    def _values(self, tlist):
        return [t.value for t in tlist.tokens]

    def test_from_splits(self):
        f = make_filter()
        tlist = sql.TokenList([
            sql.Keyword('SELECT', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('FROM', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Identifier('t'),
        ])
        f._process_default(tlist)
        # a newline token inserted before FROM
        idx = tlist.token_index(tlist.tokens[-2])
        prev = tlist.tokens[idx - 1]
        assert isinstance(prev, sql.Token)
        assert prev.ttype == T.Whitespace
        assert prev.value.startswith('\n')

    def test_multiple_keywords_split(self):
        f = make_filter()
        tlist = sql.TokenList([
            sql.Keyword('SELECT', T.Keyword),
            sql.Keyword('WHERE', T.Keyword),
            sql.Keyword('AND', T.Keyword),
        ])
        f._process_default(tlist)
        values = self._values(tlist)
        # SELECT is not a split word; WHERE and AND are
        assert values.count('WHERE') == 1
        assert values.count('AND') == 1
        # Inserted whitespace tokens start with '\n'
        newlines = [v for v in values if isinstance(v, str) and v.startswith('\n')]
        assert len(newlines) == 2

    def test_join_uses_first_word(self):
        f = make_filter()
        tlist = sql.TokenList([
            sql.Keyword('LEFT JOIN', T.Keyword),
        ])
        f._process_default(tlist)
        ws = tlist.tokens[0]
        assert ws.value.startswith('\n')
        # token_indent is 'LEFT' -> offset = -4 -> 6-4 = 2 spaces
        assert ws.value == '\n' + ' ' * (6 - len('LEFT'))

    def test_group_by_uses_first_word(self):
        f = make_filter()
        tlist = sql.TokenList([
            sql.Keyword('GROUP BY', T.Keyword),
        ])
        f._process_default(tlist)
        ws = tlist.tokens[0]
        assert ws.value == '\n' + ' ' * (6 - len('GROUP'))

    def test_between_and_skipped(self):
        f = make_filter()
        tlist = sql.TokenList([
            sql.Keyword('BETWEEN', T.Keyword),
            sql.Token(T.Number, '1'),
            sql.Keyword('AND', T.Keyword),
            sql.Token(T.Number, '2'),
            sql.Keyword('WHERE', T.Keyword),
        ])
        f._process_default(tlist)
        values = [t.value for t in tlist.tokens]
        # Only BETWEEN and WHERE should be split, not the AND inside BETWEEN
        newlines = [v for v in values if v.startswith('\n')]
        assert len(newlines) == 2

    def test_between_without_and(self):
        f = make_filter()
        tlist = sql.TokenList([
            sql.Keyword('BETWEEN', T.Keyword),
            sql.Token(T.Number, '1'),
        ])
        f._process_default(tlist)
        values = [t.value for t in tlist.tokens]
        newlines = [v for v in values if isinstance(v, str) and v.startswith('\n')]
        assert len(newlines) == 1


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

class TestIdentifierList:
    def test_identifierlist_splits_after_first(self):
        f = make_filter()
        tlist = sql.IdentifierList([
            sql.Identifier('a'),
            sql.Token(T.Punctuation, ','),
            sql.Token(T.Whitespace, ' '),
            sql.Identifier('b'),
            sql.Token(T.Punctuation, ','),
            sql.Token(T.Whitespace, ' '),
            sql.Identifier('c'),
        ])
        f._process(tlist)
        # For identifiers b and c, a newline token is inserted before each.
        newlines = [t for t in tlist.tokens
                    if isinstance(t, sql.Token)
                    and t.ttype == T.Whitespace
                    and t.value.startswith('\n')]
        assert len(newlines) == 2

    def test_identifierlist_single(self):
        f = make_filter()
        tlist = sql.IdentifierList([sql.Identifier('a')])
        f._process(tlist)
        newlines = [t for t in tlist.tokens
                    if isinstance(t, sql.Token)
                    and t.ttype == T.Whitespace
                    and t.value.startswith('\n')]
        assert len(newlines) == 0


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

class TestCase:
    def test_case_alignment(self):
        f = make_filter()
        case = sql.Case([
            sql.Keyword('CASE', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('WHEN', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('THEN', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Identifier('x'),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('ELSE', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Identifier('y'),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('END', T.Keyword),
        ])
        f._process(case)
        # Should not raise and should have inserted whitespace tokens
        values = [t.value for t in case.tokens]
        assert 'CASE' in values
        assert 'END' in values
        # at least one inserted newline whitespace token
        assert any(isinstance(v, str) and v.startswith('\n') for v in values)

    def test_case_single_when(self):
        f = make_filter()
        case = sql.Case([
            sql.Keyword('CASE', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('WHEN', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('THEN', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Identifier('x'),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('END', T.Keyword),
        ])
        f._process(case)
        values = [t.value for t in case.tokens]
        assert 'CASE' in values
        assert 'END' in values


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

class TestParenthesis:
    def test_subquery_parenthesis_indented(self):
        f = make_filter()
        inner = sql.TokenList([
            sql.Keyword('SELECT', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
        ])
        paren = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            inner,
            sql.Token(T.Punctuation, ')'),
        ])
        f._process(paren)
        # A newline inserted after '(' and before ')'
        assert paren.tokens[0].value == '('
        assert paren.tokens[1].value.startswith('\n')

    def test_non_select_parenthesis_not_indented(self):
        f = make_filter()
        paren = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            sql.Token(T.Number, '1'),
            sql.Token(T.Punctuation, ')'),
        ])
        original_len = len(paren.tokens)
        f._process(paren)
        assert len(paren.tokens) == original_len


# ---------------------------------------------------------------------------
# _process_default offset hack for GROUP/ORDER BY sublists
# ---------------------------------------------------------------------------

class TestByOffsetHack:
    def test_offset_applied_after_by_keyword(self):
        f = make_filter()
        sub = sql.TokenList([sql.Token(T.Number, '1')])
        tlist = sql.TokenList([
            sql.Keyword('ORDER BY', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sub,
        ])
        f._process_default(tlist)
        # After processing, offset should be restored to 0
        assert f.offset == 0
        assert f.indent == 0


# ---------------------------------------------------------------------------
# process full statement
# ---------------------------------------------------------------------------

class TestProcessStatement:
    def test_full_statement(self):
        f = make_filter()
        stmt = sql.Statement([
            sql.Keyword('SELECT', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('FROM', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Identifier('t'),
        ])
        result = f.process(stmt)
        assert result is stmt
        str_out = str(stmt)
        assert '\n' in str_out
        # FROM aligned at 6 - len('FROM') = 2 spaces
        assert '\n  FROM' in str_out

    def test_process_does_not_mutate_indent_state(self):
        f = make_filter()
        stmt = sql.Statement([
            sql.Keyword('SELECT', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
        ])
        f.process(stmt)
        assert f.indent == 0
        assert f.offset == 0

    def test_process_with_subquery(self):
        f = make_filter()
        stmt = sql.Statement([
            sql.Keyword('SELECT', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Number, '1'),
            sql.Token(T.Whitespace, ' '),
            sql.Keyword('FROM', T.Keyword),
            sql.Token(T.Whitespace, ' '),
            sql.Parenthesis([
                sql.Token(T.Punctuation, '('),
                sql.TokenList([
                    sql.Keyword('SELECT', T.Keyword),
                    sql.Token(T.Whitespace, ' '),
                    sql.Token(T.Number, '2'),
                ]),
                sql.Token(T.Punctuation, ')'),
            ]),
        ])
        result = f.process(stmt)
        assert result is stmt
        assert f.indent == 0
        assert f.offset == 0

    def test_process_empty_statement(self):
        f = make_filter()
        stmt = sql.Statement([])
        result = f.process(stmt)
        assert result is stmt
        assert len(stmt.tokens) == 0
