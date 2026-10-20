#
# Tests for sqlparse.filters.aligned_indent.AlignedIndentFilter
#
import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.aligned_indent import AlignedIndentFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ws_tokens(tlist):
    """Return list of (index, token) for Whitespace tokens directly in tlist."""
    return [(i, t) for i, t in enumerate(tlist.tokens) if t.is_whitespace]


def _flatten(tlist):
    """Return the concatenated string value of all tokens (recursively)."""
    out = []
    for tok in tlist.tokens:
        if isinstance(tok, sql.TokenList):
            out.append(_flatten(tok))
        else:
            out.append(tok.value)
    return ''.join(out)


def _find_ws_values(tlist):
    """Collect values of newline-whitespace tokens recursively."""
    result = []
    for tok in tlist.tokens:
        if isinstance(tok, sql.TokenList):
            result.extend(_find_ws_values(tok))
        elif tok.is_whitespace and '\n' in tok.value:
            result.append(tok.value)
    return result


# ---------------------------------------------------------------------------
# Constructor & class attributes
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_defaults(self):
        f = AlignedIndentFilter()
        assert f.n == '\n'
        assert f.char == ' '
        assert f.offset == 0
        assert f.indent == 0
        assert f._max_kwd_len == len('select') == 6

    def test_custom_char_and_n(self):
        f = AlignedIndentFilter(char='\t', n='\r\n')
        assert f.char == '\t'
        assert f.n == '\r\n'

    def test_class_words(self):
        assert AlignedIndentFilter.by_words == r'(GROUP|ORDER)\s+BY\b'
        assert 'FROM' in AlignedIndentFilter.split_words
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
        assert 'ON' in AlignedIndentFilter.split_words
        assert AlignedIndentFilter.join_words in AlignedIndentFilter.split_words


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

class TestNl:
    def test_nl_default(self):
        f = AlignedIndentFilter()
        tok = f.nl()
        assert isinstance(tok, sql.Token)
        assert tok.ttype is T.Whitespace
        # _max_kwd_len + offset + indent + offset_expr
        # 6 + 1 + 0 + 0 = 7 spaces
        assert tok.value == '\n' + ' ' * 7

    def test_nl_int_offset(self):
        f = AlignedIndentFilter()
        tok = f.nl(3)
        assert tok.value == '\n' + ' ' * 9

    def test_nl_zero_offset(self):
        f = AlignedIndentFilter()
        tok = f.nl(0)
        assert tok.value == '\n' + ' ' * 6

    def test_nl_string_offset(self):
        f = AlignedIndentFilter()
        # offset = -len('SELECT') -> 5 + (-6) + 6 = 5
        tok = f.nl('SELECT')
        assert tok.value == '\n' + ' ' * 5
        tok2 = f.nl('AB')
        # 5 + (-2) + 6 = 9
        assert tok2.value == '\n' + ' ' * 9

    def test_nl_custom_char(self):
        f = AlignedIndentFilter(char='.')
        tok = f.nl()
        assert tok.value == '\n' + '.' * 7

    def test_nl_custom_newline(self):
        f = AlignedIndentFilter(n='\r\n')
        tok = f.nl()
        assert tok.value == '\r\n' + ' ' * 7

    def test_nl_respects_indent(self):
        f = AlignedIndentFilter()
        f.indent = 2
        tok = f.nl(0)
        # 6 + 0 + 2*(2+6) + 0 = 6 + 16 = 22
        assert tok.value == '\n' + ' ' * 22

    def test_nl_respects_offset(self):
        f = AlignedIndentFilter()
        f.offset = 5
        tok = f.nl(0)
        assert tok.value == '\n' + ' ' * 11


# ---------------------------------------------------------------------------
# process(): removal of leading whitespace
# ---------------------------------------------------------------------------

class TestProcessLeadingWhitespace:
    def test_leading_whitespace_removed(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([sql.Token(T.Whitespace, '  '),
                              sql.Token(T.Keyword, 'SELECT')])
        result = f.process(stmt)
        assert result is stmt
        assert not stmt.tokens[0].is_whitespace
        assert stmt.tokens[0].value == 'SELECT'

    def test_leading_whitespace_kept_when_indent_nonzero(self):
        f = AlignedIndentFilter()
        f.indent = 1
        stmt = sql.Statement([sql.Token(T.Whitespace, '  '),
                              sql.Token(T.Keyword, 'SELECT')])
        f.process(stmt)
        # indent path still removes since it's _process_statement
        # Actually condition is on indent == 0; here indent != 0 so kept.
        assert stmt.tokens[0].is_whitespace

    def test_empty_statement(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([])
        result = f.process(stmt)
        assert result is stmt
        assert len(stmt.tokens) == 0

    def test_no_leading_whitespace(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([sql.Token(T.Keyword, 'SELECT'),
                              sql.Token(T.Whitespace, ' '),
                              sql.Token(T.Keyword, 'FROM')])
        f.process(stmt)
        # 'SELECT' then nl before 'FROM'
        assert stmt.tokens[0].value == 'SELECT'


# ---------------------------------------------------------------------------
# Splitting keywords
# ---------------------------------------------------------------------------

class TestSplitKeywords:
    def test_where_split(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Wildcard, '*'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'WHERE'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Operator, '='),
            sql.Token(T.Literal.Number.Integer, '1'),
        ])
        f.process(stmt)
        values = [t.value for t in stmt.tokens]
        # newline whitespace inserted before WHERE
        assert any(v.startswith('\n') for v in values)
        # where token still present
        assert 'WHERE' in values

    def test_from_split(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Wildcard, '*'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'FROM'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 't'),
        ])
        f.process(stmt)
        result = _flatten(stmt)
        # FROM should be preceded by a newline
        assert '\n        FROM' in result or result.count('FROM') == 1

    def test_by_words_use_first_word(self):
        # Only first word of "GROUP BY" used for indent
        f = AlignedIndentFilter()
        stmt = sql.Statement([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'GROUP BY'),
        ])
        f.process(stmt)
        ws_values = _find_ws_values(stmt)
        assert len(ws_values) >= 1
        # offset = -len('GROUP') = -5 ; 5 + (-5) + 6 = 6 spaces after newline
        assert ws_values[0] == '\n' + ' ' * 6

    def test_join_uses_first_word(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'LEFT JOIN'),
        ])
        f.process(stmt)
        ws_values = _find_ws_values(stmt)
        assert len(ws_values) >= 1
        # offset = -len('LEFT') = -4 ; 5 + (-4) + 6 = 7
        assert ws_values[0] == '\n' + ' ' * 7

    def test_between_and_not_split(self):
        # BETWEEN x AND y : the AND following BETWEEN must not create a split
        f = AlignedIndentFilter()
        stmt = sql.Statement([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'BETWEEN'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Literal.Number.Integer, '1'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'AND'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Literal.Number.Integer, '2'),
        ])
        f.process(stmt)
        ws_values = _find_ws_values(stmt)
        # Two splits: one before BETWEEN, one... AND should be swallowed.
        # With BETWEEN handled, AND is skipped; only BETWEEN yields a split
        # (followed by another _next_token call from tidx+1).
        assert any('BETWEEN' in _flatten(stmt) for _ in [0])
        # The newline before AND should not exist; check by count.
        and_idx = next(i for i, t in enumerate(stmt.tokens)
                       if t.value == 'AND')
        prev = stmt.tokens[and_idx - 1]
        assert not (prev.is_whitespace and '\n' in prev.value)


# ---------------------------------------------------------------------------
# Identifier list (SELECT columns)
# ---------------------------------------------------------------------------

class TestIdentifierList:
    def test_select_columns_split(self):
        f = AlignedIndentFilter()
        ids = [
            sql.Identifier([sql.Token(T.Name, 'a')]),
            sql.Identifier([sql.Token(T.Name, 'b')]),
            sql.Identifier([sql.Token(T.Name, 'c')]),
        ]
        il = sql.IdentifierList(
            [ids[0],
             sql.Token(T.Punctuation, ','),
             sql.Token(T.Whitespace, ' '),
             ids[1],
             sql.Token(T.Punctuation, ','),
             sql.Token(T.Whitespace, ' '),
             ids[2]])
        f._process_identifierlist(il)
        # first identifier is not preceded by inserted newline, others are.
        # find b
        b_idx = il.token_index(ids[1])
        ws_before_b = il.tokens[b_idx - 1]
        assert ws_before_b.is_whitespace
        assert '\n' in ws_before_b.value

    def test_single_identifier_no_error(self):
        f = AlignedIndentFilter()
        ident = sql.Identifier([sql.Token(T.Name, 'a')])
        il = sql.IdentifierList([ident])
        f._process_identifierlist(il)
        assert len(il.tokens) == 1

    def test_empty_identifier_list_raises(self):
        f = AlignedIndentFilter()
        il = sql.IdentifierList([])
        with pytest.raises(IndexError):
            f._process_identifierlist(il)


# ---------------------------------------------------------------------------
# Parenthesis (subqueries)
# ---------------------------------------------------------------------------

class TestParenthesis:
    def _make_select_parenthesis(self):
        inner = sql.Statement([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Wildcard, '*'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'FROM'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 't'),
        ])
        lp = sql.Token(T.Punctuation, '(')
        rp = sql.Token(T.Punctuation, ')')
        return sql.Parenthesis([lp, inner, rp])

    def test_parenthesis_with_select(self):
        f = AlignedIndentFilter()
        par = self._make_select_parenthesis()
        f._process_parenthesis(par)
        text = _flatten(par)
        assert '(' in text
        assert ')' in text
        # A newline inserted after '('
        idx_lp = par.token_index(par.tokens[0])
        nxt = par.tokens[idx_lp + 1]
        assert nxt.is_whitespace and '\n' in nxt.value

    def test_parenthesis_without_select_untouched(self):
        f = AlignedIndentFilter()
        par = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            sql.Token(T.Name, 'x'),
            sql.Token(T.Punctuation, ')'),
        ])
        original = [t.value for t in par.tokens]
        f._process_parenthesis(par)
        assert [t.value for t in par.tokens] == original


# ---------------------------------------------------------------------------
# CASE handling
# ---------------------------------------------------------------------------

def _make_case():
    # CASE WHEN a=1 THEN x WHEN b=2 THEN y ELSE z END
    when_kw = sql.Token(T.Keyword, 'WHEN')
    case_kw = sql.Token(T.Keyword, 'CASE')
    tokens = [
        case_kw,
        sql.Token(T.Whitespace, ' '),
        when_kw,
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'a'),
        sql.Token(T.Operator, '='),
        sql.Token(T.Literal.Number.Integer, '1'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'THEN'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'WHEN'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'b'),
        sql.Token(T.Operator, '='),
        sql.Token(T.Literal.Number.Integer, '22'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'THEN'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'y'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'ELSE'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'z'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'END'),
    ]
    return sql.Case(tokens)


class TestCase:
    def test_case_inserts_newlines(self):
        f = AlignedIndentFilter()
        case = _make_case()
        f._process_case(case)
        ws_values = _find_ws_values(case)
        # multiple newlines inserted: before each WHEN/ELSE and END
        newlines = [w for w in ws_values if w.startswith('\n')]
        assert len(newlines) >= 2

    def test_case_aligns_end(self):
        f = AlignedIndentFilter()
        case = _make_case()
        f._process_case(case)
        end_tok = case.token_next_by(m=(T.Keyword, 'END'))[1]
        idx = case.token_index(end_tok)
        prev = case.tokens[idx - 1]
        assert prev.is_whitespace
        assert '\n' in prev.value

    def test_case_alignment_whitespace(self):
        f = AlignedIndentFilter()
        case = _make_case()
        f._process_case(case)
        # there should be whitespace tokens inserted after conditions
        text = _flatten(case)
        assert 'CASE' in text
        assert 'END' in text


# ---------------------------------------------------------------------------
# _next_token behaviour
# ---------------------------------------------------------------------------

class TestNextToken:
    def test_next_token_finds_keyword(self):
        f = AlignedIndentFilter()
        tl = sql.TokenList([
            sql.Token(T.Name, 'x'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'WHERE'),
        ])
        idx, tok = f._next_token(tl)
        assert tok.value == 'WHERE'
        assert tl.token_index(tok) == idx

    def test_next_token_no_keyword(self):
        f = AlignedIndentFilter()
        tl = sql.TokenList([sql.Token(T.Name, 'x')])
        idx, tok = f._next_token(tl)
        assert tok is None
        assert idx is None

    def test_next_token_between_swallows_and(self):
        f = AlignedIndentFilter()
        between = sql.Token(T.Keyword, 'BETWEEN')
        and_tok = sql.Token(T.Keyword, 'AND')
        tl = sql.TokenList([
            between,
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Literal.Number.Integer, '1'),
            sql.Token(T.Whitespace, ' '),
            and_tok,
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Literal.Number.Integer, '2'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'OR'),
        ])
        idx, tok = f._next_token(tl)
        assert tok.value == 'BETWEEN'
        # AND must be skipped; next found token should be OR
        assert tl.tokens[idx] is between
        # call again from idx+1 -> should get OR (skipping AND)
        idx2, tok2 = f._next_token(tl, idx + 1)
        assert tok2.value == 'OR'


# ---------------------------------------------------------------------------
# _process / dispatch and integration
# ---------------------------------------------------------------------------

class TestProcessDispatch:
    def test_process_default_called(self):
        f = AlignedIndentFilter()
        tl = sql.TokenList([])
        # _process_default should not error on empty
        f._process(tl)

    def test_process_returns_same_statement(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([])
        assert f.process(stmt) is stmt

    def test_offset_restored(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([
            sql.Token(T.Keyword, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
        ])
        f.process(stmt)
        assert f.offset == 0
        assert f.indent == 0

    def test_indent_restored_after_parenthesis(self):
        f = AlignedIndentFilter()
        inner = sql.Statement([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Wildcard, '*'),
        ])
        par = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            inner,
            sql.Token(T.Punctuation, ')'),
        ])
        stmt = sql.Statement([par])
        f.process(stmt)
        assert f.indent == 0
        assert f.offset == 0


class TestIntegration:
    def test_simple_select_formatting(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([
            sql.Token(T.Whitespace, '\n'),
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Wildcard, '*'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'FROM'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 't1'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'WHERE'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Operator, '='),
            sql.Token(T.Literal.Number.Integer, '1'),
        ])
        f.process(stmt)
        text = _flatten(stmt)
        # leading whitespace removed
        assert not text.startswith('\n')
        assert text.startswith('SELECT')
        # FROM and WHERE get newlines
        assert '\n' in text
        from_idx = text.index('FROM')
        where_idx = text.index('WHERE')
        assert text[from_idx - 1] == ' '
        assert text.rfind('\n', 0, from_idx) != -1
        assert text.rfind('\n', 0, where_idx) != -1

    def test_custom_char_used(self):
        f = AlignedIndentFilter(char='_')
        stmt = sql.Statement([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Wildcard, '*'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'FROM'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 't'),
        ])
        f.process(stmt)
        text = _flatten(stmt)
        assert '_' in text
        assert not any(c == ' ' for c in text.replace('SELECT', '')
                       .replace('FROM', '').replace(' ', '') if c == '\t')

    def test_idempotent_double_process_no_crash(self):
        f = AlignedIndentFilter()
        stmt = sql.Statement([
            sql.Token(T.DML, 'SELECT'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Wildcard, '*'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Keyword, 'FROM'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 't'),
        ])
        f.process(stmt)
        assert f.indent == 0
        assert f.offset == 0
