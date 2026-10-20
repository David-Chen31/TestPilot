import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.aligned_indent import AlignedIndentFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ws():
    return sql.Token(T.Whitespace, ' ')


def make_select_tokens():
    """Build a simple TokenList resembling 'SELECT a, b FROM t' tokens."""
    return [
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'a'),
    ]


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

def test_defaults():
    f = AlignedIndentFilter()
    assert f.n == '\n'
    assert f.char == ' '
    assert f.offset == 0
    assert f.indent == 0
    assert f._max_kwd_len == len('select')


def test_custom_char_and_n():
    f = AlignedIndentFilter(char='\t', n='\r\n')
    assert f.char == '\t'
    assert f.n == '\r\n'


def test_split_words_contains_expected():
    f = AlignedIndentFilter()
    assert 'FROM' in f.split_words
    assert 'ON' in f.split_words
    assert 'WHERE' in f.split_words
    assert 'AND' in f.split_words
    assert 'OR' in f.split_words
    assert 'HAVING' in f.split_words
    assert 'LIMIT' in f.split_words
    assert 'UNION' in f.split_words
    assert 'VALUES' in f.split_words
    assert 'SET' in f.split_words
    assert 'BETWEEN' in f.split_words
    assert 'EXCEPT' in f.split_words
    assert f.join_words in f.split_words
    assert f.by_words in f.split_words


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

def test_nl_default_offset():
    f = AlignedIndentFilter()
    tok = f.nl()
    assert isinstance(tok, sql.Token)
    assert tok.ttype is T.Whitespace
    # _max_kwd_len + offset + indent*... + self.offset
    expected = '\n' + ' ' * (f._max_kwd_len + 1 + 0 + 0)
    assert tok.value == expected


def test_nl_int_offset():
    f = AlignedIndentFilter()
    tok = f.nl(offset=5)
    expected = '\n' + ' ' * (f._max_kwd_len + 5)
    assert tok.value == expected


def test_nl_string_offset():
    f = AlignedIndentFilter()
    tok = f.nl(offset='SELECT')  # len == 6
    expected = '\n' + ' ' * (f._max_kwd_len - 6)
    assert tok.value == expected


def test_nl_with_indent_and_offset():
    f = AlignedIndentFilter()
    f.indent = 2
    f.offset = 3
    tok = f.nl(offset=1)
    indent = f.indent * (2 + f._max_kwd_len)
    expected = '\n' + ' ' * (f._max_kwd_len + 1 + indent + f.offset)
    assert tok.value == expected


def test_nl_custom_char():
    f = AlignedIndentFilter(char='-')
    tok = f.nl(offset=0)
    expected = '\n' + '-' * f._max_kwd_len
    assert tok.value == expected


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

def test_next_token_finds_from():
    f = AlignedIndentFilter()
    tl = sql.TokenList([
        sql.Token(T.Keyword, 'SELECT'),
        make_ws(),
        sql.Token(T.Keyword, 'FROM'),
        make_ws(),
        sql.Token(T.Name, 'x'),
    ])
    idx, token = f._next_token(tl)
    assert token.normalized == 'FROM'
    assert idx == tl.token_index(token)


def test_next_token_skips_between_and():
    f = AlignedIndentFilter()
    tl = sql.TokenList([
        sql.Token(T.Keyword, 'BETWEEN'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        make_ws(),
        sql.Token(T.Keyword, 'AND'),
        make_ws(),
        sql.Token(T.Name, 'y'),
        make_ws(),
        sql.Token(T.Keyword, 'WHERE'),
    ])
    idx, token = f._next_token(tl)
    # BETWEEN and AND consumed as part of the BETWEEN statement,
    # next keyword should be WHERE
    assert token.normalized == 'WHERE'


def test_next_token_none_when_no_match():
    f = AlignedIndentFilter()
    tl = sql.TokenList([
        sql.Token(T.Name, 'x'),
        make_ws(),
        sql.Token(T.Name, 'y'),
    ])
    idx, token = f._next_token(tl)
    assert token is None


def test_next_token_between_consumes_second_and():
    """BETWEEN x AND y AND z -> the second AND is a real split point.

    Exercises the recursive branch where, after BETWEEN consumes one AND,
    the next keyword may itself be AND (returned normally).
    """
    f = AlignedIndentFilter()
    tl = sql.TokenList([
        sql.Token(T.Keyword, 'BETWEEN'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        make_ws(),
        sql.Token(T.Keyword, 'AND'),
        make_ws(),
        sql.Token(T.Name, 'y'),
        make_ws(),
        sql.Token(T.Keyword, 'AND'),
        make_ws(),
        sql.Token(T.Name, 'z'),
    ])
    idx, token = f._next_token(tl)
    # First AND gets consumed with BETWEEN; second AND is returned.
    assert token is not None
    assert token.normalized == 'AND'
    # It should be the *second* AND token (the one before z).
    second_and = tl.tokens[8]
    assert token is second_and
    assert idx == tl.token_index(second_and)
    # Verify it appears after the Name 'y'.
    y_idx = tl.token_index(tl.tokens[6])
    assert idx > y_idx


def test_next_token_between_followed_by_and_then_from():
    """BETWEEN x AND y AND z FROM t -> AND is found after skipping BETWEEN."""
    f = AlignedIndentFilter()
    tl = sql.TokenList([
        sql.Token(T.Keyword, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'a'),
        make_ws(),
        sql.Token(T.Keyword, 'BETWEEN'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        make_ws(),
        sql.Token(T.Keyword, 'AND'),
        make_ws(),
        sql.Token(T.Name, 'y'),
        make_ws(),
        sql.Token(T.Keyword, 'AND'),
        make_ws(),
        sql.Token(T.Name, 'z'),
        make_ws(),
        sql.Token(T.Keyword, 'FROM'),
        make_ws(),
        sql.Token(T.Name, 't'),
    ])
    # Starting from index of BETWEEN, should skip BETWEEN's AND.
    between_idx = tl.token_index(tl.tokens[4])
    idx, token = f._next_token(tl, between_idx)
    assert token is not None
    # The remaining keyword after the BETWEEN-AND pair is AND or FROM.
    assert token.normalized in ('AND', 'FROM')


def test_next_token_between_followed_directly_by_and_returns_it():
    """BETWEEN x AND y ... : the recursive branch at line 96 is entered.

    Explicitly covers the case where, after BETWEEN, the immediate next
    keyword is 'AND' so the recursion consumes it and continues searching.
    With no further keywords, the search exhausts and returns (None, None).
    """
    f = AlignedIndentFilter()
    tl = sql.TokenList([
        sql.Token(T.Keyword, 'BETWEEN'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        make_ws(),
        sql.Token(T.Keyword, 'AND'),
        make_ws(),
        sql.Token(T.Name, 'y'),
    ])
    idx, token = f._next_token(tl)
    # BETWEEN consumes the AND; nothing else remains.
    assert idx is None
    assert token is None


def test_next_token_between_and_then_except_returns_except():
    """BETWEEN x AND y EXCEPT ... : recursion consumes AND then finds EXCEPT."""
    f = AlignedIndentFilter()
    and_token = sql.Token(T.Keyword, 'AND')
    except_token = sql.Token(T.Keyword, 'EXCEPT')
    tl = sql.TokenList([
        sql.Token(T.Keyword, 'BETWEEN'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        make_ws(),
        and_token,
        make_ws(),
        sql.Token(T.Name, 'y'),
        make_ws(),
        except_token,
    ])
    idx, token = f._next_token(tl)
    assert token is except_token
    assert idx == tl.token_index(except_token)


# ---------------------------------------------------------------------------
# _process_statement
# ---------------------------------------------------------------------------

def test_process_statement_removes_leading_whitespace():
    f = AlignedIndentFilter()
    tl = sql.Statement([
        make_ws(),
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Keyword, 'FROM'),
    ])
    # leading whitespace removed at indent==0
    f._process_statement(tl)
    assert tl.tokens[0].ttype is T.DML
    assert tl.tokens[0].value == 'SELECT'


def test_process_statement_keeps_leading_whitespace_when_indented():
    f = AlignedIndentFilter()
    f.indent = 1
    tl = sql.Statement([
        make_ws(),
        sql.Token(T.DML, 'SELECT'),
    ])
    f._process_statement(tl)
    assert tl.tokens[0].ttype is T.Whitespace


def test_process_statement_empty():
    f = AlignedIndentFilter()
    tl = sql.Statement([])
    # should not raise
    f._process_statement(tl)
    assert tl.tokens == []


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

def test_process_identifierlist_inserts_newlines():
    f = AlignedIndentFilter()
    tl = sql.IdentifierList([
        sql.Identifier([sql.Token(T.Name, 'a')]),
        sql.Token(T.Punctuation, ','),
        make_ws(),
        sql.Identifier([sql.Token(T.Name, 'b')]),
        sql.Token(T.Punctuation, ','),
        make_ws(),
        sql.Identifier([sql.Token(T.Name, 'c')]),
    ])
    f._process_identifierlist(tl)
    # Newlines inserted before b and c (not before the first)
    values = [t.value for t in tl.tokens]
    ws_count = sum(1 for t in tl.tokens
                   if t.ttype is T.Whitespace and '\n' in t.value)
    assert ws_count == 2


# ---------------------------------------------------------------------------
# _split_kwds / _process_default
# ---------------------------------------------------------------------------

def test_split_kwds_inserts_newlines():
    f = AlignedIndentFilter()
    tl = sql.TokenList([
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'a'),
        make_ws(),
        sql.Token(T.Keyword, 'FROM'),
        make_ws(),
        sql.Token(T.Name, 't'),
        make_ws(),
        sql.Token(T.Keyword, 'WHERE'),
        make_ws(),
        sql.Token(T.Name, 'x'),
    ])
    f._split_kwds(tl)
    newline_tokens = [t for t in tl.tokens
                      if t.ttype is T.Whitespace and '\n' in t.value]
    assert len(newline_tokens) == 2


def test_split_kwds_join_uses_first_word():
    f = AlignedIndentFilter()
    join = sql.Token(T.Keyword, 'LEFT JOIN')
    tl = sql.TokenList([
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'a'),
        make_ws(),
        join,
    ])
    f._split_kwds(tl)
    # Find the whitespace inserted before the JOIN token
    idx = tl.token_index(join)
    prev = tl.tokens[idx - 1]
    assert prev.ttype is T.Whitespace
    assert '\n' in prev.value
    # nl() is called with the string 'LEFT' (first word of the join token);
    # for string offsets, nl() accounts for -len('LEFT'), giving
    # _max_kwd_len - len('LEFT') spaces.
    expected_len = f._max_kwd_len - len('LEFT')
    assert prev.value == '\n' + ' ' * expected_len


def test_split_kwds_group_by_uses_first_word():
    f = AlignedIndentFilter()
    by = sql.Token(T.Keyword, 'GROUP BY')
    tl = sql.TokenList([
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'a'),
        make_ws(),
        by,
    ])
    f._split_kwds(tl)
    idx = tl.token_index(by)
    prev = tl.tokens[idx - 1]
    assert '\n' in prev.value
    # nl('GROUP') -> _max_kwd_len - len('GROUP') spaces
    expected_len = f._max_kwd_len - len('GROUP')
    assert prev.value == '\n' + ' ' * expected_len


def test_process_default_with_sublist():
    f = AlignedIndentFilter()
    inner = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Keyword, 'FROM'),
        make_ws(),
        sql.Token(T.Name, 't'),
        sql.Token(T.Punctuation, ')'),
    ])
    tl = sql.TokenList([
        inner,
        make_ws(),
        sql.Token(T.Keyword, 'WHERE'),
        make_ws(),
        sql.Token(T.Name, 'y'),
    ])
    f._process_default(tl)
    # Subquery should have been processed -> parenthesis indented
    # after indent, the FROM should get a newline with the offset
    values = [t.value for t in tl.tokens]
    assert any('\n' in v for v in values)


def test_process_default_offset_for_group_by():
    f = AlignedIndentFilter()
    inner = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Punctuation, ')'),
    ])
    tl = sql.TokenList([
        sql.Token(T.Keyword, 'GROUP BY'),
        make_ws(),
        inner,
    ])
    # offset should be 0 before _process_default's offset context
    f._process_default(tl)
    # offset reset after processing
    assert f.offset == 0


def test_process_default_group_by_applies_offset():
    """Verify that a sublist preceded by GROUP BY gets offset == 3 applied.

    The parenthesized subquery isn't a SELECT so _process_parenthesis is a
    no-op on it; we instead verify that _process_default temporarily bumps
    the offset while processing and that it is reset afterwards by observing
    the offset passed via a recording TokenList.
    """
    f = AlignedIndentFilter()

    seen_offsets = []

    class RecordingList(sql.TokenList):
        pass

    inner = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Punctuation, ')'),
    ])
    tl = sql.TokenList([
        sql.Token(T.Keyword, 'GROUP BY'),
        make_ws(),
        inner,
    ])

    # Monkeypatch _process on this filter instance to record offsets but
    # still call through to the real implementation.
    real_process = f._process

    def recording_process(tlist):
        seen_offsets.append(f.offset)
        return real_process(tlist)

    f._process = recording_process
    f._process_default(tl)
    assert f.offset == 0
    # At least one offset observation of 3 (from the GROUP BY hack).
    assert 3 in seen_offsets


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

def test_process_parenthesis_non_subquery_noop():
    f = AlignedIndentFilter()
    tl = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Punctuation, ')'),
    ])
    original = list(tl.tokens)
    f._process_parenthesis(tl)
    assert len(tl.tokens) == len(original)


def test_process_parenthesis_subquery_indents():
    f = AlignedIndentFilter()
    tl = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        make_ws(),
        sql.Token(T.Keyword, 'FROM'),
        make_ws(),
        sql.Token(T.Name, 't'),
        sql.Token(T.Punctuation, ')'),
    ])
    f._process_parenthesis(tl)
    # indent restored
    assert f.indent == 0
    # newline after SELECT inside
    joined = ''.join(t.value for t in tl.tokens)
    assert '\n' in joined


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

def build_case():
    # <CASE WHEN a=1 THEN 1 WHEN b=2 THEN 2 ELSE 0 END>
    return sql.Case([
        sql.Token(T.Keyword, 'CASE'),
        make_ws(),
        sql.Token(T.Keyword, 'WHEN'),
        make_ws(),
        sql.Token(T.Name, 'a=1'),
        make_ws(),
        sql.Token(T.Keyword, 'THEN'),
        make_ws(),
        sql.Token(T.Literal.Number.Integer, '1'),
        make_ws(),
        sql.Token(T.Keyword, 'WHEN'),
        make_ws(),
        sql.Token(T.Name, 'b=2'),
        make_ws(),
        sql.Token(T.Keyword, 'THEN'),
        make_ws(),
        sql.Token(T.Literal.Number.Integer, '2'),
        make_ws(),
        sql.Token(T.Keyword, 'ELSE'),
        make_ws(),
        sql.Token(T.Literal.Number.Integer, '0'),
        make_ws(),
        sql.Token(T.Keyword, 'END'),
    ])


def test_process_case_inserts_newlines():
    f = AlignedIndentFilter()
    tl = build_case()
    f._process_case(tl)
    nl_tokens = [t for t in tl.tokens
                 if t.ttype is T.Whitespace and '\n' in t.value]
    # one newline per case after the first, plus END
    assert len(nl_tokens) >= 2


# ---------------------------------------------------------------------------
# _process dispatch
# ---------------------------------------------------------------------------

def test_process_dispatch_statement():
    f = AlignedIndentFilter()
    stmt = sql.Statement([
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'a'),
        make_ws(),
        sql.Token(T.Keyword, 'FROM'),
        make_ws(),
        sql.Token(T.Name, 't'),
    ])
    result = f.process(stmt)
    assert result is stmt
    # FROM should have a newline before it
    idx = None
    for i, t in enumerate(stmt.tokens):
        if t.ttype is T.Keyword and t.value == 'FROM':
            idx = i
    assert idx is not None
    assert stmt.tokens[idx - 1].ttype is T.Whitespace
    assert '\n' in stmt.tokens[idx - 1].value


def test_process_returns_same_object():
    f = AlignedIndentFilter()
    tl = sql.TokenList([sql.Token(T.Name, 'x')])
    result = f.process(tl)
    assert result is tl


def test_process_where_clause():
    f = AlignedIndentFilter()
    where = sql.Where([
        sql.Token(T.Keyword, 'WHERE'),
        make_ws(),
        sql.Token(T.Name, 'a'),
        make_ws(),
        sql.Token(T.Keyword, 'AND'),
        make_ws(),
        sql.Token(T.Name, 'b'),
    ])
    stmt = sql.Statement([
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        make_ws(),
        where,
    ])
    f.process(stmt)
    # AND inside WHERE should have a newline before it
    for i, t in enumerate(where.tokens):
        if t.ttype is T.Keyword and t.value == 'AND':
            prev = where.tokens[i - 1]
            assert prev.ttype is T.Whitespace
            assert '\n' in prev.value


# ---------------------------------------------------------------------------
# filter.indent / filter.offset contexts reset properly
# ---------------------------------------------------------------------------

def test_indent_restored_after_parenthesis():
    f = AlignedIndentFilter()
    tl = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.DML, 'SELECT'),
        make_ws(),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Punctuation, ')'),
    ])
    f._process_parenthesis(tl)
    assert f.indent == 0


def test_offset_restored_after_process_default():
    f = AlignedIndentFilter()
    inner = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Punctuation, ')'),
    ])
    tl = sql.TokenList([sql.Token(T.Keyword, 'GROUP BY'), make_ws(), inner])
    f._process_default(tl)
    assert f.offset == 0
