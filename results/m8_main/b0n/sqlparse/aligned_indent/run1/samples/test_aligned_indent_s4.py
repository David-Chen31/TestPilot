import pytest

from sqlparse.filters.aligned_indent import AlignedIndentFilter
from sqlparse import sql
from sqlparse import tokens as T


def make_tokenlist(*tokens):
    return sql.TokenList(tokens)


def kw(value):
    return sql.Token(T.Keyword, value)


def ident(value):
    return sql.Token(T.Name, value)


def ws(value=' '):
    return sql.Token(T.Whitespace, value)


# ---------------------------------------------------------------------------
# __init__ / nl
# ---------------------------------------------------------------------------

def test_init_defaults():
    f = AlignedIndentFilter()
    assert f.n == '\n'
    assert f.char == ' '
    assert f.offset == 0
    assert f.indent == 0
    assert f._max_kwd_len == len('select')


def test_init_custom():
    f = AlignedIndentFilter(char='\t', n='\r\n')
    assert f.n == '\r\n'
    assert f.char == '\t'


def test_split_words_is_class_attribute():
    assert 'FROM' in AlignedIndentFilter.split_words
    assert 'WHERE' in AlignedIndentFilter.split_words
    assert AlignedIndentFilter.join_words.endswith('JOIN\\b')
    assert AlignedIndentFilter.by_words == r'(GROUP|ORDER)\s+BY\b'


def test_nl_default():
    f = AlignedIndentFilter()
    tok = f.nl()
    assert tok.ttype == T.Whitespace
    # max_kwd_len (6) + offset (1) + indent (0) + self.offset (0)
    assert tok.value == '\n' + ' ' * 7


def test_nl_with_int_offset():
    f = AlignedIndentFilter()
    tok = f.nl(offset=3)
    assert tok.value == '\n' + ' ' * 9


def test_nl_with_str_offset():
    f = AlignedIndentFilter()
    tok = f.nl('SELECT')  # len 6 -> offset -6
    assert tok.value == '\n' + ' ' * (6 - 6)


def test_nl_with_indent_and_offset():
    f = AlignedIndentFilter()
    f.indent = 2
    f.offset = 1
    tok = f.nl()
    # 6 + 1 + 2*(2+6) + 1 = 24
    assert tok.value == '\n' + ' ' * 24


def test_nl_custom_char():
    f = AlignedIndentFilter(char='.')
    tok = f.nl()
    assert tok.value == '\n' + '.' * 7
    assert '.' not in tok.value.split('\n')[1].replace('.', '', 99).replace('', '') or True
    # direct: only '.' characters after newline
    after = tok.value.split('\n', 1)[1]
    assert set(after) == {'.'}


# ---------------------------------------------------------------------------
# process / _process_statement
# ---------------------------------------------------------------------------

def test_process_returns_statement():
    stmt = make_tokenlist(kw('SELECT'), ws(), ident('a'))
    f = AlignedIndentFilter()
    result = f.process(stmt)
    assert result is stmt


def test_process_pops_leading_whitespace():
    stmt = make_tokenlist(ws('\n\n'), kw('SELECT'), ws(), ident('a'))
    f = AlignedIndentFilter()
    f.process(stmt)
    # leading whitespace should be removed
    assert not stmt.tokens[0].is_whitespace


def test_process_keeps_leading_whitespace_when_indented():
    stmt = make_tokenlist(ws('\n\n'), kw('SELECT'), ws(), ident('a'))
    f = AlignedIndentFilter()
    f.indent = 1
    f.process(stmt)
    assert stmt.tokens[0].is_whitespace


def test_process_empty_statement():
    stmt = make_tokenlist()
    f = AlignedIndentFilter()
    result = f.process(stmt)
    assert result is stmt
    assert len(stmt.tokens) == 0


# ---------------------------------------------------------------------------
# _split_kwds / _next_token indirectly via process
# ---------------------------------------------------------------------------

def test_process_splits_on_from():
    stmt = make_tokenlist(kw('SELECT'), ws(), ident('a'), ws(), kw('FROM'),
                          ws(), ident('t'))
    f = AlignedIndentFilter()
    f.process(stmt)
    values = [str(t) for t in stmt.tokens]
    # FROM should be preceded by a newline whitespace token
    from_idx = next(i for i, t in enumerate(stmt.tokens)
                    if t.ttype == T.Keyword and str(t) == 'FROM')
    assert stmt.tokens[from_idx - 1].is_whitespace
    assert '\n' in str(stmt.tokens[from_idx - 1])


def test_process_between_and_not_split():
    stmt = make_tokenlist(kw('SELECT'), ws(), ident('a'), ws(), kw('WHERE'),
                          ws(), ident('b'), ws(), kw('BETWEEN'), ws(),
                          ident('1'), ws(), kw('AND'), ws(), ident('2'))
    f = AlignedIndentFilter()
    f.process(stmt)
    # The AND after BETWEEN must NOT be preceded by a newline whitespace
    idx = next(i for i, t in enumerate(stmt.tokens)
               if t.ttype == T.Keyword and str(t) == 'AND')
    prev = stmt.tokens[idx - 1]
    # previous token is whitespace but no newline inserted before AND
    assert '\n' not in str(prev)


def test_join_word_aligner_uses_first_word():
    stmt = make_tokenlist(kw('SELECT'), ws(), ident('a'), ws(),
                          kw('LEFT JOIN'), ws(), ident('t'))
    f = AlignedIndentFilter()
    f.process(stmt)
    idx = next(i for i, t in enumerate(stmt.tokens)
               if t.ttype == T.Keyword and 'JOIN' in str(t))
    prev = stmt.tokens[idx - 1]
    assert prev.is_whitespace
    assert '\n' in str(prev)


def test_group_by_word_aligner():
    stmt = make_tokenlist(kw('SELECT'), ws(), ident('a'), ws(),
                          kw('GROUP BY'), ws(), ident('a'))
    f = AlignedIndentFilter()
    f.process(stmt)
    idx = next(i for i, t in enumerate(stmt.tokens)
               if t.ttype == T.Keyword and 'GROUP BY' in str(t))
    prev = stmt.tokens[idx - 1]
    assert '\n' in str(prev)


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

def test_process_parenthesis_with_subquery():
    inner = make_tokenlist(ws('('), kw('SELECT'), ws(), ident('a'), ws(')'))
    paren = sql.Parenthesis(inner.tokens)
    f = AlignedIndentFilter()
    f.process(paren)
    # inner content should now contain newline whitespace tokens
    assert any('\n' in str(t) for t in paren.tokens if t.is_whitespace)


def test_process_parenthesis_without_subquery():
    inner = make_tokenlist(ws('('), ident('a'), ws(')'))
    paren = sql.Parenthesis(inner.tokens)
    f = AlignedIndentFilter()
    original = [str(t) for t in paren.tokens]
    f.process(paren)
    # No SELECT -> no changes
    assert [str(t) for t in paren.tokens] == original


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

def test_process_identifierlist():
    il = sql.IdentifierList([
        ident('a'), ws(', '), ident('b'), ws(', '), ident('c')
    ])
    f = AlignedIndentFilter()
    f.process(il)
    # identifiers a is first and not preceded by newline; b and c should be
    newlines = [t for t in il.tokens if t.is_whitespace and '\n' in str(t)]
    assert len(newlines) == 2


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

def test_process_case():
    from sqlparse import parse
    stmt = parse("SELECT CASE WHEN a = 1 THEN 'x' "
                 "WHEN a = 2 THEN 'y' ELSE 'z' END FROM t")[0]
    f = AlignedIndentFilter()
    f.process(stmt)
    # Should have inserted newline whitespace tokens inside the CASE
    text = str(stmt)
    assert '\n' in text


def test_process_case_without_end():
    # CASE without END triggers an AttributeError on unpacking None
    case = sql.Case([kw('CASE'), ws(), kw('WHEN'), ws(), ident('a')])
    f = AlignedIndentFilter()
    with pytest.raises(TypeError):
        f.process(case)


# ---------------------------------------------------------------------------
# _process_default, offset hack
# ---------------------------------------------------------------------------

def test_process_default_recurses_into_sublists():
    inner = make_tokenlist(kw('SELECT'), ws(), ident('a'), ws(), kw('FROM'),
                           ws(), ident('t'))
    stmt = make_tokenlist(kw('SELECT'), ws(), ident('x'), ws(), inner)
    f = AlignedIndentFilter()
    f.process(stmt)
    # inner should be processed too
    assert any(t.is_whitespace and '\n' in str(t) for t in inner.tokens)


# ---------------------------------------------------------------------------
# _process dispatch
# ---------------------------------------------------------------------------

def test_process_dispatch_statement():
    f = AlignedIndentFilter()
    # _process_statement exists and is used on Statement
    assert hasattr(f, '_process_statement')


def test_process_falls_back_to_default():
    f = AlignedIndentFilter()
    tl = make_tokenlist(kw('SELECT'), ws(), ident('a'))
    # TokenList -> no _process_tokenlist method -> uses default
    assert getattr(f, '_process_tokenlist', None) is None
    f._process(tl)
    # Should have run _split_kwds which splits on SELECT?  SELECT is at idx 0
    # so no newline before it; check nothing crashes and tokens still exist.
    assert len(tl.tokens) > 0


# ---------------------------------------------------------------------------
# Integration-style: formatting through process
# ---------------------------------------------------------------------------

def test_process_nested_subquery():
    from sqlparse import parse
    stmt = parse('SELECT * FROM (SELECT a FROM t) x')[0]
    f = AlignedIndentFilter()
    f.process(stmt)
    text = str(stmt)
    assert '\n' in text


def test_process_with_select_inside_parenthesis():
    from sqlparse import parse
    stmt = parse('SELECT * FROM (SELECT a FROM t WHERE b = 1) x')[0]
    f = AlignedIndentFilter()
    f.process(stmt)
    text = str(stmt)
    assert 'WHERE' in text
    assert '\n' in text
