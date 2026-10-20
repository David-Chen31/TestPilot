import io
import re

import pytest

from sqlparse import keywords, tokens
from sqlparse.lexer import Lexer, tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lexer():
    """Create a fresh, fully initialized lexer (not the singleton)."""
    lexer = Lexer()
    lexer.default_initialization()
    return lexer


# ---------------------------------------------------------------------------
# Lexer.clear / set_SQL_REGEX / add_keywords / default_initialization
# ---------------------------------------------------------------------------

def test_clear_resets_state():
    lexer = make_lexer()
    lexer.clear()
    assert lexer._SQL_REGEX == []
    assert lexer._keywords == []


def test_set_SQL_REGEX_compiles_and_ignores_case():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX(
        [
            (r'world', tokens.Name),
        ]
    )
    # One entry, a callable match.
    assert len(lexer._SQL_REGEX) == 1
    match_fn, token_type = lexer._SQL_REGEX[0]
    assert callable(match_fn)
    assert token_type is tokens.Name

    m = match_fn('HELLO WORLD', 6)
    assert m is not None
    assert m.group() == 'WORLD'


def test_set_SQL_REGEX_handles_unicode():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r'\w+', tokens.Name)])
    match_fn, _ = lexer._SQL_REGEX[0]
    m = match_fn('héllo', 0)
    assert m is not None
    assert m.group() == 'héllo'


def test_add_keywords_appends_in_order():
    lexer = Lexer()
    lexer.clear()
    d1 = {'FOO': tokens.Keyword}
    d2 = {'BAR': tokens.Name}
    lexer.add_keywords(d1)
    lexer.add_keywords(d2)
    assert lexer._keywords == [d1, d2]


def test_default_initialization_sets_regx_and_keywords():
    lexer = Lexer()
    lexer.default_initialization()
    assert len(lexer._SQL_REGEX) == len(keywords.SQL_REGEX)
    assert len(lexer._keywords) == 9


def test_default_initialization_is_idempotent():
    lexer = Lexer()
    lexer.default_initialization()
    first_regex_len = len(lexer._SQL_REGEX)
    first_kw_len = len(lexer._keywords)
    lexer.default_initialization()
    # clear() is called first, so lengths don't accumulate.
    assert len(lexer._SQL_REGEX) == first_regex_len
    assert len(lexer._keywords) == first_kw_len


# ---------------------------------------------------------------------------
# Lexer.get_default_instance (singleton + thread-safe lock usage)
# ---------------------------------------------------------------------------

def test_get_default_instance_singleton():
    Lexer._default_instance = None
    try:
        inst1 = Lexer.get_default_instance()
        inst2 = Lexer.get_default_instance()
        assert inst1 is inst2
        assert isinstance(inst1, Lexer)
        # A default-initialized lexer has keyword dicts configured.
        assert len(inst1._keywords) > 0
        assert len(inst1._SQL_REGEX) > 0
    finally:
        Lexer._default_instance = None


def test_get_default_instance_reuses_existing():
    Lexer._default_instance = None
    try:
        existing = Lexer()
        Lexer._default_instance = existing
        assert Lexer.get_default_instance() is existing
        # No default_initialization applied because instance already existed,
        # so it has no attributes set up.
        assert not hasattr(existing, '_keywords')
        assert not hasattr(existing, '_SQL_REGEX')
    finally:
        Lexer._default_instance = None


# ---------------------------------------------------------------------------
# Lexer.is_keyword
# ---------------------------------------------------------------------------

def test_is_keyword_found_returns_type_and_original_value():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({'SELECT': tokens.Keyword})
    assert lexer.is_keyword('select') == (tokens.Keyword, 'select')
    assert lexer.is_keyword('SeLeCt') == (tokens.Keyword, 'SeLeCt')


def test_is_keyword_not_found_returns_name():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({'SELECT': tokens.Keyword})
    assert lexer.is_keyword('foo') == (tokens.Name, 'foo')


def test_is_keyword_uses_first_matching_dictionary():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({'FOO': tokens.Keyword})
    lexer.add_keywords({'FOO': tokens.Name})
    assert lexer.is_keyword('foo') == (tokens.Keyword, 'foo')


def test_is_keyword_empty_keywords():
    lexer = Lexer()
    lexer.clear()
    assert lexer.is_keyword('select') == (tokens.Name, 'select')


# ---------------------------------------------------------------------------
# Lexer.get_tokens: basic types
# ---------------------------------------------------------------------------

def test_get_tokens_basic_select():
    lexer = make_lexer()
    result = [(ttype, value) for ttype, value in lexer.get_tokens('SELECT 1')]
    # There should be a SELECT keyword and a number token.
    assert (tokens.Keyword.DML, 'SELECT') in result
    assert any(tt in tokens.Number for tt, _ in result)
    # Whitespace present as a token.
    assert (tokens.Whitespace, ' ') in result


def test_get_tokens_preserves_case_of_value():
    lexer = make_lexer()
    result = list(lexer.get_tokens('select'))
    assert result[0] == (tokens.Keyword.DML, 'select')


def test_get_tokens_accepts_str():
    lexer = make_lexer()
    assert list(lexer.get_tokens('SELECT')) == [
        (tokens.Keyword.DML, 'SELECT')
    ]


def test_get_tokens_accepts_bytes_utf8_default():
    lexer = make_lexer()
    result = list(lexer.get_tokens(b'SELECT'))
    assert result == [(tokens.Keyword.DML, 'SELECT')]


def test_get_tokens_accepts_bytes_with_encoding():
    lexer = make_lexer()
    # 'abc' encoded in latin-1 still decodes fine.
    result = list(lexer.get_tokens(b'abc', encoding='latin-1'))
    assert result == [(tokens.Name, 'abc')]


def test_get_tokens_accepts_text_io():
    lexer = make_lexer()
    buf = io.StringIO('SELECT')
    assert list(lexer.get_tokens(buf)) == [(tokens.Keyword.DML, 'SELECT')]


def test_get_tokens_raises_on_unsupported_type():
    lexer = make_lexer()
    with pytest.raises(TypeError) as excinfo:
        list(lexer.get_tokens(12345))
    assert 'Expected text or file-like object' in str(excinfo.value)


def test_get_tokens_empty_string():
    lexer = make_lexer()
    assert list(lexer.get_tokens('')) == []


def test_get_tokens_whitespace_only():
    lexer = make_lexer()
    result = list(lexer.get_tokens('   '))
    # Each whitespace character is emitted as its own token.
    assert result == [
        (tokens.Whitespace, ' '),
        (tokens.Whitespace, ' '),
        (tokens.Whitespace, ' '),
    ]


# ---------------------------------------------------------------------------
# Lexer.get_tokens: literals, comments, errors
# ---------------------------------------------------------------------------

def test_get_tokens_string_literal():
    lexer = make_lexer()
    result = list(lexer.get_tokens("'hello'"))
    assert result == [(tokens.String.Single, "'hello'")]


def test_get_tokens_string_with_embedded_double_quote():
    lexer = make_lexer()
    result = list(lexer.get_tokens("'a''b'"))
    assert result == [(tokens.String.Single, "'a''b'")]


def test_get_tokens_line_comment():
    lexer = make_lexer()
    result = list(lexer.get_tokens('-- a comment'))
    assert result == [(tokens.Comment.Single, '-- a comment')]


def test_get_tokens_multiline_comment():
    lexer = make_lexer()
    result = list(lexer.get_tokens('/* a\ncomment */'))
    assert result == [(tokens.Comment.Multiline, '/* a\ncomment */')]


def test_get_tokens_delimiter_inside_string_is_not_resolved():
    # A "/*" inside a string must remain part of the string token.
    lexer = make_lexer()
    result = list(lexer.get_tokens("'/*'"))
    assert result == [(tokens.String.Single, "'/*'")]


def test_get_tokens_dollar_quoted_literal():
    lexer = make_lexer()
    text = '$$hello$$'
    result = list(lexer.get_tokens(text))
    assert len(result) == 1
    assert result[0][1] == '$$hello$$'


def test_get_tokens_unknown_character_is_error():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r'[a-z]+', tokens.Name)])
    result = list(lexer.get_tokens('abc#def'))
    # '#' not matched by any regex -> Error token.
    errors = [item for item in result if item[0] is tokens.Error]
    assert (tokens.Error, '#') in errors


def test_get_tokens_is_a_generator():
    lexer = make_lexer()
    gen = lexer.get_tokens('SELECT')
    # Not a list, supports iteration only once.
    assert hasattr(gen, '__iter__')
    assert not isinstance(gen, list)
    assert list(gen) == [(tokens.Keyword.DML, 'SELECT')]


def test_get_tokens_minimal_regex():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r'[a-zA-Z_]+', tokens.Name)])
    lexer.add_keywords({'FOO': tokens.Keyword})
    result = list(lexer.get_tokens('foo bar'))
    # The regex maps matches to Name; PROCESS_AS_KEYWORD isn't used here, so
    # keyword lookup does not apply.
    assert result[0] == (tokens.Name, 'foo')
    assert result[-1] == (tokens.Name, 'bar')


def test_get_tokens_process_as_keyword():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r'[a-zA-Z_]+', keywords.PROCESS_AS_KEYWORD)])
    lexer.add_keywords({'FOO': tokens.Keyword})
    result = list(lexer.get_tokens('foo bar'))
    # The space between "foo" and "bar" matches no regex -> Error token.
    assert result == [
        (tokens.Keyword, 'foo'),
        (tokens.Error, ' '),
        (tokens.Name, 'bar'),
    ]


# ---------------------------------------------------------------------------
# tokenize() module function
# ---------------------------------------------------------------------------

def test_tokenize_uses_default_instance():
    Lexer._default_instance = None
    try:
        stream = tokenize('SELECT')
        assert list(stream) == [(tokens.Keyword.DML, 'SELECT')]
        assert Lexer._default_instance is not None
    finally:
        Lexer._default_instance = None


def test_tokenize_returns_iterable_of_pairs():
    Lexer._default_instance = None
    try:
        stream = tokenize('SELECT 1')
        items = list(stream)
        assert len(items) > 0
        for ttype, value in items:
            assert isinstance(ttype, tokens._TokenType)
            assert isinstance(value, str)
    finally:
        Lexer._default_instance = None


def test_tokenize_with_encoding():
    Lexer._default_instance = None
    try:
        result = list(tokenize(b'abc', encoding='utf-8'))
        assert result == [(tokens.Name, 'abc')]
    finally:
        Lexer._default_instance = None
