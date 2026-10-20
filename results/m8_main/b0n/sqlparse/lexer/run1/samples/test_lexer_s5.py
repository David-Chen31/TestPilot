import io
import re
import pytest

from sqlparse import lexer, keywords, tokens
from sqlparse.lexer import Lexer, tokenize


@pytest.fixture
def fresh_lexer():
    """A fresh Lexer initialized with default settings, isolated from the singleton."""
    lex = Lexer()
    lex.default_initialization()
    return lex


def get_tokens_list(lex, text, encoding=None):
    return list(lex.get_tokens(text, encoding))


# ---------------------------------------------------------------------------
# Lexer.get_default_instance
# ---------------------------------------------------------------------------

def test_get_default_instance_is_singleton():
    a = Lexer.get_default_instance()
    b = Lexer.get_default_instance()
    assert a is b
    assert isinstance(a, Lexer)


def test_get_default_instance_is_initialized():
    inst = Lexer.get_default_instance()
    assert inst._keywords
    assert inst._SQL_REGEX


# ---------------------------------------------------------------------------
# clear / set_SQL_REGEX / add_keywords
# ---------------------------------------------------------------------------

def test_clear_resets_state(fresh_lexer):
    fresh_lexer.clear()
    assert fresh_lexer._SQL_REGEX == []
    assert fresh_lexer._keywords == []


def test_set_SQL_REGEX_compiles_patterns(fresh_lexer):
    fresh_lexer.clear()
    fresh_lexer.set_SQL_REGEX([(r'\d+', tokens.Number.Integer)])
    assert len(fresh_lexer._SQL_REGEX) == 1
    match_func, ttype = fresh_lexer._SQL_REGEX[0]
    assert ttype is tokens.Number.Integer
    m = match_func('abc 123', 4)
    assert m is not None
    assert m.group() == '123'


def test_set_SQL_REGEX_ignores_case(fresh_lexer):
    fresh_lexer.clear()
    fresh_lexer.set_SQL_REGEX([(r'select', tokens.Keyword)])
    match_func, ttype = fresh_lexer._SQL_REGEX[0]
    assert match_func('SELECT', 0) is not None
    assert match_func('SeLeCt', 0) is not None


def test_add_keywords_appends(fresh_lexer):
    fresh_lexer.clear()
    assert fresh_lexer._keywords == []
    fresh_lexer.add_keywords({'FOO': tokens.Keyword})
    assert fresh_lexer._keywords == [{'FOO': tokens.Keyword}]
    fresh_lexer.add_keywords({'BAR': tokens.Keyword})
    assert len(fresh_lexer._keywords) == 2


# ---------------------------------------------------------------------------
# is_keyword
# ---------------------------------------------------------------------------

def test_is_keyword_found(fresh_lexer):
    ttype, value = fresh_lexer.is_keyword('select')
    assert ttype is tokens.Keyword
    assert value == 'select'


def test_is_keyword_case_insensitive(fresh_lexer):
    ttype, value = fresh_lexer.is_keyword('SELECT')
    assert ttype is tokens.Keyword
    assert value == 'SELECT'


def test_is_keyword_not_found_returns_name(fresh_lexer):
    ttype, value = fresh_lexer.is_keyword('somenonexistentidentifier')
    assert ttype is tokens.Name
    assert value == 'somenonexistentidentifier'


def test_is_keyword_respects_order(fresh_lexer):
    fresh_lexer.clear()
    fresh_lexer.add_keywords({'FOO': tokens.Keyword})
    fresh_lexer.add_keywords({'FOO': tokens.Name})
    ttype, value = fresh_lexer.is_keyword('foo')
    assert ttype is tokens.Keyword
    assert value == 'foo'


# ---------------------------------------------------------------------------
# get_tokens - basic sharing and token correctness
# ---------------------------------------------------------------------------

def test_get_tokens_simple_select(fresh_lexer):
    result = get_tokens_list(fresh_lexer, 'SELECT')
    assert result == [(tokens.Keyword.DML, 'SELECT')]


def test_get_tokens_returns_generator(fresh_lexer):
    gen = fresh_lexer.get_tokens('SELECT')
    assert hasattr(gen, '__next__')


def test_get_tokens_multiple(fresh_lexer):
    result = get_tokens_list(fresh_lexer, 'SELECT a')
    types = [t for t, _ in result]
    values = [v for _, v in result]
    assert tokens.Keyword.DML in types
    assert 'SELECT' in values
    assert 'a' in values


def test_get_tokens_empty_string(fresh_lexer):
    assert get_tokens_list(fresh_lexer, '') == []


def test_get_tokens_whitespace(fresh_lexer):
    result = get_tokens_list(fresh_lexer, '  \n\t ')
    # whitespace may be grouped as a single token
    joined = ''.join(v for _, v in result)
    assert joined == '  \n\t '


def test_get_tokens_integer(fresh_lexer):
    result = get_tokens_list(fresh_lexer, '42')
    assert result == [(tokens.Number.Integer, '42')]


def test_get_tokens_string_literal(fresh_lexer):
    result = get_tokens_list(fresh_lexer, "'hello'")
    assert result == [(tokens.String.Single, "'hello'")]


def test_get_tokens_unknown_char_is_error(fresh_lexer):
    fresh_lexer.clear()
    # With no regexes, everything becomes Error tokens
    result = get_tokens_list(fresh_lexer, 'abc')
    assert result == [
        (tokens.Error, 'a'),
        (tokens.Error, 'b'),
        (tokens.Error, 'c'),
    ]


# ---------------------------------------------------------------------------
# get_tokens - input types
# ---------------------------------------------------------------------------

def test_get_tokens_from_bytes_utf8(fresh_lexer):
    result = get_tokens_list(fresh_lexer, b'SELECT')
    assert result == [(tokens.Keyword.DML, 'SELECT')]


def test_get_tokens_from_bytes_with_encoding(fresh_lexer):
    result = get_tokens_list(fresh_lexer, 'SELECT'.encode('utf-16'), 'utf-16')
    assert result == [(tokens.Keyword.DML, 'SELECT')]


def test_get_tokens_from_text_io(fresh_lexer):
    stream = io.StringIO('SELECT')
    result = get_tokens_list(fresh_lexer, stream)
    assert result == [(tokens.Keyword.DML, 'SELECT')]


def test_get_tokens_invalid_type_raises(fresh_lexer):
    with pytest.raises(TypeError):
        list(fresh_lexer.get_tokens(123))


def test_get_tokens_invalid_type_raises_list(fresh_lexer):
    with pytest.raises(TypeError):
        list(fresh_lexer.get_tokens(['SELECT']))


# ---------------------------------------------------------------------------
# tokenize function (module-level)
# ---------------------------------------------------------------------------

def test_tokenize_returns_iterator():
    result = tokenize('SELECT')
    assert hasattr(result, '__next__')
    first = next(result)
    assert first == (tokens.Keyword.DML, 'SELECT')


def test_tokenize_list_of_pairs():
    result = list(tokenize('SELECT 1'))
    assert isinstance(result, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)


def test_tokenize_uses_default_instance():
    # tokenize should use the singleton, not a fresh lexer
    gen = tokenize('SELECT')
    assert hasattr(gen, '__next__')
    assert list(gen) == [(tokens.Keyword.DML, 'SELECT')]


def test_tokenize_with_encoding():
    result = list(tokenize('SELECT'.encode('utf-16'), encoding='utf-16'))
    assert result == [(tokens.Keyword.DML, 'SELECT')]


def test_tokenize_bytes_invalid_raises():
    # Invalid utf-8 that also fails unicode-escape should raise
    with pytest.raises(UnicodeDecodeError):
        list(tokenize(b'\x80\x81\x82'))


# ---------------------------------------------------------------------------
# default_initialization restores defaults
# ---------------------------------------------------------------------------

def test_default_initialization_full_keywords(fresh_lexer):
    fresh_lexer.default_initialization()
    # after full init, a common keyword is recognized
    ttype, value = fresh_lexer.is_keyword('select')
    assert ttype is tokens.Keyword.DML
    assert value == 'select'


def test_default_initialization_clears_custom(fresh_lexer):
    fresh_lexer.add_keywords({'CUSTOMKW': tokens.Keyword})
    ttype, _ = fresh_lexer.is_keyword('customkw')
    assert ttype is tokens.Keyword
    fresh_lexer.default_initialization()
    ttype, value = fresh_lexer.is_keyword('customkw')
    assert ttype is tokens.Name
    assert value == 'customkw'


# ---------------------------------------------------------------------------
# Delimited spans / dollar-quoted strings and multiline comments
# ---------------------------------------------------------------------------

def test_dollar_quoted_string(fresh_lexer):
    text = 'SELECT $$abc$$'
    result = get_tokens_list(fresh_lexer, text)
    # find the dollar-quoted token
    dollar_tokens = [v for _, v in result if v.startswith('$$')]
    assert dollar_tokens == ['$$abc$$']


def test_multiline_comment_resolved(fresh_lexer):
    text = '/* comment */'
    result = get_tokens_list(fresh_lexer, text)
    combined = ''.join(v for _, v in result)
    assert combined == text


def test_comment_within_string_not_resolved(fresh_lexer):
    # The "/*" inside the string should not start a multiline comment
    text = "'/* not a comment */'"
    result = get_tokens_list(fresh_lexer, text)
    assert result == [(tokens.String.Single, text)]


def test_dollar_inside_string_not_resolved(fresh_lexer):
    text = "'$$'"
    result = get_tokens_list(fresh_lexer, text)
    assert result == [(tokens.String.Single, text)]
