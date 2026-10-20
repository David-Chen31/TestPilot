import io
import re

import pytest

from sqlparse import keywords, tokens
from sqlparse.lexer import Lexer, tokenize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_default_instance():
    """Ensure the singleton is reset around each test to avoid leakage."""
    old = Lexer._default_instance
    Lexer._default_instance = None
    yield
    Lexer._default_instance = old


# ---------------------------------------------------------------------------
# Lexer singleton behavior
# ---------------------------------------------------------------------------

def test_get_default_instance_returns_singleton():
    first = Lexer.get_default_instance()
    second = Lexer.get_default_instance()
    assert first is second


def test_default_initialization_sets_regex_and_keywords():
    lexer = Lexer()
    lexer.clear()
    assert lexer._SQL_REGEX == []
    assert lexer._keywords == []

    lexer.default_initialization()
    assert len(lexer._SQL_REGEX) == len(keywords.SQL_REGEX)
    # Keyword dicts added via default_initialization
    assert len(lexer._keywords) > 0


def test_clear_resets_state():
    lexer = Lexer()
    lexer.default_initialization()
    lexer.clear()
    assert lexer._SQL_REGEX == []
    assert lexer._keywords == []


def test_set_SQL_REGEX_compiles_with_flags():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r'abc', tokens.Keyword)])
    assert len(lexer._SQL_REGEX) == 1
    matcher, ttype = lexer._SQL_REGEX[0]
    assert ttype is tokens.Keyword
    m = matcher('xABCy', 1)
    assert m is not None
    assert m.group() == 'ABC'


def test_add_keywords_appends_in_order():
    lexer = Lexer()
    lexer.clear()
    d1 = {'FOO': tokens.Keyword}
    d2 = {'BAR': tokens.Name.Builtin}
    lexer.add_keywords(d1)
    lexer.add_keywords(d2)
    assert lexer._keywords == [d1, d2]


# ---------------------------------------------------------------------------
# is_keyword
# ---------------------------------------------------------------------------

def test_is_keyword_found_case_insensitive():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({'SELECT': tokens.Keyword})
    ttype, value = lexer.is_keyword('select')
    assert ttype is tokens.Keyword
    assert value == 'select'


def test_is_keyword_lookup_order():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({'FOO': tokens.Keyword})
    lexer.add_keywords({'FOO': tokens.Name.Builtin})
    ttype, _ = lexer.is_keyword('foo')
    assert ttype is tokens.Keyword


def test_is_keyword_not_found_returns_name():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({'SELECT': tokens.Keyword})
    ttype, value = lexer.is_keyword('notakeyword')
    assert ttype is tokens.Name
    assert value == 'notakeyword'


def test_is_keyword_preserves_original_case():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({'SELECT': tokens.Keyword})
    _, value = lexer.is_keyword('SeLeCt')
    assert value == 'SeLeCt'


# ---------------------------------------------------------------------------
# get_tokens - input handling
# ---------------------------------------------------------------------------

def test_get_tokens_accepts_file_like():
    lexer = Lexer.get_default_instance()
    result = list(lexer.get_tokens(io.StringIO('select 1')))
    assert result[0] == (tokens.Keyword.DML, 'select')


def test_get_tokens_accepts_bytes_utf8():
    lexer = Lexer.get_default_instance()
    result = list(lexer.get_tokens(b'select 1'))
    assert result[0] == (tokens.Keyword.DML, 'select')


def test_get_tokens_accepts_bytes_with_encoding():
    lexer = Lexer.get_default_instance()
    result = list(lexer.get_tokens('select 1'.encode('latin-1'), encoding='latin-1'))
    assert result[0] == (tokens.Keyword.DML, 'select')


def test_get_tokens_invalid_type_raises_typeerror():
    lexer = Lexer.get_default_instance()
    with pytest.raises(TypeError):
        list(lexer.get_tokens(12345))


# ---------------------------------------------------------------------------
# get_tokens - tokenization correctness
# ---------------------------------------------------------------------------

def test_get_tokens_simple_select():
    result = list(tokenize('SELECT 1'))
    assert (tokens.Keyword.DML, 'SELECT') in result


def test_get_tokens_unknown_char_is_error():
    lexer = Lexer()
    lexer.clear()
    # No regex at all => every char becomes Error
    result = list(lexer.get_tokens('ab'))
    assert result == [(tokens.Error, 'a'), (tokens.Error, 'b')]


def test_get_tokens_empty_string():
    result = list(tokenize(''))
    assert result == []


def test_get_tokens_whitespace():
    result = list(tokenize('   '))
    assert result == [(tokens.Whitespace, '   ')]


def test_get_tokens_string_literal_delimiter_not_resolved():
    # A '$' inside a string literal should not start a dollar-quoted token.
    result = list(tokenize("'$tag$ text $tag$'"))
    values = [v for _, v in result]
    # The whole thing should be tokenized as a single string literal
    assert "'$tag$ text $tag$'" in values


def test_get_tokens_multiline_comment():
    result = list(tokenize('/* comment */'))
    assert result == [(tokens.Comment.Multiline, '/* comment */')]


def test_get_tokens_line_comment():
    result = list(tokenize('-- line comment'))
    assert result[0] == (tokens.Comment.Single, '-- line comment')


def test_get_tokens_custom_regex_only():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r'\d+', tokens.Number.Integer)])
    result = list(lexer.get_tokens('123abc'))
    # 123 -> integer, then a,b,c -> each Error
    assert result == [
        (tokens.Number.Integer, '123'),
        (tokens.Error, 'a'),
        (tokens.Error, 'b'),
        (tokens.Error, 'c'),
    ]


def test_get_tokens_process_as_keyword_action():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r'[A-Za-z_]+', keywords.PROCESS_AS_KEYWORD)])
    lexer.add_keywords({'SELECT': tokens.Keyword})
    result = list(lexer.get_tokens('select foo'))
    assert result == [
        (tokens.Keyword, 'select'),
        (tokens.Whitespace, ' '),
        (tokens.Name, 'foo'),
    ]


def test_get_tokens_dollar_quoted_string():
    result = list(tokenize('$tag$ body $tag$'))
    assert result == [(tokens.String.Symbol, '$tag$ body $tag$')]


def test_get_tokens_multiline_skipped_when_inside_string():
    # The /* */ inside a string literal must not be lexed as a comment.
    result = list(tokenize("'/* not a comment */'"))
    assert result == [(tokens.String.Single, "'/* not a comment */'")]


def test_get_tokens_nested_type():
    result = list(tokenize('SELECT COUNT(*) FROM tbl'))
    assert (tokens.Keyword.DML, 'SELECT') in result
    assert (tokens.Keyword, 'FROM') in result
    assert (tokens.Name, 'COUNT') in result


def test_tokenize_function_uses_default_instance():
    result = list(tokenize('SELECT 1'))
    assert result
    # Compare with explicit default instance
    default = Lexer.get_default_instance()
    assert list(default.get_tokens('SELECT 1')) == result


# ---------------------------------------------------------------------------
# default_initialization isolates state
# ---------------------------------------------------------------------------

def test_default_initialization_reverts_customizations():
    lexer = Lexer()
    lexer.default_initialization()
    lexer.add_keywords({'MYKW': tokens.Keyword})
    assert lexer.is_keyword('MYKW')[0] is tokens.Keyword

    lexer.default_initialization()
    # Custom keyword should be gone
    assert lexer.is_keyword('MYKW')[0] is tokens.Name
    # But built-in keywords remain
    assert lexer.is_keyword('SELECT')[0] is tokens.Keyword.DML
