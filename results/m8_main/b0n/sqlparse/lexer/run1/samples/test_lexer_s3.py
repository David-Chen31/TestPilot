import io
import re
from threading import Lock

import pytest

from sqlparse import keywords, tokens
from sqlparse.lexer import Lexer, tokenize


@pytest.fixture(autouse=True)
def reset_default_instance():
    """Reset the singleton so tests don't leak state into each other."""
    saved = Lexer._default_instance
    Lexer._default_instance = None
    yield
    Lexer._default_instance = saved


# ---------------------------------------------------------------------------
# Lexer.get_default_instance
# ---------------------------------------------------------------------------

def test_get_default_instance_creates_singleton():
    inst1 = Lexer.get_default_instance()
    inst2 = Lexer.get_default_instance()
    assert inst1 is inst2
    assert isinstance(inst1, Lexer)
    assert Lexer._default_instance is inst1


def test_get_default_instance_is_initialized():
    inst = Lexer.get_default_instance()
    # default_initialization populates the regex and keyword dicts
    assert len(inst._SQL_REGEX) > 0
    assert len(inst._keywords) > 0
    assert inst._keywords[0] is keywords.KEYWORDS_COMMON


def test_get_default_instance_lazily_initialized_on_class():
    # Calling through the class creates and stores instance on the class.
    inst = Lexer.get_default_instance()
    assert Lexer._default_instance is inst


# ---------------------------------------------------------------------------
# Lexer.clear / default_initialization
# ---------------------------------------------------------------------------

def test_clear_empties_configuration():
    lex = Lexer()
    lex.default_initialization()
    assert lex._SQL_REGEX
    assert lex._keywords
    lex.clear()
    assert lex._SQL_REGEX == []
    assert lex._keywords == []


def test_default_initialization_restores_all_keyword_dicts():
    lex = Lexer()
    lex.clear()
    lex.default_initialization()
    expected = [
        keywords.KEYWORDS_COMMON,
        keywords.KEYWORDS_ORACLE,
        keywords.KEYWORDS_MYSQL,
        keywords.KEYWORDS_PLPGSQL,
        keywords.KEYWORDS_HQL,
        keywords.KEYWORDS_MSACCESS,
        keywords.KEYWORDS_SNOWFLAKE,
        keywords.KEYWORDS_BIGQUERY,
        keywords.KEYWORDS,
    ]
    assert lex._keywords == expected
    assert len(lex._SQL_REGEX) == len(keywords.SQL_REGEX)


# ---------------------------------------------------------------------------
# Lexer.set_SQL_REGEX / add_keywords / is_keyword
# ---------------------------------------------------------------------------

def test_set_SQL_REGEX_compiles_with_flags():
    lex = Lexer()
    lex.set_SQL_REGEX([(r"foo", tokens.Keyword)])
    assert len(lex._SQL_REGEX) == 1
    matcher, action = lex._SQL_REGEX[0]
    assert action is tokens.Keyword
    # ignore-case matching works
    m = matcher("FOOBAR", 0)
    assert m is not None
    assert m.group() == "FOO"


def test_add_keywords_appends_in_order():
    lex = Lexer()
    d1 = {"FOO": tokens.Keyword}
    d2 = {"FOO": tokens.Name, "BAR": tokens.Keyword}
    lex.add_keywords(d1)
    lex.add_keywords(d2)
    assert lex._keywords == [d1, d2]


def test_is_keyword_found_first_dict_wins():
    lex = Lexer()
    d1 = {"SELECT": tokens.Keyword}
    d2 = {"SELECT": tokens.Name}
    lex.add_keywords(d1)
    lex.add_keywords(d2)
    assert lex.is_keyword("SELECT") == (tokens.Keyword, "SELECT")
    assert lex.is_keyword("select") == (tokens.Keyword, "select")
    assert lex.is_keyword("SeLeCt") == (tokens.Keyword, "SeLeCt")


def test_is_keyword_not_found_returns_name():
    lex = Lexer()
    lex.add_keywords({"SELECT": tokens.Keyword})
    assert lex.is_keyword("notakeyword") == (tokens.Name, "notakeyword")


def test_is_keyword_with_no_dictionaries():
    lex = Lexer()  # empty _keywords
    # Note: clear() sets _keywords = []
    assert lex.is_keyword("SELECT") == (tokens.Name, "SELECT")


# ---------------------------------------------------------------------------
# Lexer.get_tokens - simple text
# ---------------------------------------------------------------------------

def test_get_tokens_basic_whitespace_split():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens("SELECT"))
    assert result == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_returns_list_of_tuples():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens("1 + 2"))
    assert all(isinstance(t, tuple) and len(t) == 2 for t in result)
    assert all(isinstance(v, str) for _, v in result)


# ---------------------------------------------------------------------------
# get_tokens - input types
# ---------------------------------------------------------------------------

def test_get_tokens_accepts_str():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens("SELECT 1"))
    assert result[0] == (tokens.Keyword.DML, "SELECT")


def test_get_tokens_accepts_bytes_utf8():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens(b"SELECT"))
    assert result == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_accepts_bytes_with_encoding():
    lex = Lexer()
    lex.default_initialization()
    text = "SELECT".encode("utf-16")
    result = list(lex.get_tokens(text, encoding="utf-16"))
    assert result == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_accepts_text_io_base():
    lex = Lexer()
    lex.default_initialization()
    buf = io.StringIO("SELECT")
    result = list(lex.get_tokens(buf))
    assert result == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_rejects_other_types():
    lex = Lexer()
    lex.default_initialization()
    with pytest.raises(TypeError) as excinfo:
        list(lex.get_tokens(123))
    assert "Expected text or file-like object" in str(excinfo.value)


def test_get_tokens_bytes_fallback_unicode_escape():
    # A byte sequence that is not valid utf-8 but decodes via unicode-escape
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens(b"\xff"))
    # decodes to a character; should produce something (no crash)
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# get_tokens - tokenization correctness
# ---------------------------------------------------------------------------

def test_get_tokens_whitespace():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens(" "))
    assert result == [(tokens.Whitespace, " ")]


def test_get_tokens_comment():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens("-- comment"))
    assert result == [(tokens.Comment.Single, "-- comment")]


def test_get_tokens_string_literal():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens("'hello'"))
    assert result == [(tokens.String.Single, "'hello'")]


def test_get_tokens_number():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens("42"))
    assert result == [(tokens.Number.Integer, "42")]


def test_get_tokens_punctuation():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens(";"))
    assert result == [(tokens.Punctuation, ";")]


def test_get_tokens_error_char():
    lex = Lexer()
    lex.default_initialization()
    # A character not matching any regex should fall through to Error
    result = list(lex.get_tokens("`"))
    # Backtick is actually matched by mysql regex as a name quote; just
    # verify lexing of an unmatchable character yields an Error token.
    # Use a control char that no pattern matches.
    lex2 = Lexer()
    lex2.set_SQL_REGEX([])  # no patterns at all
    result2 = list(lex2.get_tokens("a"))
    assert result2 == [(tokens.Error, "a")]


def test_get_tokens_case_insensitive_keyword():
    lex = Lexer()
    lex.default_initialization()
    result = list(lex.get_tokens("select"))
    assert result == [(tokens.Keyword.DML, "select")]


def test_get_tokens_empty_string():
    lex = Lexer()
    lex.default_initialization()
    assert list(lex.get_tokens("")) == []


# ---------------------------------------------------------------------------
# tokenize module function
# ---------------------------------------------------------------------------

def test_tokenize_uses_default_instance():
    result = list(tokenize("SELECT 1"))
    assert result[0] == (tokens.Keyword.DML, "SELECT")
    # default instance is created
    assert Lexer._default_instance is not None


def test_tokenize_multiple_statements():
    result = list(tokenize("SELECT 1; SELECT 2"))
    values = [v for _, v in result]
    assert values.count("SELECT") == 2
    assert ";" in values


def test_tokenize_with_encoding():
    text = "SELECT".encode("utf-16")
    result = list(tokenize(text, encoding="utf-16"))
    assert result == [(tokens.Keyword.DML, "SELECT")]


def test_tokenize_returns_generator():
    gen = tokenize("SELECT")
    # It is a generator (iterator), not a list
    assert hasattr(gen, "__next__")
    assert not isinstance(gen, list)
    assert next(gen) == (tokens.Keyword.DML, "SELECT")


def test_tokenize_file_like():
    buf = io.StringIO("SELECT 1")
    result = list(tokenize(buf))
    assert result[0] == (tokens.Keyword.DML, "SELECT")


def test_tokenize_type_error():
    with pytest.raises(TypeError):
        list(tokenize(3.14))


# ---------------------------------------------------------------------------
# Singleton / reset behaviour
# ---------------------------------------------------------------------------

def test_tokenize_reuses_singleton():
    list(tokenize("SELECT"))
    first = Lexer._default_instance
    list(tokenize("SELECT"))
    assert Lexer._default_instance is first


def test_default_instance_has_lock_attribute():
    assert isinstance(Lexer._lock, Lock)
