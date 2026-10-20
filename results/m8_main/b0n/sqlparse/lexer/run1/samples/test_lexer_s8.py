import io
import re

import pytest

from sqlparse import keywords, tokens
from sqlparse.lexer import Lexer, tokenize
from sqlparse.tokens import _TokenType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_default_instance():
    """Reset the singleton so tests do not leak global state."""
    Lexer._default_instance = None


@pytest.fixture(autouse=True)
def _clean_singleton():
    _reset_default_instance()
    yield
    _reset_default_instance()


# ---------------------------------------------------------------------------
# Singleton behavior
# ---------------------------------------------------------------------------

def test_get_default_instance_returns_singleton():
    inst1 = Lexer.get_default_instance()
    inst2 = Lexer.get_default_instance()
    assert inst1 is inst2
    assert isinstance(inst1, Lexer)


def test_get_default_instance_initializes():
    inst = Lexer.get_default_instance()
    assert len(inst._SQL_REGEX) > 0
    assert len(inst._keywords) > 0


def test_get_default_instance_does_not_reinitialize():
    inst = Lexer.get_default_instance()
    original = inst._SQL_REGEX
    inst2 = Lexer.get_default_instance()
    assert inst2._SQL_REGEX is original


# ---------------------------------------------------------------------------
# clear / set_SQL_REGEX / add_keywords
# ---------------------------------------------------------------------------

def test_clear_resets_state():
    lexer = Lexer()
    lexer.clear()
    assert lexer._SQL_REGEX == []
    assert lexer._keywords == []


def test_clear_on_fresh_instance():
    lexer = Lexer()
    # Attributes are only defined after clear()/init
    lexer.clear()
    assert lexer._SQL_REGEX == []
    assert lexer._keywords == []


def test_set_SQL_REGEX_compiles_regexes():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r"\d+", tokens.Number)])
    assert len(lexer._SQL_REGEX) == 1
    matcher, action = lexer._SQL_REGEX[0]
    assert action is tokens.Number
    m = matcher("abc123", 3)
    assert m is not None
    assert m.group() == "123"


def test_set_SQL_REGEX_is_case_insensitive():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r"abc", tokens.Keyword)])
    matcher, _ = lexer._SQL_REGEX[0]
    assert matcher("ABC", 0) is not None


def test_add_keywords_appends_in_order():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({"FOO": tokens.Keyword})
    lexer.add_keywords({"BAR": tokens.Keyword})
    assert lexer._keywords == [{"FOO": tokens.Keyword}, {"BAR": tokens.Keyword}]


# ---------------------------------------------------------------------------
# is_keyword
# ---------------------------------------------------------------------------

def test_is_keyword_found():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({"SELECT": tokens.Keyword})
    ttype, value = lexer.is_keyword("select")
    assert ttype is tokens.Keyword
    assert value == "select"


def test_is_keyword_case_insensitive_lookup():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({"FOO": tokens.Keyword})
    ttype, value = lexer.is_keyword("Foo")
    assert ttype is tokens.Keyword
    assert value == "Foo"


def test_is_keyword_not_found():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({"SELECT": tokens.Keyword})
    ttype, value = lexer.is_keyword("hello")
    assert ttype is tokens.Name
    assert value == "hello"


def test_is_keyword_no_dicts():
    lexer = Lexer()
    lexer.clear()
    ttype, value = lexer.is_keyword("anything")
    assert ttype is tokens.Name
    assert value == "anything"


def test_is_keyword_first_dict_wins():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({"FOO": tokens.Keyword})
    lexer.add_keywords({"FOO": tokens.Name})
    ttype, _ = lexer.is_keyword("foo")
    assert ttype is tokens.Keyword


# ---------------------------------------------------------------------------
# get_tokens - input types
# ---------------------------------------------------------------------------

def test_get_tokens_str_simple():
    lexer = Lexer()
    lexer.default_initialization()
    result = list(lexer.get_tokens("SELECT 1"))
    types = [t for t, _ in result]
    values = [v for _, v in result]
    assert tokens.Keyword in types
    assert "SELECT" in values
    assert "1" in values


def test_get_tokens_bytes_utf8_default():
    lexer = Lexer()
    lexer.default_initialization()
    result = list(lexer.get_tokens(b"SELECT 1"))
    values = [v for _, v in result]
    assert "1" in values


def test_get_tokens_bytes_with_encoding():
    lexer = Lexer()
    lexer.default_initialization()
    result = list(lexer.get_tokens("SELECT 1".encode("utf-8"), encoding="utf-8"))
    assert any(v == "1" for _, v in result)


def test_get_tokens_bytes_fallback_unicode_escape():
    # bytes that are invalid UTF-8 but valid unicode-escape
    lexer = Lexer()
    lexer.default_initialization()
    data = "\\xe9".encode("ascii")  # literal backslash-xe9
    result = list(lexer.get_tokens(data))
    assert isinstance(result, list)


def test_get_tokens_file_like():
    lexer = Lexer()
    lexer.default_initialization()
    stream = io.StringIO("SELECT 1")
    result = list(lexer.get_tokens(stream))
    values = [v for _, v in result]
    assert "1" in values


def test_get_tokens_invalid_type_raises():
    lexer = Lexer()
    lexer.default_initialization()
    with pytest.raises(TypeError):
        list(lexer.get_tokens(123))


def test_get_tokens_invalid_type_raises_list():
    lexer = Lexer()
    lexer.default_initialization()
    with pytest.raises(TypeError):
        list(lexer.get_tokens(["SELECT"]))


# ---------------------------------------------------------------------------
# get_tokens - tokenization behavior
# ---------------------------------------------------------------------------

def test_get_tokens_empty_string():
    lexer = Lexer()
    lexer.default_initialization()
    assert list(lexer.get_tokens("")) == []


def test_get_tokens_unknown_characters_are_errors():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([])
    result = list(lexer.get_tokens("ab"))
    assert result == [(tokens.Error, "a"), (tokens.Error, "b")]


def test_get_tokens_process_as_keyword_action():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r"\w+", keywords.PROCESS_AS_KEYWORD)])
    lexer.add_keywords({"SELECT": tokens.Keyword})
    result = list(lexer.get_tokens("SELECT foo"))
    assert result[0] == (tokens.Keyword, "SELECT")
    assert result[1] == (tokens.Name, "foo")


def test_get_tokens_returns_generator():
    lexer = Lexer()
    lexer.default_initialization()
    gen = lexer.get_tokens("SELECT 1")
    assert hasattr(gen, "__next__")
    # Consume to avoid unclosed generator issues
    list(gen)


def test_get_tokens_string_literal_single_line():
    lexer = Lexer()
    lexer.default_initialization()
    result = list(lexer.get_tokens("SELECT 'hello'"))
    values = [v for _, v in result]
    assert "'hello'" in values


def test_get_tokens_comment():
    lexer = Lexer()
    lexer.default_initialization()
    result = list(lexer.get_tokens("-- comment\nSELECT 1"))
    types = [t for t, _ in result]
    assert tokens.Comment.Single in types


def test_get_tokens_number_token_type():
    lexer = Lexer()
    lexer.default_initialization()
    result = list(lexer.get_tokens("1"))
    types = [t for t, _ in result]
    assert tokens.Number.Integer in types


def test_get_tokens_whitespace():
    lexer = Lexer()
    lexer.default_initialization()
    result = list(lexer.get_tokens("SELECT 1"))
    types = [t for t, _ in result]
    assert tokens.Whitespace in types


# ---------------------------------------------------------------------------
# default_initialization
# ---------------------------------------------------------------------------

def test_default_initialization_loads_keywords():
    lexer = Lexer()
    lexer.default_initialization()
    ttype, _ = lexer.is_keyword("select")
    assert ttype is not tokens.Name


def test_default_initialization_resets_prior_custom():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({"CUSTOMKW": tokens.Keyword})
    lexer.default_initialization()
    ttype, _ = lexer.is_keyword("customkw")
    assert ttype is tokens.Name


def test_default_initialization_sets_regex():
    lexer = Lexer()
    lexer.default_initialization()
    assert len(lexer._SQL_REGEX) == len(keywords.SQL_REGEX)


# ---------------------------------------------------------------------------
# tokenize() module-level function
# ---------------------------------------------------------------------------

def test_tokenize_returns_iterator():
    gen = tokenize("SELECT 1")
    assert hasattr(gen, "__next__")
    list(gen)


def test_tokenize_basic():
    result = list(tokenize("SELECT 1"))
    values = [v for _, v in result]
    assert "SELECT" in values
    assert "1" in values


def test_tokenize_with_encoding():
    result = list(tokenize(b"SELECT 1", "utf-8"))
    values = [v for _, v in result]
    assert "1" in values


def test_tokenize_invalid_type_raises():
    with pytest.raises(TypeError):
        list(tokenize(42))


def test_tokenize_empty():
    assert list(tokenize("")) == []


def test_tokenize_uses_default_instance():
    lexer = Lexer.get_default_instance()
    # Modify the default instance; tokenize should reflect it.
    lexer.add_keywords({"MAGICWORD": tokens.Keyword})
    result = list(tokenize("MAGICWORD"))
    assert result == [(tokens.Keyword, "MAGICWORD")]


# ---------------------------------------------------------------------------
# Delimited spans handling (dollar-quoted, multiline comments)
# ---------------------------------------------------------------------------

def test_get_tokens_dollar_quoted_literal():
    lexer = Lexer()
    lexer.default_initialization()
    sql = "SELECT $$hello world$$"
    result = list(lexer.get_tokens(sql))
    values = "".join(v for _, v in result)
    assert values == sql


def test_get_tokens_multiline_comment():
    lexer = Lexer()
    lexer.default_initialization()
    sql = "/* multi\nline */"
    result = list(lexer.get_tokens(sql))
    types = [t for t, _ in result]
    assert tokens.Comment.Multiline in types


def test_get_tokens_delimiter_in_string_not_resolved_as_span():
    # A $$ inside a single quoted string should not be treated as a
    # dollar-quote opener; the whole string is a single token.
    lexer = Lexer()
    lexer.default_initialization()
    result = list(lexer.get_tokens("'$$'"))
    assert result == [(tokens.String.Single, "'$$'")]
