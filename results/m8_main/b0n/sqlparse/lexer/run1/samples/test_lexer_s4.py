import builtins
import io
import re
import threading

import pytest

from sqlparse import keywords, tokens
from sqlparse.lexer import Lexer, tokenize


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_default_lexer():
    """Keep the singleton from leaking custom state between tests."""
    saved = Lexer._default_instance
    Lexer._default_instance = None
    yield
    Lexer._default_instance = saved


def _lex(sql, lexer=None, encoding=None):
    """Tokenize and return a list of (ttype, value) pairs."""
    lexer = lexer or Lexer.get_default_instance()
    return list(lexer.get_tokens(sql, encoding))


def _find(ttype_list, value):
    return [tt for tt, val in ttype_list if val == value]


# ---------------------------------------------------------------------------
# get_default_instance / singleton
# ---------------------------------------------------------------------------

def test_get_default_instance_returns_lexer():
    inst = Lexer.get_default_instance()
    assert isinstance(inst, Lexer)


def test_get_default_instance_is_singleton():
    first = Lexer.get_default_instance()
    second = Lexer.get_default_instance()
    assert first is second


def test_get_default_instance_is_initialized():
    inst = Lexer.get_default_instance()
    # default_initialization sets regex list and keyword dicts
    assert len(inst._SQL_REGEX) > 0
    assert len(inst._keywords) > 0
    assert inst._SQL_REGEX == inst._SQL_REGEX  # stability check


def test_get_default_instance_thread_safe_singleton():
    Lexer._default_instance = None
    results = []

    def worker():
        results.append(Lexer.get_default_instance())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# clear / set_SQL_REGEX / add_keywords / default_initialization
# ---------------------------------------------------------------------------

def test_clear_empties_configuration():
    lexer = Lexer()
    lexer.default_initialization()
    assert lexer._SQL_REGEX
    assert lexer._keywords
    lexer.clear()
    assert lexer._SQL_REGEX == []
    assert lexer._keywords == []


def test_set_sql_regex_compiles_with_flags():
    lexer = Lexer()
    lexer.set_SQL_REGEX([(r"\d+", tokens.Number.Integer)])
    assert len(lexer._SQL_REGEX) == 1
    match_fn, action = lexer._SQL_REGEX[0]
    assert action is tokens.Number.Integer
    m = match_fn("abc123", 3)
    assert m is not None
    assert m.group() == "123"
    # case-insensitive flag must apply
    lexer.set_SQL_REGEX([(r"abc", tokens.Keyword)])
    m = match_fn = lexer._SQL_REGEX[0][0]("XXXABC", 3)
    assert m is not None


def test_set_sql_regex_replaces_previous():
    lexer = Lexer()
    lexer.set_SQL_REGEX([(r"a", tokens.Keyword)])
    lexer.set_SQL_REGEX([(r"b", tokens.Keyword), (r"c", tokens.Name)])
    assert len(lexer._SQL_REGEX) == 2


def test_add_keywords_appends_in_order():
    lexer = Lexer()
    lexer.clear()
    d1 = {"FOO": tokens.Keyword}
    d2 = {"BAR": tokens.Keyword}
    lexer.add_keywords(d1)
    lexer.add_keywords(d2)
    assert lexer._keywords == [d1, d2]


def test_default_initialization_resets_custom_state():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({"CUSTOM": tokens.Keyword})
    lexer.default_initialization()
    # custom dict removed, defaults present
    assert {"CUSTOM": tokens.Keyword} not in lexer._keywords
    assert len(lexer._keywords) >= 8
    assert len(lexer._SQL_REGEX) == len(keywords.SQL_REGEX)


# ---------------------------------------------------------------------------
# is_keyword
# ---------------------------------------------------------------------------

def test_is_keyword_recognizes_default_keyword():
    lexer = Lexer.get_default_instance()
    expected_type, value = lexer.is_keyword("select")
    assert expected_type is tokens.Keyword
    assert value == "select"


def test_is_keyword_is_case_insensitive():
    lexer = Lexer.get_default_instance()
    lower = lexer.is_keyword("select")
    upper = lexer.is_keyword("SELECT")
    assert lower == upper


def test_is_keyword_returns_name_for_unknown():
    lexer = Lexer.get_default_instance()
    ttype, value = lexer.is_keyword("someunknownword")
    assert ttype is tokens.Name
    assert value == "someunknownword"


def test_is_keyword_respects_added_order():
    lexer = Lexer()
    lexer.clear()
    lexer.add_keywords({"FOO": tokens.Keyword})
    lexer.add_keywords({"FOO": tokens.Name})
    ttype, value = lexer.is_keyword("foo")
    assert ttype is tokens.Keyword
    assert value == "foo"


def test_is_keyword_preserves_original_case():
    lexer = Lexer.get_default_instance()
    ttype, value = lexer.is_keyword("SeLeCt")
    assert value == "SeLeCt"


# ---------------------------------------------------------------------------
# get_tokens: input types
# ---------------------------------------------------------------------------

def test_get_tokens_simple_string():
    result = _lex("select")
    assert result == [(tokens.Keyword.DML, "select")]


def test_get_tokens_accepts_text_io():
    stream = io.StringIO("select")
    result = list(Lexer.get_default_instance().get_tokens(stream))
    assert result == [(tokens.Keyword.DML, "select")]


def test_get_tokens_bytes_utf8():
    result = _lex(b"select")
    assert result == [(tokens.Keyword.DML, "select")]


def test_get_tokens_bytes_with_encoding():
    result = _lex("select".encode("utf-8"), encoding="utf-8")
    assert result == [(tokens.Keyword.DML, "select")]


def test_get_tokens_bytes_unicode_escape_fallback():
    # invalid utf-8 byte sequence (0xff) cannot be decoded as utf-8
    result = _lex(b"\xff")
    # unicode-escape decodes \xff into the char '\xff'
    assert result == [(tokens.Error, "\xff")]


def test_get_tokens_invalid_type_raises():
    lexer = Lexer.get_default_instance()
    with pytest.raises(TypeError):
        list(lexer.get_tokens(12345))


def test_get_tokens_invalid_type_message():
    lexer = Lexer.get_default_instance()
    with pytest.raises(TypeError) as excinfo:
        list(lexer.get_tokens([1, 2, 3]))
    assert "Expected text or file-like object" in str(excinfo.value)


def test_get_tokens_is_generator():
    lexer = Lexer.get_default_instance()
    result = lexer.get_tokens("select")
    assert hasattr(result, "__next__")
    assert hasattr(result, "__iter__")


# ---------------------------------------------------------------------------
# get_tokens: token types and values
# ---------------------------------------------------------------------------

def test_get_tokens_whitespace():
    result = _lex("select 1")
    values = [v for _, v in result]
    assert values == ["select", " ", "1"]


def test_get_tokens_number():
    result = _lex("123")
    assert result == [(tokens.Number.Integer, "123")]


def test_get_tokens_float_number():
    result = _lex("1.5")
    assert result[0][0] in (tokens.Number.Float, tokens.Number)


def test_get_tokens_string_literal():
    result = _lex("'abc'")
    assert result == [(tokens.String.Single, "'abc'")]


def test_get_tokens_quoted_identifier():
    result = _lex('"col"')
    assert result == [(tokens.Name, '"col"')]


def test_get_tokens_comment():
    result = _lex("-- a comment\n")
    assert any(tt in tokens.Comment for tt, _ in result)


def test_get_tokens_multiline_comment():
    result = _lex("/* block */")
    assert any(tt in tokens.Comment.Multiline for tt, _ in result)


def test_get_tokens_punctuation():
    result = _lex(";")
    assert result == [(tokens.Punctuation, ";")]


def test_get_tokens_keyword_and_name():
    result = _lex("select foo")
    types = [tt for tt, _ in result]
    assert tokens.Keyword.DML in types
    assert tokens.Name in types


# ---------------------------------------------------------------------------
# get_tokens: error / unknown characters
# ---------------------------------------------------------------------------

def test_get_tokens_unknown_char_yields_error():
    # Use an isolated lexer with only a limited regex to force the fallback.
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([(r"select", tokens.Keyword)])
    result = list(lexer.get_tokens("?"))
    assert result == [(tokens.Error, "?")]


# ---------------------------------------------------------------------------
# get_tokens: delimited spans (strings, comments, dollar quotes)
# ---------------------------------------------------------------------------

def test_dollar_quoted_string():
    result = _lex("$$abc$$")
    assert any("$$abc$$" in v for _, v in result)


def test_delimiter_inside_string_not_resolved():
    # The '$$' inside a single-quoted string must not open a dollar quote.
    sql = "'$$' foo"
    result = _lex(sql)
    values = [v for _, v in result]
    # The entire string literal stays together
    assert "'$$'" in values


def test_delimiter_inside_line_comment_ignored():
    sql = "-- $$\nselect"
    result = _lex(sql)
    # A DML keyword should follow the comment; the '$$' must not swallow rest
    assert (tokens.Keyword.DML, "select") in result


def test_multiline_comment_as_span():
    sql = "/* a\nb */select"
    result = _lex(sql)
    assert (tokens.Keyword.DML, "select") in result


# ---------------------------------------------------------------------------
# clear on default instance affects tokenization
# ---------------------------------------------------------------------------

def test_cleared_lexer_yields_errors_for_everything():
    lexer = Lexer()
    lexer.clear()
    lexer.set_SQL_REGEX([])
    result = list(lexer.get_tokens("abc"))
    assert result == [
        (tokens.Error, "a"),
        (tokens.Error, "b"),
        (tokens.Error, "c"),
    ]


def test_custom_keyword_via_add_keywords():
    lexer = Lexer()
    lexer.default_initialization()
    lexer.add_keywords({"MYCUSTOM": tokens.Keyword})
    ttype, _ = lexer.is_keyword("mycustom")
    assert ttype is tokens.Keyword


# ---------------------------------------------------------------------------
# tokenize module-level function
# ---------------------------------------------------------------------------

def test_tokenize_returns_iterable_of_pairs():
    stream = tokenize("select")
    result = list(stream)
    assert result == [(tokens.Keyword.DML, "select")]


def test_tokenize_uses_default_instance():
    Lexer._default_instance = None
    stream = tokenize("select")
    result = list(stream)
    assert result == [(tokens.Keyword.DML, "select")]
    assert Lexer._default_instance is not None


def test_tokenize_with_encoding():
    result = list(tokenize(b"select", "utf-8"))
    assert result == [(tokens.Keyword.DML, "select")]


def test_tokenize_invalid_type_raises():
    with pytest.raises(TypeError):
        list(tokenize(42))


def test_tokenize_text_io():
    result = list(tokenize(io.StringIO("select")))
    assert result == [(tokens.Keyword.DML, "select")]


def test_tokenize_empty_string():
    assert list(tokenize("")) == []
