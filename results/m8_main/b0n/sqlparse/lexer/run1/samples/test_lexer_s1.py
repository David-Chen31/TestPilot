import io
import re
from threading import Thread

import pytest

from sqlparse import tokens
from sqlparse import keywords
from sqlparse.lexer import Lexer, tokenize


# ---------------------------------------------------------------------------
# tokenize / default instance
# ---------------------------------------------------------------------------

def test_tokenize_returns_generator():
    result = tokenize("SELECT 1")
    assert not isinstance(result, list)
    assert hasattr(result, "__next__")
    toks = list(result)
    assert toks
    for ttype, value in toks:
        assert isinstance(ttype, tokens._TokenType)
        assert isinstance(value, str)


def test_tokenize_simple_select():
    toks = list(tokenize("SELECT 1"))
    # first token is the SELECT keyword (possibly whitespace separated)
    non_ws = [(tt, v) for tt, v in toks if not tt is tokens.Whitespace]
    assert non_ws[0] == (tokens.Keyword.DML, "SELECT")
    assert non_ws[1] == (tokens.Number.Integer, "1")


def test_tokenize_empty_string():
    assert list(tokenize("")) == []


def test_tokenize_bytes_utf8_default():
    toks = list(tokenize(b"SELECT 1"))
    non_ws = [(tt, v) for tt, v in toks if tt is not tokens.Whitespace]
    assert non_ws[0] == (tokens.Keyword.DML, "SELECT")


def test_tokenize_bytes_with_encoding():
    text = "SELECT 'ä'"
    toks = list(tokenize(text.encode("latin-1"), encoding="latin-1"))
    values = [v for _, v in toks]
    assert "'ä'" in values


def test_tokenize_bytes_invalid_utf8_falls_back():
    # 0xE9 alone is invalid utf-8; unicode-escape decodes as 'é'
    data = "a".encode("ascii") + b"\xe9"
    toks = list(tokenize(data))
    assert any("\xe9" in v for _, v in toks)


def test_tokenize_accepts_file_like():
    buf = io.StringIO("SELECT 1")
    toks = list(tokenize(buf))
    non_ws = [(tt, v) for tt, v in toks if tt is not tokens.Whitespace]
    assert non_ws[0] == (tokens.Keyword.DML, "SELECT")


def test_tokenize_type_error_on_other_type():
    with pytest.raises(TypeError):
        list(tokenize(12345))
    with pytest.raises(TypeError):
        list(tokenize(["SELECT"]))


def test_tokenize_default_instance_singleton():
    a = Lexer.get_default_instance()
    b = Lexer.get_default_instance()
    assert a is b
    assert isinstance(a, Lexer)


def test_tokenize_uses_default_instance():
    default = Lexer.get_default_instance()
    toks = list(tokenize("SELECT 1"))
    assert toks == list(default.get_tokens("SELECT 1"))


# ---------------------------------------------------------------------------
# Lexer clear / set_SQL_REGEX / add_keywords / is_keyword
# ---------------------------------------------------------------------------

def test_clear_empties_configuration():
    lex = Lexer()
    lex.default_initialization()
    assert lex._keywords
    assert lex._SQL_REGEX
    lex.clear()
    assert lex._keywords == []
    assert lex._SQL_REGEX == []


def test_set_SQL_REGEX_compiles_matchers_case_insensitive():
    lex = Lexer()
    lex.set_SQL_REGEX([(r"abc", tokens.Name)])
    assert len(lex._SQL_REGEX) == 1
    rexmatch, action = lex._SQL_REGEX[0]
    assert action is tokens.Name
    # case-insensitive due to re.IGNORECASE
    m = rexmatch("ABC", 0)
    assert m is not None
    assert m.group() == "ABC"


def test_set_SQL_REGEX_is_unicode():
    lex = Lexer()
    lex.set_SQL_REGEX([(r"\w+", tokens.Name)])
    rexmatch, _ = lex._SQL_REGEX[0]
    m = rexmatch("äöü", 0)
    assert m is not None
    assert m.group() == "äöü"


def test_add_keywords_single_dict():
    lex = Lexer()
    lex.clear()
    lex.add_keywords({"FOO": tokens.Keyword})
    assert len(lex._keywords) == 1
    assert lex.is_keyword("FOO") == (tokens.Keyword, "FOO")
    assert lex.is_keyword("foo") == (tokens.Keyword, "foo")
    assert lex.is_keyword("bar") == (tokens.Name, "bar")


def test_is_keyword_order_priority():
    lex = Lexer()
    lex.clear()
    lex.add_keywords({"FOO": tokens.Keyword})
    lex.add_keywords({"FOO": tokens.Keyword.DML})
    # first dictionary wins
    assert lex.is_keyword("foo")[0] is tokens.Keyword
    assert lex.is_keyword("FOO")[0] is tokens.Keyword


def test_is_keyword_not_found_returns_name():
    lex = Lexer()
    lex.clear()
    lex.add_keywords({"FOO": tokens.Keyword})
    ttype, value = lex.is_keyword("notakeyword")
    assert ttype is tokens.Name
    assert value == "notakeyword"


def test_is_keyword_case_insensitive_lookup():
    lex = Lexer()
    lex.clear()
    lex.add_keywords({"SELECT": tokens.Keyword.DML})
    ttype, value = lex.is_keyword("select")
    assert ttype is tokens.Keyword.DML
    # original value preserved in result
    assert value == "select"


# ---------------------------------------------------------------------------
# get_tokens
# ---------------------------------------------------------------------------

def test_get_tokens_simple():
    lex = Lexer.get_default_instance()
    toks = list(lex.get_tokens("SELECT"))
    assert toks == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_error_token_for_unknown_char():
    lex = Lexer()
    lex.clear()
    # no regex configured -> everything is an Error char
    toks = list(lex.get_tokens("ab"))
    assert toks == [(tokens.Error, "a"), (tokens.Error, "b")]


def test_get_tokens_with_minimal_regex():
    lex = Lexer()
    lex.clear()
    lex.set_SQL_REGEX([(r"\s+", tokens.Whitespace), (r"[a-zA-Z_]\w*",
                                                     keywords.PROCESS_AS_KEYWORD)])
    lex.add_keywords({"FOO": tokens.Keyword})
    toks = list(lex.get_tokens("FOO x"))
    assert toks == [
        (tokens.Keyword, "FOO"),
        (tokens.Whitespace, " "),
        (tokens.Name, "x"),
    ]


def test_get_tokens_process_as_keyword_returns_name():
    lex = Lexer()
    lex.clear()
    lex.set_SQL_REGEX([(r"[a-zA-Z_]\w*", keywords.PROCESS_AS_KEYWORD)])
    lex.add_keywords({"FOO": tokens.Keyword})
    assert list(lex.get_tokens("bar")) == [(tokens.Name, "bar")]


def test_get_tokens_file_like_input():
    lex = Lexer.get_default_instance()
    toks = list(lex.get_tokens(io.StringIO("SELECT")))
    assert toks == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_bytes_input():
    lex = Lexer.get_default_instance()
    toks = list(lex.get_tokens(b"SELECT"))
    assert toks == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_bad_type_raises():
    lex = Lexer.get_default_instance()
    with pytest.raises(TypeError):
        list(lex.get_tokens(1))


def test_get_tokens_string_literal_delimiter_skipped():
    # A quote inside a string must not be treated as opening a dollar quote.
    lex = Lexer.get_default_instance()
    sql = "SELECT 'a -- b'"
    toks = list(lex.get_tokens(sql))
    assert any("'a -- b'" in v for _, v in toks)


def test_get_tokens_dollar_quoted():
    lex = Lexer.get_default_instance()
    sql = "SELECT $tag$ body $tag$"
    toks = list(lex.get_tokens(sql))
    values = [v for _, v in toks]
    assert "$tag$ body $tag$" in values


def test_get_tokens_multiline_comment():
    lex = Lexer.get_default_instance()
    sql = "SELECT /* a\nb */ 1"
    toks = list(lex.get_tokens(sql))
    values = [v for _, v in toks]
    assert "/* a\nb */" in values


def test_get_tokens_positions_reindex_after_match():
    # When a multi-char match occurs, iteration must resume after it, not
    # re-scan the consumed characters.
    lex = Lexer()
    lex.clear()
    lex.set_SQL_REGEX([(r"[0-9]+", tokens.Number.Integer)])
    toks = list(lex.get_tokens("12[3"))
    assert toks == [
        (tokens.Number.Integer, "12"),
        (tokens.Error, "["),
        (tokens.Number.Integer, "3"),
    ]


# ---------------------------------------------------------------------------
# default_initialization
# ---------------------------------------------------------------------------

def test_default_initialization_populates():
    lex = Lexer()
    lex.clear()
    lex.default_initialization()
    assert lex._SQL_REGEX
    assert lex._keywords
    # multiple keyword dictionaries added
    assert len(lex._keywords) >= 5


def test_default_initialization_resets_custom():
    lex = Lexer()
    lex.clear()
    lex.add_keywords({"CUSTOM": tokens.Keyword})
    assert lex.is_keyword("CUSTOM")[0] is tokens.Keyword
    lex.default_initialization()
    assert lex.is_keyword("CUSTOM")[0] is tokens.Name


def test_default_initialization_recognizes_dialects():
    lex = Lexer()
    lex.default_initialization()
    # common keyword
    assert lex.is_keyword("select")[0] is tokens.Keyword.DML
    # unknown is Name
    assert lex.is_keyword("zzz_unknown")[0] is tokens.Name


# ---------------------------------------------------------------------------
# thread safety of get_default_instance
# ---------------------------------------------------------------------------

def test_get_default_instance_thread_safe():
    results = []

    def get():
        results.append(Lexer.get_default_instance())

    threads = [Thread(target=get) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    first = results[0]
    assert all(r is first for r in results)
