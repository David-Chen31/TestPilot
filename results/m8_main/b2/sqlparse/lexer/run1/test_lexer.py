import io

import pytest

from sqlparse import tokens, keywords
from sqlparse.lexer import Lexer, tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def lexer():
    """A fresh, fully initialized lexer instance (not the singleton)."""
    lx = Lexer()
    lx.default_initialization()
    return lx


def token_types(stream):
    return [ttype for ttype, _ in stream]


def token_values(stream):
    return [value for _, value in stream]


# ---------------------------------------------------------------------------
# Lexer.get_default_instance
# ---------------------------------------------------------------------------

def test_get_default_instance_is_singleton():
    inst1 = Lexer.get_default_instance()
    inst2 = Lexer.get_default_instance()
    assert inst1 is inst2
    assert isinstance(inst1, Lexer)


def test_get_default_instance_initialized():
    inst = Lexer.get_default_instance()
    assert len(inst._SQL_REGEX) > 0
    assert len(inst._keywords) > 0


def test_get_default_instance_usable(monkeypatch):
    monkeypatch.setattr(Lexer, "_default_instance", None)
    inst = Lexer.get_default_instance()
    assert inst is not None
    assert isinstance(inst, Lexer)
    assert list(inst.get_tokens("SELECT")) == [(tokens.Keyword.DML, "SELECT")]


# ---------------------------------------------------------------------------
# clear / default_initialization / set_SQL_REGEX / add_keywords
# ---------------------------------------------------------------------------

def test_clear_resets_configuration(lexer):
    assert lexer._SQL_REGEX
    assert lexer._keywords
    lexer.clear()
    assert lexer._SQL_REGEX == []
    assert lexer._keywords == []


def test_default_initialization_populates(lexer):
    lexer.clear()
    lexer.default_initialization()
    assert len(lexer._SQL_REGEX) == len(keywords.SQL_REGEX)
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
    assert lexer._keywords == expected


def test_set_SQL_REGEX_compiles(lexer):
    lexer.clear()
    lexer.set_SQL_REGEX([(r"\d+", tokens.Number.Integer)])
    assert len(lexer._SQL_REGEX) == 1
    rexmatch, action = lexer._SQL_REGEX[0]
    assert action is tokens.Number.Integer
    assert rexmatch("123", 0).group() == "123"
    lexer.set_SQL_REGEX([(r"abc", tokens.Keyword)])
    rexmatch, _ = lexer._SQL_REGEX[0]
    assert rexmatch("ABC", 0) is not None


def test_add_keywords_appends(lexer):
    lexer.clear()
    lexer.add_keywords({"FOO": tokens.Keyword})
    lexer.add_keywords({"BAR": tokens.Keyword})
    assert lexer._keywords == [{"FOO": tokens.Keyword}, {"BAR": tokens.Keyword}]


# ---------------------------------------------------------------------------
# is_keyword
# ---------------------------------------------------------------------------

def test_is_keyword_hit_is_case_insensitive(lexer):
    lexer.clear()
    lexer.add_keywords({"SELECT": tokens.Keyword.DML})
    ttype, value = lexer.is_keyword("select")
    assert ttype is tokens.Keyword.DML
    assert value == "select"


def test_is_keyword_returns_original_value(lexer):
    lexer.clear()
    lexer.add_keywords({"SELECT": tokens.Keyword.DML})
    ttype, value = lexer.is_keyword("SeLeCt")
    assert ttype is tokens.Keyword.DML
    assert value == "SeLeCt"


def test_is_keyword_miss_returns_name(lexer):
    lexer.clear()
    lexer.add_keywords({"SELECT": tokens.Keyword.DML})
    ttype, value = lexer.is_keyword("foo")
    assert ttype is tokens.Name
    assert value == "foo"


def test_is_keyword_order_respected(lexer):
    lexer.clear()
    lexer.add_keywords({"X": tokens.Keyword})
    lexer.add_keywords({"X": tokens.Name})
    ttype, _ = lexer.is_keyword("x")
    assert ttype is tokens.Keyword


def test_is_keyword_empty_keywords(lexer):
    lexer.clear()
    ttype, value = lexer.is_keyword("anything")
    assert ttype is tokens.Name
    assert value == "anything"


# ---------------------------------------------------------------------------
# get_tokens - input handling
# ---------------------------------------------------------------------------

def test_get_tokens_str(lexer):
    result = list(lexer.get_tokens("SELECT 1"))
    assert result[0] == (tokens.Keyword.DML, "SELECT")
    assert (tokens.Whitespace, " ") in result


def test_get_tokens_file_like(lexer):
    result = list(lexer.get_tokens(io.StringIO("SELECT 1")))
    assert result[0] == (tokens.Keyword.DML, "SELECT")


def test_get_tokens_bytes_utf8(lexer):
    result = list(lexer.get_tokens("SELECT".encode("utf-8")))
    assert result == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_bytes_utf8_multibyte(lexer):
    text = "caf\u00e9"
    result = list(lexer.get_tokens(text.encode("utf-8")))
    assert result == [(tokens.Name, "caf\u00e9")]


def test_get_tokens_bytes_with_encoding(lexer):
    result = list(lexer.get_tokens("SELECT".encode("latin-1"), "latin-1"))
    assert result == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_bytes_with_encoding_non_utf8(lexer):
    text = "caf\u00e9"
    result = list(lexer.get_tokens(text.encode("latin-1"), "latin-1"))
    assert result == [(tokens.Name, "caf\u00e9")]


def test_get_tokens_bytes_fallback_unicode_escape(lexer):
    data = b"\xff"
    result = list(lexer.get_tokens(data))
    assert isinstance(result, list)
    assert len(result) >= 1


def test_get_tokens_bytes_fallback_unicode_escape_exact(lexer):
    data = b"\xff"
    result = list(lexer.get_tokens(data))
    assert result == [(tokens.Name, "\u00ff")]


def test_get_tokens_bytes_fallback_distinguishes_from_utf8(lexer):
    data = "é".encode("utf-8")
    result = list(lexer.get_tokens(data))
    assert result == [(tokens.Name, "é")]


def test_get_tokens_bytes_utf8_uppercase_of_keyword(lexer):
    data = b"SELECT"
    assert list(lexer.get_tokens(data)) == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_invalid_type_raises(lexer):
    with pytest.raises(TypeError) as excinfo:
        list(lexer.get_tokens(12345))
    assert "Expected text or file-like object" in str(excinfo.value)


def test_get_tokens_invalid_type_message_has_type(lexer):
    with pytest.raises(TypeError) as excinfo:
        list(lexer.get_tokens(12345))
    assert str(excinfo.value) == (
        "Expected text or file-like object, got <class 'int'>"
    )


def test_get_tokens_empty_string(lexer):
    assert list(lexer.get_tokens("")) == []


# ---------------------------------------------------------------------------
# get_tokens - token classification
# ---------------------------------------------------------------------------

def test_get_tokens_name(lexer):
    result = list(lexer.get_tokens("foo"))
    assert result == [(tokens.Name, "foo")]


def test_get_tokens_number(lexer):
    result = list(lexer.get_tokens("42"))
    assert result == [(tokens.Number.Integer, "42")]


def test_get_tokens_float(lexer):
    result = list(lexer.get_tokens("3.14"))
    assert result == [(tokens.Number.Float, "3.14")]


def test_get_tokens_whitespace(lexer):
    result = list(lexer.get_tokens("   "))
    assert result == [
        (tokens.Whitespace, " "),
        (tokens.Whitespace, " "),
        (tokens.Whitespace, " "),
    ]


def test_get_tokens_newline(lexer):
    result = list(lexer.get_tokens("\n"))
    assert result == [(tokens.Newline, "\n")]


def test_get_tokens_single_quoted_string(lexer):
    result = list(lexer.get_tokens("'abc'"))
    assert result == [(tokens.String.Single, "'abc'")]


def test_get_tokens_operator(lexer):
    result = list(lexer.get_tokens("="))
    assert result == [(tokens.Operator.Comparison, "=")]


def test_get_tokens_keyword_process_as_keyword(lexer):
    result = list(lexer.get_tokens("SELECT"))
    assert result == [(tokens.Keyword.DML, "SELECT")]


def test_get_tokens_unknown_char_yields_error(lexer):
    lexer.clear()
    result = list(lexer.get_tokens("?"))
    assert result == [(tokens.Error, "?")]


def test_get_tokens_error_char_among_known(lexer):
    lexer.clear()
    lexer.add_keywords({})
    result = list(lexer.get_tokens("ab"))
    assert result == [(tokens.Error, "a"), (tokens.Error, "b")]


# ---------------------------------------------------------------------------
# get_tokens - multiple tokens and consumption
# ---------------------------------------------------------------------------

def test_get_tokens_multiple(lexer):
    result = list(lexer.get_tokens("SELECT 1;"))
    assert result == [
        (tokens.Keyword.DML, "SELECT"),
        (tokens.Whitespace, " "),
        (tokens.Number.Integer, "1"),
        (tokens.Punctuation, ";"),
    ]


def test_get_tokens_is_generator(lexer):
    gen = lexer.get_tokens("SELECT")
    assert hasattr(gen, "__next__")
    first = next(gen)
    assert first == (tokens.Keyword.DML, "SELECT")


def test_get_tokens_multiline_comment(lexer):
    sql = "/* hello\nworld */"
    result = list(lexer.get_tokens(sql))
    assert result == [(tokens.Comment.Multiline, sql)]


def test_get_tokens_comment_prefix(lexer):
    result = list(lexer.get_tokens("--foo\n"))
    assert (tokens.Comment.Single, "--foo\n") in result


def test_get_tokens_comment_prefix_without_newline(lexer):
    result = list(lexer.get_tokens("--foo"))
    assert result == [(tokens.Comment.Single, "--foo")]


def test_get_tokens_dollar_quoted(lexer):
    sql = "$$abc$$"
    result = list(lexer.get_tokens(sql))
    assert result == [(tokens.Literal, "$$abc$$")]


def test_get_tokens_dollar_quoted_then_more(lexer):
    sql = "$$abc$$ x"
    result = list(lexer.get_tokens(sql))
    assert result == [
        (tokens.Literal, "$$abc$$"),
        (tokens.Whitespace, " "),
        (tokens.Name, "x"),
    ]


def test_get_tokens_dollar_quoted_followed_immediately(lexer):
    sql = "$$ab$$;"
    result = list(lexer.get_tokens(sql))
    assert result == [
        (tokens.Literal, "$$ab$$"),
        (tokens.Punctuation, ";"),
    ]


def test_get_tokens_dollar_quoted_nonzero_pos(lexer):
    sql = "x $$abc$$ y"
    result = list(lexer.get_tokens(sql))
    assert result == [
        (tokens.Name, "x"),
        (tokens.Whitespace, " "),
        (tokens.Literal, "$$abc$$"),
        (tokens.Whitespace, " "),
        (tokens.Name, "y"),
    ]


def test_get_tokens_dollar_quoted_mid_string_overconsumption_guard(lexer):
    sql = "ab $$L$$ cd"
    result = list(lexer.get_tokens(sql))
    assert result == [
        (tokens.Name, "ab"),
        (tokens.Whitespace, " "),
        (tokens.Literal, "$$L$$"),
        (tokens.Whitespace, " "),
        (tokens.Name, "cd"),
    ]


def test_get_tokens_multiline_comment_then_more(lexer):
    sql = "/*c*/ 42"
    result = list(lexer.get_tokens(sql))
    assert result == [
        (tokens.Comment.Multiline, "/*c*/"),
        (tokens.Whitespace, " "),
        (tokens.Number.Integer, "42"),
    ]


def test_get_tokens_multiline_comment_nonzero_pos(lexer):
    sql = "x /*c*/ y"
    result = list(lexer.get_tokens(sql))
    assert result == [
        (tokens.Name, "x"),
        (tokens.Whitespace, " "),
        (tokens.Comment.Multiline, "/*c*/"),
        (tokens.Whitespace, " "),
        (tokens.Name, "y"),
    ]


# ---------------------------------------------------------------------------
# tokenize module-level function
# ---------------------------------------------------------------------------

def test_tokenize_uses_default_instance():
    result = list(tokenize("SELECT 1"))
    assert result[0] == (tokens.Keyword.DML, "SELECT")


def test_tokenize_with_encoding():
    result = list(tokenize("SELECT".encode("utf-8"), "utf-8"))
    assert result == [(tokens.Keyword.DML, "SELECT")]


def test_tokenize_with_encoding_used():
    result = list(tokenize(b"\xe9", "latin-1"))
    assert result == [(tokens.Name, "\u00e9")]


def test_tokenize_with_encoding_latin1_multibyte():
    text = "caf\u00e9"
    result = list(tokenize(text.encode("latin-1"), "latin-1"))
    assert result == [(tokens.Name, "caf\u00e9")]


def test_tokenize_encoding_actually_forwarded():
    # latin-1 decoding of b'\xc3\xa9' gives two chars: '\xc3' (Ã) and
    # '\xa9' (©). The Name regex matches the first, but '\xa9' is not
    # matched by any default regex -> Error token.
    result = list(tokenize("é".encode("utf-8"), "latin-1"))
    assert result == [(tokens.Name, "Ã"), (tokens.Error, "©")]


def test_tokenize_encoding_forwarded_vs_none():
    data = "é".encode("utf-8")
    with_enc = list(tokenize(data, "latin-1"))
    without_enc = list(tokenize(data, None))
    with_enc_default = list(tokenize(data))
    assert with_enc == [(tokens.Name, "Ã"), (tokens.Error, "©")]
    assert without_enc == [(tokens.Name, "é")]
    assert with_enc_default == [(tokens.Name, "é")]
    assert with_enc != without_enc


def test_tokenize_encoding_undecodable_required():
    # latin-1 decoding of b'\xff\x80' -> '\xff' (ÿ) then '\x80'.
    # Name regex matches the first char; '\x80' isn't matched -> Error.
    data = b"\xff\x80"
    explicit = list(tokenize(data, "latin-1"))
    assert explicit == [(tokens.Name, "\u00ff"), (tokens.Error, "\u0080")]


def test_tokenize_forwarded_encoding_distinguishes_from_none():
    # b'\xc3\xa9' as latin-1 -> 'Ã', '©' (two chars -> Name + Error).
    # With encoding=None the utf-8 path succeeds -> single 'é' Name.
    data = b"\xc3\xa9"
    result = list(tokenize(data, "latin-1"))
    assert result == [(tokens.Name, "Ã"), (tokens.Error, "©")]
    result_none = list(tokenize(data, None))
    assert result_none == [(tokens.Name, "é")]
    assert result != result_none


def test_tokenize_empty():
    assert list(tokenize("")) == []


def test_tokenize_invalid_type():
    with pytest.raises(TypeError):
        list(tokenize(12345))


def test_tokenize_returns_generator():
    stream = tokenize("SELECT")
    assert hasattr(stream, "__next__")
    assert next(stream) == (tokens.Keyword.DML, "SELECT")
