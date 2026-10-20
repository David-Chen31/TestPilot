import io
import re

import pytest

from sqlparse import tokens, keywords
from sqlparse.lexer import Lexer, tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_lexer():
    """Create a new, fully initialized lexer without touching the singleton."""
    lexer = Lexer()
    lexer.default_initialization()
    return lexer


def pairs(lexer, text, encoding=None):
    return list(lexer.get_tokens(text, encoding))


def values(lexer, text):
    return [v for _, v in lexer.get_tokens(text)]


# ---------------------------------------------------------------------------
# Singleton behavior
# ---------------------------------------------------------------------------

class TestDefaultInstance:
    def test_get_default_instance_returns_same_object(self):
        a = Lexer.get_default_instance()
        b = Lexer.get_default_instance()
        assert a is b

    def test_default_instance_is_lexer_and_initialized(self):
        inst = Lexer.get_default_instance()
        assert isinstance(inst, Lexer)
        # default_initialization populates regex and keyword lists
        assert len(inst._SQL_REGEX) > 0
        assert len(inst._keywords) > 0

    def test_default_instance_is_class_attribute(self):
        inst = Lexer.get_default_instance()
        assert Lexer._default_instance is inst


# ---------------------------------------------------------------------------
# clear / default_initialization / set_SQL_REGEX / add_keywords
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_clear_empties_both(self):
        lexer = fresh_lexer()
        lexer.clear()
        assert lexer._SQL_REGEX == []
        assert lexer._keywords == []

    def test_default_initialization_populates(self):
        lexer = Lexer()
        lexer.default_initialization()
        assert len(lexer._SQL_REGEX) == len(keywords.SQL_REGEX)
        # Callable matchers compiled with flags
        for match, _tt in lexer._SQL_REGEX:
            assert callable(match)

    def test_default_initialization_resets_previous_state(self):
        lexer = fresh_lexer()
        lexer.clear()
        lexer.default_initialization()
        # after re-init, a keyword lookup works again
        ttype, value = lexer.is_keyword("select")
        assert ttype is not tokens.Name
        assert value == "select"

    def test_set_SQL_REGEX_compiles_case_insensitive(self):
        lexer = Lexer()
        lexer.set_SQL_REGEX([(r"foo", tokens.Keyword)])
        assert len(lexer._SQL_REGEX) == 1
        match, ttype = lexer._SQL_REGEX[0]
        m = match("FOO", 0)
        assert m.group() == "FOO"
        assert ttype is tokens.Keyword

    def test_set_SQL_REGEX_replaces_existing(self):
        lexer = fresh_lexer()
        lexer.set_SQL_REGEX([(r"x", tokens.Number)])
        assert len(lexer._SQL_REGEX) == 1

    def test_add_keywords_appends(self):
        lexer = Lexer()
        lexer.clear()
        d1 = {"A": tokens.Keyword}
        d2 = {"B": tokens.Keyword}
        lexer.add_keywords(d1)
        lexer.add_keywords(d2)
        assert lexer._keywords == [d1, d2]


# ---------------------------------------------------------------------------
# is_keyword
# ---------------------------------------------------------------------------

class TestIsKeyword:
    def test_known_keyword(self):
        lexer = fresh_lexer()
        ttype, value = lexer.is_keyword("select")
        assert ttype is not tokens.Name
        assert value == "select"

    def test_keyword_case_insensitive(self):
        lexer = fresh_lexer()
        t1, _ = lexer.is_keyword("SELECT")
        t2, _ = lexer.is_keyword("SeLeCt")
        assert t1 is t2

    def test_unknown_returns_name_and_original_value(self):
        lexer = fresh_lexer()
        ttype, value = lexer.is_keyword("zzznotakeyword")
        assert ttype is tokens.Name
        assert value == "zzznotakeyword"

    def test_custom_keyword_dict(self):
        lexer = Lexer()
        lexer.clear()
        lexer.add_keywords({"CUSTOM": tokens.String.Symbol})
        ttype, value = lexer.is_keyword("custom")
        assert ttype is tokens.String.Symbol
        assert value == "custom"

    def test_lookup_order_first_dict_wins(self):
        lexer = Lexer()
        lexer.clear()
        lexer.add_keywords({"FOO": tokens.Keyword})
        lexer.add_keywords({"FOO": tokens.Number})
        ttype, _ = lexer.is_keyword("foo")
        assert ttype is tokens.Keyword

    def test_empty_keywords_returns_name(self):
        lexer = Lexer()
        lexer.clear()
        ttype, value = lexer.is_keyword("anything")
        assert ttype is tokens.Name
        assert value == "anything"


# ---------------------------------------------------------------------------
# get_tokens: input types
# ---------------------------------------------------------------------------

class TestGetTokensInputTypes:
    def test_str_input(self):
        lexer = fresh_lexer()
        result = pairs(lexer, "SELECT")
        assert result == [(tokens.Keyword.DML, "SELECT")]

    def test_bytes_utf8(self):
        lexer = fresh_lexer()
        result = pairs(lexer, b"SELECT")
        assert result == [(tokens.Keyword.DML, "SELECT")]

    def test_bytes_with_explicit_encoding(self):
        lexer = fresh_lexer()
        result = pairs(lexer, "SELECT".encode("latin-1"), encoding="latin-1")
        assert result == [(tokens.Keyword.DML, "SELECT")]

    def test_bytes_non_utf8_falls_back_to_unicode_escape(self):
        lexer = fresh_lexer()
        raw = "SELECT".encode("utf-16-le")
        # utf-16-le bytes are not valid utf-8; falls to unicode-escape path.
        result = pairs(lexer, raw)
        # Should not raise; produce some tokens as a string.
        assert isinstance(result, list)

    def test_file_like_object(self):
        lexer = fresh_lexer()
        buf = io.StringIO("SELECT")
        result = pairs(lexer, buf)
        assert result == [(tokens.Keyword.DML, "SELECT")]

    def test_invalid_type_raises(self):
        lexer = fresh_lexer()
        with pytest.raises(TypeError) as exc:
            list(lexer.get_tokens(123))
        assert "Expected text or file-like object" in str(exc.value)

    def test_empty_string(self):
        lexer = fresh_lexer()
        assert pairs(lexer, "") == []


# ---------------------------------------------------------------------------
# get_tokens: tokenization behavior
# ---------------------------------------------------------------------------

class TestGetTokensBehavior:
    def test_simple_select(self):
        lexer = fresh_lexer()
        result = pairs(lexer, "SELECT 1")
        assert (tokens.Keyword.DML, "SELECT") in result
        assert any(v == "1" for _, v in result)

    def test_whitespace_token(self):
        lexer = fresh_lexer()
        result = pairs(lexer, "SELECT 1")
        assert (tokens.Whitespace, " ") in result

    def test_unknown_char_yields_error(self):
        lexer = Lexer()
        lexer.clear()
        lexer.set_SQL_REGEX([])
        result = pairs(lexer, "ab")
        # With no regexes, every char becomes an Error token.
        assert result == [(tokens.Error, "a"), (tokens.Error, "b")]

    def test_error_token_for_unmatched(self):
        lexer = Lexer()
        lexer.clear()
        lexer.set_SQL_REGEX([(r"\d+", tokens.Number.Integer)])
        result = pairs(lexer, "1x2")
        assert result == [
            (tokens.Number.Integer, "1"),
            (tokens.Error, "x"),
            (tokens.Number.Integer, "2"),
        ]

    def test_number_token(self):
        lexer = fresh_lexer()
        result = pairs(lexer, "123")
        assert len(result) >= 1
        assert result[0][1] == "123"

    def test_string_literal(self):
        lexer = fresh_lexer()
        result = pairs(lexer, "'abc'")
        assert (tokens.String.Single, "'abc'") in result

    def test_whitespace_newline(self):
        lexer = fresh_lexer()
        result = pairs(lexer, "\n")
        assert (tokens.Whitespace.Newline, "\n") in result

    def test_symbol_punctuation(self):
        lexer = fresh_lexer()
        result = pairs(lexer, "(")
        assert (tokens.Punctuation, "(") in result

    def test_multiline_comment(self):
        lexer = fresh_lexer()
        text = "/* comment */"
        result = pairs(lexer, text)
        joined = "".join(v for _, v in result)
        assert joined == text

    def test_comment_line(self):
        lexer = fresh_lexer()
        text = "-- foo\n"
        result = pairs(lexer, text)
        joined = "".join(v for _, v in result)
        assert joined == text

    def test_reconstruction_of_full_text(self):
        lexer = fresh_lexer()
        text = "SELECT a, b FROM t WHERE x = 1;"
        result = pairs(lexer, text)
        assert "".join(v for _, v in result) == text

    def test_keyword_classification_via_process_as_keyword(self):
        lexer = fresh_lexer()
        result = pairs(lexer, "CREATE TABLE")
        joined = [v for _, v in result]
        assert "CREATE" in joined
        assert "TABLE" in joined

    def test_lazy_generator(self):
        lexer = fresh_lexer()
        gen = lexer.get_tokens("SELECT 1")
        # It is a generator, not a list
        assert not isinstance(gen, list)
        first = next(gen)
        assert first == (tokens.Keyword.DML, "SELECT")


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_tokenize_uses_default_instance(self):
        result = list(tokenize("SELECT"))
        assert result == [(tokens.Keyword.DML, "SELECT")]

    def test_tokenize_matches_default_instance(self):
        via_tokenize = list(tokenize("SELECT 1"))
        via_instance = list(Lexer.get_default_instance().get_tokens("SELECT 1"))
        assert via_tokenize == via_instance

    def test_tokenize_with_encoding(self):
        result = list(tokenize("SELECT".encode("latin-1"), "latin-1"))
        assert result == [(tokens.Keyword.DML, "SELECT")]

    def test_tokenize_returns_generator(self):
        gen = tokenize("SELECT")
        assert not isinstance(gen, list)

    def test_tokenize_invalid_type_raises(self):
        with pytest.raises(TypeError):
            list(tokenize(42))

    def test_tokenize_reconstructs_text(self):
        sql = "SELECT * FROM t;"
        assert "".join(v for _, v in tokenize(sql)) == sql
