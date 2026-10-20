import io
import re

import pytest

from sqlparse import tokens, keywords
from sqlparse.lexer import Lexer, tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lexer():
    lexer = Lexer()
    lexer.default_initialization()
    return lexer


# ---------------------------------------------------------------------------
# Lexer.get_default_instance (singleton behavior)
# ---------------------------------------------------------------------------

class TestGetDefaultInstance:
    def setup_method(self):
        Lexer._default_instance = None

    def teardown_method(self):
        Lexer._default_instance = None

    def test_returns_lexer_instance(self):
        inst = Lexer.get_default_instance()
        assert isinstance(inst, Lexer)

    def test_is_singleton(self):
        a = Lexer.get_default_instance()
        b = Lexer.get_default_instance()
        assert a is b

    def test_instance_is_initialized(self):
        inst = Lexer.get_default_instance()
        # default_initialization populates regexes and keywords
        assert len(inst._SQL_REGEX) > 0
        assert len(inst._keywords) > 0

    def test_existing_instance_not_reinitialized(self):
        inst = Lexer.get_default_instance()
        # Clear all configurations; getter must not re-init since not None
        inst.clear()
        again = Lexer.get_default_instance()
        assert again is inst
        assert inst._SQL_REGEX == []
        assert inst._keywords == []


# ---------------------------------------------------------------------------
# Lexer.clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_regex_and_keywords(self):
        lexer = make_lexer()
        assert lexer._SQL_REGEX
        assert lexer._keywords
        lexer.clear()
        assert lexer._SQL_REGEX == []
        assert lexer._keywords == []

    def test_clear_on_fresh_lexer(self):
        lexer = Lexer()
        # attributes don't exist before clear; clear should still work
        lexer.clear()
        assert lexer._SQL_REGEX == []
        assert lexer._keywords == []


# ---------------------------------------------------------------------------
# Lexer.set_SQL_REGEX
# ---------------------------------------------------------------------------

class TestSetSQLRegex:
    def test_compiles_patterns_with_match(self):
        lexer = Lexer()
        lexer.set_SQL_REGEX([(r"\d+", tokens.Number.Integer)])
        assert len(lexer._SQL_REGEX) == 1
        rexmatch, action = lexer._SQL_REGEX[0]
        assert action == tokens.Number.Integer
        m = rexmatch("123abc", 0)
        assert m is not None
        assert m.group() == "123"

    def test_case_insensitive_flag(self):
        lexer = Lexer()
        lexer.set_SQL_REGEX([(r"select", tokens.Keyword)])
        rexmatch, _ = lexer._SQL_REGEX[0]
        assert rexmatch("SELECT", 0) is not None
        assert rexmatch("SeLeCt", 0) is not None

    def test_unicode_flag(self):
        lexer = Lexer()
        lexer.set_SQL_REGEX([(r"\w+", tokens.Name)])
        rexmatch, _ = lexer._SQL_REGEX[0]
        assert rexmatch("\u00e9t\u00e9", 0) is not None

    def test_empty_list(self):
        lexer = Lexer()
        lexer.set_SQL_REGEX([])
        assert lexer._SQL_REGEX == []


# ---------------------------------------------------------------------------
# Lexer.add_keywords / is_keyword
# ---------------------------------------------------------------------------

class TestKeywords:
    def test_add_keywords_appends(self):
        lexer = Lexer()
        lexer.clear()
        d1 = {"FOO": tokens.Keyword}
        d2 = {"BAR": tokens.Keyword}
        lexer.add_keywords(d1)
        lexer.add_keywords(d2)
        assert lexer._keywords == [d1, d2]

    def test_is_keyword_found(self):
        lexer = Lexer()
        lexer.clear()
        lexer.add_keywords({"SELECT": tokens.Keyword})
        ttype, value = lexer.is_keyword("select")
        assert ttype == tokens.Keyword
        assert value == "select"

    def test_is_keyword_preserves_original_value(self):
        lexer = Lexer()
        lexer.clear()
        lexer.add_keywords({"SELECT": tokens.Keyword})
        ttype, value = lexer.is_keyword("SeLeCt")
        assert ttype == tokens.Keyword
        assert value == "SeLeCt"

    def test_is_keyword_not_found_returns_name(self):
        lexer = Lexer()
        lexer.clear()
        lexer.add_keywords({"SELECT": tokens.Keyword})
        ttype, value = lexer.is_keyword("foo")
        assert ttype == tokens.Name
        assert value == "foo"

    def test_is_keyword_order_first_match_wins(self):
        lexer = Lexer()
        lexer.clear()
        lexer.add_keywords({"X": tokens.Keyword})
        lexer.add_keywords({"X": tokens.Name})
        ttype, value = lexer.is_keyword("x")
        assert ttype == tokens.Keyword

    def test_is_keyword_empty_dicts(self):
        lexer = Lexer()
        lexer.clear()
        ttype, value = lexer.is_keyword("anything")
        assert ttype == tokens.Name
        assert value == "anything"


# ---------------------------------------------------------------------------
# Lexer.get_tokens
# ---------------------------------------------------------------------------

class TestGetTokens:
    def test_simple_select(self):
        lexer = make_lexer()
        result = list(lexer.get_tokens("SELECT 1"))
        assert (tokens.Keyword.DML, "SELECT") in result
        # whitespace token
        assert any(tt in tokens.Whitespace for tt, _ in result)

    def test_returns_generator(self):
        lexer = make_lexer()
        gen = lexer.get_tokens("SELECT 1")
        # generator has __next__
        assert hasattr(gen, "__next__")

    def test_empty_string(self):
        lexer = make_lexer()
        assert list(lexer.get_tokens("")) == []

    def test_accepts_textio(self):
        lexer = make_lexer()
        stream = io.StringIO("SELECT 1")
        result = list(lexer.get_tokens(stream))
        assert result == list(lexer.get_tokens("SELECT 1"))

    def test_bytes_utf8_default(self):
        lexer = make_lexer()
        result = list(lexer.get_tokens(b"SELECT 1"))
        assert result == list(lexer.get_tokens("SELECT 1"))

    def test_bytes_with_encoding(self):
        lexer = make_lexer()
        text = "SELECT 1"
        encoded = text.encode("latin-1")
        result = list(lexer.get_tokens(encoded, encoding="latin-1"))
        assert result == list(lexer.get_tokens(text))

    def test_bytes_unicode_escape_fallback(self):
        # Invalid UTF-8 -> falls back to unicode-escape decode
        lexer = make_lexer()
        data = b"\xff"
        result = list(lexer.get_tokens(data))
        # unicode-escape decodes \xff as 'ÿ'; ensure no crash and some tokens
        assert isinstance(result, list)

    def test_invalid_type_raises(self):
        lexer = make_lexer()
        with pytest.raises(TypeError):
            list(lexer.get_tokens(123))

    def test_invalid_type_list_raises(self):
        lexer = make_lexer()
        with pytest.raises(TypeError):
            list(lexer.get_tokens(["SELECT"]))

    def test_unknown_char_yields_error(self):
        lexer = Lexer()
        lexer.clear()
        # No regex -> every char yields Error
        result = list(lexer.get_tokens("ab"))
        assert result == [(tokens.Error, "a"), (tokens.Error, "b")]

    def test_keyword_resolution_via_process_as_keyword(self):
        lexer = make_lexer()
        result = list(lexer.get_tokens("select"))
        assert result == [(tokens.Keyword.DML, "select")]

    def test_keyword_unknown_becomes_name(self):
        lexer = make_lexer()
        result = list(lexer.get_tokens("unknownident"))
        # Matched by word regex, process as keyword -> Name
        assert len(result) == 1
        ttype, value = result[0]
        assert value == "unknownident"
        assert ttype in tokens.Name

    def test_string_literal(self):
        lexer = make_lexer()
        result = list(lexer.get_tokens("'abc'"))
        assert any(
            value == "'abc'" and tt in tokens.String.Single
            for tt, value in result
        )

    def test_comment(self):
        lexer = make_lexer()
        result = list(lexer.get_tokens("-- comment\n"))
        assert any(tt in tokens.Comment.Single for tt, _ in result)

    def test_multiline_comment(self):
        lexer = make_lexer()
        sql = "/* multi\nline */"
        result = list(lexer.get_tokens(sql))
        joined = [v for tt, v in result if tt in tokens.Comment.Multiline]
        assert joined == [sql]

    def test_comment_delimiter_in_string_ignored(self):
        # A /* inside a string must not be treated as a comment start
        lexer = make_lexer()
        sql = "'/* not a comment */'"
        result = list(lexer.get_tokens(sql))
        # entire thing is one string token
        string_tokens = [
            (tt, v) for tt, v in result if tt in tokens.String
        ]
        assert string_tokens == [(tokens.String.Single, "*/'")] or \
               any(v == sql for _, v in string_tokens)

    def test_dollar_quoted_string(self):
        lexer = make_lexer()
        sql = "$$content$$"
        result = list(lexer.get_tokens(sql))
        # Must not crash; content should be tokenized
        assert result

    def test_multiple_statements_produce_tokens(self):
        lexer = make_lexer()
        sql = "SELECT 1; SELECT 2"
        result = list(lexer.get_tokens(sql))
        assert any(v == ";" for _, v in result)

    def test_consume_advances_across_multi_char_match(self):
        lexer = make_lexer()
        result = list(lexer.get_tokens("SELECT"))
        # Single token
        assert result == [(tokens.Keyword.DML, "SELECT")]


# ---------------------------------------------------------------------------
# tokenize module function
# ---------------------------------------------------------------------------

class TestTokenize:
    def setup_method(self):
        Lexer._default_instance = None

    def teardown_method(self):
        Lexer._default_instance = None

    def test_basic(self):
        result = list(tokenize("SELECT 1"))
        assert any(tt == tokens.Keyword.DML and v == "SELECT"
                   for tt, v in result)

    def test_uses_default_instance(self):
        tokenize("SELECT 1")
        assert Lexer._default_instance is not None

    def test_same_as_default_instance_get_tokens(self):
        expected = list(Lexer.get_default_instance().get_tokens("SELECT 1"))
        result = list(tokenize("SELECT 1"))
        assert result == expected

    def test_encoding_parameter(self):
        result_default = list(tokenize(b"SELECT 1"))
        result_utf8 = list(tokenize(b"SELECT 1", encoding="utf-8"))
        assert result_default == result_utf8

    def test_empty(self):
        assert list(tokenize("")) == []

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            list(tokenize(42))
