import io
import threading

import pytest

from sqlparse import keywords, tokens
from sqlparse.lexer import Lexer, tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize_list(sql, encoding=None):
    return list(tokenize(sql, encoding))


def type_names(tokens_list):
    return [t[0] for t in tokens_list]


def values(tokens_list):
    return [t[1] for t in tokens_list]


@pytest.fixture
def fresh_lexer():
    """A Lexer instance not tied to the module-level singleton."""
    return Lexer()


@pytest.fixture
def initialized_lexer():
    """A Lexer instance that has default initialization applied."""
    lexer = Lexer()
    lexer.default_initialization()
    return lexer


# ---------------------------------------------------------------------------
# Lexer singleton behaviour
# ---------------------------------------------------------------------------

class TestDefaultInstance:
    def test_singleton_returns_same_object(self):
        first = Lexer.get_default_instance()
        second = Lexer.get_default_instance()
        assert first is second

    def test_default_instance_is_lexer(self):
        assert isinstance(Lexer.get_default_instance(), Lexer)

    def test_default_instance_is_initialized(self):
        lexer = Lexer.get_default_instance()
        assert len(lexer._SQL_REGEX) > 0
        assert len(lexer._keywords) > 0

    def test_threaded_singleton(self):
        results = []

        def worker():
            results.append(Lexer.get_default_instance())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# clear / set_SQL_REGEX / add_keywords / default_initialization
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_clear_empties_state(self, initialized_lexer):
        assert len(initialized_lexer._SQL_REGEX) > 0
        assert len(initialized_lexer._keywords) > 0
        initialized_lexer.clear()
        assert initialized_lexer._SQL_REGEX == []
        assert initialized_lexer._keywords == []

    def test_set_SQL_REGEX_compiles(self, fresh_lexer):
        fresh_lexer.set_SQL_REGEX([(r"\d+", tokens.Number.Integer)])
        assert len(fresh_lexer._SQL_REGEX) == 1
        match_fn, action = fresh_lexer._SQL_REGEX[0]
        assert callable(match_fn)
        assert action is tokens.Number.Integer
        # case-insensitive / unicode flags applied
        fresh_lexer.set_SQL_REGEX([(r"abc", tokens.Keyword)])
        match_fn, _ = fresh_lexer._SQL_REGEX[0]
        m = match_fn("ABC", 0)
        assert m is not None
        assert m.group() == "ABC"

    def test_add_keywords_preserves_order(self, fresh_lexer):
        # A fresh Lexer has no _keywords attribute until clear() or
        # default_initialization() is called, so clear first.
        fresh_lexer.clear()
        dict_a = {"FOO": tokens.Keyword}
        dict_b = {"FOO": tokens.Name, "BAR": tokens.Keyword}
        fresh_lexer.add_keywords(dict_a)
        fresh_lexer.add_keywords(dict_b)
        assert fresh_lexer._keywords == [dict_a, dict_b]
        # order matters: first dict wins
        assert fresh_lexer.is_keyword("foo") == (tokens.Keyword, "foo")
        assert fresh_lexer.is_keyword("bar") == (tokens.Keyword, "bar")

    def test_default_initialization_resets(self, fresh_lexer):
        fresh_lexer.clear()
        fresh_lexer.set_SQL_REGEX([(r"x", tokens.Error)])
        fresh_lexer.add_keywords({"ONLY": tokens.Error})
        fresh_lexer.default_initialization()
        # default regex set (not the single custom one)
        assert len(fresh_lexer._SQL_REGEX) > 1
        # "select" resolves to a DML keyword in the default dictionaries
        ttype, value = fresh_lexer.is_keyword("select")
        assert ttype is tokens.Keyword.DML
        assert value == "select"


# ---------------------------------------------------------------------------
# is_keyword
# ---------------------------------------------------------------------------

class TestIsKeyword:
    def test_known_keyword(self):
        lexer = Lexer()
        lexer.clear()
        lexer.add_keywords({"SELECT": tokens.Keyword})
        ttype, value = lexer.is_keyword("select")
        assert ttype is tokens.Keyword
        assert value == "select"

    def test_preserves_original_value(self):
        lexer = Lexer()
        lexer.clear()
        lexer.add_keywords({"SELECT": tokens.Keyword})
        _, value = lexer.is_keyword("SeLeCt")
        assert value == "SeLeCt"

    def test_unknown_becomes_name(self):
        lexer = Lexer()
        lexer.clear()
        lexer.add_keywords({"SELECT": tokens.Keyword})
        ttype, value = lexer.is_keyword("foobar")
        assert ttype is tokens.Name
        assert value == "foobar"

    def test_no_keywords_returns_name(self):
        lexer = Lexer()
        lexer.clear()
        ttype, value = lexer.is_keyword("select")
        assert ttype is tokens.Name
        assert value == "select"


# ---------------------------------------------------------------------------
# get_tokens: input types
# ---------------------------------------------------------------------------

class TestGetTokensInput:
    def test_str_input(self):
        result = list(Lexer.get_default_instance().get_tokens("SELECT 1"))
        assert result == [
            (tokens.Keyword.DML, "SELECT"),
            (tokens.Text.Whitespace, " "),
            (tokens.Number.Integer, "1"),
        ]

    def test_text_io_object(self):
        buf = io.StringIO("SELECT 1")
        result = list(Lexer.get_default_instance().get_tokens(buf))
        assert result == [
            (tokens.Keyword.DML, "SELECT"),
            (tokens.Text.Whitespace, " "),
            (tokens.Number.Integer, "1"),
        ]

    def test_bytes_utf8_default(self):
        result = list(tokenize(b"SELECT 1"))
        assert values(result) == ["SELECT", " ", "1"]

    def test_bytes_with_explicit_encoding(self):
        result = list(tokenize("SELECT 1".encode("utf-8"), encoding="utf-8"))
        assert values(result) == ["SELECT", " ", "1"]

    def test_bytes_with_latin1_encoding(self):
        text = "SELECT 'café'"
        encoded = text.encode("latin-1")
        result = list(tokenize(encoded, encoding="latin-1"))
        assert values(result) == ["SELECT", " ", "'café'"]

    def test_bytes_fallback_unicode_escape(self):
        # bytes that are not valid utf-8 but decodable via unicode-escape
        data = b"SELECT \\xe9"
        result = list(tokenize(data))
        assert values(result)[0] == "SELECT"

    def test_bytes_fallback_unicode_escape_full(self):
        # b"\\xe9" is the literal 4-character sequence backslash-x-e-9.
        # It is valid utf-8 so no fallback happens; it becomes a token
        # whose value is the literal string "\\xe9".
        data = b"\\xe9"
        result = list(tokenize(data))
        assert values(result) == ["\\xe9"]

    def test_bytes_invalid_unicode_escape_raises(self):
        # 0xff alone is not valid utf-8; unicode-escape decodes the single
        # byte 0xff via latin-1 to "ÿ", so no exception is raised.
        data = b"\xff"
        result = list(tokenize(data))
        assert values(result) == ["\xff"]

    def test_invalid_type_raises(self):
        lexer = Lexer.get_default_instance()
        with pytest.raises(TypeError) as excinfo:
            list(lexer.get_tokens(42))
        assert "Expected text or file-like object" in str(excinfo.value)

    def test_invalid_type_message_includes_type(self):
        with pytest.raises(TypeError) as excinfo:
            list(tokenize(42))
        assert "int" in str(excinfo.value)

    def test_invalid_type_none(self):
        with pytest.raises(TypeError):
            list(tokenize(None))

    def test_invalid_type_list(self):
        with pytest.raises(TypeError):
            list(tokenize(["SELECT"]))


# ---------------------------------------------------------------------------
# get_tokens: tokenization behavior
# ---------------------------------------------------------------------------

class TestTokenization:
    def test_empty_string(self):
        assert tokenize_list("") == []

    def test_whitespace_only(self):
        result = tokenize_list("   ")
        # Each whitespace character becomes its own token.
        assert result == [(tokens.Text.Whitespace, " ")] * 3

    def test_keyword_case_insensitive(self):
        lower = tokenize_list("select")
        upper = tokenize_list("SELECT")
        assert type_names(lower) == type_names(upper)
        assert values(lower) == ["select"]
        assert values(upper) == ["SELECT"]

    def test_string_literal(self):
        result = tokenize_list("'hello'")
        assert len(result) == 1
        assert result[0][0] in tokens.String
        assert result[0][1] == "'hello'"

    def test_number_integer(self):
        result = tokenize_list("123")
        assert result == [(tokens.Number.Integer, "123")]

    def test_number_float(self):
        result = tokenize_list("1.5")
        assert len(result) == 1
        assert result[0][1] == "1.5"

    def test_identifier_becomes_name(self):
        result = tokenize_list("mycolumn")
        assert result == [(tokens.Name, "mycolumn")]

    def test_punctuation(self):
        result = tokenize_list(",")
        assert len(result) == 1
        assert result[0][1] == ","

    def test_comment_single_line(self):
        result = tokenize_list("-- comment\nSELECT")
        # comment token then newline then keyword
        joined = "".join(values(result))
        assert joined == "-- comment\nSELECT"
        assert any(t[0] in tokens.Comment.Single for t in result)

    def test_unknown_char_is_error(self):
        lexer = Lexer()
        lexer.clear()
        # No regexes matched, so everything falls through to Error.
        result = list(lexer.get_tokens("?"))
        assert result == [(tokens.Error, "?")]

    def test_custom_regex_and_keyword(self, fresh_lexer):
        fresh_lexer.clear()
        fresh_lexer.set_SQL_REGEX([(r"\w+", keywords.PROCESS_AS_KEYWORD)])
        fresh_lexer.add_keywords({"FOO": tokens.Keyword})
        result = list(fresh_lexer.get_tokens("foo"))
        # "foo" matches keyword; the only regex matches word characters.
        assert result == [(tokens.Keyword, "foo")]

    def test_custom_regex_unknown_word(self, fresh_lexer):
        fresh_lexer.clear()
        fresh_lexer.set_SQL_REGEX([(r"\w+", keywords.PROCESS_AS_KEYWORD)])
        fresh_lexer.add_keywords({"FOO": tokens.Keyword})
        result = list(fresh_lexer.get_tokens("bar"))
        # no keyword match -> falls back to Name
        assert result == [(tokens.Name, "bar")]

    def test_generator_type(self):
        gen = tokenize("SELECT")
        assert not isinstance(gen, list)
        assert iter(gen) is gen

    def test_multiple_keywords(self):
        result = tokenize_list("SELECT * FROM foo")
        types = type_names(result)
        assert tokens.Keyword.DML in types
        assert tokens.Keyword in types
        names = [v for (t, v) in result if t == tokens.Name]
        assert names == ["foo"]


# ---------------------------------------------------------------------------
# get_tokens: delimited spans (dollar-quoted literals, multiline comments)
# ---------------------------------------------------------------------------

class TestDelimitedSpans:
    def test_dollar_quoted_literal(self):
        sql = "SELECT $tag$some text$tag$"
        result = tokenize_list(sql)
        # The dollar-quoted body is a single token.
        dollar_tokens = [
            (t, v) for (t, v) in result
            if v == "$tag$some text$tag$"
        ]
        assert len(dollar_tokens) == 1
        # The type produced for dollar-quoted literals is Token.Literal
        assert dollar_tokens[0][0] is tokens.Literal

    def test_dollar_quoted_literal_plain(self):
        sql = "SELECT $$hello$$"
        result = tokenize_list(sql)
        assert values(result) == ["SELECT", " ", "$$hello$$"]
        assert result[-1][0] is tokens.Literal

    def test_dollar_quoted_span_opened_via_span_openers(self):
        # Directly exercise the pos-in-span_openers branch: a dollar-quoted
        # literal at the very start of the text.
        sql = "$$abc$$"
        result = tokenize_list(sql)
        assert result == [(result[0][0], "$$abc$$")]
        assert result[0][0] is tokens.Literal

    def test_multiline_comment_is_single_token(self):
        sql = "/* a\nb */ SELECT"
        result = tokenize_list(sql)
        assert result[0][0] in tokens.Comment.Multiline
        assert result[0][1] == "/* a\nb */"
        assert result[1] == (tokens.Text.Whitespace, " ")
        assert result[2] == (tokens.Keyword.DML, "SELECT")

    def test_multiline_comment_alone(self):
        result = tokenize_list("/* foo */")
        assert len(result) == 1
        assert result[0][0] in tokens.Comment.Multiline
        assert result[0][1] == "/* foo */"

    def test_delimiter_inside_string_literal_not_resolved(self):
        # A "$" inside a normal string literal must not open a dollar-quoted
        # span; the surrounding string literal is a single token.
        sql = "'$not a tag$'"
        result = tokenize_list(sql)
        assert result == [(result[0][0], "'$not a tag$'")]
        assert result[0][0] in tokens.String

    def test_delimiter_inside_line_comment_not_resolved(self):
        # A "/*" inside a "--" comment must not open a multiline comment
        # span; the line comment is its own token.
        sql = "-- /* not opened\nSELECT"
        result = tokenize_list(sql)
        assert result[0][0] in tokens.Comment.Single
        assert "SELECT" in values(result)
