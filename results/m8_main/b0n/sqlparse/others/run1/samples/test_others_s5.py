import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.others import (
    StripCommentsFilter,
    StripWhitespaceFilter,
    SpacesAroundOperatorsFilter,
    StripTrailingSemicolonFilter,
    SerializerUnicode,
)
from sqlparse.lexer import Lexer


def parse_stmt(text):
    """Parse a SQL string into a list of statements (single statement)."""
    from sqlparse.engine import FilterStack
    stack = FilterStack()
    stmts = list(stack.run(text))
    return stmts[0]


def make_tokenlist(*tokens):
    tlist = sql.TokenList(tokens)
    return tlist


# ---------------------------------------------------------------------------
# StripCommentsFilter
# ---------------------------------------------------------------------------

class TestStripCommentsFilter:

    def test_process_returns_stmt(self):
        stmt = parse_stmt("SELECT 1")
        result = StripCommentsFilter().process(stmt)
        assert result is stmt

    def test_single_line_comment_removed(self):
        stmt = parse_stmt("SELECT 1 -- comment\nFROM foo")
        stmt = StripCommentsFilter().process(stmt)
        assert 'comment' not in str(stmt)

    def test_multiline_comment_removed(self):
        stmt = parse_stmt("SELECT /* hidden */ 1")
        stmt = StripCommentsFilter().process(stmt)
        assert 'hidden' not in str(stmt)
        assert '1' in str(stmt)

    def test_comment_at_end_removed(self):
        stmt = parse_stmt("SELECT 1 -- trailing")
        stmt = StripCommentsFilter().process(stmt)
        assert 'trailing' not in str(stmt)

    def test_comment_preserves_newline(self):
        # Comment with trailing newline should be replaced by newline token
        stmt = parse_stmt("SELECT 1 -- comment\nFROM foo")
        stmt = StripCommentsFilter().process(stmt)
        result = str(stmt)
        assert '\n' in result
        values = [t.value for t in stmt.tokens]
        assert any(v == '\n' for v in values if isinstance(v, str))

    def test_comment_between_tokens_replaced_by_space(self):
        stmt = parse_stmt("SELECT 1 /* c */ FROM foo")
        stmt = StripCommentsFilter().process(stmt)
        assert str(stmt) == "SELECT 1 FROM foo"

    def test_sql_hint_multiline_preserved(self):
        stmt = parse_stmt("SELECT /*+ INDEX(x) */ 1")
        before = str(stmt)
        stmt = StripCommentsFilter().process(stmt)
        assert '/*+ INDEX(x) */' in str(stmt)
        assert str(stmt) == before

    def test_sql_hint_single_preserved(self):
        stmt = parse_stmt("SELECT 1 --+ hint\n")
        stmt = StripCommentsFilter().process(stmt)
        assert '--+ hint' in str(stmt)

    def test_multiple_comments(self):
        stmt = parse_stmt("SELECT /* a */ 1 /* b */ FROM foo")
        stmt = StripCommentsFilter().process(stmt)
        result = str(stmt)
        assert '/* a */' not in result
        assert '/* b */' not in result
        assert result == "SELECT 1 FROM foo"

    def test_nested_in_parenthesis(self):
        stmt = parse_stmt("SELECT (/* inner */ 1) FROM foo")
        stmt = StripCommentsFilter().process(stmt)
        assert 'inner' not in str(stmt)

    def test_no_comments_untouched(self):
        stmt = parse_stmt("SELECT 1 FROM foo")
        before = str(stmt)
        stmt = StripCommentsFilter().process(stmt)
        assert str(stmt) == before

    def test_comment_after_open_paren_no_leading_ws(self):
        # prev is '(' -> don't insert whitespace, just remove
        from sqlparse import sql as s
        stmt = s.Statement([s.Token(T.Punctuation, '('),
                            s.Comment('/* c */')])
        StripCommentsFilter._process(stmt)
        assert [t.value for t in stmt.tokens] == ['(']

    def test_comment_with_newlines_replaced(self):
        stmt = parse_stmt("SELECT 1 /* multi\nline */ 2")
        stmt = StripCommentsFilter().process(stmt)
        # line break preserved
        assert '2' in str(stmt)


# ---------------------------------------------------------------------------
# StripWhitespaceFilter
# ---------------------------------------------------------------------------

class TestStripWhitespaceFilter:

    def test_process_returns_stmt(self):
        stmt = parse_stmt("SELECT 1")
        result = StripWhitespaceFilter().process(stmt)
        assert result is stmt

    def test_default_leading_whitespace_removed(self):
        stmt = parse_stmt("   SELECT 1")
        StripWhitespaceFilter().process(stmt)
        assert str(stmt) == "SELECT 1"

    def test_multiple_inner_whitespace_collapsed(self):
        stmt = parse_stmt("SELECT    1   FROM   foo")
        StripWhitespaceFilter().process(stmt)
        assert str(stmt) == "SELECT 1 FROM foo"

    def test_trailing_whitespace_removed(self):
        stmt = parse_stmt("SELECT 1   ")
        StripWhitespaceFilter().process(stmt)
        assert str(stmt) == "SELECT 1"

    def test_no_extra_leading_space_in_default(self):
        tokens = [sql.Token(T.Whitespace, '   '),
                  sql.Token(T.Keyword, 'SELECT')]
        tlist = make_tokenlist(*tokens)
        StripWhitespaceFilter._stripws_default(tlist)
        assert tlist.tokens[0].value == ''
        assert tlist.tokens[1].value == 'SELECT'

    def test_consecutive_whitespace_collapse(self):
        tokens = [sql.Token(T.Keyword, 'a'),
                  sql.Token(T.Whitespace, '  '),
                  sql.Token(T.Whitespace, '  '),
                  sql.Token(T.Keyword, 'b')]
        tlist = make_tokenlist(*tokens)
        StripWhitespaceFilter._stripws_default(tlist)
        assert [t.value for t in tlist.tokens] == ['a', ' ', '', 'b']

    def test_identifierlist_newline_before_comma_removed(self):
        # Issue 140: newline before comma
        stmt = parse_stmt("SELECT a\n, b")
        StripWhitespaceFilter().process(stmt)
        result = str(stmt)
        assert 'a\n,' not in result

    def test_parenthesis_strip(self):
        stmt = parse_stmt("SELECT (  1  )")
        StripWhitespaceFilter().process(stmt)
        assert str(stmt) == "SELECT (1)"

    def test_parenthesis_strip_nested_group(self):
        stmt = parse_stmt("SELECT (   ( 1 )   )")
        StripWhitespaceFilter().process(stmt)
        assert str(stmt) == "SELECT ((1))"

    def test_depth_zero_trailing_whitespace_popped(self):
        stmt = parse_stmt("SELECT 1")
        # append whitespace token at end
        stmt.tokens.append(sql.Token(T.Whitespace, ' '))
        StripWhitespaceFilter().process(stmt, depth=0)
        assert not stmt.tokens[-1].is_whitespace

    def test_depth_nonzero_trailing_whitespace_kept(self):
        stmt = parse_stmt("SELECT 1")
        stmt.tokens.append(sql.Token(T.Whitespace, ' '))
        last_before = len(stmt.tokens)
        StripWhitespaceFilter().process(stmt, depth=1)
        # last token is whitespace with '' value but not popped
        assert len(stmt.tokens) == last_before
        assert stmt.tokens[-1].is_whitespace

    def test_empty_stmt(self):
        stmt = parse_stmt("")
        result = StripWhitespaceFilter().process(stmt)
        assert result is stmt


# ---------------------------------------------------------------------------
# SpacesAroundOperatorsFilter
# ---------------------------------------------------------------------------

class TestSpacesAroundOperatorsFilter:

    def test_process_returns_stmt(self):
        stmt = parse_stmt("SELECT 1")
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert result is stmt

    def test_operator_gets_spaces(self):
        stmt = parse_stmt("SELECT 1+2")
        SpacesAroundOperatorsFilter().process(stmt)
        assert "1 + 2" in str(stmt)

    def test_comparison_gets_spaces(self):
        stmt = parse_stmt("SELECT * FROM t WHERE a=1")
        SpacesAroundOperatorsFilter().process(stmt)
        assert "a = 1" in str(stmt)

    def test_already_spaced_operator_unchanged(self):
        stmt = parse_stmt("SELECT 1 + 2")
        SpacesAroundOperatorsFilter().process(stmt)
        assert str(stmt) == "SELECT 1 + 2"

    def test_multiple_operators(self):
        stmt = parse_stmt("SELECT 1+2+3")
        SpacesAroundOperatorsFilter().process(stmt)
        assert "1 + 2 + 3" in str(stmt)

    def test_operator_in_parenthesis(self):
        stmt = parse_stmt("SELECT (1+2)")
        SpacesAroundOperatorsFilter().process(stmt)
        assert "(1 + 2)" in str(stmt)

    def test_no_operator_no_change(self):
        stmt = parse_stmt("SELECT 1 FROM foo")
        before = str(stmt)
        SpacesAroundOperatorsFilter().process(stmt)
        assert str(stmt) == before


# ---------------------------------------------------------------------------
# StripTrailingSemicolonFilter
# ---------------------------------------------------------------------------

class TestStripTrailingSemicolonFilter:

    def test_process_returns_stmt(self):
        stmt = parse_stmt("SELECT 1")
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result is stmt

    def test_trailing_semicolon_removed(self):
        stmt = parse_stmt("SELECT 1;")
        StripTrailingSemicolonFilter().process(stmt)
        assert str(stmt) == "SELECT 1"

    def test_trailing_semicolon_and_whitespace_removed(self):
        stmt = parse_stmt("SELECT 1;   ")
        StripTrailingSemicolonFilter().process(stmt)
        assert str(stmt) == "SELECT 1"

    def test_trailing_whitespace_removed(self):
        stmt = parse_stmt("SELECT 1   ")
        StripTrailingSemicolonFilter().process(stmt)
        assert str(stmt) == "SELECT 1"

    def test_no_trailing_semicolon_untouched(self):
        stmt = parse_stmt("SELECT 1")
        before = str(stmt)
        StripTrailingSemicolonFilter().process(stmt)
        assert str(stmt) == before

    def test_only_semicolon(self):
        from sqlparse import sql as s
        stmt = s.Statement([s.Token(T.Punctuation, ';')])
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result.tokens == []

    def test_empty_stmt(self):
        from sqlparse import sql as s
        stmt = s.Statement([])
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result.tokens == []


# ---------------------------------------------------------------------------
# SerializerUnicode
# ---------------------------------------------------------------------------

class TestSerializerUnicode:

    def test_simple_statement(self):
        stmt = parse_stmt("SELECT 1")
        result = SerializerUnicode().process(stmt)
        assert result == "SELECT 1"

    def test_trailing_whitespace_stripped_per_line(self):
        stmt = parse_stmt("SELECT 1   \nFROM foo  ")
        result = SerializerUnicode().process(stmt)
        assert "SELECT 1\nFROM foo" == result

    def test_returns_string(self):
        stmt = parse_stmt("SELECT 1")
        result = SerializerUnicode().process(stmt)
        assert isinstance(result, str)

    def test_multiline(self):
        stmt = parse_stmt("SELECT 1\nFROM foo")
        result = SerializerUnicode().process(stmt)
        lines = result.split('\n')
        assert lines[0] == "SELECT 1"
        assert lines[1] == "FROM foo"

    def test_newline_in_string_preserved(self):
        stmt = parse_stmt("SELECT 'a\nb'")
        result = SerializerUnicode().process(stmt)
        # The unquoted newline splitting keeps quoted newlines intact
        assert "'a\nb'" in result

    def test_empty_stmt(self):
        stmt = parse_stmt("")
        result = SerializerUnicode().process(stmt)
        assert result == ""
