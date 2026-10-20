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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(sql_text):
    from sqlparse.lexer import Lexer
    return Lexer.get_default_instance().get_tokens(sql_text)


def parse(stmt_text):
    from sqlparse.parser import Parser
    stream = tokenize(stmt_text)
    return Parser().parse(stream)[0]


# ---------------------------------------------------------------------------
# StripCommentsFilter
# ---------------------------------------------------------------------------

class TestStripCommentsFilter:
    def test_removes_single_line_comment(self):
        stmt = parse("SELECT 1 -- comment\n")
        result = StripCommentsFilter().process(stmt)
        assert '--' not in str(result)
        assert 'comment' not in str(result)

    def test_removes_multiline_comment(self):
        stmt = parse("SELECT /* comment */ 1")
        result = StripCommentsFilter().process(stmt)
        assert 'comment' not in str(result)

    def test_preserves_newlines_after_comment(self):
        stmt = parse("SELECT 1 -- comment\nFROM foo")
        result = StripCommentsFilter().process(stmt)
        assert '\n' in str(result)
        assert 'comment' not in str(result)

    def test_sql_hint_single_line_preserved(self):
        stmt = parse("SELECT /*+ HINT */ 1")
        result = StripCommentsFilter().process(stmt)
        assert 'HINT' in str(result)

    def test_sql_hint_single_line_slashslash(self):
        # T.Comment.Single.Hint
        stmt = sql.Statement([sql.Token(T.Comment.Single.Hint, '--+ hint')])
        result = StripCommentsFilter().process(stmt)
        assert 'hint' in str(result)

    def test_sql_hint_in_comment_group_preserved(self):
        comment = sql.Comment([sql.Token(T.Comment.Multiline.Hint, '/*+ x */')])
        stmt = sql.Statement([comment])
        result = StripCommentsFilter().process(stmt)
        assert 'x' in str(result)

    def test_hint_skipped_then_followed_by_comment_removed(self):
        stmt = sql.Statement([
            sql.Token(T.Comment.Single.Hint, '--+ hint'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Comment.Single, '-- normal'),
        ])
        result = StripCommentsFilter().process(stmt)
        text = str(result)
        assert 'hint' in text
        assert 'normal' not in text

    def test_comment_between_tokens_replaced_with_space(self):
        stmt = parse("SELECT a /* c */ FROM b")
        result = StripCommentsFilter().process(stmt)
        text = str(result)
        assert 'c' not in text
        # Still valid spacing between a and FROM
        assert 'a' in text and 'FROM' in text

    def test_leading_comment(self):
        stmt = parse("/* c */ SELECT 1")
        result = StripCommentsFilter().process(stmt)
        assert 'c' not in str(result)

    def test_trailing_comment_at_end(self):
        stmt = parse("SELECT 1 --")
        result = StripCommentsFilter().process(stmt)
        assert '--' not in str(result)

    def test_comment_preserves_line_break_value(self):
        stmt = sql.Statement([
            sql.Token(T.Punctuation, '('),
            sql.Token(T.Comment.Single, '-- x\r\n'),
            sql.Token(T.Punctuation, ')'),
        ])
        result = StripCommentsFilter().process(stmt)
        types = [(tk.ttype, tk.value) for tk in result.tokens]
        # newline should be inserted as Whitespace.Newline keeping \r\n
        newlines = [v for (t, v) in types if t in T.Whitespace.Newline]
        assert any('\r\n' in v for v in newlines)

    def test_comment_replaced_with_space_when_between_non_ws(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Comment.Single, '-- c'),
            sql.Token(T.Name, 'b'),
        ])
        result = StripCommentsFilter().process(stmt)
        values = [tk.value for tk in result.tokens]
        assert values == ['a', ' ', 'b']

    def test_comment_after_open_paren_no_space_inserted(self):
        stmt = sql.Statement([
            sql.Token(T.Punctuation, '('),
            sql.Token(T.Comment.Single, '-- c'),
            sql.Token(T.Name, 'b'),
        ])
        result = StripCommentsFilter().process(stmt)
        values = [tk.value for tk in result.tokens]
        assert values == ['(', 'b']

    def test_processes_sublists(self):
        # Comments inside parenthesis group should be removed too
        stmt = parse("SELECT (a /* inner */ b) FROM t")
        result = StripCommentsFilter().process(stmt)
        assert 'inner' not in str(result)

    def test_no_comment_returns_same(self):
        stmt = parse("SELECT 1")
        result = StripCommentsFilter().process(stmt)
        assert str(result) == str(stmt)


# ---------------------------------------------------------------------------
# StripWhitespaceFilter
# ---------------------------------------------------------------------------

class TestStripWhitespaceFilter:
    def test_collapse_whitespace(self):
        stmt = sql.Statement([
            sql.Token(T.Whitespace, '  '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, '   '),
            sql.Token(T.Name, 'b'),
        ])
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == 'a b'

    def test_strip_leading_whitespace(self):
        stmt = sql.Statement([
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
        ])
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == 'a'

    def test_collapse_multiple_ws_in_row(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'b'),
        ])
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == 'a b'

    def test_pop_trailing_ws_at_depth_zero(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
        ])
        result = StripWhitespaceFilter().process(stmt)
        assert [tk.value for tk in result.tokens] == ['a']

    def test_recurses_into_sublists(self):
        inner = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Punctuation, ')'),
        ])
        stmt = sql.Statement([inner])
        result = StripWhitespaceFilter().process(stmt)
        assert str(result) == '(a)'

    def test_default_staticmethod(self):
        tlist = sql.TokenList(None)
        tlist.tokens = [
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, '  '),
            sql.Token(T.Whitespace, '  '),
            sql.Token(T.Name, 'b'),
        ]
        StripWhitespaceFilter._stripws_default(tlist)
        assert [tk.value for tk in tlist.tokens] == ['', 'a', ' ', '', 'b']

    def test_identifierlist_removes_nl_before_comma(self):
        ws = sql.Token(T.Whitespace.Newline, '\n')
        ilist = sql.IdentifierList([
            sql.Token(T.Name, 'a'),
            ws,
            sql.Token(T.Punctuation, ','),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'b'),
        ])
        StripWhitespaceFilter()._stripws_identifierlist(ilist)
        # newline removed
        assert ws not in ilist.tokens
        assert str(ilist) == 'a, b'

    def test_parenthesis_strip_inner(self):
        paren = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Punctuation, ')'),
        ])
        StripWhitespaceFilter()._stripws_parenthesis(paren)
        assert [tk.value for tk in paren.tokens] == ['(', 'a', ')']

    def test_parenthesis_strip_inner_group_trailing_ws(self):
        inner = sql.IdentifierList([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
        ])
        paren = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            inner,
            sql.Token(T.Punctuation, ')'),
        ])
        StripWhitespaceFilter()._stripws_parenthesis(paren)
        assert inner.tokens[-1].value == 'a'

    def test_dispatch_uses_lowercase_method(self):
        f = StripWhitespaceFilter()
        called = {}
        # IdentifierList should dispatch to _stripws_identifierlist
        il = sql.IdentifierList([])
        original = f._stripws_identifierlist
        try:
            f._stripws_identifierlist = lambda tl: called.setdefault('idlist', True)
            f._stripws(il)
        finally:
            f._stripws_identifierlist = original
        assert called.get('idlist') is True


# ---------------------------------------------------------------------------
# SpacesAroundOperatorsFilter
# ---------------------------------------------------------------------------

class TestSpacesAroundOperatorsFilter:
    def test_adds_spaces_around_operator(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Operator, '+'),
            sql.Token(T.Name, 'b'),
        ])
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == 'a + b'

    def test_adds_spaces_around_comparison(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Comparison, '='),
            sql.Token(T.Name, 'b'),
        ])
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == 'a = b'

    def test_does_not_double_space_when_present(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Operator, '+'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Name, 'b'),
        ])
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == 'a + b'

    def test_multiple_operators(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Operator, '+'),
            sql.Token(T.Name, 'b'),
            sql.Token(T.Operator, '*'),
            sql.Token(T.Name, 'c'),
        ])
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == 'a + b * c'

    def test_operator_at_start(self):
        stmt = sql.Statement([
            sql.Token(T.Operator, '-'),
            sql.Token(T.Name, 'a'),
        ])
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == '- a'

    def test_operator_at_end(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Operator, '-'),
        ])
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == 'a -'

    def test_recurses_into_sublists(self):
        inner = sql.Parenthesis([
            sql.Token(T.Punctuation, '('),
            sql.Token(T.Name, 'a'),
            sql.Token(T.Operator, '+'),
            sql.Token(T.Name, 'b'),
            sql.Token(T.Punctuation, ')'),
        ])
        stmt = sql.Statement([inner])
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert str(result) == '( a + b )' or 'a + b' in str(result)


# ---------------------------------------------------------------------------
# StripTrailingSemicolonFilter
# ---------------------------------------------------------------------------

class TestStripTrailingSemicolonFilter:
    def test_removes_trailing_semicolon(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Punctuation, ';'),
        ])
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == 'a'

    def test_removes_trailing_ws_and_semicolon(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Punctuation, ';'),
            sql.Token(T.Whitespace, ' '),
        ])
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == 'a'

    def test_removes_multiple_semicolons(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Punctuation, ';'),
            sql.Token(T.Punctuation, ';'),
        ])
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == 'a'

    def test_no_trailing_semicolon(self):
        stmt = sql.Statement([sql.Token(T.Name, 'a')])
        result = StripTrailingSemicolonFilter().process(stmt)
        assert str(result) == 'a'

    def test_empty_statement(self):
        stmt = sql.Statement([])
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result.tokens == []


# ---------------------------------------------------------------------------
# SerializerUnicode
# ---------------------------------------------------------------------------

class TestSerializerUnicode:
    def test_basic(self):
        stmt = sql.Statement([sql.Token(T.Name, 'a')])
        assert SerializerUnicode().process(stmt) == 'a'

    def test_strip_trailing_whitespace_per_line(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace, '   '),
            sql.Token(T.Whitespace.Newline, '\n'),
            sql.Token(T.Name, 'b'),
        ])
        result = SerializerUnicode().process(stmt)
        assert result == 'a\nb'

    def test_multiple_lines(self):
        stmt = sql.Statement([
            sql.Token(T.Name, 'a'),
            sql.Token(T.Whitespace.Newline, '\n'),
            sql.Token(T.Name, 'b'),
            sql.Token(T.Whitespace.Newline, '\n'),
            sql.Token(T.Name, 'c'),
        ])
        result = SerializerUnicode().process(stmt)
        assert result == 'a\nb\nc'

    def test_quoted_newline_preserved(self):
        stmt = sql.Statement([
            sql.Token(T.String.Single, "'x\ny'"),
            sql.Token(T.Whitespace, ' '),
        ])
        result = SerializerUnicode().process(stmt)
        # newline inside string must not be split into lines
        assert 'x\ny' in result

    def test_returns_string(self):
        stmt = sql.Statement([sql.Token(T.Name, 'a')])
        assert isinstance(SerializerUnicode().process(stmt), str)
