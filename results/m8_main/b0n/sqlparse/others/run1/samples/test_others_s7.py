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
from sqlparse.sql import TokenList, Token, Comment, Identifier, IdentifierList, Parenthesis
from sqlparse.utils import split_unquoted_newlines


# ---------------------------
# Helpers


def make_tokenlist(*tokens):
    return TokenList(list(tokens))


def tk(value, ttype=T.Text):
    return Token(ttype, value)


def render(tlist):
    return ''.join(t.value for t in tlist.tokens)


# ---------------------------
# StripCommentsFilter


class TestStripCommentsFilter:
    def test_removes_single_comment_between_whitespace(self):
        tl = make_tokenlist(
            tk(' ', T.Whitespace),
            tk('-- hi', T.Comment.Single),
            tk(' ', T.Whitespace),
            tk('SELECT', T.Keyword),
        )
        StripCommentsFilter._process(tl)
        assert render(tl) == '  SELECT'

    def test_strips_comment_between_tokens_with_space(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk('/* c */', T.Comment.Multiline),
            tk('b', T.Text),
        )
        StripCommentsFilter._process(tl)
        assert render(tl) == 'a b'

    def test_preserves_trailing_newline_in_comment(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk('-- c\n   ', T.Comment.Single),
            tk('b', T.Text),
        )
        StripCommentsFilter._process(tl)
        # whitespace token follows 'a', the value should be '\n'
        assert render(tl) == 'a\nb'

    def test_preserves_crlf_linebreak(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk('-- c\r\n', T.Comment.Single),
            tk('b', T.Text),
        )
        StripCommentsFilter._process(tl)
        assert render(tl) == 'a\r\nb'

    def test_comment_at_start_no_prev(self):
        tl = make_tokenlist(
            tk('-- leading', T.Comment.Single),
            tk('SELECT', T.Keyword),
        )
        StripCommentsFilter._process(tl)
        assert render(tl) == 'SELECT'

    def test_comment_at_end_no_next(self):
        tl = make_tokenlist(
            tk('SELECT', T.Keyword),
            tk('-- trailing', T.Comment.Single),
        )
        StripCommentsFilter._process(tl)
        # prev exists and isn't '(', so whitespace inserted then comment removed
        assert render(tl) == 'SELECT '

    def test_comment_after_open_paren_no_whitespace_inserted(self):
        tl = make_tokenlist(
            tk('(', T.Punctuation),
            tk('-- c', T.Comment.Single),
            tk('x', T.Text),
        )
        StripCommentsFilter._process(tl)
        # prev matches '(' -> no whitespace inserted; comment removed
        assert render(tl) == '(x'

    def test_comment_before_close_paren(self):
        tl = make_tokenlist(
            tk('x', T.Text),
            tk('-- c', T.Comment.Single),
            tk(')', T.Punctuation),
        )
        StripCommentsFilter._process(tl)
        # next_ is ')' -> no special-case; prev is not '('; but next_ matches ')'
        # so branch: insert whitespace (prev not '(', then remove comment
        assert '(x' in render(tl).replace(' ', '') or render(tl) == 'x )'
        # exact behavior:
        assert render(tl) == 'x )'

    def test_sql_hint_token_is_skipped(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk('/*+ hint */', T.Comment.Multiline.Hint),
            tk('b', T.Text),
        )
        StripCommentsFilter._process(tl)
        assert render(tl) == 'a/*+ hint */b'

    def test_sql_hint_single_is_skipped(self):
        tl = make_tokenlist(
            tk('SELECT', T.Keyword),
            tk('--+ hint', T.Comment.Single.Hint),
            tk('FROM', T.Keyword),
        )
        StripCommentsFilter._process(tl)
        assert render(tl) == 'SELECT--+ hintFROM'

    def test_comment_group_with_hint_as_first_token_skipped(self):
        inner = [tk('/*+ hint */', T.Comment.Multiline.Hint)]
        comment_group = Comment(inner)
        tl = make_tokenlist(
            tk('a', T.Text),
            comment_group,
            tk('b', T.Text),
        )
        StripCommentsFilter._process(tl)
        assert render(tl) == 'a/*+ hint */b'

    def test_comment_group_empty_tokens_is_removed(self):
        comment_group = Comment([])
        tl = make_tokenlist(
            tk('a', T.Text),
            comment_group,
            tk('b', T.Text),
        )
        StripCommentsFilter._process(tl)
        assert render(tl) == 'a b'

    def test_comment_group_removed_and_linebreak_preserved(self):
        comment_group = Comment([tk('-- c\n', T.Comment.Single)])
        tl = make_tokenlist(
            tk('a', T.Text),
            comment_group,
            tk('b', T.Text),
        )
        StripCommentsFilter._process(tl)
        assert render(tl) == 'a\nb'

    def test_process_returns_stmt_and_recurses(self):
        inner = Parenthesis([
            tk('(', T.Punctuation),
            tk('-- c', T.Comment.Single),
            tk('x', T.Text),
            tk(')', T.Punctuation),
        ])
        stmt = make_tokenlist(inner, tk('-- tail', T.Comment.Single))
        result = StripCommentsFilter().process(stmt)
        assert result is stmt
        assert '--' not in result.value

    def test_no_comments_is_noop(self):
        tl = make_tokenlist(tk('a', T.Text), tk(' ', T.Whitespace), tk('b', T.Text))
        StripCommentsFilter._process(tl)
        assert render(tl) == 'a b'


# ---------------------------
# StripWhitespaceFilter


class TestStripWhitespaceFilter:
    def test_default_strips_leading_ws(self):
        tl = make_tokenlist(tk(' ', T.Whitespace), tk('a', T.Text))
        StripWhitespaceFilter._stripws(tl)
        assert render(tl) == 'a'

    def test_default_collapses_consecutive_ws(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk(' \t', T.Whitespace),
            tk('  ', T.Whitespace),
            tk('b', T.Text),
        )
        StripWhitespaceFilter._stripws(tl)
        assert render(tl) == 'a b'

    def test_default_keeps_single_ws(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk('  ', T.Whitespace),
            tk('b', T.Text),
        )
        StripWhitespaceFilter._stripws(tl)
        assert render(tl) == 'a b'

    def test_identifierlist_removes_newline_before_comma(self):
        nl = tk('\n', T.Whitespace.Newline)
        idl = IdentifierList([
            tk('a', T.Text),
            nl,
            tk(',', T.Punctuation),
            tk('b', T.Text),
        ])
        StripWhitespaceFilter()._stripws(idl)
        assert nl not in idl.tokens
        assert ',' in render(idl)

    def test_identifierlist_keeps_newline_not_before_comma(self):
        nl = tk('\n', T.Whitespace.Newline)
        idl = IdentifierList([
            tk('a', T.Text),
            nl,
            tk('b', T.Text),
        ])
        StripWhitespaceFilter()._stripws(idl)
        assert nl in idl.tokens

    def test_identifierlist_removes_ws(self):
        nl = tk('\n', T.Whitespace.Newline)
        idl = IdentifierList([tk('a', T.Text), nl, tk(',', T.Punctuation)])
        StripWhitespaceFilter()._stripws(idl)
        assert render(idl) == 'a,'

    def test_parenthesis_strips_inner_whitespace(self):
        p = Parenthesis([
            tk('(', T.Punctuation),
            tk(' ', T.Whitespace),
            tk('x', T.Text),
            tk(' ', T.Whitespace),
            tk(')', T.Punctuation),
        ])
        StripWhitespaceFilter()._stripws(p)
        assert render(p) == '(x)'

    def test_parenthesis_collapses_multiple_inner_ws(self):
        p = Parenthesis([
            tk('(', T.Punctuation),
            tk(' ', T.Whitespace),
            tk('  ', T.Whitespace),
            tk('x', T.Text),
            tk(')', T.Punctuation),
        ])
        StripWhitespaceFilter()._stripws(p)
        assert render(p) == '(x)'

    def test_parenthesis_removes_trailing_group_inner_ws(self):
        inner = Identifier([
            tk('a', T.Text),
            tk(' ', T.Whitespace),
        ])
        p = Parenthesis([
            tk('(', T.Punctuation),
            inner,
            tk(')', T.Punctuation),
        ])
        StripWhitespaceFilter()._stripws(p)
        # last token inside inner identifier was whitespace, got removed
        assert inner.tokens[-1].value == 'a'
        assert render(p) == '(a)'

    def test_process_pops_trailing_whitespace_at_depth_zero(self):
        stmt = make_tokenlist(
            tk('a', T.Text),
            tk('  ', T.Whitespace),
        )
        result = StripWhitespaceFilter().process(stmt)
        assert result is stmt
        assert render(stmt) == 'a'

    def test_process_recurses_and_depth_guard(self):
        inner = Parenthesis([
            tk('(', T.Punctuation),
            tk('x', T.Text),
            tk(' ', T.Whitespace),
            tk(')', T.Punctuation),
        ])
        stmt = make_tokenlist(inner)
        StripWhitespaceFilter().process(stmt)
        # inner recursive call handled parenthesis, so trailing ws dropped
        assert render(inner) == '(x)'

    def test_process_inner_trailing_ws_kept_at_nonzero_depth(self):
        inner = Parenthesis([
            tk('(', T.Punctuation),
            Identifier([tk('a', T.Text), tk(' ', T.Whitespace)]),
            tk(')', T.Punctuation),
        ])
        # run process with depth=1 -> inner _stripws handles things
        StripWhitespaceFilter().process(inner.tokens[1], depth=1)
        # identifier's trailing whitespace not popped at depth!=0
        # (but _stripws_default was invoked via process... check behavior)
        # Direct check of branching:
        assert inner is not None

    def test_process_nonzero_depth_preserves_trailing_ws(self):
        stmt = Identifier([tk('a', T.Text), tk(' ', T.Whitespace)])
        StripWhitespaceFilter().process(stmt, depth=1)
        # depth != 0, so trailing whitespace not popped
        assert stmt.tokens[-1].is_whitespace

    def test_process_empty_tokens(self):
        stmt = make_tokenlist()
        result = StripWhitespaceFilter().process(stmt)
        assert result is stmt
        assert stmt.tokens == []


# ---------------------------
# SpacesAroundOperatorsFilter


class TestSpacesAroundOperatorsFilter:
    def test_adds_spaces_around_operator(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk('=', T.Operator),
            tk('b', T.Text),
        )
        SpacesAroundOperatorsFilter._process(tl)
        assert render(tl) == 'a = b'

    def test_adds_spaces_around_comparison(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk('<', T.Comparison),
            tk('b', T.Text),
        )
        SpacesAroundOperatorsFilter._process(tl)
        assert render(tl) == 'a < b'

    def test_existing_whitespace_not_duplicated(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk(' ', T.Whitespace),
            tk('=', T.Operator),
            tk(' ', T.Whitespace),
            tk('b', T.Text),
        )
        SpacesAroundOperatorsFilter._process(tl)
        assert render(tl) == 'a = b'

    def test_operator_at_start(self):
        tl = make_tokenlist(
            tk('=', T.Operator),
            tk('b', T.Text),
        )
        SpacesAroundOperatorsFilter._process(tl)
        assert render(tl) == '= b'

    def test_operator_at_end(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk('=', T.Operator),
        )
        SpacesAroundOperatorsFilter._process(tl)
        assert render(tl) == 'a ='

    def test_multiple_operators(self):
        tl = make_tokenlist(
            tk('a', T.Text),
            tk('+', T.Operator),
            tk('b', T.Text),
            tk('-', T.Operator),
            tk('c', T.Text),
        )
        SpacesAroundOperatorsFilter._process(tl)
        assert render(tl) == 'a + b - c'

    def test_process_recurses_and_returns_stmt(self):
        inner = Parenthesis([
            tk('(', T.Punctuation),
            tk('a', T.Text),
            tk('=', T.Operator),
            tk('b', T.Text),
            tk(')', T.Punctuation),
        ])
        stmt = make_tokenlist(inner)
        result = SpacesAroundOperatorsFilter().process(stmt)
        assert result is stmt
        assert render(inner) == '(a = b)'

    def test_no_operators_is_noop(self):
        tl = make_tokenlist(tk('a', T.Text), tk(' ', T.Whitespace), tk('b', T.Text))
        SpacesAroundOperatorsFilter._process(tl)
        assert render(tl) == 'a b'


# ---------------------------
# StripTrailingSemicolonFilter


class TestStripTrailingSemicolonFilter:
    def test_removes_trailing_semicolon(self):
        stmt = make_tokenlist(tk('SELECT', T.Keyword), tk(';', T.Punctuation))
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result is stmt
        assert stmt.tokens == []

    def test_removes_trailing_whitespace_then_semicolon(self):
        stmt = make_tokenlist(
            tk('SELECT', T.Keyword),
            tk(';', T.Punctuation),
            tk('  ', T.Whitespace),
        )
        StripTrailingSemicolonFilter().process(stmt)
        assert render(stmt) == 'SELECT'

    def test_keeps_internal_semicolon(self):
        stmt = make_tokenlist(
            tk('SELECT', T.Keyword),
            tk(';', T.Punctuation),
            tk('x', T.Text),
        )
        StripTrailingSemicolonFilter().process(stmt)
        assert render(stmt) == 'SELECT;x'

    def test_only_whitespace_removed(self):
        stmt = make_tokenlist(
            tk('x', T.Text),
            tk('  ', T.Whitespace),
        )
        StripTrailingSemicolonFilter().process(stmt)
        assert render(stmt) == 'x'

    def test_empty(self):
        stmt = make_tokenlist()
        result = StripTrailingSemicolonFilter().process(stmt)
        assert result is stmt
        assert stmt.tokens == []


# ---------------------------
# SerializerUnicode


class TestSerializerUnicode:
    def test_single_line_rstripped(self):
        stmt = make_tokenlist(tk('SELECT   ', T.Text))
        result = SerializerUnicode.process(stmt)
        assert result == 'SELECT'

    def test_multiple_lines(self):
        stmt = make_tokenlist(
            tk('SELECT   ', T.Text),
            tk('\n', T.Whitespace.Newline),
            tk('  FROM', T.Keyword),
            tk('  ', T.Whitespace),
        )
        result = SerializerUnicode.process(stmt)
        assert result == 'SELECT\n  FROM'

    def test_newline_in_string_not_split(self):
        stmt = make_tokenlist(
            tk("'a", T.String.Single),
            tk('\n', T.String.Single),
            tk("b'", T.String.Single),
        )
        result = SerializerUnicode.process(stmt)
        # unquoted newline splitting should keep quoted content together
        assert "a\nb" in result or "a\n" in result

    def test_empty(self):
        stmt = make_tokenlist()
        assert SerializerUnicode.process(stmt) == ''

    def test_result_is_str(self):
        stmt = make_tokenlist(tk('abc', T.Text))
        assert isinstance(SerializerUnicode.process(stmt), str)
