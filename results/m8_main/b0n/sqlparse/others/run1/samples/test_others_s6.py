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

def make_token(ttype, value):
    return sql.Token(ttype, value)


def make_tlist(*tokens):
    return sql.TokenList(tokens)


def token_values(tlist):
    return [t.value for t in tlist.tokens]


def token_types(tlist):
    return [t.ttype for t in tlist.tokens]


# ---------------------------------------------------------------------------
# StripCommentsFilter
# ---------------------------------------------------------------------------

class TestStripCommentsFilter:

    def test_remove_single_comment_between_tokens(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Comment.Single, '-- comment'),
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', ' ', 'bar']
        assert token_types(tlist) == [T.Name, T.Whitespace, T.Name]

    def test_remove_multiline_comment_between_tokens(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Comment.Multiline, '/* c */'),
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', ' ', 'bar']

    def test_remove_comment_at_start_no_prev(self):
        tlist = make_tlist(
            make_token(T.Comment.Single, '-- c'),
            make_token(T.Name, 'foo'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo']

    def test_remove_comment_at_end_no_next(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Comment.Single, '-- c'),
        )
        StripCommentsFilter._process(tlist)
        # prev exists and is not '(' -> whitespace inserted
        assert token_values(tlist) == ['foo', ' ']

    def test_comment_preserves_trailing_newline(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Comment.Single, '-- c\n'),
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', '\n', 'bar']
        assert tlist.tokens[1].ttype is T.Whitespace.Newline

    def test_comment_preserves_carriage_return(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Comment.Single, '-- c\r\n'),
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', '\r\n', 'bar']
        assert tlist.tokens[1].ttype is T.Whitespace.Newline

    def test_comment_after_open_paren_no_whitespace_inserted(self):
        tlist = make_tlist(
            make_token(T.Punctuation, '('),
            make_token(T.Comment.Single, '-- c'),
            make_token(T.Name, 'foo'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['(', 'foo']

    def test_comment_before_close_paren(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Comment.Single, '-- c'),
            make_token(T.Punctuation, ')'),
        )
        StripCommentsFilter._process(tlist)
        # next is ')', but prev is not '(' -> whitespace inserted
        assert token_values(tlist) == ['foo', ' ', ')']

    def test_comment_surrounded_by_whitespace_just_removed(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Single, '-- c'),
            make_token(T.Whitespace, ' '),
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', ' ', ' ', 'bar']

    def test_sql_hint_single_kept(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Comment.Single.Hint, '--+ hint'),
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', '--+ hint', 'bar']

    def test_sql_hint_multiline_kept(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Comment.Multiline.Hint, '/*+ hint */'),
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', '/*+ hint */', 'bar']

    def test_sql_hint_in_comment_group_kept(self):
        inner = make_token(T.Comment.Single.Hint, '--+ hint')
        comment_group = sql.Comment([inner])
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            comment_group,
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', '--+ hint', 'bar']

    def test_empty_comment_group_removed(self):
        comment_group = sql.Comment([])
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            comment_group,
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', ' ', 'bar']

    def test_no_comments_no_change(self):
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            make_token(T.Whitespace, ' '),
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert token_values(tlist) == ['foo', ' ', 'bar']

    def test_process_processes_sublists(self):
        inner = sql.TokenList([
            make_token(T.Name, 'foo'),
            make_token(T.Comment.Single, '-- c'),
            make_token(T.Name, 'bar'),
        ])
        tlist = make_tlist(inner)
        StripCommentsFilter().process(tlist)
        assert token_values(inner) == ['foo', ' ', 'bar']
        assert len(tlist.tokens) == 1

    def test_process_returns_stmt(self):
        tlist = make_tlist(make_token(T.Name, 'foo'))
        result = StripCommentsFilter().process(tlist)
        assert result is tlist

    def test_get_insert_token_plain(self):
        tok = make_token(T.Comment.Single, '-- no newline')
        result = StripCommentsFilter._process.__globals__['re'] and \
            sql.Token(T.Whitespace, ' ')
        # Use the actual static helper indirectly via removal
        tlist = make_tlist(
            make_token(T.Name, 'foo'),
            tok,
            make_token(T.Name, 'bar'),
        )
        StripCommentsFilter._process(tlist)
        assert tlist.tokens[1].ttype is T.Whitespace
        assert tlist.tokens[1].value == ' '


# ---------------------------------------------------------------------------
# StripWhitespaceFilter
# ---------------------------------------------------------------------------

class TestStripWhitespaceFilter:

    def test_collapses_consecutive_whitespace(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, '   '),
            make_token(T.Whitespace, '  '),
            make_token(T.Name, 'b'),
        )
        StripWhitespaceFilter._stripws_default(tlist)
        assert token_values(tlist) == ['a', ' ', '', 'b']

    def test_leading_whitespace_removed(self):
        tlist = make_tlist(
            make_token(T.Whitespace, '   '),
            make_token(T.Name, 'a'),
        )
        StripWhitespaceFilter._stripws_default(tlist)
        assert token_values(tlist) == ['', 'a']

    def test_stripws_dispatches_to_default(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, '   '),
            make_token(T.Name, 'b'),
        )
        StripWhitespaceFilter()._stripws(tlist)
        assert token_values(tlist) == ['a', ' ', 'b']

    def test_stripws_identifierlist_removes_newline_before_comma(self):
        tlist = sql.IdentifierList([
            make_token(T.Name, 'a'),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Punctuation, ','),
            make_token(T.Name, 'b'),
        ])
        StripWhitespaceFilter()._stripws(tlist)
        assert token_values(tlist) == ['a', ',', 'b']

    def test_stripws_identifierlist_no_newline(self):
        tlist = sql.IdentifierList([
            make_token(T.Name, 'a'),
            make_token(T.Punctuation, ','),
            make_token(T.Name, 'b'),
        ])
        StripWhitespaceFilter()._stripws(tlist)
        assert token_values(tlist) == ['a', ',', 'b']

    def test_stripws_parenthesis_trims_inner_whitespace(self):
        tlist = sql.Parenthesis([
            make_token(T.Punctuation, '('),
            make_token(T.Whitespace, ' '),
            make_token(T.Whitespace, ' '),
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, ' '),
            make_token(T.Whitespace, ' '),
            make_token(T.Punctuation, ')'),
        ])
        StripWhitespaceFilter()._stripws(tlist)
        assert token_values(tlist) == ['(', 'a', ')']

    def test_stripws_parenthesis_trims_trailing_group_whitespace(self):
        inner = sql.TokenList([
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, ' '),
            make_token(T.Whitespace, ' '),
        ])
        tlist = sql.Parenthesis([
            make_token(T.Punctuation, '('),
            inner,
            make_token(T.Punctuation, ')'),
        ])
        StripWhitespaceFilter()._stripws(tlist)
        assert [t.value for t in inner.tokens] == ['a']

    def test_process_strips_trailing_whitespace_at_root(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, '   '),
        )
        StripWhitespaceFilter().process(tlist)
        assert token_values(tlist) == ['a']

    def test_process_keeps_trailing_whitespace_when_nested(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, '   '),
        )
        StripWhitespaceFilter().process(tlist, depth=1)
        # Whitespace collapsed but kept as ' ' since not depth 0
        assert token_values(tlist) == ['a', ' ']

    def test_process_processes_sublists(self):
        sub = sql.TokenList([
            make_token(T.Whitespace, '  '),
            make_token(T.Name, 'a'),
        ])
        tlist = make_tlist(sub)
        StripWhitespaceFilter().process(tlist)
        assert [t.value for t in sub.tokens] == ['', 'a']

    def test_process_returns_stmt(self):
        tlist = make_tlist(make_token(T.Name, 'a'))
        result = StripWhitespaceFilter().process(tlist)
        assert result is tlist

    def test_process_empty_tlist(self):
        tlist = make_tlist()
        StripWhitespaceFilter().process(tlist)
        assert tlist.tokens == []


# ---------------------------------------------------------------------------
# SpacesAroundOperatorsFilter
# ---------------------------------------------------------------------------

class TestSpacesAroundOperatorsFilter:

    def test_spaces_around_operator(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Operator, '+'),
            make_token(T.Name, 'b'),
        )
        SpacesAroundOperatorsFilter._process(tlist)
        assert token_values(tlist) == ['a', ' ', '+', ' ', 'b']

    def test_spaces_around_comparison(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Comparison, '='),
            make_token(T.Name, 'b'),
        )
        SpacesAroundOperatorsFilter._process(tlist)
        assert token_values(tlist) == ['a', ' ', '=', ' ', 'b']

    def test_existing_spaces_not_duplicated(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, ' '),
            make_token(T.Operator, '+'),
            make_token(T.Whitespace, ' '),
            make_token(T.Name, 'b'),
        )
        SpacesAroundOperatorsFilter._process(tlist)
        assert token_values(tlist) == ['a', ' ', '+', ' ', 'b']

    def test_operator_at_start(self):
        tlist = make_tlist(
            make_token(T.Operator, '-'),
            make_token(T.Name, 'a'),
        )
        SpacesAroundOperatorsFilter._process(tlist)
        assert token_values(tlist) == ['-', ' ', 'a']

    def test_operator_at_end(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Operator, '-'),
        )
        SpacesAroundOperatorsFilter._process(tlist)
        assert token_values(tlist) == ['a', ' ', '-']

    def test_multiple_operators(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Operator, '+'),
            make_token(T.Name, 'b'),
            make_token(T.Comparison, '='),
            make_token(T.Name, 'c'),
        )
        SpacesAroundOperatorsFilter._process(tlist)
        assert token_values(tlist) == [
            'a', ' ', '+', ' ', 'b', ' ', '=', ' ', 'c',
        ]

    def test_no_operators(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, ' '),
            make_token(T.Name, 'b'),
        )
        SpacesAroundOperatorsFilter._process(tlist)
        assert token_values(tlist) == ['a', ' ', 'b']

    def test_inserted_before_is_whitespace_token(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Operator, '+'),
            make_token(T.Name, 'b'),
        )
        SpacesAroundOperatorsFilter._process(tlist)
        assert tlist.tokens[1].ttype is T.Whitespace
        assert tlist.tokens[1].value == ' '
        assert tlist.tokens[3].ttype is T.Whitespace
        assert tlist.tokens[3].value == ' '

    def test_process_processes_sublists(self):
        sub = sql.TokenList([
            make_token(T.Name, 'a'),
            make_token(T.Operator, '+'),
            make_token(T.Name, 'b'),
        ])
        tlist = make_tlist(sub)
        SpacesAroundOperatorsFilter().process(tlist)
        assert [t.value for t in sub.tokens] == ['a', ' ', '+', ' ', 'b']

    def test_process_returns_stmt(self):
        tlist = make_tlist(make_token(T.Name, 'a'))
        result = SpacesAroundOperatorsFilter().process(tlist)
        assert result is tlist


# ---------------------------------------------------------------------------
# StripTrailingSemicolonFilter
# ---------------------------------------------------------------------------

class TestStripTrailingSemicolonFilter:

    def test_removes_trailing_semicolon(self):
        tlist = make_tlist(
            make_token(T.Name, 'SELECT'),
            make_token(T.Punctuation, ';'),
        )
        StripTrailingSemicolonFilter().process(tlist)
        assert token_values(tlist) == ['SELECT']

    def test_removes_trailing_whitespace_and_semicolon(self):
        tlist = make_tlist(
            make_token(T.Name, 'SELECT'),
            make_token(T.Punctuation, ';'),
            make_token(T.Whitespace, ' '),
            make_token(T.Whitespace.Newline, '\n'),
        )
        StripTrailingSemicolonFilter().process(tlist)
        assert token_values(tlist) == ['SELECT']

    def test_removes_multiple_trailing_semicolons(self):
        tlist = make_tlist(
            make_token(T.Name, 'SELECT'),
            make_token(T.Punctuation, ';'),
            make_token(T.Punctuation, ';'),
        )
        StripTrailingSemicolonFilter().process(tlist)
        assert token_values(tlist) == ['SELECT']

    def test_no_semicolon_no_change(self):
        tlist = make_tlist(
            make_token(T.Name, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Name, 'x'),
        )
        StripTrailingSemicolonFilter().process(tlist)
        assert token_values(tlist) == ['SELECT', ' ', 'x']

    def test_removes_only_trailing_whitespace(self):
        tlist = make_tlist(
            make_token(T.Name, 'SELECT'),
            make_token(T.Whitespace, '   '),
        )
        StripTrailingSemicolonFilter().process(tlist)
        assert token_values(tlist) == ['SELECT']

    def test_empty_stmt(self):
        tlist = make_tlist()
        StripTrailingSemicolonFilter().process(tlist)
        assert tlist.tokens == []

    def test_returns_stmt(self):
        tlist = make_tlist(make_token(T.Name, 'x'))
        result = StripTrailingSemicolonFilter().process(tlist)
        assert result is tlist


# ---------------------------------------------------------------------------
# SerializerUnicode
# ---------------------------------------------------------------------------

class TestSerializerUnicode:

    def test_simple_string(self):
        tlist = make_tlist(make_token(T.Name, 'SELECT 1'))
        assert SerializerUnicode.process(tlist) == 'SELECT 1'

    def test_strips_trailing_whitespace_per_line(self):
        tlist = make_tlist(
            make_token(T.Name, 'SELECT'),
            make_token(T.Whitespace, '   '),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Name, 'x'),
        )
        result = SerializerUnicode.process(tlist)
        assert result == 'SELECT\nx'

    def test_newlines_joined(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Name, 'b'),
        )
        assert SerializerUnicode.process(tlist) == 'a\nb'

    def test_crlf_normalized(self):
        tlist = make_tlist(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace.Newline, '\r\n'),
            make_token(T.Name, 'b'),
        )
        assert SerializerUnicode.process(tlist) == 'a\nb'

    def test_empty_stmt(self):
        tlist = make_tlist()
        assert SerializerUnicode.process(tlist) == ''

    def test_string_literal_newline_preserved(self):
        tlist = make_tlist(
            make_token(T.String.Single, "'a\nb'"),
        )
        assert SerializerUnicode.process(tlist) == "'a\nb'"
