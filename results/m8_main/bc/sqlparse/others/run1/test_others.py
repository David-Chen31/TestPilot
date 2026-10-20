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


def make_token_list(*tokens):
    # TokenList stores tokens internally; pass a list so mutating filters work.
    return sql.TokenList(list(tokens))


def stringify(stmt):
    return str(stmt)


# ---------------------------------------------------------------------------
# StripCommentsFilter
# ---------------------------------------------------------------------------

class TestStripCommentsFilter:
    def test_strip_single_comment_replaced_by_space(self):
        # Comment has whitespace on both sides, so it is *replaced* by a
        # single space (not removed): 'SELECT' + ' ' + ' '(replacement) + ' '
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Single, '-- foo'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
        )
        result = StripCommentsFilter().process(tl)
        assert str(result) == 'SELECT   1'

    def test_strip_multiline_comment(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT*'),
            make_token(T.Comment.Multiline, '/* c */'),
            make_token(T.Keyword, 'FROM'),
        )
        result = StripCommentsFilter().process(tl)
        assert str(result) == 'SELECT* FROM'

    def test_preserves_newline_after_comment(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Comment.Single, '-- comment'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Keyword, 'FROM'),
        )
        result = StripCommentsFilter().process(tl)
        assert '-- comment' not in str(result)
        assert '\n' in str(result)

    def test_comment_at_start_removed(self):
        tl = make_token_list(
            make_token(T.Comment.Single, '-- header'),
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'SELECT'),
        )
        result = StripCommentsFilter().process(tl)
        assert str(result) == ' SELECT'

    def test_comment_at_end_removed(self):
        # Comment at end: prev exists (not '('), so a whitespace is inserted
        # before removal. 'SELECT' + ' ' + ' '(inserted) = 'SELECT  '
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Single, '-- trailing'),
        )
        result = StripCommentsFilter().process(tl)
        assert str(result) == 'SELECT  '

    def test_sql_hint_single_preserved(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Single.Hint, '--+ hint'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
        )
        result = StripCommentsFilter().process(tl)
        assert '--+ hint' in str(result)

    def test_sql_hint_multiline_preserved(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Multiline.Hint, '/*+ hint */'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
        )
        result = StripCommentsFilter().process(tl)
        assert '/*+ hint */' in str(result)

    def test_comment_group_with_hint_preserved(self):
        inner = make_token(T.Comment.Multiline.Hint, '/*+ hint */')
        comment_group = sql.Comment([inner])
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            comment_group,
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
        )
        result = StripCommentsFilter().process(tl)
        assert '/*+ hint */' in str(result)

    def test_comment_group_without_hint_removed(self):
        inner = make_token(T.Comment.Multiline, '/* normal */')
        comment_group = sql.Comment([inner])
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            comment_group,
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
        )
        result = StripCommentsFilter().process(tl)
        assert '/* normal */' not in str(result)

    def test_empty_comment_group_removed(self):
        comment_group = sql.Comment([])
        tl = make_token_list(
            make_token(T.Keyword, 'A'),
            make_token(T.Whitespace, ' '),
            comment_group,
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'B'),
        )
        result = StripCommentsFilter().process(tl)
        assert result.tokens

    def test_comment_after_open_paren_no_insert(self):
        tl = make_token_list(
            make_token(T.Punctuation, '('),
            make_token(T.Comment.Single, '-- x'),
            make_token(T.Number.Integer, '1'),
            make_token(T.Punctuation, ')'),
        )
        result = StripCommentsFilter().process(tl)
        assert '-- x' not in str(result)

    def test_process_recurses_into_sublists(self):
        inner = sql.TokenList([
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Single, '-- inner'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
        ])
        tl = make_token_list(inner)
        result = StripCommentsFilter().process(tl)
        assert '-- inner' not in str(result)

    def test_no_comments_returns_unchanged(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
        )
        result = StripCommentsFilter().process(tl)
        assert str(result) == 'SELECT 1'


# ---------------------------------------------------------------------------
# StripWhitespaceFilter
# ---------------------------------------------------------------------------

class TestStripWhitespaceFilter:
    def test_stripws_default_collapses_whitespace(self):
        tl = make_token_list(
            make_token(T.Keyword, 'A'),
            make_token(T.Whitespace, '   '),
            make_token(T.Whitespace, '   '),
            make_token(T.Keyword, 'B'),
        )
        StripWhitespaceFilter._stripws_default(tl)
        assert [t.value for t in tl.tokens] == ['A', ' ', '', 'B']

    def test_stripws_default_no_leading_whitespace(self):
        tl = make_token_list(
            make_token(T.Whitespace, '  '),
            make_token(T.Keyword, 'A'),
        )
        StripWhitespaceFilter._stripws_default(tl)
        assert tl.tokens[0].value == ''
        assert tl.tokens[1].value == 'A'

    def test_process_removes_trailing_whitespace_at_depth_zero(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
        )
        result = StripWhitespaceFilter().process(tl)
        assert [t.value for t in result.tokens] == ['SELECT']

    def test_process_keeps_trailing_whitespace_for_nested(self):
        inner = sql.TokenList([
            make_token(T.Keyword, 'A'),
            make_token(T.Whitespace, ' '),
        ])
        tl = make_token_list(inner)
        result = StripWhitespaceFilter().process(tl)
        assert result.tokens[0].tokens[-1].value == ' '

    def test_process_empty_token_list(self):
        tl = make_token_list()
        result = StripWhitespaceFilter().process(tl)
        assert list(result.tokens) == []

    def test_stripws_identifierlist_removes_newline_before_comma(self):
        idlist = sql.IdentifierList([
            make_token(T.Name, 'a'),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Punctuation, ','),
            make_token(T.Name, 'b'),
        ])
        StripWhitespaceFilter()._stripws_identifierlist(idlist)
        values = [t.value for t in idlist.tokens]
        assert '\n' not in values

    def test_stripws_identifierlist_keeps_whitespace_before_comma(self):
        idlist = sql.IdentifierList([
            make_token(T.Name, 'a'),
            make_token(T.Punctuation, ','),
            make_token(T.Name, 'b'),
        ])
        StripWhitespaceFilter()._stripws_identifierlist(idlist)
        assert [t.value for t in idlist.tokens] == ['a', ',', 'b']

    def test_stripws_parenthesis_strips_leading_trailing_ws(self):
        paren = sql.Parenthesis([
            make_token(T.Punctuation, '('),
            make_token(T.Whitespace, ' '),
            make_token(T.Name, 'x'),
            make_token(T.Whitespace, ' '),
            make_token(T.Punctuation, ')'),
        ])
        StripWhitespaceFilter()._stripws_parenthesis(paren)
        values = [t.value for t in paren.tokens]
        assert values == ['(', 'x', ')']

    def test_stripws_parenthesis_inner_group_trailing_ws(self):
        inner = sql.TokenList([
            make_token(T.Name, 'x'),
            make_token(T.Whitespace, ' '),
        ])
        paren = sql.Parenthesis([
            make_token(T.Punctuation, '('),
            inner,
            make_token(T.Punctuation, ')'),
        ])
        StripWhitespaceFilter()._stripws_parenthesis(paren)
        assert inner.tokens[-1].value == 'x'

    def test_stripws_dispatch_calls_default(self):
        tl = sql.TokenList([
            make_token(T.Keyword, 'A'),
            make_token(T.Whitespace, '   '),
            make_token(T.Keyword, 'B'),
        ])
        StripWhitespaceFilter()._stripws(tl)
        assert tl.tokens[1].value == ' '


# ---------------------------------------------------------------------------
# SpacesAroundOperatorsFilter
# ---------------------------------------------------------------------------

class TestSpacesAroundOperatorsFilter:
    def test_adds_space_around_operator(self):
        tl = make_token_list(
            make_token(T.Name, 'a'),
            make_token(T.Operator, '+'),
            make_token(T.Name, 'b'),
        )
        SpacesAroundOperatorsFilter().process(tl)
        assert str(tl) == 'a + b'

    def test_adds_space_around_comparison(self):
        tl = make_token_list(
            make_token(T.Name, 'a'),
            make_token(T.Comparison, '='),
            make_token(T.Name, 'b'),
        )
        SpacesAroundOperatorsFilter().process(tl)
        assert str(tl) == 'a = b'

    def test_does_not_duplicate_existing_spaces(self):
        tl = make_token_list(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, ' '),
            make_token(T.Operator, '+'),
            make_token(T.Whitespace, ' '),
            make_token(T.Name, 'b'),
        )
        SpacesAroundOperatorsFilter().process(tl)
        assert str(tl) == 'a + b'

    def test_operator_at_start(self):
        # No prev token, so no leading space is inserted.
        tl = make_token_list(
            make_token(T.Operator, '-'),
            make_token(T.Name, 'a'),
        )
        SpacesAroundOperatorsFilter().process(tl)
        assert str(tl) == '- a'

    def test_operator_at_end(self):
        # No next token, so no trailing space is inserted.
        tl = make_token_list(
            make_token(T.Name, 'a'),
            make_token(T.Operator, '+'),
        )
        SpacesAroundOperatorsFilter().process(tl)
        assert str(tl) == 'a +'

    def test_multiple_operators(self):
        tl = make_token_list(
            make_token(T.Name, 'a'),
            make_token(T.Operator, '+'),
            make_token(T.Name, 'b'),
            make_token(T.Comparison, '='),
            make_token(T.Name, 'c'),
        )
        SpacesAroundOperatorsFilter().process(tl)
        assert str(tl) == 'a + b = c'

    def test_recurses_into_sublists(self):
        inner = sql.TokenList([
            make_token(T.Name, 'a'),
            make_token(T.Operator, '+'),
            make_token(T.Name, 'b'),
        ])
        tl = make_token_list(inner)
        SpacesAroundOperatorsFilter().process(tl)
        assert str(inner) == 'a + b'

    def test_no_operators_unchanged(self):
        tl = make_token_list(
            make_token(T.Name, 'a'),
            make_token(T.Whitespace, ' '),
            make_token(T.Name, 'b'),
        )
        SpacesAroundOperatorsFilter().process(tl)
        assert str(tl) == 'a b'


# ---------------------------------------------------------------------------
# StripTrailingSemicolonFilter
# ---------------------------------------------------------------------------

class TestStripTrailingSemicolonFilter:
    def test_strips_trailing_semicolon(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Punctuation, ';'),
        )
        result = StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in result.tokens] == ['SELECT']

    def test_strips_trailing_semicolon_and_whitespace(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Punctuation, ';'),
            make_token(T.Whitespace, ' '),
        )
        result = StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in result.tokens] == ['SELECT']

    def test_strips_multiple_trailing_semicolons_and_ws(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Punctuation, ';'),
            make_token(T.Whitespace, ' '),
            make_token(T.Punctuation, ';'),
        )
        result = StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in result.tokens] == ['SELECT']

    def test_no_trailing_semicolon_unchanged(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Number.Integer, '1'),
        )
        result = StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in result.tokens] == ['SELECT', '1']

    def test_empty_token_list(self):
        tl = make_token_list()
        result = StripTrailingSemicolonFilter().process(tl)
        assert list(result.tokens) == []

    def test_semicolon_in_middle_kept(self):
        tl = make_token_list(
            make_token(T.Keyword, 'a'),
            make_token(T.Punctuation, ';'),
            make_token(T.Keyword, 'b'),
        )
        result = StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in result.tokens] == ['a', ';', 'b']


# ---------------------------------------------------------------------------
# SerializerUnicode
# ---------------------------------------------------------------------------

class TestSerializerUnicode:
    def test_basic_roundtrip(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
        )
        out = SerializerUnicode.process(tl)
        assert out == 'SELECT 1'

    def test_strips_trailing_whitespace_per_line(self):
        tl = make_token_list(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Keyword, 'FROM'),
            make_token(T.Whitespace, '  '),
        )
        out = SerializerUnicode.process(tl)
        assert out == 'SELECT\nFROM'

    def test_multiple_lines(self):
        tl = make_token_list(
            make_token(T.Keyword, 'a'),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Keyword, 'b'),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Keyword, 'c'),
        )
        out = SerializerUnicode.process(tl)
        assert out == 'a\nb\nc'

    def test_empty(self):
        tl = make_token_list()
        assert SerializerUnicode.process(tl) == ''

    def test_is_static(self):
        tl = make_token_list(make_token(T.Keyword, 'x'))
        assert SerializerUnicode.process(tl) == 'x'
