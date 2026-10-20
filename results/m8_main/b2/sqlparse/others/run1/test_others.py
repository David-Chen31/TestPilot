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


def make_tlist(tokens=None, cls=sql.TokenList, ttype=None):
    """Create a TokenList (or subclass) with the given tokens.

    The TokenList API only accepts ``tokens`` as keyword argument in this
    version, so ``ttype`` is only applied when the class supports it.
    """
    tl = cls(tokens=list(tokens or []))
    if ttype is not None:
        try:
            tl.ttype = ttype
        except AttributeError:
            pass
    return tl


def make_comment(tokens, ttype=T.Comment):
    """Build a sql.Comment group; ttype is set after construction."""
    comment = sql.Comment(tokens=list(tokens))
    comment.ttype = ttype
    return comment


# ---------------------------------------------------------------------------
# StripCommentsFilter
# ---------------------------------------------------------------------------

class TestStripCommentsFilter:
    def test_removes_simple_comment_replaced_with_space(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Multiline, '/* c */'),
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        result = StripCommentsFilter().process(tl)
        values = [t.value for t in result.tokens]
        assert '/* c */' not in values
        assert any(v == ' ' for v in values)

    def test_comment_replacement_token_type_is_whitespace(self):
        # Kills mutations that change the insert token ttype (line 33).
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Multiline, '/* c */'),
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        # The comment was replaced by a whitespace token with one space.
        values = [(t.ttype, t.value) for t in tl.tokens]
        assert (T.Whitespace, ' ') in values

    def test_comment_replacement_between_non_ws_uses_whitespace_ttype(self):
        tokens = [
            make_token(T.Number.Integer, '1'),
            make_token(T.Comment.Multiline, '/* c */'),
            make_token(T.Number.Integer, '2'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert tl.tokens[1].ttype is T.Whitespace
        assert tl.tokens[1].value == ' '

    def test_comment_group_is_removed(self):
        # Kills mutation making idx=None in get_next_comment (line 21).
        comment = make_comment([make_token(T.Comment.Multiline, '/* c */')],
                               ttype=T.Comment.Multiline)
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            comment,
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert not any(t is comment for t in tl.tokens)
        assert not any(isinstance(t, sql.Comment) for t in tl.tokens)

    def test_comment_group_is_detected_by_comment_type(self):
        # A plain comment token with ttype T.Comment (not multiline) is removed.
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment, 'x'),
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert not any(t.value == 'x' for t in tl.tokens)

    def test_comment_at_start_removed(self):
        tokens = [
            make_token(T.Comment.Single, '-- hi'),
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'SELECT'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert not any(t.ttype in T.Comment for t in tl.tokens)

    def test_comment_at_end_removed(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Single, '-- hi'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert not any(t.ttype in T.Comment for t in tl.tokens)

    def test_comment_between_non_ws_tokens_replaced(self):
        tokens = [
            make_token(T.Number.Integer, '1'),
            make_token(T.Comment.Multiline, '/* c */'),
            make_token(T.Number.Integer, '2'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        values = [t.value for t in tl.tokens]
        assert values == ['1', ' ', '2']

    def test_sql_hint_multiline_is_preserved(self):
        hint = make_token(T.Comment.Multiline.Hint, '/*+ hint */')
        tokens = [
            make_token(T.Whitespace, ' '),
            hint,
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'SELECT'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert any(t is hint for t in tl.tokens)

    def test_sql_hint_single_is_preserved(self):
        hint = make_token(T.Comment.Single.Hint, '--+ hint')
        tokens = [
            make_token(T.Whitespace, ' '),
            hint,
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'SELECT'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert any(t is hint for t in tl.tokens)

    def test_sql_hint_inside_comment_group_is_preserved(self):
        hint_token = make_token(T.Comment.Multiline.Hint, '/*+ hint */')
        comment = make_comment([hint_token], ttype=T.Comment)
        tokens = [
            make_token(T.Whitespace, ' '),
            comment,
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'SELECT'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert any(t is comment for t in tl.tokens)

    def test_multiple_hints_and_comments_mixed(self):
        # Kills mutation of `continue` -> `break` (line 53): if a hint causes
        # the loop to break instead of continue, later comments are not removed.
        hint = make_token(T.Comment.Multiline.Hint, '/*+ h */')
        comment = make_token(T.Comment.Multiline, '/* c */')
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            hint,
            make_token(T.Whitespace, ' '),
            comment,
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert any(t is hint for t in tl.tokens)
        assert not any(t is comment for t in tl.tokens)
        assert not any(t.value == '/* c */' for t in tl.tokens)

    def test_empty_comment_group_kept_then_removed(self):
        # Covers the len(comment_tokens) > 0 boundary (line 44).
        empty_comment = make_comment([], ttype=T.Comment)
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            empty_comment,
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert not any(t is empty_comment for t in tl.tokens)

    def test_line_break_preserved_from_comment(self):
        tokens = [
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Single, '-- comment\n'),
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'SELECT'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        newline_tokens = [t for t in tl.tokens if t.ttype is T.Whitespace.Newline]
        assert len(newline_tokens) == 1
        assert newline_tokens[0].value == '\n'

    def test_carriage_return_line_break(self):
        tokens = [
            make_token(T.Whitespace, ' '),
            make_token(T.Comment.Single, '-- comment\r\n'),
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'SELECT'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        newline_tokens = [t for t in tl.tokens if t.ttype is T.Whitespace.Newline]
        assert newline_tokens[0].value == '\r\n'

    def test_comment_after_open_paren(self):
        tokens = [
            make_token(T.Punctuation, '('),
            make_token(T.Comment.Multiline, '/* c */'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
            make_token(T.Punctuation, ')'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        values = [t.value for t in tl.tokens]
        assert '/* c */' not in values
        idx = values.index('(')
        assert values[idx + 1] == ' '

    def test_comment_right_before_open_paren_not_inserted(self):
        # prev_ is '(', so no insert token should be added before removal.
        tokens = [
            make_token(T.Punctuation, '('),
            make_token(T.Comment.Multiline, '/* c */'),
            make_token(T.Number.Integer, '1'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        # No whitespace was inserted after '('.
        assert tl.tokens[0].value == '('
        assert tl.tokens[1].value == '1'

    def test_comment_right_before_close_paren(self):
        tokens = [
            make_token(T.Number.Integer, '1'),
            make_token(T.Comment.Multiline, '/* c */'),
            make_token(T.Punctuation, ')'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        values = [t.value for t in tl.tokens]
        assert '/* c */' not in values
        # A whitespace insertion happens before removal since prev isn't None
        # and isn't '('; the resulting sequence is checked as a whole.
        assert values[-1] == ')'

    def test_processes_sublists(self):
        inner = make_tlist(
            [
                make_token(T.Comment.Multiline, '/* c */'),
                make_token(T.Whitespace, ' '),
                make_token(T.Keyword, 'FROM'),
            ],
            cls=sql.Parenthesis,
        )
        outer = make_tlist(
            [
                make_token(T.Keyword, 'SELECT'),
                make_token(T.Whitespace, ' '),
                inner,
            ]
        )
        StripCommentsFilter().process(outer)
        assert not any(t.ttype in T.Comment for t in inner.tokens)

    def test_no_comments_returns_unchanged(self):
        tokens = [make_token(T.Keyword, 'SELECT'), make_token(T.Whitespace, ' ')]
        tl = make_tlist(tokens)
        result = StripCommentsFilter().process(tl)
        assert result is tl
        assert [t.value for t in result.tokens] == ['SELECT', ' ']

    def test_multiple_comments(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Comment.Multiline, '/* a */'),
            make_token(T.Keyword, 'x'),
            make_token(T.Comment.Multiline, '/* b */'),
            make_token(T.Keyword, 'y'),
        ]
        tl = make_tlist(tokens)
        StripCommentsFilter().process(tl)
        assert not any(t.ttype in T.Comment for t in tl.tokens)

    def test_process_returns_same_stmt(self):
        tl = make_tlist([make_token(T.Comment.Multiline, '/* c */')])
        assert StripCommentsFilter().process(tl) is tl


# ---------------------------------------------------------------------------
# StripWhitespaceFilter
# ---------------------------------------------------------------------------

class TestStripWhitespaceFilter:
    def test_default_collapses_leading_and_double_ws(self):
        tokens = [
            make_token(T.Whitespace, '   '),
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '  '),
            make_token(T.Whitespace, '   '),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        StripWhitespaceFilter._stripws_default(tl)
        assert tl.tokens[0].value == ''
        assert tl.tokens[2].value == ' '
        assert tl.tokens[3].value == ''

    def test_single_whitespace_becomes_single_space(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '\n\t'),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        StripWhitespaceFilter._stripws_default(tl)
        assert tl.tokens[1].value == ' '

    def test_default_whitespace_token_type_preserved(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '  '),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        StripWhitespaceFilter._stripws_default(tl)
        assert tl.tokens[1].ttype is T.Whitespace
        assert tl.tokens[1].value == ' '

    def test_identifierlist_removes_newline_before_comma(self):
        tokens = [
            make_token(T.Keyword, 'a'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Punctuation, ','),
            make_token(T.Keyword, 'b'),
        ]
        tl = make_tlist(tokens, cls=sql.IdentifierList, ttype=T.Keyword)
        StripWhitespaceFilter()._stripws_identifierlist(tl)
        values = [t.value for t in tl.tokens]
        assert values == ['a', ',', 'b']

    def test_identifierlist_no_newline_before_comma(self):
        tokens = [
            make_token(T.Keyword, 'a'),
            make_token(T.Punctuation, ','),
            make_token(T.Keyword, 'b'),
        ]
        tl = make_tlist(tokens, cls=sql.IdentifierList, ttype=T.Keyword)
        StripWhitespaceFilter()._stripws_identifierlist(tl)
        assert [t.value for t in tl.tokens] == ['a', ',', 'b']

    def test_parenthesis_strips_leading_and_trailing_ws(self):
        inner = sql.Parenthesis(
            tokens=[
                make_token(T.Punctuation, '('),
                make_token(T.Whitespace, '  '),
                make_token(T.Number.Integer, '1'),
                make_token(T.Whitespace, '  '),
                make_token(T.Punctuation, ')'),
            ]
        )
        StripWhitespaceFilter()._stripws_parenthesis(inner)
        values = [t.value for t in inner.tokens]
        assert values[0] == '('
        assert values[1] == '1'
        assert values[-1] == ')'

    def test_parenthesis_strips_trailing_ws_of_inner_group(self):
        inner_group = sql.Function(
            tokens=[
                make_token(T.Name, 'f'),
                make_token(T.Punctuation, '('),
                make_token(T.Punctuation, ')'),
                make_token(T.Whitespace, '  '),
            ]
        )
        paren = sql.Parenthesis(
            tokens=[
                make_token(T.Punctuation, '('),
                make_token(T.Whitespace, ' '),
                inner_group,
                make_token(T.Whitespace, ' '),
                make_token(T.Punctuation, ')'),
            ]
        )
        StripWhitespaceFilter()._stripws_parenthesis(paren)
        # The inner group's trailing whitespace is stripped; last token is ')'
        assert inner_group.tokens[-1].value == ')'

    def test_process_removes_trailing_ws_at_depth_zero(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '  '),
        ]
        tl = make_tlist(tokens)
        StripWhitespaceFilter().process(tl)
        assert [t.value for t in tl.tokens] == ['SELECT']

    def test_process_keeps_trailing_ws_at_depth_nonzero(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '  '),
        ]
        tl = make_tlist(tokens)
        StripWhitespaceFilter().process(tl, depth=1)
        assert len(tl.tokens) == 2

    def test_process_recurses_sublists(self):
        inner = sql.Parenthesis(
            tokens=[
                make_token(T.Punctuation, '('),
                make_token(T.Whitespace, '  '),
                make_token(T.Number.Integer, '1'),
                make_token(T.Whitespace, '  '),
                make_token(T.Punctuation, ')'),
            ]
        )
        outer = make_tlist(
            [
                make_token(T.Keyword, 'SELECT'),
                make_token(T.Whitespace, ' '),
                inner,
            ]
        )
        StripWhitespaceFilter().process(outer)
        assert inner.tokens[1].value == '1'

    def test_process_empty_tokenlist(self):
        tl = make_tlist([])
        result = StripWhitespaceFilter().process(tl)
        assert result.tokens == []

    def test_process_returns_same_stmt(self):
        tl = make_tlist([make_token(T.Keyword, 'SELECT')])
        assert StripWhitespaceFilter().process(tl) is tl


# ---------------------------------------------------------------------------
# SpacesAroundOperatorsFilter
# ---------------------------------------------------------------------------

class TestSpacesAroundOperatorsFilter:
    def test_spaces_added_around_operator(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
            make_token(T.Operator, '+'),
            make_token(T.Number.Integer, '2'),
        ]
        tl = make_tlist(tokens)
        SpacesAroundOperatorsFilter().process(tl)
        values = [t.value for t in tl.tokens]
        assert values == ['SELECT', ' ', '1', ' ', '+', ' ', '2']

    def test_spaces_added_around_comparison(self):
        tokens = [
            make_token(T.Number.Integer, '1'),
            make_token(T.Comparison, '='),
            make_token(T.Number.Integer, '2'),
        ]
        tl = make_tlist(tokens)
        SpacesAroundOperatorsFilter._process(tl)
        values = [t.value for t in tl.tokens]
        assert values == ['1', ' ', '=', ' ', '2']

    def test_inserted_space_ttype_is_whitespace(self):
        tokens = [
            make_token(T.Number.Integer, '1'),
            make_token(T.Operator, '+'),
            make_token(T.Number.Integer, '2'),
        ]
        tl = make_tlist(tokens)
        SpacesAroundOperatorsFilter._process(tl)
        assert tl.tokens[1].ttype is T.Whitespace
        assert tl.tokens[1].value == ' '
        assert tl.tokens[3].ttype is T.Whitespace
        assert tl.tokens[3].value == ' '

    def test_existing_spaces_not_duplicated(self):
        tokens = [
            make_token(T.Number.Integer, '1'),
            make_token(T.Whitespace, ' '),
            make_token(T.Operator, '+'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '2'),
        ]
        tl = make_tlist(tokens)
        SpacesAroundOperatorsFilter._process(tl)
        values = [t.value for t in tl.tokens]
        assert values == ['1', ' ', '+', ' ', '2']

    def test_operator_at_start(self):
        tokens = [
            make_token(T.Operator, '-'),
            make_token(T.Number.Integer, '1'),
        ]
        tl = make_tlist(tokens)
        SpacesAroundOperatorsFilter._process(tl)
        values = [t.value for t in tl.tokens]
        assert values == ['-', ' ', '1']

    def test_operator_at_end(self):
        tokens = [
            make_token(T.Number.Integer, '1'),
            make_token(T.Operator, '+'),
        ]
        tl = make_tlist(tokens)
        SpacesAroundOperatorsFilter._process(tl)
        values = [t.value for t in tl.tokens]
        assert values == ['1', ' ', '+']

    def test_multiple_operators(self):
        tokens = [
            make_token(T.Number.Integer, '1'),
            make_token(T.Operator, '+'),
            make_token(T.Number.Integer, '2'),
            make_token(T.Comparison, '='),
            make_token(T.Number.Integer, '3'),
        ]
        tl = make_tlist(tokens)
        SpacesAroundOperatorsFilter._process(tl)
        values = [t.value for t in tl.tokens]
        assert values == ['1', ' ', '+', ' ', '2', ' ', '=', ' ', '3']

    def test_process_recurses_sublists(self):
        inner = sql.Parenthesis(
            tokens=[
                make_token(T.Punctuation, '('),
                make_token(T.Number.Integer, '1'),
                make_token(T.Operator, '+'),
                make_token(T.Number.Integer, '2'),
                make_token(T.Punctuation, ')'),
            ]
        )
        outer = make_tlist(
            [
                make_token(T.Keyword, 'SELECT'),
                make_token(T.Whitespace, ' '),
                inner,
            ]
        )
        SpacesAroundOperatorsFilter().process(outer)
        values = [t.value for t in inner.tokens]
        assert values == ['(', '1', ' ', '+', ' ', '2', ')']

    def test_no_operators(self):
        tokens = [make_token(T.Keyword, 'SELECT'), make_token(T.Whitespace, ' ')]
        tl = make_tlist(tokens)
        result = SpacesAroundOperatorsFilter().process(tl)
        assert result is tl
        assert [t.value for t in result.tokens] == ['SELECT', ' ']

    def test_process_returns_same_stmt(self):
        tl = make_tlist([make_token(T.Number.Integer, '1'),
                         make_token(T.Operator, '+')])
        assert SpacesAroundOperatorsFilter().process(tl) is tl


# ---------------------------------------------------------------------------
# StripTrailingSemicolonFilter
# ---------------------------------------------------------------------------

class TestStripTrailingSemicolonFilter:
    def test_strips_trailing_semicolon(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Punctuation, ';'),
        ]
        tl = make_tlist(tokens)
        StripTrailingSemicolonFilter().process(tl)
        # The filter pops the semicolon and then keeps popping trailing
        # whitespace, so only 'SELECT' remains.
        assert [t.value for t in tl.tokens] == ['SELECT']

    def test_strips_trailing_whitespace(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '  '),
        ]
        tl = make_tlist(tokens)
        StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in tl.tokens] == ['SELECT']

    def test_strips_combination_semicolon_and_ws(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Punctuation, ';'),
            make_token(T.Whitespace, '  '),
        ]
        tl = make_tlist(tokens)
        StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in tl.tokens] == ['SELECT']

    def test_multiple_semicolons(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Punctuation, ';'),
            make_token(T.Punctuation, ';'),
        ]
        tl = make_tlist(tokens)
        StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in tl.tokens] == ['SELECT']

    def test_keeps_inner_semicolon(self):
        tokens = [
            make_token(T.Punctuation, ';'),
            make_token(T.Keyword, 'SELECT'),
        ]
        tl = make_tlist(tokens)
        StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in tl.tokens] == [';', 'SELECT']

    def test_non_semicolon_punctuation_kept(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Punctuation, ','),
        ]
        tl = make_tlist(tokens)
        StripTrailingSemicolonFilter().process(tl)
        assert [t.value for t in tl.tokens] == ['SELECT', ',']

    def test_empty_tokenlist(self):
        tl = make_tlist([])
        result = StripTrailingSemicolonFilter().process(tl)
        assert result.tokens == []

    def test_process_returns_same_stmt(self):
        tl = make_tlist([make_token(T.Keyword, 'SELECT')])
        assert StripTrailingSemicolonFilter().process(tl) is tl


# ---------------------------------------------------------------------------
# SerializerUnicode
# ---------------------------------------------------------------------------

class TestSerializerUnicode:
    def test_basic_serialization(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
        ]
        tl = make_tlist(tokens)
        assert SerializerUnicode.process(tl) == 'SELECT 1'

    def test_strips_trailing_whitespace_per_line(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, ' '),
            make_token(T.Number.Integer, '1'),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Keyword, 'FROM'),
            make_token(T.Whitespace, '  '),
        ]
        tl = make_tlist(tokens)
        result = SerializerUnicode.process(tl)
        for line in result.split('\n'):
            assert line == line.rstrip()

    def test_preserves_newlines_between_lines(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Keyword, 'FROM'),
        ]
        tl = make_tlist(tokens)
        result = SerializerUnicode.process(tl)
        assert '\n' in result
        assert result.split('\n')[0] == 'SELECT'

    def test_exact_output_two_lines(self):
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '  '),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Keyword, 'FROM'),
            make_token(T.Whitespace, '   '),
        ]
        tl = make_tlist(tokens)
        assert SerializerUnicode.process(tl) == 'SELECT\nFROM'

    def test_empty(self):
        tl = make_tlist([])
        assert SerializerUnicode.process(tl) == ''

    def test_only_whitespace(self):
        tl = make_tlist([make_token(T.Whitespace, '   ')])
        assert SerializerUnicode.process(tl) == ''

    def test_multiline_returns_string(self):
        tokens = [
            make_token(T.Keyword, 'SEL'),
            make_token(T.Whitespace.Newline, '\n'),
            make_token(T.Keyword, 'ECT'),
        ]
        tl = make_tlist(tokens)
        result = SerializerUnicode.process(tl)
        assert isinstance(result, str)
        assert 'SEL' in result and 'ECT' in result
        assert result == 'SEL\nECT'
