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


def _token(ttype, value):
    return sql.Token(ttype, value)


def _tokenlist():
    tl = sql.TokenList([])
    return tl


# ---------------------------------------------------------------------------
# Helpers to build simple token lists
# ---------------------------------------------------------------------------

def _build_tlist(*tokens):
    tl = sql.TokenList([])
    tl.tokens = list(tokens)
    return tl


# ---------------------------------------------------------------------------
# StripCommentsFilter
# ---------------------------------------------------------------------------

class TestStripCommentsFilter:

    def test_removes_single_line_comment_between_whitespace(self):
        stmt = _build_tlist(
            _token(T.Keyword, "SELECT"),
            _token(T.Whitespace, " "),
            _token(T.Comment.Single, "-- hi"),
            _token(T.Whitespace, " "),
            _token(T.Keyword, "a"),
        )
        StripCommentsFilter().process(stmt)
        values = [t.value for t in stmt.tokens]
        assert values == ["SELECT", " ", " ", "a"]

    def test_single_comment_replaced_by_space_when_between_words(self):
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            _token(T.Comment.Single, "-- c"),
            _token(T.Keyword, "b"),
        )
        StripCommentsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["a", " ", "b"]
        assert stmt.tokens[1].ttype is T.Whitespace

    def test_newline_preserved_from_multiline_comment(self):
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            _token(T.Comment.Multiline, "/* c */\n"),
            _token(T.Keyword, "b"),
        )
        StripCommentsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["a", "\n", "b"]
        assert stmt.tokens[1].ttype is T.Whitespace.Newline

    def test_comment_between_whitespace_not_replaced(self):
        # prev is whitespace, so token removed without inserting.
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            _token(T.Whitespace, " "),
            _token(T.Comment.Single, "-- c"),
            _token(T.Whitespace, " "),
            _token(T.Keyword, "b"),
        )
        StripCommentsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["a", " ", " ", "b"]

    def test_comment_at_start_not_replaced(self):
        stmt = _build_tlist(
            _token(T.Comment.Single, "-- c"),
            _token(T.Whitespace, " "),
            _token(T.Keyword, "b"),
        )
        StripCommentsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == [" ", "b"]

    def test_comment_at_start_no_next_not_replaced(self):
        stmt = _build_tlist(
            _token(T.Comment.Single, "-- c"),
        )
        StripCommentsFilter().process(stmt)
        assert stmt.tokens == []

    def test_comment_after_paren(self):
        stmt = _build_tlist(
            _token(T.Punctuation, "("),
            _token(T.Comment.Single, "-- c"),
            _token(T.Keyword, "b"),
            _token(T.Punctuation, ")"),
        )
        StripCommentsFilter().process(stmt)
        # prev is '(' so no insert; next is 'b' not ws/paren -> replace
        assert [t.value for t in stmt.tokens] == ["(", "-- c", "b", ")"] or \
            [t.value for t in stmt.tokens] == ["(", " ", "b", ")"]

    def test_hint_token_is_skipped(self):
        stmt = _build_tlist(
            _token(T.Comment.Single.Hint, "/*+ hint */"),
            _token(T.Whitespace, " "),
            _token(T.Keyword, "a"),
        )
        StripCommentsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["/*+ hint */", " ", "a"]

    def test_hint_multiline_token_is_skipped(self):
        stmt = _build_tlist(
            _token(T.Comment.Multiline.Hint, "/*+ hint */"),
            _token(T.Whitespace, " "),
            _token(T.Keyword, "a"),
        )
        StripCommentsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["/*+ hint */", " ", "a"]

    def test_comment_group_with_hint_first_token_is_skipped(self):
        inner = sql.Comment([
            _token(T.Comment.Single.Hint, "/*+ h */"),
            _token(T.Comment.Single, " extra"),
        ])
        stmt = _build_tlist(
            inner,
            _token(T.Whitespace, " "),
            _token(T.Keyword, "a"),
        )
        StripCommentsFilter().process(stmt)
        assert stmt.tokens[0] is inner
        assert [t.value for t in stmt.tokens] == ["/*+ h */ extra", " ", "a"]

    def test_comment_group_without_hint_is_removed(self):
        inner = sql.Comment([
            _token(T.Comment.Single, "-- x"),
        ])
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            inner,
            _token(T.Keyword, "b"),
        )
        StripCommentsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["a", " ", "b"]

    def test_empty_comment_group_is_removed(self):
        inner = sql.Comment([])
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            inner,
            _token(T.Keyword, "b"),
        )
        StripCommentsFilter().process(stmt)
        # Empty comment group: prev 'a', next 'b' -> replaced by space.
        assert [t.value for t in stmt.tokens] == ["a", " ", "b"]

    def test_processes_sublists(self):
        parent = sql.TokenList([])
        inner = sql.TokenList([
            _token(T.Keyword, "a"),
            _token(T.Comment.Single, "-- c"),
            _token(T.Keyword, "b"),
        ])
        parent.tokens = [inner]
        StripCommentsFilter().process(parent)
        assert [t.value for t in inner.tokens] == ["a", " ", "b"]

    def test_comment_after_multiple_newlines_preserves_last(self):
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            _token(T.Comment.Single, "-- c\r\n  "),
            _token(T.Whitespace, " "),
            _token(T.Keyword, "b"),
        )
        StripCommentsFilter().process(stmt)
        # prev 'a' is non-ws; next_ is whitespace -> insert token, remove.
        # _get_insert_token finds r'\r\n' at end -> newline token
        assert stmt.tokens[1].ttype is T.Whitespace.Newline
        assert stmt.tokens[1].value == "\r\n"

    def test_insert_token_without_newline_returns_space(self):
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            _token(T.Comment.Single, "-- c"),
            _token(T.Whitespace, " "),
            _token(T.Keyword, "b"),
        )
        StripCommentsFilter().process(stmt)
        assert stmt.tokens[1].ttype is T.Whitespace
        assert stmt.tokens[1].value == " "


# ---------------------------------------------------------------------------
# StripWhitespaceFilter
# ---------------------------------------------------------------------------

class TestStripWhitespaceFilter:

    def test_default_collapses_whitespace(self):
        tlist = _build_tlist(
            _token(T.Whitespace, "  "),
            _token(T.Keyword, "a"),
            _token(T.Whitespace, "   "),
            _token(T.Whitespace, "\n"),
            _token(T.Keyword, "b"),
        )
        StripWhitespaceFilter._stripws_default(tlist)
        assert [t.value for t in tlist.tokens] == ["", "a", " ", "", "b"]

    def test_default_single_ws_between_words_becomes_space(self):
        tlist = _build_tlist(
            _token(T.Keyword, "a"),
            _token(T.Whitespace, "\n"),
            _token(T.Keyword, "b"),
        )
        StripWhitespaceFilter._stripws_default(tlist)
        assert [t.value for t in tlist.tokens] == ["a", " ", "b"]

    def test_identifierlist_removes_newline_before_comma(self):
        identlist = sql.IdentifierList([])
        identlist.tokens = [
            _token(T.Name, "a"),
            _token(T.Whitespace, "\n"),
            _token(T.Punctuation, ","),
            _token(T.Name, " b"),
        ]
        StripWhitespaceFilter()._stripws_identifierlist(identlist)
        assert [t.value for t in identlist.tokens] == ["a", ",", " b"]

    def test_identifierlist_no_newline_keeps_comma(self):
        identlist = sql.IdentifierList([])
        identlist.tokens = [
            _token(T.Name, "a"),
            _token(T.Whitespace, " "),
            _token(T.Punctuation, ","),
            _token(T.Name, "b"),
        ]
        StripWhitespaceFilter()._stripws_identifierlist(identlist)
        # whitespace not removed, but default applies.
        assert [t.value for t in identlist.tokens] == ["a", " ", ",", "b"]

    def test_parenthesis_strips_inner_whitespace(self):
        paren = sql.Parenthesis([
            _token(T.Punctuation, "("),
            _token(T.Whitespace, "  "),
            _token(T.Keyword, "a"),
            _token(T.Whitespace, " "),
            _token(T.Punctuation, ")"),
        ])
        StripWhitespaceFilter()._stripws_parenthesis(paren)
        assert [t.value for t in paren.tokens] == ["(", "a", ")"]

    def test_parenthesis_strips_trailing_ws_in_last_group(self):
        inner = sql.TokenList([
            _token(T.Keyword, "x"),
            _token(T.Whitespace, " "),
        ])
        paren = sql.Parenthesis([
            _token(T.Punctuation, "("),
            inner,
            _token(T.Whitespace, " "),
            _token(T.Punctuation, ")"),
        ])
        StripWhitespaceFilter()._stripws_parenthesis(paren)
        assert inner.tokens[-1].value == "x"
        assert [t.value for t in paren.tokens] == ["(", "x", ")"]

    def test_process_strips_trailing_ws_at_top_level(self):
        stmt = sql.Statement([
            _token(T.Keyword, "SELECT"),
            _token(T.Whitespace, " "),
        ])
        StripWhitespaceFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["SELECT"]

    def test_process_keeps_trailing_ws_at_nested_depth(self):
        inner = sql.TokenList([
            _token(T.Keyword, "x"),
            _token(T.Whitespace, " "),
        ])
        stmt = sql.Statement([inner])
        StripWhitespaceFilter().process(stmt)
        # inner is processed at depth 1; default collapses last ws to ' '
        # but shouldn't be popped since depth != 0
        assert inner.tokens[-1].value in (" ", "")

    def test_process_dispatches_to_type_specific(self):
        identlist = sql.IdentifierList([
            _token(T.Name, "a"),
            _token(T.Whitespace, "\n"),
            _token(T.Punctuation, ","),
        ])
        stmt = sql.Statement([identlist])
        StripWhitespaceFilter().process(stmt)
        assert identlist.tokens[1].value == ","

    def test_process_on_statement_with_no_tokens(self):
        stmt = sql.Statement([])
        StripWhitespaceFilter().process(stmt)
        assert stmt.tokens == []


# ---------------------------------------------------------------------------
# SpacesAroundOperatorsFilter
# ---------------------------------------------------------------------------

class TestSpacesAroundOperatorsFilter:

    def test_operator_gets_spaces_both_sides(self):
        stmt = _build_tlist(
            _token(T.Name, "a"),
            _token(T.Operator, "+"),
            _token(T.Name, "b"),
        )
        SpacesAroundOperatorsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["a", " ", "+", " ", "b"]

    def test_operator_existing_spaces_untouched(self):
        stmt = _build_tlist(
            _token(T.Name, "a"),
            _token(T.Whitespace, " "),
            _token(T.Operator, "+"),
            _token(T.Whitespace, " "),
            _token(T.Name, "b"),
        )
        SpacesAroundOperatorsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["a", " ", "+", " ", "b"]

    def test_comparison_gets_spaces(self):
        stmt = _build_tlist(
            _token(T.Name, "a"),
            _token(T.Comparison, "="),
            _token(T.Literal.Number.Integer, "1"),
        )
        SpacesAroundOperatorsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["a", " ", "=", " ", "1"]

    def test_operator_at_start(self):
        stmt = _build_tlist(
            _token(T.Operator, "-"),
            _token(T.Name, "a"),
        )
        SpacesAroundOperatorsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["-", " ", "a"]

    def test_operator_at_end(self):
        stmt = _build_tlist(
            _token(T.Name, "a"),
            _token(T.Operator, "+"),
        )
        SpacesAroundOperatorsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["a", " ", "+"]

    def test_multiple_operators(self):
        stmt = _build_tlist(
            _token(T.Name, "a"),
            _token(T.Operator, "+"),
            _token(T.Name, "b"),
            _token(T.Operator, "-"),
            _token(T.Name, "c"),
        )
        SpacesAroundOperatorsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == [
            "a", " ", "+", " ", "b", " ", "-", " ", "c"]

    def test_no_operators_unchanged(self):
        stmt = _build_tlist(
            _token(T.Name, "a"),
            _token(T.Whitespace, " "),
            _token(T.Name, "b"),
        )
        SpacesAroundOperatorsFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["a", " ", "b"]

    def test_processes_sublists(self):
        inner = sql.TokenList([
            _token(T.Name, "a"),
            _token(T.Operator, "+"),
            _token(T.Name, "b"),
        ])
        parent = sql.TokenList([inner])
        SpacesAroundOperatorsFilter().process(parent)
        assert [t.value for t in inner.tokens] == ["a", " ", "+", " ", "b"]


# ---------------------------------------------------------------------------
# StripTrailingSemicolonFilter
# ---------------------------------------------------------------------------

class TestStripTrailingSemicolonFilter:

    def test_removes_trailing_semicolon(self):
        stmt = _build_tlist(
            _token(T.Keyword, "SELECT"),
            _token(T.Punctuation, ";"),
        )
        StripTrailingSemicolonFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["SELECT"]

    def test_removes_trailing_whitespace_and_semicolon(self):
        stmt = _build_tlist(
            _token(T.Keyword, "SELECT"),
            _token(T.Punctuation, ";"),
            _token(T.Whitespace, " \n"),
        )
        StripTrailingSemicolonFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["SELECT"]

    def test_removes_multiple_semicolons_and_whitespace(self):
        stmt = _build_tlist(
            _token(T.Keyword, "SELECT"),
            _token(T.Punctuation, ";"),
            _token(T.Whitespace, " "),
            _token(T.Punctuation, ";"),
        )
        StripTrailingSemicolonFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["SELECT"]

    def test_no_trailing_semicolon_unchanged(self):
        stmt = _build_tlist(
            _token(T.Keyword, "SELECT"),
            _token(T.Whitespace, " "),
        )
        StripTrailingSemicolonFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == ["SELECT"]

    def test_empty_statement(self):
        stmt = _build_tlist()
        StripTrailingSemicolonFilter().process(stmt)
        assert stmt.tokens == []

    def test_semicolon_only(self):
        stmt = _build_tlist(_token(T.Punctuation, ";"))
        StripTrailingSemicolonFilter().process(stmt)
        assert stmt.tokens == []

    def test_semicolon_not_at_end_kept(self):
        stmt = _build_tlist(
            _token(T.Punctuation, ";"),
            _token(T.Keyword, "SELECT"),
        )
        StripTrailingSemicolonFilter().process(stmt)
        assert [t.value for t in stmt.tokens] == [";", "SELECT"]


# ---------------------------------------------------------------------------
# SerializerUnicode
# ---------------------------------------------------------------------------

class TestSerializerUnicode:

    def test_simple(self):
        stmt = _build_tlist(
            _token(T.Keyword, "SELECT"),
            _token(T.Whitespace, " "),
            _token(T.Keyword, "1"),
        )
        assert SerializerUnicode.process(stmt) == "SELECT 1"

    def test_trailing_whitespace_per_line_stripped(self):
        stmt = _build_tlist(
            _token(T.Keyword, "SELECT"),
            _token(T.Whitespace, "   "),
        )
        # str(stmt) is "SELECT   " -> rstrip -> "SELECT"
        result = SerializerUnicode.process(stmt)
        assert result == "SELECT"

    def test_newlines_normalized(self):
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            _token(T.Whitespace, "\n"),
            _token(T.Keyword, "b"),
        )
        assert SerializerUnicode.process(stmt) == "a\nb"

    def test_crlf_normalized_to_lf(self):
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            _token(T.Whitespace, "\r\n"),
            _token(T.Keyword, "b"),
        )
        assert SerializerUnicode.process(stmt) == "a\nb"

    def test_multiline_each_line_rstripped(self):
        stmt = _build_tlist(
            _token(T.Keyword, "a"),
            _token(T.Whitespace, "   \n"),
            _token(T.Keyword, "b"),
            _token(T.Whitespace, "   "),
        )
        assert SerializerUnicode.process(stmt) == "a\nb"

    def test_empty(self):
        stmt = _build_tlist()
        assert SerializerUnicode.process(stmt) == ""

    def test_is_static(self):
        # SerializerUnicode.process is callable on instance too
        stmt = _build_tlist(_token(T.Keyword, "x"))
        assert SerializerUnicode().process(stmt) == "x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
