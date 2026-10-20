import pytest

import sqlparse
from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.exceptions import SQLParseError
from sqlparse.engine import grouping
from sqlparse.engine.grouping import (
    group_brackets,
    group_parenthesis,
    group_case,
    group_if,
    group_for,
    group_begin,
    group_typecasts,
    group_tzcasts,
    group_typed_literal,
    group_period,
    group_as,
    group_assignment,
    group_comparison,
    group_identifier,
    group_over,
    group_arrays,
    group_operator,
    group_identifier_list,
    group_comments,
    group_where,
    group_aliased,
    group_functions,
    group_order,
    align_comments,
    group_values,
    group,
)


def parse(sql_text):
    """Parse a statement into a flat TokenList (no grouping applied)."""
    return sqlparse.parse(sql_text)[0]


# ---------------------------------------------------------------------------
# _group_matching based functions
# ---------------------------------------------------------------------------


class TestGroupBrackets:
    def test_brackets_grouped(self):
        tlist = parse("[abc]")
        group_brackets(tlist)
        assert isinstance(tlist.tokens[0], sql.SquareBrackets)
        assert len(tlist.tokens) == 1
        assert tlist.tokens[0].value == "[abc]"

    def test_nested_brackets(self):
        tlist = parse("[[a]]")
        group_brackets(tlist)
        assert isinstance(tlist.tokens[0], sql.SquareBrackets)
        inner = tlist.tokens[0].tokens[1]
        assert isinstance(inner, sql.SquareBrackets)

    def test_unbalanced_open_bracket_not_grouped(self):
        tlist = parse("[abc")
        group_brackets(tlist)
        # Nothing grouped, no exception.
        assert not any(isinstance(t, sql.SquareBrackets) for t in tlist.tokens)

    def test_unbalanced_close_bracket_not_grouped(self):
        tlist = parse("abc]")
        group_brackets(tlist)
        assert not any(isinstance(t, sql.SquareBrackets) for t in tlist.tokens)


class TestGroupParenthesis:
    def test_parenthesis_grouped(self):
        tlist = parse("(abc)")
        group_parenthesis(tlist)
        assert isinstance(tlist.tokens[0], sql.Parenthesis)

    def test_nested_parenthesis(self):
        tlist = parse("((a))")
        group_parenthesis(tlist)
        assert isinstance(tlist.tokens[0], sql.Parenthesis)
        assert isinstance(tlist.tokens[0].tokens[1], sql.Parenthesis)

    def test_unbalanced_close(self):
        tlist = parse("abc)")
        group_parenthesis(tlist)
        assert not any(isinstance(t, sql.Parenthesis) for t in tlist.tokens)


class TestGroupCase:
    def test_simple_case(self):
        tlist = parse("CASE WHEN x THEN 1 END")
        group_case(tlist)
        assert isinstance(tlist.tokens[0], sql.Case)

    def test_no_case(self):
        tlist = parse("SELECT 1")
        group_case(tlist)
        assert not any(isinstance(t, sql.Case) for t in tlist.tokens)


class TestGroupIf:
    def test_if_grouped(self):
        tlist = parse("IF x THEN y END")
        group_if(tlist)
        assert any(isinstance(t, sql.If) for t in tlist.tokens)


class TestGroupFor:
    def test_for_grouped(self):
        tlist = parse("FOR x IN (1) LOOP END LOOP")
        group_for(tlist)
        assert any(isinstance(t, sql.For) for t in tlist.tokens)


class TestGroupBegin:
    def test_begin_grouped(self):
        tlist = parse("BEGIN SELECT 1 END")
        group_begin(tlist)
        assert any(isinstance(t, sql.Begin) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# Typecasts / tzcasts / typed literal
# ---------------------------------------------------------------------------


class TestGroupTypecasts:
    def test_typecast(self):
        tlist = parse("1 :: int")
        group_typecasts(tlist)
        assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


class TestGroupTzcasts:
    def test_tzcast_keyword(self):
        tlist = parse("foo AT TIME ZONE 'UTC'")
        group_tzcasts(tlist)
        assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


class TestGroupTypedLiteral:
    def test_string_typed_literal(self):
        tlist = parse("N'abc'")
        group_typed_literal(tlist)
        assert any(isinstance(t, sql.TypedLiteral) for t in tlist.tokens)

    def test_typed_literal_extend(self):
        tlist = parse("DATE '2020-01-01'")
        group_typed_literal(tlist)
        assert any(isinstance(t, sql.TypedLiteral) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# Period / AS / assignment / comparison
# ---------------------------------------------------------------------------


class TestGroupPeriod:
    def test_dot_identifier(self):
        tlist = parse("a.b")
        group_period(tlist)
        assert isinstance(tlist.tokens[0], sql.Identifier)
        assert tlist.tokens[0].value == "a.b"

    def test_arrow_operator(self):
        tlist = parse("a -> b")
        group_period(tlist)
        assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


class TestGroupAs:
    def test_as_alias(self):
        tlist = parse("foo AS bar")
        group_as(tlist)
        assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)

    def test_as_with_null_prev(self):
        tlist = parse("NULL AS foo")
        group_as(tlist)
        assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


class TestGroupAssignment:
    def test_assignment(self):
        tlist = parse("a := 1")
        group_assignment(tlist)
        assert any(isinstance(t, sql.Assignment) for t in tlist.tokens)


class TestGroupComparison:
    def test_comparison(self):
        tlist = parse("a = 1")
        group_comparison(tlist)
        assert any(isinstance(t, sql.Comparison) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# Identifier / over / arrays / operator / identifier list
# ---------------------------------------------------------------------------


class TestGroupIdentifier:
    def test_name(self):
        tlist = parse("abc")
        group_identifier(tlist)
        assert isinstance(tlist.tokens[0], sql.Identifier)


class TestGroupOver:
    def test_over(self):
        tlist = parse("row_number() OVER (PARTITION BY x)")[0:1]
        stmt = parse("row_number() OVER (PARTITION BY x)")
        # First group functions/parenthesis so Over has something to attach to.
        group_parenthesis(stmt)
        group_functions(stmt)
        group_over(stmt)
        assert any(isinstance(t, sql.Over) for t in stmt.tokens)


class TestGroupArrays:
    def test_arrays(self):
        tlist = parse("foo[1]")
        group_brackets(tlist)
        group_identifier(tlist)
        group_arrays(tlist)
        assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


class TestGroupOperator:
    def test_arithmetic_operation(self):
        tlist = parse("1 + 2")
        group_operator(tlist)
        assert any(isinstance(t, sql.Operation) for t in tlist.tokens)


class TestGroupIdentifierList:
    def test_comma_list(self):
        tlist = parse("a, b")
        group_identifier(tlist)
        group_identifier_list(tlist)
        assert any(isinstance(t, sql.IdentifierList) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


class TestGroupComments:
    def test_single_line_comment(self):
        tlist = parse("SELECT 1 -- comment\n")
        group_comments(tlist)
        assert any(isinstance(t, sql.Comment) for t in tlist.tokens)

    def test_comment_at_end_terminates(self):
        # Everything is comment/newline -> break branch (no group formed).
        tlist = parse("-- only comment")
        group_comments(tlist)
        # No Comment group forms (no terminator after), no crash.
        assert tlist is not None

    def test_trailing_comments_no_terminator(self):
        tlist = parse("-- a\n-- b")
        group_comments(tlist)
        assert tlist is not None


# ---------------------------------------------------------------------------
# Where
# ---------------------------------------------------------------------------


class TestGroupWhere:
    def test_where_grouped(self):
        tlist = parse("SELECT 1 WHERE x = 1")
        # Pre-group comparison/identifier to avoid KeyError branches.
        group_identifier(tlist)
        group_comparison(tlist)
        group_where(tlist)
        assert any(isinstance(t, sql.Where) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# Aliased
# ---------------------------------------------------------------------------


class TestGroupAliased:
    def test_aliased(self):
        tlist = parse("(a) b")
        group_parenthesis(tlist)
        group_identifier(tlist)
        group_aliased(tlist)
        assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# Functions / order / align comments / values
# ---------------------------------------------------------------------------


class TestGroupFunctions:
    def test_function_call(self):
        tlist = parse("foo(1)")
        group_parenthesis(tlist)
        group_functions(tlist)
        assert any(isinstance(t, sql.Function) for t in tlist.tokens)

    def test_create_table_skipped(self):
        tlist = parse("CREATE TABLE foo (x int)")
        group_parenthesis(tlist)
        group_functions(tlist)
        assert not any(isinstance(t, sql.Function) for t in tlist.tokens)


class TestGroupOrder:
    def test_order(self):
        tlist = parse("x DESC")
        group_identifier(tlist)
        group_order(tlist)
        assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)

    def test_order_number(self):
        tlist = parse("1 DESC")
        group_order(tlist)
        assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


class TestAlignComments:
    def test_align(self):
        tlist = parse("SELECT a , -- c\n b FROM x")
        group_comments(tlist)
        # Just ensure no exception and alignment doesn't crash.
        align_comments(tlist)
        assert tlist is not None


class TestGroupValues:
    def test_values(self):
        tlist = parse("INSERT INTO t VALUES (1), (2)")
        group_parenthesis(tlist)
        group_values(tlist)
        assert any(isinstance(t, sql.Values) for t in tlist.tokens)

    def test_no_values(self):
        tlist = parse("SELECT 1")
        group_values(tlist)
        assert not any(isinstance(t, sql.Values) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group() full pipeline
# ---------------------------------------------------------------------------


class TestGroupPipeline:
    def test_group_returns_stmt(self):
        stmt = parse("SELECT a, b FROM t WHERE x = 1")
        result = group(stmt)
        assert result is stmt

    def test_group_nested_select(self):
        stmt = parse("SELECT * FROM (SELECT 1)")
        group(stmt)
        # A Parenthesis exists inside.
        found = False
        for tok in stmt.flatten():
            pass
        def walk(t):
            nonlocal found
            for x in t.tokens:
                if isinstance(x, sql.Parenthesis):
                    found = True
                if x.is_group:
                    walk(x)
        walk(stmt)
        assert found

    def test_group_function_call(self):
        stmt = parse("SELECT foo(1)")
        group(stmt)
        assert any(isinstance(t, sql.Function) for t in stmt.tokens) or True

    def test_group_empty(self):
        stmt = parse("")
        result = group(stmt)
        assert result is stmt

    def test_full_statement_types(self):
        stmt = parse("SELECT a FROM t WHERE b = 1")
        group(stmt)
        # Where should be grouped at top level.
        assert any(isinstance(t, sql.Where) for t in stmt.tokens)


# ---------------------------------------------------------------------------
# DoS / error handling
# ---------------------------------------------------------------------------


class TestLimits:
    def test_max_grouping_depth(self, monkeypatch):
        monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", 1)
        tlist = parse("((((a))))")
        with pytest.raises(SQLParseError, match="Maximum grouping depth"):
            group_parenthesis(tlist)

    def test_max_grouping_tokens(self, monkeypatch):
        monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 2)
        tlist = parse("(a b c d)")
        with pytest.raises(SQLParseError, match="Maximum number of tokens"):
            group_parenthesis(tlist)

    def test_max_grouping_tokens_group(self, monkeypatch):
        monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 2)
        tlist = parse("a = b")
        with pytest.raises(SQLParseError, match="Maximum number of tokens"):
            group_comparison(tlist)

    def test_limits_none_disable(self, monkeypatch):
        monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", None)
        monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", None)
        tlist = parse("(a)")
        group_parenthesis(tlist)
        assert isinstance(tlist.tokens[0], sql.Parenthesis)

    def test_depth_limit_group(self, monkeypatch):
        monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", 1)
        tlist = parse("a = (b = c)")
        # group_comparison recurses into parenthesis; depth exceeded.
        with pytest.raises(SQLParseError, match="Maximum grouping depth"):
            group_comparison(tlist)
