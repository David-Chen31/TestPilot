"""Unit tests for sqlparse.engine.grouping."""

import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse import lexer
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
from sqlparse.exceptions import SQLParseError


def _tok(ttype, value):
    from sqlparse.sql import Token
    return Token(ttype, value)


def _tokenlist(tokens):
    from sqlparse.sql import TokenList
    return TokenList(tokens)


def _flat(sql_text):
    """Lex a statement without grouping, returning a plain TokenList."""
    tokens = list(lexer.tokenize(sql_text))
    from sqlparse.sql import TokenList
    return TokenList(tokens)


# ---------------------------------------------------------------------------
# _group_matching-based functions
# ---------------------------------------------------------------------------

class TestGroupMatching:
    def test_group_brackets(self):
        tl = _flat("[a b]")
        group_brackets(tl)
        groups = [t for t in tl.tokens if isinstance(t, sql.SquareBrackets)]
        assert len(groups) == 1
        assert groups[0].value == "[a b]"

    def test_group_parenthesis(self):
        tl = _flat("(a b)")
        group_parenthesis(tl)
        groups = [t for t in tl.tokens if isinstance(t, sql.Parenthesis)]
        assert len(groups) == 1
        assert groups[0].value == "(a b)"

    def test_group_nested_parenthesis(self):
        tl = _flat("((a))")
        group_parenthesis(tl)
        outer = [t for t in tl.tokens if isinstance(t, sql.Parenthesis)]
        assert len(outer) == 1
        outer = outer[0]
        inner = [t for t in outer.tokens if isinstance(t, sql.Parenthesis)]
        assert len(inner) == 1

    def test_group_case(self):
        tl = _flat("CASE WHEN a THEN b ELSE c END")
        group_case(tl)
        cases = [t for t in tl.tokens if isinstance(t, sql.Case)]
        assert len(cases) == 1
        assert cases[0].value.startswith("CASE")

    def test_group_if(self):
        tl = _flat("IF a THEN b END")
        group_if(tl)
        groups = [t for t in tl.tokens if isinstance(t, sql.If)]
        assert len(groups) == 1

    def test_group_for(self):
        tl = _flat("FOR i IN 1..10 LOOP END LOOP")
        group_for(tl)
        groups = [t for t in tl.tokens if isinstance(t, sql.For)]
        assert len(groups) == 1

    def test_group_begin(self):
        tl = _flat("BEGIN SELECT 1; END")
        group_begin(tl)
        groups = [t for t in tl.tokens if isinstance(t, sql.Begin)]
        assert len(groups) == 1

    def test_unbalanced_close_ignored(self):
        tl = _flat(") (")
        group_parenthesis(tl)
        # No group created because first is close (unmatched), second open.
        groups = [t for t in tl.tokens if isinstance(t, sql.Parenthesis)]
        assert groups == []

    def test_empty(self):
        tl = _flat("")
        group_parenthesis(tl)
        assert tl.tokens == []


# ---------------------------------------------------------------------------
# _group_matching depth/token limits
# ---------------------------------------------------------------------------

class TestGroupMatchingLimits:
    def test_max_depth_exceeded(self, monkeypatch):
        monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", 1)
        tl = _flat("((()))")
        with pytest.raises(SQLParseError):
            grouping._group_matching(tl, sql.Parenthesis)

    def test_max_tokens_exceeded(self, monkeypatch):
        monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 2)
        tl = _flat("(a b c d)")
        with pytest.raises(SQLParseError):
            grouping._group_matching(tl, sql.Parenthesis)


# ---------------------------------------------------------------------------
# group_typecasts
# ---------------------------------------------------------------------------

class TestGroupTypecasts:
    def test_simple(self):
        tl = _flat("a::int")
        group_typecasts(tl)
        ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
        assert len(ids) == 1
        assert ids[0].value == "a::int"


# ---------------------------------------------------------------------------
# group_tzcasts
# ---------------------------------------------------------------------------

class TestGroupTzcasts:
    def test_with_as(self):
        tl = _flat("a AT TIME ZONE AS b")
        # ensure there is a TZCast token
        from sqlparse.sql import Token
        tz = Token(T.Keyword.TZCast, "AT TIME ZONE")
        # rebuild with tz inserted between 'a' and 'AS b'
        tl2 = _flat("a AS b")
        tl2.tokens.insert(1, tz)
        group_tzcasts(tl2)
        ids = [t for t in tl2.tokens if isinstance(t, sql.Identifier)]
        assert len(ids) >= 1
        # the tzcast should be part of a grouped identifier
        assert any(tz in ident.tokens for ident in ids if hasattr(ident, "tokens"))


# ---------------------------------------------------------------------------
# group_typed_literal
# ---------------------------------------------------------------------------

class TestGroupTypedLiteral:
    def test_simple(self):
        tl = _flat("DATE '2020-01-01'")
        group_typed_literal(tl)
        groups = [t for t in tl.tokens if isinstance(t, sql.TypedLiteral)]
        assert len(groups) == 1
        assert "DATE '2020-01-01'" in groups[0].value


# ---------------------------------------------------------------------------
# group_period
# ---------------------------------------------------------------------------

class TestGroupPeriod:
    def test_simple(self):
        tl = _flat("a.b")
        group_identifier(tl)
        group_period(tl)
        ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
        assert len(ids) == 1
        assert ids[0].value == "a.b"

    def test_no_valid_prev(self):
        tl = _flat("1.b")
        group_period(tl)
        ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
        assert ids == []


# ---------------------------------------------------------------------------
# group_as
# ---------------------------------------------------------------------------

class TestGroupAs:
    def test_simple(self):
        tl = _flat("a AS b")
        group_as(tl)
        ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
        assert len(ids) == 1
        assert ids[0].value == "a AS b"


# ---------------------------------------------------------------------------
# group_assignment
# ---------------------------------------------------------------------------

class TestGroupAssignment:
    def test_simple(self):
        tl = _flat("a := b")
        group_assignment(tl)
        assigns = [t for t in tl.tokens if isinstance(t, sql.Assignment)]
        assert len(assigns) == 1
        assert assigns[0].value == "a := b"


# ---------------------------------------------------------------------------
# group_comparison
# ---------------------------------------------------------------------------

class TestGroupComparison:
    def test_simple(self):
        tl = _flat("a = b")
        group_comparison(tl)
        cmps = [t for t in tl.tokens if isinstance(t, sql.Comparison)]
        assert len(cmps) == 1
        assert cmps[0].value == "a = b"

    def test_null(self):
        tl = _flat("a IS NULL")
        # 'IS' isn't the comparison ttype; use '='
        tl = _flat("a = NULL")
        group_comparison(tl)
        cmps = [t for t in tl.tokens if isinstance(t, sql.Comparison)]
        assert len(cmps) == 1


# ---------------------------------------------------------------------------
# group_identifier
# ---------------------------------------------------------------------------

class TestGroupIdentifier:
    def test_simple(self):
        tl = _flat("foo")
        group_identifier(tl)
        ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
        assert len(ids) == 1
        assert ids[0].value == "foo"


# ---------------------------------------------------------------------------
# group_over
# ---------------------------------------------------------------------------

class TestGroupOver:
    def test_simple(self):
        tl = _flat("OVER (PARTITION BY x)")
        group_parenthesis(tl)
        group_over(tl)
        overs = [t for t in tl.tokens if isinstance(t, sql.Over)]
        assert len(overs) == 1


# ---------------------------------------------------------------------------
# group_arrays
# ---------------------------------------------------------------------------

class TestGroupArrays:
    def test_simple(self):
        tl = _flat("a[1]")
        group_brackets(tl)
        group_identifier(tl)
        group_arrays(tl)
        ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
        assert len(ids) == 1
        assert ids[0].value == "a[1]"


# ---------------------------------------------------------------------------
# group_operator
# ---------------------------------------------------------------------------

class TestGroupOperator:
    def test_simple(self):
        tl = _flat("1 + 2")
        group_operator(tl)
        ops = [t for t in tl.tokens if isinstance(t, sql.Operation)]
        assert len(ops) == 1
        assert ops[0].value == "1 + 2"

    def test_current_date(self):
        tl = _flat("CURRENT_DATE + 1")
        group_operator(tl)
        ops = [t for t in tl.tokens if isinstance(t, sql.Operation)]
        assert len(ops) == 1


# ---------------------------------------------------------------------------
# group_identifier_list
# ---------------------------------------------------------------------------

class TestGroupIdentifierList:
    def test_simple(self):
        tl = _flat("a, b, c")
        group_identifier(tl)
        group_identifier_list(tl)
        lists = [t for t in tl.tokens if isinstance(t, sql.IdentifierList)]
        assert len(lists) == 1


# ---------------------------------------------------------------------------
# group_comments
# ---------------------------------------------------------------------------

class TestGroupComments:
    def test_simple(self):
        tl = _flat("SELECT 1 -- comment\nFROM t")
        group_comments(tl)
        comments = [t for t in tl.tokens if isinstance(t, sql.Comment)]
        assert len(comments) == 1
        assert comments[0].value.startswith("-- comment")

    def test_only_comment(self):
        tl = _flat("-- comment")
        group_comments(tl)
        comments = [t for t in tl.tokens if isinstance(t, sql.Comment)]
        # No terminator -> break, no group
        assert comments == []


# ---------------------------------------------------------------------------
# group_where
# ---------------------------------------------------------------------------

class TestGroupWhere:
    def test_simple(self):
        tl = _flat("SELECT 1 WHERE a = 1")
        group_comparison(tl)
        group_where(tl)
        wheres = [t for t in tl.tokens if isinstance(t, sql.Where)]
        assert len(wheres) == 1
        assert wheres[0].value.startswith("WHERE")

    def test_no_terminator(self):
        tl = _flat("WHERE a = 1")
        group_comparison(tl)
        group_where(tl)
        wheres = [t for t in tl.tokens if isinstance(t, sql.Where)]
        assert len(wheres) == 1


# ---------------------------------------------------------------------------
# group_aliased
# ---------------------------------------------------------------------------

class TestGroupAliased:
    def test_simple(self):
        tl = _flat("(a) b")
        group_parenthesis(tl)
        group_identifier(tl)
        group_aliased(tl)
        ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
        # (a) and b merged
        assert len(ids) >= 1
        assert any(ids_entry.value == "(a) b" for ids_entry in ids)


# ---------------------------------------------------------------------------
# group_functions
# ---------------------------------------------------------------------------

class TestGroupFunctions:
    def test_simple(self):
        tl = _flat("COUNT(a)")
        group_parenthesis(tl)
        group_functions(tl)
        funcs = [t for t in tl.tokens if isinstance(t, sql.Function)]
        assert len(funcs) == 1
        assert funcs[0].value == "COUNT(a)"

    def test_create_table_skipped(self):
        tl = _flat("CREATE TABLE t")
        group_functions(tl)
        funcs = [t for t in tl.tokens if isinstance(t, sql.Function)]
        assert funcs == []

    def test_with_over(self):
        tl = _flat("COUNT(a) OVER (PARTITION BY x)")
        group_parenthesis(tl)
        group_over(tl)
        group_functions(tl)
        funcs = [t for t in tl.tokens if isinstance(t, sql.Function)]
        assert len(funcs) == 1
        assert "OVER" in funcs[0].value


# ---------------------------------------------------------------------------
# group_order
# ---------------------------------------------------------------------------

class TestGroupOrder:
    def test_simple(self):
        tl = _flat("a ASC")
        group_identifier(tl)
        group_order(tl)
        ids = [t for t in tl.tokens if isinstance(t, sql.Identifier)]
        assert len(ids) == 1
        assert ids[0].value == "a ASC"

    def test_without_identifier(self):
        tl = _flat("ASC")
        group_order(tl)
        # No grouping performed
        assert all(not isinstance(t, sql.Identifier) for t in tl.tokens)


# ---------------------------------------------------------------------------
# align_comments
# ---------------------------------------------------------------------------

class TestAlignComments:
    def test_simple(self):
        # Build list where a grouped token precedes a Comment group.
        from sqlparse.sql import Token
        ident = sql.Identifier([Token(T.Name, "x")])
        comment = sql.Comment([Token(T.Comment.Single, "-- hi")])
        tl = _tokenlist([ident, comment])
        align_comments(tl)
        # The ident + comment should be wrapped in a TokenList
        assert isinstance(tl.tokens[0], sql.TokenList)


# ---------------------------------------------------------------------------
# group_values
# ---------------------------------------------------------------------------

class TestGroupValues:
    def test_simple(self):
        tl = _flat("VALUES (1), (2)")
        group_parenthesis(tl)
        group_values(tl)
        vals = [t for t in tl.tokens if isinstance(t, sql.Values)]
        assert len(vals) == 1
        assert "VALUES" in vals[0].value


# ---------------------------------------------------------------------------
# _group depth / token limits
# ---------------------------------------------------------------------------

class TestGroupLimits:
    def test_group_depth_exceeded(self, monkeypatch):
        monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", 0)
        tl = _flat("a = b")
        with pytest.raises(SQLParseError):
            grouping._group(tl, sql.Comparison,
                            lambda t: t.ttype == T.Operator.Comparison,
                            depth=1)

    def test_group_tokens_exceeded(self, monkeypatch):
        monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 2)
        tl = _flat("a = b = c = d")
        with pytest.raises(SQLParseError):
            grouping._group(tl, sql.Comparison,
                            lambda t: t.ttype == T.Operator.Comparison)


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------

class TestGroupPipeline:
    def test_returns_same_object(self):
        tl = _flat("SELECT COUNT(a) FROM t WHERE a = 1")
        result = group(tl)
        assert result is tl

    def test_no_exception_on_basic_sql(self):
        tl = _flat("SELECT a, b AS c FROM t WHERE x = 1 ORDER BY a ASC")
        result = group(tl)
        assert isinstance(result, sql.TokenList)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_t_numerical(self):
        assert T.Number in grouping.T_NUMERICAL
        assert T.Number.Integer in grouping.T_NUMERICAL
        assert T.Number.Float in grouping.T_NUMERICAL

    def test_t_string(self):
        assert T.String in grouping.T_STRING
        assert T.String.Single in grouping.T_STRING
        assert T.String.Symbol in grouping.T_STRING

    def test_t_name(self):
        assert T.Name in grouping.T_NAME
        assert T.Name.Placeholder in grouping.T_NAME
