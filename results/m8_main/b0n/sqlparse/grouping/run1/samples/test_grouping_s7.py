"""Unit tests for sqlparse.engine.grouping.

These tests exercise the public API of the module directly at the
token-list level.  A helper parses SQL into a flat token list (without
applying any grouping) so we have full control over what each grouping
function sees.
"""

import pytest

from sqlparse import sql, tokens as T
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
from sqlparse.tokens import Token as _Token  # noqa: F401  (namespace sanity)
from sqlparse.lexer import Lexer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def lex(sql_text):
    """Return a flat token list (no grouping) for the given SQL text."""
    return list(Lexer.get_default_instance().get_tokens(sql_text))


def values(tlist):
    return [tok.value for tok in tlist.tokens]


def classnames(tlist):
    return [type(tok).__name__ for tok in tlist.tokens]


# ---------------------------------------------------------------------------
# _group_matching based groupings
# ---------------------------------------------------------------------------

def test_group_brackets():
    tlist = lex("SELECT [a, b] FROM x")
    group_brackets(tlist)
    assert classnames(tlist) == [
        "Token", "Token", "SquareBrackets", "Token", "Token",
    ]
    assert values(tlist)[2] == "[a, b]"


def test_group_brackets_unbalanced():
    # A lone closing bracket is ignored (no exception raised).
    tlist = lex("a ] b")
    group_brackets(tlist)
    assert values(tlist) == ["a", " ", "]", " ", "b"]
    assert all(not isinstance(t, sql.SquareBrackets) for t in tlist.tokens)


def test_group_parenthesis():
    tlist = lex("foo(a, b)")
    group_parenthesis(tlist)
    assert any(isinstance(t, sql.Parenthesis) for t in tlist.tokens)
    parens = [t for t in tlist.tokens if isinstance(t, sql.Parenthesis)]
    assert len(parens) == 1
    assert parens[0].value == "(a, b)"


def test_group_parenthesis_nested():
    tlist = lex("(a (b) c)")
    group_parenthesis(tlist)
    outer = [t for t in tlist.tokens if isinstance(t, sql.Parenthesis)]
    assert len(outer) == 1
    assert outer[0].value == "(a (b) c)"


def test_group_parenthesis_extra_close():
    tlist = lex("(a))")
    group_parenthesis(tlist)
    parens = [t for t in tlist.tokens if isinstance(t, sql.Parenthesis)]
    assert len(parens) == 1
    assert parens[0].value == "(a)"


def test_group_case():
    tlist = lex("CASE WHEN a THEN b ELSE c END")
    group_case(tlist)
    cases = [t for t in tlist.tokens if isinstance(t, sql.Case)]
    assert len(cases) == 1
    assert cases[0].value == "CASE WHEN a THEN b ELSE c END"


def test_group_case_unbalanced():
    tlist = lex("CASE WHEN a THEN b")
    group_case(tlist)
    assert all(not isinstance(t, sql.Case) for t in tlist.tokens)


def test_group_if():
    tlist = lex("IF a THEN b END")
    group_if(tlist)
    ifs = [t for t in tlist.tokens if isinstance(t, sql.If)]
    assert len(ifs) == 1
    assert ifs[0].value == "IF a THEN b END"


def test_group_for():
    tlist = lex("FOR x IN a END")
    group_for(tlist)
    fors = [t for t in tlist.tokens if isinstance(t, sql.For)]
    assert len(fors) == 1
    assert fors[0].value == "FOR x IN a END"


def test_group_begin():
    tlist = lex("BEGIN SELECT 1 END")
    group_begin(tlist)
    begins = [t for t in tlist.tokens if isinstance(t, sql.Begin)]
    assert len(begins) == 1
    assert begins[0].value == "BEGIN SELECT 1 END"


# ---------------------------------------------------------------------------
# group_typecasts
# ---------------------------------------------------------------------------

def test_group_typecasts():
    tlist = lex("foo::int")
    group_typecasts(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1
    assert ids[0].value == "foo::int"


def test_group_typecasts_multiple():
    tlist = lex("a::int::text")
    group_typecasts(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) >= 1
    assert any(t.value == "a::int" for t in ids)


# ---------------------------------------------------------------------------
# group_tzcasts
# ---------------------------------------------------------------------------

def test_group_tzcasts():
    tlist = lex("UPDATE foo SET x = 1::timestamp AT TIME ZONE 'UTC'")
    group_tzcasts(tlist)
    # The AT TIME ZONE keyword should be grouped into an Identifier
    # with the datetime literal before it.
    found = any(
        isinstance(t, sql.Identifier) and "AT TIME ZONE" in t.value
        for t in tlist.tokens
    )
    assert found


# ---------------------------------------------------------------------------
# group_typed_literal
# ---------------------------------------------------------------------------

def test_group_typed_literal():
    tlist = lex("DATE '2020-01-01'")
    group_typed_literal(tlist)
    lits = [t for t in tlist.tokens if isinstance(t, sql.TypedLiteral)]
    assert len(lits) == 1
    assert lits[0].value == "DATE '2020-01-01'"


def test_group_typed_literal_with_tz():
    tlist = lex("TIMESTAMP '2020-01-01 00:00:00' AT TIME ZONE 'UTC'")
    group_typed_literal(tlist)
    lits = [t for t in tlist.tokens if isinstance(t, sql.TypedLiteral)]
    assert len(lits) >= 1
    assert "TIMESTAMP" in lits[0].value


# ---------------------------------------------------------------------------
# group_period
# ---------------------------------------------------------------------------

def test_group_period():
    tlist = lex("a.b")
    # Need identifiers for valid_prev
    group_identifier(tlist)
    group_period(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert any(t.value == "a.b" for t in ids)


def test_group_period_json_operator():
    tlist = lex("a -> b")
    group_identifier(tlist)
    group_period(tlist)
    # a -> b becomes an Identifier containing a.b-ish token.
    found = any(
        isinstance(t, sql.Identifier) and "a" in t.value and "->" in t.value
        for t in tlist.tokens
    )
    assert found


# ---------------------------------------------------------------------------
# group_as
# ---------------------------------------------------------------------------

def test_group_as():
    tlist = lex("a AS b")
    group_identifier(tlist)
    group_as(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert any(t.value == "a AS b" for t in ids)


def test_group_as_null_prev():
    tlist = lex("NULL AS b")
    group_as(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1
    assert ids[0].value == "NULL AS b"


# ---------------------------------------------------------------------------
# group_assignment
# ---------------------------------------------------------------------------

def test_group_assignment():
    tlist = lex("SET @a := 1")
    group_assignment(tlist)
    assigns = [t for t in tlist.tokens if isinstance(t, sql.Assignment)]
    assert len(assigns) == 1
    assert assigns[0].value == "@a := 1"


def test_group_assignment_stops_at_semicolon():
    tlist = lex("SET @a := 1;")
    group_assignment(tlist)
    assigns = [t for t in tlist.tokens if isinstance(t, sql.Assignment)]
    assert len(assigns) == 1
    # The semicolon should be outside the assignment.
    assert not assigns[0].value.rstrip().endswith(";")


# ---------------------------------------------------------------------------
# group_comparison
# ---------------------------------------------------------------------------

def test_group_comparison():
    tlist = lex("a = b")
    group_identifier(tlist)
    group_comparison(tlist)
    cmps = [t for t in tlist.tokens if isinstance(t, sql.Comparison)]
    assert len(cmps) == 1
    assert cmps[0].value == "a = b"


def test_group_comparison_null():
    tlist = lex("a IS NULL")
    group_identifier(tlist)
    group_comparison(tlist)
    cmps = [t for t in tlist.tokens if isinstance(t, sql.Comparison)]
    assert len(cmps) == 1
    assert "NULL" in cmps[0].value


# ---------------------------------------------------------------------------
# group_identifier
# ---------------------------------------------------------------------------

def test_group_identifier():
    tlist = lex("foo bar")
    group_identifier(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 2


def test_group_identifier_placeholder():
    tlist = lex("? ?")
    group_identifier(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 2


# ---------------------------------------------------------------------------
# group_over
# ---------------------------------------------------------------------------

def test_group_over():
    tlist = lex("ROW_NUMBER() OVER (PARTITION BY x)")
    group_parenthesis(tlist)
    group_functions(tlist)
    group_over(tlist)
    overs = [t for t in tlist.tokens if isinstance(t, sql.Over)]
    assert len(overs) == 1
    assert overs[0].value.startswith("OVER")


# ---------------------------------------------------------------------------
# group_arrays
# ---------------------------------------------------------------------------

def test_group_arrays():
    tlist = lex("[1, 2, 3]")
    group_brackets(tlist)
    group_arrays(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(ids) == 1
    assert ids[0].value == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# group_operator
# ---------------------------------------------------------------------------

def test_group_operator():
    tlist = lex("1 + 2")
    group_operator(tlist)
    ops = [t for t in tlist.tokens if isinstance(t, sql.Operation)]
    assert len(ops) == 1
    assert ops[0].value == "1 + 2"


def test_group_operator_current_date():
    tlist = lex("CURRENT_DATE + 1")
    group_operator(tlist)
    ops = [t for t in tlist.tokens if isinstance(t, sql.Operation)]
    assert len(ops) == 1
    assert "CURRENT_DATE" in ops[0].value


# ---------------------------------------------------------------------------
# group_identifier_list
# ---------------------------------------------------------------------------

def test_group_identifier_list():
    tlist = lex("a, b, c")
    group_identifier(tlist)
    group_identifier_list(tlist)
    lists = [t for t in tlist.tokens if isinstance(t, sql.IdentifierList)]
    assert len(lists) == 1
    assert lists[0].value == "a, b, c"


# ---------------------------------------------------------------------------
# group_comments
# ---------------------------------------------------------------------------

def test_group_comments():
    tlist = lex("SELECT 1 -- hi\n")
    group_comments(tlist)
    comments = [t for t in tlist.tokens if isinstance(t, sql.Comment)]
    assert len(comments) == 1
    assert "-- hi" in comments[0].value


def test_group_comments_no_trailer():
    # Comment ending at end-of-string with no subsequent token: function
    # must break out of the loop rather than raising or looping.
    tlist = lex("-- hi")
    group_comments(tlist)
    comments = [t for t in tlist.tokens if isinstance(t, sql.Comment)]
    # Behaviour: no grouping possible; no exception raised.
    assert isinstance(comments, list)


# ---------------------------------------------------------------------------
# group_where
# ---------------------------------------------------------------------------

def test_group_where():
    tlist = lex("WHERE a = 1")
    group_identifier(tlist)
    group_operator(tlist)
    group_where(tlist)
    wheres = [t for t in tlist.tokens if isinstance(t, sql.Where)]
    assert len(wheres) == 1
    assert wheres[0].value == "WHERE a = 1"


def test_group_where_no_terminator():
    tlist = lex("SELECT * FROM t WHERE a = 1")
    group_identifier(tlist)
    group_where(tlist)
    wheres = [t for t in tlist.tokens if isinstance(t, sql.Where)]
    assert len(wheres) == 1


# ---------------------------------------------------------------------------
# group_aliased
# ---------------------------------------------------------------------------

def test_group_aliased():
    tlist = lex("1 foo")
    group_identifier(tlist)
    group_aliased(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert any("1" in t.value and "foo" in t.value for t in ids)


# ---------------------------------------------------------------------------
# group_functions
# ---------------------------------------------------------------------------

def test_group_functions():
    tlist = lex("COUNT(* )")
    group_parenthesis(tlist)
    group_functions(tlist)
    funcs = [t for t in tlist.tokens if isinstance(t, sql.Function)]
    assert len(funcs) == 1
    assert funcs[0].value.startswith("COUNT")


def test_group_functions_create_table_skipped():
    tlist = lex("CREATE TABLE foo (a INT)")
    group_parenthesis(tlist)
    group_functions(tlist)
    # Function grouping should be skipped for CREATE TABLE ... (no AS).
    funcs = [t for t in tlist.tokens if isinstance(t, sql.Function)]
    assert len(funcs) == 0


def test_group_functions_with_over():
    tlist = lex("SUM(x) OVER (PARTITION BY y)")
    group_parenthesis(tlist)
    group_over(tlist)
    group_functions(tlist)
    funcs = [t for t in tlist.tokens if isinstance(t, sql.Function)]
    assert len(funcs) == 1
    assert "OVER" in funcs[0].value


# ---------------------------------------------------------------------------
# group_order
# ---------------------------------------------------------------------------

def test_group_order():
    tlist = lex("a ASC")
    group_identifier(tlist)
    group_order(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert any(t.value == "a ASC" for t in ids)


def test_group_order_number():
    tlist = lex("1 DESC")
    group_order(tlist)
    ids = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert any(t.value == "1 DESC" for t in ids)


# ---------------------------------------------------------------------------
# align_comments
# ---------------------------------------------------------------------------

def test_align_comments():
    tlist = lex("SELECT 1 -- c\n")
    group_comments(tlist)
    group_parenthesis(tlist)
    # Create a leading TokenList by first grouping some things.
    tlist2 = lex("SELECT -- c\n 1")
    group_comments(tlist2)
    align_comments(tlist2)
    # After alignment some comments should be inside TokenList.
    assert any(isinstance(t, sql.TokenList) for t in tlist2.tokens)


# ---------------------------------------------------------------------------
# group_values
# ---------------------------------------------------------------------------

def test_group_values():
    tlist = lex("VALUES (1), (2)")
    group_parenthesis(tlist)
    group_values(tlist)
    vals = [t for t in tlist.tokens if isinstance(t, sql.Values)]
    assert len(vals) == 1
    assert vals[0].value.startswith("VALUES")


def test_group_values_no_parenthesis():
    tlist = lex("VALUES 1")
    group_values(tlist)
    vals = [t for t in tlist.tokens if isinstance(t, sql.Values)]
    assert vals == []


# ---------------------------------------------------------------------------
# DoS guard tests
# ---------------------------------------------------------------------------

def test_max_grouping_depth_exceeded(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", 0)
    tlist = lex("(a)")
    # _group_matching is called with depth=0, so 0 > 0 is False -> ok.
    # Force a nested call by nesting parentheses.
    tlist = lex("((a))")
    with pytest.raises(SQLParseError, match="Maximum grouping depth"):
        group_parenthesis(tlist)


def test_max_grouping_tokens_exceeded(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 1)
    tlist = lex("SELECT a FROM b")
    with pytest.raises(SQLParseError, match="Maximum number of tokens"):
        group_parenthesis(tlist)


def test_group_guard_depth_exceeded(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", 0)
    tlist = lex("(a)")
    with pytest.raises(SQLParseError, match="Maximum grouping depth"):
        grouping._group(tlist, sql.Identifier, lambda t: False)


def test_group_guard_tokens_exceeded(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", 1)
    tlist = lex("SELECT a FROM b")
    with pytest.raises(SQLParseError, match="Maximum number of tokens"):
        grouping._group(tlist, sql.Identifier, lambda t: False)


def test_guards_disabled(monkeypatch):
    monkeypatch.setattr(grouping, "MAX_GROUPING_DEPTH", None)
    monkeypatch.setattr(grouping, "MAX_GROUPING_TOKENS", None)
    tlist = lex("((((((a))))))")
    group_parenthesis(tlist)
    assert any(isinstance(t, sql.Parenthesis) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group() end-to-end
# ---------------------------------------------------------------------------

def test_group_returns_same_object():
    tlist = lex("SELECT a, b FROM t WHERE x = 1 ORDER BY a")
    result = group(tlist)
    assert result is tlist


def test_group_end_to_end():
    tlist = lex("SELECT COUNT(a) FROM foo WHERE a = 1")
    group(tlist)
    stmt_text = "".join(t.value for t in tlist.tokens)
    assert stmt_text == "SELECT COUNT(a) FROM foo WHERE a = 1"
    classes = {type(t).__name__ for t in tlist.tokens}
    assert "Function" in classes
    assert "Where" in classes


def test_group_pipeline_comment():
    tlist = lex("SELECT a -- note\nFROM t")
    group(tlist)
    assert any(isinstance(t, sql.Comment) for t in tlist.tokens)
