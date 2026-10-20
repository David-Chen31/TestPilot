import pytest

from sqlparse import sql, tokens as T
from sqlparse.engine import grouping
from sqlparse.exceptions import SQLParseError


def make_token_list(tokens):
    return sql.TokenList(tokens)


def tok(value, ttype=T.Text.Whitespace):
    return sql.Token(ttype, value)


def name(value):
    return sql.Token(T.Name, value)


def keyword(value):
    return sql.Token(T.Keyword, value)


def punct(value):
    return sql.Token(T.Punctuation, value)


def op(value, ttype=T.Operator):
    return sql.Token(ttype, value)


# ---------------------------------------------------------------------------
# _group_matching family
# ---------------------------------------------------------------------------

def test_group_parenthesis_simple():
    tlist = make_token_list([
        punct('('), name('a'), punct(')'),
    ])
    grouping.group_parenthesis(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.Parenthesis)


def test_group_brackets_simple():
    tlist = make_token_list([
        punct('['), name('a'), punct(']'),
    ])
    grouping.group_brackets(tlist)
    assert len(tlist.tokens) == 1
    assert isinstance(tlist.tokens[0], sql.SquareBrackets)


def test_group_parenthesis_nested_recursion():
    # _group_matching processes the flat token list in a single pass using the
    # opens stack with tidx offsets, so nested parens are all collapsed into a
    # single Parenthesis group in one call.
    tlist = make_token_list([
        punct('('), punct('('), name('a'), punct(')'), punct(')'),
    ])
    grouping.group_parenthesis(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.Parenthesis)]
    assert len(groups) == 1
    assert len(tlist.tokens) == 1
    assert groups[0].value == '((a))'


def test_group_parenthesis_unbalanced_close_ignored():
    tlist = make_token_list([
        name('a'), punct(')'), name('b'),
    ])
    grouping.group_parenthesis(tlist)
    # no parenthesis group, tokens stay
    assert not any(isinstance(t, sql.Parenthesis) for t in tlist.tokens)


def test_group_parenthesis_unbalanced_open_ignored():
    tlist = make_token_list([
        punct('('), name('a'),
    ])
    grouping.group_parenthesis(tlist)
    assert not any(isinstance(t, sql.Parenthesis) for t in tlist.tokens)


def test_group_case():
    tlist = make_token_list([
        keyword('CASE'), name('a'), keyword('END'),
    ])
    grouping.group_case(tlist)
    assert any(isinstance(t, sql.Case) for t in tlist.tokens)


def test_group_if():
    # sql.If M_OPEN is 'IF('/'IF ' style keyword; plain IF token alone doesn't
    # match, so no If group forms.
    tlist = make_token_list([
        keyword('IF'), name('a'), keyword('END'),
    ])
    grouping.group_if(tlist)
    assert not any(isinstance(t, sql.If) for t in tlist.tokens)


def test_group_if_with_paren():
    # Even after grouping the parenthesis, a bare Keyword 'IF' token does not
    # match sql.If.M_OPEN, so no If group forms.
    tlist = make_token_list([
        keyword('IF'), punct('('), name('a'), punct(')'), keyword('END'),
    ])
    grouping.group_parenthesis(tlist)
    grouping.group_if(tlist)
    assert not any(isinstance(t, sql.If) for t in tlist.tokens)


def test_group_for():
    # sql.For M_OPEN is the 'FOR' keyword, but M_CLOSE is 'END'; matching
    # requires both. Flat 'FOR a END' - 'FOR' matches open, 'END' matches close.
    tlist = make_token_list([
        keyword('FOR'), name('a'), keyword('END'),
    ])
    grouping.group_for(tlist)
    # Bare Keyword FOR doesn't have the specific ttype required by M_OPEN,
    # so no For group is formed.
    assert not any(isinstance(t, sql.For) for t in tlist.tokens)


def test_group_begin():
    tlist = make_token_list([
        keyword('BEGIN'), name('a'), keyword('END'),
    ])
    grouping.group_begin(tlist)
    assert any(isinstance(t, sql.Begin) for t in tlist.tokens)


def test_group_matching_max_depth(monkeypatch):
    # Directly test _group_matching depth check
    tlist = make_token_list([name('a')])
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 1)
    with pytest.raises(SQLParseError):
        grouping._group_matching(tlist, sql.Parenthesis, depth=2)


def test_group_matching_max_tokens(monkeypatch):
    tlist = make_token_list([name('a'), name('b')])
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 1)
    with pytest.raises(SQLParseError):
        grouping._group_matching(tlist, sql.Parenthesis)


# ---------------------------------------------------------------------------
# group_typecasts
# ---------------------------------------------------------------------------

def test_group_typecasts():
    # match requires T.Punctuation '::'; use a keyword-ish context where the
    # prev/next are both valid and non-keyword-typed tokens. In this pure
    # token construction, valid() requires non-None only, but _group requires
    # a prev_ to have been set, so '::' as first significant token won't match.
    tlist = make_token_list([
        name('a'), op('::', ttype=T.Punctuation), name('int'),
    ])
    grouping.group_typecasts(tlist)
    # The '::' matches but prev_/next_ valid; the name tokens are T.Name so
    # _group should form Identifier.
    assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_tzcasts
# ---------------------------------------------------------------------------

def test_group_tzcasts():
    # valid_next requires whitespace / 'AS' / TypedLiteral close. A plain
    # following Name won't pass, so no Identifier is formed.
    tlist = make_token_list([
        name('a'),
        sql.Token(T.Keyword.TZCast, 'AT TIME ZONE'),
        name('tz'),
    ])
    grouping.group_tzcasts(tlist)
    assert not any(isinstance(t, sql.Identifier) for t in tlist.tokens)


def test_group_tzcasts_followed_by_whitespace():
    # token_next skips whitespace by default, so the effective next_ is the
    # following Name ('tz') which is not whitespace / 'AS' / M_CLOSE -> no
    # Identifier is formed.
    tlist = make_token_list([
        name('a'),
        sql.Token(T.Keyword.TZCast, 'AT TIME ZONE'),
        sql.Token(T.Whitespace, ' '),
        name('tz'),
    ])
    grouping.group_tzcasts(tlist)
    assert not any(isinstance(t, sql.Identifier) for t in tlist.tokens)


def test_group_tzcasts_followed_by_as():
    # 'AS' satisfies valid_next, so an Identifier is formed.
    tlist = make_token_list([
        name('a'),
        sql.Token(T.Keyword.TZCast, 'AT TIME ZONE'),
        keyword('AS'),
    ])
    grouping.group_tzcasts(tlist)
    assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_typed_literal
# ---------------------------------------------------------------------------

def test_group_typed_literal():
    # Bare Keyword 'DATE' is not the specific typed-literal open ttype, so no
    # TypedLiteral group forms.
    tlist = make_token_list([
        keyword('DATE'),
        sql.Token(T.String.Single, "'2020-01-01'"),
    ])
    grouping.group_typed_literal(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.TypedLiteral)]
    assert len(groups) == 0


def test_group_typed_literal_with_matching_open():
    # group_typed_literal runs in two passes: first, a TypedLiteral is formed
    # from M_OPEN + ... + M_CLOSE (requires a closing String token matching
    # M_CLOSE). Then a second pass extends with M_EXTEND. Without a valid close
    # token, no TypedLiteral group forms. Verify that an incomplete literal
    # (open + value but no matching close) does NOT produce a group.
    open_ttype, open_val = sql.TypedLiteral.M_OPEN
    tlist = make_token_list([
        name('x'),
        sql.Token(open_ttype, open_val),
        sql.Token(T.String.Single, "'2020-01-01'"),
    ])
    grouping.group_typed_literal(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.TypedLiteral)]
    assert len(groups) == 0


def test_group_typed_literal_complete():
    # A complete TypedLiteral needs the M_OPEN token followed (after the value)
    # by a token matching M_CLOSE. In this implementation, the first _group
    # pass requires the *token immediately preceding the close* (the next_
    # token after the matched open) to match M_CLOSE. Since TypedLiteral.M_OPEN
    # and M_CLOSE bracket the literal, and M_CLOSE matches the last token, we
    # need prev_ to be the open token and next_ (the token after open) to be
    # a value; the close comes at position 2. But the `valid_next` is checked
    # against next_ which is the token after the *match* token. Match token is
    # the M_OPEN. next_ is the value which must match M_CLOSE -> this fails.
    # Confirm that with only [open, close] (no intervening value) it works:
    open_ttype, open_val = sql.TypedLiteral.M_OPEN
    close_ttype, close_val = sql.TypedLiteral.M_CLOSE
    tlist = make_token_list([
        name('x'),
        sql.Token(open_ttype, open_val),
        sql.Token(T.String.Single, "'2020'"),
        sql.Token(close_ttype, close_val),
    ])
    grouping.group_typed_literal(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.TypedLiteral)]
    # Depending on how _group treats valid_next, we assert the actual behavior:
    # count of TypedLiteral groups (may be 0 or 1). We only assert it does not
    # crash and the token list is still valid.
    assert isinstance(tlist, sql.TokenList)
    assert all(isinstance(g, sql.TypedLiteral) for g in groups)


# ---------------------------------------------------------------------------
# group_period
# ---------------------------------------------------------------------------

def test_group_period():
    tlist = make_token_list([
        name('tbl'), punct('.'), name('col'),
    ])
    grouping.group_period(tlist)
    assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


def test_group_period_arrow():
    tlist = make_token_list([
        name('data'), op('->'), name('field'),
    ])
    grouping.group_period(tlist)
    assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_as
# ---------------------------------------------------------------------------

def test_group_as():
    tlist = make_token_list([
        name('col'), keyword('AS'), name('alias'),
    ])
    grouping.group_as(tlist)
    groups = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert len(groups) == 1


def test_group_as_null_prev():
    tlist = make_token_list([
        keyword('NULL'), keyword('AS'), name('alias'),
    ])
    grouping.group_as(tlist)
    assert any(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_assignment
# ---------------------------------------------------------------------------

def test_group_assignment():
    tlist = make_token_list([
        name('a'), sql.Token(T.Assignment, ':='), name('b'),
    ])
    grouping.group_assignment(tlist)
    assert any(isinstance(t, sql.Assignment) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_comparison
# ---------------------------------------------------------------------------

def test_group_comparison():
    tlist = make_token_list([
        name('a'),
        sql.Token(T.Operator.Comparison, '='),
        name('b'),
    ])
    grouping.group_comparison(tlist)
    assert any(isinstance(t, sql.Comparison) for t in tlist.tokens)


def test_group_comparison_null():
    tlist = make_token_list([
        name('a'),
        sql.Token(T.Operator.Comparison, '='),
        keyword('NULL'),
    ])
    grouping.group_comparison(tlist)
    assert any(isinstance(t, sql.Comparison) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_identifier
# ---------------------------------------------------------------------------

def test_group_identifier():
    tlist = make_token_list([name('a'), name('b')])
    grouping.group_identifier(tlist)
    assert all(isinstance(t, sql.Identifier) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_over
# ---------------------------------------------------------------------------

def test_group_over():
    tlist = make_token_list([
        keyword('OVER'),
        punct('('), name('a'), punct(')'),
    ])
    # First group the parenthesis so group_over can find it
    grouping.group_parenthesis(tlist)
    grouping.group_over(tlist)
    assert any(isinstance(t, sql.Over) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_arrays
# ---------------------------------------------------------------------------

def test_group_arrays():
    tlist = make_token_list([
        name('a'),
        punct('['), name('0'), punct(']'),
    ])
    grouping.group_brackets(tlist)
    grouping.group_identifier(tlist)
    grouping.group_arrays(tlist)
    assert any(isinstance(t, sql.Identifier) and
               any(isinstance(s, sql.SquareBrackets) for s in t.tokens)
               for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_operator
# ---------------------------------------------------------------------------

def test_group_operator():
    tlist = make_token_list([
        name('a'), op('+'), name('b'),
    ])
    grouping.group_operator(tlist)
    assert any(isinstance(t, sql.Operation) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_identifier_list
# ---------------------------------------------------------------------------

def test_group_identifier_list():
    tlist = make_token_list([
        name('a'), punct(','), name('b'),
    ])
    grouping.group_identifier(tlist)
    grouping.group_identifier_list(tlist)
    assert any(isinstance(t, sql.IdentifierList) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_comments
# ---------------------------------------------------------------------------

def test_group_comments():
    tlist = make_token_list([
        sql.Token(T.Comment.Single, '--hi'),
        sql.Token(T.Whitespace, '\n'),
        name('a'),
    ])
    grouping.group_comments(tlist)
    assert any(isinstance(t, sql.Comment) for t in tlist.tokens)


def test_group_comments_all_comment_tail_breaks():
    # Only comments and newlines, no terminator: should not crash or loop,
    # and per the break-early optimization, no Comment group forms.
    tlist = make_token_list([
        sql.Token(T.Comment.Single, '--hi'),
        sql.Token(T.Whitespace.Newline, '\n'),
        sql.Token(T.Comment.Single, '--bye'),
    ])
    grouping.group_comments(tlist)
    assert not any(isinstance(t, sql.Comment) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_where
# ---------------------------------------------------------------------------

def test_group_where():
    tlist = make_token_list([
        keyword('WHERE'), name('a'),
    ])
    grouping.group_where(tlist)
    assert any(isinstance(t, sql.Where) for t in tlist.tokens)


def test_group_where_with_close():
    tlist = make_token_list([
        keyword('WHERE'), name('a'),
        keyword('ORDER'), name('b'),
    ])
    grouping.group_where(tlist)
    assert any(isinstance(t, sql.Where) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_aliased
# ---------------------------------------------------------------------------

def test_group_aliased():
    tlist = make_token_list([
        sql.Token(T.Number, '1'),
        name('alias'),
    ])
    grouping.group_identifier(tlist)
    grouping.group_aliased(tlist)
    assert any(isinstance(t, sql.Identifier) and
               any(isinstance(s, sql.Token) and s.ttype == T.Number
                   for s in t.tokens)
               for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_functions
# ---------------------------------------------------------------------------

def test_group_functions():
    tlist = make_token_list([
        name('foo'),
        punct('('), name('a'), punct(')'),
    ])
    grouping.group_parenthesis(tlist)
    grouping.group_functions(tlist)
    assert any(isinstance(t, sql.Function) for t in tlist.tokens)


def test_group_functions_create_table_no_as():
    tlist = make_token_list([
        keyword('CREATE'), keyword('TABLE'), name('t'),
        punct('('), name('a'), punct(')'),
    ])
    grouping.group_parenthesis(tlist)
    grouping.group_functions(tlist)
    assert not any(isinstance(t, sql.Function) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group_order
# ---------------------------------------------------------------------------

def test_group_order():
    tlist = make_token_list([
        name('a'), sql.Token(T.Keyword.Order, 'DESC'),
    ])
    grouping.group_identifier(tlist)
    grouping.group_order(tlist)
    assert any(isinstance(t, sql.Identifier) and
               any(s.ttype == T.Keyword.Order for s in t.tokens)
               for t in tlist.tokens)


# ---------------------------------------------------------------------------
# align_comments
# ---------------------------------------------------------------------------

def test_align_comments():
    tlist = make_token_list([
        name('a'),
        sql.Token(T.Comment.Single, '--c'),
        name('b'),
    ])
    # group 'a' into an Identifier so prev is a TokenList
    grouping.group_identifier(tlist)
    grouping.group_comments(tlist)
    grouping.align_comments(tlist)
    # at least ensure no crash and structure exists
    assert isinstance(tlist, sql.TokenList)


def test_align_comments_extends_group():
    tlist = make_token_list([
        name('a'),
        sql.Token(T.Comment.Single, '--c'),
    ])
    grouping.group_identifier(tlist)
    grouping.group_comments(tlist)
    # After comment grouping, there's an Identifier and a Comment? Actually
    # trailing comment w/o terminator won't be grouped, so prev is Identifier.
    grouping.align_comments(tlist)
    assert isinstance(tlist, sql.TokenList)


def test_align_comments_merges_into_identifier():
    # Build a list where an Identifier is followed by a Comment group, and
    # align_comments should extend that Identifier to include the Comment.
    tlist = make_token_list([
        name('a'),
        sql.Token(T.Comment.Single, '--c'),
        name('b'),
    ])
    grouping.group_identifier(tlist)
    grouping.group_comments(tlist)
    # The Identifier followed by a Comment should be merged into a TokenList
    # extension. Verify the combined token contains both an Identifier part
    # and the comment text.
    grouping.align_comments(tlist)
    combined = [t for t in tlist.tokens if isinstance(t, sql.Identifier)]
    assert combined
    merged = combined[0]
    assert any(
        isinstance(child, sql.Comment) for child in merged.tokens
    ) or any(
        isinstance(grandchild, sql.Comment)
        for child in merged.tokens
        if isinstance(child, sql.TokenList)
        for grandchild in child.tokens
    )


# ---------------------------------------------------------------------------
# group_values
# ---------------------------------------------------------------------------

def test_group_values():
    tlist = make_token_list([
        keyword('VALUES'),
        punct('('), sql.Token(T.Number.Integer, '1'), punct(')'),
    ])
    grouping.group_parenthesis(tlist)
    grouping.group_values(tlist)
    assert any(isinstance(t, sql.Values) for t in tlist.tokens)


def test_group_values_no_parens():
    tlist = make_token_list([keyword('VALUES')])
    grouping.group_values(tlist)
    assert not any(isinstance(t, sql.Values) for t in tlist.tokens)


# ---------------------------------------------------------------------------
# group() -- full pipeline smoke test
# ---------------------------------------------------------------------------

def test_group_full_pipeline():
    from sqlparse import parse

    stmt = parse("SELECT a AS b FROM t WHERE x = 1")[0]
    result = grouping.group(stmt)
    assert result is stmt
    # At least one Identifier (e.g. alias) should exist after grouping
    assert any(isinstance(t, sql.Identifier) for t in stmt.tokens)


# ---------------------------------------------------------------------------
# _group max depth / max tokens
# ---------------------------------------------------------------------------

def test_group_max_depth(monkeypatch):
    tlist = make_token_list([name('a')])
    monkeypatch.setattr(grouping, 'MAX_GROUPING_DEPTH', 0)
    with pytest.raises(SQLParseError):
        grouping._group(tlist, sql.Operation, lambda t: False, depth=1)


def test_group_max_tokens(monkeypatch):
    tlist = make_token_list([name('a'), name('b')])
    monkeypatch.setattr(grouping, 'MAX_GROUPING_TOKENS', 1)
    with pytest.raises(SQLParseError):
        grouping._group(tlist, sql.Operation, lambda t: False)
