"""Tests for sqlparse.engine.statement_splitter."""

import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tokens(ttype, value):
    """Build a single (ttype, value) stream token."""
    return [(ttype, value)]


def run(stream):
    return list(StatementSplitter().process(stream))


def values(statements):
    """Flatten statement objects back into their raw token values."""
    return [''.join(t.value for t in stmt.tokens) for stmt in statements]


# ---------------------------------------------------------------------------
# Construction / reset
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        sp = StatementSplitter()
        assert sp.tokens == []
        assert sp.level == 0
        assert sp.consume_ws is False
        assert sp._block_stack == []
        assert sp._parenthesis_level == 0
        assert sp._unconfirmed_start is None
        assert sp._is_create is False
        assert sp._seen_begin is False

    def test_reset_restores_defaults(self):
        sp = StatementSplitter()
        sp.tokens = [sql.Token(T.Text, "x")]
        sp.level = 5
        sp.consume_ws = True
        sp._block_stack = ['BEGIN']
        sp._parenthesis_level = 3
        sp._unconfirmed_start = 'FOR'
        sp._is_create = True
        sp._seen_begin = True
        sp._reset()
        assert sp.tokens == []
        assert sp.level == 0
        assert sp.consume_ws is False
        assert sp._block_stack == []
        assert sp._parenthesis_level == 0
        assert sp._unconfirmed_start is None
        assert sp._is_create is False
        assert sp._seen_begin is False


# ---------------------------------------------------------------------------
# _change_splitlevel: punctuation
# ---------------------------------------------------------------------------

class TestChangeSplitLevelPunctuation:
    def test_semicolon_outside(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Punctuation, ';') == 0

    def test_semicolon_resets_unconfirmed(self):
        sp = StatementSplitter()
        sp._unconfirmed_start = 'FOR'
        sp._change_splitlevel(T.Punctuation, ';')
        assert sp._unconfirmed_start is None

    def test_semicolon_pops_begin_when_seen(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._seen_begin = True
        res = sp._change_splitlevel(T.Punctuation, ';')
        assert res == -1
        assert sp._block_stack == []
        assert sp._seen_begin is False

    def test_semicolon_seen_begin_without_begin_stack(self):
        sp = StatementSplitter()
        sp._block_stack = ['LOOP']
        sp._seen_begin = True
        res = sp._change_splitlevel(T.Punctuation, ';')
        assert res == 0
        assert sp._seen_begin is False
        assert sp._block_stack == ['LOOP']

    def test_open_paren(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Punctuation, '(') == 1
        assert sp._parenthesis_level == 1

    def test_close_paren(self):
        sp = StatementSplitter()
        sp._parenthesis_level = 2
        assert sp._change_splitlevel(T.Punctuation, ')') == -1
        assert sp._parenthesis_level == 1

    def test_close_paren_never_negative(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Punctuation, ')') == -1
        assert sp._parenthesis_level == 0

    def test_other_punctuation(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Punctuation, ',') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: non-keyword tokens
# ---------------------------------------------------------------------------

class TestChangeSplitLevelNonKeyword:
    def test_text_token(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Text, 'foo') == 0

    def test_whitespace(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Whitespace, ' ') == 0

    def test_name_token_returns_zero(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Name, 'BEGIN') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: DDL / DECLARE
# ---------------------------------------------------------------------------

class TestCreateDeclare:
    def test_create_sets_flag(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword.DDL, 'CREATE') == 0
        assert sp._is_create is True

    def test_create_or_replace(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword.DDL,
                                     'CREATE OR REPLACE') == 0
        assert sp._is_create is True

    def test_declare_with_create_and_no_block(self):
        sp = StatementSplitter()
        sp._is_create = True
        assert sp._change_splitlevel(T.Keyword, 'DECLARE') == 1
        assert sp._block_stack == ['DECLARE']

    def test_declare_without_create(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert sp._block_stack == []

    def test_declare_with_existing_block(self):
        sp = StatementSplitter()
        sp._is_create = True
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert sp._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# _change_splitlevel: BEGIN handling
# ---------------------------------------------------------------------------

class TestBegin:
    def test_begin_pushes(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword, 'BEGIN') == 1
        assert sp._block_stack == ['BEGIN']
        assert sp._seen_begin is True

    def test_begin_transitions_declare(self):
        sp = StatementSplitter()
        sp._block_stack = ['DECLARE']
        assert sp._change_splitlevel(T.Keyword, 'BEGIN') == 0
        assert sp._block_stack == ['BEGIN']

    def test_begin_lowercase(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword, 'begin') == 1
        assert sp._block_stack == ['BEGIN']

    def test_transaction_keyword_after_begin(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._seen_begin = True
        res = sp._change_splitlevel(T.Keyword, 'TRANSACTION')
        assert res == -1
        assert sp._block_stack == []
        assert sp._seen_begin is False

    def test_transaction_work_after_begin(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._seen_begin = True
        assert sp._change_splitlevel(T.Name, 'WORK') == -1
        assert sp._block_stack == []

    def test_transaction_without_begin_stack(self):
        sp = StatementSplitter()
        sp._block_stack = []
        sp._seen_begin = True
        assert sp._change_splitlevel(T.Keyword, 'TRANSACTION') == 0
        assert sp._seen_begin is False

    def test_transaction_not_seen_begin(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._seen_begin = False
        # Because not seen_begin, and BEGIN in stack, nothing specially handled
        assert sp._change_splitlevel(T.Keyword, 'TRANSACTION') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: nested blocks
# ---------------------------------------------------------------------------

class TestNestedBlocks:
    def test_for_sets_unconfirmed(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'FOR') == 0
        assert sp._unconfirmed_start == 'FOR'

    def test_while_sets_unconfirmed(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'WHILE') == 0
        assert sp._unconfirmed_start == 'WHILE'

    def test_loop_after_for(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._unconfirmed_start = 'FOR'
        res = sp._change_splitlevel(T.Keyword, 'LOOP')
        assert res == 1
        assert sp._block_stack == ['BEGIN', 'FOR']
        assert sp._unconfirmed_start is None

    def test_loop_after_while(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._unconfirmed_start = 'WHILE'
        assert sp._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert sp._block_stack == ['BEGIN', 'WHILE']

    def test_loop_standalone(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert sp._block_stack == ['BEGIN', 'LOOP']

    def test_do_after_for(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._unconfirmed_start = 'FOR'
        assert sp._change_splitlevel(T.Keyword, 'DO') == 1
        assert sp._block_stack == ['BEGIN', 'FOR']

    def test_if_pushes(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'IF') == 1
        assert sp._block_stack == ['BEGIN', 'IF']

    def test_case_pushes(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'CASE') == 1
        assert sp._block_stack == ['BEGIN', 'CASE']

    def test_nested_keyword_outside_block_ignored(self):
        sp = StatementSplitter()
        # no BEGIN in block stack; IF should not push
        assert sp._change_splitlevel(T.Keyword, 'IF') == 0
        assert sp._block_stack == []


# ---------------------------------------------------------------------------
# _change_splitlevel: closing keywords
# ---------------------------------------------------------------------------

class TestClosingKeywords:
    def test_end_if(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'IF']
        assert sp._change_splitlevel(T.Keyword, 'END IF') == -1
        assert sp._block_stack == ['BEGIN']

    def test_end_if_wrong_top(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'LOOP']
        assert sp._change_splitlevel(T.Keyword, 'END IF') == 0
        assert sp._block_stack == ['BEGIN', 'LOOP']

    def test_end_for(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'FOR']
        assert sp._change_splitlevel(T.Keyword, 'END FOR') == -1
        assert sp._block_stack == ['BEGIN']

    def test_end_while(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'WHILE']
        assert sp._change_splitlevel(T.Keyword, 'END WHILE') == -1
        assert sp._block_stack == ['BEGIN']

    def test_end_loop_from_loop(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'LOOP']
        assert sp._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert sp._block_stack == ['BEGIN']

    def test_end_loop_from_for(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'FOR']
        assert sp._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert sp._block_stack == ['BEGIN']

    def test_end_loop_from_while(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'WHILE']
        assert sp._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert sp._block_stack == ['BEGIN']

    def test_end_case(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'CASE']
        assert sp._change_splitlevel(T.Keyword, 'END CASE') == -1
        assert sp._block_stack == ['BEGIN']

    def test_end_generic_with_stack(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'END') == -1
        assert sp._block_stack == []

    def test_end_generic_empty_stack(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword, 'END') == -1
        assert sp._block_stack == []

    def test_unknown_keyword(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'SELECT') == 0
        assert sp._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# process: splitting
# ---------------------------------------------------------------------------

class TestProcess:
    def test_empty_stream(self):
        assert run([]) == []

    def test_whitespace_only(self):
        stmts = run([(T.Whitespace, '   ')])
        assert stmts == []

    def test_single_statement(self):
        stmts = run([(T.Text, 'SELECT 1')])
        assert len(stmts) == 1
        assert values(stmts) == ['SELECT 1']

    def test_two_statements_semicolon(self):
        stream = [
            (T.Keyword.DML, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '1'),
            (T.Punctuation, ';'),
            (T.Whitespace, ' '),
            (T.Keyword.DML, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '2'),
        ]
        stmts = run(stream)
        assert len(stmts) == 2
        assert values(stmts) == ['SELECT 1; ', 'SELECT 2']

    def test_semicolon_start_of_next(self):
        stream = [
            (T.Text, 'a'),
            (T.Punctuation, ';'),
            (T.Text, 'b'),
        ]
        stmts = run(stream)
        assert len(stmts) == 2
        assert values(stmts) == ['a;', 'b']

    def test_statement_is_sql_statement(self):
        stmts = run([(T.Text, 'x')])
        assert all(isinstance(s, sql.Statement) for s in stmts)

    def test_go_splits(self):
        stream = [
            (T.Text, 'x'),
            (T.Whitespace, ' '),
            (T.Keyword, 'GO'),
            (T.Text, 'y'),
        ]
        stmts = run(stream)
        assert len(stmts) == 2
        assert values(stmts) == ['x GO', 'y']

    def test_go_with_count(self):
        stream = [
            (T.Text, 'x'),
            (T.Whitespace, ' '),
            (T.Keyword, 'GO 2'),
            (T.Text, 'y'),
        ]
        stmts = run(stream)
        assert len(stmts) == 2
        assert values(stmts) == ['x GO 2', 'y']

    def test_semicolon_inside_parens_no_split(self):
        stream = [
            (T.Text, 'f'),
            (T.Punctuation, '('),
            (T.Text, 'a'),
            (T.Punctuation, ';'),
            (T.Text, 'b'),
            (T.Punctuation, ')'),
        ]
        stmts = run(stream)
        assert len(stmts) == 1
        assert values(stmts) == ['f(a;b)']

    def test_semicolon_inside_begin_no_split(self):
        stream = [
            (T.Keyword, 'BEGIN'),
            (T.Whitespace, ' '),
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '1'),
            (T.Punctuation, ';'),
            (T.Whitespace, ' '),
            (T.Keyword, 'END'),
        ]
        stmts = run(stream)
        assert len(stmts) == 1
        assert values(stmts) == ['BEGIN SELECT 1; END']

    def test_trailing_semicolon_single_statement(self):
        stmts = run([(T.Text, 'x'), (T.Punctuation, ';')])
        assert len(stmts) == 1
        assert values(stmts) == ['x;']

    def test_multiple_semicolons_no_empty_statements(self):
        stmts = run([
            (T.Text, 'a'),
            (T.Punctuation, ';'),
            (T.Punctuation, ';'),
            (T.Text, 'b'),
        ])
        assert values(stmts) == ['a;;', 'b']

    def test_whitespace_after_semicolon_kept(self):
        stream = [
            (T.Text, 'a'),
            (T.Punctuation, ';'),
            (T.Whitespace, ' '),
            (T.Comment.Single, '-- c'),
            (T.Text, 'b'),
        ]
        stmts = run(stream)
        assert len(stmts) == 2
        assert values(stmts) == ['a; -- c', 'b']

    def test_partial_trailing_whitespace_yields_statement(self):
        # A statement followed by only whitespace still gets yielded.
        stmts = run([(T.Text, 'a'), (T.Punctuation, ';'),
                     (T.Whitespace, '  ')])
        assert len(stmts) == 1
        assert values(stmts) == ['a;  ']


# ---------------------------------------------------------------------------
# process: _seen_begin bookkeeping
# ---------------------------------------------------------------------------

class TestSeenBeginBookkeeping:
    def test_seen_begin_reset_on_other_token(self):
        sp = StatementSplitter()
        sp.consume_ws = False
        stream = [
            (T.Keyword, 'BEGIN'),
            (T.Text, 'x'),
        ]
        list(sp.process(iter(stream)))
        assert sp._seen_begin is False

    def test_begin_comment_does_not_reset_seen_begin(self):
        sp = StatementSplitter()
        stream = [
            (T.Keyword, 'BEGIN'),
            (T.Comment.Single, '-- c'),
        ]
        list(sp.process(iter(stream)))
        assert sp._seen_begin is True

    def test_level_after_begin_end(self):
        stream = [
            (T.Keyword, 'BEGIN'),
            (T.Punctuation, ';'),  # close standalone BEGIN
        ]
        sp = StatementSplitter()
        list(sp.process(iter(stream)))
        assert sp.level == 0

    def test_paren_level_tracking(self):
        stream = [
            (T.Punctuation, '('),
            (T.Punctuation, '('),
            (T.Punctuation, ')'),
        ]
        sp = StatementSplitter()
        list(sp.process(iter(stream)))
        assert sp._parenthesis_level == 1
        assert sp.level == 1
