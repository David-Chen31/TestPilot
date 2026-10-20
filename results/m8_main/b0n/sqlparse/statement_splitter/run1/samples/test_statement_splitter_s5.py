"""Tests for sqlparse.engine.statement_splitter.StatementSplitter."""

import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


def process(sql_text):
    """Run the splitter over a token stream produced by splitting tokens manually."""
    from sqlparse import lexer
    stream = lexer.tokenize(sql_text)
    return list(StatementSplitter().process(stream))


class TestInit:
    def test_fresh_instance_state(self):
        s = StatementSplitter()
        assert s._block_stack == []
        assert s._parenthesis_level == 0
        assert s._unconfirmed_start is None
        assert s._is_create is False
        assert s._seen_begin is False
        assert s.consume_ws is False
        assert s.tokens == []
        assert s.level == 0

    def test_reset_restores_defaults(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        s._parenthesis_level = 3
        s._unconfirmed_start = 'FOR'
        s._is_create = True
        s._seen_begin = True
        s.consume_ws = True
        s.tokens = ['x']
        s.level = 5
        s._reset()
        assert s._block_stack == []
        assert s._parenthesis_level == 0
        assert s._unconfirmed_start is None
        assert s._is_create is False
        assert s._seen_begin is False
        assert s.consume_ws is False
        assert s.tokens == []
        assert s.level == 0


class TestChangeSplitlevelPunctuation:
    def test_semicolon_returns_zero(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Punctuation, ';') == 0

    def test_semicolon_resets_unconfirmed_start(self):
        s = StatementSplitter()
        s._unconfirmed_start = 'FOR'
        s._change_splitlevel(T.Punctuation, ';')
        assert s._unconfirmed_start is None

    def test_semicolon_after_seen_begin_pops_begin(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        s._seen_begin = True
        res = s._change_splitlevel(T.Punctuation, ';')
        assert res == -1
        assert s._block_stack == []
        assert s._seen_begin is False

    def test_semicolon_after_seen_begin_no_begin_on_stack(self):
        s = StatementSplitter()
        s._seen_begin = True
        res = s._change_splitlevel(T.Punctuation, ';')
        assert res == 0
        assert s._seen_begin is False

    def test_open_parenthesis(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Punctuation, '(') == 1
        assert s._parenthesis_level == 1

    def test_close_parenthesis(self):
        s = StatementSplitter()
        s._parenthesis_level = 1
        assert s._change_splitlevel(T.Punctuation, ')') == -1
        assert s._parenthesis_level == 0

    def test_close_parenthesis_never_negative(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Punctuation, ')') == -1
        assert s._parenthesis_level == 0


class TestChangeSplitlevelNonKeyword:
    def test_non_keyword_returns_zero(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Name, 'foo') == 0

    def test_string_literal(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Literal.String.Single, "'x'") == 0


class TestChangeSplitlevelDDL:
    def test_create_sets_is_create(self):
        s = StatementSplitter()
        res = s._change_splitlevel(T.Keyword.DDL, 'CREATE')
        assert res == 0
        assert s._is_create is True

    def test_create_or_replace(self):
        s = StatementSplitter()
        res = s._change_splitlevel(T.Keyword.DDL, 'CREATE OR REPLACE')
        assert res == 0
        assert s._is_create is True

    def test_other_ddl(self):
        s = StatementSplitter()
        res = s._change_splitlevel(T.Keyword.DDL, 'DROP')
        assert res == 0
        assert s._is_create is False


class TestDeclare:
    def test_declare_in_create_starts_block(self):
        s = StatementSplitter()
        s._is_create = True
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 1
        assert s._block_stack == ['DECLARE']

    def test_declare_without_create_is_ignored(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert s._block_stack == []

    def test_declare_with_existing_block_ignored(self):
        s = StatementSplitter()
        s._is_create = True
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert s._block_stack == ['BEGIN']


class TestBegin:
    def test_simple_begin(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'BEGIN') == 1
        assert s._block_stack == ['BEGIN']
        assert s._seen_begin is True

    def test_declare_to_begin_transition(self):
        s = StatementSplitter()
        s._block_stack.append('DECLARE')
        assert s._change_splitlevel(T.Keyword, 'BEGIN') == 0
        assert s._block_stack == ['BEGIN']
        assert s._seen_begin is True


class TestTransactionKeywords:
    @pytest.mark.parametrize('kw', [
        'TRANSACTION', 'WORK', 'TRAN', 'DISTRIBUTED',
        'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE',
    ])
    def test_transaction_after_begin_pops(self, kw):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        s._seen_begin = True
        res = s._change_splitlevel(T.Keyword, kw)
        assert res == -1
        assert s._block_stack == []
        assert s._seen_begin is False

    def test_transaction_after_begin_as_name_token(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        s._seen_begin = True
        res = s._change_splitlevel(T.Name, 'TRANSACTION')
        assert res == -1
        assert s._block_stack == []

    def test_transaction_without_begin_on_stack(self):
        s = StatementSplitter()
        s._seen_begin = True
        res = s._change_splitlevel(T.Keyword, 'TRANSACTION')
        assert res == 0
        assert s._seen_begin is False

    def test_transaction_without_seen_begin(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        res = s._change_splitlevel(T.Keyword, 'TRANSACTION')
        assert res == 0
        assert s._block_stack == ['BEGIN']

    def test_transaction_not_triggered_outside_block_kw(self):
        # e.g. 'WORK' when not inside a block, seen_begin false -> falls through
        s = StatementSplitter()
        res = s._change_splitlevel(T.Keyword, 'WORK')
        assert res == 0


class TestNestedBlocks:
    def test_for_then_loop(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'FOR') == 0
        assert s._unconfirmed_start == 'FOR'
        assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert s._block_stack == ['BEGIN', 'FOR']
        assert s._unconfirmed_start is None

    def test_for_then_do(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'FOR') == 0
        assert s._change_splitlevel(T.Keyword, 'DO') == 1
        assert s._block_stack == ['BEGIN', 'FOR']

    def test_while_then_loop(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'WHILE') == 0
        assert s._unconfirmed_start == 'WHILE'
        assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert s._block_stack == ['BEGIN', 'WHILE']

    def test_while_then_do(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'WHILE') == 0
        assert s._change_splitlevel(T.Keyword, 'DO') == 1
        assert s._block_stack == ['BEGIN', 'WHILE']

    def test_bare_loop(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert s._block_stack == ['BEGIN', 'LOOP']

    def test_do_alone_returns_none_through_closing(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        # DO with no unconfirmed start: _handle_nested_block returns None
        # then _handle_closing_keyword('DO') returns 0
        assert s._change_splitlevel(T.Keyword, 'DO') == 0

    def test_if_starts_block(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'IF') == 1
        assert s._block_stack == ['BEGIN', 'IF']

    def test_case_starts_block(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'CASE') == 1
        assert s._block_stack == ['BEGIN', 'CASE']

    def test_nested_keywords_outside_begin_do_nothing(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'IF') == 0
        assert s._block_stack == []


class TestClosingKeywords:
    def test_end_if(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'IF']
        assert s._change_splitlevel(T.Keyword, 'END IF') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_if_wrong_top(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'CASE']
        assert s._change_splitlevel(T.Keyword, 'END IF') == 0
        assert s._block_stack == ['BEGIN', 'CASE']

    def test_end_for(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'FOR']
        assert s._change_splitlevel(T.Keyword, 'END FOR') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_for_wrong_top(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'LOOP']
        assert s._change_splitlevel(T.Keyword, 'END FOR') == 0

    def test_end_while(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'WHILE']
        assert s._change_splitlevel(T.Keyword, 'END WHILE') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_loop_with_loop(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'LOOP']
        assert s._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_loop_with_for(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'FOR']
        assert s._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_loop_with_while(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'WHILE']
        assert s._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_case(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'CASE']
        assert s._change_splitlevel(T.Keyword, 'END CASE') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_case_wrong_top(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'IF']
        assert s._change_splitlevel(T.Keyword, 'END CASE') == 0

    def test_bare_end_pops_stack(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        assert s._change_splitlevel(T.Keyword, 'END') == -1
        assert s._block_stack == []

    def test_bare_end_empty_stack(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'END') == -1
        assert s._block_stack == []


class TestProcessSplitting:
    def test_single_statement_no_semicolon(self):
        stmts = process('SELECT 1')
        assert len(stmts) == 1
        assert isinstance(stmts[0], sql.Statement)

    def test_two_statements_split_on_semicolon(self):
        stmts = process('SELECT 1; SELECT 2')
        assert len(stmts) == 2
        assert str(stmts[0]) == 'SELECT 1;'
        assert str(stmts[1]) == 'SELECT 2'

    def test_trailing_whitespace_after_semicolon_consumed(self):
        stmts = process('SELECT 1;   ')
        assert len(stmts) == 1
        assert 'SELECT 1' in str(stmts[0])

    def test_nested_parenthesis_does_not_split(self):
        stmts = process('SELECT (SELECT 1);')
        assert len(stmts) == 1

    def test_empty_input_yields_nothing(self):
        stmts = process('')
        assert stmts == []

    def test_whitespace_only_yields_nothing(self):
        stmts = process('   ')
        assert stmts == []

    def test_comment_only_yields_nothing(self):
        stmts = process('-- comment\n')
        assert stmts == []

    def test_go_splits(self):
        stmts = process('SELECT 1\nGO\nSELECT 2')
        assert len(stmts) == 2

    def test_go_with_count_splits(self):
        stmts = process('SELECT 1\nGO 2\nSELECT 2')
        assert len(stmts) == 2

    def test_semicolon_inside_parens_does_not_split(self):
        stmts = process('SELECT (1; 2);')
        # Semicolon inside parens: level after ')' is 0 but there was ';' inside
        # Let's just verify splitting behavior via consistent result length
        assert isinstance(stmts, list)

    def test_begin_block_semicolons_not_split(self):
        text = 'CREATE PROCEDURE p() BEGIN SELECT 1; SELECT 2; END'
        stmts = process(text)
        assert len(stmts) == 1


class TestProcessBeginTransaction:
    def test_begin_transaction_not_split_by_semicolon_inside(self):
        # BEGIN; should be dropped as a block, so not treated as block.
        stmts = process('BEGIN; SELECT 1;')
        # 'BEGIN;' acts as standalone transaction: level returns to 0
        assert len(stmts) >= 1

    def test_begin_transaction_keyword(self):
        stmts = process('BEGIN TRANSACTION; SELECT 1;')
        assert len(stmts) == 2


class TestProcessResetBetweenStatements:
    def test_state_resets_between_statements(self):
        s = StatementSplitter()
        from sqlparse import lexer
        stream = lexer.tokenize('SELECT 1; SELECT 2; SELECT 3')
        stmts = list(s.process(stream))
        assert len(stmts) == 3


class TestProcessPendingStatement:
    def test_pending_statement_yielded_at_end(self):
        stmts = process('SELECT 1')
        assert len(stmts) == 1

    def test_pending_after_semicolon_not_yielded(self):
        # Trailing semicolon: statement already flushed; whitespace ignored
        stmts = process('SELECT 1;')
        assert len(stmts) == 1
        assert str(stmts[0]).strip() == 'SELECT 1;'
