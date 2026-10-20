import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


def tokenize_value(value, ttype=T.Keyword):
    return [(ttype, value)]


def statements_text(splitter, stream):
    return [str(s) for s in splitter.process(stream)]


class TestResetAndInit:
    def test_initial_state(self):
        s = StatementSplitter()
        assert s._block_stack == []
        assert s._parenthesis_level == 0
        assert s._unconfirmed_start is None
        assert s._is_create is False
        assert s._seen_begin is False
        assert s.consume_ws is False
        assert s.tokens == []
        assert s.level == 0

    def test_reset_clears_state(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        s._parenthesis_level = 3
        s._unconfirmed_start = 'FOR'
        s._is_create = True
        s._seen_begin = True
        s.consume_ws = True
        s.tokens = [sql.Token(T.Keyword, 'x')]
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


class TestChangeSplitLevelBasic:
    def test_semicolon_returns_zero(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Punctuation, ';') == 0

    def test_semicolon_resets_unconfirmed_start(self):
        s = StatementSplitter()
        s._unconfirmed_start = 'FOR'
        s._change_splitlevel(T.Punctuation, ';')
        assert s._unconfirmed_start is None

    def test_open_paren_increments(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Punctuation, '(') == 1
        assert s._parenthesis_level == 1

    def test_close_paren_decrements(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Punctuation, '(')
        assert s._change_splitlevel(T.Punctuation, ')') == -1
        assert s._parenthesis_level == 0

    def test_close_paren_does_not_go_negative(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Punctuation, ')') == -1
        assert s._parenthesis_level == 0

    def test_non_keyword_token_returns_zero(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Name, 'foo') == 0
        assert s._change_splitlevel(T.Literal.String.Single, "'x'") == 0

    def test_create_ddl_returns_zero_and_sets_flag(self):
        s = StatementSplitter()
        res = s._change_splitlevel(T.Keyword.DDL, 'CREATE')
        assert res == 0
        assert s._is_create is True

    def test_create_or_replace(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword.DDL, 'CREATE OR REPLACE')
        assert s._is_create is True


class TestDeclareAndBegin:
    def test_declare_requires_create(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert s._block_stack == []

    def test_declare_with_create_opens_block(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword.DDL, 'CREATE')
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 1
        assert s._block_stack == ['DECLARE']

    def test_declare_not_opened_when_block_stack_nonempty(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword.DDL, 'CREATE')
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0

    def test_begin_opens_block(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'BEGIN') == 1
        assert s._block_stack == ['BEGIN']
        assert s._seen_begin is True

    def test_begin_transitions_declare(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword.DDL, 'CREATE')
        s._change_splitlevel(T.Keyword, 'DECLARE')
        assert s._block_stack == ['DECLARE']
        assert s._change_splitlevel(T.Keyword, 'BEGIN') == 0
        assert s._block_stack == ['BEGIN']

    def test_begin_case_insensitive(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'begin') == 1
        assert s._block_stack == ['BEGIN']


class TestTransactionKeywords:
    @pytest.mark.parametrize('kw', [
        'TRANSACTION', 'WORK', 'TRAN', 'DISTRIBUTED',
        'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE',
    ])
    def test_transaction_popped_after_begin(self, kw):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword, 'BEGIN')
        assert s._block_stack == ['BEGIN']
        assert s._change_splitlevel(T.Keyword, kw) == -1
        assert s._block_stack == []
        assert s._seen_begin is False

    def test_transaction_name_token(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword, 'BEGIN')
        assert s._change_splitlevel(T.Name, 'TRANSACTION') == -1
        assert s._block_stack == []

    def test_transaction_without_begin_returns_zero(self):
        s = StatementSplitter()
        s._seen_begin = True
        # block_stack empty
        assert s._change_splitlevel(T.Keyword, 'TRANSACTION') == 0
        assert s._seen_begin is False


class TestClosingKeywords:
    def test_end_if_pops_if(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'IF']
        assert s._change_splitlevel(T.Keyword, 'END IF') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_if_no_match_returns_zero(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'CASE']
        assert s._change_splitlevel(T.Keyword, 'END IF') == 0

    def test_end_if_empty_stack(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'END IF') == 0

    def test_end_for_pops_for(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'FOR']
        assert s._change_splitlevel(T.Keyword, 'END FOR') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_while_pops_while(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'WHILE']
        assert s._change_splitlevel(T.Keyword, 'END WHILE') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_loop_pops_loop(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'LOOP']
        assert s._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_loop_pops_for(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'FOR']
        assert s._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_loop_pops_while(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'WHILE']
        assert s._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_loop_no_match(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'IF']
        assert s._change_splitlevel(T.Keyword, 'END LOOP') == 0

    def test_end_case_pops_case(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'CASE']
        assert s._change_splitlevel(T.Keyword, 'END CASE') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_case_no_match(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'IF']
        assert s._change_splitlevel(T.Keyword, 'END CASE') == 0

    def test_plain_end_pops_any(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'IF']
        assert s._change_splitlevel(T.Keyword, 'END') == -1
        assert s._block_stack == ['BEGIN']

    def test_plain_end_empty_stack(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'END') == -1
        assert s._block_stack == []


class TestUnconfirmedLoopStarters:
    def test_for_sets_unconfirmed(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        assert s._change_splitlevel(T.Keyword, 'FOR') == 0
        assert s._unconfirmed_start == 'FOR'

    def test_while_sets_unconfirmed(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        assert s._change_splitlevel(T.Keyword, 'WHILE') == 0
        assert s._unconfirmed_start == 'WHILE'

    def test_loop_after_for(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        s._change_splitlevel(T.Keyword, 'FOR')
        assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert s._block_stack == ['BEGIN', 'FOR']
        assert s._unconfirmed_start is None

    def test_loop_after_while(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        s._change_splitlevel(T.Keyword, 'WHILE')
        assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert s._block_stack == ['BEGIN', 'WHILE']

    def test_standalone_loop(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert s._block_stack == ['BEGIN', 'LOOP']

    def test_do_after_for(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        s._change_splitlevel(T.Keyword, 'FOR')
        assert s._change_splitlevel(T.Keyword, 'DO') == 1
        assert s._block_stack == ['BEGIN', 'FOR']

    def test_do_standalone_returns_none_then_handle_closing(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        # DO without FOR/WHILE falls to closing handler -> 0
        assert s._change_splitlevel(T.Keyword, 'DO') == 0
        assert s._block_stack == ['BEGIN']

    def test_if_pushes(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        assert s._change_splitlevel(T.Keyword, 'IF') == 1
        assert s._block_stack == ['BEGIN', 'IF']

    def test_case_pushes(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        assert s._change_splitlevel(T.Keyword, 'CASE') == 1
        assert s._block_stack == ['BEGIN', 'CASE']

    def test_nested_blocks_only_inside_begin(self):
        s = StatementSplitter()
        # No BEGIN -> no nested handling
        assert s._change_splitlevel(T.Keyword, 'IF') == 0
        assert s._block_stack == []


class TestProcess:
    def test_simple_statement(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '1'),
        ]
        result = list(s.process(stream))
        assert len(result) == 1
        assert str(result[0]) == 'SELECT 1'

    def test_two_statements_split_by_semicolon(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '1'),
            (T.Punctuation, ';'),
            (T.Whitespace, ' '),
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '2'),
            (T.Punctuation, ';'),
        ]
        result = list(s.process(stream))
        assert [str(r) for r in result] == ['SELECT 1;', 'SELECT 2;']

    def test_splitter_reset_between_statements(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword, 'SELECT'),
            (T.Punctuation, ';'),
            (T.Keyword, 'SELECT'),
            (T.Punctuation, ';'),
        ]
        list(s.process(stream))
        # after processing, state is clean-ish
        assert s._block_stack == []

    def test_go_splits(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '1'),
            (T.Newline, '\n'),
            (T.Keyword, 'GO'),
            (T.Newline, '\n'),
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '2'),
        ]
        result = list(s.process(stream))
        assert len(result) == 2
        assert 'GO' in str(result[0])
        assert str(result[1]) == 'SELECT 2'

    def test_go_with_number(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword, 'GO 2'),
            (T.Newline, '\n'),
            (T.Keyword, 'SELECT'),
        ]
        result = list(s.process(stream))
        assert len(result) >= 1
        assert 'GO 2' in str(result[0])

    def test_whitespace_only_no_statement(self):
        s = StatementSplitter()
        stream = [(T.Whitespace, '   '), (T.Newline, '\n')]
        result = list(s.process(stream))
        assert result == []

    def test_empty_stream(self):
        s = StatementSplitter()
        assert list(s.process([])) == []

    def test_semicolon_inside_parenthesis_does_not_split(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword, 'SELECT'),
            (T.Punctuation, '('),
            (T.Keyword, 'a'),
            (T.Punctuation, ';'),
            (T.Keyword, 'b'),
            (T.Punctuation, ')'),
        ]
        result = list(s.process(stream))
        assert len(result) == 1

    def test_begin_end_block_does_not_split_inside(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword, 'BEGIN'),
            (T.Newline, '\n'),
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '1'),
            (T.Punctuation, ';'),
            (T.Newline, '\n'),
            (T.Keyword, 'END'),
            (T.Punctuation, ';'),
            (T.Newline, '\n'),
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Literal.Number.Integer, '2'),
        ]
        result = list(s.process(stream))
        # BEGIN...END; is one statement, then SELECT 2 is separate
        assert len(result) == 2
        assert 'BEGIN' in str(result[0])
        assert str(result[1]) == 'SELECT 2'

    def test_results_are_statements(self):
        s = StatementSplitter()
        stream = [(T.Keyword, 'SELECT'), (T.Punctuation, ';')]
        result = list(s.process(stream))
        assert len(result) == 1
        assert isinstance(result[0], sql.Statement)

    def test_process_returns_generator(self):
        s = StatementSplitter()
        gen = s.process([])
        assert hasattr(gen, '__iter__')
        assert hasattr(gen, '__next__')

    def test_create_declare_begin_end(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword.DDL, 'CREATE'),
            (T.Whitespace, ' '),
            (T.Keyword, 'PROCEDURE'),
            (T.Whitespace, ' '),
            (T.Name, 'p'),
            (T.Whitespace, ' '),
            (T.Keyword, 'DECLARE'),
            (T.Punctuation, ';'),
            (T.Newline, '\n'),
            (T.Keyword, 'BEGIN'),
            (T.Punctuation, ';'),
            (T.Keyword, 'END'),
            (T.Punctuation, ';'),
        ]
        result = list(s.process(stream))
        assert len(result) == 1

    def test_seen_begin_reset_by_non_keyword(self):
        s = StatementSplitter()
        s._seen_begin = True
        stream = [(T.Name, 'foo')]
        list(s.process(stream))
        assert s._seen_begin is False

    def test_seen_begin_not_reset_by_comment(self):
        s = StatementSplitter()
        s._seen_begin = True
        stream = [(T.Comment.Single, '-- hi')]
        list(s.process(stream))
        assert s._seen_begin is True

    def test_begin_token_does_not_reset_seen_begin(self):
        s = StatementSplitter()
        stream = [(T.Keyword, 'BEGIN')]
        list(s.process(stream))
        assert s._seen_begin is True

    def test_standalone_begin_semicolon(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword, 'BEGIN'),
            (T.Punctuation, ';'),
        ]
        result = list(s.process(stream))
        assert len(result) == 1
        assert str(result[0]) == 'BEGIN;'

    def test_consume_ws_after_semicolon(self):
        s = StatementSplitter()
        stream = [(T.Punctuation, ';')]
        list(s.process(stream))
        # consume_ws set True after semicolon at level <= 0
        assert s.consume_ws is True

    def test_go_split_with_whitespace_after(self):
        s = StatementSplitter()
        stream = [
            (T.Keyword, 'GO'),
            (T.Whitespace, ' '),
            (T.Keyword, 'SELECT'),
        ]
        result = list(s.process(stream))
        assert len(result) == 2
        assert 'GO' in str(result[0])
        assert str(result[1]) == 'SELECT'
