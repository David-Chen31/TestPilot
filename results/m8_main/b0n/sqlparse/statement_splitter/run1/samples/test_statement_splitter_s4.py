"""Unit tests for sqlparse.engine.statement_splitter.StatementSplitter."""

import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


def make_stream(sql_text):
    # Use a simple tokenizer-free split to keep tests independent of the
    # lexer: each character group is mapped by hand in individual tests.
    # This helper is only used by tests that pass explicit token streams.
    raise NotImplementedError  # not used


def run_splitter(tokens):
    """Feed (ttype, value) pairs through the splitter and collect statements."""
    splitter = StatementSplitter()
    return list(splitter.process(iter(tokens)))


def stmt_texts(statements):
    out = []
    for st in statements:
        out.append(''.join(tok.value for tok in st.flatten()))
    return out


# ---------------------------------------------------------------------------
# Initialization / _reset
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_state(self):
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
        s._block_stack = ['BEGIN']
        s._parenthesis_level = 3
        s._unconfirmed_start = 'FOR'
        s._is_create = True
        s._seen_begin = True
        s.consume_ws = True
        s.tokens = [sql.Token(T.Number, '1')]
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


# ---------------------------------------------------------------------------
# _change_splitlevel: punctuation handling
# ---------------------------------------------------------------------------

class TestChangeSplitLevelPunctuation:
    def setup_method(self):
        self.s = StatementSplitter()

    def test_semicolon_returns_zero(self):
        assert self.s._change_splitlevel(T.Punctuation, ';') == 0

    def test_semicolon_clears_unconfirmed_start(self):
        self.s._unconfirmed_start = 'FOR'
        self.s._change_splitlevel(T.Punctuation, ';')
        assert self.s._unconfirmed_start is None

    def test_semicolon_pops_begin_when_seen(self):
        self.s._block_stack = ['BEGIN']
        self.s._seen_begin = True
        result = self.s._change_splitlevel(T.Punctuation, ';')
        assert result == -1
        assert self.s._block_stack == []
        assert self.s._seen_begin is False

    def test_semicolon_without_seen_begin_does_not_pop(self):
        self.s._block_stack = ['BEGIN']
        self.s._seen_begin = False
        result = self.s._change_splitlevel(T.Punctuation, ';')
        assert result == 0
        assert self.s._block_stack == ['BEGIN']

    def test_open_paren_increments(self):
        assert self.s._change_splitlevel(T.Punctuation, '(') == 1
        assert self.s._parenthesis_level == 1
        assert self.s._change_splitlevel(T.Punctuation, '(') == 1
        assert self.s._parenthesis_level == 2

    def test_close_paren_decrements(self):
        self.s._parenthesis_level = 2
        assert self.s._change_splitlevel(T.Punctuation, ')') == -1
        assert self.s._parenthesis_level == 1

    def test_close_paren_does_not_go_negative(self):
        self.s._parenthesis_level = 0
        assert self.s._change_splitlevel(T.Punctuation, ')') == -1
        assert self.s._parenthesis_level == 0

    def test_other_punctuation_zero(self):
        assert self.s._change_splitlevel(T.Punctuation, ',') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: non-keywords
# ---------------------------------------------------------------------------

class TestChangeSplitLevelNonKeyword:
    def test_name_token(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Name, 'foo') == 0

    def test_number_token(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Number, '1') == 0

    def test_whitespace_token(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Whitespace, ' ') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: CREATE / DECLARE / BEGIN
# ---------------------------------------------------------------------------

class TestChangeSplitLevelCreateBlock:
    def test_create_ddl_sets_flag(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword.DDL, 'CREATE OR REPLACE') == 0
        assert s._is_create is True

    def test_create_ddl_lowercase(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword.DDL, 'create table') == 0
        assert s._is_create is True

    def test_declare_ignored_without_create(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert s._block_stack == []

    def test_declare_ignored_when_block_stack_nonempty(self):
        s = StatementSplitter()
        s._is_create = True
        s._block_stack = ['BEGIN']
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert s._block_stack == ['BEGIN']

    def test_declare_with_create_pushes(self):
        s = StatementSplitter()
        s._is_create = True
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 1
        assert s._block_stack == ['DECLARE']

    def test_begin_pushes(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'BEGIN') == 1
        assert s._block_stack == ['BEGIN']
        assert s._seen_begin is True

    def test_begin_transitions_declare(self):
        s = StatementSplitter()
        s._block_stack = ['DECLARE']
        result = s._change_splitlevel(T.Keyword, 'BEGIN')
        assert result == 0
        assert s._block_stack == ['BEGIN']
        assert s._seen_begin is True


# ---------------------------------------------------------------------------
# _change_splitlevel: transaction keyword after BEGIN
# ---------------------------------------------------------------------------

class TestTransactionAfterBegin:
    @pytest.mark.parametrize('kw', [
        'TRANSACTION', 'WORK', 'TRAN', 'DISTRIBUTED',
        'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE',
    ])
    def test_transaction_after_begin_pops(self, kw):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        s._seen_begin = True
        result = s._change_splitlevel(T.Keyword, kw)
        assert result == -1
        assert s._block_stack == []
        assert s._seen_begin is False

    def test_transaction_after_begin_name_token(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        s._seen_begin = True
        result = s._change_splitlevel(T.Name, 'WORK')
        assert result == -1
        assert s._block_stack == []
        assert s._seen_begin is False

    def test_transaction_without_stack_returns_zero(self):
        s = StatementSplitter()
        s._seen_begin = True
        s._block_stack = []
        result = s._change_splitlevel(T.Keyword, 'TRANSACTION')
        assert result == 0
        assert s._seen_begin is False

    def test_transaction_without_seen_begin_ignored(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN']
        s._seen_begin = False
        # Not treated as transaction branch: falls through to closing keyword
        result = s._change_splitlevel(T.Keyword, 'TRANSACTION')
        assert result == 0
        assert s._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# _handle_nested_block
# ---------------------------------------------------------------------------

class TestHandleNestedBlock:
    def test_for_sets_unconfirmed(self):
        s = StatementSplitter()
        assert s._handle_nested_block('FOR') == 0
        assert s._unconfirmed_start == 'FOR'

    def test_while_sets_unconfirmed(self):
        s = StatementSplitter()
        assert s._handle_nested_block('WHILE') == 0
        assert s._unconfirmed_start == 'WHILE'

    def test_loop_after_for_confirms(self):
        s = StatementSplitter()
        s._unconfirmed_start = 'FOR'
        assert s._handle_nested_block('LOOP') == 1
        assert s._block_stack == ['FOR']
        assert s._unconfirmed_start is None

    def test_loop_after_while_confirms(self):
        s = StatementSplitter()
        s._unconfirmed_start = 'WHILE'
        assert s._handle_nested_block('LOOP') == 1
        assert s._block_stack == ['WHILE']
        assert s._unconfirmed_start is None

    def test_loop_standalone_pushes(self):
        s = StatementSplitter()
        assert s._handle_nested_block('LOOP') == 1
        assert s._block_stack == ['LOOP']

    def test_do_after_for_confirms(self):
        s = StatementSplitter()
        s._unconfirmed_start = 'FOR'
        assert s._handle_nested_block('DO') == 1
        assert s._block_stack == ['FOR']
        assert s._unconfirmed_start is None

    def test_do_standalone_returns_none(self):
        s = StatementSplitter()
        assert s._handle_nested_block('DO') is None

    def test_if_pushes(self):
        s = StatementSplitter()
        assert s._handle_nested_block('IF') == 1
        assert s._block_stack == ['IF']

    def test_case_pushes(self):
        s = StatementSplitter()
        assert s._handle_nested_block('CASE') == 1
        assert s._block_stack == ['CASE']

    def test_unknown_returns_none(self):
        s = StatementSplitter()
        assert s._handle_nested_block('FOO') is None


# ---------------------------------------------------------------------------
# _handle_closing_keyword
# ---------------------------------------------------------------------------

class TestHandleClosingKeyword:
    def test_end_if(self):
        s = StatementSplitter()
        s._block_stack = ['IF']
        assert s._handle_closing_keyword('END IF') == -1
        assert s._block_stack == []

    def test_end_if_wrong_top(self):
        s = StatementSplitter()
        s._block_stack = ['LOOP']
        assert s._handle_closing_keyword('END IF') == 0
        assert s._block_stack == ['LOOP']

    def test_end_for(self):
        s = StatementSplitter()
        s._block_stack = ['FOR']
        assert s._handle_closing_keyword('END FOR') == -1
        assert s._block_stack == []

    def test_end_while(self):
        s = StatementSplitter()
        s._block_stack = ['WHILE']
        assert s._handle_closing_keyword('END WHILE') == -1
        assert s._block_stack == []

    @pytest.mark.parametrize('top', ['LOOP', 'FOR', 'WHILE'])
    def test_end_loop(self, top):
        s = StatementSplitter()
        s._block_stack = [top]
        assert s._handle_closing_keyword('END LOOP') == -1
        assert s._block_stack == []

    def test_end_loop_wrong_top(self):
        s = StatementSplitter()
        s._block_stack = ['IF']
        assert s._handle_closing_keyword('END LOOP') == 0
        assert s._block_stack == ['IF']

    def test_end_case(self):
        s = StatementSplitter()
        s._block_stack = ['CASE']
        assert s._handle_closing_keyword('END CASE') == -1
        assert s._block_stack == []

    def test_end_pops_any(self):
        s = StatementSplitter()
        s._block_stack = ['BEGIN', 'CASE']
        assert s._handle_closing_keyword('END') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_empty_stack(self):
        s = StatementSplitter()
        assert s._handle_closing_keyword('END') == -1
        assert s._block_stack == []

    def test_unknown_returns_zero(self):
        s = StatementSplitter()
        assert s._handle_closing_keyword('FOO') == 0


# ---------------------------------------------------------------------------
# process: statement splitting
# ---------------------------------------------------------------------------

def tok(ttype, value):
    return (ttype, value)


class TestProcessSplitting:
    def test_simple_semicolon_split(self):
        tokens = [
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
            tok(T.Punctuation, ';'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '2'),
            tok(T.Punctuation, ';'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == ['SELECT 1;', 'SELECT 2;']

    def test_trailing_statement_without_semicolon(self):
        tokens = [
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
            tok(T.Punctuation, ';'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '2'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == ['SELECT 1;', 'SELECT 2']

    def test_whitespace_only_yields_nothing(self):
        tokens = [
            tok(T.Whitespace, ' '),
            tok(T.Newline, '\n'),
            tok(T.Whitespace, '\t'),
        ]
        assert run_splitter(tokens) == []

    def test_empty_stream(self):
        assert run_splitter([]) == []

    def test_semicolon_inside_parentheses_no_split(self):
        tokens = [
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Punctuation, '('),
            tok(T.Number, '1'),
            tok(T.Punctuation, ';'),
            tok(T.Number, '2'),
            tok(T.Punctuation, ')'),
            tok(T.Punctuation, ';'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '3'),
        ]
        statements = run_splitter(tokens)
        # The first ';' inside parens must not split; second one does.
        assert len(statements) == 2
        assert stmt_texts(statements) == ['SELECT(1;2);', 'SELECT 3']

    def test_each_statement_is_statement_instance(self):
        tokens = [
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 1
        assert isinstance(statements[0], sql.Statement)

    def test_go_splits(self):
        tokens = [
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
            tok(T.Whitespace, '\n'),
            tok(T.Keyword, 'GO'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '2'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == ['SELECT 1\nGO', 'SELECT 2']

    def test_go_with_number_splits(self):
        tokens = [
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
            tok(T.Whitespace, '\n'),
            tok(T.Keyword, 'GO 2'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '2'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == ['SELECT 1\nGO 2', 'SELECT 2']


# ---------------------------------------------------------------------------
# process: BEGIN/END blocks
# ---------------------------------------------------------------------------

def build_begin_end_tokens():
    return [
        tok(T.Keyword, 'BEGIN'),
        tok(T.Whitespace, ' '),
        tok(T.Keyword, 'SELECT'),
        tok(T.Whitespace, ' '),
        tok(T.Number, '1'),
        tok(T.Punctuation, ';'),
        tok(T.Whitespace, ' '),
        tok(T.Keyword, 'SELECT'),
        tok(T.Whitespace, ' '),
        tok(T.Number, '2'),
        tok(T.Punctuation, ';'),
        tok(T.Whitespace, ' '),
        tok(T.Keyword, 'END'),
        tok(T.Punctuation, ';'),
        tok(T.Whitespace, ' '),
        tok(T.Keyword, 'SELECT'),
        tok(T.Whitespace, ' '),
        tok(T.Number, '3'),
    ]


class TestProcessBlocks:
    def test_semicolons_inside_begin_do_not_split(self):
        statements = run_splitter(build_begin_end_tokens())
        assert len(statements) == 2
        texts = stmt_texts(statements)
        assert texts[0] == 'BEGIN SELECT 1; SELECT 2; END;'
        assert texts[1] == 'SELECT 3'

    def test_standalone_begin_semicolon(self):
        tokens = [
            tok(T.Keyword, 'BEGIN'),
            tok(T.Punctuation, ';'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == ['BEGIN;', 'SELECT 1']

    def test_begin_transaction_does_not_open_block(self):
        # BEGIN TRANSACTION ... should be treated as a transaction statement.
        tokens = [
            tok(T.Keyword, 'BEGIN'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'TRANSACTION'),
            tok(T.Punctuation, ';'),
            tok(T.Keyword, 'COMMIT'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == ['BEGIN TRANSACTION;', 'COMMIT']

    def test_nested_if_end_if_no_split_inside(self):
        tokens = [
            tok(T.Keyword, 'BEGIN'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'IF'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
            tok(T.Punctuation, ';'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'END IF'),
            tok(T.Punctuation, ';'),
            tok(T.Keyword, 'END'),
            tok(T.Punctuation, ';'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '9'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == [
            'BEGIN IF 1; END IF; END;',
            'SELECT 9',
        ]


# ---------------------------------------------------------------------------
# process: CREATE ... DECLARE ... BEGIN ... END
# ---------------------------------------------------------------------------

class TestProcessCreateBlock:
    def test_create_with_declare_begin_end(self):
        tokens = [
            tok(T.Keyword.DDL, 'CREATE'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'PROCEDURE'),
            tok(T.Whitespace, ' '),
            tok(T.Name, 'p'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'DECLARE'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'BEGIN'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
            tok(T.Punctuation, ';'),
            tok(T.Whitespace, ' '),
            tok(T.Keyword, 'END'),
            tok(T.Punctuation, ';'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '2'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == [
            'CREATE PROCEDURE p DECLARE BEGIN SELECT 1; END;',
            'SELECT 2',
        ]

    def test_declare_without_create_ignored(self):
        tokens = [
            tok(T.Keyword, 'DECLARE'),
            tok(T.Punctuation, ';'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
        ]
        statements = run_splitter(tokens)
        # DECLARE did not open a block, so the ';' splits.
        assert len(statements) == 2
        assert stmt_texts(statements) == ['DECLARE;', 'SELECT 1']


# ---------------------------------------------------------------------------
# process: consume_ws behavior at boundaries
# ---------------------------------------------------------------------------

class TestConsumeWhitespace:
    def test_whitespace_after_semicolon_kept_in_previous_statement(self):
        # After a ';' triggers consume_ws, the following whitespace tokens
        # are delayed into the previous statement before the next real token.
        tokens = [
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
            tok(T.Punctuation, ';'),
            tok(T.Whitespace, ' '),
            tok(T.Newline, '\n'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '2'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == ['SELECT 1; \n', 'SELECT 2']

    def test_comment_after_semicolon_kept_in_previous_statement(self):
        tokens = [
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '1'),
            tok(T.Punctuation, ';'),
            tok(T.Comment.Single, '-- note'),
            tok(T.Keyword, 'SELECT'),
            tok(T.Whitespace, ' '),
            tok(T.Number, '2'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == ['SELECT 1;-- note', 'SELECT 2']


# ---------------------------------------------------------------------------
# process: reset between statements
# ---------------------------------------------------------------------------

class TestResetBetweenStatements:
    def test_state_reset_for_next_statement(self):
        tokens = [
            tok(T.Punctuation, '('),
            tok(T.Number, '1'),
            tok(T.Punctuation, ')'),
            tok(T.Punctuation, ';'),
            tok(T.Punctuation, '('),
            tok(T.Number, '2'),
            tok(T.Punctuation, ')'),
        ]
        statements = run_splitter(tokens)
        assert len(statements) == 2
        assert stmt_texts(statements) == ['(1);', '(2)']
