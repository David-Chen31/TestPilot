import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stream(*pairs):
    """Build a list of (ttype, value) tuples."""
    return list(pairs)


def split_sql(text):
    """Tokenize free text using sqlparse's tokenizer and split it."""
    from sqlparse import lexer
    stream = lexer.tokenize(text)
    return list(StatementSplitter().process(stream))


def statement_texts(text):
    return [str(stmt).strip() for stmt in split_sql(text)]


# ---------------------------------------------------------------------------
# Construction / reset state
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_initial_state(self):
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
        sp.tokens = [sql.Token(T.Text, 'x')]
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
    def test_semicolon_no_begin(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Punctuation, ';') == 0
        assert sp._unconfirmed_start is None
        assert sp._seen_begin is False

    def test_semicolon_resets_unconfirmed_start(self):
        sp = StatementSplitter()
        sp._unconfirmed_start = 'FOR'
        assert sp._change_splitlevel(T.Punctuation, ';') == 0
        assert sp._unconfirmed_start is None

    def test_semicolon_standalone_begin_pops_block(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._seen_begin = True
        assert sp._change_splitlevel(T.Punctuation, ';') == -1
        assert sp._block_stack == []
        assert sp._seen_begin is False

    def test_semicolon_seen_begin_no_begin_on_stack(self):
        sp = StatementSplitter()
        sp._seen_begin = True
        assert sp._change_splitlevel(T.Punctuation, ';') == 0
        assert sp._seen_begin is False

    def test_open_parenthesis(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Punctuation, '(') == 1
        assert sp._parenthesis_level == 1
        assert sp._change_splitlevel(T.Punctuation, '(') == 1
        assert sp._parenthesis_level == 2

    def test_close_parenthesis(self):
        sp = StatementSplitter()
        sp._parenthesis_level = 2
        assert sp._change_splitlevel(T.Punctuation, ')') == -1
        assert sp._parenthesis_level == 1

    def test_close_parenthesis_never_negative(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Punctuation, ')') == -1
        assert sp._parenthesis_level == 0

    def test_non_keyword_token_returns_zero(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Name, 'foo') == 0
        assert sp._change_splitlevel(T.Number.Integer, '42') == 0
        assert sp._change_splitlevel(T.Whitespace, ' ') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: keyword handling
# ---------------------------------------------------------------------------

class TestChangeSplitLevelKeywords:
    def test_create_sets_is_create(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword.DDL, 'CREATE') == 0
        assert sp._is_create is True

    def test_create_with_trailing_words(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword.DDL, 'CREATE or replace') == 0
        assert sp._is_create is True

    def test_create_case_insensitive(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword.DDL, 'create') == 0
        assert sp._is_create is True

    def test_declare_outside_create_is_ignored(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert sp._block_stack == []

    def test_declare_inside_create_starts_block(self):
        sp = StatementSplitter()
        sp._is_create = True
        assert sp._change_splitlevel(T.Keyword, 'DECLARE') == 1
        assert sp._block_stack == ['DECLARE']

    def test_declare_inside_create_with_existing_block_ignored(self):
        sp = StatementSplitter()
        sp._is_create = True
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert sp._block_stack == ['BEGIN']

    def test_begin_starts_block(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword, 'BEGIN') == 1
        assert sp._block_stack == ['BEGIN']
        assert sp._seen_begin is True

    def test_begin_after_declare_transitions(self):
        sp = StatementSplitter()
        sp._block_stack = ['DECLARE']
        assert sp._change_splitlevel(T.Keyword, 'BEGIN') == 0
        assert sp._block_stack == ['BEGIN']
        assert sp._seen_begin is True

    def test_begin_case_insensitive(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword, 'begin') == 1
        assert sp._block_stack == ['BEGIN']

    @pytest.mark.parametrize('word', [
        'TRANSACTION', 'WORK', 'TRAN', 'DISTRIBUTED',
        'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE',
    ])
    def test_transaction_keyword_after_begin(self, word):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._seen_begin = True
        assert sp._change_splitlevel(T.Keyword, word) == -1
        assert sp._block_stack == []
        assert sp._seen_begin is False

    def test_transaction_keyword_as_name(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._seen_begin = True
        assert sp._change_splitlevel(T.Name, 'WORK') == -1
        assert sp._block_stack == []

    def test_transaction_keyword_without_begin_on_stack(self):
        sp = StatementSplitter()
        sp._seen_begin = True
        assert sp._change_splitlevel(T.Keyword, 'TRANSACTION') == 0
        assert sp._seen_begin is False
        assert sp._block_stack == []

    def test_transaction_keyword_without_seen_begin_ignored(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        sp._seen_begin = False
        # not inside 'BEGIN'? it is, so _handle_nested_block is used -> None
        assert sp._change_splitlevel(T.Keyword, 'TRANSACTION') == 0
        assert sp._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# Nested block handling
# ---------------------------------------------------------------------------

class TestNestedBlocks:
    def _with_begin(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        return sp

    def test_for_sets_unconfirmed(self):
        sp = self._with_begin()
        assert sp._change_splitlevel(T.Keyword, 'FOR') == 0
        assert sp._unconfirmed_start == 'FOR'

    def test_while_sets_unconfirmed(self):
        sp = self._with_begin()
        assert sp._change_splitlevel(T.Keyword, 'WHILE') == 0
        assert sp._unconfirmed_start == 'WHILE'

    def test_loop_confirms_pending_for(self):
        sp = self._with_begin()
        sp._change_splitlevel(T.Keyword, 'FOR')
        assert sp._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert sp._block_stack == ['BEGIN', 'FOR']
        assert sp._unconfirmed_start is None

    def test_loop_confirms_pending_while(self):
        sp = self._with_begin()
        sp._change_splitlevel(T.Keyword, 'WHILE')
        assert sp._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert sp._block_stack == ['BEGIN', 'WHILE']

    def test_do_confirms_pending_for(self):
        sp = self._with_begin()
        sp._change_splitlevel(T.Keyword, 'FOR')
        assert sp._change_splitlevel(T.Keyword, 'DO') == 1
        assert sp._block_stack == ['BEGIN', 'FOR']

    def test_standalone_loop(self):
        sp = self._with_begin()
        assert sp._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert sp._block_stack == ['BEGIN', 'LOOP']

    def test_standalone_do_no_op(self):
        sp = self._with_begin()
        assert sp._change_splitlevel(T.Keyword, 'DO') == 0
        assert sp._block_stack == ['BEGIN']

    def test_if_starts_block(self):
        sp = self._with_begin()
        assert sp._change_splitlevel(T.Keyword, 'IF') == 1
        assert sp._block_stack == ['BEGIN', 'IF']

    def test_case_starts_block(self):
        sp = self._with_begin()
        assert sp._change_splitlevel(T.Keyword, 'CASE') == 1
        assert sp._block_stack == ['BEGIN', 'CASE']

    def test_for_outside_begin_ignored(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword, 'FOR') == 0
        assert sp._unconfirmed_start is None


# ---------------------------------------------------------------------------
# Closing keywords
# ---------------------------------------------------------------------------

class TestClosingKeywords:
    def test_end_if(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'IF']
        assert sp._change_splitlevel(T.Keyword, 'END IF') == -1
        assert sp._block_stack == ['BEGIN']

    def test_end_if_mismatch(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'END IF') == 0
        assert sp._block_stack == ['BEGIN']

    def test_end_for(self):
        sp = StatementSplitter()
        sp._block_stack = ['FOR']
        assert sp._change_splitlevel(T.Keyword, 'END FOR') == -1
        assert sp._block_stack == []

    def test_end_for_mismatch(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'END FOR') == 0

    def test_end_while(self):
        sp = StatementSplitter()
        sp._block_stack = ['WHILE']
        assert sp._change_splitlevel(T.Keyword, 'END WHILE') == -1
        assert sp._block_stack == []

    def test_end_while_mismatch(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'END WHILE') == 0

    @pytest.mark.parametrize('top', ['LOOP', 'FOR', 'WHILE'])
    def test_end_loop(self, top):
        sp = StatementSplitter()
        sp._block_stack = [top]
        assert sp._change_splitlevel(T.Keyword, 'END LOOP') == -1
        assert sp._block_stack == []

    def test_end_loop_mismatch(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'END LOOP') == 0

    def test_end_case(self):
        sp = StatementSplitter()
        sp._block_stack = ['CASE']
        assert sp._change_splitlevel(T.Keyword, 'END CASE') == -1
        assert sp._block_stack == []

    def test_end_case_mismatch(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN']
        assert sp._change_splitlevel(T.Keyword, 'END CASE') == 0

    def test_plain_end_pops_stack(self):
        sp = StatementSplitter()
        sp._block_stack = ['BEGIN', 'IF']
        assert sp._change_splitlevel(T.Keyword, 'END') == -1
        assert sp._block_stack == ['BEGIN']

    def test_plain_end_empty_stack(self):
        sp = StatementSplitter()
        assert sp._change_splitlevel(T.Keyword, 'END') == -1
        assert sp._block_stack == []

    def test_unknown_keyword_no_change(self):
        sp = StatementSplitter()
        sp._block_stack = ['IF']
        assert sp._change_splitlevel(T.Keyword, 'FOO') == 0
        assert sp._block_stack == ['IF']


# ---------------------------------------------------------------------------
# process() end-to-end
# ---------------------------------------------------------------------------

class TestProcess:
    def test_single_statement_no_semicolon(self):
        stmts = split_sql('SELECT 1')
        assert len(stmts) == 1
        assert str(stmts[0]).strip() == 'SELECT 1'

    def test_two_statements_split_on_semicolon(self):
        texts = statement_texts('SELECT 1; SELECT 2;')
        assert texts == ['SELECT 1;', 'SELECT 2;']

    def test_empty_stream(self):
        assert list(StatementSplitter().process([])) == []

    def test_whitespace_only_stream(self):
        stmts = split_sql('   \n  ')
        assert stmts == []

    def test_returns_sql_statement_instances(self):
        stmts = split_sql('SELECT 1')
        assert isinstance(stmts[0], sql.Statement)

    def test_semicolon_inside_parentheses_not_split(self):
        texts = statement_texts('SELECT (1; 2); SELECT 3;')
        assert len(texts) == 2

    def test_go_keyword_splits(self):
        stream = make_stream(
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Number.Integer, '1'),
            (T.Newline, '\n'),
            (T.Keyword, 'GO'),
            (T.Whitespace, ' '),
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Number.Integer, '2'),
        )
        stmts = list(StatementSplitter().process(stream))
        assert len(stmts) == 2
        assert 'SELECT 1' in str(stmts[0])
        assert 'SELECT 2' in str(stmts[1])

    def test_go_with_count_splits(self):
        stream = make_stream(
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Number.Integer, '1'),
            (T.Newline, '\n'),
            (T.Keyword, 'GO 2'),
            (T.Whitespace, ' '),
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Number.Integer, '2'),
        )
        stmts = list(StatementSplitter().process(stream))
        assert len(stmts) == 2

    def test_semicolons_inside_begin_end_not_split(self):
        sql_text = (
            "CREATE PROCEDURE p() BEGIN SELECT 1; SELECT 2; END; SELECT 3;"
        )
        texts = statement_texts(sql_text)
        assert len(texts) == 2

    def test_standalone_begin_splits_on_semicolon(self):
        stream = make_stream(
            (T.Keyword, 'BEGIN'),
            (T.Punctuation, ';'),
            (T.Whitespace, ' '),
            (T.Keyword, 'SELECT'),
            (T.Whitespace, ' '),
            (T.Number.Integer, '1'),
        )
        stmts = list(StatementSplitter().process(stream))
        assert len(stmts) == 2
        assert str(stmts[0]).strip() == 'BEGIN;'

    def test_pending_statement_yielded_at_end(self):
        stmts = split_sql('SELECT 1; SELECT 2')
        assert len(stmts) == 2
        assert str(stmts[1]).strip() == 'SELECT 2'

    def test_consume_ws_behavior_after_split(self):
        stream = make_stream(
            (T.Keyword, 'SELECT'),
            (T.Punctuation, ';'),
            (T.Whitespace, ' '),
            (T.Keyword, 'SELECT'),
        )
        stmts = list(StatementSplitter().process(stream))
        # Whitespaces do not start a new statement; the split happens on
        # the following non-whitespace token.
        assert len(stmts) == 2
        assert str(stmts[1]).strip() == 'SELECT'

    def test_no_extra_yield_for_trailing_whitespace(self):
        stream = make_stream(
            (T.Keyword, 'SELECT'),
            (T.Punctuation, ';'),
            (T.Whitespace, ' '),
        )
        stmts = list(StatementSplitter().process(stream))
        assert len(stmts) == 1
        assert str(stmts[0]).strip() == 'SELECT;'

    def test_level_increases_and_decreases_with_parentheses(self):
        sp = StatementSplitter()
        stream = make_stream(
            (T.Punctuation, '('),
            (T.Punctuation, ')'),
        )
        list(sp.process(stream))
        assert sp.level == 0

    def test_process_new_instance_each_time(self):
        a = statement_texts('SELECT 1; SELECT 2;')
        b = statement_texts('SELECT 1; SELECT 2;')
        assert a == b == ['SELECT 1;', 'SELECT 2;']

    def test_multiple_semicolons(self):
        texts = statement_texts('SELECT 1;; SELECT 2;')
        # An empty statement between ;; is still emitted as ";" chunk
        # The key assertion: last statement present.
        assert texts[-1] == 'SELECT 2;'
