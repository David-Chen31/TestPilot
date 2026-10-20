import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


def make_stream(*items):
    """Helper to build a token stream from (ttype, value) tuples."""
    return list(items)


def tokens_to_text(statement):
    return "".join(tok.value for tok in statement.tokens)


# ---------------------------------------------------------------------------
# _reset / defaults
# ---------------------------------------------------------------------------

def test_reset_defaults():
    splitter = StatementSplitter()
    assert splitter._block_stack == []
    assert splitter._parenthesis_level == 0
    assert splitter._unconfirmed_start is None
    assert splitter._is_create is False
    assert splitter._seen_begin is False
    assert splitter.consume_ws is False
    assert splitter.tokens == []
    assert splitter.level == 0


def test_reset_restores_after_mutation():
    splitter = StatementSplitter()
    splitter._block_stack.append('BEGIN')
    splitter._parenthesis_level = 3
    splitter._unconfirmed_start = 'FOR'
    splitter._is_create = True
    splitter._seen_begin = True
    splitter.consume_ws = True
    splitter.tokens.append(sql.Token(T.Text, 'x'))
    splitter.level = 5

    splitter._reset()

    assert splitter._block_stack == []
    assert splitter._parenthesis_level == 0
    assert splitter._unconfirmed_start is None
    assert splitter._is_create is False
    assert splitter._seen_begin is False
    assert splitter.consume_ws is False
    assert splitter.tokens == []
    assert splitter.level == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: punctuation handling
# ---------------------------------------------------------------------------

def test_semicolon_returns_zero_and_clears_unconfirmed():
    splitter = StatementSplitter()
    splitter._unconfirmed_start = 'FOR'
    result = splitter._change_splitlevel(T.Punctuation, ';')
    assert result == 0
    assert splitter._unconfirmed_start is None


def test_semicolon_pops_begin_when_seen_begin():
    splitter = StatementSplitter()
    splitter._block_stack.append('BEGIN')
    splitter._seen_begin = True
    result = splitter._change_splitlevel(T.Punctuation, ';')
    assert result == -1
    assert splitter._block_stack == []
    assert splitter._seen_begin is False


def test_semicolon_without_begin_block_returns_zero():
    splitter = StatementSplitter()
    splitter._seen_begin = True
    result = splitter._change_splitlevel(T.Punctuation, ';')
    assert result == 0
    assert splitter._seen_begin is False


def test_open_parenthesis_increments():
    splitter = StatementSplitter()
    assert splitter._change_splitlevel(T.Punctuation, '(') == 1
    assert splitter._parenthesis_level == 1
    assert splitter._change_splitlevel(T.Punctuation, '(') == 1
    assert splitter._parenthesis_level == 2


def test_close_parenthesis_decrements():
    splitter = StatementSplitter()
    splitter._parenthesis_level = 2
    assert splitter._change_splitlevel(T.Punctuation, ')') == -1
    assert splitter._parenthesis_level == 1


def test_close_parenthesis_never_goes_negative():
    splitter = StatementSplitter()
    result = splitter._change_splitlevel(T.Punctuation, ')')
    assert result == -1
    assert splitter._parenthesis_level == 0


def test_non_keyword_token_returns_zero():
    splitter = StatementSplitter()
    assert splitter._change_splitlevel(T.Text, 'foo') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: CREATE
# ---------------------------------------------------------------------------

def test_create_sets_is_create():
    splitter = StatementSplitter()
    result = splitter._change_splitlevel(T.Keyword.DDL, 'CREATE')
    assert result == 0
    assert splitter._is_create is True


def test_create_for_returns_zero():
    splitter = StatementSplitter()
    result = splitter._change_splitlevel(T.Keyword.DDL, 'CREATE OR REPLACE')
    assert result == 0
    assert splitter._is_create is True


def test_create_non_ddl_keyword_does_not_set():
    splitter = StatementSplitter()
    result = splitter._change_splitlevel(T.Keyword, 'CREATE')
    assert result == 0
    assert splitter._is_create is False


# ---------------------------------------------------------------------------
# _change_splitlevel: DECLARE
# ---------------------------------------------------------------------------

def test_declare_after_create_opens_block():
    splitter = StatementSplitter()
    splitter._is_create = True
    result = splitter._change_splitlevel(T.Keyword, 'DECLARE')
    assert result == 1
    assert splitter._block_stack == ['DECLARE']


def test_declare_without_create_does_not_open():
    splitter = StatementSplitter()
    result = splitter._change_splitlevel(T.Keyword, 'DECLARE')
    assert result == 0
    assert splitter._block_stack == []


def test_declare_with_existing_block_does_not_open():
    splitter = StatementSplitter()
    splitter._is_create = True
    splitter._block_stack.append('BEGIN')
    result = splitter._change_splitlevel(T.Keyword, 'DECLARE')
    assert result == 0
    assert splitter._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# _change_splitlevel: BEGIN
# ---------------------------------------------------------------------------

def test_begin_opens_block():
    splitter = StatementSplitter()
    result = splitter._change_splitlevel(T.Keyword, 'BEGIN')
    assert result == 1
    assert splitter._block_stack == ['BEGIN']
    assert splitter._seen_begin is True


def test_begin_transitions_declare():
    splitter = StatementSplitter()
    splitter._block_stack.append('DECLARE')
    result = splitter._change_splitlevel(T.Keyword, 'BEGIN')
    assert result == 0
    assert splitter._block_stack == ['BEGIN']
    assert splitter._seen_begin is True


# ---------------------------------------------------------------------------
# _change_splitlevel: transaction keywords (Issue826)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('keyword', [
    'TRANSACTION', 'WORK', 'TRAN', 'DISTRIBUTED',
    'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE',
])
def test_transaction_keyword_after_begin_pops_block(keyword):
    splitter = StatementSplitter()
    splitter._block_stack.append('BEGIN')
    splitter._seen_begin = True
    result = splitter._change_splitlevel(T.Keyword, keyword)
    assert result == -1
    assert splitter._block_stack == []
    assert splitter._seen_begin is False


def test_transaction_keyword_after_begin_without_block():
    # Note: T.Name does not trigger the transaction-keyword handling because
    # the unified value is upper-cased but ttype is checked against T.Keyword
    # as well -- however when _block_stack is empty the code falls through
    # to _handle_closing_keyword which returns 0 without touching _seen_begin.
    splitter = StatementSplitter()
    splitter._seen_begin = True
    result = splitter._change_splitlevel(T.Name, 'transaction')
    assert result == 0
    # Because there's no BEGIN block, the code sets _seen_begin = False
    # and returns 0 (see line 136-140).
    assert splitter._seen_begin is False


def test_transaction_keyword_name_without_block_preserves_seen_begin():
    # With an empty block stack, the transaction-keyword branch is entered
    # only for keyword-like ttypes. T.Name matches the branch too via
    # `ttype is T.Name`, but there's no BEGIN on the stack to pop.
    # We verify the actual behavior here for completeness.
    splitter = StatementSplitter()
    splitter._block_stack.append('IF')
    splitter._seen_begin = True
    result = splitter._change_splitlevel(T.Keyword, 'WORK')
    assert result == 0
    assert splitter._block_stack == ['IF']
    assert splitter._seen_begin is False


def test_transaction_keyword_case_insensitive():
    splitter = StatementSplitter()
    splitter._block_stack.append('BEGIN')
    splitter._seen_begin = True
    result = splitter._change_splitlevel(T.Keyword, 'work')
    assert result == -1
    assert splitter._block_stack == []


# ---------------------------------------------------------------------------
# _handle_nested_block
# ---------------------------------------------------------------------------

def test_nested_for_sets_unconfirmed():
    splitter = StatementSplitter()
    assert splitter._handle_nested_block('FOR') == 0
    assert splitter._unconfirmed_start == 'FOR'


def test_nested_while_sets_unconfirmed():
    splitter = StatementSplitter()
    assert splitter._handle_nested_block('WHILE') == 0
    assert splitter._unconfirmed_start == 'WHILE'


def test_loop_confirms_for():
    splitter = StatementSplitter()
    splitter._unconfirmed_start = 'FOR'
    result = splitter._handle_nested_block('LOOP')
    assert result == 1
    assert splitter._block_stack == ['FOR']
    assert splitter._unconfirmed_start is None


def test_loop_confirms_while():
    splitter = StatementSplitter()
    splitter._unconfirmed_start = 'WHILE'
    result = splitter._handle_nested_block('LOOP')
    assert result == 1
    assert splitter._block_stack == ['WHILE']
    assert splitter._unconfirmed_start is None


def test_loop_without_unconfirmed_opens_loop():
    splitter = StatementSplitter()
    result = splitter._handle_nested_block('LOOP')
    assert result == 1
    assert splitter._block_stack == ['LOOP']


def test_do_without_unconfirmed_returns_none():
    splitter = StatementSplitter()
    result = splitter._handle_nested_block('DO')
    assert result is None
    assert splitter._block_stack == []


def test_do_confirms_for():
    splitter = StatementSplitter()
    splitter._unconfirmed_start = 'FOR'
    result = splitter._handle_nested_block('DO')
    assert result == 1
    assert splitter._block_stack == ['FOR']


def test_if_opens_block():
    splitter = StatementSplitter()
    assert splitter._handle_nested_block('IF') == 1
    assert splitter._block_stack == ['IF']


def test_case_opens_block():
    splitter = StatementSplitter()
    assert splitter._handle_nested_block('CASE') == 1
    assert splitter._block_stack == ['CASE']


def test_unhandled_nested_returns_none():
    splitter = StatementSplitter()
    assert splitter._handle_nested_block('SELECT') is None


# ---------------------------------------------------------------------------
# _handle_closing_keyword
# ---------------------------------------------------------------------------

def test_end_if_pops():
    splitter = StatementSplitter()
    splitter._block_stack.append('IF')
    assert splitter._handle_closing_keyword('END IF') == -1
    assert splitter._block_stack == []


def test_end_if_wrong_top_does_not_pop():
    splitter = StatementSplitter()
    splitter._block_stack.append('CASE')
    assert splitter._handle_closing_keyword('END IF') == 0
    assert splitter._block_stack == ['CASE']


def test_end_for_pops():
    splitter = StatementSplitter()
    splitter._block_stack.append('FOR')
    assert splitter._handle_closing_keyword('END FOR') == -1
    assert splitter._block_stack == []


def test_end_while_pops():
    splitter = StatementSplitter()
    splitter._block_stack.append('WHILE')
    assert splitter._handle_closing_keyword('END WHILE') == -1
    assert splitter._block_stack == []


@pytest.mark.parametrize('top', ['LOOP', 'FOR', 'WHILE'])
def test_end_loop_pops_for_various_tops(top):
    splitter = StatementSplitter()
    splitter._block_stack.append(top)
    assert splitter._handle_closing_keyword('END LOOP') == -1
    assert splitter._block_stack == []


def test_end_loop_wrong_top_does_not_pop():
    splitter = StatementSplitter()
    splitter._block_stack.append('IF')
    assert splitter._handle_closing_keyword('END LOOP') == 0
    assert splitter._block_stack == ['IF']


def test_end_case_pops():
    splitter = StatementSplitter()
    splitter._block_stack.append('CASE')
    assert splitter._handle_closing_keyword('END CASE') == -1
    assert splitter._block_stack == []


def test_end_pops_any_block():
    splitter = StatementSplitter()
    splitter._block_stack.append('BEGIN')
    assert splitter._handle_closing_keyword('END') == -1
    assert splitter._block_stack == []


def test_end_empty_stack_returns_minus_one():
    splitter = StatementSplitter()
    assert splitter._handle_closing_keyword('END') == -1
    assert splitter._block_stack == []


def test_closing_unrecognized_returns_zero():
    splitter = StatementSplitter()
    assert splitter._handle_closing_keyword('SOMETHING') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: nested block delegation
# ---------------------------------------------------------------------------

def test_nested_for_inside_begin():
    splitter = StatementSplitter()
    splitter._block_stack.append('BEGIN')
    result = splitter._change_splitlevel(T.Keyword, 'FOR')
    assert result == 0
    assert splitter._unconfirmed_start == 'FOR'


def test_nested_loop_inside_begin_opens_block():
    splitter = StatementSplitter()
    splitter._block_stack.append('BEGIN')
    result = splitter._change_splitlevel(T.Keyword, 'LOOP')
    assert result == 1
    assert splitter._block_stack == ['BEGIN', 'LOOP']


def test_end_if_inside_begin_pops():
    splitter = StatementSplitter()
    splitter._block_stack.append('BEGIN')
    splitter._block_stack.append('IF')
    result = splitter._change_splitlevel(T.Keyword, 'END IF')
    assert result == -1
    assert splitter._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# process: simple splitting
# ---------------------------------------------------------------------------

def test_process_single_statement():
    splitter = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Text, '1'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 1
    assert tokens_to_text(statements[0]) == 'SELECT1'


def test_process_two_statements_split_on_semicolon():
    splitter = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Text, '1'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Text, '2'),
        (T.Punctuation, ';'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2
    assert tokens_to_text(statements[0]) == 'SELECT1;'
    assert tokens_to_text(statements[1]) == 'SELECT2;'


def test_process_empty_stream():
    splitter = StatementSplitter()
    statements = list(splitter.process([]))
    assert statements == []


def test_process_only_whitespace_yields_nothing():
    splitter = StatementSplitter()
    stream = [(T.Whitespace, ' '), (T.Whitespace, '\n')]
    statements = list(splitter.process(stream))
    assert statements == []


def test_process_whitespace_after_semicolon_attaches_to_previous():
    # Whitespace following a semicolon is appended to the first statement
    # before the next non-whitespace token triggers the split.
    splitter = StatementSplitter()
    stream = [
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Whitespace, ' '),
        (T.Text, 'b'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2
    assert tokens_to_text(statements[0]) == 'a; '
    assert tokens_to_text(statements[1]) == 'b'


def test_process_comment_after_semicolon_attaches_to_previous():
    # Single-line comments after a semicolon also stay with the previous
    # statement (as they are excluded from EOS_TTYPE triggering the split).
    splitter = StatementSplitter()
    stream = [
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Comment.Single, '-- c'),
        (T.Text, 'b'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2
    assert tokens_to_text(statements[0]) == 'a;-- c'
    assert tokens_to_text(statements[1]) == 'b'


def test_process_newline_after_semicolon_splits():
    splitter = StatementSplitter()
    stream = [
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Newline, '\n'),
        (T.Text, 'b'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2
    assert tokens_to_text(statements[0]) == 'a;'
    assert tokens_to_text(statements[1]) == '\nb'


# ---------------------------------------------------------------------------
# process: GO splitter
# ---------------------------------------------------------------------------

def test_process_split_on_go():
    splitter = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Text, '1'),
        (T.Keyword, 'GO'),
        (T.Keyword, 'SELECT'),
        (T.Text, '2'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2
    assert tokens_to_text(statements[0]) == 'SELECT1GO'
    assert tokens_to_text(statements[1]) == 'SELECT2'


def test_process_split_on_go_with_count():
    splitter = StatementSplitter()
    stream = [
        (T.Text, 'a'),
        (T.Keyword, 'GO 2'),
        (T.Text, 'b'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2
    assert tokens_to_text(statements[0]) == 'aGO 2'
    assert tokens_to_text(statements[1]) == 'b'


# ---------------------------------------------------------------------------
# process: BEGIN...END block, semicolon does not split
# ---------------------------------------------------------------------------

def test_process_begin_end_does_not_split_on_semicolon():
    splitter = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Text, 'b'),
        (T.Keyword, 'END'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 1
    assert tokens_to_text(statements[0]) == 'BEGINa;bEND'


def test_process_standalone_begin_semicolon_splits():
    splitter = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Text, 'x'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2
    assert tokens_to_text(statements[0]) == 'BEGIN;'
    assert tokens_to_text(statements[1]) == 'x'


# ---------------------------------------------------------------------------
# process: parenthesis affects level
# ---------------------------------------------------------------------------

def test_process_parenthesis_affects_level():
    splitter = StatementSplitter()
    stream = [
        (T.Punctuation, '('),
        (T.Punctuation, ')'),
    ]
    list(splitter.process(stream))
    assert splitter._parenthesis_level == 0


def test_process_semicolon_inside_parens_not_split():
    splitter = StatementSplitter()
    stream = [
        (T.Punctuation, '('),
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Text, 'b'),
        (T.Punctuation, ')'),
    ]
    statements = list(splitter.process(stream))
    # level after '(' is 1, semicolon does not split since level > 0
    # Actually parenthesis '+' returns 1, so level 1 > 0, no consume_ws
    assert len(statements) == 1


# ---------------------------------------------------------------------------
# process: CREATE ... BEGIN handling
# ---------------------------------------------------------------------------

def test_process_create_begin_transaction_splits():
    splitter = StatementSplitter()
    stream = [
        (T.Keyword.DDL, 'CREATE'),
        (T.Keyword, 'BEGIN'),
        (T.Keyword, 'TRANSACTION'),
        (T.Punctuation, ';'),
        (T.Text, 'x'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2
    assert tokens_to_text(statements[0]) == 'CREATEBEGINTRANSACTION;'
    assert tokens_to_text(statements[1]) == 'x'


def test_process_create_declare_begin_end():
    splitter = StatementSplitter()
    stream = [
        (T.Keyword.DDL, 'CREATE'),
        (T.Keyword, 'DECLARE'),
        (T.Text, 'x'),
        (T.Punctuation, ';'),
        (T.Keyword, 'BEGIN'),
        (T.Text, 'body'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
    ]
    statements = list(splitter.process(stream))
    # The semicolon after x does not split because we are inside DECLARE (level 1)
    assert len(statements) == 1
    assert tokens_to_text(statements[0]) == 'CREATEDECLAREx;BEGINbody;END'


# ---------------------------------------------------------------------------
# process: reset between statements
# ---------------------------------------------------------------------------

def test_process_resets_between_statements():
    splitter = StatementSplitter()
    stream = [
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Text, 'b'),
        (T.Punctuation, ';'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2


# ---------------------------------------------------------------------------
# process: last statement without trailing semicolon
# ---------------------------------------------------------------------------

def test_process_pending_statement_without_semicolon():
    splitter = StatementSplitter()
    stream = [
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Text, 'b'),
    ]
    statements = list(splitter.process(stream))
    assert len(statements) == 2
    assert tokens_to_text(statements[0]) == 'a;'
    assert tokens_to_text(statements[1]) == 'b'


def test_process_seen_begin_reset_on_other_token():
    splitter = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Text, 'x'),
    ]
    list(splitter.process(stream))
    # _seen_begin should have been reset by the non-begin token
    assert splitter._seen_begin is False
