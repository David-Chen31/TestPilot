import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


def stmt_texts(stream):
    """Helper: run a token stream through StatementSplitter.process and
    return the list of statements."""
    return list(StatementSplitter().process(stream))


def token_list(*pairs):
    return list(pairs)


# ---------------------------------------------------------------------------
# _reset / __init__
# ---------------------------------------------------------------------------

def test_init_state():
    s = StatementSplitter()
    assert s._block_stack == []
    assert s._parenthesis_level == 0
    assert s._unconfirmed_start is None
    assert s._is_create is False
    assert s._seen_begin is False
    assert s.consume_ws is False
    assert s.tokens == []
    assert s.level == 0


def test_reset_restores_defaults():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    s._parenthesis_level = 3
    s._unconfirmed_start = 'FOR'
    s._is_create = True
    s._seen_begin = True
    s.consume_ws = True
    s.tokens = [sql.Token(T.Text, 'x')]
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
# _change_splitlevel: punctuation
# ---------------------------------------------------------------------------

def test_semicolon_returns_zero_when_no_begin():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, ';') == 0
    assert s._unconfirmed_start is None


def test_semicolon_resets_unconfirmed_start():
    s = StatementSplitter()
    s._unconfirmed_start = 'FOR'
    assert s._change_splitlevel(T.Punctuation, ';') == 0
    assert s._unconfirmed_start is None


def test_semicolon_pops_begin_when_seen():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    s._seen_begin = True
    assert s._change_splitlevel(T.Punctuation, ';') == -1
    assert s._block_stack == []
    assert s._seen_begin is False


def test_semicolon_seen_begin_without_begin_stack():
    s = StatementSplitter()
    s._seen_begin = True
    assert s._change_splitlevel(T.Punctuation, ';') == 0
    assert s._seen_begin is False


def test_open_parenthesis():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, '(') == 1
    assert s._parenthesis_level == 1


def test_close_parenthesis_decrements_not_below_zero():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, ')') == -1
    assert s._parenthesis_level == 0


def test_close_parenthesis_after_open():
    s = StatementSplitter()
    s._change_splitlevel(T.Punctuation, '(')
    assert s._change_splitlevel(T.Punctuation, ')') == -1
    assert s._parenthesis_level == 0


def test_non_keyword_token_returns_zero():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Text, 'foo') == 0
    assert s._change_splitlevel(T.Name, 'bar') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: create / declare
# ---------------------------------------------------------------------------

def test_create_sets_is_create():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword.DDL, 'CREATE TABLE') == 0
    assert s._is_create is True


def test_declare_with_create_opens_block():
    s = StatementSplitter()
    s._is_create = True
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 1
    assert s._block_stack == ['DECLARE']


def test_declare_without_create_is_ignored():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
    assert s._block_stack == []


def test_declare_with_existing_block_is_ignored():
    s = StatementSplitter()
    s._is_create = True
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
    assert s._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# _change_splitlevel: BEGIN
# ---------------------------------------------------------------------------

def test_begin_opens_block():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'BEGIN') == 1
    assert s._block_stack == ['BEGIN']
    assert s._seen_begin is True


def test_begin_transitions_declare():
    s = StatementSplitter()
    s._block_stack.append('DECLARE')
    assert s._change_splitlevel(T.Keyword, 'BEGIN') == 0
    assert s._block_stack == ['BEGIN']
    assert s._seen_begin is True


def test_begin_lowercase():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'begin') == 1
    assert s._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# transaction keywords after BEGIN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kw', [
    'TRANSACTION', 'WORK', 'TRAN', 'DISTRIBUTED',
    'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE',
])
def test_transaction_keyword_after_begin(kw):
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, kw) == -1
    assert s._block_stack == []
    assert s._seen_begin is False


def test_transaction_keyword_after_begin_no_block():
    s = StatementSplitter()
    s._seen_begin = True
    assert s._change_splitlevel(T.Keyword, 'TRANSACTION') == 0
    assert s._seen_begin is False


def test_transaction_keyword_requires_seen_begin():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    # Not seen begin -> falls into nested block handling; TRANSACTION unknown
    assert s._change_splitlevel(T.Keyword, 'TRANSACTION') == 0


# ---------------------------------------------------------------------------
# _handle_nested_block
# ---------------------------------------------------------------------------

def test_handle_nested_for_sets_unconfirmed():
    s = StatementSplitter()
    assert s._handle_nested_block('FOR') == 0
    assert s._unconfirmed_start == 'FOR'


def test_handle_nested_while_sets_unconfirmed():
    s = StatementSplitter()
    assert s._handle_nested_block('WHILE') == 0
    assert s._unconfirmed_start == 'WHILE'


def test_handle_nested_loop_after_for():
    s = StatementSplitter()
    s._unconfirmed_start = 'FOR'
    assert s._handle_nested_block('LOOP') == 1
    assert s._block_stack == ['FOR']
    assert s._unconfirmed_start is None


def test_handle_nested_do_after_while():
    s = StatementSplitter()
    s._unconfirmed_start = 'WHILE'
    assert s._handle_nested_block('DO') == 1
    assert s._block_stack == ['WHILE']
    assert s._unconfirmed_start is None


def test_handle_nested_loop_without_unconfirmed():
    s = StatementSplitter()
    assert s._handle_nested_block('LOOP') == 1
    assert s._block_stack == ['LOOP']


def test_handle_nested_do_without_unconfirmed():
    s = StatementSplitter()
    assert s._handle_nested_block('DO') is None


def test_handle_nested_do_with_unconfirmed_wrong():
    s = StatementSplitter()
    s._unconfirmed_start = 'LOOP'
    assert s._handle_nested_block('DO') is None


def test_handle_nested_if():
    s = StatementSplitter()
    assert s._handle_nested_block('IF') == 1
    assert s._block_stack == ['IF']


def test_handle_nested_case():
    s = StatementSplitter()
    assert s._handle_nested_block('CASE') == 1
    assert s._block_stack == ['CASE']


def test_handle_nested_unknown_returns_none():
    s = StatementSplitter()
    assert s._handle_nested_block('SELECT') is None


# ---------------------------------------------------------------------------
# _handle_closing_keyword
# ---------------------------------------------------------------------------

def test_close_end_if():
    s = StatementSplitter()
    s._block_stack.append('IF')
    assert s._handle_closing_keyword('END IF') == -1
    assert s._block_stack == []


def test_close_end_if_wrong_top():
    s = StatementSplitter()
    s._block_stack.append('CASE')
    assert s._handle_closing_keyword('END IF') == 0
    assert s._block_stack == ['CASE']


def test_close_end_if_empty_stack():
    s = StatementSplitter()
    assert s._handle_closing_keyword('END IF') == 0


def test_close_end_for():
    s = StatementSplitter()
    s._block_stack.append('FOR')
    assert s._handle_closing_keyword('END FOR') == -1
    assert s._block_stack == []


def test_close_end_while():
    s = StatementSplitter()
    s._block_stack.append('WHILE')
    assert s._handle_closing_keyword('END WHILE') == -1
    assert s._block_stack == []


@pytest.mark.parametrize('top', ['LOOP', 'FOR', 'WHILE'])
def test_close_end_loop(top):
    s = StatementSplitter()
    s._block_stack.append(top)
    assert s._handle_closing_keyword('END LOOP') == -1
    assert s._block_stack == []


def test_close_end_loop_wrong_top():
    s = StatementSplitter()
    s._block_stack.append('IF')
    assert s._handle_closing_keyword('END LOOP') == 0
    assert s._block_stack == ['IF']


def test_close_end_case():
    s = StatementSplitter()
    s._block_stack.append('CASE')
    assert s._handle_closing_keyword('END CASE') == -1
    assert s._block_stack == []


def test_close_end_pops_any():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    assert s._handle_closing_keyword('END') == -1
    assert s._block_stack == []


def test_close_end_empty_stack():
    s = StatementSplitter()
    assert s._handle_closing_keyword('END') == -1
    assert s._block_stack == []


def test_close_unknown_returns_zero():
    s = StatementSplitter()
    assert s._handle_closing_keyword('FOO') == 0


# ---------------------------------------------------------------------------
# nested block via _change_splitlevel (only inside BEGIN)
# ---------------------------------------------------------------------------

def test_nested_structure_only_inside_begin():
    s = StatementSplitter()
    # Outside BEGIN, IF goes to closing handler -> 0
    assert s._change_splitlevel(T.Keyword, 'IF') == 0
    assert s._block_stack == []

    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, 'IF') == 1
    assert s._block_stack == ['BEGIN', 'IF']


# ---------------------------------------------------------------------------
# process: simple splitting
# ---------------------------------------------------------------------------

def test_process_single_statement():
    stream = token_list(
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Text, '1'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 1
    assert all(isinstance(st, sql.Statement) for st in stmts)
    assert 'SELECT 1' == str(stmts[0])


def test_process_splits_on_semicolon():
    stream = token_list(
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Text, '1'),
        (T.Punctuation, ';'),
        (T.Newline, '\n'),
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Text, '2'),
        (T.Punctuation, ';'),
    )
    stmts = list(StatementSplitter().process(stream))
    # Trailing whitespace after the first statement goes to the second one
    assert len(stmts) == 2
    assert str(stmts[0]).strip() == 'SELECT 1;'
    assert 'SELECT 2;' in str(stmts[1])


def test_process_three_statements():
    stream = token_list(
        (T.Text, 'a'), (T.Punctuation, ';'),
        (T.Text, 'b'), (T.Punctuation, ';'),
        (T.Text, 'c')
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 3
    assert str(stmts[0]) == 'a;'
    assert str(stmts[1]).strip() == 'b;'
    assert str(stmts[2]) == 'c'


def test_process_empty_stream():
    assert list(StatementSplitter().process([])) == []


def test_process_only_whitespace():
    stream = token_list(
        (T.Whitespace, ' '),
        (T.Newline, '\n'),
    )
    assert list(StatementSplitter().process(stream)) == []


def test_process_whitespace_tokens_not_split():
    # A whitespace token should not trigger a split.
    stream = token_list(
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Whitespace, ' '),
        (T.Newline, '\n'),
        (T.Text, 'b'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 2


def test_process_comment_single_is_eos_token():
    stream = token_list(
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Comment.Single, '-- c'),
        (T.Text, 'b'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 2


# ---------------------------------------------------------------------------
# process: GO splitter
# ---------------------------------------------------------------------------

def test_process_go_splits():
    stream = token_list(
        (T.Keyword, 'GO'),
        (T.Newline, '\n'),
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Text, '1'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 2
    assert str(stmts[0]).strip() == 'GO'


def test_process_go_with_count_splits():
    stream = token_list(
        (T.Keyword, 'GO 2'),
        (T.Newline, '\n'),
        (T.Text, 'x'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 2


def test_process_goto_does_not_split():
    stream = token_list(
        (T.Keyword, 'GOTO'),
        (T.Whitespace, ' '),
        (T.Text, 'label'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 1


# ---------------------------------------------------------------------------
# process: BEGIN ... END blocks
# ---------------------------------------------------------------------------

def test_process_begin_end_single_statement():
    stream = token_list(
        (T.Keyword, 'BEGIN'),
        (T.Whitespace, ' '),
        (T.Text, 'x'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
        (T.Punctuation, ';'),
    )
    stmts = list(StatementSplitter().process(stream))
    # The semicolon inside BEGIN shouldn't split; final ';' after END closes.
    assert len(stmts) == 1


def test_process_standalone_begin_semicolon_splits():
    stream = token_list(
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Newline, '\n'),
        (T.Text, 'y'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 2


def test_process_begin_transaction():
    stream = token_list(
        (T.Keyword, 'BEGIN'),
        (T.Whitespace, ' '),
        (T.Keyword, 'TRANSACTION'),
        (T.Punctuation, ';'),
        (T.Text, 'z'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 2


def test_process_parentheses_do_not_split():
    stream = token_list(
        (T.Punctuation, '('),
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Text, 'b'),
        (T.Punctuation, ')'),
        (T.Punctuation, ';'),
    )
    stmts = list(StatementSplitter().process(stream))
    # The first ';' is inside parentheses (level 1), so no split there.
    assert len(stmts) == 1


# ---------------------------------------------------------------------------
# process: reset between statements
# ---------------------------------------------------------------------------

def test_process_state_reset_after_yield():
    s = StatementSplitter()
    stream = token_list(
        (T.Text, 'a'),
        (T.Punctuation, ';'),
        (T.Text, 'b'),
    )
    gen = s.process(stream)
    first = next(gen)
    assert str(first) == 'a;'
    # State should have been reset before yielding the next statement
    second = next(gen)
    assert str(second) == 'b'
    with pytest.raises(StopIteration):
        next(gen)


def test_seen_begin_reset_on_non_keyword_token():
    s = StatementSplitter()
    s._seen_begin = True
    stream = token_list(
        (T.Text, 'foo'),
    )
    list(s.process(stream))
    assert s._seen_begin is False


def test_seen_begin_not_reset_for_begin_keyword():
    # A BEGIN keyword itself keeps the flag; then a whitespace token then
    # something like END triggers handling. We check via side effects on
    # split count.
    stream = token_list(
        (T.Keyword, 'BEGIN'),
        (T.Whitespace, ' '),
        (T.Keyword, 'END'),
        (T.Punctuation, ';'),
        (T.Text, 'next'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 2


# ---------------------------------------------------------------------------
# process: yielded objects
# ---------------------------------------------------------------------------

def test_process_yields_statement_objects():
    stream = token_list(
        (T.Text, 'x'),
        (T.Punctuation, ';'),
        (T.Text, 'y'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 2
    assert all(isinstance(st, sql.Statement) for st in stmts)


def test_process_trailing_semicolon_does_not_emit_empty():
    stream = token_list(
        (T.Text, 'x'),
        (T.Punctuation, ';'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 1
    assert str(stmts[0]) == 'x;'


def test_process_multiline_comment_does_not_split():
    stream = token_list(
        (T.Text, 'x'),
        (T.Punctuation, ';'),
        (T.Comment.Multiline, '/* c */'),
        (T.Text, 'y'),
    )
    stmts = list(StatementSplitter().process(stream))
    assert len(stmts) == 2
