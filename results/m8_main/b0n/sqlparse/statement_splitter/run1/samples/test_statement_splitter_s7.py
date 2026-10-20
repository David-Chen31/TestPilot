import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


def tokens_of(statement):
    return [(t.ttype, t.value) for t in statement.tokens]


def split(stream):
    return list(StatementSplitter().process(stream))


# ---------------------------------------------------------------------------
# _reset / initial state
# ---------------------------------------------------------------------------

def test_initial_state():
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
    s._block_stack = ['BEGIN']
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


# ---------------------------------------------------------------------------
# _change_splitlevel: punctuation
# ---------------------------------------------------------------------------

def test_semicolon_returns_zero():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, ';') == 0


def test_parenthesis_increase():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, '(') == 1
    assert s._parenthesis_level == 1


def test_parenthesis_decrease():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, '(') == 1
    assert s._change_splitlevel(T.Punctuation, ')') == -1
    assert s._parenthesis_level == 0


def test_parenthesis_decrease_never_negative():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, ')') == -1
    assert s._parenthesis_level == 0


def test_non_keyword_token_returns_zero():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Name, 'foo') == 0
    assert s._change_splitlevel(T.Number.Integer, '1') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: semicolon with BEGIN
# ---------------------------------------------------------------------------

def test_semicolon_resets_unconfirmed_start():
    s = StatementSplitter()
    s._unconfirmed_start = 'FOR'
    s._change_splitlevel(T.Punctuation, ';')
    assert s._unconfirmed_start is None


def test_semicolon_handles_standalone_begin():
    s = StatementSplitter()
    s._block_stack = ['BEGIN']
    s._seen_begin = True
    assert s._change_splitlevel(T.Punctuation, ';') == -1
    assert s._block_stack == []
    assert s._seen_begin is False


def test_semicolon_with_seen_begin_but_no_begin_block():
    s = StatementSplitter()
    s._block_stack = ['LOOP']
    s._seen_begin = True
    assert s._change_splitlevel(T.Punctuation, ';') == 0
    assert s._seen_begin is False
    assert s._block_stack == ['LOOP']


# ---------------------------------------------------------------------------
# _change_splitlevel: CREATE / DECLARE / BEGIN
# ---------------------------------------------------------------------------

def test_create_sets_is_create():
    s = StatementSplitter()
    s._change_splitlevel(T.Keyword.DDL, 'CREATE OR REPLACE')
    assert s._is_create is True


def test_declare_only_for_create_and_empty_stack():
    s = StatementSplitter()
    # not CREATE -> ignored
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
    assert s._block_stack == []

    s._is_create = True
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 1
    assert s._block_stack == ['DECLARE']


def test_declare_ignored_when_stack_nonempty():
    s = StatementSplitter()
    s._is_create = True
    s._block_stack = ['BEGIN']
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
    assert s._block_stack == ['BEGIN']


def test_begin_plain():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'BEGIN') == 1
    assert s._block_stack == ['BEGIN']
    assert s._seen_begin is True


def test_begin_transitions_from_declare():
    s = StatementSplitter()
    s._is_create = True
    s._change_splitlevel(T.Keyword, 'DECLARE')
    assert s._change_splitlevel(T.Keyword, 'BEGIN') == 0
    assert s._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# _change_splitlevel: transaction keywords after BEGIN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kw', [
    'TRANSACTION', 'WORK', 'TRAN', 'DISTRIBUTED',
    'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE',
])
def test_transaction_keyword_after_begin(kw):
    s = StatementSplitter()
    s._change_splitlevel(T.Keyword, 'BEGIN')
    assert s._change_splitlevel(T.Keyword, kw) == -1
    assert s._block_stack == []
    assert s._seen_begin is False


def test_transaction_keyword_as_name_ttype():
    s = StatementSplitter()
    s._change_splitlevel(T.Keyword, 'BEGIN')
    assert s._change_splitlevel(T.Name, 'work') == -1
    assert s._block_stack == []


def test_transaction_keyword_without_begin_block():
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack = ['LOOP']
    assert s._change_splitlevel(T.Keyword, 'TRANSACTION') == 0
    assert s._seen_begin is False
    assert s._block_stack == ['LOOP']


# ---------------------------------------------------------------------------
# _handle_nested_block
# ---------------------------------------------------------------------------

def test_handle_nested_for():
    s = StatementSplitter()
    assert s._handle_nested_block('FOR') == 0
    assert s._unconfirmed_start == 'FOR'


def test_handle_nested_while():
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


def test_handle_nested_loop_alone():
    s = StatementSplitter()
    assert s._handle_nested_block('LOOP') == 1
    assert s._block_stack == ['LOOP']


def test_handle_nested_do_alone_returns_none():
    s = StatementSplitter()
    assert s._handle_nested_block('DO') is None


def test_handle_nested_if_and_case():
    s = StatementSplitter()
    assert s._handle_nested_block('IF') == 1
    assert s._block_stack == ['IF']
    assert s._handle_nested_block('CASE') == 1
    assert s._block_stack == ['IF', 'CASE']


def test_handle_nested_unknown():
    s = StatementSplitter()
    assert s._handle_nested_block('SOMETHING') is None


# ---------------------------------------------------------------------------
# _handle_closing_keyword
# ---------------------------------------------------------------------------

def test_end_if():
    s = StatementSplitter()
    s._block_stack = ['IF']
    assert s._handle_closing_keyword('END IF') == -1
    assert s._block_stack == []


def test_end_if_mismatch():
    s = StatementSplitter()
    s._block_stack = ['CASE']
    assert s._handle_closing_keyword('END IF') == 0
    assert s._block_stack == ['CASE']


def test_end_for():
    s = StatementSplitter()
    s._block_stack = ['FOR']
    assert s._handle_closing_keyword('END FOR') == -1
    assert s._block_stack == []


def test_end_while():
    s = StatementSplitter()
    s._block_stack = ['WHILE']
    assert s._handle_closing_keyword('END WHILE') == -1
    assert s._block_stack == []


@pytest.mark.parametrize('top', ['LOOP', 'FOR', 'WHILE'])
def test_end_loop(top):
    s = StatementSplitter()
    s._block_stack = [top]
    assert s._handle_closing_keyword('END LOOP') == -1
    assert s._block_stack == []


def test_end_loop_mismatch():
    s = StatementSplitter()
    s._block_stack = ['CASE']
    assert s._handle_closing_keyword('END LOOP') == 0
    assert s._block_stack == ['CASE']


def test_end_case():
    s = StatementSplitter()
    s._block_stack = ['CASE']
    assert s._handle_closing_keyword('END CASE') == -1
    assert s._block_stack == []


def test_end_plain_pops_stack():
    s = StatementSplitter()
    s._block_stack = ['IF']
    assert s._handle_closing_keyword('END') == -1
    assert s._block_stack == []


def test_end_plain_empty_stack():
    s = StatementSplitter()
    assert s._handle_closing_keyword('END') == -1
    assert s._block_stack == []


def test_closing_unknown_keyword():
    s = StatementSplitter()
    assert s._handle_closing_keyword('FOO') == 0


# ---------------------------------------------------------------------------
# process: simple splitting
# ---------------------------------------------------------------------------

def test_empty_stream():
    assert split([]) == []


def test_only_whitespace_not_yielded():
    out = split([(T.Whitespace, '   '), (T.Newline, '\n')])
    assert out == []


def test_single_statement_no_semicolon():
    out = split([(T.Keyword, 'SELECT'), (T.Whitespace, ' '),
                 (T.Number.Integer, '1')])
    assert len(out) == 1
    assert isinstance(out[0], sql.Statement)
    assert tokens_of(out[0]) == [
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Number.Integer, '1'),
    ]


def test_two_statements_split_on_semicolon():
    out = split([
        (T.Keyword, 'SELECT'), (T.Number.Integer, '1'), (T.Punctuation, ';'),
        (T.Whitespace, ' '),
        (T.Keyword, 'SELECT'), (T.Number.Integer, '2'), (T.Punctuation, ';'),
    ])
    assert len(out) == 2
    assert tokens_of(out[0]) == [
        (T.Keyword, 'SELECT'), (T.Number.Integer, '1'), (T.Punctuation, ';'),
    ]
    assert [v for _, v in tokens_of(out[1])] == ['SELECT', '2', ';']


def test_semicolon_followed_by_whitespace_attached_to_first():
    out = split([
        (T.Keyword, 'SELECT'), (T.Punctuation, ';'),
        (T.Whitespace, ' '),
        (T.Keyword, 'SELECT'),
    ])
    assert len(out) == 2
    # whitespace after semicolon belongs to first statement's token list
    assert tokens_of(out[0]) == [
        (T.Keyword, 'SELECT'), (T.Punctuation, ';'), (T.Whitespace, ' '),
    ]
    assert tokens_of(out[1]) == [(T.Keyword, 'SELECT')]


def test_semicolon_trailing_no_second_statement():
    out = split([(T.Keyword, 'SELECT'), (T.Punctuation, ';')])
    assert len(out) == 1
    assert tokens_of(out[0]) == [(T.Keyword, 'SELECT'), (T.Punctuation, ';')]


# ---------------------------------------------------------------------------
# process: GO splitter
# ---------------------------------------------------------------------------

def test_go_splits():
    out = split([
        (T.Keyword, 'SELECT'), (T.Keyword, 'GO'),
        (T.Keyword, 'SELECT'),
    ])
    assert len(out) == 2
    assert [v for _, v in tokens_of(out[0])] == ['SELECT', 'GO']


def test_go_with_count_splits():
    out = split([
        (T.Keyword, 'SELECT'), (T.Keyword, 'GO 2'),
        (T.Keyword, 'SELECT'),
    ])
    assert len(out) == 2
    assert [v for _, v in tokens_of(out[0])] == ['SELECT', 'GO 2']


# ---------------------------------------------------------------------------
# process: semicolon inside parentheses does not split
# ---------------------------------------------------------------------------

def test_semicolon_inside_parentheses_does_not_split():
    out = split([
        (T.Keyword, 'SELECT'),
        (T.Punctuation, '('),
        (T.Punctuation, ';'),
        (T.Punctuation, ')'),
        (T.Punctuation, ';'),
    ])
    assert len(out) == 1


# ---------------------------------------------------------------------------
# process: BEGIN ... END blocks
# ---------------------------------------------------------------------------

def test_semicolon_inside_begin_end_block_not_split():
    out = split([
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
        (T.Punctuation, ';'),
    ])
    # whole thing is one statement
    assert len(out) == 1
    assert out[0].tokens[-1].value == ';'


def test_standalone_begin_semicolon_splits():
    out = split([
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Punctuation, ';'),
    ])
    # BEGIN; is its own statement, then SELECT;
    assert len(out) == 2
    assert [v for _, v in tokens_of(out[0])] == ['BEGIN', ';']


# ---------------------------------------------------------------------------
# process: CREATE ... DECLARE ... BEGIN ...
# ---------------------------------------------------------------------------

def test_create_declare_begin_block():
    out = split([
        (T.Keyword.DDL, 'CREATE'),
        (T.Whitespace, ' '),
        (T.Keyword, 'DECLARE'),
        (T.Punctuation, ';'),
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
        (T.Punctuation, ';'),
    ])
    assert len(out) == 1


# ---------------------------------------------------------------------------
# process reuses splitter state across calls to yield pending statement
# ---------------------------------------------------------------------------

def test_pending_statement_yielded_at_end():
    s = StatementSplitter()
    out = list(s.process([(T.Keyword, 'SELECT'), (T.Number.Integer, '1')]))
    assert len(out) == 1
    assert tokens_of(out[0]) == [
        (T.Keyword, 'SELECT'), (T.Number.Integer, '1'),
    ]


def test_generator_can_be_partially_consumed():
    s = StatementSplitter()
    gen = s.process([
        (T.Keyword, 'SELECT'), (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'), (T.Punctuation, ';'),
    ])
    first = next(gen)
    assert [v for _, v in tokens_of(first)] == ['SELECT', ';']
    # continue consuming the rest
    remaining = list(gen)
    assert len(remaining) == 1
    assert [v for _, v in tokens_of(remaining[0])] == ['SELECT', ';']
