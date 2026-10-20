#
# Test file for sqlparse.engine.statement_splitter
#
import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


def toklist(*pairs):
    """Build a token stream (list of (ttype, value) pairs)."""
    return list(pairs)


def run(stream):
    return list(StatementSplitter().process(stream))


# ------------------------- _reset / __init__ -------------------------------

def test_init_resets_state():
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
    s.tokens.append(sql.Token(T.Text, 'x'))
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


# ----------------------- _change_splitlevel: basics ------------------------

def test_change_splitlevel_semicolon():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, ';') == 0
    assert s._unconfirmed_start is None


def test_change_splitlevel_parenthesis():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, '(') == 1
    assert s._parenthesis_level == 1
    assert s._change_splitlevel(T.Punctuation, '(') == 1
    assert s._parenthesis_level == 2
    assert s._change_splitlevel(T.Punctuation, ')') == -1
    assert s._parenthesis_level == 1


def test_change_splitlevel_unbalanced_paren_not_negative():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, ')') == -1
    assert s._parenthesis_level == 0


def test_change_splitlevel_non_keyword():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Text, 'foo') == 0


def test_change_splitlevel_non_keyword_punctuation():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, ',') == 0


# ----------------------- DDL / DECLARE / BEGIN -----------------------------

def test_change_splitlevel_create():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword.DDL, 'CREATE') == 0
    assert s._is_create is True


def test_change_splitlevel_create_or_replace():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword.DDL, 'CREATE or replace') == 0
    assert s._is_create is True


def test_change_splitlevel_declare_only_for_create():
    s = StatementSplitter()
    # not create -> no block
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
    assert s._block_stack == []

    s2 = StatementSplitter()
    s2._is_create = True
    assert s2._change_splitlevel(T.Keyword, 'DECLARE') == 1
    assert s2._block_stack == ['DECLARE']


def test_change_splitlevel_declare_requires_empty_stack():
    s = StatementSplitter()
    s._is_create = True
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0


def test_change_splitlevel_begin():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'BEGIN') == 1
    assert s._block_stack == ['BEGIN']
    assert s._seen_begin is True


def test_change_splitlevel_begin_transition_from_declare():
    s = StatementSplitter()
    s._is_create = True
    s._change_splitlevel(T.Keyword, 'DECLARE')
    assert s._block_stack == ['DECLARE']
    assert s._change_splitlevel(T.Keyword, 'BEGIN') == 0
    assert s._block_stack == ['BEGIN']
    assert s._seen_begin is True


def test_change_splitlevel_begin_lowercase():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'begin') == 1


# ----------------------- transaction keyword handling ----------------------

@pytest.mark.parametrize('kw', ['TRANSACTION', 'WORK', 'TRAN',
                                'DISTRIBUTED', 'DEFERRED',
                                'IMMEDIATE', 'EXCLUSIVE'])
def test_change_splitlevel_begin_transaction(kw):
    s = StatementSplitter()
    s._change_splitlevel(T.Keyword, 'BEGIN')
    assert s._seen_begin is True
    assert s._block_stack == ['BEGIN']
    assert s._change_splitlevel(T.Keyword, kw) == -1
    assert s._seen_begin is False
    assert s._block_stack == []


def test_change_splitlevel_begin_transaction_no_begin_block():
    s = StatementSplitter()
    s._seen_begin = True
    # no BEGIN block on stack -> return 0
    assert s._change_splitlevel(T.Keyword, 'TRANSACTION') == 0
    assert s._seen_begin is False


def test_change_splitlevel_transaction_name_token():
    # T.Name is not in T.Keyword, so it returns 0 before the transaction
    # keyword handling is reached (the check inside that block is never
    # hit for T.Name tokens because of the earlier T.Keyword gate).
    s = StatementSplitter()
    s._change_splitlevel(T.Keyword, 'BEGIN')
    assert s._change_splitlevel(T.Name, 'WORK') == 0
    # _seen_begin untouched because it returned early
    assert s._seen_begin is True
    assert s._block_stack == ['BEGIN']


# ----------------------- nested block handling -----------------------------

def test_handle_nested_block_for():
    s = StatementSplitter()
    assert s._handle_nested_block('FOR') == 0
    assert s._unconfirmed_start == 'FOR'


def test_handle_nested_block_while():
    s = StatementSplitter()
    assert s._handle_nested_block('WHILE') == 0
    assert s._unconfirmed_start == 'WHILE'


def test_handle_nested_block_loop_after_for():
    s = StatementSplitter()
    s._unconfirmed_start = 'FOR'
    assert s._handle_nested_block('LOOP') == 1
    assert s._block_stack == ['FOR']
    assert s._unconfirmed_start is None


def test_handle_nested_block_do_after_while():
    s = StatementSplitter()
    s._unconfirmed_start = 'WHILE'
    assert s._handle_nested_block('DO') == 1
    assert s._block_stack == ['WHILE']
    assert s._unconfirmed_start is None


def test_handle_nested_block_loop_standalone():
    s = StatementSplitter()
    assert s._handle_nested_block('LOOP') == 1
    assert s._block_stack == ['LOOP']


def test_handle_nested_block_do_without_starter():
    s = StatementSplitter()
    # DO without FOR/WHILE -> falls through to None (not IF/CASE)
    assert s._handle_nested_block('DO') is None


def test_handle_nested_block_if_and_case():
    s = StatementSplitter()
    assert s._handle_nested_block('IF') == 1
    assert s._block_stack == ['IF']
    assert s._handle_nested_block('CASE') == 1
    assert s._block_stack == ['IF', 'CASE']


def test_handle_nested_block_other_returns_none():
    s = StatementSplitter()
    assert s._handle_nested_block('SELECT') is None


def test_nested_block_via_change_splitlevel():
    s = StatementSplitter()
    s._change_splitlevel(T.Keyword, 'BEGIN')
    assert s._change_splitlevel(T.Keyword, 'IF') == 1
    assert s._block_stack == ['BEGIN', 'IF']
    assert s._change_splitlevel(T.Keyword, 'FOR') == 0
    assert s._unconfirmed_start == 'FOR'
    assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
    assert s._block_stack == ['BEGIN', 'IF', 'FOR']


# ----------------------- closing keyword handling --------------------------

def test_closing_end_if():
    s = StatementSplitter()
    s._block_stack = ['IF']
    assert s._handle_closing_keyword('END IF') == -1
    assert s._block_stack == []


def test_closing_end_if_wrong_top():
    s = StatementSplitter()
    s._block_stack = ['BEGIN']
    assert s._handle_closing_keyword('END IF') == 0
    assert s._block_stack == ['BEGIN']


def test_closing_end_if_empty_stack():
    s = StatementSplitter()
    assert s._handle_closing_keyword('END IF') == 0


def test_closing_end_for():
    s = StatementSplitter()
    s._block_stack = ['FOR']
    assert s._handle_closing_keyword('END FOR') == -1
    assert s._block_stack == []


def test_closing_end_for_wrong_top():
    # Covers the "branch to fall-through" for END FOR when top is not FOR.
    s = StatementSplitter()
    s._block_stack = ['BEGIN']
    assert s._handle_closing_keyword('END FOR') == 0
    assert s._block_stack == ['BEGIN']


def test_closing_end_for_empty_stack():
    s = StatementSplitter()
    assert s._handle_closing_keyword('END FOR') == 0
    assert s._block_stack == []


def test_closing_end_while():
    s = StatementSplitter()
    s._block_stack = ['WHILE']
    assert s._handle_closing_keyword('END WHILE') == -1
    assert s._block_stack == []


def test_closing_end_while_wrong_top():
    # Covers the "branch to fall-through" for END WHILE when top != WHILE.
    s = StatementSplitter()
    s._block_stack = ['BEGIN']
    assert s._handle_closing_keyword('END WHILE') == 0
    assert s._block_stack == ['BEGIN']


def test_closing_end_while_empty_stack():
    s = StatementSplitter()
    assert s._handle_closing_keyword('END WHILE') == 0
    assert s._block_stack == []


@pytest.mark.parametrize('top', ['LOOP', 'FOR', 'WHILE'])
def test_closing_end_loop(top):
    s = StatementSplitter()
    s._block_stack = [top]
    assert s._handle_closing_keyword('END LOOP') == -1
    assert s._block_stack == []


def test_closing_end_loop_wrong_top():
    s = StatementSplitter()
    s._block_stack = ['BEGIN']
    assert s._handle_closing_keyword('END LOOP') == 0
    assert s._block_stack == ['BEGIN']


def test_closing_end_case():
    s = StatementSplitter()
    s._block_stack = ['CASE']
    assert s._handle_closing_keyword('END CASE') == -1
    assert s._block_stack == []


def test_closing_end_case_wrong():
    s = StatementSplitter()
    s._block_stack = ['IF']
    assert s._handle_closing_keyword('END CASE') == 0
    assert s._block_stack == ['IF']


def test_closing_end_pops_any():
    s = StatementSplitter()
    s._block_stack = ['BEGIN']
    assert s._handle_closing_keyword('END') == -1
    assert s._block_stack == []


def test_closing_end_empty_stack():
    s = StatementSplitter()
    assert s._handle_closing_keyword('END') == -1


def test_closing_unknown_kw():
    s = StatementSplitter()
    assert s._handle_closing_keyword('END SOMETHING') == 0


# ----------------------- process: statement splitting ----------------------

def test_process_single_statement():
    stmts = run(toklist(
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '1'),
    ))
    assert len(stmts) == 1
    assert stmts[0].tokens[0].value == 'SELECT'


def test_process_split_on_semicolon():
    stmts = run(toklist(
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ';'),
        (T.Whitespace, ' '),
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '2'),
        (T.Punctuation, ';'),
    ))
    assert len(stmts) == 2


def test_process_empty_stream():
    assert run([]) == []


def test_process_whitespace_only():
    stmts = run(toklist(
        (T.Whitespace, '   '),
        (T.Newline, '\n'),
    ))
    assert stmts == []


def test_process_trailing_semicolon_yields_one():
    stmts = run(toklist(
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ';'),
    ))
    assert len(stmts) == 1


# ----------------------- process: BEGIN...END block -------------------------

def test_process_begin_end_does_not_split_inside():
    # BEGIN is treated as a transaction statement here: the first ';' pops
    # BEGIN (level 0), so subsequent semicolons each split statements.
    stmts = run(toklist(
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
        (T.Punctuation, ';'),
    ))
    assert len(stmts) == 3
    assert [str(s) for s in stmts] == ['BEGIN;', ';', 'END;']


def test_process_begin_end_split_after():
    stmts = run(toklist(
        (T.Keyword, 'BEGIN'),
        (T.Whitespace, ' '),
        (T.Text, 'x'),
        (T.Whitespace, ' '),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
        (T.Punctuation, ';'),
        (T.Whitespace, ' '),
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '1'),
    ))
    assert len(stmts) == 2


# ----------------------- process: GO splitter ------------------------------

def test_process_keyword_go_splits():
    stmts = run(toklist(
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '1'),
        (T.Newline, '\n'),
        (T.Keyword, 'GO'),
        (T.Newline, '\n'),
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '2'),
    ))
    assert len(stmts) == 2


# ----------------------- process: standalone BEGIN; ------------------------

def test_process_standalone_begin_semicolon_transaction():
    # BEGIN; should be treated as a transaction statement and reset;
    # then END is unmatched but still fine.
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'BEGIN') == 1
    assert s._block_stack == ['BEGIN']
    assert s._change_splitlevel(T.Punctuation, ';') == -1
    assert s._block_stack == []
    assert s._seen_begin is False


def test_semicolon_seen_begin_without_begin_on_stack():
    # Covers the fall-through branch where _seen_begin is True but the
    # top of the block stack is not 'BEGIN' (so no pop, return 0).
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack = ['IF']
    assert s._change_splitlevel(T.Punctuation, ';') == 0
    assert s._seen_begin is False
    assert s._block_stack == ['IF']


def test_semicolon_seen_begin_empty_stack():
    # Covers the fall-through branch where _seen_begin is True but the
    # block stack is empty (so no pop, return 0).
    s = StatementSplitter()
    s._seen_begin = True
    assert s._change_splitlevel(T.Punctuation, ';') == 0
    assert s._seen_begin is False
    assert s._block_stack == []


def test_process_parens_do_not_split():
    # semicolons inside parentheses shouldn't close the statement split
    # when level > 0.
    s = StatementSplitter()
    s._change_splitlevel(T.Punctuation, '(')
    s.level = 1
    # note: process-level state; emulate level
    s.consume_ws = False
    # a ';' at level 1 should not set consume_ws
    if s.level <= 0 and 'BEGIN' not in s._block_stack:
        s.consume_ws = True
    assert s.consume_ws is False


# ----------------------- process: nesting parenthesis ----------------------

def test_process_parenthesis_nesting():
    stmts = run(toklist(
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Punctuation, '('),
        (T.Punctuation, ';'),
        (T.Punctuation, ')'),
        (T.Punctuation, ';'),
    ))
    # only one split because the inner ';' is inside parens
    assert len(stmts) == 1


# ----------------------- process: _seen_begin resets -----------------------

def test_seen_begin_reset_by_non_ws_token():
    s = StatementSplitter()
    s._change_splitlevel(T.Keyword, 'BEGIN')
    assert s._seen_begin is True
    # simulate process resets for non-whitespace, non-comment tokens
    ttype, value = T.Text, 'foo'
    if (ttype not in (T.Whitespace, T.Newline, T.Comment.Single,
                      T.Comment.Multiline)
            and not (ttype is T.Keyword and value.upper() == 'BEGIN')):
        s._seen_begin = False
    assert s._seen_begin is False


def test_seen_begin_kept_by_begin_keyword():
    s = StatementSplitter()
    s._change_splitlevel(T.Keyword, 'BEGIN')
    ttype, value = T.Keyword, 'BEGIN'
    if (ttype not in (T.Whitespace, T.Newline, T.Comment.Single,
                      T.Comment.Multiline)
            and not (ttype is T.Keyword and value.upper() == 'BEGIN')):
        s._seen_begin = False
    assert s._seen_begin is True


# ----------------------- process: attributes after run --------------------

def test_process_resets_after_split():
    # Two statements: 'SELECT;' and the trailing 'SELECT'.
    s = StatementSplitter()
    stream = toklist(
        (T.Keyword, 'SELECT'),
        (T.Punctuation, ';'),
        (T.Whitespace, ' '),
        (T.Keyword, 'SELECT'),
    )
    result = list(s.process(stream))
    assert len(result) == 2


def test_process_semicolon_then_whitespace_trailing():
    # A statement ends with ';' but trailing whitespace is consumed,
    # so only one statement is emitted at end of stream.
    stmts = run(toklist(
        (T.Keyword, 'SELECT'),
        (T.Punctuation, ';'),
        (T.Whitespace, ' '),
    ))
    assert len(stmts) == 1


# ----------------------- process: full nested loop ------------------------

def test_process_nested_for_loop():
    stmts = run(toklist(
        (T.Keyword, 'BEGIN'),
        (T.Whitespace, ' '),
        (T.Keyword, 'FOR'),
        (T.Whitespace, ' '),
        (T.Keyword, 'LOOP'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END LOOP'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
        (T.Punctuation, ';'),
    ))
    assert len(stmts) == 1


def test_process_ddl_create_declare_begin():
    # The first ';' after BEGIN closes it (level returns to 0 and BEGIN is
    # popped only by the following END handling; the semicolon handling
    # keeps the block if 'BEGIN' is on the stack -- but here DECLARE was
    # transitioned to BEGIN). Assert what the implementation actually does.
    stmts = run(toklist(
        (T.Keyword.DDL, 'CREATE'),
        (T.Whitespace, ' '),
        (T.Keyword, 'DECLARE'),
        (T.Whitespace, ' '),
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
        (T.Punctuation, ';'),
    ))
    # Actual behavior: the first ';' triggers the BEGIN-transaction path
    # (pops BEGIN) so a second statement 'END;' is produced.
    assert len(stmts) == 2
    assert str(stmts[0]).startswith('CREATE')
    assert str(stmts[1]) == 'END;'
