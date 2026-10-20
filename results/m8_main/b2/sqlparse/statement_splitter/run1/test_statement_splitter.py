#
# pytest tests for sqlparse.engine.statement_splitter
#
import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def toks(*pairs):
    """Build a small stream of (ttype, value) pairs."""
    return list(pairs)


def process(splitter, stream):
    """Run process and materialize both statements and remaining token list."""
    return list(splitter.process(iter(stream)))


def _values(statements):
    return [st.value for st in statements]


# ---------------------------------------------------------------------------
# Construction / _reset
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
    s._block_stack.append('BEGIN')
    s._parenthesis_level = 3
    s._unconfirmed_start = 'FOR'
    s._is_create = True
    s._seen_begin = True
    s.consume_ws = True
    s.tokens = [sql.Token(T.Keyword, 'SELECT')]
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
# _change_splitlevel: semicolon handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('syntax_ok', [True, False])
def test_change_splitlevel_semicolon_returns_zero(syntax_ok):
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, ';') == 0


def test_change_splitlevel_semicolon_resets_unconfirmed():
    s = StatementSplitter()
    s._unconfirmed_start = 'FOR'
    assert s._change_splitlevel(T.Punctuation, ';') == 0
    assert s._unconfirmed_start is None


def test_change_splitlevel_standalone_begin_semicolon():
    s = StatementSplitter()
    # Simulate BEGIN seen
    s._seen_begin = True
    s._block_stack.append('BEGIN')
    s.level = 1
    ret = s._change_splitlevel(T.Punctuation, ';')
    assert ret == -1
    assert s._block_stack == []
    assert s._seen_begin is False


def test_change_splitlevel_semicolon_with_non_begin_top():
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack.append('FOR')
    ret = s._change_splitlevel(T.Punctuation, ';')
    assert ret == 0
    assert s._block_stack == ['FOR']
    assert s._seen_begin is False


# ---------------------------------------------------------------------------
# _change_splitlevel: parentheses
# ---------------------------------------------------------------------------

def test_change_splitlevel_open_paren():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, '(') == 1
    assert s._parenthesis_level == 1
    assert s._change_splitlevel(T.Punctuation, '(') == 1
    assert s._parenthesis_level == 2


def test_change_splitlevel_close_paren():
    s = StatementSplitter()
    s._parenthesis_level = 2
    assert s._change_splitlevel(T.Punctuation, ')') == -1
    assert s._parenthesis_level == 1


def test_change_splitlevel_close_paren_floors_at_zero():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Punctuation, ')') == -1
    assert s._parenthesis_level == 0


def test_change_splitlevel_close_paren_requires_punctuation():
    # Kills mutant 'and' -> 'or' at line 98: with non-Punctuation ttype and
    # value ')' the 'or' mutation would wrongly enter the paren branch.
    s = StatementSplitter()
    s._parenthesis_level = 2
    # Non-punctuation token, but value is ')'
    assert s._change_splitlevel(T.Name, ')') == 0
    assert s._parenthesis_level == 2


def test_change_splitlevel_close_paren_requires_value():
    # Punctuation token, but value is not ')' or '(' or ';'
    s = StatementSplitter()
    s._parenthesis_level = 2
    ret = s._change_splitlevel(T.Punctuation, ',')
    # Falls through; not a keyword -> returns 0, doesn't touch paren level
    assert ret == 0
    assert s._parenthesis_level == 2


def test_change_splitlevel_non_keyword():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Name, 'foo') == 0
    assert s._change_splitlevel(T.Literal.Number.Integer, '5') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: DDL CREATE
# ---------------------------------------------------------------------------

def test_change_splitlevel_create_sets_flag():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword.DDL, 'CREATE OR REPLACE') == 0
    assert s._is_create is True


def test_change_splitlevel_create_lowercase_unified():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword.DDL, 'create') == 0
    assert s._is_create is True


def test_change_splitlevel_create_only_for_ddl():
    s = StatementSplitter()
    # Using a non-DDL keyword token type shouldn't set _is_create
    assert s._change_splitlevel(T.Keyword, 'CREATE') == 0
    assert s._is_create is False


# ---------------------------------------------------------------------------
# _change_splitlevel: DECLARE
# ---------------------------------------------------------------------------

def test_change_splitlevel_declare_with_create():
    s = StatementSplitter()
    s._is_create = True
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 1
    assert s._block_stack == ['DECLARE']


def test_change_splitlevel_declare_without_create():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
    assert s._block_stack == []


def test_change_splitlevel_declare_with_nonempty_stack():
    s = StatementSplitter()
    s._is_create = True
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
    assert s._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# _change_splitlevel: BEGIN
# ---------------------------------------------------------------------------

def test_change_splitlevel_begin_pushes():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'BEGIN') == 1
    assert s._block_stack == ['BEGIN']
    assert s._seen_begin is True


def test_change_splitlevel_begin_after_declare_transitions():
    s = StatementSplitter()
    s._block_stack.append('DECLARE')
    ret = s._change_splitlevel(T.Keyword, 'BEGIN')
    assert ret == 0
    assert s._block_stack == ['BEGIN']
    assert s._seen_begin is True


# ---------------------------------------------------------------------------
# _change_splitlevel: transaction keyword after BEGIN (issue826)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('word', [
    'TRANSACTION', 'WORK', 'TRAN', 'DISTRIBUTED',
    'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE',
])
def test_change_splitlevel_transaction_after_begin(word):
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack.append('BEGIN')
    ret = s._change_splitlevel(T.Keyword, word)
    assert ret == -1
    assert s._block_stack == []
    assert s._seen_begin is False


def test_change_splitlevel_transaction_without_begin_on_stack():
    s = StatementSplitter()
    s._seen_begin = True
    ret = s._change_splitlevel(T.Keyword, 'WORK')
    assert ret == 0
    assert s._seen_begin is False


def test_change_splitlevel_transaction_name_token():
    # T.Name is not T.Keyword, so _change_splitlevel returns early at the
    # "if normal token return" guard before reaching the transaction logic.
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack.append('BEGIN')
    ret = s._change_splitlevel(T.Name, 'WORK')
    assert ret == 0
    # State untouched since we returned early
    assert s._block_stack == ['BEGIN']
    assert s._seen_begin is True


def test_transaction_keyword_with_non_name_non_keyword_ttype_not_triggered():
    # Documented equivalence: line 132 mutant `ttype is not T.Name` is
    # equivalent because non-Keyword ttypes early-return at line 101. We pin
    # the observable behavior for T.Name inputs here.
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack.append('BEGIN')
    # T.Name with transaction keyword value must NOT pop (returns early).
    ret = s._change_splitlevel(T.Name, 'TRANSACTION')
    assert ret == 0
    assert s._block_stack == ['BEGIN']
    assert s._seen_begin is True


def test_change_splitlevel_transaction_lowercase():
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack.append('BEGIN')
    ret = s._change_splitlevel(T.Keyword, 'transaction')
    assert ret == -1


# ---------------------------------------------------------------------------
# _handle_nested_block basics (through _change_splitlevel inside BEGIN)
# ---------------------------------------------------------------------------

def test_nested_for_then_loop():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, 'FOR') == 0
    assert s._unconfirmed_start == 'FOR'
    assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
    assert s._block_stack == ['BEGIN', 'FOR']
    assert s._unconfirmed_start is None


def test_nested_while_then_do():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, 'WHILE') == 0
    assert s._unconfirmed_start == 'WHILE'
    assert s._change_splitlevel(T.Keyword, 'DO') == 1
    assert s._block_stack == ['BEGIN', 'WHILE']
    assert s._unconfirmed_start is None


def test_nested_loop_alone():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
    assert s._block_stack == ['BEGIN', 'LOOP']


def test_nested_do_without_unconfirmed_returns_none_passthrough():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    # 'DO' with no unconfirmed start returns None from helper -> closing handles
    ret = s._change_splitlevel(T.Keyword, 'DO')
    assert ret == 0
    assert s._block_stack == ['BEGIN']


def test_nested_if_pushes():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, 'IF') == 1
    assert s._block_stack == ['BEGIN', 'IF']


def test_nested_case_pushes():
    s = StatementSplitter()
    s._block_stack.append('BEGIN')
    assert s._change_splitlevel(T.Keyword, 'CASE') == 1
    assert s._block_stack == ['BEGIN', 'CASE']


def test_nested_block_ignored_outside_begin():
    s = StatementSplitter()
    # No BEGIN on the stack -> IF is not treated as a block start
    assert s._change_splitlevel(T.Keyword, 'IF') == 0
    assert s._block_stack == []


# ---------------------------------------------------------------------------
# Closing keywords (through _change_splitlevel)
# ---------------------------------------------------------------------------

def test_close_end_if():
    s = StatementSplitter()
    s._block_stack = ['BEGIN', 'IF']
    assert s._change_splitlevel(T.Keyword, 'END IF') == -1
    assert s._block_stack == ['BEGIN']


def test_close_end_if_uses_last_element():
    # Kills mutant `[-1]` -> `[+1]` at line 54. We need a stack with length
    # >= 2 where index [+1] == 'IF' but index [-1] != 'IF'. So stack with
    # 'IF' at index 1 (and something else at index 0 and at index -1).
    # Stack = ['X', 'IF', 'Y']: [-1]='Y' (not IF), [+1]='IF'.
    # Correct: no pop. Mutant: pops.
    s = StatementSplitter()
    s._block_stack = ['X', 'IF', 'Y']
    ret = s._change_splitlevel(T.Keyword, 'END IF')
    assert ret == 0
    assert s._block_stack == ['X', 'IF', 'Y']


def test_close_end_if_only_top_matches():
    s = StatementSplitter()
    s._block_stack = ['IF', 'BEGIN']
    ret = s._change_splitlevel(T.Keyword, 'END IF')
    # Top is BEGIN, not IF -> does not pop
    assert ret == 0
    assert s._block_stack == ['IF', 'BEGIN']


def test_close_end_if_empty_stack_no_crash():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'END IF') == 0
    assert s._block_stack == []


def test_close_end_if_wrong_top():
    s = StatementSplitter()
    s._block_stack = ['BEGIN', 'CASE']
    assert s._change_splitlevel(T.Keyword, 'END IF') == 0
    assert s._block_stack == ['BEGIN', 'CASE']


def test_close_end_for():
    s = StatementSplitter()
    s._block_stack = ['BEGIN', 'FOR']
    assert s._change_splitlevel(T.Keyword, 'END FOR') == -1
    assert s._block_stack == ['BEGIN']


def test_close_end_for_uses_last_element():
    # Kills mutant `[-1]` -> `[+1]` at line 58.
    s = StatementSplitter()
    s._block_stack = ['X', 'FOR', 'Y']
    ret = s._change_splitlevel(T.Keyword, 'END FOR')
    assert ret == 0
    assert s._block_stack == ['X', 'FOR', 'Y']


def test_close_end_for_empty_stack_no_crash():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'END FOR') == 0
    assert s._block_stack == []


def test_close_end_while():
    s = StatementSplitter()
    s._block_stack = ['BEGIN', 'WHILE']
    assert s._change_splitlevel(T.Keyword, 'END WHILE') == -1
    assert s._block_stack == ['BEGIN']


def test_close_end_while_uses_last_element():
    # Kills mutant `[-1]` -> `[+1]` at line 62.
    s = StatementSplitter()
    s._block_stack = ['X', 'WHILE', 'Y']
    ret = s._change_splitlevel(T.Keyword, 'END WHILE')
    assert ret == 0
    assert s._block_stack == ['X', 'WHILE', 'Y']


def test_close_end_while_empty_stack_no_crash():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'END WHILE') == 0
    assert s._block_stack == []


@pytest.mark.parametrize('top', ['LOOP', 'FOR', 'WHILE'])
def test_close_end_loop_accepts_loop_for_while(top):
    s = StatementSplitter()
    s._block_stack = ['BEGIN', top]
    assert s._change_splitlevel(T.Keyword, 'END LOOP') == -1
    assert s._block_stack == ['BEGIN']


def test_close_end_loop_uses_last_element():
    # Kills mutant `[-1]` -> `[+1]` at line 67. Need a stack length >= 2
    # where index [+1] is in (LOOP, FOR, WHILE) but index [-1] is not.
    # Stack = ['X', 'LOOP', 'IF']: [-1]='IF' (not in set), [+1]='LOOP' (in set).
    # Correct: no pop. Mutant: pops.
    s = StatementSplitter()
    s._block_stack = ['X', 'LOOP', 'IF']
    ret = s._change_splitlevel(T.Keyword, 'END LOOP')
    assert ret == 0
    assert s._block_stack == ['X', 'LOOP', 'IF']


def test_close_end_loop_checks_top_not_earlier():
    s = StatementSplitter()
    s._block_stack = ['FOR', 'BEGIN']
    ret = s._change_splitlevel(T.Keyword, 'END LOOP')
    assert ret == 0
    assert s._block_stack == ['FOR', 'BEGIN']


def test_close_end_loop_wrong_top():
    s = StatementSplitter()
    s._block_stack = ['BEGIN', 'IF']
    assert s._change_splitlevel(T.Keyword, 'END LOOP') == 0
    assert s._block_stack == ['BEGIN', 'IF']


def test_close_end_case():
    s = StatementSplitter()
    s._block_stack = ['BEGIN', 'CASE']
    assert s._change_splitlevel(T.Keyword, 'END CASE') == -1
    assert s._block_stack == ['BEGIN']


def test_close_end_case_uses_last_element():
    # Kills mutant `[-1]` -> `[+1]` at line 71.
    s = StatementSplitter()
    s._block_stack = ['X', 'CASE', 'Y']
    ret = s._change_splitlevel(T.Keyword, 'END CASE')
    assert ret == 0
    assert s._block_stack == ['X', 'CASE', 'Y']


def test_close_end_case_empty_stack_no_crash():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'END CASE') == 0
    assert s._block_stack == []


def test_close_plain_end_pops_any():
    s = StatementSplitter()
    s._block_stack = ['BEGIN', 'IF']
    assert s._change_splitlevel(T.Keyword, 'END') == -1
    assert s._block_stack == ['BEGIN']


def test_close_plain_end_empty_stack():
    s = StatementSplitter()
    assert s._change_splitlevel(T.Keyword, 'END') == -1
    assert s._block_stack == []


def test_close_unrecognized_keyword():
    s = StatementSplitter()
    s._block_stack = ['BEGIN']
    assert s._change_splitlevel(T.Keyword, 'WHATEVER') == 0
    assert s._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# process: basic splitting
# ---------------------------------------------------------------------------

def test_process_single_statement_with_semicolon():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ';'),
    ]
    statements = process(s, stream)
    assert [st.value for st in statements] == ['SELECT 1;']


def test_process_splits_on_semicolon():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '2'),
        (T.Punctuation, ';'),
    ]
    statements = process(s, stream)
    assert [st.value for st in statements] == ['SELECT 1;', 'SELECT 2;']


def test_process_no_trailing_semicolon():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Literal.Number.Integer, '1'),
    ]
    statements = process(s, stream)
    assert [st.value for st in statements] == ['SELECT 1']


def test_process_empty_stream_yields_nothing():
    s = StatementSplitter()
    assert process(s, []) == []


def test_process_only_whitespace_yields_nothing():
    s = StatementSplitter()
    stream = [
        (T.Whitespace, ' '),
        (T.Newline, '\n'),
    ]
    assert process(s, stream) == []


# ---------------------------------------------------------------------------
# process: consume_ws handling
# ---------------------------------------------------------------------------

def test_process_consumes_whitespace_after_semicolon():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ';'),
        (T.Whitespace, ' '),
        (T.Newline, '\n'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '2'),
    ]
    statements = process(s, stream)
    # whitespace after ';' is a whitespace-type token, so it's appended to
    # the first statement (consume_ws is True but we skip only whitespace).
    assert len(statements) == 2
    assert statements[0].value.startswith('SELECT1;')
    assert 'SELECT' in statements[1].value


# ---------------------------------------------------------------------------
# process: GO splitter
# ---------------------------------------------------------------------------

def test_process_go_splitter():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
        (T.Newline, '\n'),
        (T.Keyword, 'GO'),
        (T.Newline, '\n'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '2'),
    ]
    statements = process(s, stream)
    assert len(statements) >= 1
    # First statement includes the GO
    assert 'GO' in statements[0].value


def test_process_go_with_number():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'GO 2'),
        (T.Whitespace, ' '),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
    ]
    statements = process(s, stream)
    assert len(statements) >= 1
    # Whitespace following GO is appended to the first statement because
    # T.Whitespace is in EOS_TTYPE and doesn't trigger finalization.
    assert statements[0].value == 'GO 2 '


# ---------------------------------------------------------------------------
# process: semicolon inside parentheses shouldn't split
# ---------------------------------------------------------------------------

def test_process_semicolon_inside_parentheses_does_not_split():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Punctuation, '('),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ')'),
        (T.Punctuation, ';'),
    ]
    statements = process(s, stream)
    assert [st.value for st in statements] == ['SELECT(SELECT1);']


def test_process_level_accumulates_multiple_parens():
    # Kills mutant `+=` -> `=` and `-=` at line 168: with two '(' the level
    # must be 2; two nested ')' bring it back to 0.
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Punctuation, '('),
        (T.Punctuation, '('),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ')'),
        (T.Punctuation, ')'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '2'),
    ]
    statements = process(s, stream)
    # With `=` mutation the second '(' would set level back to 1, and inner
    # ';' would split early; with `-=` levels go negative and ';' splits.
    assert [st.value for st in statements] == ['SELECT((1));', 'SELECT2']


# ---------------------------------------------------------------------------
# process: BEGIN...END swallows inner semicolons
# ---------------------------------------------------------------------------

def test_process_begin_end_block_not_split_on_inner_semicolon():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
    ]
    statements = process(s, stream)
    # The BEGIN; is a standalone BEGIN transaction (issue826 handling pops the
    # BEGIN block on the semicolon). Subsequent statements split normally.
    assert [st.value for st in statements] == ['BEGIN;', 'SELECT1;', 'END']


# ---------------------------------------------------------------------------
# process: level returns to 0 via END
# ---------------------------------------------------------------------------

def test_process_multiple_statements_with_block():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '2'),
    ]
    statements = process(s, stream)
    assert len(statements) == 2
    assert 'BEGIN' in statements[0].value
    assert 'END' in statements[0].value
    assert statements[1].value == 'SELECT2'


# ---------------------------------------------------------------------------
# _seen_begin reset on register tokens
# ---------------------------------------------------------------------------

def test_seen_begin_reset_by_regular_token_in_process():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Whitespace, ' '),
        (T.Keyword, 'SELECT'),
    ]
    list(s.process(iter(stream)))
    assert s._seen_begin is False


def test_seen_begin_preserved_by_comment():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Comment.Single, '-- hi'),
    ]
    list(s.process(iter(stream)))
    # Comment.Single is in the reset-skip set, so _seen_begin remains True
    assert s._seen_begin is True


def test_seen_begin_preserved_by_multiline_comment():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Comment.Multiline, '/* hi */'),
    ]
    list(s.process(iter(stream)))
    assert s._seen_begin is True


def test_seen_begin_false_after_semicolon_inside_process():
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack.append('BEGIN')
    s._change_splitlevel(T.Punctuation, ';')
    assert s._seen_begin is False


# ---------------------------------------------------------------------------
# Standalone BEGIN; as transaction
# ---------------------------------------------------------------------------

def test_process_standalone_begin_semicolon_splits():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
    ]
    statements = process(s, stream)
    assert statements[0].value == 'BEGIN;'
    assert statements[1].value == 'SELECT1'


def test_semicolon_handler_resets_seen_begin_to_false():
    s = StatementSplitter()
    s._seen_begin = True
    s._block_stack = ['BEGIN']
    ret = s._change_splitlevel(T.Punctuation, ';')
    assert ret == -1
    assert s._seen_begin is False


# ---------------------------------------------------------------------------
# _is_create drives DECLARE inside process
# ---------------------------------------------------------------------------

def test_process_create_declare_begin_sequence():
    s = StatementSplitter()
    stream = [
        (T.Keyword.DDL, 'CREATE PROCEDURE p'),
        (T.Whitespace, ' '),
        (T.Keyword, 'DECLARE'),
        (T.Whitespace, ' '),
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Keyword, 'END'),
    ]
    statements = process(s, stream)
    # The standalone BEGIN; (after DECLARE->BEGIN transition) pops the block
    # on the semicolon, splitting into a CREATE..; statement and 'END'.
    assert len(statements) == 2
    assert statements[0].value.startswith('CREATE PROCEDURE p')
    assert 'BEGIN' in statements[0].value
    assert statements[1].value == 'END'


# ---------------------------------------------------------------------------
# Non statement tokens after a finished statement start a new one
# ---------------------------------------------------------------------------

def test_process_comment_after_semicolon_stays_in_previous():
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ';'),
        (T.Comment.Single, '-- note'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '2'),
    ]
    statements = process(s, stream)
    assert len(statements) == 2
    # Single-line comments are in EOS_TTYPE, so they attach to first statement
    assert '-- note' in statements[0].value
    assert statements[1].value == 'SELECT2'


def test_semicolon_inside_begin_block_does_not_split():
    s = StatementSplitter()
    stream = [
        (T.Punctuation, '('),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '2'),
    ]
    statements = process(s, stream)
    assert len(statements) == 1
    assert statements[0].value == '(SELECT1;SELECT2'


def test_process_begin_semicolon_clears_seen_begin():
    s = StatementSplitter()
    list(s.process(iter([
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ';'),
        (T.Whitespace, ' '),
    ])))
    assert s._seen_begin is False


def test_process_nested_begin_end_level_tracking():
    s = StatementSplitter()
    stream = [
        (T.Punctuation, '('),
        (T.Punctuation, '('),
        (T.Punctuation, '('),
        (T.Punctuation, ')'),
        (T.Punctuation, ')'),
        (T.Punctuation, ')'),
        (T.Keyword, 'SELECT'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
    ]
    statements = process(s, stream)
    assert [st.value for st in statements] == ['((()))SELECT;', 'SELECT1']


# ---------------------------------------------------------------------------
# Line 183: semicolon at level <= 0 with 'BEGIN' on the block stack
# ---------------------------------------------------------------------------

def test_semicolon_at_zero_level_with_begin_on_stack_does_not_split():
    # Kills mutants at line 183 `'BEGIN' not in self._block_stack`:
    #   - 'BEGIN' -> 'XXBEGINXX'  (string no longer matches stack content)
    #   - 'BEGIN' -> 'begin'      (case-sensitive mismatch against 'BEGIN')
    #
    # Setup: 'BEGIN' returns +1 and pushes 'BEGIN' onto the block stack,
    # raising level to 1. The ')' returns -1, driving level back to 0. The
    # ')' is a non-BEGIN, non-whitespace/comment token which resets
    # _seen_begin to False in process(), so at the ';' the semicolon handler
    # does NOT pop 'BEGIN' from the stack. At that point level == 0 and
    # 'BEGIN' is still on the stack, so the split condition at line 183
    # evaluates to False and no split occurs (all tokens stay bundled).
    # Under the mutants, the split condition evaluates to True and the
    # statement is split after the ';'.
    s = StatementSplitter()
    stream = [
        (T.Keyword, 'BEGIN'),
        (T.Punctuation, ')'),
        (T.Punctuation, ';'),
        (T.Keyword, 'SELECT'),
        (T.Literal.Number.Integer, '1'),
    ]
    statements = process(s, stream)
    assert len(statements) == 1
    assert statements[0].value == 'BEGIN);SELECT1'
