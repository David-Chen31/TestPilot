#
# Tests for sqlparse.engine.statement_splitter
#
import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter


def _stream_of(*tokens):
    """Helper to build a stream from (ttype, value) tuples."""
    return list(tokens)


def _kw(value):
    return (T.Keyword, value)


def _kwddl(value):
    return (T.Keyword.DDL, value)


def _punc(value):
    return (T.Punctuation, value)


def _ws(value=' '):
    return (T.Whitespace, value)


def _nl(value='\n'):
    return (T.Newline, value)


def _name(value):
    return (T.Name, value)


def _comment_single(value='-- x'):
    return (T.Comment.Single, value)


def _comment_multi(value='/* x */'):
    return (T.Comment.Multiline, value)


# ---------------------------------------------------------------------------
# Constructor / reset / initial attributes
# ---------------------------------------------------------------------------

class TestInitAndReset:
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

    def test_reset_after_mutation(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        s._parenthesis_level = 3
        s._unconfirmed_start = 'FOR'
        s._is_create = True
        s._seen_begin = True
        s.consume_ws = True
        s.tokens.append(sql.Token(T.Keyword, 'X'))
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

class TestChangeSplitlevelPunctuation:
    def test_semicolon_returns_zero(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Punctuation, ';') == 0

    def test_semicolon_resets_unconfirmed_start(self):
        s = StatementSplitter()
        s._unconfirmed_start = 'FOR'
        s._change_splitlevel(T.Punctuation, ';')
        assert s._unconfirmed_start is None

    def test_semicolon_pops_begin_when_seen_begin(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        s._seen_begin = True
        result = s._change_splitlevel(T.Punctuation, ';')
        assert result == -1
        assert s._block_stack == []
        assert s._seen_begin is False

    def test_semicolon_seen_begin_wrong_top_returns_zero(self):
        s = StatementSplitter()
        s._block_stack.append('IF')
        s._seen_begin = True
        result = s._change_splitlevel(T.Punctuation, ';')
        assert result == 0
        assert s._seen_begin is False
        # stack top unchanged
        assert s._block_stack == ['IF']

    def test_semicolon_seen_begin_empty_stack_one(self):
        s = StatementSplitter()
        s._seen_begin = True
        result = s._change_splitlevel(T.Punctuation, ';')
        assert result == 0
        assert s._seen_begin is False

    def test_open_paren(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Punctuation, '(') == 1
        assert s._parenthesis_level == 1
        assert s._change_splitlevel(T.Punctuation, '(') == 1
        assert s._parenthesis_level == 2

    def test_close_paren(self):
        s = StatementSplitter()
        s._parenthesis_level = 2
        assert s._change_splitlevel(T.Punctuation, ')') == -1
        assert s._parenthesis_level == 1
        assert s._change_splitlevel(T.Punctuation, ')') == -1
        assert s._parenthesis_level == 0

    def test_close_paren_does_not_go_negative(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Punctuation, ')') == -1
        assert s._parenthesis_level == 0

    def test_non_keyword_non_punctuation_returns_zero(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Name, 'foo') == 0
        assert s._change_splitlevel(T.Number, '42') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: CREATE / DECLARE / BEGIN
# ---------------------------------------------------------------------------

class TestChangeSplitlevelCreateDeclareBegin:
    def test_create_sets_flag(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword.DDL, 'CREATE') == 0
        assert s._is_create is True

    def test_create_or_replace_sets_flag(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword.DDL, 'CREATE OR REPLACE') == 0
        assert s._is_create is True

    def test_create_lowercase(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword.DDL, 'create') == 0
        assert s._is_create is True

    def test_non_ddl_create_keyword_does_not_set_flag(self):
        s = StatementSplitter()
        result = s._change_splitlevel(T.Keyword, 'CREATE')
        # Not DDL -> not handled here; falls through closing keywords (0)
        assert s._is_create is False
        assert result == 0

    def test_declare_inside_create_no_stack(self):
        s = StatementSplitter()
        s._is_create = True
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 1
        assert s._block_stack == ['DECLARE']

    def test_declare_not_in_create(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert s._block_stack == []

    def test_declare_with_nonempty_stack(self):
        s = StatementSplitter()
        s._is_create = True
        s._block_stack.append('BEGIN')
        assert s._change_splitlevel(T.Keyword, 'DECLARE') == 0
        assert s._block_stack == ['BEGIN']

    def test_begin_pushes_and_returns_one(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'BEGIN') == 1
        assert s._block_stack == ['BEGIN']
        assert s._seen_begin is True

    def test_begin_transitions_from_declare(self):
        s = StatementSplitter()
        s._is_create = True
        s._change_splitlevel(T.Keyword, 'DECLARE')
        assert s._block_stack == ['DECLARE']
        assert s._change_splitlevel(T.Keyword, 'BEGIN') == 0
        assert s._block_stack == ['BEGIN']
        assert s._seen_begin is True

    def test_begin_lowercase(self):
        s = StatementSplitter()
        assert s._change_splitlevel(T.Keyword, 'begin') == 1
        assert s._block_stack == ['BEGIN']


# ---------------------------------------------------------------------------
# _change_splitlevel: transaction keywords after BEGIN (Issue826)
# ---------------------------------------------------------------------------

class TestTransactionAfterBegin:
    @pytest.mark.parametrize('word', [
        'TRANSACTION', 'WORK', 'TRAN', 'DISTRIBUTED',
        'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE',
    ])
    def test_transaction_keyword_pops_begin(self, word):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword, 'BEGIN')
        assert s._block_stack == ['BEGIN']
        result = s._change_splitlevel(T.Keyword, word)
        assert result == -1
        assert s._block_stack == []
        assert s._seen_begin is False

    def test_transaction_as_name_pops_begin(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword, 'BEGIN')
        result = s._change_splitlevel(T.Name, 'TRANSACTION')
        assert result == -1
        assert s._block_stack == []
        assert s._seen_begin is False

    def test_transaction_only_after_seen_begin(self):
        s = StatementSplitter()
        # No BEGIN seen -> keyword falls through to closing keyword handling.
        result = s._change_splitlevel(T.Keyword, 'TRANSACTION')
        assert result == 0
        assert s._block_stack == []

    def test_transaction_no_begin_on_stack(self):
        s = StatementSplitter()
        s._seen_begin = True
        s._block_stack.append('IF')
        result = s._change_splitlevel(T.Keyword, 'WORK')
        assert result == 0
        assert s._seen_begin is False
        assert s._block_stack == ['IF']


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

    def test_loop_after_for(self):
        s = StatementSplitter()
        s._unconfirmed_start = 'FOR'
        assert s._handle_nested_block('LOOP') == 1
        assert s._block_stack == ['FOR']
        assert s._unconfirmed_start is None

    def test_loop_after_while(self):
        s = StatementSplitter()
        s._unconfirmed_start = 'WHILE'
        assert s._handle_nested_block('LOOP') == 1
        assert s._block_stack == ['WHILE']
        assert s._unconfirmed_start is None

    def test_do_after_for(self):
        s = StatementSplitter()
        s._unconfirmed_start = 'FOR'
        assert s._handle_nested_block('DO') == 1
        assert s._block_stack == ['FOR']
        assert s._unconfirmed_start is None

    def test_loop_without_unconfirmed(self):
        s = StatementSplitter()
        assert s._handle_nested_block('LOOP') == 1
        assert s._block_stack == ['LOOP']
        assert s._unconfirmed_start is None

    def test_do_without_unconfirmed_returns_none(self):
        s = StatementSplitter()
        assert s._handle_nested_block('DO') is None
        assert s._block_stack == []

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
        assert s._handle_nested_block('WHATEVER') is None
        assert s._block_stack == []


# ---------------------------------------------------------------------------
# _handle_closing_keyword
# ---------------------------------------------------------------------------

class TestHandleClosingKeyword:
    def test_end_if_pops_if(self):
        s = StatementSplitter()
        s._block_stack.append('IF')
        assert s._handle_closing_keyword('END IF') == -1
        assert s._block_stack == []

    def test_end_if_wrong_top_returns_zero(self):
        s = StatementSplitter()
        s._block_stack.append('CASE')
        assert s._handle_closing_keyword('END IF') == 0
        assert s._block_stack == ['CASE']

    def test_end_for_pops_for(self):
        s = StatementSplitter()
        s._block_stack.append('FOR')
        assert s._handle_closing_keyword('END FOR') == -1
        assert s._block_stack == []

    def test_end_for_wrong_top(self):
        s = StatementSplitter()
        s._block_stack.append('IF')
        assert s._handle_closing_keyword('END FOR') == 0
        assert s._block_stack == ['IF']

    def test_end_while_pops_while(self):
        s = StatementSplitter()
        s._block_stack.append('WHILE')
        assert s._handle_closing_keyword('END WHILE') == -1
        assert s._block_stack == []

    def test_end_while_wrong_top(self):
        s = StatementSplitter()
        s._block_stack.append('LOOP')
        assert s._handle_closing_keyword('END WHILE') == 0
        assert s._block_stack == ['LOOP']

    @pytest.mark.parametrize('top', ['LOOP', 'FOR', 'WHILE'])
    def test_end_loop_pops_any_loop_ish(self, top):
        s = StatementSplitter()
        s._block_stack.append(top)
        assert s._handle_closing_keyword('END LOOP') == -1
        assert s._block_stack == []

    def test_end_loop_wrong_top(self):
        s = StatementSplitter()
        s._block_stack.append('IF')
        assert s._handle_closing_keyword('END LOOP') == 0
        assert s._block_stack == ['IF']

    def test_end_case_pops_case(self):
        s = StatementSplitter()
        s._block_stack.append('CASE')
        assert s._handle_closing_keyword('END CASE') == -1
        assert s._block_stack == []

    def test_end_case_wrong_top(self):
        s = StatementSplitter()
        s._block_stack.append('IF')
        assert s._handle_closing_keyword('END CASE') == 0
        assert s._block_stack == ['IF']

    def test_plain_end_pops_anything(self):
        s = StatementSplitter()
        s._block_stack.append('BEGIN')
        assert s._handle_closing_keyword('END') == -1
        assert s._block_stack == []

    def test_plain_end_empty_stack(self):
        s = StatementSplitter()
        assert s._handle_closing_keyword('END') == -1
        assert s._block_stack == []

    def test_unknown_returns_zero(self):
        s = StatementSplitter()
        assert s._handle_closing_keyword('FOO') == 0


# ---------------------------------------------------------------------------
# _change_splitlevel: nested blocks via BEGIN
# ---------------------------------------------------------------------------

class TestChangeSplitlevelNested:
    def test_if_inside_begin(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword, 'BEGIN')
        assert s._change_splitlevel(T.Keyword, 'IF') == 1
        assert s._block_stack == ['BEGIN', 'IF']

    def test_for_loop_inside_begin(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword, 'BEGIN')
        assert s._change_splitlevel(T.Keyword, 'FOR') == 0
        assert s._unconfirmed_start == 'FOR'
        assert s._change_splitlevel(T.Keyword, 'LOOP') == 1
        assert s._block_stack == ['BEGIN', 'FOR']
        assert s._unconfirmed_start is None

    def test_end_if_inside_begin(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword, 'BEGIN')
        s._change_splitlevel(T.Keyword, 'IF')
        assert s._change_splitlevel(T.Keyword, 'END IF') == -1
        assert s._block_stack == ['BEGIN']

    def test_end_pops_begin_outside_nested(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword, 'BEGIN')
        assert s._change_splitlevel(T.Keyword, 'END') == -1
        assert s._block_stack == []

    def test_unified_uppercased(self):
        s = StatementSplitter()
        s._change_splitlevel(T.Keyword, 'begin')
        assert s._change_splitlevel(T.Keyword, 'if') == 1
        assert s._block_stack == ['BEGIN', 'IF']


# ---------------------------------------------------------------------------
# process(): splitting logic
# ---------------------------------------------------------------------------

class TestProcess:
    def test_single_statement_without_semicolon(self):
        s = StatementSplitter()
        stmts = list(s.process(_stream_of(
            _kw('SELECT'), _ws(), _name('a'),
        )))
        assert len(stmts) == 1
        assert isinstance(stmts[0], sql.Statement)

    def test_empty_stream_yields_nothing(self):
        s = StatementSplitter()
        assert list(s.process([])) == []

    def test_only_whitespace_yields_nothing(self):
        s = StatementSplitter()
        assert list(s.process(_stream_of(_ws(), _nl()))) == []

    def test_split_on_semicolon_followed_by_token(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('SELECT'), _ws(), _name('a'), _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('b'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2
        # First statement ends with semicolon
        assert stmts[0]._tokens[-1].value == ';'

    def test_semicolon_at_end_yields_single(self):
        s = StatementSplitter()
        stream = _stream_of(_kw('SELECT'), _ws(), _name('a'), _punc(';'))
        stmts = list(s.process(stream))
        assert len(stmts) == 1
        assert stmts[0]._tokens[-1].value == ';'

    def test_multiple_statements_each_semicolon(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('SELECT'), _ws(), _name('a'), _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('b'), _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('c'), _punc(';'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 3

    def test_go_splits(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('SELECT'), _ws(), _name('a'),
            _nl(), _kw('GO'), _nl(),
            _kw('SELECT'), _ws(), _name('b'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2

    def test_go_with_count_splits(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('SELECT'), _ws(), _name('a'),
            _nl(), _kw('GO 2'), _nl(),
            _kw('SELECT'), _ws(), _name('b'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2

    def test_semicolon_inside_parens_no_split(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('SELECT'), _ws(),
            _punc('('), _name('a'), _punc(';'), _name('b'), _punc(')'),
            _ws(), _kw('SELECT'), _ws(), _name('c'),
        )
        stmts = list(s.process(stream))
        # Only split at very end -> one statement (no terminal split token)
        assert len(stmts) == 1

    def test_paren_level_tracks_balance(self):
        s = StatementSplitter()
        list(s.process(_stream_of(
            _punc('('), _name('x'), _punc(')'),
        )))
        # After full processing the splitter isn't reset (no trailing token to
        # trigger reset), so state persists.
        assert s._parenthesis_level == 0

    def test_comment_single_does_not_trigger_split(self):
        s = StatementSplitter()
        # Statement ended with ';', then single-line comment, then next token.
        stream = _stream_of(
            _kw('SELECT'), _ws(), _name('a'), _punc(';'),
            _ws(), _comment_single(), _nl(),
            _kw('SELECT'), _ws(), _name('b'),
        )
        stmts = list(s.process(stream))
        # Comments after ';' are consumed as part of EOS_TTYPE; but ws is also
        # consumed. The first token after ws+comment triggers the split.
        # The split happens when a non-EOS token appears -> 2 statements.
        assert len(stmts) == 2

    def test_semicolon_inside_begin_block_no_split(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('CREATE'),
            _ws(), _kw('BEGIN'),
            _ws(), _kw('SELECT'), _punc(';'),
            _ws(), _kw('END'),
            _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('x'),
        )
        # Note: BEGIN here is a plain Keyword, and 'CREATE' would be DDL; but
        # we use plain Keyword for CREATE so _is_create stays False.
        # Still, BEGIN sets _seen_begin and pushes 'BEGIN'.
        stmts = list(s.process(stream))
        # There should be a split only after 'END ;' (level back to 0,
        # 'BEGIN' removed from stack), so 2 statements.
        assert len(stmts) == 2

    def test_begin_transaction_splits_on_semicolon(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('BEGIN'), _ws(), _kw('TRANSACTION'), _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('a'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2

    def test_begin_standalone_semicolon(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('BEGIN'), _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('a'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2

    def test_empty_begin_block_then_statement(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('BEGIN'), _ws(), _kw('END'), _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('a'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2

    def test_create_declare_begin_block(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kwddl('CREATE'), _ws(), _kw('DECLARE'),
            _ws(), _kw('BEGIN'),
            _ws(), _kw('END'), _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('a'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2

    def test_nested_if_block_splits_after_end(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('BEGIN'),
            _ws(), _kw('IF'),
            _ws(), _name('cond'),
            _ws(), _kw('END IF'), _punc(';'),
            _ws(), _kw('END'), _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('x'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2

    def test_seen_begin_reset_by_non_ws_token(self):
        s = StatementSplitter()
        # BEGIN followed by a Name -> _seen_begin should be reset to False.
        list(s.process(_stream_of(_kw('BEGIN'), _ws(), _name('foo'))))
        # After process, no semicolon, so no reset; but the name token reset
        # _seen_begin during processing.
        assert s._seen_begin is False

    def test_seen_begin_preserved_across_begin_keyword(self):
        s = StatementSplitter()
        list(s.process(_stream_of(_kw('BEGIN'), _ws())))
        # Whitespace does not reset _seen_begin and no further token.
        assert s._seen_begin is True

    def test_process_resets_between_statements(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kwddl('CREATE'), _ws(), _name('t'), _punc(';'),
            _ws(), _kw('SELECT'), _ws(), _name('a'), _punc(';'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2
        # After _reset triggered by the SELECT token, _is_create should be
        # False until 'CREATE' seen again; final state after full processing:
        assert s._is_create is False

    def test_returns_statement_objects_with_tokens(self):
        s = StatementSplitter()
        stream = _stream_of(_kw('SELECT'), _ws(), _name('a'), _punc(';'),
                            _ws(), _kw('SELECT'))
        stmts = list(s.process(stream))
        assert len(stmts) == 2
        assert all(isinstance(st, sql.Statement) for st in stmts)
        # First statement's tokens should include ';'
        assert any(t.value == ';' for t in stmts[0]._tokens)

    def test_whitespace_newline_consumed_after_split(self):
        s = StatementSplitter()
        stream = _stream_of(
            _kw('SELECT'), _ws(), _name('a'), _punc(';'),
            _nl(), _nl(), _ws(),
            _kw('SELECT'), _ws(), _name('b'),
        )
        stmts = list(s.process(stream))
        assert len(stmts) == 2
        # Second statement should NOT include the leading whitespace
        # (it was appended to first statement / consumed).
        second_first = stmts[1]._tokens[0]
        assert second_first.value == 'SELECT'

    def test_instance_reuse_after_process(self):
        s = StatementSplitter()
        list(s.process(_stream_of(_kw('SELECT'), _punc(';'))))
        # ensure object can be reused
        stmts = list(s.process(_stream_of(
            _kw('SELECT'), _punc(';'), _ws(), _kw('SELECT'),
        )))
        assert len(stmts) == 2
