import sys
import traceback

import pytest

from boltons.excutils import (
    ExceptionCauseMixin,
    MathError,
    whoops_math,
    math_lol,
    main,
    _extract_from_tb,
    _extract_from_frame,
    _DeferredLine,
    _TBItem,
    _format_exc,
)


# ---------------------------------------------------------------------------
# Basic construction / cause handling
# ---------------------------------------------------------------------------

def test_mixin_stored_as_cause_first_arg():
    cause = ValueError('boom')
    exc = ExceptionCauseMixin(cause, 'msg')
    assert exc.cause is cause
    assert exc.args == ('msg',)
    assert exc.root_cause is cause


def test_mixin_without_cause():
    exc = ExceptionCauseMixin('just a message')
    assert exc.cause is None
    assert exc.args == ('just a message',)
    # root_cause should be the class default None
    assert getattr(exc, 'root_cause', None) is None


def test_mixin_cause_class_attribute_default_is_none():
    assert ExceptionCauseMixin.cause is None


def test_mixin_chained_root_cause():
    root = ValueError('root')
    mid = ExceptionCauseMixin(root, 'mid')
    assert mid.root_cause is root

    outer = ExceptionCauseMixin(mid, 'outer')
    assert outer.cause is mid
    # root cause should be propagated from the inner wrap
    assert outer.root_cause is root


def test_mixin_non_exception_first_arg_is_normal_arg():
    exc = ExceptionCauseMixin('notexc', 'second')
    assert exc.cause is None
    assert exc.args == ('notexc', 'second')


def test_mixin_no_args():
    exc = ExceptionCauseMixin()
    assert exc.cause is None
    assert exc.args == ()


# ---------------------------------------------------------------------------
# MathError is a subclass of both mixin and ValueError
# ---------------------------------------------------------------------------

def test_math_error_subclassing():
    assert issubclass(MathError, ExceptionCauseMixin)
    assert issubclass(MathError, ValueError)


def test_math_error_construction():
    cause = ZeroDivisionError('div')
    exc = MathError(cause, 'ya done messed up')
    assert isinstance(exc, ValueError)
    assert isinstance(exc, ExceptionCauseMixin)
    assert exc.cause is cause
    assert exc.args == ('ya done messed up',)


# ---------------------------------------------------------------------------
# String formatting methods
# ---------------------------------------------------------------------------

def test_get_message_no_cause():
    exc = ExceptionCauseMixin('hello')
    # cause None -> args kept intact
    assert exc._get_message() == 'hello'


def test_get_message_with_cause():
    cause = ValueError('c')
    exc = ExceptionCauseMixin(cause, 'real message')
    assert exc._get_message() == 'real message'


def test_get_message_empty_when_no_args():
    exc = ExceptionCauseMixin()
    assert exc._get_message() == ''


def test_get_message_falsy_first_arg():
    cause = ValueError('c')
    exc = ExceptionCauseMixin(cause, '', 'ignored')
    # args[1:] -> ('', 'ignored'); first is falsy -> ''
    assert exc._get_message() == ''


def test_get_exc_str_with_message():
    cause = ValueError('c')
    exc = ExceptionCauseMixin(cause, 'my msg')
    s = exc._get_exc_str()
    assert s.startswith('ExceptionCauseMixin: ')
    assert 'my msg (caused by ' in s
    assert 'ValueError: c' in s


def test_get_exc_str_without_message():
    cause = ValueError('c')
    exc = ExceptionCauseMixin(cause)
    s = exc._get_exc_str()
    assert s == 'ExceptionCauseMixin: ' + ' caused by ValueError: c'


def test_get_exc_str_incl_name_false():
    cause = ValueError('c')
    exc = ExceptionCauseMixin(cause, 'msg')
    s = exc._get_exc_str(incl_name=False)
    assert s.startswith('msg (caused by ')
    assert 'ValueError: c' in s


def test_str_without_cause():
    exc = ExceptionCauseMixin('plain message')
    # no cause -> falls through to super().__str__
    assert str(exc) == 'plain message'


def test_str_with_cause_no_trace():
    # Cause exists but full_trace not populated (not raised)
    cause = ValueError('c')
    exc = ExceptionCauseMixin(cause, 'msg')
    # full_trace attribute won't be set because we weren't handling exc
    assert not hasattr(exc, 'full_trace')
    s = str(exc)
    # falls to _get_exc_str(incl_name=False) branch
    assert s.startswith('msg (caused by ')
    assert 'ValueError: c' in s


def test_get_str_without_cause():
    exc = ExceptionCauseMixin('plain')
    s = exc.get_str()
    # No cause -> _get_trace_str returns super().__repr__() which is non-empty
    assert 'ExceptionCauseMixin' in s
    assert s.endswith('plain')


# ---------------------------------------------------------------------------
# Whoops / math_lol / main
# ---------------------------------------------------------------------------

def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


def test_math_lol_raises_math_error():
    with pytest.raises(MathError) as excinfo:
        math_lol()
    exc = excinfo.value
    assert isinstance(exc.cause, ZeroDivisionError)
    assert exc.args == ('ya done messed up',)
    assert 'ya done messed up' in str(exc)


def test_math_lol_recursion_depth():
    # math_lol recurses until n == 3 then raises.
    with pytest.raises(MathError) as excinfo:
        math_lol(0)
    exc = excinfo.value
    assert isinstance(exc.cause, ZeroDivisionError)


def test_math_lol_n_ge_three_still_raises():
    # When n >= 3, it goes directly to whoops_math
    with pytest.raises(MathError) as excinfo:
        math_lol(5)
    assert isinstance(excinfo.value.cause, ZeroDivisionError)


def test_main_raises_math_error():
    with pytest.raises(MathError) as excinfo:
        main()
    exc = excinfo.value
    # top-level wrap catches a MathError (which is a ValueError)
    assert isinstance(exc.cause, MathError)
    assert exc.args == ('hi',)
    # chained root cause
    assert isinstance(exc.root_cause, ZeroDivisionError)


def test_main_raises_caught_as_value_error():
    with pytest.raises(ValueError):
        main()


def test_raised_math_error_populates_full_trace():
    try:
        math_lol()
    except MathError as exc:
        assert hasattr(exc, 'full_trace')
        assert hasattr(exc, '_tb')
        assert hasattr(exc, '_stack')
        # full_trace is a non-empty list of _TBItem entries
        assert len(exc.full_trace) > 0
        assert all(isinstance(item, _TBItem) for item in exc.full_trace)


def test_raised_chain_preserves_full_trace():
    try:
        main()
    except MathError as exc:
        assert hasattr(exc, 'full_trace')
        assert len(exc.full_trace) > 0
        # inner cause is also MathError and should carry its trace
        assert hasattr(exc.cause, 'full_trace')


# ---------------------------------------------------------------------------
# _format_exc
# ---------------------------------------------------------------------------

def test_format_exc_basic():
    exc = ValueError('boo')
    s = _format_exc(exc)
    assert s == 'ValueError: boo'


def test_format_exc_with_message_override():
    exc = ValueError('boo')
    s = _format_exc(exc, message='custom')
    assert s == 'ValueError: custom'


def test_format_exc_no_args():
    exc = ValueError()
    s = _format_exc(exc)
    assert s == 'ValueError'


# ---------------------------------------------------------------------------
# _DeferredLine
# ---------------------------------------------------------------------------

def test_deferred_line_equality(tmp_path):
    f = tmp_path / 'x.py'
    f.write_text('line1\nline2\n')
    a = _DeferredLine(str(f), 1)
    b = _DeferredLine(str(f), 1)
    c = _DeferredLine(str(f), 2)
    assert a == b
    assert a != c
    assert not (a != b)


def test_deferred_line_str_reads_file(tmp_path):
    f = tmp_path / 'sample.py'
    f.write_text('first line\nsecond line\n')
    dl = _DeferredLine(str(f), 2)
    assert str(dl) == 'second line'
    # cached
    assert hasattr(dl, '_line')


def test_deferred_line_missing_file_returns_none():
    dl = _DeferredLine('/nonexistent/path/xyz_zzz.py', 1)
    assert str(dl) is None


def test_deferred_line_repr(tmp_path):
    f = tmp_path / 'r.py'
    f.write_text('hello\n')
    dl = _DeferredLine(str(f), 1)
    assert repr(dl) == "'hello'"


def test_deferred_line_len(tmp_path):
    f = tmp_path / 'l.py'
    f.write_text('abcd\n')
    dl = _DeferredLine(str(f), 1)
    assert len(dl) == 4


def test_deferred_line_strip(tmp_path):
    f = tmp_path / 's.py'
    f.write_text('   spaces   \n')
    dl = _DeferredLine(str(f), 1)
    assert dl.strip() == 'spaces'


def test_deferred_line_module_globals_filtered():
    dl = _DeferredLine('/x', 1, {'__name__': 'm', '__loader__': 1, 'other': 2})
    assert dl.module_globals == {'__name__': 'm', '__loader__': 1}


def test_deferred_line_module_globals_none():
    dl = _DeferredLine('/x', 1)
    assert dl.module_globals == {}


# ---------------------------------------------------------------------------
# _TBItem
# ---------------------------------------------------------------------------

def test_tbitem_repr_includes_frame_id():
    dl = _DeferredLine('/x', 1)
    item = _TBItem('/x', 1, 'name', dl)
    item.frame_id = 12345
    r = repr(item)
    assert '12345' in r
    assert '_TBItem' in r


# ---------------------------------------------------------------------------
# _extract_from_tb
# ---------------------------------------------------------------------------

def test_extract_from_tb_basic():
    try:
        raise ValueError('x')
    except ValueError:
        _, _, tb = sys.exc_info()
        items = _extract_from_tb(tb)
    assert len(items) >= 1
    assert all(isinstance(i, _TBItem) for i in items)
    # Each item has frame_id
    for item in items:
        assert hasattr(item, 'frame_id')
    # The filename includes this test file's path
    assert any(__file__ in i.filename for i in items)


def test_extract_from_tb_limit():
    try:
        raise ValueError('x')
    except ValueError:
        _, _, tb = sys.exc_info()
        items = _extract_from_tb(tb, limit=0)
    assert items == []


def test_extract_from_tb_limit_one():
    try:
        raise ValueError('x')
    except ValueError:
        _, _, tb = sys.exc_info()
        items = _extract_from_tb(tb, limit=1)
    assert len(items) == 1


def test_extract_from_tb_none():
    items = _extract_from_tb(None)
    assert items == []


# ---------------------------------------------------------------------------
# _extract_from_frame
# ---------------------------------------------------------------------------

def test_extract_from_frame_default():
    items = _extract_from_frame()
    assert len(items) >= 1
    assert all(isinstance(i, _TBItem) for i in items)
    # reversed so caller-outermost first
    assert any(__file__ in i.filename for i in items)


def test_extract_from_frame_limit():
    items = _extract_from_frame(limit=1)
    assert len(items) == 1


def test_extract_from_frame_explicit_frame():
    frame = sys._getframe(0)
    items = _extract_from_frame(frame, limit=1)
    assert len(items) == 1
    assert items[0].frame_id == id(frame)
    assert items[0].name == 'test_extract_from_frame_explicit_frame'
