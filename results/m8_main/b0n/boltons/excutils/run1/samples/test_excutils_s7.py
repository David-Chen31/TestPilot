import sys
import traceback

import pytest

from boltons.excutils import (
    ExceptionCauseMixin,
    MathError,
    whoops_math,
    math_lol,
    main,
    _format_exc,
    _extract_from_tb,
    _extract_from_frame,
    _DeferredLine,
    _TBItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleExc(ExceptionCauseMixin, Exception):
    pass


class SimpleValueError(ExceptionCauseMixin, ValueError):
    pass


# ---------------------------------------------------------------------------
# whoops_math
# ---------------------------------------------------------------------------

def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


# ---------------------------------------------------------------------------
# MathError
# ---------------------------------------------------------------------------

def test_math_error_is_exceptioncause_mixin_and_valueerror():
    assert issubclass(MathError, ExceptionCauseMixin)
    assert issubclass(MathError, ValueError)


# ---------------------------------------------------------------------------
# ExceptionCauseMixin - basic construction
# ---------------------------------------------------------------------------

def test_no_cause_sets_cause_none():
    exc = SimpleExc('a', 'b')
    assert exc.cause is None
    assert exc.args == ('a', 'b')
    assert not hasattr(exc, 'root_cause')
    assert not hasattr(exc, 'full_trace')


def test_cause_first_arg_extracted():
    cause = ValueError('original')
    exc = SimpleExc(cause, 'message')
    assert exc.cause is cause
    assert exc.root_cause is cause
    # remaining args become the exception args
    assert exc.args == ('message',)


def test_cause_without_extra_args():
    cause = ValueError('original')
    exc = SimpleExc(cause)
    assert exc.cause is cause
    assert exc.root_cause is cause
    assert exc.args == ()


def test_non_exception_first_arg_not_treated_as_cause():
    exc = SimpleExc('not an exception', 'second')
    assert exc.cause is None
    assert exc.args == ('not an exception', 'second')


def test_empty_args_no_cause():
    exc = SimpleExc()
    assert exc.cause is None
    assert exc.args == ()


# ---------------------------------------------------------------------------
# ExceptionCauseMixin - nested causes and root_cause propagation
# ---------------------------------------------------------------------------

def test_nested_cause_uses_existing_root_cause():
    root = ValueError('root')
    middle = SimpleExc(root, 'middle')
    assert middle.root_cause is root

    outer = SimpleExc(middle, 'outer')
    assert outer.cause is middle
    # root_cause propagated from middle's root_cause
    assert outer.root_cause is root


def test_cause_that_itself_has_no_root_cause_attr():
    # A plain exception with no root_cause attribute
    cause = RuntimeError('plain')
    exc = SimpleExc(cause, 'wrapped')
    assert exc.cause is cause
    assert exc.root_cause is cause


def test_full_trace_copied_when_present():
    cause = ValueError('root')
    cause.full_trace = ['line1', 'line2']
    cause._tb = ['tb1']
    cause._stack = ['stack1']

    exc = SimpleExc(cause, 'msg')
    assert exc.full_trace == ['line1', 'line2']
    assert exc._tb == ['tb1']
    assert exc._stack == ['stack1']
    # Specifically a copy, not the same list
    assert exc.full_trace is not cause.full_trace
    assert exc._tb is not cause._tb
    assert exc._stack is not cause._stack


# ---------------------------------------------------------------------------
# ExceptionCauseMixin - extracting trace when cause is current exception
# ---------------------------------------------------------------------------

def test_cause_is_current_exception_captures_trace():
    try:
        raise ValueError('boom')
    except ValueError as current:
        exc = SimpleExc(current, 'wrapped')
    # The current exception is the cause, so trace info is captured
    assert exc.cause is current
    assert exc.root_cause is current
    assert hasattr(exc, '_tb')
    assert hasattr(exc, '_stack')
    assert hasattr(exc, 'full_trace')
    assert isinstance(exc.full_trace, list)
    assert len(exc._tb) >= 1
    # full_trace = stack[:-1] + tb
    assert exc.full_trace == exc._stack[:-1] + exc._tb


def test_cause_root_cause_is_current_exception_captures_trace():
    root = ValueError('root')
    try:
        raise ValueError('current')
    except ValueError as current:
        # cause is an unrelated exception whose root_cause happens to be current
        cause = SimpleExc(root, 'mid')
        # Force root_cause == current to exercise that branch
        cause.root_cause = current
        exc = SimpleExc(cause, 'wrapped')
    assert exc.cause is cause
    assert hasattr(exc, 'full_trace')


def test_cause_no_current_exception_no_trace():
    cause = ValueError('plain')
    exc = SimpleExc(cause, 'wrapped')
    # No exception being handled -> no trace attributes set
    assert not hasattr(exc, 'full_trace')
    assert not hasattr(exc, '_tb')
    assert not hasattr(exc, '_stack')


# ---------------------------------------------------------------------------
# ExceptionCauseMixin - _get_message
# ---------------------------------------------------------------------------

def test_get_message_without_cause():
    exc = SimpleExc('hello')
    assert exc._get_message() == 'hello'


def test_get_message_with_cause():
    cause = ValueError('cause')
    exc = SimpleExc(cause, 'hello')
    assert exc._get_message() == 'hello'


def test_get_message_empty_args():
    exc = SimpleExc()
    assert exc._get_message() == ''


def test_get_message_with_cause_and_no_extra_args():
    cause = ValueError('cause')
    exc = SimpleExc(cause)
    assert exc._get_message() == ''


# ---------------------------------------------------------------------------
# ExceptionCauseMixin - str formatting
# ---------------------------------------------------------------------------

def test_str_without_cause_simple():
    exc = SimpleExc('hello')
    assert str(exc) == 'hello'


def test_str_without_cause_empty():
    exc = SimpleExc()
    assert str(exc) == ''


def test_str_with_cause_no_trace():
    cause = ValueError('cause')
    exc = SimpleExc(cause, 'hello')
    expected = 'hello (caused by ValueError: cause)'
    # incl_name=False when no trace
    expected = 'hello (caused by ValueError: cause)'
    assert str(exc) == 'hello (caused by ValueError: cause)'


def test_str_with_cause_no_message():
    cause = ValueError('cause')
    exc = SimpleExc(cause)
    assert str(exc) == ' caused by ValueError: cause'


def test_str_with_cause_with_trace():
    try:
        raise ValueError('boom')
    except ValueError as current:
        exc = SimpleExc(current, 'wrapped')
    s = str(exc)
    assert ' --- ' in s
    assert 'Wrapped traceback (most recent call last):' in s
    assert 'SimpleExc' in s
    assert 'ValueError: boom' in s


def test_get_str_without_cause():
    exc = SimpleExc('hello')
    result = exc.get_str()
    # No trace, so just the exception string
    assert result == 'hello'


def test_get_str_with_cause_no_trace():
    cause = ValueError('cause')
    exc = SimpleExc(cause, 'hello')
    result = exc.get_str()
    assert result == 'hello (caused by ValueError: cause)'


def test_get_str_with_trace():
    try:
        raise ValueError('boom')
    except ValueError as current:
        exc = SimpleExc(current, 'wrapped')
    result = exc.get_str()
    assert result.startswith('Traceback (most recent call last):\n')
    assert 'SimpleExc: wrapped (caused by ValueError: boom)' in result


# ---------------------------------------------------------------------------
# _get_exc_str
# ---------------------------------------------------------------------------

def test_get_exc_str_no_cause_with_name():
    exc = SimpleExc('msg')
    # root_cause is None -> _format_exc(None) would fail; but with no cause
    # _get_exc_str isn't normally called. Just check the incl_name path.
    exc.cause = ValueError('root')
    exc.root_cause = ValueError('root')
    assert exc._get_exc_str(incl_name=True) == 'SimpleExc: msg (caused by ValueError: root)'
    assert exc._get_exc_str(incl_name=False) == 'msg (caused by ValueError: root)'


# ---------------------------------------------------------------------------
# _format_exc
# ---------------------------------------------------------------------------

def test_format_exc_uses_exception_as_message():
    exc = ValueError('boom')
    assert _format_exc(exc) == 'ValueError: boom'


def test_format_exc_with_message():
    exc = ValueError('boom')
    assert _format_exc(exc, 'custom') == 'ValueError: custom'


def test_format_exc_empty_message():
    exc = ValueError()
    # Default message is the exception itself; str of empty ValueError is ''
    assert _format_exc(exc) == 'ValueError'


# ---------------------------------------------------------------------------
# MathError flow helpers
# ---------------------------------------------------------------------------

def test_math_lol_raises_math_error_with_cause():
    with pytest.raises(MathError) as excinfo:
        math_lol()
    exc = excinfo.value
    assert isinstance(exc, ValueError)
    assert isinstance(exc.cause, ZeroDivisionError)
    assert exc.root_cause is exc.cause
    assert exc.args == ('ya done messed up',)


def test_main_raises_math_error_wrapping_math_error():
    with pytest.raises(MathError) as excinfo:
        main()
    outer = excinfo.value
    assert isinstance(outer, ValueError)
    assert isinstance(outer.cause, MathError)
    # root_cause should point to the ZeroDivisionError at the bottom
    assert isinstance(outer.root_cause, ZeroDivisionError)


# ---------------------------------------------------------------------------
# _extract_from_tb
# ---------------------------------------------------------------------------

def test_extract_from_tb_returns_items():
    try:
        raise ValueError('boom')
    except ValueError:
        tb = sys.exc_info()[2]
        items = _extract_from_tb(tb)
    assert len(items) >= 1
    for item in items:
        assert isinstance(item, _TBItem)
        assert isinstance(item, _BaseTBItemType())
        assert hasattr(item, 'frame_id')
    # Filename should be this test file
    assert any('test' in item.filename for item in items)


def _BaseTBItemType():
    from boltons.excutils import _BaseTBItem
    return _BaseTBItem


def test_extract_from_tb_limit():
    try:
        try:
            raise ValueError('inner')
        except ValueError:
            raise RuntimeError('outer')
    except RuntimeError:
        tb = sys.exc_info()[2]
        full = _extract_from_tb(tb)
        limited = _extract_from_tb(tb, limit=1)
    assert len(limited) == 1
    assert len(full) >= len(limited)


def test_extract_from_tb_empty():
    assert _extract_from_tb(None) == []


# ---------------------------------------------------------------------------
# _extract_from_frame
# ---------------------------------------------------------------------------

def test_extract_from_frame_default_frame():
    items = _extract_from_frame()
    assert len(items) >= 1
    for item in items:
        assert isinstance(item, _TBItem)
        assert hasattr(item, 'frame_id')


def test_extract_from_frame_with_frame():
    frame = sys._getframe()
    items = _extract_from_frame(frame)
    assert len(items) >= 1
    # Last (most recent after reverse) should be this frame
    assert items[-1].name == 'test_extract_from_frame_with_frame'


def test_extract_from_frame_limit():
    frame = sys._getframe()
    items = _extract_from_frame(frame, limit=1)
    assert len(items) == 1


# ---------------------------------------------------------------------------
# _DeferredLine
# ---------------------------------------------------------------------------

def test_deferred_line_equality():
    a = _DeferredLine('file.py', 10)
    b = _DeferredLine('file.py', 10)
    c = _DeferredLine('file.py', 11)
    assert a == b
    assert a != c
    assert not (a != b)
    assert a != object()


def test_deferred_line_str_nonexistent_file():
    # getline returns '' -> line becomes None -> returned None
    dl = _DeferredLine('does_not_exist_xyz.py', 1)
    result = str(dl)
    assert result is None


def test_deferred_line_str_and_repr(tmp_path):
    p = tmp_path / 'sample.py'
    p.write_text('x = 1\n')
    dl = _DeferredLine(str(p), 1)
    assert str(dl) == 'x = 1'
    assert repr(dl) == "'x = 1'"
    assert len(dl) == len('x = 1')
    assert dl.strip() == 'x = 1'


def test_deferred_line_str_cached(tmp_path):
    p = tmp_path / 'sample.py'
    p.write_text('first\nsecond\n')
    dl = _DeferredLine(str(p), 1)
    assert str(dl) == 'first'
    # Overwrite file, value should remain cached
    p.write_text('changed\n')
    assert str(dl) == 'first'


def test_deferred_line_module_globals_filtered():
    dl = _DeferredLine('f.py', 1, {'__name__': 'm', 'other': 1, '__loader__': None})
    assert dl.module_globals == {'__name__': 'm', '__loader__': None}


# ---------------------------------------------------------------------------
# _TBItem repr
# ---------------------------------------------------------------------------

def test_tbitem_repr_includes_frame_id():
    dl = _DeferredLine('f.py', 1)
    item = _TBItem('f.py', 1, 'name', dl)
    item.frame_id = 12345
    r = repr(item)
    assert '<12345>' in r
    assert 'f.py' in r


# ---------------------------------------------------------------------------
# Exception identity / args handling
# ---------------------------------------------------------------------------

def test_multiple_causes_chain():
    a = ValueError('a')
    b = SimpleExc(a, 'b')
    c = SimpleExc(b, 'c')
    d = SimpleExc(c, 'd')
    assert d.cause is c
    assert c.cause is b
    assert b.cause is a
    assert d.root_cause is a


def test_cause_arg_shifting_preserves_other_args():
    cause = ValueError('cause')
    exc = SimpleExc(cause, 'msg1', 'msg2')
    assert exc.args == ('msg1', 'msg2')
    assert exc._get_message() == 'msg1'
