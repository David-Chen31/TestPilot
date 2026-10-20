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
    _extract_from_frame,
    _extract_from_tb,
    _TBItem,
    _DeferredLine,
)


# ---------------------------------------------------------------------------
# ExceptionCauseMixin basics
# ---------------------------------------------------------------------------

def test_cause_is_stored_and_stripped_from_args():
    cause = ValueError('boom')
    exc = ExceptionCauseMixin(cause, 'message')
    assert exc.cause is cause
    # the cause arg is stripped from args by __new__
    assert exc.args == ('message',)


def test_no_cause_sets_cause_none():
    exc = ExceptionCauseMixin('just a message')
    assert exc.cause is None
    assert exc.args == ('just a message',)


def test_first_arg_not_exception_kept_in_args():
    exc = ExceptionCauseMixin('a', 'b')
    assert exc.cause is None
    assert exc.args == ('a', 'b')


def test_root_cause_when_cause_has_none():
    cause = ValueError('root')
    exc = ExceptionCauseMixin(cause)
    assert exc.cause is cause
    assert exc.root_cause is cause


def test_root_cause_inherited_from_chain():
    root = ValueError('root')
    middle = ExceptionCauseMixin(root, 'mid')
    assert middle.root_cause is root
    outer = ExceptionCauseMixin(middle, 'outer')
    assert outer.cause is middle
    assert outer.root_cause is root


def test_message_getter_with_cause():
    cause = ValueError('boom')
    exc = ExceptionCauseMixin(cause, 'my message')
    assert exc._get_message() == 'my message'


def test_message_getter_no_message():
    cause = ValueError('boom')
    exc = ExceptionCauseMixin(cause)
    assert exc._get_message() == ''


def test_message_getter_without_cause():
    exc = ExceptionCauseMixin('hello')
    assert exc._get_message() == 'hello'


def test_class_attr_cause_default_none():
    assert ExceptionCauseMixin.cause is None


# ---------------------------------------------------------------------------
# __str__ behavior
# ---------------------------------------------------------------------------

def test_str_without_cause_delegates_to_exception():
    exc = ExceptionCauseMixin('plain message')
    assert str(exc) == 'plain message'


def test_str_with_cause_no_trace():
    cause = ValueError('boom')
    exc = ExceptionCauseMixin(cause, 'wrapped')
    s = str(exc)
    # no traceback available in this context
    assert 'caused by' in s
    assert 'wrapped' in s
    assert 'ValueError' in s


def test_str_with_cause_no_message():
    cause = ValueError('boom')
    exc = ExceptionCauseMixin(cause)
    s = str(exc)
    assert s.startswith(' caused by ')
    assert 'ValueError: boom' in s


# ---------------------------------------------------------------------------
# Traceback capture on raise
# ---------------------------------------------------------------------------

def test_traceback_capture_within_except():
    cause = None
    wrapper = None
    try:
        raise ValueError('inner')
    except ValueError as ve:
        cause = ve
        wrapper = ExceptionCauseMixin(ve, 'wrapped')
    assert wrapper.cause is cause
    assert wrapper.root_cause is cause
    assert wrapper._tb is not None
    assert len(wrapper._tb) >= 1
    assert wrapper._stack is not None
    assert isinstance(wrapper.full_trace, list)
    assert len(wrapper.full_trace) >= 1


def test_get_str_with_trace():
    try:
        raise ValueError('inner')
    except ValueError as ve:
        exc = ExceptionCauseMixin(ve, 'wrapped')
    s = exc.get_str()
    assert 'Traceback (most recent call last):' in s
    assert 'ValueError' in s
    assert 'wrapped' in s


def test_get_str_without_cause():
    exc = ExceptionCauseMixin('nothing wrong')
    s = exc.get_str()
    # no cause -> falls through to repr-based trace str then exc str
    assert 'nothing wrong' in s or 'ExceptionCauseMixin' in s


def test_inheriting_mixin_with_value_error():
    # MathError inherits mixin first, then ValueError
    assert issubclass(MathError, ValueError)
    assert issubclass(MathError, ExceptionCauseMixin)


# ---------------------------------------------------------------------------
# whoops_math / math_lol / main
# ---------------------------------------------------------------------------

def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


def test_math_lol_raises_math_error():
    with pytest.raises(MathError) as excinfo:
        math_lol()
    exc = excinfo.value
    # cause is the ZeroDivisionError; message "ya done messed up"
    assert isinstance(exc.cause, ZeroDivisionError)
    assert exc.args == ('ya done messed up',)
    assert 'ya done messed up' in str(exc)


def test_math_lol_traceback_captured():
    with pytest.raises(MathError) as excinfo:
        math_lol()
    exc = excinfo.value
    assert exc.full_trace is not None
    assert len(exc.full_trace) > 1


def test_main_raises_math_error_wrapping_math_error():
    with pytest.raises(MathError) as excinfo:
        main()
    exc = excinfo.value
    assert isinstance(exc.cause, MathError)
    assert exc.args == ('hi',)
    # root cause should be the inner MathError
    assert isinstance(exc.root_cause, MathError)
    assert exc.root_cause.root_cause is exc.root_cause.root_cause  # stable


def test_main_root_cause_is_deepest():
    with pytest.raises(MathError) as excinfo:
        main()
    exc = excinfo.value
    # inner MathError's root_cause is the ZeroDivisionError
    assert isinstance(exc.root_cause, MathError)
    assert isinstance(exc.root_cause.root_cause, ZeroDivisionError)


# ---------------------------------------------------------------------------
# _format_exc / _extract_from_frame / _extract_from_tb / _TBItem
# ---------------------------------------------------------------------------

def test_format_exc_default_message():
    exc = ValueError('bad')
    out = _format_exc(exc)
    assert 'ValueError' in out
    assert 'bad' in out
    # trailing whitespace should be stripped
    assert not out.endswith('\n')
    assert not out.endswith(' ')


def test_format_exc_custom_message():
    exc = ValueError('bad')
    out = _format_exc(exc, 'custom')
    assert 'ValueError' in out
    assert 'custom' in out


def test_extract_from_frame_returns_list_of_tb_items():
    def sample():
        return _extract_from_frame(None)
    items = sample()
    assert isinstance(items, list)
    assert len(items) >= 1
    for item in items:
        assert isinstance(item, _TBItem)
        assert hasattr(item, 'frame_id')
        assert isinstance(item.frame_id, int)


def test_extract_from_frame_limit():
    def outer():
        def inner():
            return _extract_from_frame(None, limit=1)
        return inner()
    items = outer()
    assert len(items) == 1


def test_extract_from_tb():
    try:
        raise RuntimeError('err')
    except RuntimeError:
        tb = sys.exc_info()[2]
        items = _extract_from_tb(tb)
    assert len(items) >= 1
    for item in items:
        assert isinstance(item, _TBItem)
        assert isinstance(item.frame_id, int)


def test_extract_from_tb_limit():
    try:
        raise RuntimeError('err')
    except RuntimeError:
        tb = sys.exc_info()[2]
        items = _extract_from_tb(tb, limit=1)
    assert len(items) == 1


def test_tb_item_repr_contains_frame_id():
    try:
        raise RuntimeError('err')
    except RuntimeError:
        tb = sys.exc_info()[2]
        items = _extract_from_tb(tb)
    item = items[0]
    r = repr(item)
    assert '<' in r and '>' in r


# ---------------------------------------------------------------------------
# _DeferredLine behavior
# ---------------------------------------------------------------------------

def test_deferred_line_equality_and_inequality():
    dl1 = _DeferredLine('a.py', 10)
    dl2 = _DeferredLine('a.py', 10)
    dl3 = _DeferredLine('a.py', 11)
    assert dl1 == dl2
    assert dl1 != dl3
    assert not (dl1 != dl2)


def test_deferred_line_str_caches(tmp_path):
    f = tmp_path / 'sample.py'
    f.write_text('line one\nline two\n')
    dl = _DeferredLine(str(f), 2)
    s = str(dl)
    assert s == 'line two'
    # calling again returns cached value
    assert str(dl) == 'line two'
    # strip and len operate on the string
    assert dl.strip() == 'line two'
    assert len(dl) == len('line two')
    # repr wraps str
    assert repr(dl) == repr('line two')


def test_deferred_line_missing_line_returns_none(tmp_path):
    f = tmp_path / 'empty.py'
    f.write_text('')
    dl = _DeferredLine(str(f), 5)
    assert str(dl) is None


def test_deferred_line_filters_module_globals():
    dl = _DeferredLine('x.py', 1, {'__name__': 'm', '__loader__': None, 'other': 1})
    assert dl.module_globals == {'__name__': 'm', '__loader__': None}
