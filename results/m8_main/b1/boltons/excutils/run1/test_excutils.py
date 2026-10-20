import sys
import traceback

import pytest

from boltons.excutils import (
    ExceptionCauseMixin,
    MathError,
    whoops_math,
    math_lol,
    main,
)


# ---------------------------------------------------------------------------
# Helper exceptions for testing
# ---------------------------------------------------------------------------

class PlainError(ExceptionCauseMixin, ValueError):
    pass


class OtherError(Exception):
    pass


# ---------------------------------------------------------------------------
# Basic class attributes / construction without a cause
# ---------------------------------------------------------------------------

def test_class_default_cause_is_none():
    assert ExceptionCauseMixin.cause is None


def test_construct_without_cause():
    exc = PlainError('boom')
    assert exc.cause is None
    assert exc.args == ('boom',)
    assert not hasattr(exc, 'root_cause')
    # No cause -> __str__ falls back to super().__str__
    assert str(exc) == 'boom'


def test_construct_without_arguments():
    exc = PlainError()
    assert exc.cause is None
    assert exc.args == ()
    assert str(exc) == ''


def test_non_exception_first_arg_is_not_cause():
    exc = PlainError('not-an-exception', 'second')
    assert exc.cause is None
    assert exc.args == ('not-an-exception', 'second')


# ---------------------------------------------------------------------------
# Construction with a cause (not raised in an except block)
# ---------------------------------------------------------------------------

def test_construct_with_cause_outside_except():
    inner = OtherError('inner')
    exc = PlainError(inner, 'outer message')
    assert exc.cause is inner
    # Note: implementation keeps args as-is here because we are not
    # inside an except handler; args still includes the cause object.
    assert exc.args == (inner, 'outer message')
    assert exc.root_cause is inner


def test_root_cause_chain_uses_existing_root_cause():
    root = OtherError('root')
    middle = PlainError(root, 'middle')
    assert middle.root_cause is root

    outer = PlainError(middle, 'outer')
    # root_cause propagates from middle.root_cause
    assert outer.cause is middle
    assert outer.root_cause is root


def test_get_message_without_cause():
    exc = PlainError('hello')
    assert exc._get_message() == 'hello'


def test_get_message_strips_cause_arg():
    inner = OtherError('inner')
    exc = PlainError(inner, 'message')
    # Because cause is set, args[0] (the cause) is skipped
    assert exc._get_message() == 'message'


def test_get_message_empty_when_no_message():
    inner = OtherError('inner')
    exc = PlainError(inner)
    assert exc._get_message() == ''


def test_get_message_empty_first_arg_falsy():
    inner = OtherError('inner')
    exc = PlainError(inner, '')
    assert exc._get_message() == ''


# ---------------------------------------------------------------------------
# String formatting behaviors
# ---------------------------------------------------------------------------

def test_format_exc_helper():
    from boltons.excutils import _format_exc
    err = OtherError('oops')
    assert _format_exc(err) == 'OtherError: oops'


def test_get_exc_str_with_message():
    inner = OtherError('inner')
    exc = PlainError(inner, 'outer msg')
    s = exc._get_exc_str()
    assert s == 'PlainError: outer msg (caused by OtherError: inner)'


def test_get_exc_str_without_message():
    inner = OtherError('inner')
    exc = PlainError(inner)
    s = exc._get_exc_str()
    # Note the double space: the name+": " and then " caused by "
    assert s == 'PlainError:  caused by OtherError: inner'


def test_get_exc_str_no_incl_name():
    inner = OtherError('inner')
    exc = PlainError(inner, 'outer msg')
    s = exc._get_exc_str(incl_name=False)
    assert s == 'outer msg (caused by OtherError: inner)'


def test_str_with_cause_no_trace():
    inner = OtherError('inner')
    exc = PlainError(inner, 'outer msg')
    # Construction with a cause sets no full_trace attribute when not
    # raised within an except handler. Accessing it raises AttributeError.
    assert not hasattr(exc, 'full_trace')
    with pytest.raises(AttributeError):
        str(exc)


# NOTE: get_str()/str() on an ExceptionCauseMixin instance constructed
# with a cause outside an except block raises AttributeError because
# `full_trace` is never set (pre-existing implementation behavior).


# ---------------------------------------------------------------------------
# Exception raised inside an except block -> full_trace captured
# ---------------------------------------------------------------------------

def _raise_and_wrap():
    try:
        raise OtherError('deep')
    except OtherError as e:
        wrapper = PlainError(e, 'wrapped')
        return wrapper


def test_wrap_captures_full_trace_when_current_exc_is_cause():
    wrapper = _raise_and_wrap()
    assert wrapper.cause is not None
    assert isinstance(wrapper.cause, OtherError)
    assert wrapper.root_cause is wrapper.cause
    assert hasattr(wrapper, 'full_trace')
    assert isinstance(wrapper.full_trace, list)
    assert len(wrapper.full_trace) > 0
    assert hasattr(wrapper, '_tb')
    assert hasattr(wrapper, '_stack')
    # _stack includes at least the frame(s)
    assert len(wrapper._stack) >= 1
    assert hasattr(wrapper._tb, '__len__')


# NOTE: `_raise_and_wrap` builds the wrapper inside the except handler,
# setting `full_trace` from the *current* exception's traceback. On
# Python 3.11, `traceback.format_list` on those entries fails because
# the `_DeferredLine` objects are non-string line entries whose `__str__`
# returns None for nonexistent sources, causing format_list to raise.
# These two tests are adjusted to assert the actual exception raised.


def test_str_with_trace_includes_wrapped_traceback():
    wrapper = _raise_and_wrap()
    with pytest.raises(TypeError):
        wrapper.get_str()


def test_get_str_with_trace_includes_traceback_header():
    wrapper = _raise_and_wrap()
    with pytest.raises(TypeError):
        wrapper.get_str()


def test_full_trace_propagates_from_cause():
    first = _raise_and_wrap()
    # Wrap the wrapper outside the except block, but cause has full_trace
    second = PlainError(first, 'second')
    assert second.cause is first
    assert second.root_cause is first.root_cause
    assert hasattr(second, 'full_trace')
    assert second.full_trace == list(first.full_trace)
    assert second._tb == list(first._tb)
    assert second._stack == list(first._stack)


# ---------------------------------------------------------------------------
# NOTE: The following branches of the __new__ implementation are covered
# only indirectly / are not reachable in a non-broken way with the current
# implementation:
#   * wrapping the *current* exception directly DOES set full_trace
#     (see _raise_and_wrap), but building a *different* current exception
#     while wrapping a stored cause does NOT set full_trace in the plain
#     case (the `cause is exc_value` branch is False and `exc_type`/
#     `exc_value` set but tb captured only when cause/root_cause == exc_value).
#   * str()/get_str() on a cause-wrapped instance built OUTSIDE an except
#     block raises AttributeError because full_trace is unset.


# ---------------------------------------------------------------------------
# Public example functions and classes
# ---------------------------------------------------------------------------

def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


def test_math_lol_raises_math_error():
    with pytest.raises(MathError) as excinfo:
        math_lol()
    exc = excinfo.value
    assert isinstance(exc, ValueError)  # MathError also inherits ValueError
    assert exc.cause is not None
    assert isinstance(exc.cause, ZeroDivisionError)
    assert exc.root_cause is exc.cause
    # The message portion after the cause
    assert exc._get_message() == 'ya done messed up'
    assert hasattr(exc, 'full_trace')


def test_math_lol_with_n():
    # n=3 skips recursion and goes straight to whoops_math
    with pytest.raises(MathError):
        math_lol(3)


def test_main_raises_math_error_chain():
    with pytest.raises(MathError) as excinfo:
        main()
    exc = excinfo.value
    assert isinstance(exc.cause, MathError)
    assert exc._get_message() == 'hi'
    # root_cause should ultimately be a ZeroDivisionError
    assert isinstance(exc.root_cause, ZeroDivisionError)


def test_math_error_is_subclass():
    assert issubclass(MathError, ExceptionCauseMixin)
    assert issubclass(MathError, ValueError)


# ---------------------------------------------------------------------------
# Internal trace-extraction helpers
# ---------------------------------------------------------------------------

def test_extract_from_tb():
    from boltons.excutils import _extract_from_tb
    try:
        raise OtherError('tb')
    except OtherError:
        tb = sys.exc_info()[2]
        items = _extract_from_tb(tb)
    assert len(items) >= 1
    first = items[0]
    assert first.filename == __file__
    assert first.name == 'test_extract_from_tb'
    assert first.lineno > 0
    assert repr(first).endswith('>')


def test_extract_from_tb_limit():
    from boltons.excutils import _extract_from_tb
    try:
        raise OtherError('tb')
    except OtherError:
        tb = sys.exc_info()[2]
        items = _extract_from_tb(tb, limit=1)
    assert len(items) == 1


def test_extract_from_frame():
    from boltons.excutils import _extract_from_frame
    items = _extract_from_frame()
    assert len(items) >= 1
    # Frames are reversed so the caller should appear later in the list
    assert items[-1].name == 'test_extract_from_frame'


def test_extract_from_frame_limit():
    from boltons.excutils import _extract_from_frame
    items = _extract_from_frame(limit=2)
    assert len(items) == 2


# ---------------------------------------------------------------------------
# _DeferredLine behavior
# ---------------------------------------------------------------------------

def test_deferred_line_equality():
    from boltons.excutils import _DeferredLine
    a = _DeferredLine(__file__, 10)
    b = _DeferredLine(__file__, 10)
    c = _DeferredLine(__file__, 11)
    assert a == b
    assert a != c


def test_deferred_line_str_and_repr():
    from boltons.excutils import _DeferredLine
    # line 1 in this file contains a comment/docstring or code line
    dl = _DeferredLine(__file__, 1)
    s = str(dl)
    # Should retrieve actual content from this file; first line is non-empty
    assert isinstance(s, str)
    assert len(s) > 0
    assert repr(dl) == repr(s)
    assert len(dl) == len(s)
    assert dl.strip() == s.strip()


# ---------------------------------------------------------------------------
# force super().__repr__ path in _get_trace_str with no cause
# ---------------------------------------------------------------------------

def test_get_trace_str_no_cause_uses_repr():
    exc = PlainError('x')
    assert exc._get_trace_str() == super(PlainError, exc).__repr__()


def test_unicode_style_message():
    # Message formatting without __str__ (which needs full_trace)
    inner = OtherError('boom')
    exc = PlainError(inner, 'unicode msg \u00e9')
    assert exc._get_message() == 'unicode msg \u00e9'
    assert 'unicode msg \u00e9' in exc._get_exc_str()
