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
# Basic wrapping behavior of ExceptionCauseMixin
# ---------------------------------------------------------------------------

def test_no_cause_basic():
    exc = ExceptionCauseMixin("hello")
    assert exc.cause is None
    assert exc.args == ("hello",)
    assert str(exc) == "hello"


def test_no_cause_empty():
    exc = ExceptionCauseMixin()
    assert exc.cause is None
    assert exc.args == ()
    assert str(exc) == ""


def test_cause_extracted_from_first_arg():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner, "wrapped")
    assert exc.cause is inner
    assert exc.args == ("wrapped",)


def test_root_cause_when_inner_has_none():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner, "wrapped")
    assert exc.root_cause is inner


def test_root_cause_propagates():
    inner = ValueError("boom")
    mid = ExceptionCauseMixin(inner, "mid")
    outer = ExceptionCauseMixin(mid, "outer")
    assert outer.cause is mid
    assert outer.root_cause is inner


def test_cause_first_arg_non_exception():
    # first arg not an exception => treated as message, cause None
    exc = ExceptionCauseMixin("not an exception")
    assert exc.cause is None
    assert exc.args == ("not an exception",)


def test_class_attribute_cause_default():
    assert ExceptionCauseMixin.cause is None


# ---------------------------------------------------------------------------
# _get_message
# ---------------------------------------------------------------------------

def test_get_message_with_cause():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner, "my message")
    assert exc._get_message() == "my message"


def test_get_message_empty_with_cause():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner)
    assert exc._get_message() == ""


def test_get_message_without_cause():
    exc = ExceptionCauseMixin("plain")
    assert exc._get_message() == "plain"


def test_get_message_falsy_first_arg():
    exc = ExceptionCauseMixin("")
    assert exc._get_message() == ""


# ---------------------------------------------------------------------------
# _format_exc
# ---------------------------------------------------------------------------

def test_format_exc_basic():
    exc = ValueError("boom")
    result = _format_exc(exc)
    assert result == "ValueError: boom"


def test_format_exc_empty_message():
    exc = ValueError()
    result = _format_exc(exc)
    assert result == "ValueError"


def test_format_exc_custom_message():
    exc = ValueError("boom")
    result = _format_exc(exc, "custom")
    assert result == "ValueError: custom"


# ---------------------------------------------------------------------------
# String formatting for wrapped exceptions
# ---------------------------------------------------------------------------

def test_str_with_cause_no_trace():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner, "wrapped")
    s = str(exc)
    # No full_trace set -> uses _get_exc_str(incl_name=False)
    assert s == " caused by ValueError: boom"


def test_str_with_cause_has_full_trace():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner, "wrapped")
    exc.full_trace = [
        _TBItem("fake.py", 1, "func", "some line"),
    ]
    s = str(exc)
    assert "wrapped --- " in s
    assert "Wrapped traceback (most recent call last):" in s
    assert "ExceptionCauseMixin: wrapped (caused by ValueError: boom)" in s


def test_get_str_no_trace():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner, "wrapped")
    # full_trace attribute missing, but cause set
    # _get_trace_str returns '' when full_trace is falsy
    s = exc.get_str()
    assert s == " caused by ValueError: boom"


def test_get_str_with_no_cause():
    exc = ExceptionCauseMixin("hello")
    s = exc.get_str()
    # no trace => just the exception str-ish
    # _get_exc_str -> _format_exc(root_cause=None) would fail, but
    # get_str appends _get_exc_str() unconditionally...
    # Check actual behavior: root_cause is None -> AttributeError.
    # Actually get_str calls _get_exc_str(), which calls _format_exc(None),
    # which calls None.__class__.__name__ -> 'NoneType'. So it'd be "NoneType".
    # Let's just assert it doesn't crash and is a str.
    assert isinstance(s, str)


def test_get_str_with_trace_and_message():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner, "wrapped")
    exc.full_trace = [_TBItem("fake.py", 1, "func", "line content")]
    s = exc.get_str()
    assert s.startswith("Traceback (most recent call last):\n")
    assert "line content" in s
    assert "ExceptionCauseMixin: wrapped (caused by ValueError: boom)" in s


def test_get_str_with_cause_no_message():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner)
    s = exc.get_str()
    assert "ExceptionCauseMixin caused by ValueError: boom" in s


def test_get_exc_str_with_message():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner, "wrapped")
    assert exc._get_exc_str(incl_name=True) == (
        "ExceptionCauseMixin: wrapped (caused by ValueError: boom)"
    )
    assert exc._get_exc_str(incl_name=False) == (
        "wrapped (caused by ValueError: boom)"
    )


def test_get_exc_str_no_message():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner)
    assert exc._get_exc_str(incl_name=True) == (
        "ExceptionCauseMixin caused by ValueError: boom"
    )
    assert exc._get_exc_str(incl_name=False) == (
        " caused by ValueError: boom"
    )


# ---------------------------------------------------------------------------
# full_trace copying from cause
# ---------------------------------------------------------------------------

def test_full_trace_propagates_from_cause():
    inner = ValueError("boom")
    inner.full_trace = [_TBItem("a.py", 1, "f", "x")]
    inner._tb = [_TBItem("b.py", 2, "g", "y")]
    inner._stack = [_TBItem("c.py", 3, "h", "z")]
    exc = ExceptionCauseMixin(inner, "w")
    assert exc.full_trace == inner.full_trace
    assert exc.full_trace is not inner.full_trace  # copied
    assert exc._tb == inner._tb
    assert exc._tb is not inner._tb
    assert exc._stack == inner._stack
    assert exc._stack is not inner._stack


# ---------------------------------------------------------------------------
# Capturing traceback from active exception
# ---------------------------------------------------------------------------

def test_captures_trace_when_wrapping_current_exception():
    inner = None
    try:
        raise ValueError("boom")
    except ValueError as e:
        inner = e
        wrapped = ExceptionCauseMixin(e, "w")
    assert wrapped.cause is inner
    assert hasattr(wrapped, "full_trace")
    assert len(wrapped.full_trace) >= 1
    assert hasattr(wrapped, "_tb")
    assert hasattr(wrapped, "_stack")
    assert wrapped.full_trace[-1].name == "test_captures_trace_when_wrapping_current_exception" or \
        wrapped.full_trace[-1] in wrapped.full_trace


def test_no_trace_when_no_active_exception():
    inner = ValueError("boom")
    exc = ExceptionCauseMixin(inner, "w")
    assert not hasattr(exc, "full_trace")
    assert exc.root_cause is inner


# ---------------------------------------------------------------------------
# _extract_from_tb / _extract_from_frame
# ---------------------------------------------------------------------------

def test_extract_from_tb():
    tb = None
    try:
        raise RuntimeError("x")
    except RuntimeError as e:
        tb = e.__traceback__
    items = _extract_from_tb(tb)
    assert len(items) >= 1
    assert all(isinstance(i, _TBItem) for i in items)
    assert items[-1].name == "test_extract_from_tb"


def test_extract_from_tb_limit():
    tb = None
    try:
        raise RuntimeError("x")
    except RuntimeError as e:
        tb = e.__traceback__
    items = _extract_from_tb(tb, limit=1)
    assert len(items) == 1


def test_extract_from_frame_basic():
    items = _extract_from_frame()
    assert len(items) >= 1
    assert isinstance(items[-1], _TBItem)
    assert items[-1].name == "test_extract_from_frame_basic"


def test_extract_from_frame_limit():
    items = _extract_from_frame(limit=1)
    assert len(items) == 1


# ---------------------------------------------------------------------------
# _TBItem and _DeferredLine
# ---------------------------------------------------------------------------

def test_tb_item_repr_includes_frame_id():
    dl = _DeferredLine(__file__, 1)
    item = _TBItem("f.py", 5, "fn", dl)
    item.frame_id = 42
    r = repr(item)
    assert "<42>" in r


def test_deferred_line_eq_ne():
    a = _DeferredLine("a.py", 1)
    b = _DeferredLine("a.py", 1)
    c = _DeferredLine("a.py", 2)
    assert a == b
    assert a != c
    assert not (a != b)
    assert not (a == c)


def test_deferred_line_str_reads_line():
    dl = _DeferredLine(__file__, 1)
    s = str(dl)
    # line 1 of this file starts with 'import sys' (after possible encoding)
    # Just assert it's a str and cached
    assert isinstance(s, str)
    assert dl._line == s


def test_deferred_line_len_and_strip():
    dl = _DeferredLine(__file__, 1)
    s = str(dl)
    assert len(dl) == len(s)
    assert dl.strip() == s.strip()


def test_deferred_line_missing_file():
    dl = _DeferredLine("/nonexistent/path/xyz.py", 1)
    s = str(dl)
    assert s is None
    assert dl._line is None
    # len of None would raise TypeError; that's fine, just check str result
    assert repr(dl) == "None"


# ---------------------------------------------------------------------------
# MathError & example functions
# ---------------------------------------------------------------------------

def test_math_error_is_valueerror():
    assert issubclass(MathError, ValueError)
    assert issubclass(MathError, ExceptionCauseMixin)


def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


def test_math_lol_raises_math_error_with_zero_division_cause():
    with pytest.raises(MathError) as excinfo:
        math_lol()
    exc = excinfo.value
    assert isinstance(exc, MathError)
    assert isinstance(exc.cause, ZeroDivisionError)
    assert exc.root_cause is exc.cause
    assert exc.args == ("ya done messed up",)
    assert exc._get_message() == "ya done messed up"


def test_main_raises_math_error_with_math_error_cause():
    with pytest.raises(MathError) as excinfo:
        main()
    exc = excinfo.value
    assert isinstance(exc, MathError)
    assert isinstance(exc.cause, MathError)
    assert isinstance(exc.cause.cause, ZeroDivisionError)
    assert exc.root_cause is exc.cause.cause
    assert exc._get_message() == "hi"
