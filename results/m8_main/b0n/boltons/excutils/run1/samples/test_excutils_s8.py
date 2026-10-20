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
    _DeferredLine,
    _TBItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Simple(ExceptionCauseMixin, Exception):
    pass


class SimpleValue(ExceptionCauseMixin, ValueError):
    pass


# ---------------------------------------------------------------------------
# ExceptionCauseMixin - basic construction
# ---------------------------------------------------------------------------

def test_new_without_cause():
    exc = Simple("just a message")
    assert exc.cause is None
    assert getattr(exc, "root_cause", None) is None
    assert exc.args == ("just a message",)


def test_new_with_no_args():
    exc = Simple()
    assert exc.cause is None
    assert exc.args == ()


def test_new_with_cause_first_arg():
    cause = ValueError("boom")
    exc = Simple(cause, "wrapper msg")
    assert exc.cause is cause
    assert exc.root_cause is cause
    assert exc.args == ("wrapper msg",)


def test_new_with_cause_and_no_message():
    cause = ValueError("boom")
    exc = Simple(cause)
    assert exc.cause is cause
    assert exc.root_cause is cause
    assert exc.args == ()


def test_first_arg_not_exception_is_message():
    exc = Simple("not an exception")
    assert exc.cause is None
    assert exc.args == ("not an exception",)


def test_root_cause_chain_uses_cause_root_cause():
    root = ValueError("root")
    middle = SimpleValue(root, "middle")
    assert middle.root_cause is root
    outer = SimpleValue(middle, "outer")
    assert outer.cause is middle
    assert outer.root_cause is root


def test_root_cause_none_falls_back_to_cause():
    cause = ValueError("c")
    # A plain ValueError has no root_cause attribute; getattr -> None
    exc = Simple(cause)
    assert exc.root_cause is cause


# ---------------------------------------------------------------------------
# full_trace inheritance
# ---------------------------------------------------------------------------

def test_full_trace_copied_from_cause():
    cause = Simple("c")
    cause.full_trace = ["a", "b"]
    cause._tb = ["tb1"]
    cause._stack = ["st1"]
    exc = Simple(cause, "outer")
    assert exc.full_trace == ["a", "b"]
    # Must be a separate list object
    assert exc.full_trace is not cause.full_trace
    assert exc._tb == ["tb1"]
    assert exc._tb is not cause._tb
    assert exc._stack == ["st1"]
    assert exc._stack is not cause._stack


# ---------------------------------------------------------------------------
# capturing current exception
# ---------------------------------------------------------------------------

def test_captures_current_exception_traceback():
    try:
        raise ValueError("inner")
    except ValueError as ve:
        exc = Simple(ve, "wrapped")
    assert exc.cause is ve
    assert exc.root_cause is ve
    assert hasattr(exc, "full_trace")
    assert isinstance(exc.full_trace, list)
    assert isinstance(exc._tb, list)
    assert isinstance(exc._stack, list)
    # full_trace = _stack[:-1] + _tb
    assert exc.full_trace == exc._stack[:-1] + exc._tb


def test_corrects_root_cause_when_cause_not_current_exception():
    root = ValueError("root")
    try:
        raise ValueError("current")
    except ValueError as current:
        # Pass root (not the current exception), but root_cause is current
        # so branch `cause is exc_value or root_cause is exc_value` is False
        # because root is not current. But if root is current:
        exc = Simple(current)
    assert exc.cause is current
    assert exc.root_cause is current


def test_no_current_exception_leaves_no_full_trace():
    # No exception currently being handled
    exc = Simple(ValueError("standalone"))
    assert exc.cause is not None
    assert not hasattr(exc, "full_trace")


# ---------------------------------------------------------------------------
# get_str / __str__ / _get_message / _get_exc_str
# ---------------------------------------------------------------------------

def test_get_str_without_cause():
    exc = Simple("plain")
    s = exc.get_str()
    assert s == "Simple: plain"


def test_get_str_without_cause_no_message():
    exc = Simple()
    assert exc.get_str() == "Simple: "


def test_str_without_cause():
    exc = Simple("plain")
    assert str(exc) == "plain"


def test_get_message_with_cause_skips_message():
    cause = ValueError("inner")
    exc = Simple(cause)
    assert exc._get_message() == ""


def test_get_message_without_message():
    exc = Simple()
    assert exc._get_message() == ""


def test_get_exc_str_with_cause_and_message():
    cause = ValueError("inner")
    exc = Simple(cause, "outer")
    s = exc._get_exc_str()
    assert s == "Simple: outer (caused by ValueError: inner)"


def test_get_exc_str_with_cause_no_message():
    cause = ValueError("inner")
    exc = Simple(cause)
    s = exc._get_exc_str()
    assert s == "Simple:  caused by ValueError: inner"


def test_get_exc_str_no_name():
    cause = ValueError("inner")
    exc = Simple(cause, "outer")
    s = exc._get_exc_str(incl_name=False)
    assert s == "outer (caused by ValueError: inner)"


def test_get_exc_str_root_cause_used():
    root = ValueError("root")
    middle = SimpleValue(root, "middle")
    outer = SimpleValue(middle, "outer")
    s = outer._get_exc_str()
    assert "ValueError: root" in s
    assert "middle" not in s


def test_str_with_cause_without_trace():
    cause = ValueError("inner")
    exc = Simple(cause, "outer")
    # No full_trace: goes to _get_exc_str(incl_name=False)
    assert str(exc) == "outer (caused by ValueError: inner)"


def test_str_with_cause_with_trace():
    cause = ValueError("inner")
    cause.full_trace = ["  line1\n", "  line2\n"]
    cause._tb = []
    cause._stack = []
    exc = Simple(cause, "outer")
    s = str(exc)
    assert s.startswith("outer --- Wrapped traceback (most recent call last):\n")
    assert "  line1\n" in s
    assert "  line2\n" in s
    assert s.endswith("Simple: outer (caused by ValueError: inner)")


def test_str_with_cause_with_trace_no_message():
    cause = ValueError("inner")
    cause.full_trace = ["  line1\n"]
    cause._tb = []
    cause._stack = []
    exc = Simple(cause)
    s = str(exc)
    assert s.startswith("Wrapped traceback (most recent call last):\n  line1\n")
    assert s.endswith("Simple:  caused by ValueError: inner")


def test_get_str_with_wrapped_trace():
    cause = ValueError("inner")
    cause.full_trace = ["  line1\n"]
    cause._tb = []
    cause._stack = []
    exc = Simple(cause, "outer")
    s = exc.get_str()
    assert s.startswith("Traceback (most recent call last):\n  line1\n")
    assert s.endswith("Simple: outer (caused by ValueError: inner)")


# ---------------------------------------------------------------------------
# _format_exc
# ---------------------------------------------------------------------------

def test_format_exc_default_message():
    exc = ValueError("some message")
    assert _format_exc(exc) == "ValueError: some message"


def test_format_exc_custom_message():
    exc = ValueError("some message")
    assert _format_exc(exc, "custom") == "ValueError: custom"


def test_format_exc_empty_message():
    exc = ValueError()
    assert _format_exc(exc) == "ValueError"


# ---------------------------------------------------------------------------
# _DeferredLine
# ---------------------------------------------------------------------------

def test_deferred_line_equality():
    a = _DeferredLine("file.py", 10)
    b = _DeferredLine("file.py", 10)
    assert a == b
    assert not (a != b)


def test_deferred_line_inequality():
    a = _DeferredLine("file.py", 10)
    b = _DeferredLine("file.py", 11)
    assert a != b
    c = _DeferredLine("other.py", 10)
    assert a != c
    assert not (a == b)


def test_deferred_line_str_caches(tmp_path):
    f = tmp_path / "sample_mod.py"
    f.write_text("line one\nline two\n")
    dl = _DeferredLine(str(f), 2)
    assert str(dl) == "line two"
    # cached
    assert hasattr(dl, "_line")
    assert str(dl) == "line two"


def test_deferred_line_str_missing_line(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("only line\n")
    dl = _DeferredLine(str(f), 100)
    assert str(dl) is None
    assert dl._line is None


def test_deferred_line_repr_len_strip(tmp_path):
    f = tmp_path / "rep.py"
    f.write_text("  spaced   \n")
    dl = _DeferredLine(str(f), 1)
    assert repr(dl) == repr("spaced")
    assert len(dl) == len("spaced")
    assert dl.strip() == "spaced"


# ---------------------------------------------------------------------------
# _TBItem
# ---------------------------------------------------------------------------

def test_tbitem_repr_includes_frame_id():
    dl = _DeferredLine("f.py", 1)
    item = _TBItem("f.py", 1, "func", dl)
    item.frame_id = 12345
    r = repr(item)
    assert "12345" in r
    assert "func" in r


# ---------------------------------------------------------------------------
# _extract_from_frame
# ---------------------------------------------------------------------------

def test_extract_from_frame_default():
    items = _extract_from_frame()
    assert len(items) >= 1
    for item in items:
        assert isinstance(item, _TBItem)
        assert hasattr(item, "frame_id")


def test_extract_from_frame_explicit():
    def inner():
        return _extract_from_frame(sys._getframe(0))

    items = inner()
    assert len(items) >= 1
    assert items[-1].name == "inner"


def test_extract_from_frame_limit():
    def a():
        return b()

    def b():
        return c()

    def c():
        return _extract_from_frame(sys._getframe(0), limit=1)

    items = c()
    assert len(items) == 1
    assert items[0].name == "c"


# ---------------------------------------------------------------------------
# _extract_from_tb
# ---------------------------------------------------------------------------

def test_extract_from_tb():
    try:
        raise ValueError("x")
    except ValueError:
        tb = sys.exc_info()[2]
    items = _extract_from_tb(tb)
    assert len(items) == 1
    assert items[0].name == "test_extract_from_tb"
    assert hasattr(items[0], "frame_id")


def test_extract_from_tb_limit():
    def a():
        raise ValueError("x")

    def b():
        a()

    try:
        b()
    except ValueError:
        tb = sys.exc_info()[2]
    items = _extract_from_tb(tb, limit=1)
    assert len(items) == 1


# ---------------------------------------------------------------------------
# Example classes / functions
# ---------------------------------------------------------------------------

def test_math_error_is_value_error():
    assert issubclass(MathError, ValueError)
    assert issubclass(MathError, ExceptionCauseMixin)


def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


def test_math_lol_raises_math_error():
    with pytest.raises(MathError) as excinfo:
        math_lol()
    exc = excinfo.value
    assert isinstance(exc.cause, ZeroDivisionError)
    assert exc.root_cause is exc.cause
    assert exc._get_message() == "ya done messed up"
    assert "ya done messed up" in str(exc)


def test_math_lol_recursion_depth_ok():
    # n >= 3 skips recursion
    with pytest.raises(MathError):
        math_lol(n=3)


def test_main_raises_math_error():
    with pytest.raises(MathError) as excinfo:
        main()
    exc = excinfo.value
    assert isinstance(exc.cause, MathError)
    assert isinstance(exc.root_cause, ZeroDivisionError)
    assert "hi" in exc._get_exc_str()


def test_main_str_contains_wrapped_traceback():
    with pytest.raises(MathError) as excinfo:
        main()
    s = str(excinfo.value)
    assert "Wrapped traceback (most recent call last):" in s
    assert "ZeroDivisionError" in s


# ---------------------------------------------------------------------------
# Cause provided but no current exception - edge case
# ---------------------------------------------------------------------------

def test_cause_alone_no_trace_no_full_trace():
    cause = ValueError("x")
    exc = Simple(cause)
    assert exc.cause is cause
    assert not hasattr(exc, "full_trace")
    assert str(exc) == "Simple:  caused by ValueError: x"
