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
import boltons.excutils as excutils


# ---------------------------------------------------------------------------
# Helpers / small subclasses used by tests
# ---------------------------------------------------------------------------

class WrapError(ExceptionCauseMixin, ValueError):
    pass


class PlainWrap(ExceptionCauseMixin, Exception):
    pass


# ---------------------------------------------------------------------------
# Basic subclass sanity
# ---------------------------------------------------------------------------

def test_subclass_is_mixin_and_valueerror():
    assert issubclass(MathError, ExceptionCauseMixin)
    assert issubclass(MathError, ValueError)


def test_default_cause_is_none_on_class():
    assert ExceptionCauseMixin.cause is None


# ---------------------------------------------------------------------------
# __new__ behavior
# ---------------------------------------------------------------------------

def test_no_cause_no_args():
    exc = MathError()
    assert exc.cause is None
    assert exc.args == ()
    assert not hasattr(exc, "full_trace")
    assert str(exc) == ""


def test_cause_extracted_from_first_arg():
    inner = ValueError("inner")
    exc = MathError(inner)
    assert exc.cause is inner
    assert exc.args == ()


def test_cause_with_message():
    inner = ValueError("inner")
    exc = MathError(inner, "outer msg")
    assert exc.cause is inner
    assert exc.args == ("outer msg",)


def test_non_exception_first_arg_is_not_cause():
    exc = MathError("not an exception")
    assert exc.cause is None
    assert exc.args == ("not an exception",)


def test_root_cause_of_simple_cause_is_cause():
    inner = ValueError("inner")
    exc = MathError(inner, "msg")
    assert exc.root_cause is inner


def test_root_cause_chains_from_nested_wrapped():
    inner = ValueError("inner")
    mid = MathError(inner, "mid")
    outer = MathError(mid, "outer")
    # mid.root_cause is inner, so outer.root_cause should be inner
    assert mid.root_cause is inner
    assert outer.cause is mid
    assert outer.root_cause is inner


# ---------------------------------------------------------------------------
# _get_message / _get_exc_str / __str__
# ---------------------------------------------------------------------------

def test_get_message_no_cause():
    exc = MathError("hello")
    assert exc._get_message() == "hello"


def test_get_message_empty():
    exc = MathError()
    assert exc._get_message() == ""


def test_get_message_with_cause_and_message():
    # Note: __new__ strips a leading Exception arg from args, so when
    # cause is set, args does not actually contain the cause.
    exc = MathError(ValueError("x"), "msg")
    # args == ("msg",); _get_message slices args[1:] when cause is set,
    # so the message is dropped (reflecting current behavior).
    assert exc.args == ("msg",)
    assert exc._get_message() == ""


def test_get_exc_str_with_message():
    inner = ValueError("inner")
    exc = MathError(inner, "msg")
    # Because _get_message returns '' here (see above), _get_exc_str
    # uses the "caused by" branch without message.
    s = exc._get_exc_str()
    assert s == "MathError:  caused by ValueError: inner"


def test_get_exc_str_without_incl_name():
    inner = ValueError("inner")
    exc = MathError(inner, "msg")
    s = exc._get_exc_str(incl_name=False)
    assert s == " caused by ValueError: inner"


def test_str_no_cause():
    exc = MathError("hello")
    assert str(exc) == "hello"


def test_str_no_cause_empty():
    exc = MathError()
    assert str(exc) == ""


# ---------------------------------------------------------------------------
# _format_exc
# ---------------------------------------------------------------------------

def test_format_exc_basic():
    inner = ValueError("inner")
    assert excutils._format_exc(inner) == "ValueError: inner"


def test_format_exc_no_message_strips():
    inner = ValueError()
    assert excutils._format_exc(inner) == "ValueError"


# ---------------------------------------------------------------------------
# Captured-with-traceback behavior
# ---------------------------------------------------------------------------

def test_new_captures_current_exception_context():
    try:
        raise ValueError("boom")
    except ValueError as e:
        exc = MathError(e, "wrapped")
        caught = e
    assert exc.cause is caught
    assert exc.root_cause is caught
    assert hasattr(exc, "full_trace")
    assert isinstance(exc.full_trace, list)
    assert isinstance(exc.full_trace[-1].filename, str)


def test_get_str_contains_traceback_and_cause():
    try:
        raise ValueError("boom")
    except ValueError as e:
        exc = MathError(e, "wrapped")
    s = exc.get_str()
    assert s.startswith("Traceback (most recent call last):\n")
    assert "ValueError: boom" in s


def test_get_str_no_cause():
    exc = MathError("hello")
    s = exc.get_str()
    assert "Traceback (most recent call last):" not in s
    assert "hello" in s
    assert s.endswith("hello")


# ---------------------------------------------------------------------------
# _extract_from_tb / _extract_from_frame
# ---------------------------------------------------------------------------

def test_extract_from_tb():
    try:
        raise RuntimeError("x")
    except RuntimeError:
        tb = sys.exc_info()[2]
        items = excutils._extract_from_tb(tb)
    assert len(items) >= 1
    assert isinstance(items[0].filename, str)
    assert items[0].name == "test_extract_from_tb"


def test_extract_from_tb_limit():
    try:
        raise RuntimeError("x")
    except RuntimeError:
        tb = sys.exc_info()[2]
        items = excutils._extract_from_tb(tb, limit=0)
    assert items == []


def test_extract_from_frame():
    items = excutils._extract_from_frame(limit=1)
    assert len(items) >= 1
    assert items[-1].name == "test_extract_from_frame"


def test_extract_from_frame_none_uses_caller():
    items = excutils._extract_from_frame(limit=1)
    assert len(items) >= 1


# ---------------------------------------------------------------------------
# _TBItem / _DeferredLine
# ---------------------------------------------------------------------------

def test_tbitem_repr_contains_frame_id():
    item = excutils._TBItem("f.py", 1, "func", "line")
    item.frame_id = 123
    assert "123" in repr(item)


def test_deferred_line_eq_non_eq():
    dl1 = excutils._DeferredLine("a.py", 1)
    dl2 = excutils._DeferredLine("a.py", 1)
    dl3 = excutils._DeferredLine("a.py", 2)
    assert dl1 == dl2
    assert dl1 != dl3


def test_deferred_line_missing_file_returns_none_str():
    dl = excutils._DeferredLine("/nonexistent/xyz_fake_file.py", 1)
    assert str(dl) is None
    assert repr(dl) == "repr(None)"


def test_deferred_line_module_globals_filtered():
    dl = excutils._DeferredLine("a.py", 1, {"__name__": "m", "other": 1})
    assert dl.module_globals == {"__name__": "m"}
    dl2 = excutils._DeferredLine("a.py", 1, None)
    assert dl2.module_globals == {}


# ---------------------------------------------------------------------------
# whoops_math / math_lol / main
# ---------------------------------------------------------------------------

def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


def test_math_lol_raises_matherror():
    with pytest.raises(MathError) as ei:
        math_lol()
    exc = ei.value
    assert isinstance(exc.cause, ZeroDivisionError)
    assert isinstance(exc.root_cause, ZeroDivisionError)


def test_math_lol_with_arg(monkeypatch):
    calls = []

    def fake_whoops():
        calls.append(1)
        return 42

    monkeypatch.setattr(excutils, "whoops_math", fake_whoops)
    assert math_lol() == 42
    assert len(calls) == 1


def test_main_raises_matherror():
    with pytest.raises(MathError) as ei:
        main()
    exc = ei.value
    # Outer cause is a MathError raised in math_lol
    assert isinstance(exc.cause, MathError)
    assert isinstance(exc.root_cause, ZeroDivisionError)


def test_main_raising_caught_as_valueerror():
    with pytest.raises(ValueError):
        main()
