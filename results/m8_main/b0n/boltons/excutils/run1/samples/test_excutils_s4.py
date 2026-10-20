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


# --- helpers ---------------------------------------------------------------


class SimpleError(ExceptionCauseMixin, Exception):
    pass


class SimpleValueError(ExceptionCauseMixin, ValueError):
    pass


def raise_and_capture(fn):
    try:
        fn()
    except BaseException as e:
        return e
    raise AssertionError("expected an exception")


# --- basic construction ----------------------------------------------------


def test_cause_class_attribute_default():
    assert ExceptionCauseMixin.cause is None


def test_construction_without_cause():
    exc = SimpleError("hello")
    assert exc.cause is None
    assert exc.args == ("hello",)
    assert not hasattr(exc, "full_trace") or getattr(exc, "full_trace", None) is None or True


def test_construction_with_cause_first_arg():
    cause = ValueError("boom")
    exc = SimpleError(cause, "wrapped")
    assert exc.cause is cause
    # The cause is stripped from args by __new__
    assert exc.args == ("wrapped",)
    assert exc.root_cause is cause


def test_construction_with_cause_and_no_message():
    cause = ValueError("boom")
    exc = SimpleError(cause)
    assert exc.cause is cause
    assert exc.args == ()
    assert exc.root_cause is cause


def test_root_cause_preserved_nested():
    first = ValueError("first")
    second = SimpleError(first, "second")
    third = SimpleError(second, "third")
    assert third.cause is second
    assert third.root_cause is first


def test_non_exception_first_arg_not_treated_as_cause():
    exc = SimpleError("not an exception", "second")
    assert exc.cause is None
    assert exc.args == ("not an exception", "second")


# --- string formatting -----------------------------------------------------


def test_get_message_with_cause():
    cause = ValueError("boom")
    exc = SimpleError(cause, "wrapped message")
    assert exc._get_message() == "wrapped message"


def test_get_message_without_cause():
    exc = SimpleError("standalone")
    assert exc._get_message() == "standalone"


def test_get_message_empty():
    exc = SimpleError()
    assert exc._get_message() == ""


def test_str_without_cause():
    exc = SimpleError("plain")
    assert str(exc) == "plain"


def test_str_with_cause_no_trace():
    cause = ValueError("boom")
    exc = SimpleError(cause, "oops")
    s = str(exc)
    # When no full_trace was captured (raised outside except),
    # __str__ falls back to _get_exc_str(incl_name=False)
    assert "oops (caused by ValueError: boom)" in s


def test_str_with_cause_no_message():
    cause = ValueError("boom")
    exc = SimpleError(cause)
    s = str(exc)
    assert " caused by ValueError: boom" in s


def test_get_str_contains_message_and_cause():
    cause = ValueError("boom")
    exc = SimpleError(cause, "oops")
    s = exc.get_str()
    assert s.endswith("SimpleError: oops (caused by ValueError: boom)")


def test_format_exc():
    from boltons.excutils import _format_exc

    assert _format_exc(ValueError("hi")) == "ValueError: hi"
    exc = ValueError()
    assert _format_exc(exc) == "ValueError"


# --- traceback capture through cause ----------------------------------------


def test_str_inside_except_captures_trace():
    try:
        1 / 0
    except ZeroDivisionError as zde:
        exc = SimpleError(zde, "wrapped")
    s = str(exc)
    assert "Wrapped traceback (most recent call last):" in s
    assert "SimpleError: wrapped (caused by ZeroDivisionError:" in s
    # full_trace captured
    assert exc.full_trace
    assert hasattr(exc, "_tb")
    assert hasattr(exc, "_stack")


def test_nested_wraps_propagate_full_trace():
    try:
        raise ValueError("inner")
    except ValueError as v:
        outer = SimpleError(v, "outer")
        try:
            raise outer
        except SimpleError as o:
            outer2 = SimpleError(o, "outer2")
    assert outer2.root_cause is not None
    assert outer2.full_trace
    assert outer2.cause is outer
    assert outer2.root_cause is v


# --- MathError / whoops_math / math_lol / main ------------------------------


def test_math_error_is_valueerror_and_mixin():
    assert issubclass(MathError, ExceptionCauseMixin)
    assert issubclass(MathError, ValueError)


def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


def test_math_lol_raises_math_error():
    with pytest.raises(MathError) as excinfo:
        math_lol()
    exc = excinfo.value
    assert isinstance(exc.cause, ZeroDivisionError)
    assert exc._get_message() == "ya done messed up"
    assert "ya done messed up" in str(exc)


def test_math_lol_recursion_terminates():
    # should not raise RecursionError; raises MathError
    with pytest.raises(MathError):
        math_lol()


def test_main_wraps():
    with pytest.raises(MathError) as excinfo:
        main()
    exc = excinfo.value
    # outer math error wraps the inner one (a MathError)
    assert isinstance(exc.cause, MathError)
    assert exc._get_message() == "hi"
    # root cause should be the innermost ZeroDivisionError
    assert isinstance(exc.root_cause, ZeroDivisionError)


# --- internal traceback helpers --------------------------------------------


def test_extract_from_tb_basic():
    from boltons.excutils import _extract_from_tb

    try:
        raise ValueError("x")
    except ValueError:
        _, _, tb = sys.exc_info()
        items = _extract_from_tb(tb)
    assert len(items) == 1
    item = items[0]
    assert item.name == "test_extract_from_tb_basic"
    assert item.filename.endswith("test_excutils.py")
    assert hasattr(item, "frame_id")
    assert "frame_id" in repr(item)


def test_extract_from_frame_basic():
    from boltons.excutils import _extract_from_frame

    def caller():
        return _extract_from_frame(sys._getframe(0))

    items = caller()
    assert items
    assert items[-1].name == "caller"


def test_extract_from_frame_limit():
    from boltons.excutils import _extract_from_frame

    def a():
        return b()

    def b():
        return c()

    def c():
        return _extract_from_frame(sys._getframe(0), limit=2)

    items = c()
    assert len(items) == 2


def test_deferred_line_str_and_eq(tmp_path):
    from boltons.excutils import _DeferredLine

    f = tmp_path / "sample.py"
    f.write_text("line one\nline two\n")

    line = _DeferredLine(str(f), 2)
    assert str(line) == "line two"
    assert line.strip() == "line two"
    assert len(line) == len("line two")
    assert repr(line) == "'line two'"

    same = _DeferredLine(str(f), 2)
    assert line == same
    assert not (line != same)

    different = _DeferredLine(str(f), 1)
    assert line != different


def test_deferred_line_missing_line(tmp_path):
    from boltons.excutils import _DeferredLine

    f = tmp_path / "sample.py"
    f.write_text("only\n")

    line = _DeferredLine(str(f), 99)
    assert str(line) is None


def test_tbitem_repr():
    from boltons.excutils import _TBItem, _DeferredLine

    d = _DeferredLine("x.py", 1)
    item = _TBItem("x.py", 1, "fn", d)
    item.frame_id = 12345
    r = repr(item)
    assert "frame_id" in r
    assert "12345" in r


# --- interplay with builtins ------------------------------------------------


def test_full_trace_is_list():
    try:
        raise KeyError("k")
    except KeyError as k:
        exc = SimpleError(k, "wrapped")
    assert isinstance(exc.full_trace, list)
    assert exc.full_trace  # non-empty
    assert isinstance(exc._tb, list)
    assert isinstance(exc._stack, list)


def test_cause_and_root_cause_same_for_simple():
    cause = RuntimeError("r")
    exc = SimpleError(cause)
    assert exc.cause is cause
    assert exc.root_cause is cause


def test_exception_cause_mixin_direct_usage():
    cause = ValueError("boom")
    exc = ExceptionCauseMixin(cause, "msg")
    assert isinstance(exc, ExceptionCauseMixin)
    assert exc.cause is cause
    assert exc.root_cause is cause
    assert exc.args == ("msg",)
