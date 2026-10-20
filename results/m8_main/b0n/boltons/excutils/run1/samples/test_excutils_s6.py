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
    _TBItem,
    _DeferredLine,
)


# ---------------------------------------------------------------------------
# Basic construction / cause handling
# ---------------------------------------------------------------------------

def test_plain_construction_no_cause():
    exc = ExceptionCauseMixin("some message")
    assert exc.cause is None
    assert not hasattr(exc, "full_trace")
    assert str(exc) == "some message"


def test_construction_with_cause():
    cause = ValueError("boom")
    exc = ExceptionCauseMixin(cause, "wrapper message")
    assert exc.cause is cause
    assert exc.args == ("wrapper message",)
    # Root cause is set to the cause since it has no root_cause.
    assert exc.root_cause is cause


def test_construction_root_cause_chain():
    inner = ValueError("inner")
    middle = ExceptionCauseMixin(inner, "middle")
    assert middle.root_cause is inner

    outer = ExceptionCauseMixin(middle, "outer")
    assert outer.cause is middle
    # root_cause should collapse to the innermost.
    assert outer.root_cause is inner


def test_first_arg_exception_treated_as_cause_only():
    cause = ValueError("boom")
    exc = ExceptionCauseMixin(cause)
    assert exc.cause is cause
    assert exc.args == ()


def test_non_exception_first_arg_is_not_cause():
    exc = ExceptionCauseMixin("just a message")
    assert exc.cause is None
    assert exc.args == ("just a message",)


# ---------------------------------------------------------------------------
# _get_message / _get_exc_str / get_str / __str__
# ---------------------------------------------------------------------------

def test_get_message_no_cause():
    exc = ExceptionCauseMixin("hello")
    assert exc._get_message() == "hello"


def test_get_message_with_cause_uses_second_arg():
    exc = ExceptionCauseMixin(ValueError("boom"), "wrapped message")
    assert exc._get_message() == "wrapped message"


def test_get_message_empty():
    exc = ExceptionCauseMixin()
    assert exc._get_message() == ""


def test_get_str_no_cause():
    exc = ExceptionCauseMixin("plain")
    # When there's no cause, _get_trace_str returns super().__repr__(),
    # so get_str includes the repr prefixed by a traceback header.
    result = exc.get_str()
    assert "Traceback (most recent call last):\n" in result
    assert repr(exc) in result


def test_get_str_with_cause():
    cause = ValueError("original")
    exc = ExceptionCauseMixin(cause, "wrapped")
    result = exc.get_str()
    assert "Traceback (most recent call last):\n" in result
    assert "ExceptionCauseMixin: wrapped (caused by ValueError: original)" in result


def test_str_no_cause_is_super_str():
    exc = ExceptionCauseMixin("only message")
    assert str(exc) == "only message"


def test_str_with_cause_no_trace():
    cause = ValueError("original")
    exc = ExceptionCauseMixin(cause, "wrapped")
    # No full_trace was extracted when not in an exception context.
    result = str(exc)
    assert result == "wrapped (caused by ValueError: original)"


def test_str_with_cause_no_message():
    cause = ValueError("original")
    exc = ExceptionCauseMixin(cause)
    result = str(exc)
    assert result == " caused by ValueError: original"


# ---------------------------------------------------------------------------
# full_trace extraction from a live exception
# ---------------------------------------------------------------------------

def test_full_trace_extraction_inside_except():
    captured = {}

    def raiser():
        try:
            raise ValueError("boom")
        except ValueError as e:
            wrapped = ExceptionCauseMixin(e, "wrapped")
            captured["wrapped"] = wrapped
            return wrapped

    wrapped = raiser()
    assert wrapped.cause is not None
    assert wrapped.root_cause is wrapped.cause
    assert hasattr(wrapped, "full_trace")
    assert isinstance(wrapped.full_trace, list)
    assert len(wrapped.full_trace) > 0

    # All items should be _TBItem
    for item in wrapped.full_trace:
        assert isinstance(item, _TBItem)

    # _tb and _stack were also populated.
    assert hasattr(wrapped, "_tb")
    assert hasattr(wrapped, "_stack")


# ---------------------------------------------------------------------------
# ExceptionCauseMixin as a mixin
# ---------------------------------------------------------------------------

def test_math_error_inheritance():
    assert issubclass(MathError, ExceptionCauseMixin)
    assert issubclass(MathError, ValueError)


def test_math_error_wrapping_value_error():
    ve = ValueError("orig")
    me = MathError(ve, "wrapped")
    assert isinstance(me, ValueError)
    assert me.cause is ve
    assert me._get_message() == "wrapped"


# ---------------------------------------------------------------------------
# whoops_math raises ZeroDivisionError
# ---------------------------------------------------------------------------

def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


# ---------------------------------------------------------------------------
# math_lol behavior
# ---------------------------------------------------------------------------

def test_math_lol_raises_math_error():
    with pytest.raises(MathError) as excinfo:
        math_lol()
    me = excinfo.value
    assert isinstance(me, ValueError)
    assert isinstance(me.cause, ZeroDivisionError)
    assert me.root_cause is me.cause
    assert me._get_message() == "ya done messed up"


def test_math_lol_recursion_then_error():
    # Call with n < 3 goes recursively then errors.
    for n in (0, 1, 2):
        with pytest.raises(MathError):
            math_lol(n=n)


def test_math_lol_n_3_directly():
    with pytest.raises(MathError) as excinfo:
        math_lol(3)
    assert isinstance(excinfo.value.cause, ZeroDivisionError)


def test_math_lol_has_full_trace():
    try:
        math_lol()
    except MathError as me:
        assert hasattr(me, "full_trace")
        assert len(me.full_trace) > 0
        # Include the frame from whoops_math's division.
        names = [item.name for item in me.full_trace]
        assert "whoops_math" in names


# ---------------------------------------------------------------------------
# main behavior
# ---------------------------------------------------------------------------

def test_main_raises_math_error_wrapping_math_error():
    with pytest.raises(MathError) as excinfo:
        main()
    outer = excinfo.value
    assert outer._get_message() == "hi"
    assert isinstance(outer.cause, MathError)
    # The inner MathError wraps ZeroDivisionError
    inner = outer.cause
    assert isinstance(inner.cause, ZeroDivisionError)
    # root_cause chains down to ZeroDivisionError
    assert isinstance(outer.root_cause, ZeroDivisionError)


def test_main_outer_str_mentions_message():
    try:
        main()
    except MathError as me:
        s = str(me)
        assert "hi" in s
        # The chain ultimately points to ZeroDivisionError
        assert "ZeroDivisionError" in s


# ---------------------------------------------------------------------------
# _format_exc helper
# ---------------------------------------------------------------------------

def test_format_exc_default_message():
    exc = ValueError("boom")
    result = _format_exc(exc)
    assert result == "ValueError: boom"


def test_format_exc_custom_message():
    exc = ValueError("boom")
    result = _format_exc(exc, "custom")
    assert result == "ValueError: custom"


def test_format_exc_empty_message():
    exc = ValueError()
    result = _format_exc(exc)
    # _format_final_exc_line with empty str message just yields the name
    assert result.startswith("ValueError")


# ---------------------------------------------------------------------------
# _DeferredLine
# ---------------------------------------------------------------------------

def test_deferred_line_str_reads_current_file(tmp_path):
    p = tmp_path / "sample.py"
    p.write_text("line one\nline two\nline three\n")
    dl = _DeferredLine(str(p), 2)
    assert str(dl) == "line two"
    assert dl.strip() == "line two"
    assert dl.__repr__() == repr("line two")
    assert len(dl) == len("line two")


def test_deferred_line_missing_file():
    dl = _DeferredLine("/nonexistent/path/xyz.py", 1)
    # getline returns '' -> line is None
    assert str(dl) is None


def test_deferred_line_equality():
    a = _DeferredLine("/some/file.py", 10)
    b = _DeferredLine("/some/file.py", 10)
    c = _DeferredLine("/some/file.py", 11)
    d = _DeferredLine("/other/file.py", 10)
    assert a == b
    assert a != c
    assert a != d


def test_deferred_line_caches_line(tmp_path):
    p = tmp_path / "cache.py"
    p.write_text("hello\n")
    dl = _DeferredLine(str(p), 1)
    first = str(dl)
    assert first == "hello"
    # Modify file; cached value should be returned unchanged.
    p.write_text("changed\n")
    assert str(dl) == "hello"


def test_deferred_line_module_globals_filtered():
    dl = _DeferredLine("/some/file.py", 1, {"__name__": "mod", "other": 1, "__loader__": None})
    assert dl.module_globals == {"__name__": "mod", "__loader__": None}


# ---------------------------------------------------------------------------
# _extract_from_tb
# ---------------------------------------------------------------------------

def test_extract_from_tb_basic():
    try:
        raise ValueError("boom")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        try:
            items = _extract_from_tb(exc_tb)
            assert isinstance(items, list)
            assert len(items) >= 1
            for item in items:
                assert isinstance(item, _TBItem)
                assert hasattr(item, "frame_id")
        finally:
            del exc_tb


def test_extract_from_tb_limit():
    try:
        raise ValueError("boom")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        try:
            all_items = _extract_from_tb(exc_tb)
            limited = _extract_from_tb(exc_tb, limit=1)
            assert len(limited) == 1
            if all_items:
                assert limited[0] == all_items[0]
        finally:
            del exc_tb


def test_extract_from_tb_zero_limit():
    try:
        raise ValueError("boom")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        try:
            items = _extract_from_tb(exc_tb, limit=0)
            assert items == []
        finally:
            del exc_tb


# ---------------------------------------------------------------------------
# _extract_from_frame
# ---------------------------------------------------------------------------

def test_extract_from_frame_explicit():
    frame = sys._getframe()
    items = _extract_from_frame(frame)
    assert isinstance(items, list)
    assert len(items) >= 1
    assert items[-1].name == "test_extract_from_frame_explicit"
    for item in items:
        assert isinstance(item, _TBItem)
        assert hasattr(item, "frame_id")


def test_extract_from_frame_default_uses_caller():
    items = _extract_from_frame()
    assert isinstance(items, list)
    assert len(items) >= 1
    # The last item (after reverse) is the current frame, i.e. this function.
    assert items[-1].name == "test_extract_from_frame_default_uses_caller"


def test_extract_from_frame_limit():
    frame = sys._getframe()
    items = _extract_from_frame(frame, limit=1)
    assert len(items) == 1


# ---------------------------------------------------------------------------
# _TBItem repr
# ---------------------------------------------------------------------------

def test_tb_item_repr_contains_frame_id():
    dl = _DeferredLine("/some/file.py", 1)
    item = _TBItem("/some/file.py", 1, "funcname", dl)
    item.frame_id = 12345
    r = repr(item)
    assert "12345" in r
    assert "funcname" in r


# ---------------------------------------------------------------------------
# Exception inside construction when no active exception
# ---------------------------------------------------------------------------

def test_wrapping_outside_except_no_full_trace():
    cause = ValueError("boom")
    exc = ExceptionCauseMixin(cause, "wrapped")
    # There is no active exception, so no full_trace should be set.
    assert not hasattr(exc, "full_trace")
    assert str(exc) == "wrapped (caused by ValueError: boom)"
