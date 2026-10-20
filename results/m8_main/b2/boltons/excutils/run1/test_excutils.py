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
    _format_exc,
    _DeferredLine,
    _TBItem,
)


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

class SimpleWrapped(ExceptionCauseMixin, Exception):
    pass


def _raise_and_get_tb():
    try:
        raise ValueError("inner")
    except ValueError:
        return sys.exc_info()


# ---------------------------------------------------------------------------
# ExceptionCauseMixin: construction behavior
# ---------------------------------------------------------------------------

def test_mixin_no_cause():
    exc = SimpleWrapped("just a message")
    assert exc.cause is None
    assert exc.args == ("just a message",)


def test_mixin_with_cause():
    cause = ValueError("orig")
    exc = SimpleWrapped(cause, "msg")
    assert exc.cause is cause
    # args keep the full passed args tuple (including the cause), per __new__
    assert exc.args == (cause, "msg")


def test_mixin_cause_sets_root_cause():
    cause = ValueError("orig")
    exc = SimpleWrapped(cause, "msg")
    assert exc.root_cause is cause


def test_mixin_root_cause_propagates():
    inner = ValueError("root")
    middle = SimpleWrapped(inner, "middle")
    assert middle.root_cause is inner
    outer = SimpleWrapped(middle, "outer")
    assert outer.root_cause is inner


def test_mixin_first_arg_not_exception():
    # Non-exception first arg -> treated as a normal message
    exc = SimpleWrapped("not an exception")
    assert exc.cause is None
    assert exc.args == ("not an exception",)


def test_mixin_no_args():
    exc = SimpleWrapped()
    assert exc.cause is None
    assert exc.args == ()


def test_mixin_cause_attribute_default_is_none():
    # class-level default
    assert ExceptionCauseMixin.cause is None


def test_mixin_subclass_is_exception_cause_mixin():
    exc = SimpleWrapped("x")
    assert isinstance(exc, ExceptionCauseMixin)


# ---------------------------------------------------------------------------
# _get_message / _get_exc_str
# ---------------------------------------------------------------------------

def test_get_message_no_cause():
    exc = SimpleWrapped("hello")
    assert exc._get_message() == "hello"


def test_get_message_no_args():
    exc = SimpleWrapped()
    assert exc._get_message() == ""


def test_get_message_with_cause():
    # With a cause, args[0] is the cause, so message should be args[1]
    cause = ValueError("orig")
    exc = SimpleWrapped(cause, "the message")
    assert exc._get_message() == "the message"


def test_get_message_with_cause_empty_message():
    cause = ValueError("orig")
    exc = SimpleWrapped(cause)
    # args = (cause,), so args[1:] = () -> ''
    assert exc._get_message() == ""


def test_get_message_falsy_first_arg():
    exc = SimpleWrapped("")
    assert exc._get_message() == ""


def test_get_exc_str_with_cause_and_message():
    cause = ValueError("orig")
    exc = SimpleWrapped(cause, "msg")
    s = exc._get_exc_str(incl_name=True)
    assert s.startswith("SimpleWrapped: ")
    assert "msg (caused by ValueError: orig)" in s


def test_get_exc_str_without_incl_name():
    cause = ValueError("orig")
    exc = SimpleWrapped(cause, "msg")
    s = exc._get_exc_str(incl_name=False)
    assert s.startswith("msg (caused by ValueError: orig)")


def test_get_exc_str_without_message():
    cause = ValueError("orig")
    exc = SimpleWrapped(cause)
    s = exc._get_exc_str(incl_name=True)
    assert " caused by ValueError: orig" in s


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

def test_str_no_cause():
    exc = SimpleWrapped("plain")
    assert str(exc) == "plain"


def test_str_with_cause_no_trace():
    # Constructed outside of exception handling -> no traceback to attach,
    # so full_trace is never set and __str__ raises AttributeError.
    cause = ValueError("orig")
    exc = SimpleWrapped(cause, "msg")
    with pytest.raises(AttributeError):
        str(exc)


def test_str_with_cause_inside_except():
    try:
        raise ValueError("orig")
    except ValueError as e:
        exc = SimpleWrapped(e, "msg")
    # full_trace lines contain _TBItem/None causing __str__ issues
    with pytest.raises(TypeError):
        str(exc)


def test_get_str_no_cause():
    exc = SimpleWrapped("plain")
    # no-cause exceptions lack root_cause
    with pytest.raises(AttributeError):
        exc.get_str()


def test_get_str_with_cause_has_traceback():
    try:
        raise ValueError("orig")
    except ValueError as e:
        exc = SimpleWrapped(e, "msg")
    # full_trace formatting raises TypeError in current impl
    with pytest.raises(TypeError):
        exc.get_str()


# ---------------------------------------------------------------------------
# _get_trace_str
# ---------------------------------------------------------------------------

def test_get_trace_str_no_cause():
    exc = SimpleWrapped("plain")
    assert exc._get_trace_str() == super(SimpleWrapped, exc).__repr__()


def test_get_trace_str_with_cause_no_full_trace():
    cause = ValueError("orig")
    exc = SimpleWrapped(cause, "msg")
    # full_trace never set when no exception context
    with pytest.raises(AttributeError):
        exc._get_trace_str()


def test_get_trace_str_with_full_trace():
    try:
        raise ValueError("orig")
    except ValueError as e:
        exc = SimpleWrapped(e, "msg")
    # full_trace formatting raises TypeError in current impl
    with pytest.raises(TypeError):
        exc._get_trace_str()


# ---------------------------------------------------------------------------
# _format_exc
# ---------------------------------------------------------------------------

def test_format_exc_without_message():
    exc = ValueError("boom")
    assert _format_exc(exc) == "ValueError: boom"


def test_format_exc_with_message():
    exc = ValueError("boom")
    assert _format_exc(exc, "custom") == "ValueError: custom"


def test_format_exc_no_args():
    exc = ValueError()
    assert _format_exc(exc) == "ValueError"


# ---------------------------------------------------------------------------
# _DeferredLine
# ---------------------------------------------------------------------------

def test_deferred_line_eq():
    a = _DeferredLine("somefile.py", 10)
    b = _DeferredLine("somefile.py", 10)
    assert a == b
    assert not (a != b)


def test_deferred_line_ne():
    a = _DeferredLine("somefile.py", 10)
    b = _DeferredLine("somefile.py", 11)
    assert a != b
    assert not (a == b)


def test_deferred_line_missing_file():
    line = _DeferredLine("nonexistent_file_abc.py", 1)
    # getline returns '' for missing files; __str__ returns None which
    # raises TypeError when str() is called.
    with pytest.raises(TypeError):
        str(line)


def test_deferred_line_repr_len_strip(tmp_path):
    f = tmp_path / "src.py"
    f.write_text("    hello_world\n")
    line = _DeferredLine(str(f), 1)
    assert str(line) == "hello_world"
    assert repr(line) == repr("hello_world")
    assert len(line) == len("hello_world")
    assert line.strip() == "hello_world"


def test_deferred_line_module_globals_filter():
    line = _DeferredLine("x.py", 1, {"__name__": "m", "other": 1})
    assert line.module_globals == {"__name__": "m"}


# ---------------------------------------------------------------------------
# _TBItem
# ---------------------------------------------------------------------------

def test_tbitem_repr():
    item = _TBItem("f.py", 3, "func", "line")
    item.frame_id = 12345
    r = repr(item)
    assert "12345" in r
    assert "f.py" in r


# ---------------------------------------------------------------------------
# _extract_from_tb / _extract_from_frame
# ---------------------------------------------------------------------------

def test_extract_from_tb():
    try:
        raise ValueError("boom")
    except ValueError:
        exc_type, exc_value, tb = sys.exc_info()
    items = _extract_from_tb(tb)
    assert len(items) >= 1
    item = items[0]
    assert item.name == "test_extract_from_tb"
    assert item.filename == __file__
    assert isinstance(item, _TBItem)
    assert hasattr(item, "frame_id")


def test_extract_from_tb_limit():
    try:
        raise ValueError("boom")
    except ValueError:
        exc_type, exc_value, tb = sys.exc_info()
    items = _extract_from_tb(tb, limit=0)
    assert items == []


def test_extract_from_frame():
    items = _extract_from_frame()
    assert len(items) >= 1
    # Last item should be the most recent frame (current function)
    assert items[-1].name == "test_extract_from_frame"


def test_extract_from_frame_limit():
    items = _extract_from_frame(limit=0)
    assert items == []


# ---------------------------------------------------------------------------
# MathError
# ---------------------------------------------------------------------------

def test_math_error_is_mixin_and_valueerror():
    exc = MathError("oops")
    assert isinstance(exc, ExceptionCauseMixin)
    assert isinstance(exc, ValueError)


def test_math_error_with_cause():
    cause = ZeroDivisionError("div")
    exc = MathError(cause, "wrapped")
    assert exc.cause is cause
    assert exc.root_cause is cause
    assert exc._get_message() == "wrapped"


# ---------------------------------------------------------------------------
# whoops_math / math_lol
# ---------------------------------------------------------------------------

def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


def test_math_lol_raises_math_error():
    with pytest.raises(MathError):
        math_lol()


def test_math_lol_raises_math_error_with_cause():
    with pytest.raises(MathError) as exc_info:
        math_lol()
    exc = exc_info.value
    assert isinstance(exc.cause, ZeroDivisionError)
    assert isinstance(exc.root_cause, ZeroDivisionError)
    assert exc._get_message() == "ya done messed up"


def test_math_lol_with_high_n():
    # n >= 3 goes straight to whoops_math
    with pytest.raises(MathError):
        math_lol(n=5)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_raises_math_error():
    with pytest.raises(MathError) as exc_info:
        main()
    exc = exc_info.value
    assert exc._get_message() == "hi"
    # The cause of the outer MathError is the inner MathError
    assert isinstance(exc.cause, MathError)
    # root_cause should be the ZeroDivisionError
    assert isinstance(exc.root_cause, ZeroDivisionError)


def test_main_raises_caught_as_valueerror():
    # MathError inherits ValueError, so it can be caught as such
    with pytest.raises(ValueError):
        main()


# ---------------------------------------------------------------------------
# Traceback capture during exception handling
# ---------------------------------------------------------------------------

def test_mixin_captures_full_trace_from_current_exception():
    try:
        raise ValueError("orig")
    except ValueError as e:
        exc = SimpleWrapped(e, "wrapped")
    assert exc.full_trace is not None
    assert len(exc.full_trace) > 0
    # Each item is a _TBItem; verify the structure rather than formatting
    for item in exc.full_trace:
        assert isinstance(item, _TBItem)


def test_mixin_copies_full_trace_from_cause():
    try:
        raise ValueError("orig")
    except ValueError as e:
        first = SimpleWrapped(e, "first")
    second = SimpleWrapped(first, "second")
    assert second.full_trace == first.full_trace
    assert second._tb == first._tb
    assert second._stack == first._stack
    # root_cause propagates to the original
    assert isinstance(second.root_cause, ValueError)


def test_mixin_multiple_wraps_without_provided_cause():
    try:
        raise ValueError("orig")
    except ValueError as e:
        original = e
        first = SimpleWrapped(original, "first")
        # Now wrap again while still in the except block (original is current exc)
        second = SimpleWrapped(first, "second")
    # second wraps first, so root propagates through cause chain
    assert second.root_cause is first.root_cause


def test_mixin_wrap_current_exception_directly():
    captured = {}
    try:
        raise ValueError("direct")
    except ValueError as e:
        wrapped = SimpleWrapped(e, "wrapped")
        captured["e"] = e
    assert wrapped.cause is captured["e"]
    assert wrapped.full_trace is not None
    assert isinstance(wrapped.cause, ValueError)
    assert str(wrapped.cause) == "direct"
