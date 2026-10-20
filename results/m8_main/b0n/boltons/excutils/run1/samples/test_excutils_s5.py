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


# ---------- ExceptionCauseMixin basics ----------

def test_cause_default_none():
    exc = ExceptionCauseMixin()
    assert exc.cause is None
    assert exc.args == ()


def test_cause_set_with_exception_first_arg():
    inner = ValueError("inner")
    exc = ExceptionCauseMixin(inner)
    assert exc.cause is inner
    assert exc.args == ()
    assert exc.root_cause is inner


def test_cause_with_message_after_exception():
    inner = ValueError("inner")
    exc = ExceptionCauseMixin(inner, "outer message")
    assert exc.cause is inner
    assert exc.args == ("outer message",)
    assert exc.root_cause is inner


def test_string_first_arg_not_cause():
    exc = ExceptionCauseMixin("just a message")
    assert exc.cause is None
    assert exc.args == ("just a message",)


def test_no_cause_means_no_root_cause_attribute():
    exc = ExceptionCauseMixin("msg")
    # root_cause only set when cause given
    assert getattr(exc, "root_cause", None) is None


def test_root_cause_uses_existing_root_cause():
    root = ValueError("root")
    mid = ExceptionCauseMixin(root, "mid")
    assert mid.root_cause is root

    outer = ExceptionCauseMixin(mid, "outer")
    assert outer.cause is mid
    assert outer.root_cause is root


def test_full_trace_copied_from_cause():
    inner = ValueError("inner")
    # Fake a full_trace-carrying cause
    inner.full_trace = ["line1", "line2"]
    inner._tb = ["tb1"]
    inner._stack = ["stack1"]

    exc = ExceptionCauseMixin(inner, "msg")
    assert exc.full_trace == ["line1", "line2"]
    assert exc._tb == ["tb1"]
    assert exc._stack == ["stack1"]
    # copies, not same objects
    assert exc.full_trace is not inner.full_trace


def test_get_message_without_cause():
    exc = ExceptionCauseMixin("hello")
    assert exc._get_message() == "hello"


def test_get_message_with_cause_and_message():
    inner = ValueError("inner")
    exc = ExceptionCauseMixin(inner, "hello")
    assert exc._get_message() == "hello"


def test_get_message_with_cause_no_message():
    inner = ValueError("inner")
    exc = ExceptionCauseMixin(inner)
    assert exc._get_message() == ""


def test_get_message_empty_args():
    exc = ExceptionCauseMixin()
    assert exc._get_message() == ""


# ---------- str / repr behavior ----------

def test_str_no_cause():
    exc = ExceptionCauseMixin("some message")
    assert str(exc) == "some message"


def test_str_with_cause_no_trace():
    inner = ValueError("inner")
    exc = ExceptionCauseMixin(inner, "outer")
    s = str(exc)
    # no traceback info so _get_exc_str(incl_name=False)
    assert s.startswith(" caused by ")
    assert "ValueError: inner" in s


def test_str_with_cause_no_message():
    inner = ValueError("inner")
    exc = ExceptionCauseMixin(inner)
    s = str(exc)
    assert s.startswith(" caused by ")
    assert "ValueError: inner" in s


def test_get_str_includes_traceback_header_when_full_trace():
    inner = ValueError("inner")
    inner.full_trace = [("file.py", 1, "func", "line")]
    exc = ExceptionCauseMixin(inner, "outer")
    s = exc.get_str()
    assert s.startswith("Traceback (most recent call last):\n")
    assert s.endswith("ExceptionCauseMixin: outer (caused by ValueError: inner)")


def test_get_str_no_trace_only_exc_line():
    inner = ValueError("inner")
    exc = ExceptionCauseMixin(inner, "outer")
    s = exc.get_str()
    assert "Traceback" not in s
    assert s == "ExceptionCauseMixin: outer (caused by ValueError: inner)"


def test_get_exc_str_without_name():
    inner = ValueError("inner")
    exc = ExceptionCauseMixin(inner, "outer")
    assert exc._get_exc_str(incl_name=False) == "outer (caused by ValueError: inner)"


def test_get_exc_str_without_message():
    inner = ValueError("inner")
    exc = ExceptionCauseMixin(inner)
    assert exc._get_exc_str() == "ExceptionCauseMixin:  caused by ValueError: inner"


def test_format_exc_helper():
    exc = ValueError("boom")
    assert _format_exc(exc) == "ValueError: boom"


def test_format_exc_custom_message():
    exc = ValueError("boom")
    assert _format_exc(exc, "custom") == "ValueError: custom"


def test_format_exc_strips_newline():
    exc = ValueError("boom")
    out = _format_exc(exc, "msg\n")
    assert not out.endswith("\n")


# ---------- _DeferredLine ----------

def test_deferred_line_attributes():
    dl = _DeferredLine("nonexistent_file_xyz.py", 10)
    assert dl.filename == "nonexistent_file_xyz.py"
    assert dl.lineno == 10


def test_deferred_line_module_globals_filtering():
    dl = _DeferredLine("f.py", 1, {"__name__": "m", "__loader__": "l", "other": 1})
    assert dl.module_globals == {"__name__": "m", "__loader__": "l"}


def test_deferred_line_eq_ne():
    a = _DeferredLine("f.py", 1)
    b = _DeferredLine("f.py", 1)
    c = _DeferredLine("f.py", 2)
    assert a == b
    assert a != c
    assert not (a != b)
    assert not (a == c)


def test_deferred_line_str_caches():
    dl = _DeferredLine("nonexistent_file_xyz.py", 1)
    s1 = str(dl)
    assert s1 is None  # no such file, getline returns ''
    assert hasattr(dl, "_line")
    s2 = str(dl)
    assert s2 is None


def test_deferred_line_repr_and_len():
    dl = _DeferredLine("nonexistent_file_xyz.py", 1)
    # repr of None is "None"
    assert repr(dl) == repr(str(dl))
    assert len(dl) == len(str(dl))


def test_deferred_line_with_real_source_line():
    dl = _DeferredLine(__file__, 1)
    s = str(dl)
    assert s is not None
    assert "import" in s


def test_deferred_line_strip():
    dl = _DeferredLine(__file__, 1)
    assert dl.strip() == str(dl).strip()


# ---------- frame / tb extraction ----------

def test_extract_from_frame_returns_list():
    frame = sys._getframe()
    items = _extract_from_frame(frame)
    assert isinstance(items, list)
    assert len(items) >= 1
    assert all(isinstance(it, _TBItem) for it in items)


def test_extract_from_frame_order_outer_to_inner():
    frame = sys._getframe()
    items = _extract_from_frame(frame)
    # last item should be the innermost (this frame's function)
    assert items[-1].name == "test_extract_from_frame_order_outer_to_inner"


def test_extract_from_frame_limit():
    frame = sys._getframe()
    items = _extract_from_frame(frame, limit=1)
    assert len(items) == 1


def test_extract_from_frame_item_fields():
    frame = sys._getframe()
    items = _extract_from_frame(frame)
    last = items[-1]
    assert last.filename == __file__
    assert last.name == "test_extract_from_frame_item_fields"
    assert isinstance(last.lineno, int)
    assert hasattr(last, "frame_id")
    assert last.frame_id == id(frame)


def test_extract_from_frame_default_frame():
    items = _extract_from_frame()
    assert len(items) >= 1


def test_extract_from_tb_returns_list():
    try:
        raise ValueError("boom")
    except ValueError:
        tb = sys.exc_info()[2]
        items = _extract_from_tb(tb)
    assert isinstance(items, list)
    assert len(items) >= 1
    assert all(isinstance(it, _TBItem) for it in items)


def test_extract_from_tb_limit():
    try:
        raise ValueError("boom")
    except ValueError:
        tb = sys.exc_info()[2]
        items = _extract_from_tb(tb, limit=1)
    assert len(items) == 1


def test_extract_from_tb_fields():
    try:
        raise ValueError("boom")
    except ValueError:
        tb = sys.exc_info()[2]
        items = _extract_from_tb(tb)
    item = items[0]
    assert item.name == "test_extract_from_tb_fields"
    assert item.filename == __file__
    assert item.frame_id == id(tb.tb_frame)


def test_tbitem_repr_contains_frame_id():
    item = _TBItem("f.py", 1, "foo", "line")
    item.frame_id = 1234
    r = repr(item)
    assert "1234" in r
    assert "foo" in r


# ---------- MathError / whoops_math / math_lol / main ----------

def test_matcherror_is_exception_cause_mixin_and_valueerror():
    assert issubclass(MathError, ExceptionCauseMixin)
    assert issubclass(MathError, ValueError)


def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


def test_math_lol_raises_matcherror_with_zde_cause():
    with pytest.raises(MathError) as ei:
        math_lol()
    exc = ei.value
    assert isinstance(exc.cause, ZeroDivisionError)
    assert exc.args == ("ya done messed up",)
    assert exc.root_cause is exc.cause
    assert exc._get_message() == "ya done messed up"


def test_math_lol_mutually_recursive_depth():
    # calling with n>=3 goes straight to failure
    with pytest.raises(MathError):
        math_lol(n=3)


def test_main_raises_wrapped_error():
    with pytest.raises(MathError) as ei:
        main()
    exc = ei.value
    inner = exc.cause
    # inner is the MathError raised inside math_lol
    assert isinstance(inner, MathError)
    assert isinstance(inner.cause, ZeroDivisionError)
    assert exc.args == ("hi",)
    # root cause propagated through wrapping chain
    assert isinstance(exc.root_cause, ZeroDivisionError)


def test_main_outer_str_mentions_message_and_root():
    with pytest.raises(MathError) as ei:
        main()
    s = str(ei.value)
    assert "hi" in s
    assert "ZeroDivisionError" in s
