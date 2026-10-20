import sys

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
# Basic construction without cause
# ---------------------------------------------------------------------------

def test_mixin_no_cause_args():
    exc = ExceptionCauseMixin("hello")
    assert exc.args == ("hello",)
    assert exc.cause is None
    assert not hasattr(exc, "root_cause")
    assert not hasattr(exc, "full_trace")


def test_mixin_no_args():
    exc = ExceptionCauseMixin()
    assert exc.args == ()
    assert exc.cause is None


def test_mixin_first_arg_not_exception():
    exc = ExceptionCauseMixin("not-an-exception", "extra")
    assert exc.args == ("not-an-exception", "extra")
    assert exc.cause is None


# ---------------------------------------------------------------------------
# Construction with cause
# ---------------------------------------------------------------------------
# NOTE: Exception.__new__ is called with the full original args (including the
# cause) because of how the mixin's __new__ passes *args to super().__new__.

def test_mixin_with_cause():
    cause = ValueError("bad")
    exc = ExceptionCauseMixin(cause, "wrapper message")
    assert exc.cause is cause
    assert exc.root_cause is cause
    assert exc.args == (cause, "wrapper message")


def test_mixin_with_cause_no_message():
    cause = ValueError("bad")
    exc = ExceptionCauseMixin(cause)
    assert exc.cause is cause
    assert exc.root_cause is cause
    assert exc.args == (cause,)


def test_mixin_root_cause_chain():
    root = ValueError("root")
    mid = ExceptionCauseMixin(root)
    assert mid.root_cause is root
    outer = ExceptionCauseMixin(mid)
    assert outer.cause is mid
    assert outer.root_cause is root


# ---------------------------------------------------------------------------
# _get_message / _get_exc_str / get_str / __str__
# ---------------------------------------------------------------------------
# The implementation's _get_message() does "args = args[1:]" when cause is
# set, but args includes the cause at index 0, so the message remains at [0].

def test_get_message_no_cause():
    exc = ExceptionCauseMixin("msg")
    assert exc._get_message() == "msg"


def test_get_message_no_args_no_cause():
    exc = ExceptionCauseMixin()
    assert exc._get_message() == ""


def test_get_message_with_cause():
    # With cause set, args[1:] is taken; original args[0] is the cause,
    # so the remaining args[0] is the message argument.
    exc = ExceptionCauseMixin(ValueError("cause"), "outer")
    assert exc._get_message() == "outer"


def test_get_message_with_cause_no_args():
    exc = ExceptionCauseMixin(ValueError("cause"))
    # args == (cause,); args[1:] == (); no message
    assert exc._get_message() == ""


def test_get_message_with_cause_empty_first_arg():
    exc = ExceptionCauseMixin(ValueError("cause"), "")
    assert exc._get_message() == ""


def test_get_exc_str_with_message():
    exc = ExceptionCauseMixin(ValueError("cause"), "outer")
    s = exc._get_exc_str()
    assert s.startswith("ExceptionCauseMixin: ")
    assert "outer (caused by ValueError: cause)" in s


def test_get_exc_str_without_message():
    exc = ExceptionCauseMixin(ValueError("cause"))
    s = exc._get_exc_str()
    assert s == "ExceptionCauseMixin:  caused by ValueError: cause"


def test_get_exc_str_no_name():
    exc = ExceptionCauseMixin(ValueError("cause"), "outer")
    s = exc._get_exc_str(incl_name=False)
    assert s == "outer (caused by ValueError: cause)"


def test_str_no_cause():
    exc = ExceptionCauseMixin("simple")
    assert str(exc) == "simple"


def test_get_str_no_cause():
    # get_str() relies on root_cause / cause handling and raises when there
    # is no cause present (no root_cause attribute).
    exc = ExceptionCauseMixin("simple")
    with pytest.raises(AttributeError):
        exc.get_str()


# ---------------------------------------------------------------------------
# Cause captured from current exception context
# ---------------------------------------------------------------------------

def test_cause_captured_from_exc_info():
    ve = None
    try:
        raise ValueError("boom")
    except ValueError as e:
        ve = e
        wrapper = ExceptionCauseMixin(e, "wrapped")
    assert wrapper.cause is ve
    assert wrapper.root_cause is ve
    assert hasattr(wrapper, "full_trace")
    assert isinstance(wrapper.full_trace, list)
    assert len(wrapper.full_trace) > 0
    assert hasattr(wrapper, "_tb")
    assert hasattr(wrapper, "_stack")


def test_full_trace_copied_when_cause_has_full_trace():
    try:
        raise ValueError("boom")
    except ValueError as ve:
        inner = ExceptionCauseMixin(ve, "inner")
    # Now create a new wrapper outside of an exception context,
    # exercising the full_trace copy branch.
    outer = ExceptionCauseMixin(inner, "outer")
    assert outer.full_trace == inner.full_trace
    assert outer.full_trace is not inner.full_trace
    assert outer._tb == inner._tb
    assert outer._stack == inner._stack
    assert outer._tb is not inner._tb
    assert outer._stack is not inner._stack


def test_no_exc_info_returns_early():
    # Construct wrapper with an explicit cause that has no full_trace,
    # but no active exception context.
    cause = ValueError("bare")
    wrapper = ExceptionCauseMixin(cause, "msg")
    assert wrapper.cause is cause
    assert wrapper.args == (cause, "msg")
    assert not hasattr(wrapper, "full_trace")


# ---------------------------------------------------------------------------
# __repr__ / get_str with trace
# ---------------------------------------------------------------------------

def test_trace_str_no_cause_uses_repr():
    exc = ExceptionCauseMixin("x")
    assert exc._get_trace_str() == repr(exc)


def test_str_with_cause_no_trace():
    # Constructed outside an exception context: no full_trace attribute set,
    # so accessing it raises AttributeError inside __str__.
    cause = ValueError("bad")
    exc = ExceptionCauseMixin(cause, "outer")
    assert not hasattr(exc, "full_trace")
    with pytest.raises(AttributeError):
        str(exc)


def test_str_with_trace_contains_traceback_header():
    try:
        raise ValueError("boom")
    except ValueError as ve:
        wrapper = ExceptionCauseMixin(ve, "wrapped")
    # traceback.format_list on _TBItem entries whose .line is a _DeferredLine
    # (not a str) fails because the line object is not iterable as expected.
    with pytest.raises(TypeError):
        str(wrapper)


def test_get_str_with_trace():
    try:
        raise ValueError("boom")
    except ValueError as ve:
        wrapper = ExceptionCauseMixin(ve, "wrapped")
    # Same formatting failure as __str__ when a full_trace exists.
    with pytest.raises(TypeError):
        wrapper.get_str()


# ---------------------------------------------------------------------------
# get_str with a fake full_trace (string lines) exercises the trace branch
# ---------------------------------------------------------------------------

def test_get_str_with_fake_trace_lines():
    exc = ExceptionCauseMixin(ValueError("cause"), "outer")
    # Provide _TBItem-like tuples whose "line" is a plain string so that
    # traceback.format_list can render them.
    exc._tb = []
    exc._stack = []
    exc.full_trace = [
        ('fake.py', 1, 'func', 'some line\n'),
    ]
    s = exc.get_str()
    assert s.startswith('Traceback (most recent call last):\n')
    assert 'some line' in s
    assert s.rstrip().endswith(
        'outer (caused by ValueError: cause)')


def test_str_with_fake_trace_lines():
    exc = ExceptionCauseMixin(ValueError("cause"), "outer")
    exc._tb = []
    exc._stack = []
    exc.full_trace = [
        ('fake.py', 1, 'func', 'some line\n'),
    ]
    s = str(exc)
    assert 'outer --- ' in s
    assert 'Wrapped traceback (most recent call last):\n' in s
    assert 'some line' in s
    assert s.endswith('ExceptionCauseMixin: outer (caused by ValueError: cause)')


def test_str_with_empty_fake_trace_lines():
    exc = ExceptionCauseMixin(ValueError("cause"), "outer")
    exc._tb = []
    exc._stack = []
    exc.full_trace = []
    # Empty list is falsy; takes the else branch of __str__.
    s = str(exc)
    assert s == 'outer (caused by ValueError: cause)'


def test_get_str_with_empty_trace():
    exc = ExceptionCauseMixin(ValueError("cause"), "outer")
    exc._tb = []
    exc._stack = []
    exc.full_trace = []
    s = exc.get_str()
    assert s == 'ExceptionCauseMixin: outer (caused by ValueError: cause)'


def test_get_str_with_empty_trace_no_message():
    exc = ExceptionCauseMixin(ValueError("cause"))
    exc._tb = []
    exc._stack = []
    exc.full_trace = []
    s = exc.get_str()
    assert s == 'ExceptionCauseMixin:  caused by ValueError: cause'


# ---------------------------------------------------------------------------
# __str__ with a non-empty trace AND an empty message (line 150 false branch)
# ---------------------------------------------------------------------------

def test_str_with_trace_and_empty_message():
    # cause set, full_trace non-empty, but _get_message() returns '' so the
    # `if message:` branch (line 150) is False; the " --- " separator is
    # not inserted.
    exc = ExceptionCauseMixin(ValueError("cause"))
    exc._tb = []
    exc._stack = []
    exc.full_trace = [
        ('fake.py', 1, 'func', 'some line\n'),
    ]
    s = str(exc)
    assert 'some line' in s
    assert 'Wrapped traceback (most recent call last):\n' in s
    assert ' --- ' not in s
    assert not s.startswith('---')
    assert s.endswith('ExceptionCauseMixin:  caused by ValueError: cause')
    # ensure header directly precedes the trace
    assert 'Wrapped traceback (most recent call last):\n' in s


def test_str_with_trace_and_empty_message_exact_prefix():
    exc = ExceptionCauseMixin(ValueError("cause"))
    exc._tb = []
    exc._stack = []
    exc.full_trace = [
        ('fake.py', 1, 'func', 'some line\n'),
    ]
    s = str(exc)
    expected_trace = ''.join(
        __import__('traceback').format_list(exc.full_trace))
    expected = ('Wrapped traceback (most recent call last):\n'
                + expected_trace
                + 'ExceptionCauseMixin:  caused by ValueError: cause')
    assert s == expected


# ---------------------------------------------------------------------------
# cause is exc_value / root_cause is exc_value branch (line 91)
# ---------------------------------------------------------------------------

def test_cause_is_current_exception_branch():
    # Raising inside the except block with the raised exception wrapping
    # itself triggers the `cause is exc_value` branch.
    try:
        raise ValueError("self-cause")
    except ValueError as e:
        wrapper = ExceptionCauseMixin(e, "self")
        # cause is exc_value -> branch entered
        assert wrapper.cause is e
        assert wrapper.root_cause is e
    assert hasattr(wrapper, "full_trace")
    assert wrapper.full_trace


def test_root_cause_is_current_exception_branch():
    inner = ValueError("root")
    mid = ExceptionCauseMixin(inner)
    try:
        raise mid
    except ExceptionCauseMixin as e:
        outer = ExceptionCauseMixin(inner, "outer")
        assert outer.root_cause is inner
        assert outer.cause is inner
        assert outer.root_cause is e.root_cause


# ---------------------------------------------------------------------------
# _format_exc helper
# ---------------------------------------------------------------------------

def test_format_exc_default_message():
    exc = ValueError("oops")
    s = _format_exc(exc)
    assert s == "ValueError: oops"


def test_format_exc_custom_message():
    exc = ValueError("oops")
    s = _format_exc(exc, "custom")
    assert s == "ValueError: custom"


def test_format_exc_message_stripped():
    exc = RuntimeError("x")
    s = _format_exc(exc, "trailing   ")
    assert not s.endswith(" ")


# ---------------------------------------------------------------------------
# MathError example class
# ---------------------------------------------------------------------------

def test_matcherror_is_valueerror():
    assert issubclass(MathError, ValueError)
    assert issubclass(MathError, ExceptionCauseMixin)


def test_matcherror_construct():
    cause = ValueError("c")
    exc = MathError(cause, "msg")
    assert exc.cause is cause
    assert exc.args == (cause, "msg")


# ---------------------------------------------------------------------------
# whoops_math
# ---------------------------------------------------------------------------

def test_whoops_math_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        whoops_math()


# ---------------------------------------------------------------------------
# math_lol
# ---------------------------------------------------------------------------

def test_math_lol_raises_matcherror():
    with pytest.raises(MathError) as ei:
        math_lol()
    exc = ei.value
    assert isinstance(exc.cause, ZeroDivisionError)
    assert exc.args[1:] == ("ya done messed up",)
    assert isinstance(exc.args[0], ZeroDivisionError)


def test_math_lol_with_near_default():
    with pytest.raises(MathError) as ei:
        math_lol(n=2)
    assert isinstance(ei.value.cause, ZeroDivisionError)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_raises_matcherror():
    with pytest.raises(MathError) as ei:
        main()
    exc = ei.value
    assert exc.args[1:] == ("hi",)
    assert isinstance(exc.args[0], MathError)
    # cause is the inner MathError
    assert isinstance(exc.cause, MathError)
    assert isinstance(exc.cause.cause, ZeroDivisionError)


def test_main_root_cause_is_zero_division():
    with pytest.raises(MathError) as ei:
        main()
    assert isinstance(ei.value.root_cause, ZeroDivisionError)


# ---------------------------------------------------------------------------
# _extract_from_tb / _extract_from_frame basic shape
# ---------------------------------------------------------------------------

def test_extract_from_tb_empty():
    assert _extract_from_tb(None) == []


def test_extract_from_tb_structure():
    try:
        raise ValueError("boom")
    except ValueError:
        _, _, tb = sys.exc_info()
        items = _extract_from_tb(tb)
    assert len(items) >= 1
    for item in items:
        assert item.filename
        assert isinstance(item.lineno, int)
        assert item.name
        assert hasattr(item, "frame_id")


def test_extract_from_tb_limit():
    try:
        raise ValueError("boom")
    except ValueError:
        _, _, tb = sys.exc_info()
        items = _extract_from_tb(tb, limit=1)
    assert len(items) == 1


def test_extract_from_frame_structure():
    items = _extract_from_frame()
    assert len(items) >= 1
    # reversed order: outermost last frame is first in the test call
    for item in items:
        assert item.filename
        assert isinstance(item.lineno, int)
        assert item.name


def test_extract_from_frame_limit():
    items = _extract_from_frame(limit=1)
    assert len(items) == 1


# ---------------------------------------------------------------------------
# _TBItem.__repr__ (lines 172-174)
# ---------------------------------------------------------------------------

def test_tbitem_repr():
    item = _TBItem('foo.py', 10, 'func', 'the line')
    item.frame_id = 12345
    r = repr(item)
    assert 'foo.py' in r
    assert '10' in r
    assert 'func' in r
    assert 'the line' in r
    assert '12345' in r


# ---------------------------------------------------------------------------
# _DeferredLine behavior (__eq__, __ne__, __str__, __repr__, __len__, strip)
# ---------------------------------------------------------------------------

def test_deferred_line_eq_ne():
    a = _DeferredLine('some_file.py', 1)
    b = _DeferredLine('some_file.py', 1)
    c = _DeferredLine('some_file.py', 2)
    d = _DeferredLine('other.py', 1)
    assert a == b
    assert not (a != b)
    assert a != c
    assert not (a == c)
    assert a != d
    assert not (a == d)


def test_deferred_line_str_caches(tmp_path):
    f = tmp_path / "sample_mod.py"
    f.write_text("line one\nline two\n")
    dl = _DeferredLine(str(f), 2)
    s = str(dl)
    assert s == "line two"
    # _line should now be cached.
    assert hasattr(dl, '_line')
    assert dl._line == "line two"
    # Second access hits the cached branch.
    assert str(dl) == "line two"


def test_deferred_line_missing_file():
    dl = _DeferredLine('/nonexistent/path/to/file_xyz.py', 1)
    # getline returns empty, the code sets self._line to None, and since
    # __str__ returns that None value, str() raises TypeError.
    with pytest.raises(TypeError):
        str(dl)
    # The cached _line attribute is None after the failed access.
    assert dl._line is None


def test_deferred_line_repr(tmp_path):
    f = tmp_path / "repr_mod.py"
    f.write_text("hello world\n")
    dl = _DeferredLine(str(f), 1)
    r = repr(dl)
    assert r == repr("hello world")


def test_deferred_line_len(tmp_path):
    f = tmp_path / "len_mod.py"
    f.write_text("abc\n")
    dl = _DeferredLine(str(f), 1)
    assert len(dl) == 3


def test_deferred_line_strip(tmp_path):
    f = tmp_path / "strip_mod.py"
    f.write_text("   spaced   \n")
    dl = _DeferredLine(str(f), 1)
    assert dl.strip() == "spaced"


def test_deferred_line_module_globals_filtered():
    dl = _DeferredLine('x.py', 1, {
        '__name__': 'mod',
        '__loader__': 'loader',
        'other': 'ignored',
    })
    assert dl.module_globals == {'__name__': 'mod', '__loader__': 'loader'}
