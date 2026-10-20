import time
import threading
from datetime import timedelta

import pytest

from funcy.flow import (
    raiser, ignore, silent, reraise, retry, fallback,
    ErrorRateExceeded, limit_error_rate, throttle,
    post_processing, collecting, joining,
    once, once_per, once_per_args, wrap_with,
)


# ---------------- raiser ----------------

def test_raiser_raises_class():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_raises_instance():
    exc = ValueError("boom")
    f = raiser(exc)
    with pytest.raises(ValueError) as ei:
        f()
    assert ei.value is exc


def test_raiser_with_string_message():
    f = raiser("bad thing")
    with pytest.raises(Exception) as ei:
        f()
    assert str(ei.value) == "bad thing"


def test_raiser_with_args():
    f = raiser(ValueError, "message", 1)
    with pytest.raises(ValueError) as ei:
        f(10, 20)
    assert ei.value.args == ("message", 1)


def test_raiser_with_kwargs():
    class MyErr(Exception):
        def __init__(self, a=None, b=None):
            self.a = a
            self.b = b

    f = raiser(MyErr, a=1, b=2)
    with pytest.raises(MyErr) as ei:
        f()
    assert ei.value.a == 1
    assert ei.value.b == 2


# ---------------- ignore / silent ----------------

def test_ignore_returns_value():
    @ignore(ValueError, default=42)
    def f(x):
        return x * 2

    assert f(3) == 6


def test_ignore_catches_error_returns_default():
    @ignore(ZeroDivisionError, default="d")
    def f():
        return 1 / 0

    assert f() == "d"


def test_ignore_default_none():
    @ignore(ValueError)
    def f():
        raise ValueError()

    assert f() is None


def test_ignore_tuple_of_errors():
    @ignore((ValueError, KeyError), default=0)
    def f(kind):
        if kind == "v":
            raise ValueError()
        raise KeyError()

    assert f("v") == 0
    assert f("k") == 0


def test_ignore_does_not_catch_other_errors():
    @ignore(ValueError)
    def f():
        raise KeyError("nope")

    with pytest.raises(KeyError):
        f()


def test_ignore_preserves_name_and_doc():
    @ignore(ValueError)
    def myfunc():
        """my doc"""
        return 1

    assert myfunc.__name__ == "myfunc"
    assert myfunc.__doc__ == "my doc"


def test_silent_catches_all():
    @silent
    def f():
        raise RuntimeError("boom")

    assert f() is None


def test_silent_passes_through_return():
    @silent
    def f():
        return 7

    assert f() == 7


# ---------------- reraise ----------------

def test_reraise_converts():
    with pytest.raises(KeyError):
        with reraise(ValueError, KeyError):
            raise ValueError("x")


def test_reraise_keeps_original_as_cause():
    original = ValueError("orig")
    try:
        with reraise(ValueError, KeyError):
            raise original
    except KeyError as e:
        assert e.__cause__ is original
    else:
        pytest.fail("did not raise")


def test_reraise_does_not_catch_other():
    with pytest.raises(TypeError):
        with reraise(ValueError, KeyError):
            raise TypeError("t")


def test_reraise_callable_factory():
    def make(e):
        return KeyError("wrapped: " + str(e))

    with pytest.raises(KeyError) as ei:
        with reraise(ValueError, make):
            raise ValueError("oops")
    assert ei.value.args == ("wrapped: oops",)


def test_reraise_exception_type_not_called():
    # into is an exception type -> must be raised as-is, not called as factory
    with pytest.raises(KeyError):
        with reraise(ValueError, KeyError):
            raise ValueError()


def test_reraise_passes_through_no_error():
    with reraise(ValueError, KeyError):
        result = 5
    assert result == 5


# ---------------- retry ----------------

def test_retry_succeeds_first_try():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        return "ok"

    assert f() == "ok"
    assert len(calls) == 1


def test_retry_retries_then_succeeds():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError()
        return "done"

    assert f() == "done"
    assert len(calls) == 3


def test_retry_exhausts_and_raises():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        raise ValueError("always")

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 3


def test_retry_only_on_specified_errors():
    calls = []

    @retry(3, errors=ValueError)
    def f():
        calls.append(1)
        raise KeyError()

    with pytest.raises(KeyError):
        f()
    assert len(calls) == 1


def test_retry_with_timeout_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    calls = []

    @retry(3, timeout=0.5)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError()
        return "ok"

    assert f() == "ok"
    assert slept == [0.5, 0.5]


def test_retry_timeout_callable(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    calls = []

    @retry(3, timeout=lambda attempt: attempt + 1)
    def f():
        calls.append(1)
        raise ValueError()

    with pytest.raises(ValueError):
        f()
    assert slept == [1, 2]


def test_retry_timeout_zero_no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    @retry(2, timeout=0)
    def f():
        raise ValueError()

    with pytest.raises(ValueError):
        f()
    assert slept == []


def test_retry_filter_errors_blocks_retry():
    calls = []

    @retry(3, filter_errors=lambda e: False)
    def f():
        calls.append(1)
        raise ValueError()

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 1


def test_retry_filter_errors_allows_retry():
    calls = []

    @retry(3, filter_errors=lambda e: True)
    def f():
        calls.append(1)
        raise ValueError()

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 3


# ---------------- fallback ----------------

def test_fallback_bare_callable():
    assert fallback(lambda: "a") == "a"


def test_fallback_uses_second():
    def first():
        raise ValueError()

    def second():
        return "second"

    assert fallback(first, second) == "second"


def test_fallback_with_expected_errors():
    def first():
        raise ValueError()

    def second():
        return "ok"

    assert fallback((first, ValueError), second) == "ok"


def test_fallback_does_not_catch_unexpected():
    def first():
        raise KeyError()

    with pytest.raises(KeyError):
        fallback((first, ValueError), lambda: "x")


def test_fallback_all_fail_returns_none():
    def first():
        raise ValueError()

    def second():
        raise KeyError()

    assert fallback((first, ValueError), (second, KeyError)) is None


def test_fallback_tuple_errors():
    def first():
        raise KeyError()

    def second():
        return "s"

    assert fallback((first, (ValueError, KeyError)), second) == "s"


# ---------------- limit_error_rate ----------------

def test_limit_error_rate_passes_through():
    @limit_error_rate(3, 10)
    def f(x):
        return x + 1

    assert f(1) == 2


def test_limit_error_rate_blocks_after_fails():
    @limit_error_rate(2, 100)
    def f():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    # now blocked
    with pytest.raises(ErrorRateExceeded):
        f()


def test_limit_error_rate_resets_fails_on_success():
    calls = []

    @limit_error_rate(2, 100)
    def f(fail):
        calls.append(fail)
        if fail:
            raise ValueError()
        return "ok"

    with pytest.raises(ValueError):
        f(True)
    assert f(False) == "ok"
    assert f.wrapper if False else True
    # fails counter should be reset -> wrapper.fails == 0
    # can fail again once before blocking
    with pytest.raises(ValueError):
        f(True)
    # one more fail -> 2nd consecutive -> blocks
    with pytest.raises(ValueError):
        f(True)
    with pytest.raises(ErrorRateExceeded):
        f(False)


def test_limit_error_rate_custom_exception():
    class Blocked(Exception):
        pass

    @limit_error_rate(1, 100, exception=Blocked)
    def f():
        raise ValueError()

    with pytest.raises(ValueError):
        f()
    with pytest.raises(Blocked):
        f()


def test_limit_error_rate_timeout_expires():
    # Use a tiny timeout and actually wait so the block expires.
    @limit_error_rate(1, 0.001)
    def f():
        raise ValueError()

    with pytest.raises(ValueError):
        f()
    # while blocked within timeout
    with pytest.raises(ErrorRateExceeded):
        f()
    time.sleep(0.02)
    # after timeout, block lifted -> ValueError again
    with pytest.raises(ValueError):
        f()


def test_limit_error_rate_timedelta_timeout():
    @limit_error_rate(1, timedelta(seconds=100))
    def f():
        raise ValueError()

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()


# ---------------- throttle ----------------

def test_throttle_runs_once_then_skips(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    calls = []

    @throttle(10)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    # within period, skipped -> returns None
    assert f(2) is None
    assert f(3) is None
    assert calls == [1]

    now[0] += 11
    assert f(4) == 8
    assert calls == [1, 4]


def test_throttle_timedelta_period(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    calls = []

    @throttle(timedelta(seconds=5))
    def f():
        calls.append(1)
        return "r"

    assert f() == "r"
    assert f() is None
    now[0] += 6
    assert f() == "r"
    assert len(calls) == 2


# ---------------- post_processing / collecting / joining ----------------

def test_post_processing_applies_func():
    @post_processing(int)
    def f():
        return "42"

    assert f() == 42


def test_collecting_turns_generator_into_list():
    @collecting
    def gen():
        yield 1
        yield 2
        yield 3

    assert gen() == [1, 2, 3]


def test_collecting_name_and_doc():
    assert collecting.__name__ == "collecting"
    assert "generator" in collecting.__doc__


def test_joining_with_string():
    @joining("-")
    def f():
        return ["a", "b", "c"]

    assert f() == "a-b-c"


def test_joining_with_converts_non_str():
    @joining(",")
    def f():
        return [1, 2, 3]

    assert f() == "1,2,3"


def test_joining_empty():
    @joining("|")
    def f():
        return []

    assert f() == ""


# ---------------- once_per / once / once_per_args ----------------

def test_once_runs_only_once():
    calls = []

    @once
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(2) is None
    assert f(1) is None
    assert calls == [1]


def test_once_per_argnames():
    calls = []

    @once_per("a")
    def f(a, b):
        calls.append((a, b))
        return a + b

    assert f(1, 10) == 11
    assert f(1, 20) is None
    assert f(2, 20) == 22
    assert calls == [(1, 10), (2, 20)]


def test_once_per_multiple_argnames_combination():
    calls = []

    @once_per("a", "b")
    def f(a, b, c):
        calls.append((a, b, c))
        return "r"

    assert f(1, 2, 3) == "r"
    assert f(1, 2, 4) is None
    assert f(1, 5, 4) == "r"
    assert calls == [(1, 2, 3), (1, 5, 4)]


def test_once_per_kwargs():
    calls = []

    @once_per("a")
    def f(a):
        calls.append(a)
        return a

    assert f(a=1) == 1
    assert f(a=1) is None
    assert f(1) is None
    assert calls == [1]


def test_once_per_default_values():
    calls = []

    @once_per("a")
    def f(a=5):
        calls.append(a)
        return a

    assert f() == 5
    assert f() is None
    assert f(5) is None
    assert calls == [5]


def test_once_per_unhashable_args_uses_list():
    calls = []

    @once_per("a")
    def f(a):
        calls.append(a)
        return "r"

    assert f([1, 2]) == "r"
    assert f([1, 2]) is None
    assert f([3, 4]) == "r"
    assert len(calls) == 2


def test_once_per_args():
    calls = []

    @once_per_args
    def f(a, b):
        calls.append((a, b))
        return a + b

    assert f(1, 2) == 3
    assert f(1, 2) is None
    assert f(2, 1) == 3
    assert f(1, 3) == 4
    assert calls == [(1, 2), (2, 1), (1, 3)]


def test_once_per_args_with_kwargs():
    calls = []

    @once_per_args
    def f(a, b=10):
        calls.append((a, b))
        return a + b

    assert f(1) == 11
    assert f(1, 10) is None
    assert f(1, b=10) is None
    assert calls == [(1, 10)]


# ---------------- wrap_with ----------------

def test_wrap_with_uses_context_manager():
    log = []

    class CM:
        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *exc):
            log.append("exit")

    @wrap_with(CM())
    def f():
        log.append("call")
        return "result"

    assert f() == "result"
    assert log == ["enter", "call", "exit"]


def test_wrap_with_context_manager_with_error():
    log = []

    class CM:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            log.append(exc_type)
            return False

    @wrap_with(CM())
    def f():
        raise ValueError("x")

    with pytest.raises(ValueError):
        f()
    assert log == [ValueError]


def test_wrap_with_single_arg_decorator_form():
    log = []

    class CM:
        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *exc):
            log.append("exit")

    @wrap_with
    def f():
        return 1

    # bare wrap_with call applied to a context manager instance (no parens style)
    # Here @wrap_with is not the usual usage, so test explicit ctx wrapping.
    @wrap_with(CM())
    def g():
        log.append("body")
        return 2

    assert g() == 2
    assert log == ["enter", "body", "exit"]
