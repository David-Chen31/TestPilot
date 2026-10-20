import time
import threading
from datetime import datetime, timedelta

import pytest

from funcy.flow import (
    raiser, ignore, silent, reraise, retry, fallback,
    ErrorRateExceeded, limit_error_rate, throttle,
    post_processing, collecting, joining,
    once, once_per, once_per_args,
    wrap_with,
)


# ---------------------- raiser ----------------------

def test_raiser_raises_exception_instance():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_with_args_and_kwargs():
    f = raiser(ValueError, "boom", code=3)
    with pytest.raises(ValueError) as ei:
        f()
    assert ei.value.args == ("boom",)
    assert ei.value.code == 3


def test_raiser_raises_specific_instance():
    exc = ValueError("instance")
    f = raiser(exc)
    with pytest.raises(ValueError) as ei:
        f()
    assert ei.value is exc


def test_raiser_string_becomes_exception_message():
    f = raiser("boom")
    with pytest.raises(Exception) as ei:
        f()
    assert str(ei.value) == "boom"


def test_raiser_default_exception():
    f = raiser()
    with pytest.raises(Exception):
        f()


# ---------------------- ignore / silent ----------------------

def test_ignore_returns_value_when_no_error():
    @ignore(Exception, default="d")
    def f(x):
        return x * 2
    assert f(3) == 6


def test_ignore_returns_default_on_error():
    @ignore(ZeroDivisionError, default=-1)
    def f():
        return 1 / 0
    assert f() == -1


def test_ignore_default_is_none():
    @ignore(ValueError)
    def f():
        raise ValueError
    assert f() is None


def test_ignore_only_catches_listed_errors():
    @ignore(ValueError)
    def f():
        raise TypeError
    with pytest.raises(TypeError):
        f()


def test_ignore_accepts_tuple():
    @ignore((ValueError, TypeError), default=0)
    def f(e):
        raise e
    assert f(ValueError) == 0
    assert f(TypeError) == 0


def test_silent_swallows_all_exceptions():
    @silent
    def f():
        raise ZeroDivisionError
    assert f() is None


def test_silent_preserves_name():
    @silent
    def my_func():
        return 1
    assert my_func.__name__ == "my_func"
    assert my_func() == 1


# ---------------------- reraise ----------------------

def test_reraise_reraises_as_other_exception():
    with pytest.raises(RuntimeError) as ei:
        with reraise(ValueError, RuntimeError):
            raise ValueError("orig")
    assert isinstance(ei.value.__cause__, ValueError)


def test_reraise_callable_into():
    def make(e):
        return RuntimeError(str(e))
    with pytest.raises(RuntimeError) as ei:
        with reraise(ValueError, make):
            raise ValueError("orig")
    assert str(ei.value) == "orig"


def test_reraise_passes_through_unmatched():
    with pytest.raises(TypeError):
        with reraise(ValueError, RuntimeError):
            raise TypeError


def test_reraise_no_error_ok():
    with reraise(ValueError, RuntimeError):
        x = 1
    assert x == 1


# ---------------------- retry ----------------------

def test_retry_succeeds_first_time():
    calls = []

    @retry(tries=3)
    def f():
        calls.append(1)
        return 42
    assert f() == 42
    assert len(calls) == 1


def test_retry_retries_and_succeeds():
    state = {"n": 0}

    @retry(tries=3, timeout=0)
    def f():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError
        return "ok"
    assert f() == "ok"
    assert state["n"] == 3


def test_retry_reraises_after_last_attempt():
    state = {"n": 0}

    @retry(tries=2)
    def f():
        state["n"] += 1
        raise ValueError("x")
    with pytest.raises(ValueError):
        f()
    assert state["n"] == 2


def test_retry_only_on_specified_errors():
    state = {"n": 0}

    @retry(tries=3, errors=ValueError)
    def f():
        state["n"] += 1
        raise TypeError
    with pytest.raises(TypeError):
        f()
    assert state["n"] == 1


def test_retry_filter_errors_stops_retry():
    state = {"n": 0}

    @retry(tries=3, filter_errors=lambda e: False)
    def f():
        state["n"] += 1
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert state["n"] == 1


def test_retry_timeout_callable_uses_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    state = {"n": 0}

    @retry(tries=3, timeout=lambda attempt: attempt + 1)
    def f():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError
        return "done"
    assert f() == "done"
    assert sleeps == [1, 2]


def test_retry_timeout_zero_no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    @retry(tries=2, timeout=0)
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert sleeps == []


# ---------------------- fallback ----------------------

def test_fallback_first_works():
    assert fallback(lambda: 1, lambda: 2) == 1


def test_fallback_skips_failing():
    def bad():
        raise ValueError
    assert fallback(bad, lambda: 2) == 2


def test_fallback_with_expected_errors_tuple():
    def bad():
        raise ValueError

    def good():
        return "ok"
    assert fallback((bad, ValueError), good) == "ok"


def test_fallback_raises_when_all_fail():
    def bad1():
        raise ValueError

    def bad2():
        raise TypeError
    with pytest.raises(TypeError):
        fallback(bad1, bad2)


def test_fallback_only_catches_expected():
    def bad():
        raise TypeError

    with pytest.raises(TypeError):
        fallback((bad, ValueError), lambda: 2)


# ---------------------- limit_error_rate ----------------------

def test_limit_error_rate_raises_after_fails():
    state = {"n": 0}

    @limit_error_rate(fails=2, timeout=10)
    def f():
        state["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    # Now blocked
    with pytest.raises(ErrorRateExceeded):
        f()
    assert state["n"] == 2


def test_limit_error_rate_resets_on_success():
    state = {"n": 0}
    fail = {"v": True}

    @limit_error_rate(fails=2, timeout=10)
    def f():
        state["n"] += 1
        if fail["v"]:
            raise ValueError
        return "ok"

    with pytest.raises(ValueError):
        f()
    fail["v"] = False
    assert f() == "ok"
    assert f.__wrapped__ if hasattr(f, "__wrapped__") else True
    # fails should have reset; simulate new failures
    fail["v"] = True
    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()


def test_limit_error_rate_custom_exception():
    @limit_error_rate(fails=1, timeout=10, exception=RuntimeError)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(RuntimeError):
        f()


def test_limit_error_rate_unblocks_after_timeout(monkeypatch):
    state = {"n": 0}

    @limit_error_rate(fails=1, timeout=timedelta(seconds=1))
    def f():
        state["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()

    # Manipulate blocked time into the past to simulate timeout expiry
    past = datetime.now() - timedelta(seconds=5)
    f.blocked = past
    with pytest.raises(ValueError):
        f()
    assert state["n"] == 2


# ---------------------- throttle ----------------------

def test_throttle_allows_first_then_blocks(monkeypatch):
    times = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    calls = []

    @throttle(5)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(2) is None
    assert f(3) is None
    assert calls == [1]


def test_throttle_allows_after_period(monkeypatch):
    times = iter([100.0, 106.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    calls = []

    @throttle(5)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(2) == 2
    assert calls == [1, 2]


def test_throttle_accepts_timedelta(monkeypatch):
    times = iter([100.0, 102.0, 106.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    calls = []

    @throttle(timedelta(seconds=5))
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(2) is None
    assert f(3) == 3
    assert calls == [1, 3]


# ---------------------- post_processing / collecting / joining ----------------------

def test_post_processing_applies_func():
    @post_processing(lambda r: r * 2)
    def f():
        return 21
    assert f() == 42


def test_collecting():
    @collecting
    def gen():
        yield 1
        yield 2
    assert gen() == [1, 2]
    assert collecting.__name__ == "collecting"


def test_joining_with_sep():
    @joining(", ")
    def gen():
        return [1, 2, 3]
    assert gen() == "1, 2, 3"


def test_joining_casts_via_sep_class():
    @joining("-")
    def gen():
        return [1, 2, 3]
    assert gen() == "1-2-3"


def test_joining_strings():
    @joining("")
    def gen():
        return ["a", "b", "c"]
    assert gen() == "abc"


# ---------------------- once / once_per / once_per_args ----------------------

def test_once_calls_only_first_time():
    calls = []

    @once
    def f():
        calls.append(1)
        return "x"
    assert f() == "x"
    assert f() is None
    assert f() is None
    assert calls == [1]


def test_once_per_uses_named_args():
    calls = []

    @once_per("a")
    def f(a, b, c=0):
        calls.append((a, b, c))
        return a
    assert f(1, 2) == 1
    assert f(1, 3) is None
    assert f(1, 4, c=9) is None
    assert f(2, 2) == 2
    assert calls == [(1, 2, 0), (2, 2, 0)]


def test_once_per_kwargs():
    calls = []

    @once_per("a")
    def f(a, b):
        calls.append((a, b))
        return a
    assert f(a=1, b=2) == 1
    assert f(a=1, b=3) is None
    assert f(a=2, b=3) == 2
    assert calls == [(1, 2), (2, 3)]


def test_once_per_unhashable_args_uses_list():
    calls = []

    @once_per("a")
    def f(a):
        calls.append(a)
        return a
    assert f([1, 2]) == [1, 2]
    assert f([1, 2]) is None
    assert f([3]) == [3]
    assert calls == [[1, 2], [3]]


def test_once_per_missing_arg():
    @once_per("a")
    def f(b):
        return b
    with pytest.raises(TypeError):
        f(1)


def test_once_per_args_uses_all_args():
    calls = []

    @once_per_args
    def f(a, b):
        calls.append((a, b))
        return a + b
    assert f(1, 2) == 3
    assert f(1, 2) is None
    assert f(1, 3) == 4
    assert calls == [(1, 2), (1, 3)]


def test_once_per_args_method_style(monkeypatch):
    calls = []

    class C:
        @once_per_args
        def m(self, a):
            calls.append(a)
            return a

    c = C()
    assert c.m(1) == 1
    assert c.m(1) is None
    assert c.m(2) == 2
    assert calls == [1, 2]


# ---------------------- wrap_with ----------------------

def test_wrap_with_provides_context():
    entered = []
    exited = []

    class CM:
        def __enter__(self):
            entered.append(1)
            return "value"

        def __exit__(self, *a):
            exited.append(1)
            return False

    @wrap_with(CM())
    def f():
        return "result"

    assert f() == "result"
    assert entered == [1]
    assert exited == [1]


def test_wrap_with_propagates_exception():
    class CM:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    @wrap_with(CM())
    def f():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        f()
