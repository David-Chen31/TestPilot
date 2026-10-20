import time
import threading
from datetime import timedelta

import pytest

from funcy.flow import (
    raiser,
    ignore,
    silent,
    reraise,
    retry,
    fallback,
    ErrorRateExceeded,
    limit_error_rate,
    throttle,
    post_processing,
    collecting,
    joining,
    once,
    once_per,
    once_per_args,
    wrap_with,
)


# ---------------------------------------------------------------------------
# raiser
# ---------------------------------------------------------------------------

def test_raiser_default_is_exception():
    f = raiser()
    with pytest.raises(Exception):
        f()


def test_raiser_class():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_instance():
    exc = ValueError("boom")
    f = raiser(exc)
    with pytest.raises(ValueError) as excinfo:
        f()
    assert excinfo.value is exc


def test_raiser_string_becomes_exception():
    f = raiser("some message")
    with pytest.raises(Exception) as excinfo:
        f()
    assert str(excinfo.value) == "some message"


def test_raiser_with_args():
    f = raiser(ValueError, "a", "b")
    with pytest.raises(ValueError) as excinfo:
        excinfo
        f()
    # args are passed to the exception constructor
    assert excinfo.value.args == ("a", "b")


def test_raiser_with_kwargs():
    class MyError(Exception):
        def __init__(self, code=None):
            self.code = code

    f = raiser(MyError, code=42)
    with pytest.raises(MyError) as excinfo:
        f()
    assert excinfo.value.code == 42


def test_raiser_accepts_call_args_ignored():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f(1, 2, keyword=3)


# ---------------------------------------------------------------------------
# ignore / silent
# ---------------------------------------------------------------------------

def test_ignore_returns_default():
    @ignore(ValueError, default=42)
    def f():
        raise ValueError

    assert f() == 42


def test_ignore_returns_none_by_default():
    @ignore(ValueError)
    def f():
        raise ValueError

    assert f() is None


def test_ignore_passes_through_non_matching():
    @ignore(ValueError)
    def f():
        raise KeyError

    with pytest.raises(KeyError):
        f()


def test_ignore_returns_value_on_success():
    @ignore(ValueError, default=0)
    def f(x):
        return x * 2

    assert f(21) == 42


def test_ignore_tuple_of_errors():
    @ignore((ValueError, KeyError), default="c")
    def f(e):
        raise e("x")

    assert f(ValueError) == "c"
    assert f(KeyError) == "c"


def test_ignore_list_of_errors():
    @ignore([ValueError], default=7)
    def f():
        raise ValueError

    assert f() == 7


def test_silent_ignores_all():
    @silent
    def f():
        raise RuntimeError("nope")

    assert f() is None


def test_silent_passes_through_value():
    @silent
    def f():
        return 5

    assert f() == 5


def test_ignore_preserves_wrapped_metadata():
    @ignore(ValueError)
    def my_func():
        pass

    assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------------------
# reraise
# ---------------------------------------------------------------------------

def test_reraise_converts_exception():
    with pytest.raises(KeyError) as excinfo:
        with reraise(ValueError, KeyError):
            raise ValueError("orig")

    assert isinstance(excinfo.value.__cause__, ValueError)


def test_reraise_with_callable():
    def conv(e):
        return KeyError("converted: " + str(e))

    with pytest.raises(KeyError) as excinfo:
        with reraise(ValueError, conv):
            raise ValueError("orig")

    assert str(excinfo.value) == "converted: orig"


def test_reraise_factory_class_is_not_called():
    # When ``into`` is an exception type, it is raised directly (not called with
    # the caught exception as argument, which would be handled by callable check).
    with pytest.raises(KeyError) as excinfo:
        with reraise(ValueError, KeyError):
            raise ValueError("orig")
    assert excinfo.value.args == ()


def test_reraise_passes_through_other_errors():
    with pytest.raises(TypeError):
        with reraise(ValueError, KeyError):
            raise TypeError("other")


def test_reraise_passes_through_success():
    with reraise(ValueError, KeyError):
        result = 1 + 1
    assert result == 2


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

def test_retry_success_first_try():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        return "ok"

    assert f() == "ok"
    assert len(calls) == 1


def test_retry_eventually_succeeds():
    calls = [0]

    @retry(3)
    def f():
        calls[0] += 1
        if calls[0] < 3:
            raise ValueError
        return "done"

    assert f() == "done"
    assert calls[0] == 3


def test_retry_reraises_after_final_attempt():
    calls = [0]

    @retry(3)
    def f():
        calls[0] += 1
        raise ValueError("fail")

    with pytest.raises(ValueError):
        f()
    assert calls[0] == 3


def test_retry_only_on_specified_errors():
    calls = [0]

    @retry(3, errors=ValueError)
    def f():
        calls[0] += 1
        raise KeyError

    with pytest.raises(KeyError):
        f()
    assert calls[0] == 1


def test_retry_filter_errors_rejects():
    calls = [0]

    @retry(3, filter_errors=lambda e: False)
    def f():
        calls[0] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert calls[0] == 1


def test_retry_filter_errors_accepts():
    calls = [0]

    @retry(3, filter_errors=lambda e: True)
    def f():
        calls[0] += 1
        if calls[0] < 2:
            raise ValueError
        return "ok"

    assert f() == "ok"
    assert calls[0] == 2


def test_retry_with_timeout_callable(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    calls = [0]

    @retry(3, timeout=lambda attempt: attempt + 1)
    def f():
        calls[0] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    # sleeps happen between attempts: attempts 0 and 1 -> sleeps of 1 and 2
    assert slept == [1, 2]


def test_retry_zero_timeout_does_not_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    @retry(2, timeout=0)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert slept == []


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------

def test_fallback_first_works():
    def a():
        return "a"

    def b():
        return "b"

    assert fallback(a, b) == "a"


def test_fallback_uses_next_on_failure():
    def a():
        raise ValueError

    def b():
        return "b"

    assert fallback(a, b) == "b"


def test_fallback_with_expected_errors():
    def a():
        raise KeyError

    def b():
        return "b"

    # a catches ValueError only, so KeyError propagates
    with pytest.raises(KeyError):
        fallback((a, ValueError), b)


def test_fallback_catches_specified_then_uses_next():
    def a():
        raise ValueError

    def b():
        return "b"

    assert fallback((a, ValueError), b) == "b"


def test_fallback_all_fail_raises_from_last():
    def a():
        raise ValueError("a")

    def b():
        raise KeyError("b")

    with pytest.raises(KeyError):
        fallback(a, b)


def test_fallback_plain_callable_catches_exception():
    def a():
        raise RuntimeError

    assert fallback(a, lambda: "fallback") == "fallback"


# ---------------------------------------------------------------------------
# ErrorRateExceeded / limit_error_rate
# ---------------------------------------------------------------------------

def test_error_rate_exceeded_is_exception():
    assert issubclass(ErrorRateExceeded, Exception)


def test_limit_error_rate_success_resets():
    @limit_error_rate(2, 100)
    def f():
        return "ok"

    assert f() == "ok"
    assert f.fails == 0
    assert f.blocked is None


def test_limit_error_rate_blocks_after_fails():
    @limit_error_rate(2, timedelta(seconds=100))
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    # now blocked
    with pytest.raises(ErrorRateExceeded):
        f()


def test_limit_error_rate_custom_exception():
    class MyErr(Exception):
        pass

    @limit_error_rate(1, timedelta(seconds=100), exception=MyErr)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(MyErr):
        f()


def test_limit_error_rate_timeout_numeric_becomes_timedelta():
    @limit_error_rate(1, 0)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    # timeout is zero, so block expires immediately
    with pytest.raises(ValueError):
        f()


# ---------------------------------------------------------------------------
# throttle
# ---------------------------------------------------------------------------

def test_throttle_first_call_runs():
    @throttle(10)
    def f(x):
        return x * 2

    assert f(3) == 6


def test_throttle_blocks_within_period(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    @throttle(10)
    def f():
        return "ran"

    assert f() == "ran"
    assert f() is None
    now[0] += 5
    assert f() is None
    now[0] += 6  # total +11 since first successful
    assert f() == "ran"


def test_throttle_accepts_timedelta(monkeypatch):
    now = [50.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    @throttle(timedelta(seconds=5))
    def f():
        return "yes"

    assert f() == "yes"
    assert f() is None
    now[0] += 6
    assert f() == "yes"


# ---------------------------------------------------------------------------
# post_processing / collecting / joining
# ---------------------------------------------------------------------------

def test_post_processing():
    @post_processing(lambda x: x + 1)
    def f():
        return 41

    assert f() == 42


def test_collecting():
    @collecting
    def f():
        yield 1
        yield 2
        yield 3

    assert f() == [1, 2, 3]


def test_collecting_name_doc():
    assert collecting.__name__ == "collecting"
    assert "list" in collecting.__doc__


def test_joining():
    @joining(",")
    def f():
        return [1, 2, 3]

    assert f() == "1,2,3"


def test_joining_strings():
    @joining("-")
    def f():
        return ["a", "b", "c"]

    assert f() == "a-b-c"


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

def test_once_runs_only_once():
    calls = []

    @once
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(2) is None
    assert f(3) is None
    assert calls == [1]


# ---------------------------------------------------------------------------
# once_per / once_per_args
# ---------------------------------------------------------------------------

def test_once_per_on_given_args():
    calls = []

    @once_per("a")
    def f(a, b):
        calls.append((a, b))
        return a

    assert f(1, "x") == 1
    assert f(2, "y") == 2
    assert f(1, "z") is None
    assert calls == [(1, "x"), (2, "y")]


def test_once_per_with_kwargs():
    calls = []

    @once_per("a")
    def f(a, b=0):
        calls.append((a, b))
        return a

    assert f(a=1, b=5) == 1
    assert f(a=1, b=6) is None
    assert calls == [(1, 5)]


def test_once_per_unhashable_args_uses_list():
    calls = []

    @once_per("a")
    def f(a):
        calls.append(a)
        return a

    assert f([1, 2]) == [1, 2]
    assert f([1, 2]) is None
    assert f([1, 3]) == [1, 3]
    assert calls == [[1, 2], [1, 3]]


def test_once_per_args_all_args():
    calls = []

    @once_per_args
    def f(a, b):
        calls.append((a, b))
        return a

    assert f(1, 2) == 1
    assert f(1, 2) is None
    assert f(1, 3) == 1
    assert calls == [(1, 2), (1, 3)]


def test_once_per_args_only_varargs_func():
    calls = []

    @once_per_args
    def f(a):
        calls.append(a)
        return a

    assert f(1) == 1
    assert f(1) is None
    assert f(2) == 2
    assert calls == [1, 2]


# ---------------------------------------------------------------------------
# wrap_with
# ---------------------------------------------------------------------------

def test_wrap_with_enters_context():
    events = []

    class CM:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *a):
            events.append("exit")

    @wrap_with(CM())
    def f():
        events.append("call")
        return 99

    assert f() == 99
    assert events == ["enter", "call", "exit"]


def test_wrap_with_runs_context_even_if_call_unchanged():
    class CM:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True

        def __exit__(self, *a):
            pass

    cm = CM()

    @wrap_with(cm)
    def f():
        return "x"

    assert f() == "x"
    assert cm.entered is True
