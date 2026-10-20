import time
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager

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
    once_per,
    once_per_args,
    wrap_with,
    nullcontext,
    suppress,
)


# ---------------------------------------------------------------------------
# raiser
# ---------------------------------------------------------------------------

def test_raiser_raises_default_exception():
    f = raiser()
    with pytest.raises(Exception):
        f()


def test_raiser_raises_given_class():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_raises_instance():
    f = raiser(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        f()


def test_raiser_raises_with_args():
    f = raiser(ValueError, "custom message")
    with pytest.raises(ValueError, match="custom message"):
        f()


def test_raiser_raises_with_kwargs():
    class MyErr(Exception):
        def __init__(self, x):
            super().__init__(x)
            self.x = x

    f = raiser(MyErr, x=42)
    with pytest.raises(MyErr) as ei:
        f()
    assert ei.value.x == 42


def test_raiser_from_string_message():
    f = raiser("a message")
    with pytest.raises(Exception, match="a message"):
        f()


def test_raiser_ignores_call_args():
    f = raiser(ValueError, "fixed")
    with pytest.raises(ValueError, match="fixed"):
        f(1, 2, 3, foo="bar")


# ---------------------------------------------------------------------------
# ignore / silent
# ---------------------------------------------------------------------------

def test_ignore_returns_value_when_no_error():
    @ignore(ValueError)
    def f(x):
        return x * 2

    assert f(5) == 10


def test_ignore_returns_default_on_error():
    @ignore(ValueError)
    def f():
        raise ValueError("nope")

    assert f() is None


def test_ignore_custom_default():
    @ignore(ValueError, default=99)
    def f():
        raise ValueError

    assert f() == 99


def test_ignore_only_catches_specified():
    @ignore(ValueError)
    def f():
        raise TypeError("not ignored")

    with pytest.raises(TypeError):
        f()


def test_ignore_accepts_tuple():
    @ignore((ValueError, KeyError), default="x")
    def f():
        raise KeyError

    assert f() == "x"


def test_ignore_accepts_list():
    @ignore([ValueError, KeyError], default="y")
    def f():
        raise ValueError

    assert f() == "y"


def test_ignore_preserves_name():
    @ignore(ValueError)
    def my_func():
        return 1

    assert my_func.__name__ == "my_func"


def test_silent_catches_everything():
    @silent
    def f():
        raise RuntimeError("anything")

    assert f() is None


def test_silent_returns_value():
    @silent
    def f():
        return 7

    assert f() == 7


# ---------------------------------------------------------------------------
# reraise
# ---------------------------------------------------------------------------

def test_reraise_converts():
    with pytest.raises(KeyError) as ei:
        with reraise(ValueError, KeyError):
            raise ValueError("orig")

    assert isinstance(ei.value.__cause__, ValueError)


def test_reraise_passes_through_when_no_error():
    with reraise(ValueError, KeyError):
        x = 1
    assert x == 1


def test_reraise_does_not_catch_other():
    with pytest.raises(TypeError):
        with reraise(ValueError, KeyError):
            raise TypeError("other")


def test_reraise_callable_into():
    def make(e):
        return TypeError(str(e))

    with pytest.raises(TypeError) as ei:
        with reraise(ValueError, make):
            raise ValueError("msg")

    assert isinstance(ei.value.__cause__, ValueError)


def test_reraise_accepts_tuple():
    with pytest.raises(RuntimeError):
        with reraise((ValueError, KeyError), RuntimeError):
            raise KeyError("k")


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


def test_retry_succeeds_after_failures():
    attempts = []

    @retry(5)
    def f():
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError
        return "done"

    assert f() == "done"
    assert len(attempts) == 3


def test_retry_exhausts_and_raises():
    attempts = []

    @retry(3)
    def f():
        attempts.append(1)
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        f()
    assert len(attempts) == 3


def test_retry_default_errors_is_exception():
    attempts = []

    @retry(2)
    def f():
        attempts.append(1)
        raise RuntimeError

    with pytest.raises(RuntimeError):
        f()
    assert len(attempts) == 2


def test_retry_non_matching_error_not_retried():
    attempts = []

    @retry(5, errors=ValueError)
    def f():
        attempts.append(1)
        raise TypeError("nope")

    with pytest.raises(TypeError):
        f()
    assert len(attempts) == 1


def test_retry_filter_errors_rejects():
    attempts = []

    @retry(5, errors=ValueError, filter_errors=lambda e: False)
    def f():
        attempts.append(1)
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert len(attempts) == 1


def test_retry_filter_errors_accepts():
    attempts = []

    @retry(3, errors=ValueError, filter_errors=lambda e: True)
    def f():
        attempts.append(1)
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert len(attempts) == 3


def test_retry_timeout_sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    @retry(3, timeout=0.01)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    # sleeps between attempts only, i.e. tries-1 times
    assert slept == [0.01, 0.01]


def test_retry_timeout_callable(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    @retry(3, timeout=lambda attempt: attempt + 1)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert slept == [1, 2]


def test_retry_zero_timeout_no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    @retry(2)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert slept == []


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------

def test_fallback_first_succeeds():
    assert fallback(lambda: 1) == 1


def test_fallback_returns_first_success():
    def a():
        raise ValueError

    def b():
        return 2

    def c():
        return 3

    assert fallback(a, b, c) == 2


def test_fallback_with_expected_errors():
    def a():
        raise ValueError

    def b():
        raise KeyError

    def c():
        return "c"

    result = fallback((a, ValueError), (b, KeyError), c)
    assert result == "c"


def test_fallback_all_fail_returns_none():
    def a():
        raise ValueError

    assert fallback(a) is None


def test_fallback_only_catches_specified_errors():
    def a():
        raise ValueError

    def b():
        return 2

    result = fallback((a, ValueError), b)
    assert result == 2


def test_fallback_error_not_matching_raises():
    def a():
        raise TypeError

    def b():
        return 2

    with pytest.raises(TypeError):
        fallback((a, ValueError), b)


# ---------------------------------------------------------------------------
# limit_error_rate
# ---------------------------------------------------------------------------

def test_limit_error_rate_passes_success():
    @limit_error_rate(fails=3, timeout=10)
    def f(x):
        return x * 2

    assert f(5) == 10
    assert f.fails == 0
    assert f.blocked is None


def test_limit_error_rate_counts_failures_and_blocks():
    err = ErrorRateExceeded

    @limit_error_rate(fails=2, timeout=10)
    def f():
        raise ValueError("x")

    with pytest.raises(ValueError):
        f()
    assert f.fails == 1
    assert f.blocked is None

    with pytest.raises(ValueError):
        f()
    assert f.fails == 2
    assert f.blocked is not None

    with pytest.raises(err):
        f()


def test_limit_error_rate_reset_on_success():
    state = {"fail": True}

    @limit_error_rate(fails=3, timeout=10)
    def f():
        if state["fail"]:
            raise ValueError
        return "ok"

    with pytest.raises(ValueError):
        f()
    assert f.fails == 1

    state["fail"] = False
    assert f() == "ok"
    assert f.fails == 0


def test_limit_error_rate_custom_exception():
    class MyExceeded(Exception):
        pass

    @limit_error_rate(fails=1, timeout=10, exception=MyExceeded)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(MyExceeded):
        f()


def test_limit_error_rate_unblocks_after_timeout(monkeypatch):
    # datetime.now() must return an actual datetime, since (now - blocked) is
    # compared against a timedelta. Returning floats would break that.
    base = datetime(2020, 1, 1, 12, 0, 0)
    offset = {"seconds": 0.0}

    class FakeDateTime:
        @staticmethod
        def now():
            return base + timedelta(seconds=offset["seconds"])

    monkeypatch.setattr("funcy.flow.datetime", FakeDateTime)

    @limit_error_rate(fails=1, timeout=10)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert f.blocked is not None

    # Still blocked, raises ErrorRateExceeded
    with pytest.raises(ErrorRateExceeded):
        f()

    # Advance time beyond timeout
    offset["seconds"] = 20.0
    # Now unblocked, runs again and fails again
    with pytest.raises(ValueError):
        f()


def test_limit_error_rate_accepts_timedelta():
    @limit_error_rate(fails=1, timeout=timedelta(seconds=10))
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()


# ---------------------------------------------------------------------------
# throttle
# ---------------------------------------------------------------------------

def test_throttle_first_call_runs():
    @throttle(10)
    def f(x):
        return x * 2

    assert f(5) == 10


def test_throttle_second_call_skipped():
    @throttle(10)
    def f():
        return "called"

    assert f() == "called"
    # Within the period, further calls are skipped and return None.
    assert f() is None
    assert f() is None


def test_throttle_timedelta(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)

    @throttle(timedelta(seconds=5))
    def f():
        return "x"

    assert f() == "x"
    assert f() is None


def test_throttle_after_period_runs_again(monkeypatch):
    times = iter([100.0, 100.0, 200.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))

    @throttle(10)
    def f():
        return "run"

    assert f() == "run"       # t=100, sets blocked_until=110
    assert f() is None        # t=100, still blocked
    assert f() == "run"       # t=200, allowed again


# ---------------------------------------------------------------------------
# post_processing / collecting / joining
# ---------------------------------------------------------------------------

def test_post_processing_applies_func():
    @post_processing(lambda x: x * 10)
    def f():
        return 2

    assert f() == 20


def test_post_processing_with_args():
    @post_processing(sum)
    def f(a, b):
        return [a, b]

    assert f(1, 2) == 3


def test_collecting():
    @collecting
    def gen():
        yield 1
        yield 2
        yield 3

    assert gen() == [1, 2, 3]


def test_collecting_empty():
    @collecting
    def gen():
        return iter([])

    assert gen() == []


def test_collecting_name_and_doc():
    assert collecting.__name__ == "collecting"
    assert "list" in collecting.__doc__


def test_joining_strings():
    @joining("-")
    def f():
        return ["a", "b", "c"]

    assert f() == "a-b-c"


def test_joining_numbers():
    @joining(", ")
    def f():
        return [1, 2, 3]

    assert f() == "1, 2, 3"


def test_joining_empty():
    @joining(",")
    def f():
        return []

    assert f() == ""


# ---------------------------------------------------------------------------
# once / once_per / once_per_args
# ---------------------------------------------------------------------------

def test_once_runs_only_once():
    calls = []

    @once_per()
    def f():
        calls.append(1)
        return "v"

    assert f() == "v"
    assert f() is None
    assert f() is None
    assert len(calls) == 1


def test_once_per_specific_args():
    calls = []

    @once_per("x")
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 10) == 11
    assert f(1, 20) is None  # x=1 already done
    assert f(2, 10) == 12
    assert f(2, 20) is None
    assert len(calls) == 2


def test_once_per_kwargs():
    calls = []

    @once_per("x")
    def f(x):
        calls.append(x)
        return x

    assert f(x=1) == 1
    assert f(x=1) is None
    assert f(2) == 2
    assert len(calls) == 2


def test_once_per_unhashable_args():
    calls = []

    @once_per("x")
    def f(x):
        calls.append(x)
        return len(x)

    assert f([1, 2]) == 2
    assert f([1, 2]) is None
    assert f([3]) == 1
    assert len(calls) == 2


def test_once_per_args_decorator():
    calls = []

    @once_per_args
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 2) == 3
    assert f(1, 2) is None
    assert f(2, 3) == 5
    assert len(calls) == 2


def test_once_per_missing_arg_raises():
    @once_per("missing")
    def f(x):
        return x

    with pytest.raises(TypeError):
        f(1)


def test_once_thread_safety():
    from funcy.flow import once_per as once_per_fab

    calls = []
    lock = threading.Lock()

    @once_per_fab("x")
    def f(x):
        with lock:
            calls.append(x)
        return x

    threads = [threading.Thread(target=f, args=(1,)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == [1]


# ---------------------------------------------------------------------------
# wrap_with
# ---------------------------------------------------------------------------

def test_wrap_with_uses_context_manager():
    events = []

    @contextmanager
    def cm():
        events.append("enter")
        yield
        events.append("exit")

    @wrap_with(cm())
    def f():
        events.append("call")
        return "r"

    assert f() == "r"
    assert events == ["enter", "call", "exit"]


def test_wrap_with_nullcontext():
    @wrap_with(nullcontext())
    def f(x):
        return x + 1

    assert f(1) == 2


def test_wrap_with_returns_call_result():
    class CM:
        def __enter__(self):
            return 1

        def __exit__(self, *a):
            return False

    @wrap_with(CM())
    def f(x):
        return x * 3

    assert f(4) == 12


# ---------------------------------------------------------------------------
# nullcontext backport
# ---------------------------------------------------------------------------

def test_nullcontext_enter_result():
    with nullcontext(42) as v:
        assert v == 42


def test_nullcontext_default_none():
    with nullcontext() as v:
        assert v is None


def test_nullcontext_does_not_suppress():
    with pytest.raises(ValueError):
        with nullcontext():
            raise ValueError


# ---------------------------------------------------------------------------
# suppress reexport
# ---------------------------------------------------------------------------

def test_suppress_reexport():
    with suppress(ValueError):
        raise ValueError

    with pytest.raises(ValueError):
        with suppress(KeyError):
            raise ValueError
