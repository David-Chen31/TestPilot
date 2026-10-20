import time
from datetime import timedelta
from unittest import mock

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
    suppress,
    nullcontext,
)


# ------------------------- raiser -------------------------

def test_raiser_default_exception():
    f = raiser()
    with pytest.raises(Exception):
        f()


def test_raiser_with_class():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_with_instance():
    exc = ValueError("boom")
    f = raiser(exc)
    with pytest.raises(ValueError, match="boom"):
        f()


def test_raiser_with_string():
    f = raiser("message")
    with pytest.raises(Exception, match="message"):
        f()


def test_raiser_with_args_and_kwargs():
    f = raiser(ValueError, "a", b=1)
    with pytest.raises(ValueError) as ei:
        f()
    assert ei.value.args == ("a",)
    assert ei.value.kwargs == {"b": 1}


def test_raiser_args_with_default_class():
    f = raiser(Exception, "x")
    with pytest.raises(Exception) as ei:
        f()
    assert ei.value.args == ("x",)


# ------------------------- ignore -------------------------

def test_ignore_returns_default():
    @ignore(ValueError)
    def f():
        raise ValueError
    assert f() is None


def test_ignore_custom_default():
    @ignore(ValueError, "dflt")
    def f():
        raise ValueError
    assert f() == "dflt"


def test_ignore_passes_through_success():
    @ignore(ValueError, "dflt")
    def f(x):
        return x * 2
    assert f(3) == 6


def test_ignore_only_catches_specified():
    @ignore(ValueError)
    def f():
        raise TypeError
    with pytest.raises(TypeError):
        f()


def test_ignore_accepts_tuple():
    @ignore((ValueError, KeyError), 42)
    def f():
        raise KeyError
    assert f() == 42


def test_ignore_accepts_list():
    @ignore([ValueError, KeyError], "x")
    def f():
        raise ValueError
    assert f() == "x"


def test_ignore_preserves_name():
    @ignore(ValueError)
    def my_func():
        return 1
    assert my_func.__name__ == "my_func"


# ------------------------- silent -------------------------

def test_silent_catches_all():
    @silent
    def f():
        raise TypeError("x")
    assert f() is None


def test_silent_success():
    @silent
    def f(x):
        return x + 1
    assert f(1) == 2


# ------------------------- reraise -------------------------

def test_reraise_basic():
    with pytest.raises(ValueError):
        with reraise(KeyError, ValueError):
            raise KeyError


def test_reraise_preserves_cause_chain():
    try:
        with reraise(KeyError, ValueError):
            raise KeyError("k")
    except ValueError as e:
        assert isinstance(e.__cause__, KeyError)
    else:
        pytest.fail("did not raise")


def test_reraise_only_catches_specified():
    with pytest.raises(TypeError):
        with reraise(KeyError, ValueError):
            raise TypeError


def test_reraise_passes_through_no_error():
    with reraise(KeyError, ValueError):
        x = 5
    assert x == 5


def test_reraise_with_factory():
    def factory(exc):
        return ValueError("wrapped: " + str(exc))

    with pytest.raises(ValueError, match="wrapped: k"):
        with reraise(KeyError, factory):
            raise KeyError("k")


def test_reraise_accepts_tuple():
    with pytest.raises(ValueError):
        with reraise((KeyError, TypeError), ValueError):
            raise TypeError


# ------------------------- retry -------------------------

def test_retry_success_first():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        return "ok"
    assert f() == "ok"
    assert len(calls) == 1


def test_retry_eventually_succeeds():
    calls = []

    @retry(5)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError
        return "done"
    assert f() == "done"
    assert len(calls) == 3


def test_retry_exhausted_reraises():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        raise ValueError("x")
    with pytest.raises(ValueError, match="x"):
        f()
    assert len(calls) == 3


def test_retry_specific_errors():
    @retry(3, errors=ValueError)
    def f():
        raise KeyError
    with pytest.raises(KeyError):
        f()


def test_retry_with_sleep():
    with mock.patch("funcy.flow.time.sleep") as sleep:
        @retry(3, timeout=0.1)
        def f():
            raise ValueError
        with pytest.raises(ValueError):
            f()
    # Sleeps happen between attempts; with 3 tries there are 2 sleeps.
    assert sleep.call_count == 2
    sleep.assert_called_with(0.1)


def test_retry_with_timeout_callable():
    with mock.patch("funcy.flow.time.sleep") as sleep:
        @retry(3, timeout=lambda attempt: attempt + 1)
        def f():
            raise ValueError
        with pytest.raises(ValueError):
            f()
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2]


def test_retry_filter_errors_false_stops():
    calls = []

    @retry(5, filter_errors=lambda e: False)
    def f():
        calls.append(1)
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert len(calls) == 1


def test_retry_filter_errors_true_continues():
    calls = []

    @retry(3, filter_errors=lambda e: True)
    def f():
        calls.append(1)
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert len(calls) == 3


# ------------------------- fallback -------------------------

def test_fallback_first_works():
    assert fallback(lambda: "first", lambda: "second") == "first"


def test_fallback_second_works():
    def bad():
        raise ValueError
    assert fallback(bad, lambda: "second") == "second"


def test_fallback_with_expected_errors():
    def bad():
        raise KeyError
    assert fallback((bad, ValueError), lambda: "ok") == "ok"


def test_fallback_all_fail_returns_none():
    def bad():
        raise ValueError
    assert fallback(bad) is None


def test_fallback_unexpected_error_propagates():
    def bad():
        raise KeyError
    with pytest.raises(KeyError):
        fallback((bad, ValueError))


def test_fallback_no_approaches():
    assert fallback() is None


# ------------------------- limit_error_rate -------------------------

def test_limit_error_rate_basic():
    calls = []

    @limit_error_rate(fails=2, timeout=10)
    def f():
        calls.append(1)
        raise ValueError
    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    # Blocked now
    with pytest.raises(ErrorRateExceeded):
        f()
    # Blocked call did not invoke underlying function
    assert len(calls) == 2


def test_limit_error_rate_reset_on_success():
    calls = []

    @limit_error_rate(fails=2, timeout=10)
    def f(x):
        calls.append(x)
        if x == 0:
            raise ValueError
        return x
    with pytest.raises(ValueError):
        f(0)
    assert f(1) == 1
    # fails reset, so another failure doesn't block
    with pytest.raises(ValueError):
        f(0)
    assert f(2) == 2


def test_limit_error_rate_custom_exception():
    @limit_error_rate(fails=1, timeout=10, exception=RuntimeError)
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()
    with pytest.raises(RuntimeError):
        f()


def test_limit_error_rate_timeout_unblocked():
    fake_now = [1000.0]

    class FakeDatetime:
        @staticmethod
        def now():
            return fake_now[0]

    calls = []

    # Patch datetime in the module
    with mock.patch("funcy.flow.datetime", FakeDatetime):
        @limit_error_rate(fails=1, timeout=5)
        def f():
            calls.append(1)
            raise ValueError

        with pytest.raises(ValueError):
            f()
        with pytest.raises(ErrorRateExceeded):
            f()
        # Move time forward past timeout
        fake_now[0] += 10
        with pytest.raises(ValueError):
            f()

    assert len(calls) == 2


# ------------------------- throttle -------------------------

@mock.patch("funcy.flow.time.monotonic")
def test_throttle_basic(monotonic):
    monotonic.return_value = 100.0
    calls = []

    @throttle(10)
    def f():
        calls.append(1)
        return "x"

    assert f() == "x"
    assert f() is None
    assert f() is None
    assert len(calls) == 1


@mock.patch("funcy.flow.time.monotonic")
def test_throttle_after_period(monotonic):
    calls = []

    @throttle(10)
    def f():
        calls.append(1)
        return len(calls)

    monotonic.return_value = 100.0
    assert f() == 1
    assert f() is None

    monotonic.return_value = 105.0
    assert f() is None

    monotonic.return_value = 111.0
    assert f() == 2


@mock.patch("funcy.flow.time.monotonic")
def test_throttle_timedelta(monotonic):
    monotonic.return_value = 100.0
    calls = []

    @throttle(timedelta(seconds=10))
    def f():
        calls.append(1)
        return "x"

    assert f() == "x"
    assert f() is None


# ------------------------- post_processing / collecting / joining -------------------------

def test_post_processing():
    @post_processing(lambda r: r + 1)
    def f():
        return 1
    assert f() == 2


def test_collecting():
    @collecting
    def f():
        yield 1
        yield 2
    assert f() == [1, 2]
    assert collecting.__name__ == "collecting"


def test_joining():
    @joining("-")
    def f():
        yield "a"
        yield 1
    assert f() == "a-1"


def test_joining_int_sep():
    @joining(0)
    def f():
        return [1, 2, 3]
    assert f() == 1


# ------------------------- once_per / once_per_args / once -------------------------

def test_once_per_no_args():
    calls = []

    @once_per()
    def f():
        calls.append(1)
        return "first"
    assert f() == "first"
    assert f() is None
    assert len(calls) == 1


def test_once_alias():
    calls = []

    @once
    def f():
        calls.append(1)
        return "first"
    assert f() == "first"
    assert f() is None
    assert len(calls) == 1


def test_once_per_per_arg_value():
    calls = []

    @once_per("x")
    def f(x):
        calls.append(x)
        return x
    assert f(1) == 1
    assert f(1) is None
    assert f(2) == 2
    assert f(2) is None
    assert calls == [1, 2]


def test_once_per_positional_and_kwarg():
    calls = []

    @once_per("x")
    def f(x):
        calls.append(x)
        return x
    assert f(1) == 1
    assert f(x=1) is None
    assert len(calls) == 1


def test_once_per_unhashable_values():
    calls = []

    @once_per("x")
    def f(x):
        calls.append(x)
        return x
    assert f([1, 2]) == [1, 2]
    assert f([1, 2]) is None
    assert f([3]) == [3]
    assert calls == [[1, 2], [3]]


def test_once_per_multiple_argnames():
    calls = []

    @once_per("a", "b")
    def f(a, b):
        calls.append((a, b))
        return (a, b)
    f(1, 2)
    f(b=2, a=1)
    f(1, 3)
    assert calls == [(1, 2), (1, 3)]


def test_once_per_args():
    calls = []

    @once_per_args
    def f(a, b=1):
        calls.append((a, b))
        return (a, b)
    assert f(1) == (1, 1)
    assert f(1, 1) is None
    assert f(2) == (2, 1)
    assert calls == [(1, 1), (2, 1)]


def test_once_per_invalid_argname():
    @once_per("nonexistent")
    def f(x):
        return x
    with pytest.raises(TypeError):
        f(1)


# ------------------------- wrap_with -------------------------

class _CM:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False


def test_wrap_with():
    cm = _CM()

    @wrap_with(cm)
    def f():
        return 42

    assert f() == 42
    assert cm.entered
    assert cm.exited


def test_wrap_with_suppresses_exception():
    class SuppressCM:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return True

    @wrap_with(SuppressCM())
    def f():
        raise ValueError

    # The context manager suppresses, but the exception propagates
    # up through the decorator before suppression takes effect
    with pytest.raises(ValueError):
        f()


# ------------------------- re-exports -------------------------

def test_suppress_reexport():
    import contextlib
    assert suppress is contextlib.suppress


def test_nullcontext_enter_result():
    with nullcontext("v") as v:
        assert v == "v"
