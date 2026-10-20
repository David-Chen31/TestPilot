import time
import pytest
from datetime import timedelta

from funcy.flow import (
    raiser, ignore, silent, reraise, retry, fallback,
    ErrorRateExceeded, limit_error_rate, throttle,
    post_processing, collecting, joining,
    once_per, once_per_args, wrap_with,
)


# ---------------------------------------------------------------------------
# raiser
# ---------------------------------------------------------------------------

def test_raiser_raises_nothing_special_class():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_with_str_wraps_in_exception():
    f = raiser("boom")
    with pytest.raises(Exception) as ei:
        f()
    assert str(ei.value) == "boom"


def test_raiser_with_args():
    f = raiser(ValueError, "a", "b")
    with pytest.raises(ValueError) as ei:
        f()
    assert ei.value.args == ("a", "b")


def test_raiser_with_kwargs():
    f = raiser(ValueError, code=42)
    with pytest.raises(ValueError) as ei:
        f()
    assert ei.value.args == ()  # kwargs won't appear in args
    # but constructing with kwargs would normally fail for ValueError;
    # the code path passes them to constructor
def test_raiser_passes_args_and_kwargs(monkeypatch):
    calls = []
    class MyErr(Exception):
        def __init__(self, *a, **kw):
            calls.append((a, kw))
    f = raiser(MyErr, 1, x=2)
    with pytest.raises(MyErr):
        f()
    assert calls == [((1,), {"x": 2})]


def test_raiser_default_exception():
    f = raiser()
    with pytest.raises(Exception):
        f()


# ---------------------------------------------------------------------------
# ignore / silent
# ---------------------------------------------------------------------------

def test_ignore_returns_value():
    @ignore(ZeroDivisionError, default=42)
    def f(x):
        return 10 // x
    assert f(2) == 5
    assert f(0) == 42


def test_ignore_default_none():
    @ignore(ValueError)
    def f():
        raise ValueError
    assert f() is None


def test_ignore_does_not_catch_other():
    @ignore(ValueError)
    def f():
        raise KeyError
    with pytest.raises(KeyError):
        f()


def test_ignore_accepts_tuple():
    @ignore((ValueError, KeyError), default="x")
    def f(e):
        raise e
    assert f(ValueError) == "x"
    assert f(KeyError) == "x"


def test_ignore_accepts_list():
    @ignore([ValueError, KeyError], default="y")
    def f():
        raise KeyError
    assert f() == "y"


def test_ignore_preserves_name_and_doc():
    @ignore(ValueError)
    def myfunc():
        """doc here"""
        return 1
    assert myfunc.__name__ == "myfunc"
    assert myfunc.__doc__ == "doc here"


def test_silent_catches_everything():
    @silent
    def f():
        raise KeyError("x")
    assert f() is None


def test_silent_returns_value_and_preserves_metadata():
    @silent
    def f(x):
        """the doc"""
        return x * 2
    assert f(3) == 6
    assert f.__name__ == "f"
    assert f.__doc__ == "the doc"


# ---------------------------------------------------------------------------
# reraise
# ---------------------------------------------------------------------------

def test_reraise_with_class():
    with pytest.raises(ValueError):
        with reraise(KeyError, ValueError):
            raise KeyError("k")


def test_reraise_chains_cause():
    try:
        with reraise(KeyError, ValueError):
            raise KeyError("orig")
    except ValueError as e:
        assert isinstance(e.__cause__, KeyError)
        assert str(e.__cause__) == "'orig'"


def test_reraise_callable_factory():
    def factory(e):
        return RuntimeError("wrapped " + str(e))
    with pytest.raises(RuntimeError) as ei:
        with reraise(KeyError, factory):
            raise KeyError("boom")
    assert str(ei.value) == "wrapped 'boom'"
    assert isinstance(ei.value.__cause__, KeyError)


def test_reraise_no_exception_passes():
    with reraise(KeyError, ValueError):
        result = 7
    assert result == 7


def test_reraise_does_not_catch_other():
    with pytest.raises(TypeError):
        with reraise(KeyError, ValueError):
            raise TypeError


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


def test_retry_recovers():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError
        return "done"
    assert f() == "done"
    assert len(calls) == 3


def test_retry_exhausts_and_raises_last():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        raise ValueError("v")
    with pytest.raises(ValueError):
        f()
    assert len(calls) == 3


def test_retry_only_on_specified_errors():
    @retry(3, errors=ValueError)
    def f():
        raise KeyError
    with pytest.raises(KeyError):
        f()


def test_retry_filter_errors_false_reraises_immediately():
    calls = []

    @retry(5, errors=ValueError, filter_errors=lambda e: False)
    def f():
        calls.append(1)
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert len(calls) == 1


def test_retry_filter_errors_true_retries():
    calls = []

    @retry(3, errors=ValueError, filter_errors=lambda e: True)
    def f():
        calls.append(1)
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert len(calls) == 3


def test_retry_timeout_fixed_uses_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    calls = []

    @retry(3, timeout=0.5)
    def f():
        calls.append(1)
        raise ValueError
    with pytest.raises(ValueError):
        f()
    # no sleep after the final attempt
    assert slept == [0.5, 0.5]


def test_retry_timeout_callable(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    calls = []

    @retry(3, timeout=lambda attempt: 2 * (attempt + 1))
    def f():
        calls.append(1)
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert slept == [2, 4]


def test_retry_zero_timeout_does_not_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    @retry(3, timeout=0)
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert slept == []


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------

def test_fallback_first_wins():
    assert fallback(lambda: "a", lambda: "b") == "a"


def test_fallback_skips_failing():
    def bad():
        raise ValueError
    assert fallback(bad, lambda: "good") == "good"


def test_fallback_with_expected_errors_tuple():
    def bad():
        raise KeyError
    assert fallback((bad, KeyError), lambda: 5) == 5


def test_fallback_does_not_catch_unexpected():
    def bad():
        raise TypeError

    def other():
        raise ValueError
    with pytest.raises(TypeError):
        fallback((bad, KeyError), (other, ValueError))


def test_fallback_all_fail_returns_none():
    def bad1():
        raise ValueError

    def bad2():
        raise KeyError
    assert fallback(bad1, bad2) is None


# ---------------------------------------------------------------------------
# limit_error_rate
# ---------------------------------------------------------------------------

def test_limit_error_rate_resets_on_success():
    @limit_error_rate(fails=2, timeout=10)
    def f(x):
        if x < 0:
            raise ValueError
        return x
    assert f(1) == 1
    # one failure then success resets counter
    with pytest.raises(ValueError):
        f(-1)
    assert f(2) == 2
    # fails counter reset, so one more failure should not block
    with pytest.raises(ValueError):
        f(-1)


def test_limit_error_rate_blocks_after_fails():
    @limit_error_rate(fails=2, timeout=100)
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    # blocked now
    with pytest.raises(ErrorRateExceeded):
        f()


def test_limit_error_rate_custom_exception():
    @limit_error_rate(fails=1, timeout=100, exception=RuntimeError)
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()
    with pytest.raises(RuntimeError):
        f()


def test_limit_error_rate_timeout_expires(monkeypatch):
    import funcy.flow as flow_mod

    fake_now = [1000.0]

    class FakeDatetime:
        @staticmethod
        def now():
            from datetime import datetime as real
            return real.fromtimestamp(fake_now[0])

    monkeypatch.setattr(flow_mod, "datetime", FakeDatetime)

    attempts = []

    @limit_error_rate(fails=1, timeout=10)
    def f():
        attempts.append(1)
        raise ValueError
    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()
    assert len(attempts) == 1  # blocked, func not called

    # advance beyond timeout
    fake_now[0] += 100
    with pytest.raises(ValueError):
        f()
    assert len(attempts) == 2


def test_limit_error_rate_timedelta_timeout():
    from datetime import datetime
    @limit_error_rate(fails=1, timeout=timedelta(seconds=100))
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()


# ---------------------------------------------------------------------------
# throttle
# ---------------------------------------------------------------------------

def test_throttle_within_period(monkeypatch):
    fake_time = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

    calls = []

    @throttle(10)
    def f(x):
        calls.append(x)
        return x
    assert f(1) == 1
    fake_time[0] = 5.0
    assert f(2) is None
    fake_time[0] = 10.0
    assert f(3) == 3
    assert calls == [1, 3]


def test_throttle_timedelta(monkeypatch):
    fake_time = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])
    calls = []

    @throttle(timedelta(seconds=5))
    def f():
        calls.append(1)
        return "run"
    assert f() == "run"
    assert f() is None
    fake_time[0] = 6
    assert f() == "run"
    assert calls == [1, 1]


# ---------------------------------------------------------------------------
# post_processing / collecting / joining
# ---------------------------------------------------------------------------

def test_post_processing():
    @post_processing(lambda x: x * 2)
    def f():
        return 5
    assert f() == 10


def test_collecting():
    @collecting
    def gen():
        yield 1
        yield 2
        yield 3
    assert gen() == [1, 2, 3]


def test_collecting_metadata():
    assert collecting.__name__ == "collecting"
    assert collecting.__doc__ == "Transforms a generator into list returning function."


def test_joining():
    @joining("-")
    def f():
        yield 1
        yield 2
        yield 3
    assert f() == "1-2-3"


def test_joining_custom_sep_class():
    @joining("*")
    def f():
        return [1, 2, 3]
    assert f() == "1*2*3"


# ---------------------------------------------------------------------------
# once / once_per / once_per_args
# ---------------------------------------------------------------------------

def test_once_runs_only_once():
    calls = []

    @once_per()
    def f(x):
        calls.append(x)
        return x * 2
    assert f(1) == 2
    assert f(1) is None
    assert f(2) == 4
    assert calls == [1, 2]


def test_once_per_specified_args():
    calls = []

    @once_per("a")
    def f(a, b):
        calls.append((a, b))
        return (a, b)
    assert f(1, 2) == (1, 2)
    # same 'a', different b -> skipped
    assert f(1, 9) is None
    assert f(2, 3) == (2, 3)
    assert calls == [(1, 2), (2, 3)]


def test_once_per_multiple_args():
    calls = []

    @once_per("a", "b")
    def f(a, b, c):
        calls.append((a, b, c))
        return c
    assert f(1, 2, 3) == 3
    assert f(1, 2, 4) is None
    assert f(1, 5, 6) == 6


def test_once_per_default_argument():
    calls = []

    @once_per("a")
    def f(a=1):
        calls.append(a)
        return a
    assert f() == 1
    assert f() is None  # same default value already seen
    assert f(2) == 2
    assert calls == [1, 2]


def test_once_per_unhashable_uses_list():
    calls = []

    @once_per("a")
    def f(a):
        calls.append(a)
        return a
    assert f([1, 2]) == [1, 2]
    assert f([1, 2]) is None
    assert calls == [[1, 2]]
    # different equal-but-distinct list counts as same values
    assert f([1, 2]) is None


def test_once_per_missing_arg():
    @once_per("missing")
    def f(a):
        return a
    with pytest.raises(TypeError):
        f(1)


def test_once_per_args():
    calls = []

    @once_per_args
    def f(a, b):
        calls.append((a, b))
        return (a, b)
    assert f(1, 2) == (1, 2)
    assert f(1, 2) is None
    assert f(1, 3) == (1, 3)
    assert calls == [(1, 2), (1, 3)]


def test_once_per_args_kwargs():
    calls = []

    @once_per_args
    def f(a, b=10):
        calls.append((a, b))
        return (a, b)
    assert f(1) == (1, 10)
    assert f(1, b=10) is None
    assert f(1, 20) == (1, 20)


# ---------------------------------------------------------------------------
# wrap_with
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self, calls, value="ctx-value"):
        self.calls = calls
        self.value = value

    def __enter__(self):
        self.calls.append("enter")
        return self.value

    def __exit__(self, *excinfo):
        self.calls.append(("exit", excinfo))
        return False


def test_wrap_with_enters_context():
    calls = []

    @wrap_with(_Recorder(calls))
    def f():
        calls.append("body")
        return 7
    assert f() == 7
    assert calls[0] == "enter"
    assert calls[1] == "body"
    assert calls[2][0] == "exit"
    assert calls[2][1] == (None, None, None)


def test_wrap_with_propagates_exception():
    calls = []

    @wrap_with(_Recorder(calls))
    def f():
        raise ValueError("x")
    with pytest.raises(ValueError):
        f()
    assert calls[0] == "enter"
    exit_info = calls[1]
    assert exit_info[0] == "exit"
    assert exit_info[1][0] is ValueError
