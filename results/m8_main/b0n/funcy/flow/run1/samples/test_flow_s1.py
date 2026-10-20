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

def test_raiser_default_exception():
    f = raiser()
    with pytest.raises(Exception):
        f()


def test_raiser_specific_exception_class():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_with_args():
    f = raiser(ValueError, "bad", 42)
    with pytest.raises(ValueError) as exc:
        f()
    assert exc.value.args == ("bad", 42)


def test_raiser_with_kwargs():
    f = raiser(ValueError, 1, x=2)
    with pytest.raises(ValueError) as exc:
        f()
    assert exc.value.args == (1,)
    # kwargs passed to exception constructor are not stored in args
    # but should be accepted without error


def test_raiser_with_string_message():
    f = raiser("some message")
    with pytest.raises(Exception) as exc:
        f()
    assert str(exc.value) == "some message"


def test_raiser_accepts_arguments_at_call_time():
    f = raiser(ValueError, "fixed")
    with pytest.raises(ValueError) as exc:
        f("ignored")
    assert exc.value.args == ("fixed",)


# ---------------------------------------------------------------------------
# ignore / silent
# ---------------------------------------------------------------------------

def test_ignore_returns_value():
    @ignore(ValueError)
    def f(x):
        return x * 2

    assert f(3) == 6


def test_ignore_swallows_error():
    @ignore(ValueError)
    def f():
        raise ValueError("boom")

    assert f() is None


def test_ignore_default():
    @ignore(ValueError, default=99)
    def f():
        raise ValueError

    assert f() == 99


def test_ignore_tuple_of_errors():
    @ignore((ValueError, KeyError), default="d")
    def f(e):
        raise e

    assert f(ValueError()) == "d"
    assert f(KeyError()) == "d"


def test_ignore_does_not_catch_other():
    @ignore(ValueError)
    def f():
        raise TypeError

    with pytest.raises(TypeError):
        f()


def test_ignore_single_error_class():
    @ignore(KeyError)
    def f():
        raise KeyError

    assert f() is None


def test_silent_catches_all():
    @silent
    def f(x):
        if x == 0:
            raise RuntimeError("zero")
        return x

    assert f(1) == 1
    assert f(0) is None


def test_silent_preserves_name():
    @silent
    def myfunc():
        return 1

    assert myfunc.__name__ == "myfunc"


# ---------------------------------------------------------------------------
# reraise
# ---------------------------------------------------------------------------

def test_reraise_changes_exception():
    with pytest.raises(ValueError):
        with reraise(KeyError, ValueError):
            raise KeyError


def test_reraise_does_not_trigger_on_other():
    with pytest.raises(TypeError):
        with reraise(KeyError, ValueError):
            raise TypeError


def test_reraise_normal_exit():
    with reraise(KeyError, ValueError):
        result = 5
    assert result == 5


def test_reraise_with_factory_callable():
    class MyError(Exception):
        pass

    def factory(e):
        return MyError(str(e))

    with pytest.raises(MyError) as exc:
        with reraise(KeyError, factory):
            raise KeyError("original")
    assert str(exc.value) == "'original'"
    assert isinstance(exc.value.__cause__, KeyError)


def test_reraise_with_exception_class_as_into():
    with pytest.raises(ValueError) as exc:
        with reraise(KeyError, ValueError):
            raise KeyError("msg")
    assert isinstance(exc.value.__cause__, KeyError)


def test_reraise_tuple_errors():
    with pytest.raises(RuntimeError):
        with reraise((KeyError, IndexError), RuntimeError):
            raise IndexError


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

def test_retry_succeeds_first_time():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        return 42

    assert f() == 42
    assert len(calls) == 1


def test_retry_eventually_succeeds():
    state = {"n": 0}

    @retry(3)
    def f():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError
        return "ok"

    assert f() == "ok"
    assert state["n"] == 3


def test_retry_reraises_after_tries():
    state = {"n": 0}

    @retry(3)
    def f():
        state["n"] += 1
        raise ValueError("fail")

    with pytest.raises(ValueError):
        f()
    assert state["n"] == 3


def test_retry_only_on_specified_errors():
    state = {"n": 0}

    @retry(3, errors=KeyError)
    def f():
        state["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert state["n"] == 1


def test_retry_filter_errors_rejects():
    state = {"n": 0}

    @retry(3, filter_errors=lambda e: False)
    def f():
        state["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert state["n"] == 1


def test_retry_filter_errors_accepts():
    state = {"n": 0}

    @retry(3, filter_errors=lambda e: True)
    def f():
        state["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert state["n"] == 3


def test_retry_timeout_callable(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    state = {"n": 0}

    @retry(3, timeout=lambda attempt: attempt + 1)
    def f():
        state["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    # sleeps between attempts: attempt 0 -> 1, attempt 1 -> 2
    assert sleeps == [1, 2]


def test_retry_timeout_numeric(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    @retry(2, timeout=0.5)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert sleeps == [0.5]


def test_retry_negative_timeout_no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    @retry(2, timeout=-1)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert sleeps == []


def test_retry_with_args_and_kwargs():
    state = {"n": 0}

    @retry(3)
    def f(a, b=2):
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError
        return a + b

    assert f(1, b=3) == 4


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------

def test_fallback_first_succeeds():
    result = fallback(lambda: 1, lambda: 2)
    assert result == 1


def test_fallback_second_succeeds():
    def bad():
        raise ValueError

    result = fallback(bad, lambda: 2)
    assert result == 2


def test_fallback_with_expected_errors_tuple():
    def bad():
        raise KeyError

    def good():
        return "ok"

    result = fallback((bad, KeyError), good)
    assert result == "ok"


def test_fallback_uncaught_error_propagates():
    def bad():
        raise TypeError

    with pytest.raises(TypeError):
        fallback((bad, KeyError), lambda: 2)


def test_fallback_all_fail_returns_none():
    def bad():
        raise ValueError

    assert fallback(bad, bad) is None


# ---------------------------------------------------------------------------
# limit_error_rate
# ---------------------------------------------------------------------------

def test_limit_error_rate_disabled_until_threshold():
    calls = {"n": 0}

    @limit_error_rate(2, 60)
    def f():
        calls["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    # threshold reached, next call blocked
    with pytest.raises(ErrorRateExceeded):
        f()
    assert calls["n"] == 2


def test_limit_error_rate_resets_on_success():
    state = {"fail": True}

    @limit_error_rate(2, 60)
    def f():
        if state["fail"]:
            raise ValueError
        return "ok"

    with pytest.raises(ValueError):
        f()
    state["fail"] = False
    assert f() == "ok"
    # after success, fails reset
    state["fail"] = True
    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()


def test_limit_error_rate_custom_exception():
    @limit_error_rate(1, 60, exception=RuntimeError)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(RuntimeError):
        f()


def test_limit_error_rate_unblocks_after_timeout(monkeypatch):
    fake_now = [1000.0]

    class FakeDateTime:
        @staticmethod
        def now():
            from datetime import datetime as real_datetime
            return real_datetime.now()  # not used

    # Simpler: patch wrapper internals via monkeypatching datetime in module
    import funcy.flow as flow_mod

    class FakeDatetime:
        @staticmethod
        def now():
            from datetime import datetime as DT, timedelta
            return DT(2020, 1, 1) + timedelta(seconds=fake_now[0])

    monkeypatch.setattr(flow_mod, "datetime", FakeDatetime)

    @limit_error_rate(1, 10)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    # blocked
    with pytest.raises(ErrorRateExceeded):
        f()
    # move forward past timeout
    fake_now[0] = 100.0
    with pytest.raises(ValueError):
        f()


def test_limit_error_rate_preserves_metadata():
    @limit_error_rate(1, 60)
    def myfunc():
        return 1

    assert myfunc.__name__ == "myfunc"
    assert myfunc.fails == 0
    assert myfunc.blocked is None


# ---------------------------------------------------------------------------
# throttle
# ---------------------------------------------------------------------------

def test_throttle_runs_once_per_period(monkeypatch):
    fake = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake[0])

    calls = []

    @throttle(5)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(2) is None  # blocked
    fake[0] = 3.0
    assert f(3) is None  # still blocked
    fake[0] = 6.0
    assert f(4) == 4
    assert calls == [1, 4]


def test_throttle_with_timedelta(monkeypatch):
    fake = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake[0])

    @throttle(timedelta(seconds=2))
    def f():
        return "x"

    assert f() == "x"
    fake[0] = 1.0
    assert f() is None
    fake[0] = 2.0
    assert f() == "x"


def test_throttle_preserves_metadata():
    @throttle(1)
    def myfunc():
        pass

    assert myfunc.__name__ == "myfunc"
    assert myfunc.blocked_until == 0


# ---------------------------------------------------------------------------
# post_processing / collecting / joining
# ---------------------------------------------------------------------------

def test_post_processing_transforms_result():
    @post_processing(lambda x: x + 1)
    def f():
        return 41

    assert f() == 42


def test_collecting():
    @collecting
    def gen(n):
        for i in range(n):
            yield i

    assert gen(3) == [0, 1, 2]


def test_collecting_name_and_doc():
    assert collecting.__name__ == "collecting"
    assert "generator" in collecting.__doc__.lower()


def test_joining():
    @joining(",")
    def gen():
        yield 1
        yield 2
        yield 3

    assert gen() == "1,2,3"


# ---------------------------------------------------------------------------
# once / once_per / once_per_args
# ---------------------------------------------------------------------------

def test_once_runs_only_once():
    calls = []

    @once
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(2) is None
    assert calls == [1]


def test_once_per_distinct_args():
    calls = []

    @once_per("x")
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 10) == 11
    assert f(1, 20) is None  # same x
    assert f(2, 20) == 22
    assert calls == [(1, 10), (2, 20)]


def test_once_per_with_defaults():
    calls = []

    @once_per("x")
    def f(x=5):
        calls.append(x)
        return x

    assert f() == 5
    assert f(x=5) is None
    assert f(6) == 6
    assert calls == [5, 6]


def test_once_per_unhashable_args():
    calls = []

    @once_per("x")
    def f(x):
        calls.append(x)
        return x

    assert f([1, 2]) == [1, 2]
    assert f([1, 2]) is None
    assert f([3]) == [3]
    assert calls == [[1, 2], [3]]


def test_once_per_args_all_arguments():
    calls = []

    @once_per_args
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 2) == 3
    assert f(1, 2) is None
    assert f(1, 3) == 4
    assert calls == [(1, 2), (1, 3)]


def test_once_per_preserves_name():
    @once_per("x")
    def myfunc(x):
        return x

    assert myfunc.__name__ == "myfunc"


# ---------------------------------------------------------------------------
# wrap_with
# ---------------------------------------------------------------------------

def test_wrap_with_enters_context():
    entered = []

    class CM:
        def __enter__(self):
            entered.append("enter")
            return self

        def __exit__(self, *exc):
            entered.append("exit")
            return False

    @wrap_with(CM())
    def f():
        entered.append("call")
        return 42

    assert f() == 42
    assert entered == ["enter", "call", "exit"]


def test_wrap_with_propagates_exceptions():
    class CM:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    @wrap_with(CM())
    def f():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        f()


def test_wrap_with_uses_fresh_context_manager_instance():
    # The context manager passed to wrap_with is shared; check that usage
    # works with contextlib-like object that returns itself
    entered = []

    class CM:
        def __enter__(self):
            entered.append("enter")
            return self

        def __exit__(self, *exc):
            entered.append("exit")
            return False

    cm = CM()

    @wrap_with(cm)
    def f(x):
        return x * 2

    assert f(1) == 2
    assert f(2) == 4
    assert entered == ["enter", "exit", "enter", "exit"]
