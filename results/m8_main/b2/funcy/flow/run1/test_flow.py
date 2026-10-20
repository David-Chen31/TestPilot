import pytest
from datetime import timedelta

from funcy.flow import (
    raiser, ignore, silent, reraise, retry, fallback,
    ErrorRateExceeded, limit_error_rate, throttle,
    post_processing, joining, once_per, once_per_args, wrap_with,
    nullcontext,
    _ensure_exceptable,
)


# ---------------- raiser ----------------

def test_raiser_default_raises_exception():
    f = raiser()
    with pytest.raises(Exception):
        f()


def test_raiser_class_with_args():
    f = raiser(ValueError, "boom")
    with pytest.raises(ValueError) as ei:
        f()
    assert ei.value.args == ("boom",)


def test_raiser_instance():
    exc = ValueError("inst")
    f = raiser(exc)
    with pytest.raises(ValueError) as ei:
        f()
    assert ei.value is exc


def test_raiser_string_becomes_exception_instance():
    f = raiser("some error")
    with pytest.raises(Exception) as ei:
        f()
    assert ei.value.args == ("some error",)


def test_raiser_custom_class_with_kwargs_raises_typeerror():
    # Exception subclasses don't accept arbitrary keyword arguments, so the
    # construction itself fails with a TypeError while raising.
    class MyError(Exception):
        pass
    f = raiser(MyError, "msg", code=42)
    with pytest.raises(TypeError):
        f()


def test_raiser_kwargs_only_raises_typeerror():
    class MyError(Exception):
        pass
    f = raiser(MyError, code=1)
    with pytest.raises(TypeError):
        f()


def test_raiser_class_no_args_raises_instance():
    class MyError(Exception):
        pass
    f = raiser(MyError)
    with pytest.raises(MyError) as ei:
        f()
    assert ei.value.args == ()


# ---------------- _ensure_exceptable ----------------

def test_ensure_exceptable_exception_class_returns_class():
    assert _ensure_exceptable(ValueError) is ValueError


def test_ensure_exceptable_tuple_returns_tuple():
    result = _ensure_exceptable((ValueError, KeyError))
    assert result == (ValueError, KeyError)


def test_ensure_exceptable_list_of_exceptions_becomes_tuple():
    result = _ensure_exceptable([ValueError, KeyError])
    assert isinstance(result, tuple)
    assert result == (ValueError, KeyError)


def test_ensure_exceptable_generator_becomes_tuple():
    result = _ensure_exceptable(iter([ValueError, KeyError]))
    assert isinstance(result, tuple)
    assert result == (ValueError, KeyError)


def test_ignore_with_list_of_errors():
    # If _ensure_exceptable did not convert lists, this would fail.
    @ignore([ValueError, KeyError])
    def f():
        raise KeyError("k")
    assert f() is None


# ---------------- ignore / silent ----------------

def test_ignore_returns_value():
    @ignore(Exception)
    def f():
        return 5
    assert f() == 5


def test_ignore_returns_default_on_error():
    @ignore(ValueError, default="dflt")
    def f():
        raise ValueError("x")
    assert f() == "dflt"


def test_ignore_default_is_none():
    @ignore(ValueError)
    def f():
        raise ValueError
    assert f() is None


def test_ignore_does_not_catch_others():
    @ignore(ValueError)
    def f():
        raise TypeError("t")
    with pytest.raises(TypeError):
        f()


def test_ignore_with_tuple_of_errors():
    @ignore((ValueError, KeyError))
    def f():
        raise KeyError("k")
    assert f() is None


def test_ignore_preserves_name():
    @ignore(Exception)
    def myfunc():
        return 1
    assert myfunc.__name__ == "myfunc"


def test_silent_catches_all():
    @silent
    def f():
        raise RuntimeError("x")
    assert f() is None


def test_silent_returns_value():
    @silent
    def f():
        return 7
    assert f() == 7


# ---------------- nullcontext ----------------

def test_nullcontext_enter_result():
    with nullcontext("v") as x:
        assert x == "v"


def test_nullcontext_default_none():
    with nullcontext() as x:
        assert x is None


# ---------------- reraise ----------------

def test_reraise_translates_error():
    with pytest.raises(ValueError) as ei:
        with reraise(KeyError, ValueError):
            raise KeyError("k")
    assert isinstance(ei.value.__cause__, KeyError)


def test_reraise_no_error_passes():
    with reraise(KeyError, ValueError):
        x = 1
    assert x == 1


def test_reraise_callable_into():
    with pytest.raises(ValueError) as ei:
        with reraise(KeyError, lambda e: ValueError(str(e))):
            raise KeyError("orig")
    assert ei.value.args == ("'orig'",)
    assert isinstance(ei.value.__cause__, KeyError)


def test_reraise_exception_instance():
    exc = ValueError("inst")
    with pytest.raises(ValueError) as ei:
        with reraise(KeyError, exc):
            raise KeyError
    assert ei.value is exc


def test_reraise_only_catches_specified():
    with pytest.raises(TypeError):
        with reraise(KeyError, ValueError):
            raise TypeError("t")


# ---------------- retry ----------------

def test_retry_succeeds_first():
    calls = []
    @retry(3)
    def f():
        calls.append(1)
        return 10
    assert f() == 10
    assert len(calls) == 1


def test_retry_succeeds_after_failures():
    state = {"n": 0}
    @retry(3)
    def f():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError
        return "ok"
    assert f() == "ok"
    assert state["n"] == 3


def test_retry_raises_after_tries():
    state = {"n": 0}
    @retry(2)
    def f():
        state["n"] += 1
        raise ValueError("fail")
    with pytest.raises(ValueError):
        f()
    assert state["n"] == 2


def test_retry_only_specified_errors():
    @retry(3, errors=ValueError)
    def f():
        raise KeyError("k")
    with pytest.raises(KeyError):
        f()


def test_retry_filter_errors_rejects():
    state = {"n": 0}
    @retry(3, errors=ValueError, filter_errors=lambda e: False)
    def f():
        state["n"] += 1
        raise ValueError("v")
    with pytest.raises(ValueError):
        f()
    assert state["n"] == 1


def test_retry_filter_errors_accepts():
    state = {"n": 0}
    @retry(3, errors=ValueError, filter_errors=lambda e: True)
    def f():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError
        return "ok"
    assert f() == "ok"


def test_retry_timeout_zero_no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr("funcy.flow.time.sleep", lambda s: slept.append(s))
    state = {"n": 0}
    @retry(2, timeout=0)
    def f():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError
        return 1
    assert f() == 1
    assert slept == []


def test_retry_timeout_number_sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr("funcy.flow.time.sleep", lambda s: slept.append(s))
    state = {"n": 0}
    @retry(3, timeout=0.5)
    def f():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError
        return 1
    f()
    assert slept == [0.5, 0.5]


def test_retry_timeout_callable(monkeypatch):
    slept = []
    monkeypatch.setattr("funcy.flow.time.sleep", lambda s: slept.append(s))
    state = {"n": 0}
    @retry(3, timeout=lambda attempt: attempt + 1)
    def f():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError
        return 1
    f()
    assert slept == [1, 2]


# ---------------- fallback ----------------

def test_fallback_first_works():
    assert fallback(lambda: 1, lambda: 2) == 1


def test_fallback_second_works():
    def bad():
        raise ValueError
    assert fallback(bad, lambda: 2) == 2


def test_fallback_tuple_form():
    def bad():
        raise KeyError
    assert fallback((bad, KeyError), lambda: 9) == 9


def test_fallback_all_fail():
    # Every error is swallowed; if all approaches fail, fallback returns None.
    def bad1():
        raise ValueError
    def bad2():
        raise KeyError
    assert fallback(bad1, bad2) is None


def test_fallback_catch_specific():
    def bad():
        raise TypeError
    # ValueError catch does not apply, so TypeError propagates
    with pytest.raises(TypeError):
        fallback((bad, ValueError), lambda: 1)


def test_fallback_returns_last_success():
    assert fallback(lambda: 1, lambda: 2, lambda: 3) == 1


# ---------------- limit_error_rate ----------------

def test_limit_error_rate_blocks_after_fails():
    @limit_error_rate(2, timeout=60)
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert f.fails == 1
    with pytest.raises(ValueError):
        f()
    assert f.fails == 2
    # blocked now
    with pytest.raises(ErrorRateExceeded):
        f()


def test_limit_error_rate_initial_blocked_is_none():
    @limit_error_rate(2, timeout=60)
    def f():
        return "ok"
    # Must be None (not falsy value like "" or 0)
    assert f.blocked is None
    assert f.fails == 0


def test_limit_error_rate_resets_on_success():
    state = {"n": 0}
    @limit_error_rate(2, timeout=60)
    def f():
        state["n"] += 1
        if state["n"] == 1:
            raise ValueError
        return "ok"
    with pytest.raises(ValueError):
        f()
    assert f.fails == 1
    assert f() == "ok"
    assert f.fails == 0


def test_limit_error_rate_custom_exception():
    @limit_error_rate(1, timeout=60, exception=RuntimeError)
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()
    with pytest.raises(RuntimeError):
        f()


def test_limit_error_rate_unblocks_after_timeout(monkeypatch):
    fake_now = {"t": 0.0}

    class FakeDT:
        @classmethod
        def now(cls):
            return fake_now["t"]

    import funcy.flow as flowmod
    real_datetime = flowmod.datetime
    monkeypatch.setattr(flowmod, "datetime", FakeDT)

    @limit_error_rate(1, timeout=1)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    fake_now["t"] = 10.0
    # after timeout, allowed to call again -> raises original error
    with pytest.raises(ValueError):
        f()
    assert f.blocked is not None


def test_limit_error_rate_blocked_exact_datetime_value(monkeypatch):
    # The value stored in `blocked` after a successful unblocking call must be
    # exactly the datetime returned by `now`, not just "some truthy value".
    fake_now = {"t": "T0"}

    class FakeDT:
        @classmethod
        def now(cls):
            return fake_now["t"]

    import funcy.flow as flowmod
    monkeypatch.setattr(flowmod, "datetime", FakeDT)

    @limit_error_rate(1, timeout=0)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert f.blocked == "T0"


# ---------------- throttle ----------------

def test_throttle_first_call_runs(monkeypatch):
    times = iter([100.0, 100.5])
    monkeypatch.setattr("funcy.flow.time.monotonic", lambda: next(times))
    calls = []
    @throttle(10)
    def f(x):
        calls.append(x)
        return x
    assert f(1) == 1
    assert f(2) is None
    assert calls == [1]


def test_throttle_runs_after_period(monkeypatch):
    times = iter([100.0, 200.0])
    monkeypatch.setattr("funcy.flow.time.monotonic", lambda: next(times))
    @throttle(10)
    def f():
        return "r"
    assert f() == "r"
    assert f() == "r"


def test_throttle_timedelta(monkeypatch):
    times = iter([0.0, 5.0])
    monkeypatch.setattr("funcy.flow.time.monotonic", lambda: next(times))
    @throttle(timedelta(seconds=10))
    def f():
        return 1
    assert f() == 1
    assert f() is None


# ---------------- post_processing ----------------

def test_post_processing_applies_func():
    @post_processing(lambda x: x * 2)
    def f():
        return 5
    assert f() == 10


def test_post_processing_call_args_forwarded():
    @post_processing(lambda x: x + 1)
    def f(a, b):
        return a + b
    assert f(2, 3) == 6


# ---------------- joining ----------------

def test_joining_ints():
    @joining(",")
    def f():
        return [1, 2, 3]
    assert f() == "1,2,3"


def test_joining_strings():
    @joining("-")
    def f():
        return ["a", "b"]
    assert f() == "a-b"


def test_joining_generator():
    @joining("+")
    def f():
        return (str(i) for i in range(3))
    assert f() == "0+1+2"


# ---------------- once / once_per / once_per_args ----------------

def test_once_runs_once():
    calls = []
    @once_per()
    def f():
        calls.append(1)
        return 1
    assert f() == 1
    assert f() is None
    assert calls == [1]


def test_once_per_argnames():
    calls = []
    @once_per("a")
    def f(a, b):
        calls.append((a, b))
        return a + b
    assert f(1, 2) == 3
    assert f(1, 5) is None
    assert f(2, 5) == 7
    assert calls == [(1, 2), (2, 5)]


def test_once_per_args_decorator():
    calls = []
    @once_per_args
    def f(a, b):
        calls.append((a, b))
        return a + b
    assert f(1, 2) == 3
    assert f(1, 2) is None
    assert f(3, 4) == 7
    assert calls == [(1, 2), (3, 4)]


def test_once_per_unhashable_args():
    calls = []
    @once_per("a")
    def f(a):
        calls.append(a)
        return len(a)
    assert f([1, 2]) == 2
    assert f([1, 2]) is None
    assert f([3]) == 1
    assert len(calls) == 2


def test_once_per_kwargs():
    calls = []
    @once_per("a")
    def f(a, b=0):
        calls.append((a, b))
        return a + b
    assert f(a=1, b=2) == 3
    assert f(a=1, b=3) is None
    assert calls == [(1, 2)]


def test_once_per_args_with_defaults():
    calls = []
    @once_per_args
    def f(a, b=10):
        calls.append((a, b))
        return a + b
    assert f(1) == 11
    assert f(1) is None
    assert f(1, 20) == 21
    assert calls == [(1, 10), (1, 20)]


# ---------------- wrap_with ----------------

def test_wrap_with_context_manager():
    log = []
    class CM:
        def __enter__(self):
            log.append("enter")
            return self
        def __exit__(self, *exc):
            log.append("exit")
            return False

    @wrap_with(CM())
    def f():
        log.append("call")
        return 5
    assert f() == 5
    assert log == ["enter", "call", "exit"]


def test_wrap_with_propagates_errors():
    log = []
    class CM:
        def __enter__(self):
            log.append("enter")
            return self
        def __exit__(self, *exc):
            log.append("exit")
            return False

    @wrap_with(CM())
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()
    assert log == ["enter", "exit"]
