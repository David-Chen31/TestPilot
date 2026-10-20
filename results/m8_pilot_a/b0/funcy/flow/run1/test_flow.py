import time
from datetime import datetime, timedelta

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


# ---------------- raiser ----------------

def test_raiser_default():
    f = raiser()
    with pytest.raises(Exception):
        f()


def test_raiser_with_class():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_with_string():
    f = raiser("boom")
    with pytest.raises(Exception) as ei:
        f()
    assert str(ei.value) == "boom"


def test_raiser_with_args():
    f = raiser(ValueError, "msg")
    with pytest.raises(ValueError) as ei:
        f()
    assert str(ei.value) == "msg"


def test_raiser_ignores_invocation_args():
    f = raiser(ValueError, "msg")
    with pytest.raises(ValueError) as ei:
        f(1, 2, x=3)
    assert str(ei.value) == "msg"


# ---------------- ignore / silent ----------------

def test_ignore_returns_value_when_no_error():
    @ignore(ValueError, default=42)
    def f():
        return 1
    assert f() == 1


def test_ignore_returns_default_on_error():
    @ignore(ValueError, default=42)
    def f():
        raise ValueError("x")
    assert f() == 42


def test_ignore_default_none():
    @ignore(ValueError)
    def f():
        raise ValueError("x")
    assert f() is None


def test_ignore_does_not_catch_other_errors():
    @ignore(ValueError)
    def f():
        raise TypeError("x")
    with pytest.raises(TypeError):
        f()


def test_ignore_accepts_tuple():
    @ignore((ValueError, TypeError), default="d")
    def f():
        raise TypeError
    assert f() == "d"


def test_ignore_accepts_list():
    @ignore([ValueError, TypeError], default="d")
    def f():
        raise TypeError
    assert f() == "d"


def test_ignore_preserves_name():
    @ignore(ValueError)
    def my_func():
        return 1
    assert my_func.__name__ == "my_func"


def test_silent_catches_everything():
    @silent
    def f():
        raise KeyError("x")
    assert f() is None


def test_silent_passes_args():
    @silent
    def f(a, b):
        return a + b
    assert f(1, 2) == 3


# ---------------- reraise ----------------

def test_reraise_passthrough():
    with reraise(ValueError, TypeError):
        pass  # no error, nothing happens


def test_reraise_converts_error():
    with pytest.raises(TypeError):
        with reraise(ValueError, TypeError):
            raise ValueError("x")


def test_reraise_chains_cause():
    with pytest.raises(TypeError) as ei:
        with reraise(ValueError, TypeError):
            raise ValueError("x")
    assert isinstance(ei.value.__cause__, ValueError)


def test_reraise_callable_into():
    with pytest.raises(TypeError) as ei:
        with reraise(ValueError, lambda e: TypeError("converted: %s" % e)):
            raise ValueError("orig")
    assert str(ei.value) == "converted: orig"


def test_reraise_does_not_catch_unmatched():
    with pytest.raises(KeyError):
        with reraise(ValueError, TypeError):
            raise KeyError("x")


def test_reraise_accepts_tuple_of_errors():
    with pytest.raises(TypeError):
        with reraise((ValueError, KeyError), TypeError):
            raise KeyError("x")


# ---------------- retry ----------------

def test_retry_returns_on_first_success():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        return "ok"

    assert f() == "ok"
    assert calls == [1]


def test_retry_succeeds_after_failures():
    state = {"n": 0}

    @retry(3)
    def f():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("fail")
        return "done"

    assert f() == "done"
    assert state["n"] == 3


def test_retry_raises_after_all_attempts():
    state = {"n": 0}

    @retry(3)
    def f():
        state["n"] += 1
        raise ValueError("fail")

    with pytest.raises(ValueError):
        f()
    assert state["n"] == 3


def test_retry_only_catches_given_errors():
    state = {"n": 0}

    @retry(5, errors=ValueError)
    def f():
        state["n"] += 1
        raise TypeError("nope")

    with pytest.raises(TypeError):
        f()
    assert state["n"] == 1


def test_retry_filter_errors_stops_retry():
    state = {"n": 0}

    @retry(5, filter_errors=lambda e: False)
    def f():
        state["n"] += 1
        raise ValueError("x")

    with pytest.raises(ValueError):
        f()
    assert state["n"] == 1


def test_retry_filter_errors_allows_retry():
    state = {"n": 0}

    @retry(3, filter_errors=lambda e: True)
    def f():
        state["n"] += 1
        raise ValueError("x")

    with pytest.raises(ValueError):
        f()
    assert state["n"] == 3


def test_retry_timeout_sleeps(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    @retry(3, timeout=0.1)
    def f():
        raise ValueError("x")

    with pytest.raises(ValueError):
        f()
    assert sleeps == [0.1, 0.1]


def test_retry_timeout_callable(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    @retry(3, timeout=lambda a: (a + 1) * 0.01)
    def f():
        raise ValueError("x")

    with pytest.raises(ValueError):
        f()
    assert sleeps == [0.01, 0.02]


def test_retry_passes_arguments():
    @retry(2)
    def f(a, b):
        return a + b
    assert f(1, 2) == 3


# ---------------- fallback ----------------

def test_fallback_first_works():
    assert fallback(lambda: 1, lambda: 2) == 1


def test_fallback_falls_through():
    def bad():
        raise ValueError
    assert fallback(bad, lambda: 2) == 2


def test_fallback_with_error_tuple():
    def bad():
        raise ValueError

    def bad2():
        raise KeyError

    assert fallback((bad, ValueError), (bad2, KeyError), lambda: 3) == 3


def test_fallback_reraises_when_all_fail():
    def bad():
        raise ValueError("x")

    def bad2():
        raise KeyError("y")

    with pytest.raises(KeyError):
        fallback(bad, bad2)


def test_fallback_does_not_catch_unexpected():
    def bad():
        raise TypeError

    with pytest.raises(TypeError):
        fallback((bad, ValueError))


# ---------------- limit_error_rate ----------------

def test_limit_error_rate_blocks_after_fails():
    state = {"n": 0}

    @limit_error_rate(2, 100)
    def f():
        state["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    # now blocked
    with pytest.raises(ErrorRateExceeded):
        f()
    # function not called while blocked
    assert state["n"] == 2


def test_limit_error_rate_resets_on_success():
    state = {"n": 0}

    @limit_error_rate(2, 100)
    def f():
        state["n"] += 1
        if state["n"] == 1:
            raise ValueError
        return "ok"

    with pytest.raises(ValueError):
        f()
    assert f() == "ok"
    assert f.wrapper if False else f.fails == 0


def test_limit_error_rate_not_blocked_before_threshold():
    state = {"n": 0}

    @limit_error_rate(3, 100)
    def f():
        state["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()


def test_limit_error_rate_unblocks_after_timeout():
    state = {"n": 0}

    @limit_error_rate(1, timedelta(seconds=10))
    def f():
        state["n"] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()
    # Manually simulate passing of time
    f.blocked = datetime.now() - timedelta(seconds=20)
    with pytest.raises(ValueError):
        f()
    assert state["n"] == 2


def test_limit_error_rate_success_passes_through():
    @limit_error_rate(2, 100)
    def f(a):
        return a * 2

    assert f(3) == 6


def test_limit_error_rate_custom_exception():
    @limit_error_rate(1, 100, exception=KeyError)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(KeyError):
        f()


# ---------------- throttle ----------------

def test_throttle_first_call_runs():
    @throttle(10)
    def f():
        return "x"
    assert f() == "x"


def test_throttle_second_call_skipped():
    state = {"n": 0}

    @throttle(10)
    def f():
        state["n"] += 1
        return state["n"]

    assert f() == 1
    assert f() is None
    assert state["n"] == 1


def test_throttle_with_timedelta():
    @throttle(timedelta(seconds=10))
    def f():
        return "y"
    assert f() == "y"
    assert f() is None


# ---------------- post_processing / collecting / joining ----------------

def test_post_processing_applies_func():
    @post_processing(lambda x: x * 10)
    def f():
        return 5
    assert f() == 50


def test_collecting_turns_generator_into_list():
    @collecting
    def g():
        yield 1
        yield 2
        yield 3
    assert g() == [1, 2, 3]


def test_collecting_name_and_doc():
    assert collecting.__name__ == "collecting"
    assert "generator" in collecting.__doc__


def test_joining_joins_strings():
    @joining("-")
    def g():
        yield "a"
        yield "b"
        yield "c"
    assert g() == "a-b-c"


def test_joining_joins_non_strings_via_class():
    @joining("-")
    def g():
        yield 1
        yield 2
    # sep.__class__ is str so ints are converted via str
    assert g() == "1-2"


# ---------------- once / once_per / once_per_args ----------------

def test_once_runs_only_once():
    calls = []

    @once
    def f():
        calls.append(1)
        return "ok"

    assert f() == "ok"
    assert f() is None
    assert f() is None
    assert calls == [1]


def test_once_passes_args():
    @once
    def f(a, b):
        return a + b
    assert f(1, 2) == 3
    assert f(3, 4) is None


def test_once_per_called_once_per_arg_combo():
    calls = []

    @once_per("a")
    def f(a, b):
        calls.append((a, b))
        return a + b

    assert f(1, 2) == 3
    assert f(1, 5) is None  # same a, skipped
    assert f(2, 5) == 7
    assert calls == [(1, 2), (2, 5)]


def test_once_per_multiple_args():
    calls = []

    @once_per("a", "b")
    def f(a, b):
        calls.append((a, b))
        return a + b

    assert f(1, 2) == 3
    assert f(1, 2) is None
    assert f(1, 3) == 4
    assert calls == [(1, 2), (1, 3)]


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


def test_once_per_args_with_kwargs():
    calls = []

    @once_per_args
    def f(a, b=10):
        calls.append((a, b))
        return a + b

    assert f(1) == 11
    assert f(1) is None
    assert f(1, b=10) is None
    assert f(1, b=20) == 21
    assert calls == [(1, 10), (1, 20)]


# ---------------- wrap_with ----------------

def test_wrap_with_enters_context():
    log = []

    class CM:
        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *exc):
            log.append("exit")

    @wrap_with(CM())
    def f():
        return "val"

    assert f() == "val"
    assert log == ["enter", "exit"]
