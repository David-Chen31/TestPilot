import contextlib
import threading
import time

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
    nullcontext,
    suppress,
)


# ---------------------------------------------------------------------------
# raiser
# ---------------------------------------------------------------------------

def test_raiser_raises_exception_instance():
    f = raiser(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        f()


def test_raiser_raises_exception_class():
    f = raiser(KeyError)
    with pytest.raises(KeyError):
        f()


def test_raiser_with_string_message():
    f = raiser("bad thing")
    with pytest.raises(Exception, match="bad thing"):
        f()


def test_raiser_with_args():
    f = raiser(ValueError, "hello", 42)
    with pytest.raises(ValueError) as excinfo:
        f()
    assert excinfo.value.args == ("hello", 42)


def test_raiser_with_kwargs():
    f = raiser(ValueError, foo="bar")
    with pytest.raises(ValueError) as excinfo:
        f()
    # ValueError("foo", "bar") -- kwargs converted to positional-style args
    assert excinfo.value.args == ("foo", "bar")


def test_raiser_accepts_arbitrary_invocation_args():
    f = raiser(RuntimeError)
    with pytest.raises(RuntimeError):
        f(1, 2, x=3)


# ---------------------------------------------------------------------------
# ignore / silent
# ---------------------------------------------------------------------------

def test_ignore_returns_value_on_success():
    @ignore(ZeroDivisionError)
    def f(x):
        return 100 // x

    assert f(2) == 50


def test_ignore_swallows_and_returns_default():
    @ignore(ZeroDivisionError)
    def f(x):
        return 100 // x

    assert f(0) is None


def test_ignore_custom_default():
    @ignore(ZeroDivisionError, default="oops")
    def f(x):
        return 100 // x

    assert f(0) == "oops"
    assert f(4) == 25


def test_ignore_does_not_catch_unlisted_errors():
    @ignore(ZeroDivisionError)
    def f():
        raise ValueError("not ignored")

    with pytest.raises(ValueError):
        f()


def test_ignore_accepts_tuple_of_errors():
    @ignore((ZeroDivisionError, ValueError), default=-1)
    def f(kind):
        if kind == "div":
            return 1 / 0
        if kind == "val":
            raise ValueError
        return 7

    assert f("div") == -1
    assert f("val") == -1
    assert f("ok") == 7


def test_ignore_accepts_list_of_errors():
    @ignore([ZeroDivisionError], default=0)
    def f():
        return 1 / 0

    assert f() == 0


def test_ignore_preserves_wraps_metadata():
    @ignore(Exception)
    def original_func():
        """docstring here"""
        return 1

    assert original_func.__name__ == "original_func"
    assert original_func.__doc__ == "docstring here"


def test_silent_catches_all_exceptions():
    @silent
    def f(x):
        if x < 0:
            raise RuntimeError
        return x * 2

    assert f(3) == 6
    assert f(-1) is None


def test_silent_does_not_catch_base_exception():
    @silent
    def f():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        f()


# ---------------------------------------------------------------------------
# reraise
# ---------------------------------------------------------------------------

def test_reraise_reraises_as_class():
    with pytest.raises(ValueError):
        with reraise(KeyError, ValueError):
            raise KeyError("orig")


def test_reraise_chains_from_original():
    with pytest.raises(ValueError) as excinfo:
        with reraise(KeyError, ValueError):
            raise KeyError("orig")
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_reraise_with_callable_factory():
    with pytest.raises(ValueError, match="got KeyError"):
        with reraise(KeyError, lambda e: ValueError("got %s" % type(e).__name__)):
            raise KeyError


def test_reraise_with_exception_instance():
    inst = ValueError("instantiated")
    with pytest.raises(ValueError) as excinfo:
        with reraise(KeyError, inst):
            raise KeyError
    assert excinfo.value is inst


def test_reraise_tuple_of_errors():
    with pytest.raises(ValueError):
        with reraise((KeyError, IndexError), ValueError):
            raise IndexError


def test_reraise_passes_through_non_matching():
    with pytest.raises(TypeError):
        with reraise(KeyError, ValueError):
            raise TypeError("passthrough")


def test_reraise_no_error_does_nothing():
    with reraise(KeyError, ValueError):
        pass


def test_reraise_exception_class_with_callable_intent_is_not_called():
    # ValueError is itself callable but is an exception type; must not be called with e
    with pytest.raises(ValueError) as excinfo:
        with reraise(KeyError, ValueError):
            raise KeyError("x")
    assert excinfo.value.__cause__ is not None


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

def test_retry_succeeds_first_time():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        return "ok"

    assert f() == "ok"
    assert len(calls) == 1


def test_retry_retries_until_success():
    calls = []

    @retry(5)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError
        return "done"

    assert f() == "done"
    assert len(calls) == 3


def test_retry_raises_after_tries():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        raise ValueError("fail")

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 3


def test_retry_only_on_specified_errors():
    calls = []

    @retry(5, errors=KeyError)
    def f():
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 1


def test_retry_timeout_once(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    calls = []

    @retry(3, timeout=0.5)
    def f():
        calls.append(1)
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 3
    assert slept == [0.5, 0.5]


def test_retry_timeout_callable(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    @retry(3, timeout=lambda attempt: attempt + 1)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert slept == [1, 2]


def test_retry_filter_errors_blocks_retry():
    calls = []

    @retry(5, filter_errors=lambda e: False)
    def f():
        calls.append(1)
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 1


def test_retry_filter_errors_returns_true_retries():
    calls = []

    @retry(5, filter_errors=lambda e: True)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError
        return "x"

    assert f() == "x"
    assert len(calls) == 3


def test_retry_no_sleep_when_timeout_zero(monkeypatch):
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

def test_fallback_first_succeeds():
    assert fallback(lambda: 1, lambda: 2) == 1


def test_fallback_passes_single_callables():
    def bad():
        raise ValueError

    assert fallback(bad, lambda: "second") == "second"


def test_fallback_with_tuples_catch():
    def bad():
        raise KeyError

    assert fallback((bad, KeyError), lambda: "fallback") == "fallback"


def test_fallback_skips_unexpected_exceptions_are_not_caught():
    def bad():
        raise KeyError

    with pytest.raises(KeyError):
        fallback((bad, ValueError), lambda: "never")


def test_fallback_all_fail_returns_none():
    def bad1():
        raise ValueError

    def bad2():
        raise KeyError

    assert fallback(bad1, bad2) is None


def test_fallback_reraises_type_error_of_last_when_mutated_tuple_condition():
    # any non-matching error is caught by fallback since default catch is Exception
    calls = []

    def bad():
        calls.append("bad")
        raise RuntimeError

    def good():
        calls.append("good")
        return 42

    assert fallback(bad, good) == 42
    assert calls == ["bad", "good"]


def test_fallback_with_list_of_exceptions():
    def bad():
        raise TypeError

    assert fallback((bad, [TypeError, ValueError]), lambda: "ok") == "ok"


# ---------------------------------------------------------------------------
# limit_error_rate
# ---------------------------------------------------------------------------

def test_limit_error_rate_raises_after_limit():
    @limit_error_rate(2, 60)
    def f():
        raise ValueError("x")

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    # now blocked
    with pytest.raises(ErrorRateExceeded):
        f()


def test_limit_error_rate_custom_exception():
    class Custom(Exception):
        pass

    @limit_error_rate(1, 60, exception=Custom)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(Custom):
        f()


def test_limit_error_rate_success_resets():
    out = []

    @limit_error_rate(2, 60)
    def f(fail):
        if fail:
            raise ValueError
        out.append("ok")
        return "ok"

    with pytest.raises(ValueError):
        f(True)
    assert f(False) == "ok"
    # fails got reset; can fail again without immediate block until threshold
    with pytest.raises(ValueError):
        f(True)
    with pytest.raises(ValueError):
        f(True)
    with pytest.raises(ErrorRateExceeded):
        f(True)


def test_limit_error_rate_uses_timedelta():
    from datetime import timedelta

    @limit_error_rate(1, timedelta(seconds=60))
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()


def test_limit_error_rate_unblocks_after_timeout(monkeypatch):
    from datetime import datetime as real_datetime
    import funcy.flow as flow_mod

    class FakeDatetime(real_datetime):
        now_value = real_datetime(2020, 1, 1, 0, 0, 0)
        calls = 0

        @classmethod
        def now(cls):
            # each now() call approximates a lot of passed time
            cls.calls += 1
            if cls.calls == 1:
                return cls.now_value
            return cls.now_value + flow_mod.timedelta(seconds=1000)

    monkeypatch.setattr(flow_mod, "datetime", FakeDatetime)

    @limit_error_rate(1, 10)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    # blocked time recorded; next call's now() is far in future so unblocks
    calls = []

    def g():  # different function
        pass

    with pytest.raises(ValueError):
        f()


# ---------------------------------------------------------------------------
# throttle
# ---------------------------------------------------------------------------

def test_throttle_allows_first_then_blocks(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    calls = []

    @throttle(10)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(2) is None
    now[0] += 5
    assert f(3) is None
    now[0] += 10  # past blocked_until = 1010
    assert f(4) == 4
    assert calls == [1, 4]


def test_throttle_with_timedelta(monkeypatch):
    from datetime import timedelta

    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    @throttle(timedelta(seconds=100))
    def f():
        return "ran"

    assert f() == "ran"
    assert f() is None
    now[0] = 100
    assert f() == "ran"


def test_throttle_does_not_call_when_blocked(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    calls = []

    @throttle(5)
    def f():
        calls.append(1)
        return len(calls)

    assert f() == 1
    assert f() is None
    assert f() is None
    assert calls == [1]


# ---------------------------------------------------------------------------
# post_processing / collecting / joining
# ---------------------------------------------------------------------------

def test_post_processing():
    @post_processing(lambda x: x * 10)
    def f():
        return 4

    assert f() == 40


def test_post_processing_receives_result():
    @post_processing(sorted)
    def f():
        return [3, 1, 2]

    assert f() == [1, 2, 3]


def test_collecting():
    @collecting
    def gen():
        yield 1
        yield 2
        yield 3

    assert gen() == [1, 2, 3]


def test_collecting_metadata():
    assert collecting.__name__ == "collecting"


def test_joining():
    @joining("-")
    def gen():
        yield 1
        yield 2
        yield 3

    assert gen() == "1-2-3"


def test_joining_strings():
    @joining(",")
    def gen():
        yield "a"
        yield "b"

    assert gen() == "a,b"


def test_joining_empty():
    @joining("")
    def gen():
        return []

    assert gen() == ""


# ---------------------------------------------------------------------------
# once / once_per / once_per_args
# ---------------------------------------------------------------------------

def test_once_calls_only_once():
    calls = []

    @once
    def f():
        calls.append(1)
        return "value"

    assert f() == "value"
    assert f() is None
    assert f() is None
    assert len(calls) == 1


def test_once_per_no_args_once():
    calls = []

    @once_per()
    def f():
        calls.append(1)
        return 1

    assert f() == 1
    assert f() is None
    assert len(calls) == 1


def test_once_per_different_arg_values():
    calls = []

    @once_per("x")
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(2) == 4
    assert f(1) is None
    assert f(2) is None
    assert calls == [1, 2]


def test_once_per_kwargs_and_pos():
    calls = []

    @once_per("a", "b")
    def f(a, b):
        calls.append((a, b))
        return a + b

    assert f(1, 2) == 3
    assert f(a=1, b=2) is None
    assert f(2, 1) == 3
    assert calls == [(1, 2), (2, 1)]


def test_once_per_unhashable_args():
    calls = []

    @once_per("x")
    def f(x):
        calls.append(x)
        return "done"

    assert f([1, 2]) == "done"
    assert f([1, 2]) is None
    assert f([3, 4]) == "done"
    assert calls == [[1, 2], [3, 4]]


def test_once_per_default_argument():
    calls = []

    @once_per("x")
    def f(x=5):
        calls.append(x)
        return x

    assert f() == 5
    assert f(5) is None
    assert calls == [5]


def test_once_per_missing_arg_raises_type_error():
    @once_per("missing")
    def f(x):
        return x

    with pytest.raises(TypeError):
        f(1)


def test_once_per_thread_safety():
    calls = []

    @once_per()
    def f():
        calls.append(1)
        return 1

    results = []

    def worker():
        results.append(f())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert results.count(1) == 1
    assert results.count(None) == 9


def test_once_per_args():
    calls = []

    @once_per_args
    def f(a, b=2):
        calls.append((a, b))
        return a + b

    assert f(1) == 3
    assert f(1, 2) is None
    assert f(1, 3) == 4
    assert calls == [(1, 2), (1, 3)]


def test_once_per_args_varargs():
    calls = []

    @once_per_args
    def f(*args):
        calls.append(args)
        return sum(args)

    assert f(1, 2) == 3
    assert f(1, 2) is None
    assert f(2, 1) == 3
    assert calls == [(1, 2), (2, 1)]


# ---------------------------------------------------------------------------
# wrap_with
# ---------------------------------------------------------------------------

def test_wrap_with_enters_context():
    entered = []
    exited = []

    @contextlib.contextmanager
    def cm():
        entered.append(True)
        try:
            yield "ctx-value"
        finally:
            exited.append(True)

    @wrap_with(cm())
    def f():
        return "result"

    assert f() == "result"
    assert entered == [True]
    assert exited == [True]


def test_wrap_with_passes_exception():
    exited = []

    @contextlib.contextmanager
    def cm():
        try:
            yield
        except ValueError:
            exited.append("caught")
            raise
        finally:
            exited.append("finally")

    @wrap_with(cm())
    def f():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        f()
    assert exited == ["caught", "finally"]


def test_wrap_with_suppressing_context():
    @wrap_with(contextlib.suppress(ValueError))
    def f():
        raise ValueError

    assert f() is None


# ---------------------------------------------------------------------------
# nullcontext (reexport) sanity
# ---------------------------------------------------------------------------

def test_nullcontext_enter_result():
    with nullcontext("x") as v:
        assert v == "x"
