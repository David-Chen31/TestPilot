import time
import threading
from datetime import timedelta

import pytest

from funcy.flow import (
    raiser, ignore, silent, suppress, nullcontext, reraise, retry, fallback,
    limit_error_rate, ErrorRateExceeded, throttle,
    post_processing, collecting, joining,
    once, once_per, once_per_args,
    wrap_with,
)


# --------------------------------------------------------------------------
# raiser
# --------------------------------------------------------------------------
def test_raiser_raises_exception_class():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_with_string_message():
    f = raiser("boom")
    with pytest.raises(Exception) as exc_info:
        f()
    assert str(exc_info.value) == "boom"


def test_raiser_with_instance():
    exc = KeyError("k")
    f = raiser(exc)
    with pytest.raises(KeyError) as exc_info:
        f()
    assert exc_info.value is exc


def test_raiser_with_args_kwargs():
    f = raiser(ValueError, "a", "b", c=1)
    with pytest.raises(ValueError) as exc_info:
        f()
    assert exc_info.value.args == ("a", "b")
    assert exc_info.value.kwargs == {"c": 1}


def test_raiser_ignores_call_args():
    f = raiser(TypeError)
    with pytest.raises(TypeError):
        f(1, 2, x=3)


# --------------------------------------------------------------------------
# ignore / silent
# --------------------------------------------------------------------------
def test_ignore_returns_default_on_matching_error():
    @ignore(ValueError, default="dflt")
    def f():
        raise ValueError

    assert f() == "dflt"


def test_ignore_default_is_none():
    @ignore(ValueError)
    def f():
        raise ValueError

    assert f() is None


def test_ignore_passes_through_on_non_matching():
    @ignore(ValueError)
    def f():
        raise KeyError("k")

    with pytest.raises(KeyError):
        f()


def test_ignore_returns_result_normally():
    @ignore(ValueError)
    def f(x):
        return x * 2

    assert f(3) == 6


def test_ignore_accepts_tuple_errors():
    @ignore((ValueError, KeyError), default=0)
    def f(e):
        raise e

    assert f(ValueError()) == 0
    assert f(KeyError()) == 0


def test_ignore_accepts_generator_of_errors():
    @ignore(iter([ValueError, KeyError]), default=-1)
    def f(e):
        raise e

    assert f(ValueError()) == -1
    assert f(KeyError()) == -1


def test_ignore_preserves_name():
    @ignore(ValueError)
    def my_func():
        pass

    assert my_func.__name__ == "my_func"


def test_silent_ignores_all():
    @silent
    def f():
        raise RuntimeError("whatever")

    assert f() is None


def test_silent_returns_value():
    @silent
    def f():
        return 42

    assert f() == 42


# --------------------------------------------------------------------------
# reraise
# --------------------------------------------------------------------------
def test_reraise_converts_error():
    with pytest.raises(KeyError):
        with reraise(ValueError, KeyError):
            raise ValueError("orig")


def test_reraise_keeps_cause():
    with pytest.raises(KeyError) as exc_info:
        with reraise(ValueError, KeyError):
            raise ValueError("orig")
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_reraise_callable_into():
    with pytest.raises(KeyError) as exc_info:
        with reraise(ValueError, lambda e: KeyError(str(e))):
            raise ValueError("msg")
    assert exc_info.value.args == ("msg",)


def test_reraise_exception_instance_via_callable_type_not_called():
    # A class that's an exception type is used directly, not called with e
    with pytest.raises(KeyError):
        with reraise(ValueError, KeyError):
            raise ValueError


def test_reraise_non_matching_passes_through():
    with pytest.raises(RuntimeError):
        with reraise(ValueError, KeyError):
            raise RuntimeError("x")


def test_reraise_no_error():
    with reraise(ValueError, KeyError):
        pass


def test_reraise_accepts_tuple():
    with pytest.raises(KeyError):
        with reraise((ValueError, TypeError), KeyError):
            raise TypeError


# --------------------------------------------------------------------------
# retry
# --------------------------------------------------------------------------
def test_retry_succeeds_immediately():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        return "ok"

    assert f() == "ok"
    assert len(calls) == 1


def test_retry_eventually_succeeds():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError
        return "done"

    assert f() == "done"
    assert len(calls) == 3


def test_retry_reraises_on_last_attempt():
    calls = []

    @retry(2)
    def f():
        calls.append(1)
        raise ValueError("boom")

    with pytest.raises(ValueError) as exc_info:
        f()
    assert exc_info.value.args == ("boom",)
    assert len(calls) == 2


def test_retry_only_specified_errors():
    calls = []

    @retry(3, errors=ValueError)
    def f():
        calls.append(1)
        raise KeyError("no")

    with pytest.raises(KeyError):
        f()
    assert len(calls) == 1


def test_retry_filter_errors_true_retries():
    calls = []

    @retry(3, filter_errors=lambda e: True)
    def f():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError
        return "ok"

    assert f() == "ok"
    assert len(calls) == 2


def test_retry_filter_errors_false_reraises():
    calls = []

    @retry(3, filter_errors=lambda e: False)
    def f():
        calls.append(1)
        raise ValueError("stop")

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 1


def test_retry_timeout_callable(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    calls = []

    @retry(3, timeout=lambda attempt: attempt + 1)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError
        return "ok"

    assert f() == "ok"
    assert sleeps == [1, 2]


def test_retry_timeout_non_callable(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    calls = []

    @retry(2, timeout=0.5)
    def f():
        calls.append(1)
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert sleeps == [0.5]


def test_retry_zero_timeout_no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    calls = []

    @retry(3, timeout=0)
    def f():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError
        return 1

    assert f() == 1
    assert sleeps == []


# --------------------------------------------------------------------------
# fallback
# --------------------------------------------------------------------------
def test_fallback_first_succeeds():
    assert fallback(lambda: 1, lambda: 2) == 1


def test_fallback_second_after_failure():
    def bad():
        raise ValueError

    assert fallback(bad, lambda: 2) == 2


def test_fallback_with_error_tuples():
    def bad():
        raise ValueError

    result = fallback((bad, ValueError), lambda: "fallback")
    assert result == "fallback"


def test_fallback_non_matching_error_propagates():
    def bad():
        raise KeyError

    with pytest.raises(KeyError):
        fallback((bad, ValueError), lambda: "nope")


def test_fallback_all_fail_returns_none():
    def bad1():
        raise ValueError

    def bad2():
        raise ValueError

    assert fallback(bad1, bad2) is None


def test_fallback_empty_returns_none():
    assert fallback() is None


# --------------------------------------------------------------------------
# limit_error_rate
# --------------------------------------------------------------------------
def test_limit_error_rate_blocks_after_fails(monkeypatch):
    now = [1000000.0]

    class FakeDatetime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime.fromtimestamp(now[0])

    import funcy.flow as flow_mod
    monkeypatch.setattr(flow_mod, "datetime", FakeDatetime)

    calls = []

    @limit_error_rate(2, 100)
    def f():
        calls.append(1)
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    assert len(calls) == 2
    # Now blocked
    with pytest.raises(ErrorRateExceeded):
        f()
    assert len(calls) == 2


def test_limit_error_rate_unblocks_after_timeout(monkeypatch):
    now = [1000000.0]

    class FakeDatetime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime.fromtimestamp(now[0])

    import funcy.flow as flow_mod
    monkeypatch.setattr(flow_mod, "datetime", FakeDatetime)

    calls = []

    @limit_error_rate(1, 10)
    def f(should_fail=True):
        calls.append(1)
        if should_fail:
            raise ValueError
        return "ok"

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ErrorRateExceeded):
        f()
    # advance time past timeout
    now[0] += 20
    assert f(should_fail=False) == "ok"
    assert f(should_fail=False) == "ok"


def test_limit_error_rate_success_resets(monkeypatch):
    now = [0.0]

    class FakeDatetime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime.fromtimestamp(now[0])

    import funcy.flow as flow_mod
    monkeypatch.setattr(flow_mod, "datetime", FakeDatetime)

    outcomes = [False, True, False, False, True]

    @limit_error_rate(2, 100)
    def f():
        ok = outcomes.pop(0)
        if not ok:
            raise ValueError
        return "ok"

    with pytest.raises(ValueError):
        f()
    assert f() == "ok"
    with pytest.raises(ValueError):
        f()
    # one fail since reset, but not yet blocked
    assert f() == "ok"


def test_limit_error_rate_custom_exception(monkeypatch):
    now = [0.0]

    class FakeDatetime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime.fromtimestamp(now[0])

    import funcy.flow as flow_mod
    monkeypatch.setattr(flow_mod, "datetime", FakeDatetime)

    class MyErr(Exception):
        pass

    @limit_error_rate(1, 1000, exception=MyErr)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(MyErr):
        f()


def test_limit_error_rate_preserves_name():
    @limit_error_rate(1, 1)
    def named_func():
        pass

    assert named_func.__name__ == "named_func"


# --------------------------------------------------------------------------
# throttle
# --------------------------------------------------------------------------
def test_throttle_runs_once_within_period(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    calls = []

    @throttle(10)
    def f():
        calls.append(1)
        return len(calls)

    assert f() == 1
    assert f() is None
    clock[0] = 5
    assert f() is None
    clock[0] = 11
    assert f() == 2


def test_throttle_timedelta_period(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    calls = []

    @throttle(timedelta(seconds=10))
    def f():
        calls.append(1)

    f()
    clock[0] = 5
    f()
    assert len(calls) == 1
    clock[0] = 15
    f()
    assert len(calls) == 2


def test_throttle_preserves_name():
    @throttle(1)
    def my_func():
        pass

    assert my_func.__name__ == "my_func"


# --------------------------------------------------------------------------
# post_processing / collecting / joining
# --------------------------------------------------------------------------
def test_post_processing():
    @post_processing(sum)
    def f():
        return [1, 2, 3]

    assert f() == 6


def test_post_processing_named_decorator_direct():
    def f():
        return 5

    assert post_processing(lambda x: x * 2)(f)() == 10


def test_collecting():
    @collecting
    def gen():
        yield 1
        yield 2
        yield 3

    assert gen() == [1, 2, 3]


def test_collecting_name_and_doc():
    assert collecting.__name__ == "collecting"
    assert "generator" in collecting.__doc__


def test_joining():
    @joining(", ")
    def f():
        return [1, 2, 3]

    assert f() == "1, 2, 3"


def test_joining_empty():
    @joining("-")
    def f():
        return []

    assert f() == ""


def test_joining_non_str_items_converted():
    @joining("|")
    def f():
        return [1, 2]

    assert f() == "1|2"


# --------------------------------------------------------------------------
# once / once_per / once_per_args
# --------------------------------------------------------------------------
def test_once_only_runs_once():
    calls = []

    @once
    def f():
        calls.append(1)
        return len(calls)

    assert f() == 1
    assert f() is None
    assert f() is None
    assert len(calls) == 1


def test_once_doc():
    assert "execute once" in once.__doc__


def test_once_per_argnames():
    calls = []

    @once_per("a")
    def f(a, b):
        calls.append((a, b))
        return (a, b)

    assert f(1, 2) == (1, 2)
    assert f(1, 3) is None
    assert f(2, 3) == (2, 3)
    assert f(2, 4) is None


def test_once_per_multiple_argnames():
    calls = []

    @once_per("a", "b")
    def f(a, b, c=0):
        calls.append((a, b, c))
        return (a, b, c)

    assert f(1, 2) == (1, 2, 0)
    assert f(1, 2, 5) is None
    assert f(1, 3) == (1, 3, 0)
    assert f(2, 2) == (2, 2, 0)


def test_once_per_kwargs():
    @once_per("a")
    def f(a):
        return a

    assert f(a=1) == 1
    assert f(a=1) is None
    assert f(a=2) == 2


def test_once_per_unhashable_arg():
    calls = []

    @once_per("a")
    def f(a):
        calls.append(a)
        return len(calls)

    assert f([1, 2]) == 1
    assert f([1, 2]) is None
    assert f([3, 4]) == 2
    assert f([3, 4]) is None


def test_once_per_args():
    calls = []

    @once_per_args
    def f(a, b):
        calls.append((a, b))
        return (a, b)

    assert f(1, 2) == (1, 2)
    assert f(1, 2) is None
    assert f(1, 3) == (1, 3)
    assert f(2, 2) == (2, 2)


def test_once_per_args_with_kwargs():
    @once_per_args
    def f(a, b, c=1):
        return (a, b, c)

    assert f(1, 2) == (1, 2, 1)
    assert f(1, 2) is None
    assert f(1, 2, c=3) == (1, 2, 3)


def test_once_per_args_varargs():
    @once_per_args
    def f(*args):
        return args

    assert f(1, 2) == (1, 2)
    assert f(1, 2) is None
    assert f(1, 2, 3) == (1, 2, 3)


def test_once_per_missing_arg_raises():
    @once_per("missing")
    def f(a):
        return a

    with pytest.raises(TypeError):
        f(1)


# --------------------------------------------------------------------------
# wrap_with
# --------------------------------------------------------------------------
def test_wrap_with_enter_used():
    entered = []

    class Ctx:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, *args):
            entered.append("exit")

    @wrap_with(Ctx())
    def f():
        return "value"

    assert f() == "value"
    assert entered == [True, "exit"]


def test_wrap_with_exception_propagates_and_exits():
    events = []

    class Ctx:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append(exc_type)
            return False

    @wrap_with(Ctx())
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    assert events == ["enter", ValueError]


# --------------------------------------------------------------------------
# misc reexports
# --------------------------------------------------------------------------
def test_nullcontext():
    with nullcontext("val") as v:
        assert v == "val"


def test_nullcontext_default_none():
    with nullcontext() as v:
        assert v is None


def test_suppress_reexported():
    with suppress(ValueError):
        raise ValueError
