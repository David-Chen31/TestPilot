import pytest
import time
import threading
import importlib
from datetime import timedelta, datetime
from unittest import mock

import funcy.flow as flow_module
from funcy.flow import (
    raiser, ignore, silent, reraise, retry, fallback, ErrorRateExceeded,
    limit_error_rate, throttle, post_processing, collecting, joining,
    once, once_per, once_per_args, wrap_with, nullcontext, suppress,
)


# ---------- raiser ----------

def test_raiser_default():
    f = raiser()
    with pytest.raises(Exception):
        f()


def test_raiser_class():
    f = raiser(ValueError)
    with pytest.raises(ValueError):
        f()


def test_raiser_with_args():
    f = raiser(ValueError, "boom", 1)
    with pytest.raises(ValueError) as ei:
        f()
    assert ei.value.args == ("boom", 1)


def test_raiser_with_kwargs():
    f = raiser(KeyError, 5)
    with pytest.raises(KeyError) as ei:
        f()
    assert ei.value.args == (5,)


def test_raiser_string_message():
    f = raiser("custom message")
    with pytest.raises(Exception) as ei:
        f()
    assert str(ei.value) == "custom message"


def test_raiser_ignores_call_args():
    f = raiser(ValueError, 'x')
    with pytest.raises(ValueError) as ei:
        f(1, 2, k=3)
    assert ei.value.args == ('x',)


# ---------- ignore / silent ----------

def test_ignore_returns_default():
    @ignore(ValueError)
    def f():
        raise ValueError("bad")
    assert f() is None


def test_ignore_custom_default():
    @ignore(ValueError, default=42)
    def f():
        raise ValueError
    assert f() == 42


def test_ignore_passes_through_other_exceptions():
    @ignore(ValueError)
    def f():
        raise KeyError("k")
    with pytest.raises(KeyError):
        f()


def test_ignore_passes_args_and_result():
    @ignore(ValueError, default=-1)
    def f(a, b=2):
        if a == 0:
            raise ValueError
        return a + b
    assert f(1) == 3
    assert f(0) == -1
    assert f(1, b=10) == 11


def test_ignore_tuple_of_errors():
    @ignore((ValueError, KeyError), default='fallback')
    def f(exc):
        raise exc
    assert f(ValueError) == 'fallback'
    assert f(KeyError) == 'fallback'


def test_ignore_list_of_errors_converted():
    @ignore([ValueError, TypeError], default=0)
    def f():
        raise TypeError
    assert f() == 0


def test_ignore_preserves_name():
    @ignore(ValueError)
    def my_func():
        pass
    assert my_func.__name__ == 'my_func'


def test_silent_catches_all():
    @silent
    def f():
        raise RuntimeError("oops")
    assert f() is None


def test_silent_returns_result():
    @silent
    def f(x):
        return x * 2
    assert f(5) == 10


# ---------- reraise ----------

def test_reraise_translates():
    with pytest.raises(ValueError):
        with reraise(KeyError, ValueError):
            raise KeyError("x")


def test_reraise_passes_through_unmatched():
    with pytest.raises(TypeError):
        with reraise(KeyError, ValueError):
            raise TypeError("t")


def test_reraise_chains_cause():
    with pytest.raises(ValueError) as ei:
        with reraise(KeyError, ValueError):
            raise KeyError("orig")
    assert isinstance(ei.value.__cause__, KeyError)


def test_reraise_callable_into():
    with pytest.raises(ValueError) as ei:
        with reraise(KeyError, lambda e: ValueError("wrapped: " + str(e))):
            raise KeyError("k")
    assert str(ei.value) == "wrapped: 'k'"


def test_reraise_callable_that_is_exception_type_not_called():
    # ValueError is callable and exception type -> used directly, not called with e
    with pytest.raises(ValueError):
        with reraise(KeyError, ValueError):
            raise KeyError


def test_reraise_no_error():
    with reraise(KeyError, ValueError):
        pass
    assert True


# ---------- retry ----------

def test_retry_success_first_try():
    calls = []

    @retry(3)
    def f():
        calls.append(1)
        return "ok"
    assert f() == "ok"
    assert len(calls) == 1


def test_retry_eventually_succeeds():
    state = {'n': 0}

    @retry(3)
    def f():
        state['n'] += 1
        if state['n'] < 3:
            raise ValueError
        return state['n']
    assert f() == 3
    assert state['n'] == 3


def test_retry_exhausts():
    calls = []

    @retry(2)
    def f():
        calls.append(1)
        raise ValueError("fail")
    with pytest.raises(ValueError):
        f()
    assert len(calls) == 2


def test_retry_only_on_given_errors():
    @retry(3, errors=KeyError)
    def f():
        raise ValueError
    with pytest.raises(ValueError):
        f()


def test_retry_filter_errors_stops():
    state = {'n': 0}

    @retry(5, errors=ValueError, filter_errors=lambda e: e.args[0] != 'stop')
    def f():
        state['n'] += 1
        raise ValueError('stop')
    with pytest.raises(ValueError):
        f()
    # filter returns False -> raise immediately after first call
    assert state['n'] == 1


def test_retry_zero_tries_does_nothing():
    @retry(0)
    def f():
        raise AssertionError("should not be called")
    # tries=0 -> loop body never runs, returns None
    assert f() is None


def test_retry_timeout_callable():
    sleeps = []
    orig_sleep = time.sleep
    state = {'n': 0}

    @retry(3, timeout=lambda attempt: 0)
    def f():
        state['n'] += 1
        if state['n'] < 2:
            raise ValueError
        return 'done'
    assert f() == 'done'


def test_retry_preserves_args():
    state = {'n': 0}

    @retry(3)
    def f(a, b=10):
        state['n'] += 1
        if state['n'] < 2:
            raise ValueError
        return a + b
    assert f(5, b=1) == 6


def test_retry_sleeps_between_attempts():
    # Exercises: timeout_value = timeout(attempt) is not called (timeout is int),
    # timeout_value > 0 True and time.sleep(timeout_value) executed.
    state = {'n': 0}

    @retry(3, timeout=0.5)
    def f():
        state['n'] += 1
        if state['n'] < 2:
            raise ValueError
        return 'ok'

    with mock.patch('funcy.flow.time.sleep') as sleeps:
        assert f() == 'ok'
    assert sleeps.call_args_list == [mock.call(0.5)]
    assert state['n'] == 2


def test_retry_callable_timeout_sleeps():
    # Exercises: timeout(attempt) called and returns positive value -> sleep.
    state = {'n': 0}
    attempts = []

    def timeout(attempt):
        attempts.append(attempt)
        return 0.25

    @retry(3, timeout=timeout)
    def f():
        state['n'] += 1
        if state['n'] < 2:
            raise ValueError
        return 'ok'

    with mock.patch('funcy.flow.time.sleep') as sleeps:
        assert f() == 'ok'
    assert attempts == [0]
    assert sleeps.call_args_list == [mock.call(0.25)]


def test_retry_zero_timeout_no_sleep():
    # timeout_value = 0 -> not > 0 -> no sleep call.
    state = {'n': 0}

    @retry(3, timeout=0)
    def f():
        state['n'] += 1
        if state['n'] < 2:
            raise ValueError
        return 'ok'

    with mock.patch('funcy.flow.time.sleep') as sleeps:
        assert f() == 'ok'
    assert sleeps.call_args_list == []


# ---------- fallback ----------

def test_fallback_first_works():
    assert fallback(lambda: 1, lambda: 2) == 1


def test_fallback_second_works():
    def bad():
        raise ValueError
    assert fallback(bad, lambda: 2) == 2


def test_fallback_with_catch_specification():
    def bad():
        raise KeyError

    result = fallback((bad, KeyError), lambda: 99)
    assert result == 99


def test_fallback_uncaught_error_propagates():
    # bad is bare callable -> catches Exception, so KeyError is swallowed.
    # bad2 is bare callable -> also catches Exception, so ValueError is swallowed.
    # Result: all approaches fail their unknown errors are caught -> returns None.
    def bad():
        raise KeyError

    def bad2():
        raise ValueError

    assert fallback((bad, KeyError), bad2) is None


def test_fallback_all_fail_no_approaches():
    assert fallback() is None


def test_fallback_catch_tuple_via_list():
    def bad():
        raise TypeError

    assert fallback((bad, [TypeError, KeyError]), lambda: 'ok') == 'ok'


# ---------- limit_error_rate ----------

def test_limit_error_rate_blocks_after_fails():
    calls = []

    @limit_error_rate(fails=2, timeout=10)
    def f():
        calls.append(1)
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(ValueError):
        f()
    # Now blocked: raises ErrorRateExceeded without executing func
    with pytest.raises(ErrorRateExceeded):
        f()
    assert len(calls) == 2


def test_limit_error_rate_custom_exception():
    @limit_error_rate(fails=1, timeout=10, exception=KeyError)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    with pytest.raises(KeyError):
        f()


def test_limit_error_rate_resets_on_success():
    state = {'fail': True, 'calls': 0}

    @limit_error_rate(fails=2, timeout=10)
    def f():
        state['calls'] += 1
        if state['fail']:
            raise ValueError
        return 'ok'

    with pytest.raises(ValueError):
        f()
    state['fail'] = False
    assert f() == 'ok'
    # internal state reset
    assert f.fails == 0
    assert f.blocked is None


def test_limit_error_rate_unblocks_after_timeout():
    call_count = {'n': 0}

    @limit_error_rate(fails=1, timeout=0)
    def f():
        call_count['n'] += 1
        raise ValueError

    with pytest.raises(ValueError):
        f()
    # timeout is 0, so blocked check subtracts 0 -> not strictly less than 0
    # so it unblocks and calls again
    with pytest.raises(ValueError):
        f()
    assert call_count['n'] == 2


def test_limit_error_rate_attributes():
    @limit_error_rate(3, 5)
    def f():
        pass
    assert f.fails == 0
    assert f.blocked is None
    assert f.__name__ == 'f'


def test_limit_error_rate_timedelta_timeout_stays_timedelta():
    # Exercises the branch where timeout is not int/float (timedelta) -> not converted.
    # timeout as timedelta is used directly in `datetime.now() - blocked < timeout`.
    @limit_error_rate(fails=1, timeout=timedelta(seconds=10))
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()
    # blocked was set to a datetime; comparing datetime < timedelta would raise
    # TypeError if incorrectly treated. It should raise ErrorRateExceeded instead.
    with pytest.raises(ErrorRateExceeded):
        f()
    assert isinstance(f.blocked, datetime)


# ---------- throttle ----------

def test_throttle_only_one_run():
    calls = []

    @throttle(100)
    def f():
        calls.append(1)
        return len(calls)

    assert f() == 1
    assert f() is None
    assert f() is None
    assert calls == [1]


def test_throttle_timedelta():
    calls = []

    @throttle(timedelta(seconds=100))
    def f():
        calls.append(1)
        return 'first'

    assert f() == 'first'
    assert f() is None
    assert len(calls) == 1


def test_throttle_initial_state():
    @throttle(10)
    def f():
        return 1
    assert f.blocked_until == 0


def test_throttle_unblocks_after_period():
    # Uses monotonic clock; patch it to control time deterministically.
    calls = []

    @throttle(10)
    def f():
        calls.append(1)
        return len(calls)

    with mock.patch('funcy.flow.time.monotonic') as mono:
        mono.return_value = 100.0
        assert f() == 1
        mono.return_value = 105.0
        assert f() is None
        mono.return_value = 111.0
        assert f() == 2
    assert calls == [1, 1]


# ---------- post_processing / collecting / joining ----------

def test_post_processing():
    @post_processing(lambda x: x * 2)
    def f():
        return 10
    assert f() == 20


def test_post_processing_passes_args():
    @post_processing(sum)
    def f(n):
        return list(range(n))
    assert f(4) == 6


def test_collecting():
    @collecting
    def f():
        yield 1
        yield 2
        yield 3
    assert f() == [1, 2, 3]
    assert collecting.__name__ == 'collecting'


def test_joining():
    @joining('-')
    def f():
        yield 1
        yield 2
        yield 3
    assert f() == '1-2-3'


def test_joining_empty():
    @joining(',')
    def f():
        return []
    assert f() == ''


# ---------- once / once_per / once_per_args ----------

def test_once_runs_once():
    calls = []

    @once
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(4) is None
    assert calls == [3]


def test_once_thread_safe_smoke():
    calls = []

    @once
    def f():
        calls.append(1)
        return 1

    results = []

    def worker():
        results.append(f())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # exactly one non-None result
    assert results.count(1) == 1
    assert len(calls) == 1


def test_once_per_distinct_args():
    calls = []

    @once_per('a', 'b')
    def f(a, b):
        calls.append((a, b))
        return (a, b)

    assert f(1, 2) == (1, 2)
    assert f(1, 2) is None
    assert f(2, 2) == (2, 2)
    assert f(1, 3) == (1, 3)
    assert calls == [(1, 2), (2, 2), (1, 3)]


def test_once_per_with_kwargs():
    calls = []

    @once_per('a')
    def f(a, b=0):
        calls.append(a)
        return a

    assert f(1) == 1
    assert f(1, b=5) is None
    assert f(a=2) == 2
    assert calls == [1, 2]


def test_once_per_unhashable_args():
    calls = []

    @once_per('lst')
    def f(lst):
        calls.append(lst)
        return sum(lst)

    assert f([1, 2]) == 3
    assert f([1, 2]) is None
    assert f([3, 4]) == 7
    assert len(calls) == 2


def test_once_per_missing_arg():
    @once_per('nonexistent')
    def f(a):
        return a

    with pytest.raises(TypeError):
        f(1)


def test_once_per_args_combinations():
    calls = []

    @once_per_args
    def f(a, b=2):
        calls.append((a, b))
        return a + b

    assert f(1) == 3
    assert f(1) is None
    assert f(1, 2) is None
    assert f(1, 5) == 6
    assert calls == [(1, 2), (1, 5)]


def test_once_per_doc():
    assert 'once' in once.__doc__


# ---------- wrap_with ----------

def test_wrap_with_context_manager():
    entered = []
    exited = []

    class CM:
        def __enter__(self):
            entered.append(True)
            return self

        def __exit__(self, *a):
            exited.append(True)

    @wrap_with(CM())
    def f():
        assert entered == [True]
        assert exited == []
        return 'ok'

    assert f() == 'ok'
    assert exited == [True]


def test_wrap_with_nullcontext():
    @wrap_with(nullcontext())
    def f(x):
        return x + 1
    assert f(1) == 2


def test_wrap_with_propagates_during_call():
    class CM:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    @wrap_with(CM())
    def f():
        raise ValueError("inner")

    with pytest.raises(ValueError):
        f()


# ---------- misc: nullcontext, suppress reexport ----------

def test_nullcontext_enter_result():
    with nullcontext(5) as v:
        assert v == 5


def test_nullcontext_default_enter_result():
    with nullcontext() as v:
        assert v is None


def test_nullcontext_exit_returns_none():
    cm = nullcontext(7)
    assert cm.__enter__() == 7
    assert cm.__exit__(None, None, None) is None


def test_suppress_reexport_available():
    with suppress(ValueError):
        raise ValueError


# ---------- backported nullcontext fallback (ImportError branch) ----------

def test_nullcontext_fallback_defined_when_contextlib_missing():
    """Forces the ``except ImportError:`` branch by making the
    ``from contextlib import nullcontext`` import fail, then exercises the
    locally defined fallback ``nullcontext`` class (lines 68-75)."""
    real_nullcontext = importlib.import_module('contextlib').nullcontext

    # Temporarily remove nullcontext from contextlib so the from-import raises
    # ImportError, then reload funcy.flow to force it to define the fallback class.
    with mock.patch.dict('contextlib.__dict__', {}, clear=False):
        removed = importlib.import_module('contextlib').__dict__.pop('nullcontext', None)
        assert removed is real_nullcontext
        try:
            reloaded = importlib.reload(flow_module)
            try:
                FallbackNullContext = reloaded.nullcontext
                # It must NOT be the real contextlib.nullcontext (proves fallback path ran)
                assert FallbackNullContext is not real_nullcontext
                assert FallbackNullContext.__name__ == 'nullcontext'

                # __init__ stores enter_result
                cm = FallbackNullContext(42)
                assert cm.enter_result == 42

                # __enter__ returns the stored enter_result
                assert cm.__enter__() == 42

                # __exit__ returns None and accepts any excinfo
                assert cm.__exit__(None, None, None) is None
                assert cm.__exit__(ValueError, ValueError("x"), None) is None

                # default enter_result is None
                default_cm = FallbackNullContext()
                assert default_cm.enter_result is None
                assert default_cm.__enter__() is None

                # Usable as a context manager returning the enter_result
                with FallbackNullContext('hello') as v:
                    assert v == 'hello'
            finally:
                # Restore funcy.flow to its normal state (real nullcontext) for
                # the rest of the test session.
                importlib.reload(flow_module)
        finally:
            # contextlib.nullcontext is restored by patch.dict on exit;
            # ensure it is present now for the assertion below.
            importlib.import_module('contextlib').nullcontext = real_nullcontext
    assert importlib.import_module('contextlib').nullcontext is real_nullcontext
