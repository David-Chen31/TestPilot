import asyncio
import logging
import operator

import pytest

import backon._retry._api as api_module
from backon._retry._api import (
    _retry_async,
    _retry_sync,
    retry,
)
from backon._retry import _api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_sleep(_):
    return None


async def _no_async_sleep(_):
    return None


# A trivial wait generator that yields deterministic values.
def _zero_wait_gen(**kwargs):
    while True:
        yield 0


def _make_failing_sync(fail_times, exc=ValueError):
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] <= fail_times:
            raise exc("boom")
        return "ok"

    fn.state = state
    return fn


def _make_failing_async(fail_times, exc=ValueError):
    state = {"n": 0}

    async def fn():
        state["n"] += 1
        if state["n"] <= fail_times:
            raise exc("boom")
        return "ok"

    fn.state = state
    return fn


# ---------------------------------------------------------------------------
# retry dispatch: sync vs async
# ---------------------------------------------------------------------------

def test_retry_dispatch_sync_returns_value():
    calls = {"n": 0}

    @retry(max_tries=3, sleep=_no_sleep, wait_gen=_zero_wait_gen)
    def fn():
        calls["n"] += 1
        return "hello"

    assert fn() == "hello"
    assert calls["n"] == 1


def test_retry_dispatch_async_returns_coroutine_run():
    calls = {"n": 0}

    @retry(max_tries=3, sleep=_no_async_sleep, wait_gen=_zero_wait_gen)
    async def fn():
        calls["n"] += 1
        return "hello-async"

    result = asyncio.run(fn())
    assert result == "hello-async"
    assert calls["n"] == 1


def test_retry_dispatch_async_retries_then_succeeds():
    fn = _make_failing_async(2)

    @retry(max_tries=5, sleep=_no_async_sleep, wait_gen=_zero_wait_gen)
    async def _wrapped():
        return await fn()

    assert asyncio.run(_wrapped()) == "ok"
    assert fn.state["n"] == 3


def test_retry_dispatch_sync_retries_then_succeeds():
    fn = _make_failing_sync(2)

    wrapped = retry(
        fn, max_tries=5, sleep=_no_sleep, wait_gen=_zero_wait_gen
    )
    assert wrapped() == "ok"
    assert fn.state["n"] == 3


def test_retry_sync_giveup_raises():
    fn = _make_failing_sync(100)

    wrapped = retry(
        fn, max_tries=3, sleep=_no_sleep, wait_gen=_zero_wait_gen
    )
    with pytest.raises(ValueError):
        wrapped()
    assert fn.state["n"] == 3


def test_retry_async_giveup_raises():
    fn = _make_failing_async(100)

    wrapped = retry(
        fn, max_tries=3, sleep=_no_async_sleep, wait_gen=_zero_wait_gen
    )
    with pytest.raises(ValueError):
        asyncio.run(wrapped())
    assert fn.state["n"] == 3


# ---------------------------------------------------------------------------
# is_enabled() early-return branch
# ---------------------------------------------------------------------------

def test_retry_sync_disabled_calls_target_once(monkeypatch):
    monkeypatch.setattr(api_module, "is_enabled", lambda: False)

    calls = {"n": 0}

    def fn(a, b):
        calls["n"] += 1
        return a + b

    wrapped = retry(fn, sleep=_no_sleep)
    assert wrapped(1, 2) == 3
    assert calls["n"] == 1


def test_retry_async_disabled_calls_target_once(monkeypatch):
    monkeypatch.setattr(api_module, "is_enabled", lambda: False)

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        return "disabled"

    wrapped = retry(fn, sleep=_no_async_sleep)
    assert asyncio.run(wrapped()) == "disabled"
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# _apply_test_overrides mapping
# ---------------------------------------------------------------------------

def test_retry_sync_max_tries_override_applied(monkeypatch):
    fn = _make_failing_sync(100)
    monkeypatch.setattr(
        _api,
        "_apply_test_overrides",
        lambda max_tries, max_time: (2, max_time),
    )

    wrapped = retry(fn, max_tries=10, sleep=_no_sleep, wait_gen=_zero_wait_gen)
    with pytest.raises(ValueError):
        wrapped()
    assert fn.state["n"] == 2


# ---------------------------------------------------------------------------
# Handler configuration: default handlers are built with the prepared logger
# ---------------------------------------------------------------------------

class _SpyLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg, args))


def test_retry_sync_prepares_logger_and_logs_backoff():
    logger = _SpyLogger()

    def fn():
        raise ValueError("nope")

    wrapped = retry(
        fn,
        max_tries=2,
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
        logger=logger,
        backoff_log_level=logging.WARNING,
        giveup_log_level=logging.ERROR,
    )
    with pytest.raises(ValueError):
        wrapped()

    # Should have logged backoff at least once with the backoff level, and
    # giveup at the giveup level.
    backoff_levels = [r[0] for r in logger.records if "Backing off" in r[1]]
    giveup_levels = [r[0] for r in logger.records if "Giving up" in r[1]]
    assert backoff_levels and all(l == logging.WARNING for l in backoff_levels)
    assert giveup_levels and all(l == logging.ERROR for l in giveup_levels)


def test_retry_sync_logger_string_resolved():
    # Passing a string logger should be resolved through logging.getLogger.
    wrapped = retry(
        _make_failing_sync(100),
        max_tries=1,
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
        logger="backon.tests.some.logger",
    )
    with pytest.raises(ValueError):
        wrapped()


def test_retry_sync_user_handlers_receive_events():
    events = {"attempt": 0, "backoff": 0, "giveup": 0, "success": 0}

    def on_attempt(details):
        events["attempt"] += 1

    def on_backoff(details):
        events["backoff"] += 1

    def on_giveup(details):
        events["giveup"] += 1

    def on_success(details):
        events["success"] += 1

    fn = _make_failing_sync(1)
    wrapped = retry(
        fn,
        max_tries=3,
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
        on_attempt=on_attempt,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
        on_success=on_success,
    )
    assert wrapped() == "ok"
    assert events["attempt"] == 2
    assert events["backoff"] == 1
    assert events["giveup"] == 0
    assert events["success"] == 1


def test_retry_sync_list_handlers():
    attempts = []

    def h1(details):
        attempts.append("h1")

    def h2(details):
        attempts.append("h2")

    wrapped = retry(
        lambda: "ok",
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
        on_attempt=[h1, h2],
    )
    assert wrapped() == "ok"
    assert attempts == ["h1", "h2"]


# ---------------------------------------------------------------------------
# Condition / stop construction paths
# ---------------------------------------------------------------------------

def test_retry_sync_explicit_condition_and_stop_used():
    from backon._conditions import RetryCondition, Stop

    class AlwaysRetry(RetryCondition):
        def __call__(self, state):
            return True

    class NeverStop(Stop):
        def __call__(self, state):
            return False

    condition_calls = {"n": 0}
    stop_calls = {"n": 0}

    class CountingCondition(AlwaysRetry):
        def __call__(self, state):
            condition_calls["n"] += 1
            return True

    class CountingStop(NeverStop):
        def __call__(self, state):
            stop_calls["n"] += 1
            return state["tries"] >= 2 if isinstance(state, dict) else False

    # Simpler: use built-in helpers so we don't rely on state internals.
    from backon._conditions import retry_always

    wrapped = retry(
        lambda: "ok",
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
        condition=retry_always,
        stop=NeverStop(),
    )
    assert wrapped() == "ok"


def test_retry_sync_default_condition_with_exception():
    # exception given -> default condition built via _make_default_condition
    fn = _make_failing_sync(1, exc=KeyError)
    wrapped = retry(
        fn,
        exception=KeyError,
        max_tries=3,
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
    )
    assert wrapped() == "ok"
    assert fn.state["n"] == 2


def test_retry_sync_default_condition_with_predicate():
    values = iter([None, None, "done"])
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return next(values)

    wrapped = retry(
        fn,
        predicate=operator.not_,
        max_tries=5,
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
    )
    assert wrapped() == "done"
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# Threading kwargs/args through
# ---------------------------------------------------------------------------

def test_retry_sync_passes_args_and_kwargs():
    received = {}

    def fn(a, b, *, c=None):
        received["a"] = a
        received["b"] = b
        received["c"] = c
        return a + b + (c or 0)

    wrapped = retry(fn, sleep=_no_sleep, wait_gen=_zero_wait_gen)
    # Args aren't passed at the wrapper level normally; ensure direct wrapper
    # call works through *args / **kwargs forwarding.
    assert wrapped(1, 2, c=3) == 6
    assert received == {"a": 1, "b": 2, "c": 3}


def test_retry_sync_passes_args_kwargs_low_level():
    received = {}

    def fn(a, b, *, c=None):
        received["a"] = a
        received["b"] = b
        received["c"] = c
        return a + b + (c or 0)

    result = _retry_sync(
        fn,
        _zero_wait_gen,
        sleep=_no_sleep,
        args=(1, 2),
        kwargs={"c": 4},
    )
    assert result == 7
    assert received == {"a": 1, "b": 2, "c": 4}


# ---------------------------------------------------------------------------
# retry_error_callback and raise_on_giveup
# ---------------------------------------------------------------------------

def test_retry_sync_retry_error_callback_returns_value():
    fn = _make_failing_sync(100)

    wrapped = retry(
        fn,
        max_tries=2,
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
        retry_error_callback=lambda details: "fallback",
        raise_on_giveup=False,
    )
    assert wrapped() == "fallback"


def test_retry_async_retry_error_callback_returns_value():
    fn = _make_failing_async(100)

    wrapped = retry(
        fn,
        max_tries=2,
        sleep=_no_async_sleep,
        wait_gen=_zero_wait_gen,
        retry_error_callback=lambda details: "fallback-async",
        raise_on_giveup=False,
    )
    assert asyncio.run(wrapped()) == "fallback-async"


# ---------------------------------------------------------------------------
# Low level _retry_sync / _retry_async
# ---------------------------------------------------------------------------

def test_retry_async_low_level_success():
    async def fn():
        return "x"

    result = asyncio.run(
        _retry_async(fn, _zero_wait_gen, sleep=_no_async_sleep)
    )
    assert result == "x"


def test_retry_sync_custom_sleep_used():
    sleeps = []

    def my_sleep(seconds):
        sleeps.append(seconds)

    fn = _make_failing_sync(2)
    wrapped = retry(fn, max_tries=5, sleep=my_sleep, wait_gen=_zero_wait_gen)
    assert wrapped() == "ok"
    assert len(sleeps) == 2


def test_retry_sync_condition_and_stop_none_defaults_built():
    # Exercises the `condition is None` and `stop is None` branches.
    wrapped = retry(
        _make_failing_sync(1),
        max_tries=3,
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
        exception=ValueError,
    )
    assert wrapped() == "ok"


def test_retry_sync_before_after_handlers_called():
    before_calls = []
    after_calls = []

    wrapped = retry(
        lambda: "ok",
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
        before=lambda details: before_calls.append(1),
        after=lambda details: after_calls.append(1),
    )
    assert wrapped() == "ok"
    assert len(before_calls) == 1
    assert len(after_calls) == 1


def test_retry_async_user_handlers_receive_events():
    events = {"backoff": 0, "giveup": 0, "success": 0}

    def on_backoff(details):
        events["backoff"] += 1

    def on_giveup(details):
        events["giveup"] += 1

    def on_success(details):
        events["success"] += 1

    fn = _make_failing_async(1)
    wrapped = retry(
        fn,
        max_tries=3,
        sleep=_no_async_sleep,
        wait_gen=_zero_wait_gen,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
        on_success=on_success,
    )
    assert asyncio.run(wrapped()) == "ok"
    assert events == {"backoff": 1, "giveup": 0, "success": 1}


# ---------------------------------------------------------------------------
# giveup predicate
# ---------------------------------------------------------------------------

def test_retry_sync_giveup_predicate_stops_early():
    fn = _make_failing_sync(100)

    wrapped = retry(
        fn,
        exception=ValueError,
        giveup=lambda e: True,
        max_tries=10,
        sleep=_no_sleep,
        wait_gen=_zero_wait_gen,
    )
    with pytest.raises(ValueError):
        wrapped()
    # giveup immediately -> only a single attempt executed
    assert fn.state["n"] == 1
