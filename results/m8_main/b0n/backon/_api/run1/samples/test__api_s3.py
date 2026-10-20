import asyncio
import inspect
import logging
import operator

import pytest

from backon._retry import _api
from backon._retry._api import retry, _retry_sync, _retry_async


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_sleep(_):
    return None


async def _async_noop_sleep(_):
    return None


def _identity_wait():
    # A wait generator that never actually waits.
    while True:
        yield 0.0


@pytest.fixture
def fast_wait():
    return _identity_wait


@pytest.fixture
def reset_globals():
    from backon import _common

    old_enabled = _common._GLOBAL_ENABLED
    old_test = dict(_common._TEST_CONFIG)
    try:
        yield
    finally:
        _common._GLOBAL_ENABLED = old_enabled
        _common._TEST_CONFIG.update(old_test)


@pytest.fixture
def enable_globals(reset_globals):
    from backon import _common

    _common._GLOBAL_ENABLED = True
    _common._TEST_CONFIG["max_retries"] = None


# ---------------------------------------------------------------------------
# retry() dispatch
# ---------------------------------------------------------------------------

def test_retry_dispatches_sync(enable_globals, fast_wait):
    calls = []

    def target():
        calls.append(1)
        return "ok"

    wrapped = retry(target, fast_wait, max_tries=3, sleep=_noop_sleep)
    assert wrapped() == "ok"
    assert calls == [1]


def test_retry_dispatches_async(enable_globals, fast_wait):
    calls = []

    async def target():
        calls.append(1)
        return "async-ok"

    wrapped = retry(target, fast_wait, max_tries=3, sleep=_async_noop_sleep)
    assert inspect.iscoroutinefunction(target)
    result = asyncio.run(wrapped())
    assert result == "async-ok"
    assert calls == [1]


def test_retry_sync_returns_callable_wrapping_sync(enable_globals, fast_wait):
    def target():
        return 42

    wrapped = retry(target, fast_wait, sleep=_noop_sleep)
    # sync branch returns coroutine? No: it returns the wrapper callable result.
    assert wrapped() == 42


# ---------------------------------------------------------------------------
# retry() retries on exception
# ---------------------------------------------------------------------------

def test_retry_sync_retries_on_exception(enable_globals, fast_wait):
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("boom")
        return "done"

    wrapped = retry(
        target,
        fast_wait,
        exception=(ValueError,),
        max_tries=5,
        sleep=_noop_sleep,
    )
    assert wrapped() == "done"
    assert state["n"] == 3


def test_retry_sync_raises_after_max_tries(enable_globals, fast_wait):
    def target():
        raise ValueError("always")

    wrapped = retry(
        target,
        fast_wait,
        exception=ValueError,
        max_tries=3,
        sleep=_noop_sleep,
    )
    with pytest.raises(ValueError, match="always"):
        wrapped()


def test_retry_async_retries_on_exception(enable_globals, fast_wait):
    state = {"n": 0}

    async def target():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("boom")
        return "done"

    wrapped = retry(
        target,
        fast_wait,
        exception=(ValueError,),
        max_tries=5,
        sleep=_async_noop_sleep,
    )
    assert asyncio.run(wrapped()) == "done"
    assert state["n"] == 3


def test_retry_async_raises_after_max_tries(enable_globals, fast_wait):
    async def target():
        raise ValueError("always")

    wrapped = retry(
        target,
        fast_wait,
        exception=ValueError,
        max_tries=2,
        sleep=_async_noop_sleep,
    )
    with pytest.raises(ValueError, match="always"):
        asyncio.run(wrapped())


# ---------------------------------------------------------------------------
# retry() retries on predicate (return value)
# ---------------------------------------------------------------------------

def test_retry_sync_predicate_retries_until_falsy(enable_globals, fast_wait):
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 3:
            return False
        return True

    wrapped = retry(
        target,
        fast_wait,
        predicate=operator.not_,  # retry while result is falsy
        max_tries=5,
        sleep=_noop_sleep,
    )
    assert wrapped() is True
    assert state["n"] == 3


def test_retry_sync_predicate_gives_up(enable_globals, fast_wait):
    def target():
        return False

    wrapped = retry(
        target,
        fast_wait,
        predicate=operator.not_,
        max_tries=2,
        sleep=_noop_sleep,
        raise_on_giveup=False,
    )
    # With raise_on_giveup=False, returns final value instead of raising.
    assert wrapped() is False


def test_retry_raise_on_giveup_true_raises_predicate(enable_globals, fast_wait):
    def target():
        return False

    wrapped = retry(
        target,
        fast_wait,
        predicate=operator.not_,
        max_tries=2,
        sleep=_noop_sleep,
        raise_on_giveup=True,
    )
    with pytest.raises(Exception):
        wrapped()


# ---------------------------------------------------------------------------
# retry() with explicit condition / stop
# ---------------------------------------------------------------------------

def test_retry_with_explicit_condition_and_stop(enable_globals, fast_wait):
    from backon._conditions import retry_if_exception_type, stop_after_attempt

    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("x")
        return "value"

    wrapped = retry(
        target,
        fast_wait,
        condition=retry_if_exception_type(ValueError),
        stop=stop_after_attempt(3),
        sleep=_noop_sleep,
    )
    assert wrapped() == "value"
    assert state["n"] == 2


# ---------------------------------------------------------------------------
# retry() passthrough of handlers
# ---------------------------------------------------------------------------

def test_retry_on_backoff_handler_called(enable_globals, fast_wait):
    events = []

    def target():
        state = events.count("attempt")
        if state < 2:
            raise ValueError("x")
        return "ok"

    def on_backoff(details):
        events.append(("backoff", details))

    wrapped = retry(
        target,
        fast_wait,
        exception=ValueError,
        max_tries=5,
        sleep=_noop_sleep,
        on_backoff=on_backoff,
        logger=None,
    )
    assert wrapped() == "ok"
    backoffs = [e for e in events if isinstance(e, tuple) and e[0] == "backoff"]
    assert len(backoffs) == 2
    for _, details in backoffs:
        assert details["tries"] >= 1
        assert details["target"] is target


def test_retry_on_success_handler_called(enable_globals, fast_wait):
    events = []

    def target():
        return "value"

    def on_success(details):
        events.append(details)

    wrapped = retry(
        target,
        fast_wait,
        sleep=_noop_sleep,
        on_success=on_success,
        logger=None,
    )
    assert wrapped() == "value"
    assert len(events) == 1
    assert events[0]["value"] == "value"


def test_retry_on_giveup_handler_called(enable_globals, fast_wait):
    events = []

    def target():
        raise ValueError("x")

    def on_giveup(details):
        events.append(details)

    wrapped = retry(
        target,
        fast_wait,
        exception=ValueError,
        max_tries=2,
        sleep=_noop_sleep,
        on_giveup=on_giveup,
        logger=None,
        raise_on_giveup=False,
    )
    wrapped()
    assert len(events) == 1
    assert events[0]["tries"] == 2


def test_retry_on_attempt_handler_called(enable_globals, fast_wait):
    events = []

    def target():
        state = len(events)
        if state < 1:
            raise ValueError("x")
        return "ok"

    def on_attempt(_):
        events.append(1)

    wrapped = retry(
        target,
        fast_wait,
        exception=ValueError,
        max_tries=5,
        sleep=_noop_sleep,
        on_attempt=on_attempt,
        logger=None,
    )
    assert wrapped() == "ok"
    assert len(events) == 2


# ---------------------------------------------------------------------------
# retry() with retry_error_callback
# ---------------------------------------------------------------------------

def test_retry_error_callback_returns_value(enable_globals, fast_wait):
    def target():
        raise ValueError("x")

    def err_cb(details):
        return "cb-result"

    wrapped = retry(
        target,
        fast_wait,
        exception=ValueError,
        max_tries=2,
        sleep=_noop_sleep,
        retry_error_callback=err_cb,
        logger=None,
    )
    assert wrapped() == "cb-result"


# ---------------------------------------------------------------------------
# retry() when disabled
# ---------------------------------------------------------------------------

def test_retry_sync_disabled_calls_target_once(fast_wait):
    from backon import _common

    _common._GLOBAL_ENABLED = False
    old = dict(_common._TEST_CONFIG)
    try:
        _common._TEST_CONFIG["max_retries"] = None
        calls = []

        def target():
            calls.append(1)
            return "r"

        wrapped = retry(target, fast_wait, max_tries=5, sleep=_noop_sleep)
        assert wrapped() == "r"
        assert calls == [1]
    finally:
        _common._GLOBAL_ENABLED = True
        _common._TEST_CONFIG.update(old)


def test_retry_async_disabled_calls_target_once(fast_wait):
    from backon import _common

    _common._GLOBAL_ENABLED = False
    old = dict(_common._TEST_CONFIG)
    try:
        _common._TEST_CONFIG["max_retries"] = None
        calls = []

        async def target():
            calls.append(1)
            return "r"

        wrapped = retry(target, fast_wait, max_tries=5, sleep=_async_noop_sleep)
        assert asyncio.run(wrapped()) == "r"
        assert calls == [1]
    finally:
        _common._GLOBAL_ENABLED = True
        _common._TEST_CONFIG.update(old)


# ---------------------------------------------------------------------------
# retry() with before/after hooks
# ---------------------------------------------------------------------------

def test_retry_before_after_hooks(enable_globals, fast_wait):
    events = []

    def target():
        events.append("target")
        return "ok"

    def before_h(details):
        events.append("before")

    def after_h(details):
        events.append("after")

    wrapped = retry(
        target,
        fast_wait,
        sleep=_noop_sleep,
        before=before_h,
        after=after_h,
        logger=None,
    )
    assert wrapped() == "ok"
    assert events == ["before", "target", "after"]


# ---------------------------------------------------------------------------
# retry() with args/kwargs? Those are internal API params not exposed by retry.
# Test underlying _retry_sync / _retry_async directly for args/kwargs.
# ---------------------------------------------------------------------------

def test_retry_sync_args_kwargs_internal(enable_globals, fast_wait):
    def target(a, b):
        return a + b

    result = _retry_sync(
        target,
        fast_wait,
        max_tries=2,
        sleep=_noop_sleep,
        logger=None,
        args=(1,),
        kwargs={"b": 2},
    )
    assert result == 3


def test_retry_async_args_kwargs_internal(enable_globals, fast_wait):
    async def target(a, b):
        return a + b

    result = asyncio.run(
        _retry_async(
            target,
            fast_wait,
            max_tries=2,
            sleep=_async_noop_sleep,
            logger=None,
            args=(2,),
            kwargs={"b": 3},
        )
    )
    assert result == 5


# ---------------------------------------------------------------------------
# retry() default wait generator (expo) and defaults
# ---------------------------------------------------------------------------

def test_retry_default_wait_gen(enable_globals):
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("x")
        return "ok"

    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        return None

    wrapped = retry(
        target,
        exception=ValueError,
        max_tries=3,
        sleep=fake_sleep,
    )
    assert wrapped() == "ok"
    assert len(sleeps) == 1
    assert sleeps[0] >= 0


def test_retry_raises_on_giveup_default_true(enable_globals, fast_wait):
    def target():
        raise ValueError("boom")

    wrapped = retry(
        target,
        fast_wait,
        exception=ValueError,
        max_tries=1,
        sleep=_noop_sleep,
    )
    with pytest.raises(ValueError, match="boom"):
        wrapped()


# ---------------------------------------------------------------------------
# retry() with logger string triggers default logging handlers
# ---------------------------------------------------------------------------

def test_retry_logs_giveup(enable_globals, fast_wait, caplog):
    def target():
        raise ValueError("nope")

    wrapped = retry(
        target,
        fast_wait,
        exception=ValueError,
        max_tries=1,
        sleep=_noop_sleep,
        logger="test.backon",
        backoff_log_level=logging.DEBUG,
        giveup_log_level=logging.WARNING,
        raise_on_giveup=False,
    )
    with caplog.at_level(logging.DEBUG, logger="test.backon"):
        wrapped()
    assert any("Giving up" in rec.message for rec in caplog.records)


def test_retry_logs_backoff(enable_globals, fast_wait, caplog):
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("x")
        return "ok"

    wrapped = retry(
        target,
        fast_wait,
        exception=ValueError,
        max_tries=3,
        sleep=_noop_sleep,
        logger="test.backon.backoff",
        backoff_log_level=logging.DEBUG,
    )
    with caplog.at_level(logging.DEBUG, logger="test.backon.backoff"):
        wrapped()
    assert any("Backing off" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# retry() with rate limiter
# ---------------------------------------------------------------------------

def test_retry_rate_limiter_eventually_succeeds(enable_globals, fast_wait):
    from backon._rate_limiter import RateLimiter

    rl = RateLimiter(max_calls=100, period=1.0)
    calls = []

    def target():
        calls.append(1)
        return "ok"

    wrapped = retry(
        target,
        fast_wait,
        sleep=_noop_sleep,
        rate_limit=rl,
        logger=None,
    )
    assert wrapped() == "ok"
    assert calls == [1]


# ---------------------------------------------------------------------------
# retry() giveup predicate on exceptions
# ---------------------------------------------------------------------------

def test_retry_giveup_predicate_stops_retries(enable_globals, fast_wait):
    calls = []

    def target():
        calls.append(1)
        raise KeyError("stop")

    def giveup(exc):
        return isinstance(exc, KeyError)

    wrapped = retry(
        target,
        fast_wait,
        exception=KeyError,
        giveup=giveup,
        max_tries=10,
        sleep=_noop_sleep,
        logger=None,
    )
    with pytest.raises(KeyError):
        wrapped()
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Default arguments sanity
# ---------------------------------------------------------------------------

def test_retry_default_wait_gen_is_expo():
    sig = inspect.signature(retry)
    default = sig.parameters["wait_gen"].default
    from backon._wait_gen import expo

    assert default is expo


def test_retry_default_predicate_is_not():
    sig = inspect.signature(retry)
    default = sig.parameters["predicate"].default
    assert default is operator.not_
