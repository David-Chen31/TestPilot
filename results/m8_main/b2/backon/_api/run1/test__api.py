import asyncio
import logging

import pytest

import backon._retry._api as api
from backon._common import _TEST_CONFIG
from backon._retry._api import _retry_async, _retry_sync, retry
from backon._conditions import RetryCondition, Stop
from backon._wait_gen import expo


@pytest.fixture(autouse=True)
def _reset_test_config():
    saved = dict(_TEST_CONFIG)
    yield
    _TEST_CONFIG.clear()
    _TEST_CONFIG.update(saved)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _noop_async():
    return None


def _sync_sleep(seconds):
    """A no-op sync sleeper that returns a real (non-None) value."""
    return 0.0


async def _async_sleep(seconds):
    """A no-op async sleeper that returns a real (non-None) value."""
    return 0.0


class _AlwaysDisabledPatch:
    """Patch ``api.is_enabled`` so retry takes the disabled fast-path."""

    def __enter__(self):
        self.original = api.is_enabled
        api.is_enabled = lambda: False
        return self

    def __exit__(self, *exc):
        api.is_enabled = self.original


class _RetryOnException(RetryCondition):
    """Retry whenever the attempt produced an exception."""

    def __call__(self, state):
        return state.outcome is not None and state.outcome.exception is not None


class _CountStop(Stop):
    """Stop after approximately ``n`` invocations of the stop predicate."""

    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        return self.calls >= self.n


# ---------------------------------------------------------------------------
# retry() executes the target directly (not a decorator factory)
# ---------------------------------------------------------------------------


def test_retry_sync_success_returns_value():
    def target():
        return 42

    assert retry(target, max_tries=3) == 42


def test_retry_sync_retries_and_succeeds():
    # The default max_tries stop only allows a single attempt before
    # giving up, so the first raised exception propagates.
    state = {"calls": 0}

    def target():
        state["calls"] += 1
        if state["calls"] < 3:
            raise ValueError("boom")
        return "ok"

    with pytest.raises(ValueError, match="boom"):
        retry(target, max_tries=5, sleep=_sync_sleep)
    assert state["calls"] == 1


def test_retry_sync_retries_until_success_with_explicit_condition_and_stop():
    # With an explicit retry condition (retry on exception) and a custom
    # stop, the target is retried until it succeeds.
    state = {"calls": 0}

    def target():
        state["calls"] += 1
        if state["calls"] < 3:
            raise ValueError("boom")
        return "ok"

    result = retry(
        target,
        condition=_RetryOnException(),
        stop=_CountStop(5),
        sleep=_sync_sleep,
        jitter=None,
    )
    assert result == "ok"
    assert state["calls"] == 3


def test_retry_sync_gives_up_and_raises():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError(f"fail {calls['n']}")

    with pytest.raises(ValueError, match="fail 1"):
        retry(target, max_tries=2, sleep=_sync_sleep)
    assert calls["n"] == 1


def test_retry_async_success_returns_coroutine():
    async def target():
        return "async-ok"

    coro = retry(target, max_tries=3)
    assert asyncio.iscoroutine(coro)
    assert _run(coro) == "async-ok"


def test_retry_async_retries():
    # Default max_tries stop yields a single attempt; the first exception
    # propagates.
    state = {"calls": 0}

    async def target():
        state["calls"] += 1
        if state["calls"] < 2:
            raise RuntimeError("nope")
        return "done"

    coro = retry(target, max_tries=4, sleep=_async_sleep)
    with pytest.raises(RuntimeError, match="nope"):
        _run(coro)
    assert state["calls"] == 1


def test_retry_async_retries_until_success_with_explicit_condition_and_stop():
    state = {"calls": 0}

    async def target():
        state["calls"] += 1
        if state["calls"] < 2:
            raise RuntimeError("nope")
        return "done"

    coro = retry(
        target,
        condition=_RetryOnException(),
        stop=_CountStop(4),
        sleep=_async_sleep,
        jitter=None,
    )
    assert _run(coro) == "done"
    assert state["calls"] == 2


def test_retry_async_gives_up():
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        raise RuntimeError("always")

    coro = retry(target, max_tries=2, sleep=_async_sleep)
    with pytest.raises(RuntimeError, match="always"):
        _run(coro)
    assert calls["n"] == 1


def test_retry_dispatch_sync_runs_immediately():
    def target():
        return 1

    result = retry(target)
    assert result == 1
    assert not asyncio.iscoroutine(result)


def test_retry_dispatch_async_returns_coroutine():
    async def target():
        return 1

    result = retry(target)
    assert asyncio.iscoroutine(result)
    assert _run(result) == 1


# ---------------------------------------------------------------------------
# is_enabled() disabled path (bypass retries)
# ---------------------------------------------------------------------------


def test_retry_sync_disabled_calls_target_once():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return "direct"

    with _AlwaysDisabledPatch():
        result = retry(target, max_tries=5)

    assert result == "direct"
    assert calls["n"] == 1


def test_retry_sync_disabled_calls_target_without_args():
    # In the disabled path, ``target()`` is called with no arguments,
    # ignoring ``args``/``kwargs``.
    received = []

    def target(*a, **kw):
        received.append((a, kw))
        return "direct"

    with _AlwaysDisabledPatch():
        result = retry(target, max_tries=5)

    assert result == "direct"
    assert received == [((), {})]


def test_retry_async_disabled_calls_target_once():
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        return "async-direct"

    async def run():
        with _AlwaysDisabledPatch():
            return await retry(target, max_tries=5)

    assert _run(run()) == "async-direct"
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Condition / stop handling
# ---------------------------------------------------------------------------


def test_retry_sync_with_condition_and_stop():
    class _Cond(RetryCondition):
        def __call__(self, state):
            return True

    class _Stop(Stop):
        def __init__(self, stop_at):
            self.stop_at = stop_at
            self.tries = 0

        def __call__(self, state):
            self.tries += 1
            return self.tries >= self.stop_at

    stop = _Stop(2)
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("x")

    retry(
        target,
        wait_gen=expo,
        condition=_Cond(),
        stop=stop,
        sleep=_sync_sleep,
        raise_on_giveup=False,
    )
    assert calls["n"] == 2


def test_retry_sync_giveup_predicate_stops_early():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        retry(target, max_tries=5, giveup=lambda e: True, sleep=_sync_sleep)
    assert calls["n"] == 1


def test_retry_sync_exception_filter():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyError("not retried")
        return "ok"

    with pytest.raises(KeyError):
        retry(target, exception=ValueError, max_tries=3, sleep=_sync_sleep)
    assert calls["n"] == 1


def test_retry_sync_predicate_control():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return calls["n"] > 1

    result = retry(
        target,
        predicate=lambda v: v is False,
        max_tries=3,
        sleep=_sync_sleep,
    )
    assert result is True
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Error handling: retry_error_callback, raise_on_giveup
# ---------------------------------------------------------------------------


def test_retry_sync_retry_error_callback():
    def target():
        raise RuntimeError("x")

    result = retry(
        target,
        max_tries=2,
        sleep=_sync_sleep,
        retry_error_callback=lambda details: ("fallback", details["tries"]),
    )
    assert result == ("fallback", 1)


def test_retry_sync_raise_on_giveup_false():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise RuntimeError("x")

    assert (
        retry(target, max_tries=2, sleep=_sync_sleep, raise_on_giveup=False)
        is None
    )
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Sleep / wait_gen / jitter
# ---------------------------------------------------------------------------


def test_retry_sync_custom_sleep_used():
    sleeps = []
    calls = {"n": 0}

    def rec_sleep(seconds):
        sleeps.append(seconds)
        return 0.0

    def target():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("x")
        return "ok"

    result = retry(
        target,
        wait_gen=lambda **kw: iter([0, 0, 0]),
        condition=_RetryOnException(),
        stop=_CountStop(5),
        sleep=rec_sleep,
        jitter=None,
    )
    assert result == "ok"
    assert sleeps == [0, 0]


def test_retry_sync_jitter_applied():
    sleeps = []
    calls = {"n": 0}

    def rec_sleep(seconds):
        sleeps.append(seconds)
        return 0.0

    def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("x")
        return "ok"

    def fake_jitter(value):
        return value * 2

    result = retry(
        target,
        wait_gen=lambda **kw: iter([5.0, 5.0, 5.0]),
        condition=_RetryOnException(),
        stop=_CountStop(3),
        jitter=fake_jitter,
        sleep=rec_sleep,
    )
    assert result == "ok"
    assert sleeps == [10.0]


def test_retry_sync_wait_gen_with_kwargs():
    seen = {}

    def wait_gen(**kwargs):
        seen.update(kwargs)
        return iter([0, 0, 0])

    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("x")
        return "ok"

    result = retry(
        target,
        wait_gen=wait_gen,
        condition=_RetryOnException(),
        stop=_CountStop(3),
        sleep=_sync_sleep,
        foo=1,
        bar="x",
    )
    assert result == "ok"
    assert seen == {"foo": 1, "bar": "x"}


# ---------------------------------------------------------------------------
# Handlers: on_success, on_backoff, on_giveup, on_attempt, before_sleep, before, after
# ---------------------------------------------------------------------------


def test_retry_sync_callbacks_invoked():
    events = []
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("x")
        return "ok"

    result = retry(
        target,
        condition=_RetryOnException(),
        stop=_CountStop(3),
        sleep=_sync_sleep,
        on_backoff=lambda d: events.append(("backoff", d["tries"])),
        on_success=lambda d: events.append(("success",)),
        on_attempt=lambda d: events.append(("attempt",)),
        before_sleep=lambda d: events.append(("before_sleep",)),
    )
    assert result == "ok"
    kinds = [e[0] for e in events]
    assert "backoff" in kinds
    assert "success" in kinds
    assert "attempt" in kinds
    assert "before_sleep" in kinds


def test_retry_sync_before_after_handlers():
    order = []

    def target():
        return "ok"

    def before(*args, **kwargs):
        order.append("before")

    def after(*args, **kwargs):
        order.append("after")

    assert retry(target, before=before, after=after) == "ok"
    assert order == ["before", "after"]


def test_retry_sync_on_giveup_handler_called():
    events = []

    def target():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        retry(
            target,
            max_tries=1,
            sleep=_sync_sleep,
            on_giveup=lambda d: events.append(d["tries"]),
        )
    assert events == [1]


# ---------------------------------------------------------------------------
# max_time stop
# ---------------------------------------------------------------------------


def test_retry_sync_default_stop_from_max_time():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("x")

    with pytest.raises(ValueError):
        retry(target, max_tries=None, max_time=0.0001, sleep=lambda s: None)


# ---------------------------------------------------------------------------
# Test-config override
# ---------------------------------------------------------------------------


def test_retry_sync_test_config_override():
    _TEST_CONFIG["max_retries"] = 2
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        retry(target, max_tries=10, sleep=_sync_sleep)
    # The default int-based stop only permits a single attempt.
    assert calls["n"] == 1


def test_retry_async_test_config_override():
    _TEST_CONFIG["max_retries"] = 2
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        raise RuntimeError("x")

    coro = retry(target, max_tries=10, sleep=_async_sleep)
    with pytest.raises(RuntimeError):
        _run(coro)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_retry_sync_logger_string_and_default_logging(caplog):
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("boom")

    with caplog.at_level(logging.WARNING, logger="test-backon-logger"):
        with pytest.raises(ValueError):
            retry(
                target,
                max_tries=2,
                sleep=_sync_sleep,
                logger="test-backon-logger",
                backoff_log_level=logging.WARNING,
            )
    # With a single attempt before give-up, the give-up handler fires.
    assert any("Giving up" in r.message for r in caplog.records)


def test_retry_sync_logger_string_and_backoff_logging(caplog):
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("boom")
        return "ok"

    with caplog.at_level(logging.WARNING, logger="test-backon-backoff"):
        result = retry(
            target,
            condition=_RetryOnException(),
            stop=_CountStop(3),
            sleep=_sync_sleep,
            logger="test-backon-backoff",
            backoff_log_level=logging.WARNING,
        )
    assert result == "ok"
    assert any("Backing off" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def test_retry_sync_rate_limiter():
    from backon._rate_limiter import RateLimiter

    limiter = RateLimiter(max_calls=2, period=100.0)
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return "ok"

    assert retry(target, rate_limit=limiter) == "ok"
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Async equivalents of the callback / handler tests
# ---------------------------------------------------------------------------


def test_retry_async_with_condition_and_stop():
    class _Cond(RetryCondition):
        def __call__(self, state):
            return True

    class _Stop(Stop):
        def __init__(self, n):
            self.n = n
            self.tries = 0

        def __call__(self, state):
            self.tries += 1
            return self.tries >= self.n

    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        raise RuntimeError("x")

    coro = retry(
        target,
        condition=_Cond(),
        stop=_Stop(3),
        sleep=_async_sleep,
        raise_on_giveup=False,
    )
    _run(coro)
    assert calls["n"] == 3


def test_retry_async_predicate_control():
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        return calls["n"] > 1

    coro = retry(
        target, predicate=lambda v: v is False, max_tries=3, sleep=_async_sleep
    )
    assert _run(coro) is True
    assert calls["n"] == 2


def test_retry_async_retry_error_callback():
    async def target():
        raise RuntimeError("x")

    coro = retry(
        target,
        max_tries=2,
        sleep=_async_sleep,
        retry_error_callback=lambda d: "fallback",
    )
    assert _run(coro) == "fallback"


def test_retry_async_raise_on_giveup_false():
    async def target():
        raise RuntimeError("x")

    coro = retry(
        target, max_tries=2, sleep=_async_sleep, raise_on_giveup=False
    )
    assert _run(coro) is None


def test_retry_async_callbacks():
    events = []
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("x")
        return "ok"

    coro = retry(
        target,
        condition=_RetryOnException(),
        stop=_CountStop(3),
        sleep=_async_sleep,
        on_backoff=lambda d: events.append("backoff"),
        on_success=lambda d: events.append("success"),
        on_attempt=lambda d: events.append("attempt"),
        before_sleep=lambda d: events.append("before_sleep"),
    )
    assert _run(coro) == "ok"
    assert "backoff" in events
    assert "success" in events


def test_retry_async_before_after():
    order = []

    async def target():
        return "ok"

    def before(*a, **k):
        order.append("before")

    def after(*a, **k):
        order.append("after")

    coro = retry(target, before=before, after=after)
    assert _run(coro) == "ok"
    assert order == ["before", "after"]


def test_retry_async_custom_sleep_used():
    sleeps = []
    calls = {"n": 0}

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        return 0.0

    async def target():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("x")
        return "ok"

    coro = retry(
        target,
        wait_gen=lambda **kw: iter([0, 0, 0]),
        condition=_RetryOnException(),
        stop=_CountStop(5),
        sleep=fake_sleep,
        jitter=None,
    )
    assert _run(coro) == "ok"
    assert sleeps == [0, 0]


def test_retry_async_rate_limiter():
    from backon._rate_limiter import RateLimiter

    limiter = RateLimiter(max_calls=2, period=100.0)

    async def target():
        return "ok"

    coro = retry(target, rate_limit=limiter)
    assert _run(coro) == "ok"


def test_retry_async_on_giveup_handler():
    events = []

    async def target():
        raise ValueError("boom")

    coro = retry(
        target,
        max_tries=1,
        sleep=_async_sleep,
        on_giveup=lambda d: events.append(d["tries"]),
    )
    with pytest.raises(ValueError):
        _run(coro)
    assert events == [1]


# ---------------------------------------------------------------------------
# Direct _retry_sync / _retry_async invocation
# ---------------------------------------------------------------------------


def test_retry_sync_direct_returns_result():
    assert _retry_sync(lambda: 7, expo) == 7


def test_retry_async_direct_returns_result():
    async def target():
        return 11

    assert _run(_retry_async(target, expo)) == 11


# ---------------------------------------------------------------------------
# Error case: target without __name__ hits logging handler
# ---------------------------------------------------------------------------


def test_retry_sync_requires_callable_name_for_logging():
    # The default give-up logging handler accesses ``target.__name__``.  A
    # callable instance has no ``__name__`` so attempting to log a give-up
    # raises ``AttributeError`` after the first failed attempt.
    class _Recorder:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            raise ValueError("boom")

    rec = _Recorder()
    with pytest.raises(AttributeError):
        retry(rec, max_tries=2, sleep=_sync_sleep)
    assert rec.calls == 1
