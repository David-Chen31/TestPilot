import asyncio
import logging
import operator

import pytest

import backon._retry._api as api
from backon._retry._api import (
    _retry_async,
    _retry_sync,
    retry,
)
from backon._conditions import RetryCondition, Stop
from backon._rate_limiter import RateLimiter
from backon._wait_gen import constant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Counter:
    def __init__(self, fail_times=0, exc=ValueError, value="ok"):
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc
        self.value = value

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc("boom")
        return self.value


class AsyncCounter:
    def __init__(self, fail_times=0, exc=ValueError, value="ok"):
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc
        self.value = value

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc("boom")
        return self.value


def no_sleep(_delay):
    return None


@pytest.fixture(autouse=True)
def _reset_global_state(monkeypatch):
    """Ensure global enabled flag and test config do not leak between tests."""
    import backon._common as common

    monkeypatch.setattr(common, "_GLOBAL_ENABLED", True, raising=False)
    # Also patch the imported reference in the api module helpers by patching
    # the common module's underlying state.
    monkeypatch.setattr(common, "_TEST_CONFIG", {"max_retries": None}, raising=False)
    yield


# ---------------------------------------------------------------------------
# retry() dispatch
# ---------------------------------------------------------------------------

def test_retry_dispatches_sync():
    calls = []

    @retry(max_tries=1, sleep=no_sleep)
    def fn(x):
        calls.append(x)
        return x * 2

    assert fn(3) == 6
    assert calls == [3]


def test_retry_dispatches_async():
    calls = []

    @retry(max_tries=1, sleep=no_sleep)
    async def fn(x):
        calls.append(x)
        return x * 2

    result = asyncio.run(fn(4))
    assert result == 8
    assert calls == [4]


def test_retry_uses_default_wait_gen_expo():
    captured = {}

    def fake_sync(target, wait_gen, **kwargs):
        captured["wait_gen"] = wait_gen
        return "sync-result"

    import backon._retry._api as mod
    from backon._wait_gen import expo as expo_mod

    original = mod._retry_sync
    mod._retry_sync = fake_sync
    try:
        def target():
            return 1

        result = retry(target)
    finally:
        mod._retry_sync = original

    assert result == "sync-result"
    assert captured["wait_gen"] is expo_mod


def test_retry_passes_wait_gen_kwargs_to_sync():
    captured = {}

    def fake_sync(target, wait_gen, **kwargs):
        captured.update(kwargs)
        captured["wait_gen"] = wait_gen
        return "ok"

    import backon._retry._api as mod
    orig = mod._retry_sync
    mod._retry_sync = fake_sync
    try:
        def target():
            return 1

        retry(target, max_tries=3, base=2.0, factor=3)
    finally:
        mod._retry_sync = orig

    assert captured["wait_gen_kwargs"] == {"base": 2.0, "factor": 3}


def test_retry_passes_wait_gen_kwargs_to_async():
    captured = {}

    async def fake_async(target, wait_gen, **kwargs):
        captured.update(kwargs)
        return "async-ok"

    import backon._retry._api as mod
    orig = mod._retry_async
    mod._retry_async = fake_async
    try:
        async def target():
            return 1

        result = retry(target, max_tries=3, base=2.0)
    finally:
        mod._retry_async = orig

    assert result == "async-ok"
    assert captured["wait_gen_kwargs"] == {"base": 2.0}


# ---------------------------------------------------------------------------
# _retry_sync basic behavior
# ---------------------------------------------------------------------------

def test_sync_success_first_try():
    counter = Counter()
    result = _retry_sync(counter, constant(1), max_tries=3, sleep=no_sleep)
    assert result == "ok"
    assert counter.calls == 1


def test_sync_retries_and_succeeds():
    counter = Counter(fail_times=2)
    result = _retry_sync(
        counter,
        constant(1),
        exception=ValueError,
        max_tries=5,
        sleep=no_sleep,
    )
    assert result == "ok"
    assert counter.calls == 3


def test_sync_gives_up_raises():
    counter = Counter(fail_times=10)
    with pytest.raises(ValueError):
        _retry_sync(
            counter,
            constant(1),
            exception=ValueError,
            max_tries=3,
            sleep=no_sleep,
        )
    assert counter.calls == 3


def test_sync_raise_on_giveup_false_returns_value_or_exception():
    counter = Counter(fail_times=10)
    # When raise_on_giveup is False, giveup returns last exception/value.
    with pytest.raises(Exception):
        # default retry_error_callback may still raise; but if None, returns value.
        _retry_sync(
            counter,
            constant(1),
            exception=ValueError,
            max_tries=3,
            sleep=no_sleep,
            raise_on_giveup=False,
        )


def test_sync_raise_on_giveup_false_with_callback():
    counter = Counter(fail_times=10)

    def cb(details):
        return "callback-value"

    result = _retry_sync(
        counter,
        constant(1),
        exception=ValueError,
        max_tries=2,
        sleep=no_sleep,
        raise_on_giveup=False,
        retry_error_callback=cb,
    )
    assert result == "callback-value"
    assert counter.calls == 2


def test_sync_forwards_args_and_kwargs():
    received = {}

    def target(a, b=0, *, c=0):
        received["a"] = a
        received["b"] = b
        received["c"] = c
        return a + b + c

    result = _retry_sync(
        target,
        constant(1),
        max_tries=1,
        sleep=no_sleep,
        args=(1,),
        kwargs={"b": 2, "c": 3},
    )
    assert result == 6
    assert received == {"a": 1, "b": 2, "c": 3}


def test_sync_uses_explicit_sleep():
    delays = []

    def fake_sleep(d):
        delays.append(d)

    counter = Counter(fail_times=2)
    _retry_sync(
        counter,
        constant(5),
        exception=ValueError,
        max_tries=5,
        sleep=fake_sleep,
        jitter=None,
        wait_gen_kwargs={},
    )
    assert counter.calls == 3
    # Without jitter the wait should be exactly 5 each iteration.
    assert delays == [5, 5]


def test_sync_default_condition_and_stop_built_when_none():
    counter = Counter(fail_times=1)
    result = _retry_sync(
        counter,
        constant(1),
        exception=ValueError,
        max_tries=3,
        sleep=no_sleep,
    )
    assert result == "ok"
    assert counter.calls == 2


def test_sync_custom_condition_and_stop_used():
    counter = Counter(fail_times=10)

    class NeverStop(Stop):
        def __call__(self, state):
            return False

    class StopAfter2(Stop):
        def __call__(self, state):
            return state.tries >= 2

    class RetryOnValueError(RetryCondition):
        def __call__(self, state):
            if state.outcome and state.outcome.exception:
                return True
            return False

    with pytest.raises(ValueError):
        _retry_sync(
            counter,
            constant(1),
            condition=RetryOnValueError(),
            stop=StopAfter2(),
            sleep=no_sleep,
        )
    assert counter.calls == 2


def test_sync_predicate_default_condition():
    values = [0, 0, 1]
    idx = [0]

    def target():
        v = values[idx[0]]
        idx[0] += 1
        return v

    result = _retry_sync(
        target,
        constant(1),
        max_tries=5,
        sleep=no_sleep,
        predicate=operator.not_,
    )
    assert result == 1
    assert idx[0] == 3


def test_sync_rate_limit_passed_through():
    limiter = RateLimiter(max_calls=1, period=1000.0)
    counter = Counter()

    result = _retry_sync(
        counter,
        constant(1),
        max_tries=1,
        sleep=no_sleep,
        rate_limit=limiter,
    )
    assert result == "ok"
    assert counter.calls == 1


# ---------------------------------------------------------------------------
# Logging behavior
# ---------------------------------------------------------------------------

def test_sync_logs_on_backoff(caplog):
    counter = Counter(fail_times=1)
    with caplog.at_level(logging.INFO, logger="backon"):
        _retry_sync(
            counter,
            constant(1),
            exception=ValueError,
            max_tries=3,
            sleep=no_sleep,
        )
    assert any("Backing off" in rec.getMessage() for rec in caplog.records)


def test_sync_logs_on_giveup(caplog):
    counter = Counter(fail_times=10)
    with caplog.at_level(logging.ERROR, logger="backon"):
        with pytest.raises(ValueError):
            _retry_sync(
                counter,
                constant(1),
                exception=ValueError,
                max_tries=1,
                sleep=no_sleep,
            )
    assert any("Giving up" in rec.getMessage() for rec in caplog.records)


def test_sync_custom_logger_object(caplog):
    counter = Counter(fail_times=1)
    logger = logging.getLogger("custom-test-logger")
    with caplog.at_level(logging.INFO, logger="custom-test-logger"):
        _retry_sync(
            counter,
            constant(1),
            exception=ValueError,
            max_tries=3,
            sleep=no_sleep,
            logger=logger,
        )
    assert any("Backing off" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def test_sync_on_success_handler_called():
    counter = Counter()
    events = []

    def on_success(details):
        events.append(details)

    _retry_sync(
        counter,
        constant(1),
        max_tries=1,
        sleep=no_sleep,
        on_success=on_success,
    )
    assert len(events) == 1
    details = events[0]
    assert details["tries"] == 1
    assert details["target"] is counter


def test_sync_on_backoff_and_on_giveup_handlers():
    backoff_events = []
    giveup_events = []
    counter = Counter(fail_times=10)

    with pytest.raises(ValueError):
        _retry_sync(
            counter,
            constant(1),
            exception=ValueError,
            max_tries=3,
            sleep=no_sleep,
            on_backoff=backoff_events.append,
            on_giveup=giveup_events.append,
        )
    # Two backoffs (after tries 1 and 2) then giveup on try 3
    assert len(backoff_events) == 2
    assert len(giveup_events) == 1
    assert giveup_events[0]["tries"] == 3


def test_sync_on_attempt_handler_called():
    attempts = []
    counter = Counter()
    _retry_sync(
        counter,
        constant(1),
        max_tries=1,
        sleep=no_sleep,
        on_attempt=attempts.append,
    )
    assert len(attempts) == 1


def test_sync_before_and_after_handlers():
    order = []

    def before(details):
        order.append(("before", details["tries"]))

    def after(details):
        order.append(("after", details["tries"]))

    counter = Counter(fail_times=1)
    _retry_sync(
        counter,
        constant(1),
        exception=ValueError,
        max_tries=3,
        sleep=no_sleep,
        before=before,
        after=after,
    )
    assert ("before", 1) in order
    assert ("before", 2) in order
    assert ("after", 1) in order
    assert ("after", 2) in order


def test_sync_before_sleep_handler():
    sleeps = []
    counter = Counter(fail_times=1)
    _retry_sync(
        counter,
        constant(1),
        exception=ValueError,
        max_tries=3,
        sleep=no_sleep,
        before_sleep=sleeps.append,
    )
    assert len(sleeps) == 1


def test_sync_list_of_handlers():
    calls_a = []
    calls_b = []
    counter = Counter(fail_times=1)
    _retry_sync(
        counter,
        constant(1),
        exception=ValueError,
        max_tries=3,
        sleep=no_sleep,
        on_backoff=[calls_a.append, calls_b.append],
    )
    assert len(calls_a) == 1
    assert len(calls_b) == 1


# ---------------------------------------------------------------------------
# Async behavior
# ---------------------------------------------------------------------------

def test_async_success_first_try():
    counter = AsyncCounter()
    result = asyncio.run(
        _retry_async(counter, constant(1), max_tries=3, sleep=no_sleep)
    )
    assert result == "ok"
    assert counter.calls == 1


def test_async_retries_and_succeeds():
    counter = AsyncCounter(fail_times=2)
    result = asyncio.run(
        _retry_async(
            counter,
            constant(1),
            exception=ValueError,
            max_tries=5,
            sleep=no_sleep,
        )
    )
    assert result == "ok"
    assert counter.calls == 3


def test_async_gives_up():
    counter = AsyncCounter(fail_times=10)
    with pytest.raises(ValueError):
        asyncio.run(
            _retry_async(
                counter,
                constant(1),
                exception=ValueError,
                max_tries=3,
                sleep=no_sleep,
            )
        )
    assert counter.calls == 3


def test_async_forwards_args_kwargs():
    received = {}

    async def target(a, b=0):
        received.update({"a": a, "b": b})
        return a + b

    result = asyncio.run(
        _retry_async(
            target,
            constant(1),
            max_tries=1,
            sleep=no_sleep,
            args=(1,),
            kwargs={"b": 2},
        )
    )
    assert result == 3
    assert received == {"a": 1, "b": 2}


def test_async_uses_explicit_sleep():
    delays = []

    async def fake_sleep(d):
        delays.append(d)

    counter = AsyncCounter(fail_times=2)
    asyncio.run(
        _retry_async(
            counter,
            constant(5),
            exception=ValueError,
            max_tries=5,
            sleep=fake_sleep,
            jitter=None,
        )
    )
    assert delays == [5, 5]


def test_async_retry_error_callback():
    counter = AsyncCounter(fail_times=10)

    def cb(details):
        return "cb"

    result = asyncio.run(
        _retry_async(
            counter,
            constant(1),
            exception=ValueError,
            max_tries=2,
            sleep=no_sleep,
            raise_on_giveup=False,
            retry_error_callback=cb,
        )
    )
    assert result == "cb"


def test_async_logs_on_backoff(caplog):
    counter = AsyncCounter(fail_times=1)
    with caplog.at_level(logging.INFO, logger="backon"):
        asyncio.run(
            _retry_async(
                counter,
                constant(1),
                exception=ValueError,
                max_tries=3,
                sleep=no_sleep,
            )
        )
    assert any("Backing off" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Disabled behavior (is_enabled False)
# ---------------------------------------------------------------------------

def test_sync_disabled_calls_target_directly(monkeypatch):
    import backon._common as common

    monkeypatch.setattr(common, "_GLOBAL_ENABLED", False, raising=False)

    counter = Counter(fail_times=5)
    result = _retry_sync(counter, constant(1), exception=ValueError, max_tries=3)
    # Even though it would fail, only one call is made (no retry).
    assert result == "ok"
    assert counter.calls == 1


def test_sync_disabled_returns_failure_directly(monkeypatch):
    import backon._common as common

    monkeypatch.setattr(common, "_GLOBAL_ENABLED", False, raising=False)

    counter = Counter(fail_times=5)
    with pytest.raises(ValueError):
        _retry_sync(counter, constant(1), exception=ValueError, max_tries=3)
    assert counter.calls == 1


def test_async_disabled_calls_target_directly(monkeypatch):
    import backon._common as common

    monkeypatch.setattr(common, "_GLOBAL_ENABLED", False, raising=False)

    counter = AsyncCounter(fail_times=5)
    result = asyncio.run(_retry_async(counter, constant(1), max_tries=3))
    assert result == "ok"
    assert counter.calls == 1


# ---------------------------------------------------------------------------
# _apply_test_overrides interaction
# ---------------------------------------------------------------------------

def test_sync_uses_test_override_max_retries(monkeypatch):
    import backon._common as common

    monkeypatch.setattr(common, "_TEST_CONFIG", {"max_retries": 1}, raising=False)

    counter = Counter(fail_times=10)
    with pytest.raises(ValueError):
        _retry_sync(
            counter,
            constant(1),
            exception=ValueError,
            max_tries=100,
            sleep=no_sleep,
        )
    # override forces max_tries=1
    assert counter.calls == 1


# ---------------------------------------------------------------------------
# Giveup predicate / exception types
# ---------------------------------------------------------------------------

def test_sync_giveup_predicate_prevents_retries():
    counter = Counter(fail_times=10, exc=RuntimeError)

    with pytest.raises(RuntimeError):
        _retry_sync(
            counter,
            constant(1),
            exception=RuntimeError,
            giveup=lambda e: True,
            max_tries=5,
            sleep=no_sleep,
        )
    assert counter.calls == 1


def test_sync_exception_tuple_matches():
    counter = Counter(fail_times=1, exc=KeyError)
    result = _retry_sync(
        counter,
        constant(1),
        exception=(ValueError, KeyError),
        max_tries=3,
        sleep=no_sleep,
    )
    assert result == "ok"
    assert counter.calls == 2


def test_sync_jitter_none_no_jitter():
    delays = []

    def fake_sleep(d):
        delays.append(d)

    counter = Counter(fail_times=1)
    _retry_sync(
        counter,
        constant(7),
        exception=ValueError,
        max_tries=3,
        sleep=fake_sleep,
        jitter=None,
    )
    assert delays == [7]
