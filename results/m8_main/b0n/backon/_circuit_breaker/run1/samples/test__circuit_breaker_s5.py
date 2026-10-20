from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from backon._circuit_breaker import (
    BreakerRetrying,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitOpenError,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class FakeClock:
    """Deterministic stand-in for time.monotonic used by the breaker."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


@pytest.fixture
def clock(monkeypatch):
    fc = FakeClock()
    monkeypatch.setattr("backon._circuit_breaker.time.monotonic", fc)
    return fc


def make_breaker(**kwargs) -> CircuitBreaker:
    kwargs.setdefault("name", "test-breaker")
    return CircuitBreaker(**kwargs)


# --------------------------------------------------------------------------- #
# CircuitBreakerState / CircuitOpenError
# --------------------------------------------------------------------------- #

def test_state_enum_values():
    assert CircuitBreakerState.CLOSED.value == "CLOSED"
    assert CircuitBreakerState.OPEN.value == "OPEN"
    assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"
    assert CircuitBreakerState("CLOSED") is CircuitBreakerState.CLOSED


def test_circuit_open_error_is_exception():
    err = CircuitOpenError("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"


# --------------------------------------------------------------------------- #
# CircuitBreaker basics
# --------------------------------------------------------------------------- #

def test_initial_state_and_counters(clock):
    cb = make_breaker()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.name == "test-breaker"
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_default_name_empty():
    assert CircuitBreaker().name == ""


def test_record_failure_below_threshold_stays_closed(clock):
    cb = make_breaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    assert cb.success_count == 0
    assert cb.state is CircuitBreakerState.CLOSED


def test_record_failure_reaches_threshold_opens(clock):
    cb = make_breaker(failure_threshold=2)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.failure_count == 2
    assert cb.state is CircuitBreakerState.OPEN


def test_record_success_resets_failure_count_and_increments_successes(clock):
    cb = make_breaker(failure_threshold=5)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.success_count == 1
    cb.record_success()
    assert cb.success_count == 2
    assert cb.state is CircuitBreakerState.CLOSED


def test_record_failure_resets_success_count(clock):
    cb = make_breaker(failure_threshold=5)
    cb.record_success()
    cb.record_success()
    assert cb.success_count == 2
    cb.record_failure()
    assert cb.success_count == 0
    assert cb.failure_count == 1


# --------------------------------------------------------------------------- #
# Recovery / half-open transitions
# --------------------------------------------------------------------------- #

def test_open_transitions_to_half_open_after_recovery_timeout(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=10.0)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN

    clock.advance(9.9)
    assert cb.state is CircuitBreakerState.OPEN

    clock.advance(0.1)
    assert cb.state is CircuitBreakerState.HALF_OPEN


def test_state_property_half_open_resets_half_open_calls(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0,
                      half_open_max_calls=5)
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    # half_open_calls should have been reset to 0
    assert cb._half_open_calls == 0


def test_record_success_in_half_open_closes(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0)
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 1


# --------------------------------------------------------------------------- #
# CircuitBreaker.call
# --------------------------------------------------------------------------- #

def test_call_success_returns_value_and_records(clock):
    cb = make_breaker()
    result = cb.call(lambda a, b=0: a + b, 2, b=3)
    assert result == 5
    assert cb.success_count == 1
    assert cb.failure_count == 0
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_passes_args_and_kwargs(clock):
    cb = make_breaker()
    seen = {}

    def fn(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "ok"

    assert cb.call(fn, 1, 2, x=3, y=4) == "ok"
    assert seen == {"args": (1, 2), "kwargs": {"x": 3, "y": 4}}


def test_call_failure_records_and_reraises(clock):
    cb = make_breaker(failure_threshold=5)

    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        cb.call(boom)
    assert cb.failure_count == 1
    assert cb.success_count == 0
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_opens_after_enough_failures(clock):
    cb = make_breaker(failure_threshold=2)

    def boom():
        raise RuntimeError("x")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.state is CircuitBreakerState.OPEN


def test_call_raises_when_open(clock):
    cb = make_breaker(failure_threshold=1, name="my-breaker")
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert cb.state is CircuitBreakerState.OPEN

    fn = MagicMock(return_value="never")
    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(fn)
    assert str(exc_info.value) == "my-breaker"
    fn.assert_not_called()


def test_call_base_exception_is_recorded_and_reraised(clock):
    cb = make_breaker(failure_threshold=5)

    def boom():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cb.call(boom)
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_call_half_open_limits_calls(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0,
                      half_open_max_calls=1, name="ho")
    # open it
    with pytest.raises(ValueError):
        cb.call(lambda: (_ for _ in ()).throw(ValueError()))
    assert cb.state is CircuitBreakerState.OPEN

    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    # First call in half-open increments the slot count then the call
    # fails, which re-opens the breaker.
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        cb.call(boom)

    # Since it failed the breaker is open again.
    assert cb.state is CircuitBreakerState.OPEN


def test_call_half_open_max_calls_exceeded(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0,
                      half_open_max_calls=1, name="ho")
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    # Manually bump the slot count to simulate an in-flight call.
    cb._half_open_calls = 1

    fn = MagicMock(return_value="x")
    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(fn)
    assert str(exc_info.value) == "ho"
    fn.assert_not_called()


def test_call_half_open_success_closes(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0,
                      half_open_max_calls=1)
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    assert cb.call(lambda: 42) == 42
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 1


# --------------------------------------------------------------------------- #
# CircuitBreaker.async_call
# --------------------------------------------------------------------------- #

def test_async_call_success(clock):
    cb = make_breaker()

    async def coro(a, b=0):
        return a + b

    result = asyncio.run(cb.async_call(coro, 2, b=3))
    assert result == 5
    assert cb.success_count == 1
    assert cb.state is CircuitBreakerState.CLOSED


def test_async_call_failure_records(clock):
    cb = make_breaker(failure_threshold=5)

    async def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        asyncio.run(cb.async_call(boom))
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_async_call_open_raises(clock):
    cb = make_breaker(failure_threshold=1, name="async-breaker")
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN

    async def coro():
        return "never"

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(cb.async_call(coro))
    assert str(exc_info.value) == "async-breaker"


def test_async_call_half_open_closes_on_success(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0)
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    async def coro():
        return "ok"

    assert asyncio.run(cb.async_call(coro)) == "ok"
    assert cb.state is CircuitBreakerState.CLOSED


def test_async_call_half_open_limit(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0,
                      half_open_max_calls=1, name="ho")
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1

    async def coro():
        return "never"

    with pytest.raises(CircuitOpenError):
        asyncio.run(cb.async_call(coro))


# --------------------------------------------------------------------------- #
# __call__ and context manager protocols
# --------------------------------------------------------------------------- #

def test_dunder_call_delegates_to_call(clock):
    cb = make_breaker()
    assert cb(lambda x: x * 2, 21) == 42
    assert cb.success_count == 1


def test_context_manager_sync(clock):
    cb = make_breaker()
    with cb as ctx:
        assert ctx is cb


def test_context_manager_sync_does_not_record_on_exception(clock):
    cb = make_breaker()
    with pytest.raises(ValueError):
        with cb:
            raise ValueError("x")
    # __exit__ ignores exceptions, so nothing recorded
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_context_manager_async(clock):
    async def run():
        cb = make_breaker()
        async with cb as ctx:
            assert ctx is cb

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# BreakerRetrying
# --------------------------------------------------------------------------- #

def _no_sleep(value: float) -> None:
    return None


def test_breaker_retrying_default_creates_breaker():
    br = BreakerRetrying(sleep=_no_sleep)
    assert isinstance(br.breaker, CircuitBreaker)
    assert br.breaker.name == ""


def test_breaker_retrying_uses_provided_breaker(clock):
    cb = make_breaker(name="p")
    br = BreakerRetrying(breaker=cb, sleep=_no_sleep)
    assert br.breaker is cb


def test_breaker_retrying_success(clock):
    br = BreakerRetrying(sleep=_no_sleep)
    result = br.call(lambda a, b=0: a + b, 1, b=2)
    assert result == 3
    assert br.breaker.success_count == 1
    assert br.breaker.state is CircuitBreakerState.CLOSED


def test_breaker_retrying_retries_until_success(clock):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("fail")
        return "done"

    br = BreakerRetrying(
        sleep=_no_sleep,
        predicate=operator_not_truthy,
        exception=ValueError,
        max_tries=5,
        jitter=None,
    )
    assert br.call(flaky) == "done"
    assert attempts["n"] == 3
    assert br.breaker.success_count == 1


def operator_not_truthy(x):
    # predicate: retry when result falsy -> treat as failure
    return not x


def test_breaker_retrying_failure_open_on_threshold(clock):
    br = BreakerRetrying(
        breaker=make_breaker(failure_threshold=2, name="br"),
        sleep=_no_sleep,
        exception=ValueError,
        max_tries=1,
        jitter=None,
    )

    def boom():
        raise ValueError("no")

    with pytest.raises(ValueError):
        br.call(boom)
    assert br.breaker.failure_count == 1
    assert br.breaker.state is CircuitBreakerState.CLOSED

    with pytest.raises(ValueError):
        br.call(boom)
    assert br.breaker.failure_count == 2
    assert br.breaker.state is CircuitBreakerState.OPEN


def test_breaker_retrying_raises_when_open(clock):
    cb = make_breaker(failure_threshold=1, name="open-br")
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN

    br = BreakerRetrying(breaker=cb, sleep=_no_sleep)
    fn = MagicMock(return_value="never")
    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(fn)
    assert str(exc_info.value) == "open-br"
    fn.assert_not_called()


def test_breaker_retrying_half_open_success_closes(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0,
                      half_open_max_calls=1)
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    br = BreakerRetrying(breaker=cb, sleep=_no_sleep)
    assert br.call(lambda: "ok") == "ok"
    assert cb.state is CircuitBreakerState.CLOSED


def test_breaker_retrying_half_open_limit(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0,
                      half_open_max_calls=1, name="ho-br")
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1

    br = BreakerRetrying(breaker=cb, sleep=_no_sleep)
    fn = MagicMock(return_value="never")
    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(fn)
    assert str(exc_info.value) == "ho-br"
    fn.assert_not_called()


def test_breaker_retrying_async_success(clock):
    br = BreakerRetrying(sleep=_no_sleep)

    async def coro(a, b=0):
        return a + b

    assert asyncio.run(br.async_call(coro, 2, b=3)) == 5
    assert br.breaker.success_count == 1


def test_breaker_retrying_async_failure(clock):
    br = BreakerRetrying(
        breaker=make_breaker(failure_threshold=5),
        sleep=_no_sleep,
        exception=ValueError,
        max_tries=1,
        jitter=None,
    )

    async def boom():
        raise ValueError("async nope")

    with pytest.raises(ValueError, match="async nope"):
        asyncio.run(br.async_call(boom))
    assert br.breaker.failure_count == 1


def test_breaker_retrying_async_open_raises(clock):
    cb = make_breaker(failure_threshold=1, name="async-open-br")
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN

    br = BreakerRetrying(breaker=cb, sleep=_no_sleep)

    async def coro():
        return "never"

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(br.async_call(coro))
    assert str(exc_info.value) == "async-open-br"


def test_breaker_retrying_async_half_open(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0,
                      half_open_max_calls=1)
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    br = BreakerRetrying(breaker=cb, sleep=_no_sleep)

    async def coro():
        return "ok"

    assert asyncio.run(br.async_call(coro)) == "ok"
    assert cb.state is CircuitBreakerState.CLOSED


def test_breaker_retrying_async_half_open_limit(clock):
    cb = make_breaker(failure_threshold=1, recovery_timeout=1.0,
                      half_open_max_calls=1, name="ho-async-br")
    cb.record_failure()
    clock.advance(2.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb._half_open_calls = 1

    br = BreakerRetrying(breaker=cb, sleep=_no_sleep)

    async def coro():
        return "never"

    with pytest.raises(CircuitOpenError):
        asyncio.run(br.async_call(coro))
