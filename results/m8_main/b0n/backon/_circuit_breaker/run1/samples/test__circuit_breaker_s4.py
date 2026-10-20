import asyncio
import logging
import operator
import threading

import pytest

from backon._circuit_breaker import (
    BreakerRetrying,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitOpenError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(value=42):
    return value


def boom(exc=ValueError):
    raise exc("boom")


async def aok(value=42):
    return value


async def aboom(exc=ValueError):
    raise exc("boom")


class FakeClock:
    """Deterministic monotonic replacement."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, delta):
        self.now += delta


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr("backon._circuit_breaker.time.monotonic", fake)
    return fake


# ---------------------------------------------------------------------------
# CircuitBreakerState
# ---------------------------------------------------------------------------

def test_state_enum_values():
    assert CircuitBreakerState.CLOSED.value == "CLOSED"
    assert CircuitBreakerState.OPEN.value == "OPEN"
    assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"
    assert set(CircuitBreakerState) == {
        CircuitBreakerState.CLOSED,
        CircuitBreakerState.OPEN,
        CircuitBreakerState.HALF_OPEN,
    }


def test_circuit_open_error_is_exception():
    assert issubclass(CircuitOpenError, Exception)
    err = CircuitOpenError("name")
    assert str(err) == "name"


# ---------------------------------------------------------------------------
# CircuitBreaker construction / defaults
# ---------------------------------------------------------------------------

def test_defaults():
    cb = CircuitBreaker()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.name == ""
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_custom_params_and_name():
    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=10.0,
        half_open_max_calls=2,
        name="svc",
    )
    assert cb.name == "svc"
    assert cb._failure_threshold == 3
    assert cb._recovery_timeout == 10.0
    assert cb._half_open_max_calls == 2


# ---------------------------------------------------------------------------
# record_success / record_failure in CLOSED state
# ---------------------------------------------------------------------------

def test_record_success_resets_failure_count(clock):
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.success_count == 1
    assert cb.state is CircuitBreakerState.CLOSED


def test_record_failure_increments_and_resets_successes(clock):
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_success()
    assert cb.success_count == 1
    cb.record_failure()
    assert cb.failure_count == 1
    assert cb.success_count == 0
    assert cb.state is CircuitBreakerState.CLOSED


def test_failure_count_tracks_and_opens_at_threshold(clock):
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    assert cb.state is CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.failure_count == 3
    assert cb.state is CircuitBreakerState.OPEN


def test_threshold_one_opens_immediately(clock):
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# state transitions with recovery_timeout
# ---------------------------------------------------------------------------

def test_state_stays_open_before_recovery_timeout(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    clock.advance(59.999)
    assert cb.state is CircuitBreakerState.OPEN


def test_state_transitions_to_half_open_after_timeout(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    clock.advance(60.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    # Stays half-open on repeated reads
    assert cb.state is CircuitBreakerState.HALF_OPEN


def test_half_open_calls_reset_on_transition(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0, half_open_max_calls=2)
    cb.record_failure()
    cb._half_open_calls = 99  # simulate stale value
    clock.advance(5.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0


def test_record_success_in_half_open_closes(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    clock.advance(5.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 1


def test_record_failure_in_half_open_reopens(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    clock.advance(5.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    clock.advance(1.0)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    # timeout now measured from the new failure time
    clock.advance(4.0)
    assert cb.state is CircuitBreakerState.OPEN
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN


# ---------------------------------------------------------------------------
# CircuitBreaker.call
# ---------------------------------------------------------------------------

def test_call_success(clock):
    cb = CircuitBreaker(failure_threshold=3)
    result = cb.call(ok, 10)
    assert result == 10
    assert cb.failure_count == 0
    assert cb.success_count == 1
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_passes_kwargs(clock):
    cb = CircuitBreaker()

    def double(x, *, factor=2):
        return x * factor

    assert cb.call(double, 3, factor=5) == 15


def test_call_failure_records_and_reraises(clock):
    cb = CircuitBreaker(failure_threshold=3)
    with pytest.raises(ValueError, match="boom"):
        cb.call(boom)
    assert cb.failure_count == 1
    assert cb.success_count == 0
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_reaches_threshold_opens(clock):
    cb = CircuitBreaker(failure_threshold=2)
    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(boom)
    assert cb.state is CircuitBreakerState.OPEN


def test_call_when_open_raises_circuit_open(clock):
    cb = CircuitBreaker(failure_threshold=1, name="myservice")
    with pytest.raises(ValueError):
        cb.call(boom)
    assert cb.state is CircuitBreakerState.OPEN

    called = []
    with pytest.raises(CircuitOpenError) as excinfo:
        cb.call(lambda: called.append(1))
    assert called == []
    assert excinfo.value.args[0] == "myservice"


def test_call_catches_base_exception(clock):
    cb = CircuitBreaker(failure_threshold=2)

    def raise_base():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        cb.call(raise_base)
    assert cb.failure_count == 1


def test_call_half_open_limited_calls(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1)
    with pytest.raises(ValueError):
        cb.call(boom)
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN

    # Manually occupy the single half-open slot.
    with cb._lock:
        cb._half_open_calls = 1
    with pytest.raises(CircuitOpenError):
        cb.call(ok)


def test_call_half_open_success_closes(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=2)
    with pytest.raises(ValueError):
        cb.call(boom)
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb.call(ok, 7) == 7
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_half_open_failure_reopens(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=2)
    with pytest.raises(ValueError):
        cb.call(boom)
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    with pytest.raises(ValueError):
        cb.call(boom)
    assert cb.state is CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# CircuitBreaker.async_call
# ---------------------------------------------------------------------------

def test_async_call_success(clock):
    cb = CircuitBreaker()
    result = asyncio.run(cb.async_call(aok, 5))
    assert result == 5
    assert cb.success_count == 1
    assert cb.state is CircuitBreakerState.CLOSED


def test_async_call_failure(clock):
    cb = CircuitBreaker(failure_threshold=2)
    with pytest.raises(ValueError):
        asyncio.run(cb.async_call(aboom))
    assert cb.failure_count == 1


def test_async_call_opens(clock):
    cb = CircuitBreaker(failure_threshold=1, name="async-svc")

    async def go():
        for _ in range(1):
            with pytest.raises(ValueError):
                await cb.async_call(aboom)
        with pytest.raises(CircuitOpenError) as excinfo:
            await cb.async_call(aok)
        return excinfo.value

    err = asyncio.run(go())
    assert err.args[0] == "async-svc"


def test_async_call_half_open_success_closes(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)

    async def go():
        with pytest.raises(ValueError):
            await cb.async_call(aboom)
        clock.advance(1.0)
        assert cb.state is CircuitBreakerState.HALF_OPEN
        return await cb.async_call(aok, 3)

    assert asyncio.run(go()) == 3
    assert cb.state is CircuitBreakerState.CLOSED


def test_async_call_half_open_limit(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1)

    async def go():
        with pytest.raises(ValueError):
            await cb.async_call(aboom)
        clock.advance(1.0)
        with cb._lock:
            cb._half_open_calls = 1
        with pytest.raises(CircuitOpenError):
            await cb.async_call(aok)

    asyncio.run(go())


def test_async_call_catches_cancelled_error(clock):
    cb = CircuitBreaker(failure_threshold=2)

    async def cancelled():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cb.async_call(cancelled))
    assert cb.failure_count == 1


# ---------------------------------------------------------------------------
# __call__ and context managers
# ---------------------------------------------------------------------------

def test_dunder_call_delegates(clock):
    cb = CircuitBreaker()
    assert cb(ok, 99) == 99
    assert cb.success_count == 1


def test_sync_context_manager(clock):
    cb = CircuitBreaker()
    with cb as entered:
        assert entered is cb


def test_async_context_manager(clock):
    cb = CircuitBreaker()

    async def go():
        async with cb as entered:
            assert entered is cb

    asyncio.run(go())


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------

def test_breaker_retrying_defaults():
    br = BreakerRetrying()
    assert isinstance(br.breaker, CircuitBreaker)
    assert br.breaker.state is CircuitBreakerState.CLOSED


def test_breaker_retrying_custom_breaker():
    cb = CircuitBreaker(name="custom")
    br = BreakerRetrying(breaker=cb)
    assert br.breaker is cb


def test_breaker_retrying_success_records_success(clock):
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None)
    result = br.call(ok, 7)
    assert result == 7
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_breaker_retrying_retries_then_fails_records_single_failure(clock):
    cb = CircuitBreaker(failure_threshold=5)
    br = BreakerRetrying(
        breaker=cb,
        sleep=lambda _s: None,
        max_tries=3,
        exception=ValueError,
    )
    calls = []

    def flaky():
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):
        br.call(flaky)
    assert len(calls) == 3
    assert cb.failure_count == 1


def test_breaker_retrying_eventually_succeeds(clock):
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None, max_tries=5, exception=ValueError)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError()
        return "done"

    assert br.call(flaky) == "done"
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_breaker_retrying_open_breaker_raises(clock):
    cb = CircuitBreaker(failure_threshold=1, name="br-name")
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    br = BreakerRetrying(breaker=cb)
    with pytest.raises(CircuitOpenError) as excinfo:
        br.call(ok)
    assert excinfo.value.args[0] == "br-name"


def test_breaker_retrying_half_open_limit(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1)
    cb.record_failure()
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    with cb._lock:
        cb._half_open_calls = 1
    br = BreakerRetrying(breaker=cb)
    with pytest.raises(CircuitOpenError):
        br.call(ok)


def test_breaker_retrying_half_open_success_closes(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    cb.record_failure()
    clock.advance(1.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None)
    assert br.call(ok, 11) == 11
    assert cb.state is CircuitBreakerState.CLOSED


def test_breaker_retrying_failure_opens(clock):
    cb = CircuitBreaker(failure_threshold=1)
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None, max_tries=2, exception=ValueError)
    with pytest.raises(ValueError):
        br.call(boom)
    assert cb.state is CircuitBreakerState.OPEN


def test_breaker_retrying_disabled_bypasses_retry_but_records(clock):
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, enabled=False, sleep=lambda _s: None)
    calls = []

    def flaky():
        calls.append(1)
        raise ValueError()

    with pytest.raises(ValueError):
        br.call(flaky)
    assert len(calls) == 1
    assert cb.failure_count == 1


# ---------------------------------------------------------------------------
# BreakerRetrying.async_call
# ---------------------------------------------------------------------------

def test_breaker_retrying_async_success(clock):
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None)
    result = asyncio.run(br.async_call(aok, 9))
    assert result == 9
    assert cb.success_count == 1


def test_breaker_retrying_async_failure_records(clock):
    cb = CircuitBreaker(failure_threshold=1)
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None, exception=ValueError)

    async def go():
        with pytest.raises(ValueError):
            await br.async_call(aboom)
        assert cb.state is CircuitBreakerState.OPEN
        with pytest.raises(CircuitOpenError) as excinfo:
            await br.async_call(aok)
        return excinfo.value

    err = asyncio.run(go())
    assert err.args[0] == ""


def test_breaker_retrying_async_half_open_success_closes(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)

    async def go():
        cb.record_failure()
        clock.advance(1.0)
        br = BreakerRetrying(breaker=cb, sleep=lambda _s: None)
        return await br.async_call(aok, 4)

    assert asyncio.run(go()) == 4
    assert cb.state is CircuitBreakerState.CLOSED


def test_breaker_retrying_async_half_open_limit(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1)

    async def go():
        cb.record_failure()
        clock.advance(1.0)
        with cb._lock:
            cb._half_open_calls = 1
        br = BreakerRetrying(breaker=cb, sleep=lambda _s: None)
        with pytest.raises(CircuitOpenError):
            await br.async_call(aok)

    asyncio.run(go())


def test_thread_safety_of_record_operations(clock):
    cb = CircuitBreaker(failure_threshold=1000)

    def worker():
        for _ in range(100):
            cb.record_failure()

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cb.failure_count == 500
    assert cb.state is CircuitBreakerState.CLOSED


def test_existing_lock_is_a_threading_lock():
    cb = CircuitBreaker()
    assert isinstance(cb._lock, type(threading.Lock()))
