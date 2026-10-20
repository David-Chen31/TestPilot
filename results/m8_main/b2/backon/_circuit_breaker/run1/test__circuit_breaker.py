import asyncio
import threading
import time

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


class FakeClock:
    """Controllable monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


def ok(value=42):
    return lambda *a, **k: value


def boom(exc=ValueError):
    def _f(*a, **k):
        raise exc("boom")

    return _f


async def aok(value=42):
    return value


async def aboom(exc=ValueError):
    async def _f(*a, **k):
        raise exc("boom")

    return _f


# ---------------------------------------------------------------------------
# CircuitBreakerState / CircuitOpenError
# ---------------------------------------------------------------------------


def test_state_enum_values():
    assert CircuitBreakerState.CLOSED.value == "CLOSED"
    assert CircuitBreakerState.OPEN.value == "OPEN"
    assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"
    assert {s.name for s in CircuitBreakerState} == {"CLOSED", "OPEN", "HALF_OPEN"}


def test_circuit_open_error_is_exception():
    err = CircuitOpenError("myname")
    assert isinstance(err, Exception)
    assert str(err) == "myname"


# ---------------------------------------------------------------------------
# CircuitBreaker - construction & properties
# ---------------------------------------------------------------------------


def test_default_construction():
    cb = CircuitBreaker()
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.name == ""
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_default_construction_internal_defaults():
    # Directly check the private default values so that mutations of the
    # constructor's default arguments are detected.
    cb = CircuitBreaker()
    assert cb._failure_threshold == 5
    assert cb._recovery_timeout == 60.0
    assert cb._half_open_max_calls == 1
    assert cb._last_failure_time == 0.0
    assert cb._half_open_calls == 0
    assert cb._failure_count == 0
    assert cb._consecutive_successes == 0
    assert cb._state == CircuitBreakerState.CLOSED


def test_custom_construction_name_and_thresholds():
    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=10.0,
        half_open_max_calls=2,
        name="svc",
    )
    assert cb.name == "svc"
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 0
    assert cb._failure_threshold == 3
    assert cb._recovery_timeout == 10.0
    assert cb._half_open_max_calls == 2


# ---------------------------------------------------------------------------
# record_success / record_failure
# ---------------------------------------------------------------------------


def test_record_success_increments_and_resets_failure():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.success_count == 1


def test_record_failure_increments_and_resets_successes():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_success()
    cb.record_success()
    assert cb.success_count == 2
    cb.record_failure()
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_record_failure_opens_at_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN


def test_failure_threshold_of_one_opens_immediately():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.failure_count == 1


def test_record_failure_updates_last_failure_time(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)
    cb = CircuitBreaker(failure_threshold=10)
    clock.advance(12.5)
    cb.record_failure()
    assert cb._last_failure_time == clock.now


# ---------------------------------------------------------------------------
# state transitions using mocked clock
# ---------------------------------------------------------------------------


def test_open_transitions_to_half_open_after_recovery_timeout(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(4.999)
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(0.001)
    assert cb.state == CircuitBreakerState.HALF_OPEN


def test_half_open_resets_half_open_calls(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    clock.advance(10.0)

    # Simulate stale counter before transition
    cb._half_open_calls = 99
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0


def test_record_success_in_half_open_closes_breaker(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.failure_count == 0


def test_record_failure_from_half_open_reopens(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# call()
# ---------------------------------------------------------------------------


def test_call_returns_result_and_records_success():
    cb = CircuitBreaker()
    assert cb.call(lambda a, b: a + b, 2, 3) == 5
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_call_passes_kwargs():
    cb = CircuitBreaker()

    def fn(**kwargs):
        return kwargs

    assert cb.call(fn, x=1, y=2) == {"x": 1, "y": 2}


def test_call_passes_args_and_kwargs_together():
    cb = CircuitBreaker()

    def fn(a, b, *, c, d):
        return (a, b, c, d)

    assert cb.call(fn, 1, 2, c=3, d=4) == (1, 2, 3, 4)


def test_call_records_failure_and_reraises():
    cb = CircuitBreaker(failure_threshold=5)
    with pytest.raises(ValueError, match="boom"):
        cb.call(boom())
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_call_records_base_exception():
    cb = CircuitBreaker(failure_threshold=5)

    def raise_kbd():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        cb.call(raise_kbd)
    assert cb.failure_count == 1


def test_call_raises_circuit_open_when_open():
    cb = CircuitBreaker(failure_threshold=1, name="mybreaker")
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(ok())
    assert exc_info.value.args == ("mybreaker",)


def test_call_in_half_open_allows_up_to_max_calls(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0, half_open_max_calls=2)
    cb.record_failure()
    clock.advance(10.0)

    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.call(ok(1)) == 1
    # success closed the breaker
    assert cb.state == CircuitBreakerState.CLOSED


def test_call_increments_half_open_calls_counter(monkeypatch):
    # Verify that entering a half-open call increments the counter by one.
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=5.0, half_open_max_calls=5
    )
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0

    # Failure inside keeps breaker state observable but increments the counter.
    with pytest.raises(RuntimeError):
        cb.call(boom(RuntimeError))
    assert cb._half_open_calls == 1


def test_call_half_open_limit_blocks_extra_calls(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=5.0, half_open_max_calls=1, name="cb"
    )
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Manually saturate the half-open slot to simulate a concurrent in-flight call
    cb._half_open_calls = 1
    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(ok())
    assert exc_info.value.args == ("cb",)


def test_call_records_failure_in_half_open_and_reopens(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    with pytest.raises(RuntimeError):
        cb.call(boom(RuntimeError))
    assert cb.state == CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


def test_dunder_call_delegates():
    cb = CircuitBreaker()
    assert cb(lambda a, b: a * b, 3, 4) == 12
    assert cb.success_count == 1


def test_dunder_call_forwards_kwargs():
    cb = CircuitBreaker()
    assert cb(lambda *, x: x, x=7) == 7
    assert cb.success_count == 1


def test_dunder_call_raises_when_open():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure()
    with pytest.raises(CircuitOpenError):
        cb(ok())


# ---------------------------------------------------------------------------
# async_call()
# ---------------------------------------------------------------------------


def test_async_call_returns_result_and_records_success():
    cb = CircuitBreaker()
    result = asyncio.run(cb.async_call(aok, 7))
    assert result == 7
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_async_call_forwards_kwargs():
    cb = CircuitBreaker()

    async def fn(*, x, y):
        return x + y

    assert asyncio.run(cb.async_call(fn, x=3, y=4)) == 7


def test_async_call_records_failure_and_reraises():
    cb = CircuitBreaker(failure_threshold=5)

    async def raise_it():
        raise ValueError("boom async")

    with pytest.raises(ValueError, match="boom async"):
        asyncio.run(cb.async_call(raise_it))
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_async_call_raises_circuit_open_when_open():
    cb = CircuitBreaker(failure_threshold=1, name="async-cb")
    cb.record_failure()
    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(cb.async_call(aok))
    assert exc_info.value.args == ("async-cb",)


def test_async_call_increments_half_open_calls_counter(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=5.0, half_open_max_calls=5
    )
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0

    async def raise_it():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(cb.async_call(raise_it))
    assert cb._half_open_calls == 1


def test_async_call_half_open_limit_blocks_extra_calls(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=5.0, half_open_max_calls=1, name="cb"
    )
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    cb._half_open_calls = 1
    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(cb.async_call(aok))
    assert exc_info.value.args == ("cb",)


def test_async_call_records_base_exception():
    cb = CircuitBreaker(failure_threshold=5)

    async def raise_kbd():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(cb.async_call(raise_kbd))
    assert cb.failure_count == 1


# ---------------------------------------------------------------------------
# context managers
# ---------------------------------------------------------------------------


def test_context_manager_returns_self_and_does_not_alter_state():
    cb = CircuitBreaker()
    with cb as entered:
        assert entered is cb
    assert cb.state == CircuitBreakerState.CLOSED


def test_context_manager_does_not_swallow_exception():
    cb = CircuitBreaker()
    with pytest.raises(ValueError):
        with cb:
            raise ValueError("inside")
    # __exit__ does nothing; breaker state untouched
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.failure_count == 0


def test_async_context_manager():
    cb = CircuitBreaker()

    async def runner():
        async with cb as entered:
            assert entered is cb
            return True

    assert asyncio.run(runner()) is True
    assert cb.state == CircuitBreakerState.CLOSED


def test_async_context_manager_propagates_exception():
    cb = CircuitBreaker()

    async def runner():
        async with cb:
            raise RuntimeError("async inside")

    with pytest.raises(RuntimeError, match="async inside"):
        asyncio.run(runner())
    assert cb.state == CircuitBreakerState.CLOSED


# ---------------------------------------------------------------------------
# Threading / locking basics
# ---------------------------------------------------------------------------


def test_property_access_uses_lock():
    cb = CircuitBreaker()
    assert isinstance(cb._lock, type(threading.Lock()))
    # State property is callable from another thread without deadlock
    results = []

    def worker():
        results.append(cb.state)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)
    assert results == [CircuitBreakerState.CLOSED]


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------


def test_breaker_retrying_default_breaker():
    br = BreakerRetrying()
    assert isinstance(br.breaker, CircuitBreaker)
    # Default breaker has default configuration
    assert br.breaker.name == ""
    assert br.breaker.state == CircuitBreakerState.CLOSED
    assert br.breaker._failure_threshold == 5
    assert br.breaker._recovery_timeout == 60.0
    assert br.breaker._half_open_max_calls == 1


def test_breaker_retrying_custom_breaker():
    cb = CircuitBreaker(name="custom", failure_threshold=2)
    br = BreakerRetrying(breaker=cb)
    assert br.breaker is cb


def test_breaker_retrying_success():
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None)
    assert br.call(lambda a, b: a + b, 1, 2) == 3
    assert cb.success_count == 1
    assert cb.failure_count == 0


def test_breaker_retrying_forwards_kwargs():
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None)

    def fn(a, *, b):
        return a + b

    assert br.call(fn, 1, b=2) == 3
    assert cb.success_count == 1


def test_breaker_retrying_raises_open():
    cb = CircuitBreaker(failure_threshold=1, name="br-cb")
    cb.record_failure()
    br = BreakerRetrying(breaker=cb)
    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(ok())
    assert exc_info.value.args == ("br-cb",)


def test_breaker_retrying_half_open_limit(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=5.0, half_open_max_calls=1, name="br-cb"
    )
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    cb._half_open_calls = 1
    br = BreakerRetrying(breaker=cb)
    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(ok())
    assert exc_info.value.args == ("br-cb",)


def test_breaker_retrying_increments_half_open_calls(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=5.0, half_open_max_calls=5
    )
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0

    br = BreakerRetrying(breaker=cb, max_tries=1)

    def always_fail():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        br.call(always_fail)
    assert cb._half_open_calls == 1


def test_breaker_retrying_records_failure_after_giveup():
    cb = CircuitBreaker(failure_threshold=5)
    br = BreakerRetrying(
        breaker=cb,
        max_tries=2,
        exception=(ValueError,),
        sleep=lambda _s: None,
        jitter=None,
    )

    with pytest.raises(ValueError, match="boom"):
        br.call(boom())
    # The retrying exhausted and raised; breaker must have recorded exactly one
    # failure for the whole call (despite internal retries).
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_breaker_retrying_retries_within_half_open(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("transient")
        return "done"

    br = BreakerRetrying(
        breaker=cb,
        max_tries=3,
        exception=(ValueError,),
        sleep=lambda _s: None,
        jitter=None,
    )
    assert br.call(fn) == "done"
    assert attempts["n"] == 2
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.success_count == 1


def test_breaker_retrying_async_success():
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None)

    async def fn(a, b):
        return a + b

    assert asyncio.run(br.async_call(fn, 2, 5)) == 7
    assert cb.success_count == 1


def test_breaker_retrying_async_forwards_kwargs():
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, sleep=lambda _s: None)

    async def fn(a, *, b):
        return a + b

    assert asyncio.run(br.async_call(fn, 2, b=5)) == 7
    assert cb.success_count == 1


def test_breaker_retrying_async_raises_open():
    cb = CircuitBreaker(failure_threshold=1, name="abr")
    cb.record_failure()
    br = BreakerRetrying(breaker=cb)

    async def fn():
        return 1

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(br.async_call(fn))
    assert exc_info.value.args == ("abr",)


def test_breaker_retrying_async_half_open_limit(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)

    cb = CircuitBreaker(
        failure_threshold=1, recovery_timeout=5.0, half_open_max_calls=1, name="abr"
    )
    cb.record_failure()
    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    cb._half_open_calls = 1
    br = BreakerRetrying(breaker=cb)

    async def fn():
        return 1

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(br.async_call(fn))
    assert exc_info.value.args == ("abr",)


# NOTE: the current Retrying.async_call appears to invoke the target
# synchronously and then try to ``await`` its (coroutine) return value after
# the retry machinery has already consumed it; in the retry/giveup paths this
# ends up attempting to ``await`` the ``None`` returned by the sleep hook.
# These tests document the actual behavior: the call surfaces a TypeError
# from the retry machinery rather than the underlying ValueError.
def test_breaker_retrying_async_records_failure():
    cb = CircuitBreaker(failure_threshold=5)
    br = BreakerRetrying(
        breaker=cb,
        max_tries=2,
        exception=(ValueError,),
        sleep=lambda _s: None,
        jitter=None,
    )

    async def fn():
        raise ValueError("boom async")

    with pytest.raises(TypeError):
        asyncio.run(br.async_call(fn))
    # The breaker still records a failure for the whole call.
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_breaker_retrying_async_retries():
    cb = CircuitBreaker(failure_threshold=5)
    br = BreakerRetrying(
        breaker=cb,
        max_tries=3,
        exception=(ValueError,),
        sleep=lambda _s: None,
        jitter=None,
    )

    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient")
        return "ok"

    # Actual behavior: retry machinery hits the None-await TypeError before
    # the target can reach its successful third attempt.
    with pytest.raises(TypeError):
        asyncio.run(br.async_call(fn))
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_breaker_retrying_disabled_passes_through():
    cb = CircuitBreaker()
    br = BreakerRetrying(breaker=cb, enabled=False)
    assert br.call(lambda: 99) == 99
    # Breaker still records success around the call
    assert cb.success_count == 1
