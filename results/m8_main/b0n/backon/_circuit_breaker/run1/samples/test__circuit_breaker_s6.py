from __future__ import annotations

import asyncio
from unittest import mock

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

class _FakeClock:
    """Deterministic replacement for time.monotonic."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, delta: float) -> None:
        self._now += delta


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr("backon._circuit_breaker.time.monotonic", fake)
    return fake


def _open_breaker(breaker: CircuitBreaker) -> None:
    """Drive a breaker into the OPEN state."""
    for _ in range(breaker.failure_count if False else 0):  # pragma: no cover
        pass
    while breaker.state == CircuitBreakerState.CLOSED:
        breaker.record_failure()


# ---------------------------------------------------------------------------
# Enum / Exception
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
    err = CircuitOpenError("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"


# ---------------------------------------------------------------------------
# CircuitBreaker construction / properties
# ---------------------------------------------------------------------------

def test_initial_state():
    cb = CircuitBreaker(name="svc")
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.name == "svc"
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_default_name_is_empty():
    assert CircuitBreaker().name == ""


def test_initial_attributes_stored():
    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=12.5,
        half_open_max_calls=7,
        name="x",
    )
    assert cb._failure_threshold == 3
    assert cb._recovery_timeout == 12.5
    assert cb._half_open_max_calls == 7
    assert cb._name == "x"
    assert cb._state is CircuitBreakerState.CLOSED
    assert cb._failure_count == 0
    assert cb._last_failure_time == 0.0
    assert cb._half_open_calls == 0
    assert cb._consecutive_successes == 0


# ---------------------------------------------------------------------------
# record_success / record_failure
# ---------------------------------------------------------------------------

def test_record_success_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.success_count == 1


def test_record_success_increments_consecutive_successes():
    cb = CircuitBreaker()
    cb.record_success()
    cb.record_success()
    cb.record_success()
    assert cb.success_count == 3
    cb.record_failure()
    assert cb.success_count == 0


def test_record_failure_increments_and_tracks_time(clock):
    cb = CircuitBreaker(failure_threshold=5)
    clock.advance(5)
    cb.record_failure()
    assert cb.failure_count == 1
    assert cb._last_failure_time == clock._now
    assert cb.success_count == 0


def test_record_failure_opens_at_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitBreakerState.CLOSED
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN


def test_record_failure_beyond_threshold_keeps_open():
    cb = CircuitBreaker(failure_threshold=2)
    for _ in range(5):
        cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    assert cb.failure_count == 5


def test_record_success_in_half_open_closes():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    clock = mock.Mock(return_value=0.0)
    with mock.patch("backon._circuit_breaker.time.monotonic", clock):
        cb.record_failure()
        assert cb.state is CircuitBreakerState.OPEN
        clock.return_value = 100.0
        assert cb.state is CircuitBreakerState.HALF_OPEN
        cb.record_success()
    assert cb.state is CircuitBreakerState.CLOSED
    assert cb.failure_count == 0


def test_record_success_outside_half_open_keeps_closed():
    cb = CircuitBreaker()
    cb.record_success()
    assert cb.state is CircuitBreakerState.CLOSED


# ---------------------------------------------------------------------------
# state transition through recovery timeout
# ---------------------------------------------------------------------------

def test_state_transitions_open_to_half_open_after_timeout(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    clock.advance(5.0)
    assert cb.state is CircuitBreakerState.OPEN
    clock.advance(5.0)
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0


def test_state_half_open_resets_half_open_calls(clock):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    cb.record_failure()
    clock.advance(2.0)
    cb._half_open_calls = 99
    assert cb.state is CircuitBreakerState.HALF_OPEN
    assert cb._half_open_calls == 0


# ---------------------------------------------------------------------------
# CircuitBreaker.call
# ---------------------------------------------------------------------------

def test_call_success_returns_value_and_records_success():
    cb = CircuitBreaker()
    result = cb.call(lambda a, b=0: a + b, 3, b=4)
    assert result == 7
    assert cb.failure_count == 0
    assert cb.success_count == 1


def test_call_passes_args_and_kwargs():
    cb = CircuitBreaker()
    seen = {}

    def fn(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "ok"

    assert cb.call(fn, 1, 2, x=3) == "ok"
    assert seen == {"args": (1, 2), "kwargs": {"x": 3}}


def test_call_failure_records_failure_and_reraises():
    cb = CircuitBreaker(failure_threshold=5)
    err = ValueError("bad")

    def fn():
        raise err

    with pytest.raises(ValueError) as exc_info:
        cb.call(fn)
    assert exc_info.value is err
    assert cb.failure_count == 1
    assert cb.success_count == 0


def test_call_failure_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert cb.state is CircuitBreakerState.OPEN


def test_call_when_open_raises_circuit_open_error():
    cb = CircuitBreaker(failure_threshold=1, name="mybreaker")
    cb.record_failure()
    assert cb.state is CircuitBreakerState.OPEN
    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(lambda: 1)
    assert str(exc_info.value) == "mybreaker"


def test_call_open_error_does_not_invoke_fn():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure()
    calls = []
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: calls.append(1))
    assert calls == []


def test_call_half_open_allows_up_to_max_calls():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0,
                        half_open_max_calls=2)
    clock = mock.Mock(return_value=0.0)
    with mock.patch("backon._circuit_breaker.time.monotonic", clock):
        cb.record_failure()
        clock.return_value = 100.0
        assert cb.state is CircuitBreakerState.HALF_OPEN

        def boom():
            raise RuntimeError("half-open")

        # First two half-open calls are allowed (and each fails, keeping open).
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(boom)
        # The third is rejected because max half-open calls is exhausted.
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "never")


def test_call_half_open_success_closes_breaker():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0,
                        half_open_max_calls=1)
    clock = mock.Mock(return_value=0.0)
    with mock.patch("backon._circuit_breaker.time.monotonic", clock):
        cb.record_failure()
        clock.return_value = 100.0
        assert cb.state is CircuitBreakerState.HALF_OPEN
        result = cb.call(lambda: "done")
    assert result == "done"
    assert cb.state is CircuitBreakerState.CLOSED


def test_call_half_open_limit_uses_half_open_calls():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0,
                        half_open_max_calls=0)
    clock = mock.Mock(return_value=0.0)
    with mock.patch("backon._circuit_breaker.time.monotonic", clock):
        cb.record_failure()
        clock.return_value = 100.0
        assert cb.state is CircuitBreakerState.HALF_OPEN
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "nope")


def test_call_base_exception_recorded():
    cb = CircuitBreaker()
    exc = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        cb.call(lambda: (_ for _ in ()).throw(exc))
    assert cb.failure_count == 1


# ---------------------------------------------------------------------------
# CircuitBreaker.async_call
# ---------------------------------------------------------------------------

def test_async_call_success():
    cb = CircuitBreaker()

    async def fn(a, b=0):
        return a + b

    result = asyncio.run(cb.async_call(fn, 2, b=5))
    assert result == 7
    assert cb.success_count == 1


def test_async_call_failure_records_and_reraises():
    cb = CircuitBreaker()

    async def fn():
        raise ValueError("async bad")

    with pytest.raises(ValueError):
        asyncio.run(cb.async_call(fn))
    assert cb.failure_count == 1


def test_async_call_when_open_raises():
    cb = CircuitBreaker(failure_threshold=1, name="asyncname")
    cb.record_failure()

    async def fn():
        return 1

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(cb.async_call(fn))
    assert str(exc_info.value) == "asyncname"


def test_async_call_half_open_max_calls():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0,
                        half_open_max_calls=1)
    clock = mock.Mock(return_value=0.0)
    with mock.patch("backon._circuit_breaker.time.monotonic", clock):
        cb.record_failure()
        clock.return_value = 100.0
        assert cb.state is CircuitBreakerState.HALF_OPEN

        async def boom():
            raise RuntimeError("half")

        with pytest.raises(RuntimeError):
            asyncio.run(cb.async_call(boom))
        with pytest.raises(CircuitOpenError):
            asyncio.run(cb.async_call(boom))


# ---------------------------------------------------------------------------
# __call__, context managers
# ---------------------------------------------------------------------------

def test_dunder_call_delegates_to_call():
    cb = CircuitBreaker()
    assert cb(lambda x: x * 2, 5) == 10
    assert cb.success_count == 1


def test_sync_context_manager():
    cb = CircuitBreaker()
    with cb as ctx:
        assert ctx is cb


def test_async_context_manager():
    cb = CircuitBreaker()

    async def runner():
        async with cb as ctx:
            return ctx

    assert asyncio.run(runner()) is cb


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------

def test_breaker_retrying_default_breaker():
    br = BreakerRetrying()
    assert isinstance(br.breaker, CircuitBreaker)


def test_breaker_retrying_custom_breaker():
    cb = CircuitBreaker(name="custom")
    br = BreakerRetrying(breaker=cb)
    assert br.breaker is cb


def test_breaker_retrying_call_success():
    br = BreakerRetrying(max_tries=1, sleep=lambda _: None)
    result = br.call(lambda a, b=1: a + b, 3, b=4)
    assert result == 7
    assert br.breaker.success_count == 1
    assert br.breaker.failure_count == 0


def test_breaker_retrying_call_retries_then_succeeds():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient")
        return "success"

    br = BreakerRetrying(max_tries=5, sleep=lambda _: None)
    assert br.call(fn) == "success"
    assert attempts["n"] == 3
    assert br.breaker.success_count == 1
    assert br.breaker.failure_count == 0


def test_breaker_retrying_call_failure_records_failure():
    br = BreakerRetrying(max_tries=1, sleep=lambda _: None)

    with pytest.raises(ValueError):
        br.call(lambda: (_ for _ in ()).throw(ValueError("nope")))
    assert br.breaker.failure_count == 1


def test_breaker_retrying_call_when_open():
    cb = CircuitBreaker(failure_threshold=1, name="openbr")
    cb.record_failure()
    br = BreakerRetrying(breaker=cb, sleep=lambda _: None)
    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(lambda: 1)
    assert str(exc_info.value) == "openbr"


def test_breaker_retrying_call_half_open_limit():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0,
                        half_open_max_calls=1)
    br = BreakerRetrying(breaker=cb, max_tries=1, sleep=lambda _: None)
    clock = mock.Mock(return_value=0.0)
    with mock.patch("backon._circuit_breaker.time.monotonic", clock):
        cb.record_failure()
        clock.return_value = 100.0
        assert cb.state is CircuitBreakerState.HALF_OPEN

        with pytest.raises(ValueError):
            br.call(lambda: (_ for _ in ()).throw(ValueError("x")))
        # half_open_calls now exhausted.
        with pytest.raises(CircuitOpenError):
            br.call(lambda: "unused")


def test_breaker_retrying_async_call_success():
    br = BreakerRetrying(max_tries=1, sleep=lambda _: None)

    async def fn(a, b=0):
        return a + b

    assert asyncio.run(br.async_call(fn, 1, b=2)) == 3
    assert br.breaker.success_count == 1


def test_breaker_retrying_async_call_failure():
    br = BreakerRetrying(max_tries=1, sleep=lambda _: None)

    async def fn():
        raise ValueError("async fail")

    with pytest.raises(ValueError):
        asyncio.run(br.async_call(fn))
    assert br.breaker.failure_count == 1


def test_breaker_retrying_async_call_when_open():
    cb = CircuitBreaker(failure_threshold=1, name="asyncopen")
    cb.record_failure()
    br = BreakerRetrying(breaker=cb, sleep=lambda _: None)

    async def fn():
        return 1

    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(br.async_call(fn))
    assert str(exc_info.value) == "asyncopen"


def test_breaker_retrying_async_half_open_limit():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0,
                        half_open_max_calls=1)
    br = BreakerRetrying(breaker=cb, max_tries=1, sleep=lambda _: None)
    clock = mock.Mock(return_value=0.0)
    with mock.patch("backon._circuit_breaker.time.monotonic", clock):
        cb.record_failure()
        clock.return_value = 100.0
        assert cb.state is CircuitBreakerState.HALF_OPEN

        async def boom():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            asyncio.run(br.async_call(boom))
        with pytest.raises(CircuitOpenError):
            asyncio.run(br.async_call(boom))


def test_breaker_retrying_returns_value_with_cast():
    br = BreakerRetrying(max_tries=1, sleep=lambda _: None)
    sentinel = object()
    assert br.call(lambda: sentinel) is sentinel
