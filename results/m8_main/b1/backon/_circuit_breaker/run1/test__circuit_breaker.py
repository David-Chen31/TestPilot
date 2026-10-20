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

def _open_breaker(breaker: CircuitBreaker) -> None:
    """Drive a breaker into the OPEN state without relying on real time."""
    for _ in range(breaker._failure_threshold):
        breaker.record_failure()


def _move_to_half_open(breaker: CircuitBreaker) -> None:
    """Force a breaker from OPEN to HALF_OPEN by rewinding last failure time."""
    _open_breaker(breaker)
    assert breaker.state == CircuitBreakerState.OPEN
    breaker._last_failure_time = time.monotonic() - breaker._recovery_timeout - 1
    assert breaker.state == CircuitBreakerState.HALF_OPEN


def _always_fail(*args, **kwargs):
    raise ValueError("boom")


def _return_value(value):
    def _fn(*args, **kwargs):
        return value

    return _fn


async def _async_return(value):
    return value


async def _async_fail():
    raise ValueError("async boom")


# ---------------------------------------------------------------------------
# CircuitBreakerState / CircuitOpenError
# ---------------------------------------------------------------------------

def test_state_enum_values():
    assert CircuitBreakerState.CLOSED.value == "CLOSED"
    assert CircuitBreakerState.OPEN.value == "OPEN"
    assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"
    assert len(CircuitBreakerState) == 3


def test_circuit_open_error_is_exception():
    err = CircuitOpenError("my-breaker")
    assert isinstance(err, Exception)
    assert str(err) == "my-breaker"


# ---------------------------------------------------------------------------
# CircuitBreaker basics
# ---------------------------------------------------------------------------

def test_breaker_defaults():
    breaker = CircuitBreaker()
    assert breaker.name == ""
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.success_count == 0


def test_breaker_custom_name():
    breaker = CircuitBreaker(name="svc")
    assert breaker.name == "svc"


def test_breaker_starts_closed():
    breaker = CircuitBreaker(failure_threshold=3)
    assert breaker.state == CircuitBreakerState.CLOSED


def test_record_success_resets_failure_count_and_increments_success():
    breaker = CircuitBreaker(failure_threshold=5)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.failure_count == 2
    assert breaker.success_count == 0

    breaker.record_success()
    assert breaker.failure_count == 0
    assert breaker.success_count == 1

    breaker.record_success()
    assert breaker.success_count == 2


def test_record_failure_increments_then_opens_at_threshold():
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.failure_count == 1
    breaker.record_failure()
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.failure_count == 2
    breaker.record_failure()
    assert breaker.state == CircuitBreakerState.OPEN
    assert breaker.failure_count == 3


def test_record_failure_resets_consecutive_successes():
    breaker = CircuitBreaker()
    breaker.record_success()
    breaker.record_success()
    assert breaker.success_count == 2
    breaker.record_failure()
    assert breaker.success_count == 0


def test_record_success_in_open_state_does_not_close():
    # record_success only closes from HALF_OPEN; an OPEN stays OPEN
    # (but failure_count resets and success_count increments).
    breaker = CircuitBreaker(failure_threshold=2)
    _open_breaker(breaker)
    assert breaker.state == CircuitBreakerState.OPEN
    breaker.record_success()
    assert breaker.state == CircuitBreakerState.OPEN
    assert breaker.failure_count == 0
    assert breaker.success_count == 1


def test_open_transitions_to_half_open_after_timeout():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    _open_breaker(breaker)
    assert breaker.state == CircuitBreakerState.OPEN

    # Not enough time elapsed
    breaker._last_failure_time = time.monotonic()
    assert breaker.state == CircuitBreakerState.OPEN

    # Timeout elapsed
    breaker._last_failure_time = time.monotonic() - 11.0
    assert breaker.state == CircuitBreakerState.HALF_OPEN


def test_half_open_transition_resets_half_open_calls():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    _open_breaker(breaker)
    breaker._half_open_calls = 7
    breaker._last_failure_time = time.monotonic() - 6.0
    assert breaker.state == CircuitBreakerState.HALF_OPEN
    assert breaker._half_open_calls == 0


def test_half_open_record_success_closes():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    _move_to_half_open(breaker)
    breaker.record_success()
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.success_count == 1


# ---------------------------------------------------------------------------
# CircuitBreaker.call
# ---------------------------------------------------------------------------

def test_call_returns_result_and_records_success():
    breaker = CircuitBreaker(failure_threshold=3)
    result = breaker.call(lambda a, b: a + b, 1, 2)
    assert result == 3
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.success_count == 1
    assert breaker.failure_count == 0


def test_call_passes_kwargs():
    breaker = CircuitBreaker()
    result = breaker.call(lambda *, x: x * 2, x=5)
    assert result == 10


def test_call_records_failure_and_reraises():
    breaker = CircuitBreaker(failure_threshold=3)
    with pytest.raises(ValueError, match="boom"):
        breaker.call(_always_fail)
    assert breaker.failure_count == 1
    assert breaker.success_count == 0
    assert breaker.state == CircuitBreakerState.CLOSED


def test_call_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=2)
    for _ in range(2):
        with pytest.raises(ValueError):
            breaker.call(_always_fail)
    assert breaker.state == CircuitBreakerState.OPEN


def test_call_raises_circuit_open_error_with_name():
    breaker = CircuitBreaker(failure_threshold=1, name="mybreaker")
    _open_breaker(breaker)
    with pytest.raises(CircuitOpenError) as exc_info:
        breaker.call(_return_value(1))
    assert str(exc_info.value) == "mybreaker"
    # The underlying function was never called, so no success recorded.
    assert breaker.success_count == 0


def test_call_does_not_execute_when_open():
    breaker = CircuitBreaker(failure_threshold=1)
    _open_breaker(breaker)
    calls = []

    def fn():
        calls.append(1)
        return "x"

    with pytest.raises(CircuitOpenError):
        breaker.call(fn)
    assert calls == []


def test_call_recovers_after_timeout_and_success():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    _move_to_half_open(breaker)
    result = breaker.call(_return_value(42))
    assert result == 42
    assert breaker.state == CircuitBreakerState.CLOSED


def test_call_in_half_open_failure_reopens():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    _move_to_half_open(breaker)
    with pytest.raises(ValueError):
        breaker.call(_always_fail)
    assert breaker.state == CircuitBreakerState.OPEN
    assert breaker.failure_count == 2


def test_call_half_open_limit_exceeded():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0,
                             half_open_max_calls=1)
    # Manually set half-open state without consuming the single call.
    breaker._state = CircuitBreakerState.HALF_OPEN
    breaker._half_open_calls = breaker._half_open_max_calls

    with pytest.raises(CircuitOpenError):
        breaker.call(_return_value(1))


def test_call_half_open_within_limit_increments_count():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0,
                             half_open_max_calls=2)
    breaker._state = CircuitBreakerState.HALF_OPEN
    assert breaker._half_open_calls == 0

    result = breaker.call(_return_value("ok"))
    assert result == "ok"
    assert breaker._half_open_calls == 1


def test_call_bases_exception_not_swallowed():
    breaker = CircuitBreaker(failure_threshold=5)

    def raise_keyboard():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        breaker.call(raise_keyboard)
    assert breaker.failure_count == 1


def test_call_via_dunder_call():
    breaker = CircuitBreaker()
    assert breaker(lambda x: x + 1, 4) == 5


# ---------------------------------------------------------------------------
# CircuitBreaker.async_call
# ---------------------------------------------------------------------------

def test_async_call_returns_result():
    breaker = CircuitBreaker()

    async def _fn(a, b):
        return a * b

    result = asyncio.run(breaker.async_call(_fn, 3, 4))
    assert result == 12
    assert breaker.success_count == 1
    assert breaker.state == CircuitBreakerState.CLOSED


def test_async_call_records_failure():
    breaker = CircuitBreaker(failure_threshold=3)
    with pytest.raises(ValueError, match="async boom"):
        asyncio.run(breaker.async_call(_async_fail))
    assert breaker.failure_count == 1


def test_async_call_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=1)

    async def _fail():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        asyncio.run(breaker.async_call(_fail))
    assert breaker.state == CircuitBreakerState.OPEN


def test_async_call_raises_when_open():
    breaker = CircuitBreaker(failure_threshold=1, name="abreaker")
    _open_breaker(breaker)
    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(breaker.async_call(_async_return, 1))
    assert str(exc_info.value) == "abreaker"


def test_async_call_half_open_limit():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0,
                             half_open_max_calls=1)
    breaker._state = CircuitBreakerState.HALF_OPEN
    breaker._half_open_calls = 1
    with pytest.raises(CircuitOpenError):
        asyncio.run(breaker.async_call(_async_return, 1))


def test_async_call_half_open_recovery():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
    _move_to_half_open(breaker)
    result = asyncio.run(breaker.async_call(_async_return, "recovered"))
    assert result == "recovered"
    assert breaker.state == CircuitBreakerState.CLOSED


# ---------------------------------------------------------------------------
# Context manager support
# ---------------------------------------------------------------------------

def test_context_manager_returns_self():
    breaker = CircuitBreaker()
    with breaker as b:
        assert b is breaker


def test_context_manager_does_not_record():
    breaker = CircuitBreaker()
    with pytest.raises(ValueError):
        with breaker:
            raise ValueError("inner")
    # __exit__ returns None which is falsy -> exception propagates,
    # and no bookkeeping occurs.
    assert breaker.failure_count == 0
    assert breaker.success_count == 0


def test_async_context_manager_returns_self():
    breaker = CircuitBreaker()

    async def _run():
        async with breaker as b:
            return b

    assert asyncio.run(_run()) is breaker


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------

def test_breaker_retrying_default_breaker():
    br = BreakerRetrying()
    assert isinstance(br.breaker, CircuitBreaker)
    assert br.breaker.state == CircuitBreakerState.CLOSED


def test_breaker_retrying_uses_given_breaker():
    breaker = CircuitBreaker(name="given")
    br = BreakerRetrying(breaker=breaker)
    assert br.breaker is breaker


def test_breaker_retrying_call_success():
    breaker = CircuitBreaker(failure_threshold=3)
    br = BreakerRetrying(breaker=breaker, wait_gen=lambda *a, **k: (i for i in [0]),
                         max_tries=1, jitter=None, sleep=lambda _: None)
    result = br.call(lambda x: x + 1, 1)
    assert result == 2
    assert breaker.success_count == 1
    assert breaker.failure_count == 0


def test_breaker_retrying_call_failure_records():
    breaker = CircuitBreaker(failure_threshold=2)
    br = BreakerRetrying(breaker=breaker, max_tries=1, jitter=None,
                         wait_gen=lambda *a, **k: (i for i in [0]),
                         sleep=lambda _: None)
    with pytest.raises(ValueError, match="boom"):
        br.call(_always_fail)
    assert breaker.failure_count >= 1


def test_breaker_retrying_call_opens_circuit():
    breaker = CircuitBreaker(failure_threshold=1)
    br = BreakerRetrying(breaker=breaker, max_tries=1, jitter=None,
                         wait_gen=lambda *a, **k: (i for i in [0]),
                         sleep=lambda _: None)
    with pytest.raises(ValueError):
        br.call(_always_fail)
    assert breaker.state == CircuitBreakerState.OPEN

    # Second call should short-circuit.
    with pytest.raises(CircuitOpenError) as exc_info:
        br.call(lambda: "never")
    assert str(exc_info.value) == breaker.name


def test_breaker_retrying_half_open_limit():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0,
                             half_open_max_calls=1)
    breaker._state = CircuitBreakerState.HALF_OPEN
    breaker._half_open_calls = 1
    br = BreakerRetrying(breaker=breaker, max_tries=1, jitter=None,
                         wait_gen=lambda *a, **k: (i for i in [0]),
                         sleep=lambda _: None)
    with pytest.raises(CircuitOpenError):
        br.call(lambda: "never")


def test_breaker_retrying_async_call_success():
    breaker = CircuitBreaker()

    async def _fn(x):
        return x * 2

    br = BreakerRetrying(breaker=breaker, max_tries=1, jitter=None,
                         wait_gen=lambda *a, **k: (i for i in [0]),
                         sleep=lambda _: None)
    result = asyncio.run(br.async_call(_fn, 5))
    assert result == 10
    assert breaker.success_count == 1


def test_breaker_retrying_async_call_failure():
    breaker = CircuitBreaker(failure_threshold=1)
    br = BreakerRetrying(breaker=breaker, max_tries=1, jitter=None,
                         wait_gen=lambda *a, **k: (i for i in [0]),
                         sleep=lambda _: None)

    async def _fail():
        raise ValueError("async boom")

    with pytest.raises(ValueError, match="async boom"):
        asyncio.run(br.async_call(_fail))
    assert breaker.state == CircuitBreakerState.OPEN


def test_breaker_retrying_async_call_when_open():
    breaker = CircuitBreaker(failure_threshold=1, name="abr")
    _open_breaker(breaker)
    br = BreakerRetrying(breaker=breaker, max_tries=1, jitter=None,
                         wait_gen=lambda *a, **k: (i for i in [0]),
                         sleep=lambda _: None)
    with pytest.raises(CircuitOpenError) as exc_info:
        asyncio.run(br.async_call(_async_return, 1))
    assert str(exc_info.value) == "abr"


def test_breaker_retrying_async_half_open_limit():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0,
                             half_open_max_calls=1)
    breaker._state = CircuitBreakerState.HALF_OPEN
    breaker._half_open_calls = 1
    br = BreakerRetrying(breaker=breaker, max_tries=1, jitter=None,
                         wait_gen=lambda *a, **k: (i for i in [0]),
                         sleep=lambda _: None)
    with pytest.raises(CircuitOpenError):
        asyncio.run(br.async_call(_async_return, 1))


# ---------------------------------------------------------------------------
# Threading / lock sanity (light)
# ---------------------------------------------------------------------------

def test_concurrent_calls_track_failures():
    breaker = CircuitBreaker(failure_threshold=1000)

    def _fail():
        raise ValueError("x")

    def _worker():
        for _ in range(10):
            with pytest.raises(ValueError):
                breaker.call(_fail)

    threads = [threading.Thread(target=_worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert breaker.failure_count == 50
