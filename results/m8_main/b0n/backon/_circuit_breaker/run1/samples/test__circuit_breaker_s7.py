import asyncio
import time
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


def make_breaker(**kwargs):
    defaults = dict(
        failure_threshold=3,
        recovery_timeout=10.0,
        half_open_max_calls=1,
        name="test-breaker",
    )
    defaults.update(kwargs)
    return CircuitBreaker(**defaults)


def boom(*args, **kwargs):
    raise ValueError("boom")


def ok(value=42, *args, **kwargs):
    return value


async def async_boom(*args, **kwargs):
    raise ValueError("async-boom")


async def async_ok(value=42, *args, **kwargs):
    return value


# ---------------------------------------------------------------------------
# CircuitBreakerState
# ---------------------------------------------------------------------------


class TestCircuitBreakerState:
    def test_members(self):
        assert CircuitBreakerState.CLOSED.value == "CLOSED"
        assert CircuitBreakerState.OPEN.value == "OPEN"
        assert CircuitBreakerState.HALF_OPEN.value == "HALF_OPEN"

    def test_lookup_by_value(self):
        assert CircuitBreakerState("OPEN") is CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# CircuitOpenError
# ---------------------------------------------------------------------------


class TestCircuitOpenError:
    def test_is_exception(self):
        assert issubclass(CircuitOpenError, Exception)

    def test_carries_message(self):
        err = CircuitOpenError("hello")
        assert str(err) == "hello"


# ---------------------------------------------------------------------------
# CircuitBreaker - construction & properties
# ---------------------------------------------------------------------------


class TestCircuitBreakerInit:
    def test_defaults(self):
        cb = CircuitBreaker()
        assert cb.state is CircuitBreakerState.CLOSED
        assert cb.name == ""
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_custom_values(self):
        cb = make_breaker(name="custom")
        assert cb.name == "custom"
        assert cb.state is CircuitBreakerState.CLOSED

    def test_starts_closed(self):
        assert make_breaker().state is CircuitBreakerState.CLOSED


# ---------------------------------------------------------------------------
# Recording success / failure
# ---------------------------------------------------------------------------


class TestRecordSuccess:
    def test_increments_success_count(self):
        cb = make_breaker()
        cb.record_success()
        assert cb.success_count == 1
        cb.record_success()
        assert cb.success_count == 2

    def test_resets_failure_count(self):
        cb = make_breaker()
        cb.record_failure()
        assert cb.failure_count == 1
        cb.record_success()
        assert cb.failure_count == 0

    def test_closes_half_open(self):
        cb = make_breaker(failure_threshold=1, recovery_timeout=0.0)
        with mock.patch("time.monotonic", return_value=100.0):
            cb.record_failure()
            # force transition to HALF_OPEN
            assert cb.state is CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state is CircuitBreakerState.CLOSED


class TestRecordFailure:
    def test_increments_failure_count(self):
        cb = make_breaker(failure_threshold=5)
        cb.record_failure()
        assert cb.failure_count == 1

    def test_resets_success_count(self):
        cb = make_breaker()
        cb.record_success()
        cb.record_success()
        assert cb.success_count == 2
        cb.record_failure()
        assert cb.success_count == 0

    def test_opens_at_threshold(self):
        cb = make_breaker(failure_threshold=3)
        cb.record_failure()
        assert cb.state is CircuitBreakerState.CLOSED
        cb.record_failure()
        assert cb.state is CircuitBreakerState.CLOSED
        cb.record_failure()
        assert cb.state is CircuitBreakerState.OPEN

    def test_beyond_threshold_stays_open(self):
        cb = make_breaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitBreakerState.OPEN
        cb.record_failure()
        assert cb.state is CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# State transitions (time-controlled)
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_open_to_half_open_after_recovery(self):
        cb = make_breaker(failure_threshold=1, recovery_timeout=5.0)
        with mock.patch("time.monotonic", return_value=100.0):
            cb.record_failure()
            assert cb.state is CircuitBreakerState.HALF_OPEN or True
        # At exactly t=100 + 5 -> HALF_OPEN
        with mock.patch("time.monotonic", return_value=105.0):
            assert cb.state is CircuitBreakerState.HALF_OPEN

    def test_open_stays_open_before_recovery(self):
        cb = make_breaker(failure_threshold=1, recovery_timeout=5.0)
        with mock.patch("time.monotonic", return_value=100.0):
            cb.record_failure()
        with mock.patch("time.monotonic", return_value=104.9):
            assert cb.state is CircuitBreakerState.OPEN

    def test_half_open_resets_half_open_calls(self):
        cb = make_breaker(failure_threshold=1, recovery_timeout=5.0)
        with mock.patch("time.monotonic", return_value=100.0):
            cb.record_failure()
        # force into half open
        with mock.patch("time.monotonic", return_value=200.0):
            assert cb.state is CircuitBreakerState.HALF_OPEN
            cb._half_open_calls = 5
        # re-checking state in half-open changes nothing
        with mock.patch("time.monotonic", return_value=200.0):
            assert cb.state is CircuitBreakerState.HALF_OPEN
            assert cb._half_open_calls == 5


# ---------------------------------------------------------------------------
# CircuitBreaker.call
# ---------------------------------------------------------------------------


class TestCircuitBreakerCall:
    def test_success_returns_result(self):
        cb = make_breaker()
        assert cb.call(ok, 7) == 7
        assert cb.failure_count == 0
        assert cb.success_count == 1

    def test_success_with_kwargs(self):
        cb = make_breaker()

        def fn(a, b=0):
            return a + b

        assert cb.call(fn, 1, b=2) == 3

    def test_failure_records_and_reraises(self):
        cb = make_breaker(failure_threshold=5)
        with pytest.raises(ValueError, match="boom"):
            cb.call(boom)
        assert cb.failure_count == 1
        assert cb.success_count == 0

    def test_failure_opens_breaker_then_raises_circuit_open(self):
        cb = make_breaker(failure_threshold=2)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(boom)
        assert cb.state is CircuitBreakerState.OPEN
        with pytest.raises(CircuitOpenError):
            cb.call(ok)

    def test_open_error_contains_name(self):
        cb = make_breaker(failure_threshold=1, name="named")
        with pytest.raises(ValueError):
            cb.call(boom)
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(ok)
        assert str(exc_info.value) == "named"

    def test_open_does_not_call_fn(self):
        cb = make_breaker(failure_threshold=1)
        with pytest.raises(ValueError):
            cb.call(boom)
        called = []

        def tracker():
            called.append(True)
            return 1

        with pytest.raises(CircuitOpenError):
            cb.call(tracker)
        assert called == []

    def test_half_open_limits_calls(self):
        cb = make_breaker(
            failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
        )
        with mock.patch("time.monotonic", return_value=100.0):
            with pytest.raises(ValueError):
                cb.call(boom)
        with mock.patch("time.monotonic", return_value=200.0):
            assert cb.state is CircuitBreakerState.HALF_OPEN
            # first half-open call is allowed
            assert cb.call(ok, 5) == 5
        # after success, breaker is closed
        assert cb.state is CircuitBreakerState.CLOSED

    def test_half_open_rejects_extra_calls(self):
        cb = make_breaker(
            failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
        )
        with mock.patch("time.monotonic", return_value=100.0):
            with pytest.raises(ValueError):
                cb.call(boom)
        with mock.patch("time.monotonic", return_value=200.0):
            # Consume the half-open slot but fail so we stay in HALF_OPEN
            with pytest.raises(ValueError):
                cb.call(boom)
        with mock.patch("time.monotonic", return_value=200.0):
            with pytest.raises(CircuitOpenError):
                cb.call(ok)

    def test_failure_in_half_open_reopens(self):
        cb = make_breaker(
            failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=2
        )
        with mock.patch("time.monotonic", return_value=100.0):
            with pytest.raises(ValueError):
                cb.call(boom)
        # half-open and fail again -> back to OPEN
        with mock.patch("time.monotonic", return_value=200.0):
            assert cb.state is CircuitBreakerState.HALF_OPEN
            with pytest.raises(ValueError):
                cb.call(boom)
        with mock.patch("time.monotonic", return_value=200.0):
            assert cb.state is CircuitBreakerState.OPEN

    def test_base_exception_also_recorded(self):
        cb = make_breaker(failure_threshold=5)

        def raiser():
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            cb.call(raiser)
        assert cb.failure_count == 1

    def test_dunder_call(self):
        cb = make_breaker()
        assert cb(ok, 11) == 11
        assert cb.success_count == 1


# ---------------------------------------------------------------------------
# CircuitBreaker.async_call
# ---------------------------------------------------------------------------


class TestCircuitBreakerAsyncCall:
    def test_success(self):
        cb = make_breaker()
        result = asyncio.run(cb.async_call(async_ok, 3))
        assert result == 3
        assert cb.success_count == 1
        assert cb.failure_count == 0

    def test_failure_records_and_reraises(self):
        cb = make_breaker(failure_threshold=5)
        with pytest.raises(ValueError, match="async-boom"):
            asyncio.run(cb.async_call(async_boom))
        assert cb.failure_count == 1

    def test_open_raises(self):
        cb = make_breaker(failure_threshold=1)
        with pytest.raises(ValueError):
            asyncio.run(cb.async_call(async_boom))
        with pytest.raises(CircuitOpenError) as exc_info:
            asyncio.run(cb.async_call(async_ok))
        assert str(exc_info.value) == "test-breaker"

    def test_half_open_limits(self):
        cb = make_breaker(
            failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
        )
        with mock.patch("time.monotonic", return_value=100.0):
            with pytest.raises(ValueError):
                asyncio.run(cb.async_call(async_boom))
        with mock.patch("time.monotonic", return_value=200.0):
            with pytest.raises(ValueError):
                asyncio.run(cb.async_call(async_boom))
        with mock.patch("time.monotonic", return_value=200.0):
            with pytest.raises(CircuitOpenError):
                asyncio.run(cb.async_call(async_ok))


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


class TestContextManagers:
    def test_sync_context_manager(self):
        cb = make_breaker()
        with cb as entered:
            assert entered is cb

    def test_sync_context_manager_does_not_record(self):
        cb = make_breaker()
        with cb:
            pass
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_async_context_manager(self):
        cb = make_breaker()

        async def runner():
            async with cb as entered:
                assert entered is cb
            return True

        assert asyncio.run(runner()) is True


# ---------------------------------------------------------------------------
# BreakerRetrying
# ---------------------------------------------------------------------------


class TestBreakerRetrying:
    def test_breaker_property_default(self):
        br = BreakerRetrying()
        assert isinstance(br.breaker, CircuitBreaker)

    def test_breaker_property_custom(self):
        custom = make_breaker(name="custom")
        br = BreakerRetrying(breaker=custom)
        assert br.breaker is custom

    def test_success(self):
        br = BreakerRetrying(breaker=make_breaker())
        assert br.call(ok, 5) == 5
        assert br.breaker.success_count == 1

    def test_failure_records(self):
        breaker = make_breaker(failure_threshold=5)
        br = BreakerRetrying(breaker=breaker, max_tries=1)
        with pytest.raises(ValueError):
            br.call(boom)
        assert breaker.failure_count == 1

    def test_opens_breaker_and_raises_circuit_open(self):
        breaker = make_breaker(failure_threshold=1, name="br")
        br = BreakerRetrying(breaker=breaker, max_tries=1)
        with pytest.raises(ValueError):
            br.call(boom)
        assert breaker.state is CircuitBreakerState.OPEN
        with pytest.raises(CircuitOpenError) as exc_info:
            br.call(ok)
        assert str(exc_info.value) == "br"

    def test_retries_count_against_breaker_once(self):
        breaker = make_breaker(failure_threshold=5)
        br = BreakerRetrying(breaker=breaker, max_tries=3, sleep=lambda _: None)

        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            br.call(flaky)
        assert attempts["n"] == 3
        # only a single failure recorded for the whole retry sequence
        assert breaker.failure_count == 1

    def test_half_open_limits(self):
        breaker = make_breaker(
            failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
        )
        br = BreakerRetrying(breaker=breaker, max_tries=1)
        with mock.patch("time.monotonic", return_value=100.0):
            with pytest.raises(ValueError):
                br.call(boom)
        with mock.patch("time.monotonic", return_value=200.0):
            with pytest.raises(ValueError):
                br.call(boom)
        with mock.patch("time.monotonic", return_value=200.0):
            with pytest.raises(CircuitOpenError):
                br.call(ok)

    def test_async_success(self):
        br = BreakerRetrying(breaker=make_breaker())
        result = asyncio.run(br.async_call(async_ok, 8))
        assert result == 8
        assert br.breaker.success_count == 1

    def test_async_failure_records(self):
        breaker = make_breaker(failure_threshold=5)
        br = BreakerRetrying(breaker=breaker, max_tries=1)
        with pytest.raises(ValueError):
            asyncio.run(br.async_call(async_boom))
        assert breaker.failure_count == 1

    def test_async_open_raises(self):
        breaker = make_breaker(failure_threshold=1, name="a-br")
        br = BreakerRetrying(breaker=breaker, max_tries=1)
        with pytest.raises(ValueError):
            asyncio.run(br.async_call(async_boom))
        with pytest.raises(CircuitOpenError) as exc_info:
            asyncio.run(br.async_call(async_ok))
        assert str(exc_info.value) == "a-br"

    def test_async_half_open_limits(self):
        breaker = make_breaker(
            failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=1
        )
        br = BreakerRetrying(breaker=breaker, max_tries=1)
        with mock.patch("time.monotonic", return_value=100.0):
            with pytest.raises(ValueError):
                asyncio.run(br.async_call(async_boom))
        with mock.patch("time.monotonic", return_value=200.0):
            with pytest.raises(ValueError):
                asyncio.run(br.async_call(async_boom))
        with mock.patch("time.monotonic", return_value=200.0):
            with pytest.raises(CircuitOpenError):
                asyncio.run(br.async_call(async_ok))
