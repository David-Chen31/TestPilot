from __future__ import annotations

import threading

import pytest

from backon._state import (
    Attempt,
    AttemptResult,
    AttemptTimeoutError,
    RetryCallState,
    RetryError,
    RetryState,
    TryAgain,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def test_try_again_is_exception():
    assert issubclass(TryAgain, Exception)
    with pytest.raises(TryAgain):
        raise TryAgain()


def test_attempt_timeout_error_is_exception():
    assert issubclass(AttemptTimeoutError, Exception)
    with pytest.raises(AttemptTimeoutError):
        raise AttemptTimeoutError()


def test_retry_error_with_cause():
    exc = ValueError("boom")
    attempt = Attempt(tries=3, exception=exc)
    err = RetryError(attempt)

    assert err.last_attempt is attempt
    assert str(err) == "Retry failed after 3 tries"
    assert err.__cause__ is exc


def test_retry_error_without_cause():
    attempt = Attempt(tries=2, exception=None)
    err = RetryError(attempt)

    assert err.last_attempt is attempt
    assert str(err) == "Retry failed after 2 tries"
    assert err.__cause__ is None


def test_retry_error_reraise_with_cause():
    exc = RuntimeError("original failure")
    attempt = Attempt(tries=1, exception=exc)
    err = RetryError(attempt)

    with pytest.raises(RuntimeError) as info:
        err.reraise()

    assert info.value is exc
    # raised from None -> __cause__ is None on the raised exception context
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


def test_retry_error_reraise_without_cause_returns_none():
    attempt = Attempt(tries=4, exception=None)
    err = RetryError(attempt)

    # No exception raised; returns None implicitly
    assert err.reraise() is None


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------


def test_attempt_defaults():
    attempt = Attempt()
    assert attempt.tries == 0
    assert attempt.exception is None
    assert attempt.value is None
    assert attempt.wait == 0.0
    assert attempt.elapsed == 0.0


def test_attempt_custom_values():
    exc = KeyError("k")
    attempt = Attempt(tries=5, exception=exc, value="val", wait=1.5, elapsed=2.25)
    assert attempt.tries == 5
    assert attempt.exception is exc
    assert attempt.value == "val"
    assert attempt.wait == 1.5
    assert attempt.elapsed == 2.25


# ---------------------------------------------------------------------------
# AttemptResult
# ---------------------------------------------------------------------------


def test_attempt_result_defaults():
    result = AttemptResult()
    assert result.value is None
    assert result.exception is None


def test_attempt_result_custom():
    exc = ValueError("bad")
    result = AttemptResult(value=42, exception=exc)
    assert result.value == 42
    assert result.exception is exc


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------


def test_retry_state_defaults():
    state = RetryState()
    assert state.args == ()
    assert state.kwargs == {}
    assert state.tries == 0
    assert state.start_time == 0.0
    assert state.elapsed == 0.0
    assert state.idle_for == 0.0
    assert state.outcome is None
    assert callable(state.target)
    assert state.target() is None


def test_retry_state_custom_values():
    def target(a, b=0):
        return a + b

    state = RetryState(
        target=target,
        args=(1,),
        kwargs={"b": 2},
        tries=7,
        start_time=100.0,
        elapsed=3.5,
        idle_for=0.25,
        outcome=Attempt(tries=7),
    )

    assert state.target is target
    assert state.args == (1,)
    assert state.kwargs == {"b": 2}
    assert state.tries == 7
    assert state.start_time == 100.0
    assert state.elapsed == 3.5
    assert state.idle_for == 0.25
    assert isinstance(state.outcome, Attempt)


def test_retry_state_kwargs_default_factory_independent():
    s1 = RetryState()
    s2 = RetryState()
    s1.kwargs["x"] = 1
    assert s2.kwargs == {}


def test_retry_state_post_init_creates_lock():
    state = RetryState()
    assert hasattr(state, "_lock")
    lock = state._lock.lock
    assert isinstance(lock, type(threading.Lock()))

    # The lock is usable
    acquired = lock.acquire(blocking=False)
    assert acquired is True
    lock.release()


def test_retry_state_lock_is_thread_local():
    state = RetryState()
    main_lock = state._lock.lock

    captured = {}

    def worker():
        # In another thread, a new threading.Lock should be created on access
        captured["lock"] = state._lock.lock

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert captured["lock"] is not main_lock
    # Each thread got its own lock object
    assert isinstance(captured["lock"], type(threading.Lock()))


def test_retry_state_statistics():
    state = RetryState(start_time=10.0, tries=3, idle_for=1.5, elapsed=4.0)
    stats = state.statistics
    assert stats == {
        "start_time": 10.0,
        "attempt_number": 3,
        "idle_for": 1.5,
        "elapsed": 4.0,
    }


def test_retry_state_statistics_defaults():
    state = RetryState()
    assert state.statistics == {
        "start_time": 0.0,
        "attempt_number": 0,
        "idle_for": 0.0,
        "elapsed": 0.0,
    }


def test_retry_state_to_details():
    def target():
        pass

    state = RetryState(
        target=target,
        args=(1, 2),
        kwargs={"a": 3},
        tries=2,
        elapsed=5.5,
    )
    details = state.to_details()
    assert details == {
        "target": target,
        "args": (1, 2),
        "kwargs": {"a": 3},
        "tries": 2,
        "elapsed": 5.5,
    }


# ---------------------------------------------------------------------------
# RetryCallState
# ---------------------------------------------------------------------------


def test_retry_call_state_defaults():
    state = RetryCallState()
    assert state.fn is None
    assert state.args == ()
    assert state.kwargs == {}
    assert state.attempt_number == 1
    assert state.outcome is None
    assert state.outcome_timestamp is None
    assert state.start_time == 0.0
    assert state.idle_for == 0.0
    assert state.upcoming_sleep == 0.0


def test_retry_call_state_elapsed_zero_start():
    state = RetryCallState(start_time=0)
    assert state.elapsed == 0.0
    assert state.seconds_since_start == 0.0


def test_retry_call_state_elapsed_nonzero_start():
    state = RetryCallState(start_time=1.0)
    assert state.elapsed > 0.0
    assert state.seconds_since_start == state.elapsed


def test_retry_call_state_custom_values():
    def fn(a):
        return a

    outcome = Attempt(tries=2, value=5)
    state = RetryCallState(
        fn=fn,
        args=(1,),
        kwargs={"k": "v"},
        attempt_number=4,
        outcome=outcome,
        outcome_timestamp=123.0,
        start_time=0.0,
        idle_for=0.5,
        upcoming_sleep=1.0,
    )
    assert state.fn is fn
    assert state.args == (1,)
    assert state.kwargs == {"k": "v"}
    assert state.attempt_number == 4
    assert state.outcome is outcome
    assert state.outcome_timestamp == 123.0
    assert state.idle_for == 0.5
    assert state.upcoming_sleep == 1.0


def test_retry_call_state_kwargs_default_factory_independent():
    s1 = RetryCallState()
    s2 = RetryCallState()
    s1.kwargs["y"] = 2
    assert s2.kwargs == {}


def test_retry_call_state_statistics():
    state = RetryCallState(
        start_time=0.0,
        attempt_number=3,
        idle_for=2.0,
    )
    stats = state.statistics
    assert stats == {
        "start_time": 0.0,
        "attempt_number": 3,
        "idle_for": 2.0,
        "elapsed": 0.0,
    }


def test_retry_call_state_statistics_nonzero_start():
    state = RetryCallState(start_time=5.0, attempt_number=2)
    stats = state.statistics
    assert stats["start_time"] == 5.0
    assert stats["attempt_number"] == 2
    assert stats["idle_for"] == 0.0
    assert stats["elapsed"] == state.elapsed
    assert stats["elapsed"] > 0.0


def test_retry_call_state_to_details():
    def fn(a):
        return a

    state = RetryCallState(
        fn=fn,
        args=(10,),
        kwargs={"b": 20},
        attempt_number=6,
    )
    details = state.to_details()
    assert details["target"] is fn
    assert details["args"] == (10,)
    assert details["kwargs"] == {"b": 20}
    assert details["tries"] == 6
    assert details["elapsed"] == state.elapsed
