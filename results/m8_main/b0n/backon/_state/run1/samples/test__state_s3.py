import threading
import time

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
# Exception types
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_try_again_is_exception(self):
        assert issubclass(TryAgain, Exception)

    def test_attempt_timeout_error_is_exception(self):
        assert issubclass(AttemptTimeoutError, Exception)

    def test_retry_error_is_exception(self):
        assert issubclass(RetryError, Exception)

    def test_raise_try_again(self):
        with pytest.raises(TryAgain):
            raise TryAgain("retry me")

    def test_raise_attempt_timeout_error(self):
        with pytest.raises(AttemptTimeoutError):
            raise AttemptTimeoutError()


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------

class TestAttempt:
    def test_defaults(self):
        a = Attempt()
        assert a.tries == 0
        assert a.exception is None
        assert a.value is None
        assert a.wait == 0.0
        assert a.elapsed == 0.0

    def test_positional_construction(self):
        exc = ValueError("bad")
        a = Attempt(3, exc, "val", 1.5, 2.5)
        assert a.tries == 3
        assert a.exception is exc
        assert a.value == "val"
        assert a.wait == 1.5
        assert a.elapsed == 2.5

    def test_keyword_construction(self):
        exc = RuntimeError("boom")
        a = Attempt(tries=2, exception=exc, value=42, wait=0.1, elapsed=0.9)
        assert a.tries == 2
        assert a.exception is exc
        assert a.value == 42
        assert a.wait == 0.1
        assert a.elapsed == 0.9

    def test_dataclass_equality(self):
        assert Attempt(tries=1, value=5) == Attempt(tries=1, value=5)
        assert Attempt(tries=1) != Attempt(tries=2)

    def test_mutability(self):
        a = Attempt()
        a.tries = 7
        a.value = "x"
        assert a.tries == 7
        assert a.value == "x"


# ---------------------------------------------------------------------------
# AttemptResult
# ---------------------------------------------------------------------------

class TestAttemptResult:
    def test_defaults(self):
        r = AttemptResult()
        assert r.value is None
        assert r.exception is None

    def test_value_result(self):
        r = AttemptResult(value=123)
        assert r.value == 123
        assert r.exception is None

    def test_exception_result(self):
        exc = ValueError("nope")
        r = AttemptResult(exception=exc)
        assert r.value is None
        assert r.exception is exc

    def test_equality(self):
        exc = ValueError("e")
        assert AttemptResult(value=1) == AttemptResult(value=1)
        assert AttemptResult(exception=exc) == AttemptResult(exception=exc)
        assert AttemptResult(value=1) != AttemptResult(value=2)


# ---------------------------------------------------------------------------
# RetryError
# ---------------------------------------------------------------------------

class TestRetryError:
    def test_message_and_last_attempt(self):
        exc = ValueError("original")
        attempt = Attempt(tries=4, exception=exc)
        err = RetryError(attempt)
        assert err.last_attempt is attempt
        assert str(err) == "Retry failed after 4 tries"

    def test_sets_cause_when_exception_present(self):
        exc = ValueError("original")
        err = RetryError(Attempt(tries=1, exception=exc))
        assert err.__cause__ is exc

    def test_no_cause_when_exception_is_none(self):
        err = RetryError(Attempt(tries=2))
        assert err.__cause__ is None
        assert str(err) == "Retry failed after 2 tries"

    def test_reraise_raises_cause(self):
        exc = ValueError("original")
        err = RetryError(Attempt(tries=3, exception=exc))
        with pytest.raises(ValueError) as info:
            err.reraise()
        assert info.value is exc

    def test_reraise_from_none_no_context(self):
        exc = ValueError("original")
        err = RetryError(Attempt(tries=3, exception=exc))
        try:
            err.reraise()
        except ValueError as raised:
            assert raised.__cause__ is None
            assert raised.__suppress_context__ is True
        else:  # pragma: no cover
            pytest.fail("reraise did not raise")

    def test_reraise_no_exception_returns_none(self):
        err = RetryError(Attempt(tries=1))
        # Should not raise anything.
        assert err.reraise() is None

    def test_specific_exception_type_preserved(self):
        exc = KeyError("missing")
        err = RetryError(Attempt(tries=2, exception=exc))
        with pytest.raises(KeyError):
            err.reraise()


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------

class TestRetryState:
    def test_defaults(self):
        st = RetryState()
        assert st.args == ()
        assert st.kwargs == {}
        assert st.tries == 0
        assert st.start_time == 0.0
        assert st.elapsed == 0.0
        assert st.idle_for == 0.0
        assert st.outcome is None
        assert callable(st.target)

    def test_default_target_is_callable_returning_none(self):
        st = RetryState()
        assert st.target() is None

    def test_construction_with_fields(self):
        def fn():
            return 1

        st = RetryState(target=fn, args=(1, 2), kwargs={"a": 3}, tries=2)
        assert st.target is fn
        assert st.args == (1, 2)
        assert st.kwargs == {"a": 3}
        assert st.tries == 2

    def test_kwargs_default_factory_independent(self):
        a = RetryState()
        b = RetryState()
        a.kwargs["x"] = 1
        assert "x" not in b.kwargs

    def test_lock_created_in_post_init(self):
        st = RetryState()
        assert isinstance(st._lock, threading.local)
        assert isinstance(st._lock.lock, type(threading.Lock()))

    def test_lock_is_distinct_per_instance(self):
        a = RetryState()
        b = RetryState()
        assert a._lock is not b._lock
        assert a._lock.lock is not b._lock.lock

    def test_statistics(self):
        st = RetryState(tries=3, start_time=10.0, idle_for=1.5, elapsed=2.5)
        assert st.statistics == {
            "start_time": 10.0,
            "attempt_number": 3,
            "idle_for": 1.5,
            "elapsed": 2.5,
        }

    def test_statistics_defaults(self):
        st = RetryState()
        assert st.statistics == {
            "start_time": 0.0,
            "attempt_number": 0,
            "idle_for": 0.0,
            "elapsed": 0.0,
        }

    def test_statistics_reflects_mutation(self):
        st = RetryState()
        st.tries = 5
        st.elapsed = 9.0
        stats = st.statistics
        assert stats["attempt_number"] == 5
        assert stats["elapsed"] == 9.0

    def test_to_details(self):
        def fn():
            return None

        st = RetryState(target=fn, args=(1,), kwargs={"k": "v"}, tries=2, elapsed=3.0)
        details = st.to_details()
        assert details == {
            "target": fn,
            "args": (1,),
            "kwargs": {"k": "v"},
            "tries": 2,
            "elapsed": 3.0,
        }

    def test_to_details_defaults(self):
        st = RetryState()
        details = st.to_details()
        assert set(details) == {"target", "args", "kwargs", "tries", "elapsed"}
        assert details["args"] == ()
        assert details["kwargs"] == {}
        assert details["tries"] == 0
        assert details["elapsed"] == 0.0


# ---------------------------------------------------------------------------
# RetryCallState
# ---------------------------------------------------------------------------

class TestRetryCallState:
    def test_defaults(self):
        st = RetryCallState()
        assert st.fn is None
        assert st.args == ()
        assert st.kwargs == {}
        assert st.attempt_number == 1
        assert st.outcome is None
        assert st.outcome_timestamp is None
        assert st.start_time == 0.0
        assert st.idle_for == 0.0
        assert st.upcoming_sleep == 0.0

    def test_kwargs_default_factory_independent(self):
        a = RetryCallState()
        b = RetryCallState()
        a.kwargs["x"] = 1
        assert "x" not in b.kwargs

    def test_elapsed_zero_when_start_time_zero(self):
        st = RetryCallState(start_time=0.0)
        assert st.elapsed == 0.0

    def test_elapsed_positive(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 105.0)
        st = RetryCallState(start_time=100.0)
        assert st.elapsed == pytest.approx(5.0)

    def test_seconds_since_start_matches_elapsed(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 200.0)
        st = RetryCallState(start_time=190.0)
        assert st.seconds_since_start == pytest.approx(10.0)

    def test_seconds_since_start_zero(self):
        st = RetryCallState()
        assert st.seconds_since_start == 0.0

    def test_statistics(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 120.0)
        st = RetryCallState(
            start_time=100.0, attempt_number=4, idle_for=1.0
        )
        stats = st.statistics
        assert stats == {
            "start_time": 100.0,
            "attempt_number": 4,
            "idle_for": 1.0,
            "elapsed": pytest.approx(20.0),
        }

    def test_statistics_zero_start(self):
        st = RetryCallState(attempt_number=2, idle_for=3.0)
        assert st.statistics == {
            "start_time": 0.0,
            "attempt_number": 2,
            "idle_for": 3.0,
            "elapsed": 0.0,
        }

    def test_to_details(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 50.0)

        def fn():
            return 1

        st = RetryCallState(
            fn=fn, args=(1, 2), kwargs={"k": 3}, attempt_number=2, start_time=40.0
        )
        details = st.to_details()
        assert details["target"] is fn
        assert details["args"] == (1, 2)
        assert details["kwargs"] == {"k": 3}
        assert details["tries"] == 2
        assert details["elapsed"] == pytest.approx(10.0)

    def test_to_details_defaults(self):
        st = RetryCallState()
        details = st.to_details()
        assert details["target"] is None
        assert details["args"] == ()
        assert details["kwargs"] == {}
        assert details["tries"] == 1
        assert details["elapsed"] == 0.0

    def test_outcome_assignment(self):
        st = RetryCallState()
        outcome = Attempt(tries=1, value=5)
        st.outcome = outcome
        st.outcome_timestamp = 1.23
        assert st.outcome is outcome
        assert st.outcome_timestamp == 1.23

    def test_elapsed_uses_actual_monotonic(self):
        st = RetryCallState(start_time=time.monotonic() - 1000.0)
        # Elapsed should be at least ~1000 seconds (no sleeping needed).
        assert st.elapsed >= 999.0
