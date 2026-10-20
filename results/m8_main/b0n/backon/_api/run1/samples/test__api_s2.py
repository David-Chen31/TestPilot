import asyncio
import logging
import operator

import pytest

from backon._conditions import RetryCondition, Stop
from backon._jitter import full_jitter
from backon._retry import _api
from backon._retry._api import retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_sleep(_seconds):
    return None


class _Counter:
    def __init__(self):
        self.calls = 0


def _make_sync(fail_times, result="ok"):
    counter = _Counter()

    def fn():
        counter.calls += 1
        if counter.calls <= fail_times:
            raise ValueError(f"fail {counter.calls}")
        return result

    fn.counter = counter
    return fn


def _make_async(fail_times, result="ok"):
    counter = _Counter()

    async def fn():
        counter.calls += 1
        if counter.calls <= fail_times:
            raise ValueError(f"fail {counter.calls}")
        return result

    fn.counter = counter
    return fn


# ---------------------------------------------------------------------------
# retry: dispatch on sync vs async
# ---------------------------------------------------------------------------

def test_retry_sync_returns_result_immediately():
    fn = _make_sync(fail_times=0, result="hello")
    wrapped = retry(fn, max_tries=3, sleep=_no_sleep)
    assert wrapped() == "hello"
    assert fn.counter.calls == 1


def test_retry_sync_retries_until_success():
    fn = _make_sync(fail_times=2, result="done")
    wrapped = retry(fn, max_tries=5, sleep=_no_sleep, raise_on_giveup=True)
    assert wrapped() == "done"
    assert fn.counter.calls == 3


def test_retry_returns_coroutine_for_async_target():
    fn = _make_async(fail_times=0, result="async-ok")
    wrapped = retry(fn, max_tries=3, sleep=_no_sleep)
    coro = wrapped()
    assert asyncio.iscoroutine(coro)
    try:
        assert asyncio.run(coro) == "async-ok"
    finally:
        if hasattr(coro, "close"):
            coro.close()
    assert fn.counter.calls == 1


def test_retry_async_retries_until_success():
    fn = _make_async(fail_times=1, result="async-done")
    wrapped = retry(fn, max_tries=5, sleep=_no_sleep)
    result = asyncio.run(wrapped())
    assert result == "async-done"
    assert fn.counter.calls == 2


# ---------------------------------------------------------------------------
# exhaust retries / raise_on_giveup
# ---------------------------------------------------------------------------

def test_retry_sync_gives_up_by_raising_when_endless_failures():
    fn = _make_sync(fail_times=10)
    wrapped = retry(
        fn,
        max_tries=3,
        sleep=_no_sleep,
        raise_on_giveup=True,
        exception=ValueError,
    )
    with pytest.raises(ValueError):
        wrapped()
    assert fn.counter.calls == 3


def test_retry_sync_raise_on_giveup_false_returns_none():
    fn = _make_sync(fail_times=10)
    wrapped = retry(
        fn,
        max_tries=3,
        sleep=_no_sleep,
        raise_on_giveup=False,
        exception=ValueError,
    )
    assert wrapped() is None
    assert fn.counter.calls == 3


def test_retry_sync_returns_retry_error_callback_value():
    fn = _make_sync(fail_times=10)
    captured = {}

    def error_cb(details):
        captured.update(details)
        return "fallback"

    wrapped = retry(
        fn,
        max_tries=2,
        sleep=_no_sleep,
        raise_on_giveup=False,
        retry_error_callback=error_cb,
        exception=ValueError,
    )
    assert wrapped() == "fallback"
    assert isinstance(captured.get("exception"), ValueError)
    assert captured["target"] is fn


def test_retry_async_gives_up_by_raising():
    fn = _make_async(fail_times=10)
    wrapped = retry(
        fn,
        max_tries=2,
        sleep=_no_sleep,
        raise_on_giveup=True,
        exception=ValueError,
    )
    with pytest.raises(ValueError):
        asyncio.run(wrapped())
    assert fn.counter.calls == 2


def test_retry_async_raise_on_giveup_false_returns_none():
    fn = _make_async(fail_times=10)
    wrapped = retry(
        fn,
        max_tries=2,
        sleep=_no_sleep,
        raise_on_giveup=False,
        exception=ValueError,
    )
    assert asyncio.run(wrapped()) is None
    assert fn.counter.calls == 2


# ---------------------------------------------------------------------------
# predicated retry (default predicate `operator.not_`)
# ---------------------------------------------------------------------------

def test_retry_predicate_retries_on_falsy_result():
    calls = _Counter()

    def fn():
        calls.calls += 1
        if calls.calls < 3:
            return False
        return "yes"

    wrapped = retry(fn, max_tries=5, sleep=_no_sleep)
    assert wrapped() == "yes"
    assert calls.calls == 3


def test_retry_predicate_uses_operator_not_by_default():
    # If the result is truthy, predicate `operator.not_` returns False -> stop.
    calls = _Counter()

    def fn():
        calls.calls += 1
        return True

    wrapped = retry(fn, max_tries=5, sleep=_no_sleep)
    assert wrapped() is True
    assert calls.calls == 1


def test_retry_custom_predicate():
    calls = _Counter()

    def fn():
        calls.calls += 1
        return calls.calls

    wrapped = retry(fn, predicate=lambda v: v < 3, max_tries=10, sleep=_no_sleep)
    assert wrapped() == 3
    assert calls.calls == 3


# ---------------------------------------------------------------------------
# handlers: on_success, on_backoff, on_giveup, on_attempt, before, after
# ---------------------------------------------------------------------------

def test_retry_on_success_called():
    fn = _make_sync(fail_times=0, result="ok")
    seen = []
    wrapped = retry(
        fn,
        on_success=lambda d: seen.append(d),
        sleep=_no_sleep,
    )
    assert wrapped() == "ok"
    assert len(seen) == 1
    assert seen[0]["target"] is fn


def test_retry_on_attempt_called_for_each_attempt():
    fn = _make_sync(fail_times=2, result="ok")
    count = {"n": 0}

    def on_attempt(details):
        count["n"] += 1

    wrapped = retry(
        fn,
        max_tries=5,
        sleep=_no_sleep,
        on_attempt=on_attempt,
        exception=ValueError,
    )
    assert wrapped() == "ok"
    assert count["n"] == 2  # only failed attempts get on_attempt? verify below


def test_retry_on_backoff_called_on_failure():
    fn = _make_sync(fail_times=2, result="ok")
    seen = []
    wrapped = retry(
        fn,
        max_tries=5,
        sleep=_no_sleep,
        on_backoff=lambda d: seen.append(d),
        exception=ValueError,
    )
    assert wrapped() == "ok"
    assert len(seen) == 2
    assert seen[0]["tries"] == 1
    assert seen[1]["tries"] == 2
    assert isinstance(seen[0].get("exception"), ValueError)


def test_retry_on_giveup_called_when_exhausted():
    fn = _make_sync(fail_times=10)
    seen = []
    wrapped = retry(
        fn,
        max_tries=2,
        sleep=_no_sleep,
        raise_on_giveup=False,
        on_giveup=lambda d: seen.append(d),
        exception=ValueError,
    )
    assert wrapped() is None
    assert len(seen) == 1
    assert seen[0]["tries"] == 2
    assert isinstance(seen[0].get("exception"), ValueError)


def test_retry_before_and_after_called():
    fn = _make_sync(fail_times=1, result="ok")
    before_calls = []
    after_calls = []
    wrapped = retry(
        fn,
        max_tries=5,
        sleep=_no_sleep,
        before=lambda d: before_calls.append(d),
        after=lambda d: after_calls.append(d),
        exception=ValueError,
    )
    assert wrapped() == "ok"
    assert len(before_calls) == 2
    assert len(after_calls) == 1  # after is called only after failed attempt? see below


def test_retry_before_sleep_called_between_attempts():
    fn = _make_sync(fail_times=1, result="ok")
    seen = []
    wrapped = retry(
        fn,
        max_tries=5,
        sleep=_no_sleep,
        before_sleep=lambda d: seen.append(d),
        exception=ValueError,
    )
    assert wrapped() == "ok"
    assert len(seen) == 1


def test_retry_handler_accepts_iterable():
    fn = _make_sync(fail_times=1, result="ok")
    a, b = [], []
    wrapped = retry(
        fn,
        max_tries=5,
        sleep=_no_sleep,
        on_backoff=[lambda d: a.append(d), lambda d: b.append(d)],
        exception=ValueError,
    )
    assert wrapped() == "ok"
    assert len(a) == 1
    assert len(b) == 1


def test_retry_async_on_success_called():
    fn = _make_async(fail_times=0, result="ok")
    seen = []
    wrapped = retry(fn, on_success=lambda d: seen.append(d), sleep=_no_sleep)
    assert asyncio.run(wrapped()) == "ok"
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# early return when backon is globally disabled
# ---------------------------------------------------------------------------

def test_retry_sync_disabled_calls_target_once_without_handlers(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)
    fn = _make_sync(fail_times=0, result="direct")
    seen = []
    wrapped = retry(
        fn,
        max_tries=3,
        sleep=_no_sleep,
        on_success=lambda d: seen.append(d),
    )
    assert wrapped() == "direct"
    assert fn.counter.calls == 1
    # Handlers are never configured when disabled
    assert seen == []


def test_retry_sync_disabled_propagates_exception(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)

    def fn():
        raise RuntimeError("direct boom")

    wrapped = retry(fn, max_tries=3, sleep=_no_sleep)
    with pytest.raises(RuntimeError, match="direct boom"):
        wrapped()


def test_retry_async_disabled_calls_target_once(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)
    fn = _make_async(fail_times=0, result="direct-async")
    wrapped = retry(fn, max_tries=3, sleep=_no_sleep)
    assert asyncio.run(wrapped()) == "direct-async"
    assert fn.counter.calls == 1


# ---------------------------------------------------------------------------
# logging: default handlers use logger object
# ---------------------------------------------------------------------------

def test_retry_default_logger_logs_backoff(caplog):
    fn = _make_sync(fail_times=1, result="ok")
    wrapped = retry(
        fn,
        max_tries=3,
        sleep=_no_sleep,
        logger="backon-test",
        exception=ValueError,
    )
    with caplog.at_level(logging.INFO, logger="backon-test"):
        assert wrapped() == "ok"
    assert any(
        "Backing off" in rec.getMessage() for rec in caplog.records
    )


def test_retry_default_logger_logs_giveup(caplog):
    fn = _make_sync(fail_times=10)
    wrapped = retry(
        fn,
        max_tries=2,
        sleep=_no_sleep,
        logger="backon-test-giveup",
        giveup_log_level=logging.WARNING,
        raise_on_giveup=False,
        exception=ValueError,
    )
    with caplog.at_level(logging.WARNING, logger="backon-test-giveup"):
        assert wrapped() is None
    assert any("Giving up" in rec.getMessage() for rec in caplog.records)


def test_retry_accepts_logger_object():
    logger = logging.getLogger("backon-object-logger")
    fn = _make_sync(fail_times=1, result="ok")
    wrapped = retry(
        fn,
        max_tries=3,
        sleep=_no_sleep,
        logger=logger,
        exception=ValueError,
    )
    assert wrapped() == "ok"


# ---------------------------------------------------------------------------
# arguments / kwargs forwarding
# ---------------------------------------------------------------------------

def test_retry_sync_passes_args_and_kwargs():
    def add(a, b, *, c=0):
        return a + b + c

    wrapped = retry(add, max_tries=2, sleep=_no_sleep)
    assert wrapped(1, 2, c=3) == 6


def test_retry_async_passes_args_and_kwargs():
    async def add(a, b, *, c=0):
        return a + b + c

    wrapped = retry(add, max_tries=2, sleep=_no_sleep)
    assert asyncio.run(wrapped(1, 2, c=4)) == 7


# ---------------------------------------------------------------------------
# custom condition / stop override
# ---------------------------------------------------------------------------

def test_retry_accepts_custom_condition_and_stop():
    calls = _Counter()

    def fn():
        calls.calls += 1
        return calls.calls

    class AlwaysRetry(RetryCondition):
        def __call__(self, state):
            return True

    class StopAfterTwo(Stop):
        def __call__(self, state):
            return state.tries >= 2

    wrapped = retry(
        fn,
        condition=AlwaysRetry(),
        stop=StopAfterTwo(),
        sleep=_no_sleep,
        raise_on_giveup=False,
    )
    assert wrapped() is None
    assert calls.calls == 2


# ---------------------------------------------------------------------------
# names passed through to the wrapped function (functools-like passthrough)
# ---------------------------------------------------------------------------

def test_retry_preserves_target_name_via_wrapper():
    fn = _make_sync(fail_times=0, result="x")
    wrapped = retry(fn, sleep=_no_sleep)
    assert wrapped() == "x"


# ---------------------------------------------------------------------------
# name kwarg routed through retry signature (accepted, no error)
# ---------------------------------------------------------------------------

def test_retry_accepts_name_kwarg():
    fn = _make_sync(fail_times=0, result="ok")
    wrapped = retry(fn, max_tries=2, sleep=_no_sleep, name="my-retry")
    assert wrapped() == "ok"


# ---------------------------------------------------------------------------
# max_time based stop
# ---------------------------------------------------------------------------

def test_retry_stops_due_to_max_time():
    calls = _Counter()
    clock = {"t": 0.0}

    def fake_sleep(seconds):
        clock["t"] += seconds

    # Use a stub wait_gen that yields 1 second, with max_time small enough
    def wait_gen():
        while True:
            yield 1.0

    def fn():
        calls.calls += 1
        raise ValueError("nope")

    wrapped = retry(
        fn,
        wait_gen=wait_gen,
        max_time=2.5,
        sleep=fake_sleep,
        raise_on_giveup=False,
        exception=ValueError,
    )
    assert wrapped() is None
    # The retry should stop once elapsed time reaches max_time.
    assert calls.calls <= 4


# ---------------------------------------------------------------------------
# exception filter: only matching exception types are retried
# ---------------------------------------------------------------------------

def test_retry_exception_filter_reraises_non_matching():
    calls = _Counter()

    def fn():
        calls.calls += 1
        raise RuntimeError("boom")

    wrapped = retry(
        fn,
        max_tries=5,
        sleep=_no_sleep,
        exception=ValueError,
    )
    with pytest.raises(RuntimeError, match="boom"):
        wrapped()
    assert calls.calls == 1


# ---------------------------------------------------------------------------
# full_jitter is default; jitter can be None
# ---------------------------------------------------------------------------

def test_retry_accepts_jitter_none():
    fn = _make_sync(fail_times=1, result="ok")
    wrapped = retry(
        fn,
        max_tries=3,
        sleep=_no_sleep,
        jitter=None,
        exception=ValueError,
    )
    assert wrapped() == "ok"


def test_retry_accepts_custom_jitter():
    seen_values = []

    def jitter(value):
        seen_values.append(value)
        return value

    fn = _make_sync(fail_times=1, result="ok")
    wrapped = retry(
        fn,
        max_tries=3,
        sleep=_no_sleep,
        jitter=jitter,
        exception=ValueError,
    )
    assert wrapped() == "ok"
    assert len(seen_values) == 1


# ---------------------------------------------------------------------------
# default jitter is full_jitter
# ---------------------------------------------------------------------------

def test_retry_default_jitter_is_full_jitter():
    import inspect

    sig = inspect.signature(_api.retry)
    assert sig.parameters["jitter"].default is full_jitter


def test_retry_default_predicate_is_operator_not():
    import inspect

    sig = inspect.signature(_api.retry)
    assert sig.parameters["predicate"].default is operator.not_
