"""Tests for backon._retry._api (public `retry` and internal helpers)."""

from __future__ import annotations

import asyncio
import logging

import pytest

from backon._retry import _api
from backon._retry._api import retry, _retry_sync, _retry_async


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Boom(Exception):
    pass


def _noop_sleep(_):
    return None


async def _noop_async_sleep(_):
    return None


# ---------------------------------------------------------------------------
# retry: disabled path
# ---------------------------------------------------------------------------

def test_retry_sync_disabled_calls_target_without_args(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)
    calls = []

    def target():
        calls.append(1)
        return "result"

    out = retry(target)
    assert out == "result"
    assert calls == [1]


def test_retry_async_disabled_path(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)

    async def target():
        return "async-result"

    coro = retry(target)
    assert asyncio.iscoroutine(coro)
    out = asyncio.run(coro)
    assert out == "async-result"


# ---------------------------------------------------------------------------
# retry: sync basic success
# ---------------------------------------------------------------------------

def test_retry_sync_success_first_try():
    calls = []

    def target():
        calls.append(1)
        return 42

    wrapped = retry(target, sleep=_noop_sleep)
    # retry() returns result of _retry_sync which wraps target; it should
    # actually return a callable? Check actual behavior: retry(target) returns
    # _retry_sync(target, ...) result. Since target has no args, it is invoked.
    assert calls == [1]
    # Based on implementation, retry calls _retry_sync directly which invokes
    # target with no args and returns the value.
    assert wrapped == 42


def test_retry_sync_retries_on_exception_then_succeeds():
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 3:
            raise _Boom("nope")
        return "ok"

    result = retry(
        target,
        exception=_Boom,
        max_tries=5,
        sleep=_noop_sleep,
        jitter=None,
    )
    assert result == "ok"
    assert state["n"] == 3


def test_retry_sync_gives_up_and_raises():
    state = {"n": 0}

    def target():
        state["n"] += 1
        raise _Boom("always")

    with pytest.raises(_Boom):
        retry(
            target,
            exception=_Boom,
            max_tries=2,
            sleep=_noop_sleep,
            jitter=None,
        )
    assert state["n"] == 2


def test_retry_sync_raise_on_giveup_false_returns_value():
    state = {"n": 0}

    def target():
        state["n"] += 1
        raise _Boom("always")

    result = retry(
        target,
        exception=_Boom,
        max_tries=2,
        raise_on_giveup=False,
        sleep=_noop_sleep,
        jitter=None,
    )
    # Without retry_error_callback and raise_on_giveup=False, the retry loop
    # returns the last value/exception info; typically returns None here.
    assert result is None
    assert state["n"] == 2


def test_retry_sync_retry_error_callback():
    state = {"n": 0}

    def target():
        state["n"] += 1
        raise _Boom("fail")

    def on_fail(details):
        return f"callback:{details['tries']}"

    result = retry(
        target,
        exception=_Boom,
        max_tries=3,
        retry_error_callback=on_fail,
        sleep=_noop_sleep,
        jitter=None,
    )
    assert result == "callback:3"
    assert state["n"] == 3


def test_retry_sync_predicate_condition():
    state = {"n": 0}

    def target():
        state["n"] += 1
        return state["n"] < 3  # Falsy when done

    # predicate = operator.not_ by default: retries while predicate(value) True
    # Here not_(value) is True while value is False i.e., n<3 True -> value True?
    # Careful: value is bool. not_(True)=False -> stop. not_(False)=True -> retry.
    result = retry(
        target,
        max_tries=5,
        sleep=_noop_sleep,
        jitter=None,
    )
    assert result is False
    assert state["n"] == 3


def test_retry_sync_custom_condition_and_stop():
    from backon._conditions import Stop, RetryCondition

    state = {"n": 0}

    def target():
        state["n"] += 1
        return "val"

    attempts = {"count": 0}

    class MyCondition(RetryCondition):
        def __call__(self, s):
            return True  # always retry

    class MyStop(Stop):
        def __call__(self, s):
            attempts["count"] += 1
            return attempts["count"] >= 3

    result = retry(
        target,
        condition=MyCondition(),
        stop=MyStop(),
        sleep=_noop_sleep,
        jitter=None,
        raise_on_giveup=False,
    )
    assert state["n"] == 3
    assert result == "val"


def test_retry_sync_custom_sleep_called():
    slept = []
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 2:
            raise _Boom()
        return "done"

    def my_sleep(wait):
        slept.append(wait)

    result = retry(
        target,
        exception=_Boom,
        max_tries=3,
        sleep=my_sleep,
        jitter=None,
        wait_gen_kwargs={"base": 2, "factor": 1, "max_value": 1},
    )
    assert result == "done"
    assert len(slept) == 1


def test_retry_sync_handlers_on_backoff_and_giveup():
    backoff_calls = []
    giveup_calls = []

    def target():
        raise _Boom("boom")

    def on_backoff(details):
        backoff_calls.append(dict(details))

    def on_giveup(details):
        giveup_calls.append(dict(details))

    with pytest.raises(_Boom):
        retry(
            target,
            exception=_Boom,
            max_tries=3,
            on_backoff=on_backoff,
            on_giveup=on_giveup,
            sleep=_noop_sleep,
            jitter=None,
            logger=None,
        )
    assert len(backoff_calls) == 2
    assert len(giveup_calls) == 1
    assert giveup_calls[0]["tries"] == 3


def test_retry_sync_handlers_as_lists():
    events = []

    def target():
        raise _Boom()

    def h1(d):
        events.append("h1")

    def h2(d):
        events.append("h2")

    with pytest.raises(_Boom):
        retry(
            target,
            exception=_Boom,
            max_tries=2,
            on_backoff=[h1, h2],
            on_giveup=[h1, h2],
            sleep=_noop_sleep,
            jitter=None,
            logger=None,
        )
    # 1 backoff attempt (between 2 tries) and 1 giveup, each firing both handlers
    assert events == ["h1", "h2", "h1", "h2"]


def test_retry_sync_on_success_handler():
    success_calls = []

    def target():
        return "yes"

    def on_success(details):
        success_calls.append(dict(details))

    result = retry(target, on_success=on_success, sleep=_noop_sleep)
    assert result == "yes"
    assert len(success_calls) == 1
    assert success_calls[0]["value"] == "yes"


def test_retry_sync_before_after_handlers():
    events = []

    def target():
        return "ok"

    def before(d):
        events.append("before")

    def after(d):
        events.append("after")

    retry(target, before=before, after=after, sleep=_noop_sleep)
    assert events == ["before", "after"]


def test_retry_sync_on_attempt_handler():
    attempts = []

    def target():
        raise _Boom()

    def on_attempt(d):
        attempts.append(dict(d))

    with pytest.raises(_Boom):
        retry(
            target,
            exception=_Boom,
            max_tries=2,
            on_attempt=on_attempt,
            sleep=_noop_sleep,
            jitter=None,
            logger=None,
        )
    assert len(attempts) == 2


def test_retry_sync_before_sleep_handler():
    sleep_events = []

    def target():
        raise _Boom()

    def before_sleep(d):
        sleep_events.append(dict(d))

    with pytest.raises(_Boom):
        retry(
            target,
            exception=_Boom,
            max_tries=3,
            before_sleep=before_sleep,
            sleep=_noop_sleep,
            jitter=None,
            logger=None,
        )
    assert len(sleep_events) == 2


def test_retry_sync_giveup_predicate():
    state = {"n": 0}

    def target():
        state["n"] += 1
        raise _Boom("x")

    def giveup(exc):
        return state["n"] >= 2

    with pytest.raises(_Boom):
        retry(
            target,
            exception=_Boom,
            giveup=giveup,
            max_tries=10,
            sleep=_noop_sleep,
            jitter=None,
            logger=None,
        )
    assert state["n"] == 2


def test_retry_sync_logging_default_handler():
    logger = logging.getLogger("test_retry_api_logging")
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        def target():
            raise _Boom("boom")

        with pytest.raises(_Boom):
            retry(
                target,
                exception=_Boom,
                max_tries=2,
                sleep=_noop_sleep,
                jitter=None,
                logger=logger,
                giveup_log_level=logging.ERROR,
            )
        # At least one backoff log and one giveup log
        messages = [r.getMessage() for r in records]
        assert any("Backing off" in m for m in messages)
        assert any("Giving up" in m for m in messages)
    finally:
        logger.removeHandler(handler)


def test_retry_sync_string_logger_prepared():
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("test_retry_api_string_logger")
    logger.handlers = []
    handler = ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        def target():
            raise _Boom()

        with pytest.raises(_Boom):
            retry(
                target,
                exception=_Boom,
                max_tries=2,
                sleep=_noop_sleep,
                jitter=None,
                logger="test_retry_api_string_logger",
            )
        assert any("Giving up" in r.getMessage() for r in records)
    finally:
        logger.removeHandler(handler)


def test_retry_sync_custom_condition_without_stop():
    from backon._conditions import RetryCondition

    state = {"n": 0}

    class Always(RetryCondition):
        def __call__(self, s):
            return True

    def target():
        state["n"] += 1
        raise _Boom()

    with pytest.raises(_Boom):
        retry(
            target,
            condition=Always(),
            max_tries=3,
            sleep=_noop_sleep,
            jitter=None,
            logger=None,
        )
    assert state["n"] == 3


# ---------------------------------------------------------------------------
# retry (dispatch): sync vs async
# ---------------------------------------------------------------------------

def test_retry_dispatches_to_retry_sync(monkeypatch):
    captured = {}

    def fake_sync(target, wait_gen, **kwargs):
        captured["target"] = target
        captured["wait_gen"] = wait_gen
        captured["kwargs"] = kwargs
        return "sync-result"

    monkeypatch.setattr(_api, "_retry_sync", fake_sync)

    def target():
        return 1

    result = retry(target, name="ignored", max_tries=5, sleep=_noop_sleep)
    assert result == "sync-result"
    assert captured["target"] is target
    assert captured["kwargs"]["max_tries"] == 5
    # name is not forwarded
    assert "name" not in captured["kwargs"]
    # sleep forwarded
    assert captured["kwargs"]["sleep"] is _noop_sleep


def test_retry_dispatches_to_retry_async(monkeypatch):
    captured = {}

    async def fake_async(target, wait_gen, **kwargs):
        captured["target"] = target
        captured["kwargs"] = kwargs
        return "async-result"

    monkeypatch.setattr(_api, "_retry_async", fake_async)

    async def target():
        return 1

    coro = retry(target, max_tries=7, sleep=_noop_async_sleep)
    result = asyncio.run(coro)
    assert result == "async-result"
    assert captured["kwargs"]["max_tries"] == 7


def test_retry_wait_gen_kwargs_forwarded(monkeypatch):
    captured = {}

    def fake_sync(target, wait_gen, **kwargs):
        captured["kw"] = kwargs
        return "ok"

    monkeypatch.setattr(_api, "_retry_sync", fake_sync)

    def target():
        return 1

    retry(target, base=2, factor=3)
    assert captured["kw"]["wait_gen_kwargs"] == {"base": 2, "factor": 3}


def test_retry_default_wait_gen_is_expo(monkeypatch):
    from backon._wait_gen import expo

    captured = {}

    def fake_sync(target, wait_gen, **kwargs):
        captured["wait_gen"] = wait_gen
        return "ok"

    monkeypatch.setattr(_api, "_retry_sync", fake_sync)

    def target():
        return 1

    retry(target)
    assert captured["wait_gen"] is expo


# ---------------------------------------------------------------------------
# _retry_sync direct call: wait_gen_kwargs default
# ---------------------------------------------------------------------------

def test_retry_sync_none_wait_gen_kwargs(monkeypatch):
    # When wait_gen_kwargs is None, it should be replaced with {}.
    received = {}

    def fake_inner(target, wait_gen, **kwargs):
        received.update(kwargs)
        return "x"

    monkeypatch.setattr(_api, "_retry_fast_sync_inner", fake_inner)

    def target():
        return 1

    result = _retry_sync(target, lambda *a, **k: 0, wait_gen_kwargs=None)
    assert result == "x"
    assert received["wait_gen_kwargs"] == {}


def test_retry_async_none_wait_gen_kwargs(monkeypatch):
    received = {}

    async def fake_inner(target, wait_gen, **kwargs):
        received.update(kwargs)
        return "y"

    monkeypatch.setattr(_api, "_retry_fast_async_inner", fake_inner)

    async def target():
        return 1

    coro = _retry_async(target, lambda *a, **k: 0, wait_gen_kwargs=None)
    result = asyncio.run(coro)
    assert result == "y"
    assert received["wait_gen_kwargs"] == {}


# ---------------------------------------------------------------------------
# _retry_async direct: success path
# ---------------------------------------------------------------------------

def test_retry_async_retries_and_succeeds():
    state = {"n": 0}

    async def target():
        state["n"] += 1
        if state["n"] < 3:
            raise _Boom()
        return "ok"

    async def run():
        return await _retry_async(
            target,
            _api.expo,
            exception=_Boom,
            max_tries=5,
            sleep=_noop_async_sleep,
            jitter=None,
            logger=None,
        )

    result = asyncio.run(run())
    assert result == "ok"
    assert state["n"] == 3


def test_retry_async_gives_up():
    state = {"n": 0}

    async def target():
        state["n"] += 1
        raise _Boom("boom")

    async def run():
        return await _retry_async(
            target,
            _api.expo,
            exception=_Boom,
            max_tries=2,
            sleep=_noop_async_sleep,
            jitter=None,
            logger=None,
        )

    with pytest.raises(_Boom):
        asyncio.run(run())
    assert state["n"] == 2


def test_retry_async_retry_error_callback():
    async def target():
        raise _Boom()

    def cb(details):
        return "err-cb"

    async def run():
        return await _retry_async(
            target,
            _api.expo,
            exception=_Boom,
            max_tries=2,
            sleep=_noop_async_sleep,
            jitter=None,
            logger=None,
            retry_error_callback=cb,
        )

    assert asyncio.run(run()) == "err-cb"


def test_retry_async_custom_sleep_used():
    slept = []

    state = {"n": 0}

    async def target():
        state["n"] += 1
        if state["n"] < 2:
            raise _Boom()
        return "done"

    async def my_sleep(wait):
        slept.append(wait)

    async def run():
        return await _retry_async(
            target,
            _api.expo,
            exception=_Boom,
            max_tries=3,
            sleep=my_sleep,
            jitter=None,
            logger=None,
        )

    assert asyncio.run(run()) == "done"
    assert len(slept) == 1


# ---------------------------------------------------------------------------
# retry with real sync target end-to-end
# ---------------------------------------------------------------------------

def test_retry_end_to_end_sync_returns_value():
    def target():
        return "hello"

    assert retry(target) == "hello"


def test_retry_end_to_end_async_returns_value():
    async def target():
        return "hello-async"

    coro = retry(target)
    assert asyncio.run(coro) == "hello-async"


def test_retry_end_to_end_sync_retries():
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 2:
            raise _Boom()
        return "ok"

    result = retry(
        target,
        exception=_Boom,
        max_tries=3,
        sleep=_noop_sleep,
        jitter=None,
        logger=None,
    )
    assert result == "ok"
    assert state["n"] == 2


# ---------------------------------------------------------------------------
# rate limiter forwarding
# ---------------------------------------------------------------------------

def test_retry_forwards_rate_limiter(monkeypatch):
    received = {}

    def fake_sync(target, wait_gen, **kwargs):
        received.update(kwargs)
        return "ok"

    monkeypatch.setattr(_api, "_retry_sync", fake_sync)

    from backon._rate_limiter import RateLimiter

    rl = RateLimiter(max_calls=1, period=1.0)

    def target():
        return 1

    retry(target, rate_limit=rl)
    assert received["rate_limit"] is rl


def test_retry_forwards_attempt_timeout(monkeypatch):
    received = {}

    def fake_async(target, wait_gen, **kwargs):
        received.update(kwargs)
        return "ok"

    monkeypatch.setattr(_api, "_retry_async", fake_async)

    async def target():
        return 1

    retry(target, attempt_timeout=0.5)
    assert received["attempt_timeout"] == 0.5
