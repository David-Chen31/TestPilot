from __future__ import annotations

import logging
import time as time_module
from typing import Any
from unittest import mock

import pytest

import backon._common as _common
from backon._common import (
    disable,
    enable,
    is_enabled,
    _apply_test_overrides,
    _check_hot_loop,
    _config_handlers,
    _elapsed,
    _hot_loop_data,
    _init_wait_gen,
    _is_custom_wait,
    _LazyLogHandler,
    _log_backoff,
    _log_giveup,
    _maybe_call,
    _next_wait,
    _now,
    _prepare_logger,
)


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Ensure global enable state and test config are restored between tests."""
    original_enabled = _common._GLOBAL_ENABLED
    original_config = dict(_common._TEST_CONFIG)
    original_hot = dict(_common._hot_loop_data)
    yield
    _common._GLOBAL_ENABLED = original_enabled
    _common._TEST_CONFIG.clear()
    _common._TEST_CONFIG.update(original_config)
    _common._hot_loop_data.clear()
    _common._hot_loop_data.update(original_hot)


# --------------------------- enable / disable ---------------------------

def test_is_enabled_defaults_true():
    enable()
    assert is_enabled() is True


def test_disable_sets_disabled():
    enable()
    disable()
    assert is_enabled() is False
    assert _common._GLOBAL_ENABLED is False


def test_enable_after_disable():
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True


def test_enable_is_idempotent():
    enable()
    enable()
    assert is_enabled() is True


def test_disable_is_idempotent():
    disable()
    disable()
    assert is_enabled() is False


# --------------------------- _maybe_call ---------------------------

def test_maybe_call_callable_no_args():
    assert _maybe_call(lambda: 42) == 42


def test_maybe_call_callable_with_args():
    assert _maybe_call(lambda x, y: x + y, 3, 4) == 7


def test_maybe_call_callable_with_kwargs():
    assert _maybe_call(lambda x, y=0: x + y, 3, y=5) == 8


def test_maybe_call_non_callable_returned():
    assert _maybe_call(42) == 42
    assert _maybe_call("hello") == "hello"
    assert _maybe_call(None) is None


def test_maybe_call_type_error_returns_callable():
    # A function requiring an argument, called with none.
    def f(x):
        return x

    result = _maybe_call(f)
    assert result is f


def test_maybe_call_type_error_from_body_propagates():
    def f():
        raise TypeError("boom unrelated error")

    with pytest.raises(TypeError, match="boom unrelated error"):
        _maybe_call(f)


def test_maybe_call_type_error_mismatched_args_propagates():
    def f(x):
        return x

    with pytest.raises(TypeError):
        _maybe_call(f, 1, 2)


# --------------------------- _init_wait_gen ---------------------------

def test_init_wait_gen_basic():
    def gen(**kwargs):
        return kwargs

    result = _init_wait_gen(gen, {"a": 1, "b": 2})
    assert result == {"a": 1, "b": 2}


def test_init_wait_gen_calls_callable_values():
    def gen(**kwargs):
        return kwargs

    result = _init_wait_gen(gen, {"a": lambda: 10, "b": 2})
    assert result == {"a": 10, "b": 2}


def test_init_wait_gen_empty_kwargs():
    def gen(**kwargs):
        assert kwargs == {}
        return "ok"

    assert _init_wait_gen(gen, {}) == "ok"


# --------------------------- _next_wait ---------------------------

class _FakeWait:
    def __init__(self, values):
        self.values = list(values)
        self.sent = []

    def next(self, v):
        self.sent.append(v)
        return self.values.pop(0)


def test_next_wait_basic():
    w = _FakeWait([1.0])
    result = _next_wait(w, None, None, 0.0, None)
    assert result == 1.0
    assert w.sent == [None]


def test_next_wait_sends_value():
    w = _FakeWait([2.0])
    _next_wait(w, "val", None, 0.0, None)
    assert w.sent == ["val"]


def test_next_wait_with_jitter():
    w = _FakeWait([2.0])
    result = _next_wait(w, None, lambda v: v * 3, 0.0, None)
    assert result == 6.0


def test_next_wait_applies_multiplier():
    _common._TEST_CONFIG["backoff_multiplier"] = 2.0
    w = _FakeWait([1.5])
    result = _next_wait(w, None, None, 0.0, None)
    assert result == 3.0


def test_next_wait_with_max_time():
    w = _FakeWait([10.0])
    result = _next_wait(w, None, None, 2.0, 5.0)
    assert result == 3.0


def test_next_wait_max_time_no_clamp():
    w = _FakeWait([1.0])
    result = _next_wait(w, None, None, 2.0, 100.0)
    assert result == 1.0


# --------------------------- _apply_test_overrides ---------------------------

def test_apply_test_overrides_none():
    mt, mtm = _apply_test_overrides()
    assert mt is None
    assert mtm is None


def test_apply_test_overrides_passthrough():
    mt, mtm = _apply_test_overrides(5, 10)
    assert mt == 5
    assert mtm == 10


def test_apply_test_overrides_max_retries_override():
    _common._TEST_CONFIG["max_retries"] = 3
    mt, mtm = _apply_test_overrides(5, 10)
    assert mt == 3
    assert mtm == 10


# --------------------------- _prepare_logger ---------------------------

def test_prepare_logger_string():
    lg = _prepare_logger("my_logger")
    assert isinstance(lg, logging.Logger)
    assert lg.name == "my_logger"


def test_prepare_logger_object_passthrough():
    lg = logging.getLogger("other")
    assert _prepare_logger(lg) is lg


# --------------------------- _LazyLogHandler ---------------------------

def test_lazy_log_handler_lazy_creation():
    calls = []

    def factory(details, logger, log_level):
        calls.append((details, logger, log_level))

    h = _LazyLogHandler(factory, "lg", logging.INFO)
    assert h._handler is None
    h({"x": 1})
    assert len(calls) == 1
    h({"x": 2})
    assert len(calls) == 2
    assert h._handler is not None


def test_lazy_log_handler_creates_once():
    created = []

    def factory(details, logger, log_level):
        created.append((logger, log_level))

    h = _LazyLogHandler(factory, "lg", logging.INFO)
    h({})
    h({})
    h({})
    assert len(created) == 1


# --------------------------- _config_handlers ---------------------------

def test_config_handlers_list_no_logger_returned_as_is():
    handlers = [1, 2, 3]
    result = _config_handlers(handlers)
    assert result is handlers


def test_config_handlers_none_returns_empty():
    assert _config_handlers(None) == []


def test_config_handlers_iterable_without_logger():
    result = _config_handlers((1, 2))
    assert result == [1, 2]


def test_config_handlers_single_without_logger():
    result = _config_handlers(42)
    assert result == [42]


def test_config_handlers_logger_none_user():
    result = _config_handlers(None, default_handler="dh", logger="lg", log_level=10)
    assert len(result) == 1
    assert isinstance(result[0], _LazyLogHandler)


def test_config_handlers_logger_with_user_iterable():
    def dh(details, logger, log_level):
        pass

    result = _config_handlers(
        [1, 2], default_handler=dh, logger="lg", log_level=10
    )
    assert len(result) == 3
    assert result[1:] == [1, 2]


def test_config_handlers_logger_with_single_user():
    def dh(details, logger, log_level):
        pass

    result = _config_handlers(7, default_handler=dh, logger="lg", log_level=10)
    assert len(result) == 2
    assert result[-1] == 7


def test_config_handlers_requires_log_level():
    with pytest.raises(AssertionError):
        _config_handlers(None, default_handler="dh", logger="lg")


# --------------------------- _log_backoff ---------------------------

class _FakeLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg, args))


def test_log_backoff_with_exception():
    logger = _FakeLogger()

    def target():
        pass

    exc = ValueError("bad")
    _log_backoff(
        {"target": target, "wait": 1.23, "exception": exc}, logger, logging.WARNING
    )
    assert len(logger.records) == 1
    level, msg, args = logger.records[0]
    assert level == logging.WARNING
    assert args[0] == "target"
    assert f"{args[1]:.1f}" == "1.2"
    assert args[2] == "ValueError: bad"


def test_log_backoff_without_exception_uses_value():
    logger = _FakeLogger()

    def target():
        pass

    _log_backoff({"target": target, "wait": 2.0, "value": "retval"}, logger, 20)
    level, msg, args = logger.records[0]
    assert args[2] == "retval"


def test_log_backoff_without_exception_no_value():
    logger = _FakeLogger()

    def target():
        pass

    _log_backoff({"target": target, "wait": 2.0}, logger, 20)
    _, _, args = logger.records[0]
    assert args[2] == "unknown"


# --------------------------- _log_giveup ---------------------------

def test_log_giveup_with_exception():
    logger = _FakeLogger()

    def target():
        pass

    exc = RuntimeError("oops")
    _log_giveup(
        {"target": target, "tries": 5, "exception": exc}, logger, logging.ERROR
    )
    level, msg, args = logger.records[0]
    assert level == logging.ERROR
    assert args == ["target", 5, "RuntimeError: oops"]


def test_log_giveup_without_exception_value():
    logger = _FakeLogger()

    def target():
        pass

    _log_giveup({"target": target, "tries": 3, "value": "v"}, logger, 20)
    _, _, args = logger.records[0]
    assert args == ["target", 3, "v"]


def test_log_giveup_without_exception_no_value():
    logger = _FakeLogger()

    def target():
        pass

    _log_giveup({"target": target, "tries": 3}, logger, 20)
    _, _, args = logger.records[0]
    assert args[2] == "unknown"


# --------------------------- time helpers ---------------------------

def test_now_uses_monotonic_ns(monkeypatch):
    monkeypatch.setattr(time_module, "monotonic_ns", lambda: 2_000_000_000)
    assert _now() == 2.0


def test_elapsed(monkeypatch):
    monkeypatch.setattr(time_module, "monotonic_ns", lambda: 5_000_000_000)
    assert _elapsed(2.0) == 3.0


# --------------------------- _is_custom_wait ---------------------------

@pytest.mark.parametrize("value", [0, 1, 1.5, -2.0, 100])
def test_is_custom_wait_numbers(value):
    assert _is_custom_wait(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_is_custom_wait_bools(value):
    assert _is_custom_wait(value) is False


@pytest.mark.parametrize("value", ["1", None, [1], object()])
def test_is_custom_wait_non_numbers(value):
    assert _is_custom_wait(value) is False


# --------------------------- _check_hot_loop ---------------------------

def test_check_hot_loop_warns_after_threshold(monkeypatch, caplog):
    _common._hot_loop_data["last_retry"] = 0.0
    _common._hot_loop_data["count"] = 0
    t = {"v": 1_000_000_000}

    def fake_ns():
        return t["v"]

    monkeypatch.setattr(time_module, "monotonic_ns", fake_ns)

    with caplog.at_level(logging.WARNING, logger="backon"):
        # First call sets baseline (prev == 0).
        _check_hot_loop()
        # Subsequent quick calls within 0.1s increment count.
        for i in range(5):
            _check_hot_loop()
        # After the 6th quick call total, count reached >= 5 -> warning.
        assert any("Hot loop detected" in r.message for r in caplog.records)
        assert _common._hot_loop_data["count"] == 0


def test_check_hot_loop_resets_on_slow_call(monkeypatch):
    _common._hot_loop_data["last_retry"] = 1.0
    _common._hot_loop_data["count"] = 4
    monkeypatch.setattr(time_module, "monotonic_ns", lambda: 2_000_000_000)
    _check_hot_loop()
    # now - prev = 1.0 > 0.1 -> count reset to 0.
    assert _common._hot_loop_data["count"] == 0
    assert _common._hot_loop_data["last_retry"] == 2.0


def test_check_hot_loop_increments_count_within_window(monkeypatch):
    _common._hot_loop_data["last_retry"] = 1.0
    _common._hot_loop_data["count"] = 1
    monkeypatch.setattr(time_module, "monotonic_ns", lambda: 1_050_000_000)
    _check_hot_loop()
    assert _common._hot_loop_data["count"] == 2
    assert _common._hot_loop_data["last_retry"] == 1.05
