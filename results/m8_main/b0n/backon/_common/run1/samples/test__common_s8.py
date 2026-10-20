from __future__ import annotations

import logging
import time

import pytest

import backon._common as common
from backon._common import (
    disable,
    enable,
    is_enabled,
    _maybe_call,
    _init_wait_gen,
    _next_wait,
    _apply_test_overrides,
    _prepare_logger,
    _LazyLogHandler,
    _config_handlers,
    _log_backoff,
    _log_giveup,
    _now,
    _elapsed,
    _is_custom_wait,
    _check_hot_loop,
)


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Keep the module's mutable global state isolated between tests."""
    original_enabled = common._GLOBAL_ENABLED
    original_test_config = dict(common._TEST_CONFIG)
    original_hot_loop = dict(common._hot_loop_data)
    yield
    common._GLOBAL_ENABLED = original_enabled
    common._TEST_CONFIG.clear()
    common._TEST_CONFIG.update(original_test_config)
    common._hot_loop_data.clear()
    common._hot_loop_data.update(original_hot_loop)


# ---------------------------------------------------------------------------
# Public API: enable / disable / is_enabled
# ---------------------------------------------------------------------------


def test_is_enabled_default_true():
    enable()
    assert is_enabled() is True


def test_disable():
    enable()
    disable()
    assert is_enabled() is False


def test_enable_after_disable():
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True


def test_disable_is_idempotent():
    disable()
    disable()
    assert is_enabled() is False


def test_enable_is_idempotent():
    enable()
    enable()
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# _maybe_call
# ---------------------------------------------------------------------------


def test_maybe_call_non_callable_returns_value():
    assert _maybe_call(5) == 5
    assert _maybe_call("abc") == "abc"
    assert _maybe_call(None) is None


def test_maybe_call_callable_no_args():
    assert _maybe_call(lambda: 42) == 42


def test_maybe_call_callable_with_args():
    assert _maybe_call(lambda a, b: a + b, 1, 2) == 3


def test_maybe_call_callable_with_kwargs():
    def f(*, x):
        return x * 2

    assert _maybe_call(f, x=3) == 6


def test_maybe_call_typeerror_required_positional_returns_func():
    def f(a):
        return a

    result = _maybe_call(f)
    assert result is f


def test_maybe_call_typeerror_takes_returns_func():
    def f(a):
        return a

    result = _maybe_call(f, 1, 2)
    assert result is f


def test_maybe_call_legit_typeerror_propagates():
    def f(a):
        return a + 1

    with pytest.raises(TypeError):
        # This raises inside the function body with a non-"required
        # positional argument" / "takes " message.
        _maybe_call(lambda: f("not-an-int" + 1))


# ---------------------------------------------------------------------------
# _init_wait_gen
# ---------------------------------------------------------------------------


def test_init_wait_gen_calls_wait_gen_with_kwargs():
    def gen(scale, base):
        def inner():
            return scale * base

        return inner

    result = _init_wait_gen(gen, {"scale": 2, "base": 3})
    assert result() == 6


def test_init_wait_gen_resolves_callable_values():
    def gen(value):
        return value

    result = _init_wait_gen(gen, {"value": lambda: 99})
    assert result == 99


def test_init_wait_gen_passes_through_non_callables():
    def gen(value):
        return value

    result = _init_wait_gen(gen, {"value": 7})
    assert result == 7


# ---------------------------------------------------------------------------
# _next_wait
# ---------------------------------------------------------------------------


class _FakeWait:
    def __init__(self, values):
        self.values = list(values)
        self.sent = []

    def next(self, value):
        self.sent.append(value)
        return self.values.pop(0)


def test_next_wait_basic():
    wait = _FakeWait([1.0])
    result = _next_wait(wait, None, None, 0.0, None)
    assert result == 1.0
    assert wait.sent == [None]


def test_next_wait_sends_value():
    wait = _FakeWait([2.5])
    result = _next_wait(wait, "sentinel", None, 0.0, None)
    assert result == 2.5
    assert wait.sent == ["sentinel"]


def test_next_wait_applies_jitter():
    wait = _FakeWait([1.0])
    result = _next_wait(wait, None, lambda v: v * 3, 0.0, None)
    assert result == 3.0


def test_next_wait_applies_test_multiplier():
    common._TEST_CONFIG["backoff_multiplier"] = 2.0
    wait = _FakeWait([1.5])
    result = _next_wait(wait, None, None, 0.0, None)
    assert result == 3.0


def test_next_wait_jitter_and_multiplier():
    common._TEST_CONFIG["backoff_multiplier"] = 0.5
    wait = _FakeWait([4.0])
    result = _next_wait(wait, None, lambda v: v + 1, 0.0, None)
    assert result == 2.5


def test_next_wait_caps_to_remaining_time():
    wait = _FakeWait([10.0])
    result = _next_wait(wait, None, None, 3.0, 5.0)
    assert result == 2.0


def test_next_wait_does_not_cap_when_under_max():
    wait = _FakeWait([1.0])
    result = _next_wait(wait, None, None, 3.0, 5.0)
    assert result == 1.0


def test_next_wait_no_max_time_no_cap():
    wait = _FakeWait([100.0])
    result = _next_wait(wait, None, None, 0.0, None)
    assert result == 100.0


# ---------------------------------------------------------------------------
# _apply_test_overrides
# ---------------------------------------------------------------------------


def test_apply_test_overrides_no_override():
    common._TEST_CONFIG["max_retries"] = None
    max_tries, max_time = _apply_test_overrides(max_tries=3, max_time=10)
    assert max_tries == 3
    assert max_time == 10


def test_apply_test_overrides_with_override():
    common._TEST_CONFIG["max_retries"] = 7
    max_tries, max_time = _apply_test_overrides(max_tries=3, max_time=10)
    assert max_tries == 7
    assert max_time == 10


def test_apply_test_overrides_defaults():
    common._TEST_CONFIG["max_retries"] = None
    max_tries, max_time = _apply_test_overrides()
    assert max_tries is None
    assert max_time is None


def test_apply_test_overrides_with_override_and_defaults():
    common._TEST_CONFIG["max_retries"] = 2
    max_tries, max_time = _apply_test_overrides()
    assert max_tries == 2
    assert max_time is None


# ---------------------------------------------------------------------------
# _prepare_logger
# ---------------------------------------------------------------------------


def test_prepare_logger_with_string():
    logger = _prepare_logger("some.logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "some.logger"


def test_prepare_logger_with_logger_instance():
    original = logging.getLogger("instance.logger")
    result = _prepare_logger(original)
    assert result is original


def test_prepare_logger_with_none():
    assert _prepare_logger(None) is None


# ---------------------------------------------------------------------------
# _LazyLogHandler
# ---------------------------------------------------------------------------


def test_lazy_log_handler_creates_lazily():
    calls = []

    def handler(details, *, logger, log_level):
        calls.append((details, logger, log_level))

    lazy = _LazyLogHandler(handler, "lazy.logger", logging.INFO)
    assert lazy._handler is None
    assert calls == []

    lazy("first")
    assert lazy._handler is not None
    assert len(calls) == 1
    assert calls[0][0] == "first"
    assert isinstance(calls[0][1], logging.Logger)
    assert calls[0][1].name == "lazy.logger"
    assert calls[0][2] == logging.INFO


def test_lazy_log_handler_reuses_handler():
    count = {"n": 0}

    def handler(details, *, logger, log_level):
        count["n"] += 1

    lazy = _LazyLogHandler(handler, logging.getLogger("x"), logging.WARNING)
    lazy("a")
    lazy("b")
    assert count["n"] == 2
    assert lazy._handler is not None


# ---------------------------------------------------------------------------
# _config_handlers
# ---------------------------------------------------------------------------


def test_config_handlers_list_and_no_logger_returns_same_list():
    handlers = ["a", "b"]
    result = _config_handlers(handlers, logger=None)
    assert result is handlers


def test_config_handlers_none_no_logger_returns_empty():
    result = _config_handlers(None, logger=None)
    assert result == []


def test_config_handlers_single_handler_no_logger():
    handler = object()
    result = _config_handlers(handler, logger=None)
    assert result == [handler]


def test_config_handlers_iterable_no_logger():
    result = _config_handlers(("x", "y"), logger=None)
    assert result == ["x", "y"]


def test_config_handlers_logger_none_user_handlers_uses_lazy():
    def default_handler(details, *, logger, log_level):
        pass

    result = _config_handlers(
        None,
        default_handler=default_handler,
        logger="cfg.logger",
        log_level=logging.INFO,
    )
    assert len(result) == 1
    assert isinstance(result[0], _LazyLogHandler)


def test_config_handlers_logger_with_user_handlers_iterable():
    user1, user2 = object(), object()

    def default_handler(details, *, logger, log_level):
        pass

    result = _config_handlers(
        [user1, user2],
        default_handler=default_handler,
        logger="cfg.logger",
        log_level=logging.INFO,
    )
    assert len(result) == 3
    assert result[1] is user1
    assert result[2] is user2
    partial = result[0]
    assert isinstance(partial, __import__("functools").partial)
    # Verify the partial's bound kwargs are correct.
    assert partial.keywords == {
        "logger": logging.getLogger("cfg.logger"),
        "log_level": logging.INFO,
    }


def test_config_handlers_logger_with_single_user_handler():
    user = object()

    def default_handler(details, *, logger, log_level):
        pass

    result = _config_handlers(
        user,
        default_handler=default_handler,
        logger="cfg.logger",
        log_level=logging.INFO,
    )
    assert len(result) == 2
    assert result[1] is user


def test_config_handlers_logger_none_user_handlers_none():
    result = _config_handlers(None, logger=None)
    assert result == []


# ---------------------------------------------------------------------------
# _log_backoff / _log_giveup
# ---------------------------------------------------------------------------


class _RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg, args))


def _target():
    pass


def test_log_backoff_with_exception():
    logger = _RecordingLogger()
    exc = ValueError("bad")
    details = {"target": _target, "wait": 1.5, "exception": exc}
    _log_backoff(details, logger, logging.WARNING)
    assert len(logger.records) == 1
    level, msg, args = logger.records[0]
    assert level == logging.WARNING
    assert msg == "Backing off %s(...) for %.1fs (%s)"
    assert args == ("_target", 1.5, "ValueError: bad")


def test_log_backoff_with_value():
    logger = _RecordingLogger()
    details = {"target": _target, "wait": 2.25, "value": "some-value"}
    _log_backoff(details, logger, logging.INFO)
    level, msg, args = logger.records[0]
    assert level == logging.INFO
    assert args == ("_target", 2.25, "some-value")


def test_log_backoff_default_unknown():
    logger = _RecordingLogger()
    details = {"target": _target, "wait": 0.5}
    _log_backoff(details, logger, logging.DEBUG)
    _, _, args = logger.records[0]
    assert args == ("_target", 0.5, "unknown")


def test_log_backoff_exception_none_uses_value():
    logger = _RecordingLogger()
    details = {"target": _target, "wait": 1.0, "exception": None, "value": "v"}
    _log_backoff(details, logger, logging.INFO)
    _, _, args = logger.records[0]
    assert args == ("_target", 1.0, "v")


def test_log_giveup_with_exception():
    logger = _RecordingLogger()
    exc = RuntimeError("boom")
    details = {"target": _target, "tries": 3, "exception": exc}
    _log_giveup(details, logger, logging.ERROR)
    level, msg, args = logger.records[0]
    assert level == logging.ERROR
    assert msg == "Giving up %s(...) after %d tries (%s)"
    assert args == ("_target", 3, "RuntimeError: boom")


def test_log_giveup_with_value():
    logger = _RecordingLogger()
    details = {"target": _target, "tries": 5, "value": "some-value"}
    _log_giveup(details, logger, logging.WARNING)
    _, _, args = logger.records[0]
    assert args == ("_target", 5, "some-value")


def test_log_giveup_default_unknown():
    logger = _RecordingLogger()
    details = {"target": _target, "tries": 1}
    _log_giveup(details, logger, logging.WARNING)
    _, _, args = logger.records[0]
    assert args == ("_target", 1, "unknown")


# ---------------------------------------------------------------------------
# _now / _elapsed (monkeypatched, no reliance on real clock)
# ---------------------------------------------------------------------------


def test_now_uses_monotonic_ns(monkeypatch):
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 2_000_000_000)
    assert _now() == 2.0


def test_now_zero(monkeypatch):
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 0)
    assert _now() == 0.0


def test_elapsed(monkeypatch):
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 5_000_000_000)
    assert _elapsed(2.0) == 3.0


def test_elapsed_negative(monkeypatch):
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 1_000_000_000)
    assert _elapsed(2.5) == -1.5


# ---------------------------------------------------------------------------
# _is_custom_wait
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, -1, 3.14, 0.0, -2.5])
def test_is_custom_wait_numbers(value):
    assert _is_custom_wait(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_is_custom_wait_bools_are_false(value):
    assert _is_custom_wait(value) is False


@pytest.mark.parametrize("value", ["1", None, [1], (1.0,), {"a": 1}, object()])
def test_is_custom_wait_non_numbers(value):
    assert _is_custom_wait(value) is False


# ---------------------------------------------------------------------------
# _check_hot_loop
# ---------------------------------------------------------------------------


def test_check_hot_loop_no_warning_under_threshold(monkeypatch, caplog):
    clock = {"t": 100.0}
    monkeypatch.setattr(
        common.time_module, "monotonic_ns", lambda: int(clock["t"] * 1_000_000_000)
    )
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0

    with caplog.at_level(logging.WARNING, logger="backon"):
        # 4 rapid retries in quick succession: count grows but not yet 5.
        for i in range(4):
            clock["t"] += 0.01
            _check_hot_loop()

    assert common._hot_loop_data["count"] == 3
    assert common._hot_loop_data["last_retry"] == pytest.approx(100.04)
    assert len(caplog.records) == 0


def test_check_hot_loop_warns_at_threshold(monkeypatch, caplog):
    clock = {"t": 200.0}
    monkeypatch.setattr(
        common.time_module, "monotonic_ns", lambda: int(clock["t"] * 1_000_000_000)
    )
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0

    with caplog.at_level(logging.WARNING, logger="backon"):
        # first call establishes prev=0 -> count=0
        _check_hot_loop()
        # subsequent rapid calls increment count; at count==5 it warns/resets.
        for i in range(5):
            clock["t"] += 0.01
            _check_hot_loop()

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "Hot loop detected" in record.getMessage()
    assert "5 retries" in record.getMessage()
    # After warning, count reset to 0.
    assert common._hot_loop_data["count"] == 0


def test_check_hot_loop_resets_count_on_slow_retry(monkeypatch, caplog):
    clock = {"t": 300.0}
    monkeypatch.setattr(
        common.time_module, "monotonic_ns", lambda: int(clock["t"] * 1_000_000_000)
    )
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0

    with caplog.at_level(logging.WARNING, logger="backon"):
        _check_hot_loop()  # count 0
        clock["t"] += 0.01
        _check_hot_loop()  # count 1
        clock["t"] += 0.01
        _check_hot_loop()  # count 2
        clock["t"] += 1.0  # slow retry -> reset count
        _check_hot_loop()  # count 0

    assert common._hot_loop_data["count"] == 0
    assert len(caplog.records) == 0


def test_check_hot_loop_first_call_sets_last_retry(monkeypatch):
    clock = {"t": 50.0}
    monkeypatch.setattr(
        common.time_module, "monotonic_ns", lambda: int(clock["t"] * 1_000_000_000)
    )
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0

    _check_hot_loop()
    assert common._hot_loop_data["last_retry"] == pytest.approx(50.0)
    assert common._hot_loop_data["count"] == 0
