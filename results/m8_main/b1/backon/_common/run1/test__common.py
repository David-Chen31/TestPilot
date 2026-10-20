import logging
import pytest

import backon._common as common
from backon._common import disable, enable, is_enabled


@pytest.fixture(autouse=True)
def reset_global_state():
    """Ensure the global enabled flag is restored after each test."""
    original = common._GLOBAL_ENABLED
    yield
    common._GLOBAL_ENABLED = original


# ---------------------------------------------------------------------------
# disable / enable / is_enabled
# ---------------------------------------------------------------------------

def test_is_enabled_default_true():
    common._GLOBAL_ENABLED = True
    assert is_enabled() is True


def test_disable_sets_false():
    common._GLOBAL_ENABLED = True
    disable()
    assert is_enabled() is False
    assert common._GLOBAL_ENABLED is False


def test_enable_sets_true():
    common._GLOBAL_ENABLED = False
    enable()
    assert is_enabled() is True
    assert common._GLOBAL_ENABLED is True


def test_disable_is_idempotent():
    common._GLOBAL_ENABLED = True
    disable()
    disable()
    assert is_enabled() is False


def test_enable_is_idempotent():
    common._GLOBAL_ENABLED = False
    enable()
    enable()
    assert is_enabled() is True


def test_toggle_sequence():
    common._GLOBAL_ENABLED = True
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True
    disable()
    assert is_enabled() is False


# ---------------------------------------------------------------------------
# _maybe_call
# ---------------------------------------------------------------------------

def test_maybe_call_non_callable():
    assert common._maybe_call(5) == 5
    assert common._maybe_call("abc") == "abc"
    assert common._maybe_call(None) is None


def test_maybe_call_callable_no_args():
    assert common._maybe_call(lambda: 42) == 42


def test_maybe_call_callable_with_args():
    assert common._maybe_call(lambda x, y: x + y, 1, 2) == 3
    assert common._maybe_call(lambda x, y=0: x * y, 3, y=4) == 12


def test_maybe_call_callable_missing_args_returns_callable():
    def needs_arg(x):
        return x

    result = common._maybe_call(needs_arg)
    assert result is needs_arg


def test_maybe_call_rejects_too_many_args():
    def fn(a, b):
        return a + b

    # TypeError message contains "takes ", so it should return the callable
    result = common._maybe_call(fn, 1, 2, 3)
    assert result is fn


def test_maybe_call_reraises_unrelated_type_error():
    def fn(x):
        raise TypeError("some unrelated problem")

    with pytest.raises(TypeError, match="some unrelated problem"):
        common._maybe_call(fn, 1)


# ---------------------------------------------------------------------------
# _init_wait_gen
# ---------------------------------------------------------------------------

def test_init_wait_gen_calls_gen_with_kwargs():
    captured = {}

    def gen(**kwargs):
        captured.update(kwargs)
        return "gen-instance"

    class Gen:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return "gen-instance"

    result = common._init_wait_gen(Gen(), {"a": 1, "b": lambda: 2})
    assert result == "gen-instance"
    assert captured == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# _next_wait
# ---------------------------------------------------------------------------

class _FakeWait:
    def __init__(self, values):
        self._values = list(values)
        self.sent = []

    def next(self, send_value):
        self.sent.append(send_value)
        return self._values.pop(0)


def test_next_wait_no_jitter_no_max():
    fake = _FakeWait([3])
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    assert common._next_wait(fake, None, None, 0.0, None) == 3
    assert fake.sent == [None]


def test_next_wait_with_jitter():
    fake = _FakeWait([3])
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    result = common._next_wait(fake, "send", lambda v: v + 1, 0.0, None)
    assert result == 4
    assert fake.sent == ["send"]


def test_next_wait_multiplier():
    fake = _FakeWait([10])
    common._TEST_CONFIG["backoff_multiplier"] = 0.5
    assert common._next_wait(fake, None, None, 0.0, None) == 5


def test_next_wait_clamped_by_max_time():
    fake = _FakeWait([10])
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    # max_time - elapsed = 4
    assert common._next_wait(fake, None, None, 1.0, 5.0) == 4


def test_next_wait_not_clamped_below_max_time():
    fake = _FakeWait([2])
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    assert common._next_wait(fake, None, None, 1.0, 5.0) == 2


# ---------------------------------------------------------------------------
# _apply_test_overrides
# ---------------------------------------------------------------------------

def test_apply_test_overrides_none():
    common._TEST_CONFIG["max_retries"] = None
    assert common._apply_test_overrides(3, 10) == (3, 10)


def test_apply_test_overrides_set():
    common._TEST_CONFIG["max_retries"] = 7
    assert common._apply_test_overrides(3, 10) == (7, 10)
    common._TEST_CONFIG["max_retries"] = None


# ---------------------------------------------------------------------------
# _prepare_logger
# ---------------------------------------------------------------------------

def test_prepare_logger_string():
    logger = common._prepare_logger("my.test.logger")
    assert logger is logging.getLogger("my.test.logger")


def test_prepare_logger_object():
    obj = logging.getLogger("other")
    assert common._prepare_logger(obj) is obj


# ---------------------------------------------------------------------------
# _config_handlers
# ---------------------------------------------------------------------------

def test_config_handlers_list_and_no_logger_returns_same_list():
    handlers = [lambda d: None]
    result = common._config_handlers(handlers)
    assert result is handlers


def test_config_handlers_none_logger_none_returns_empty():
    assert common._config_handlers(None) == []


def test_config_handlers_single_user_handler():
    handler = lambda d: None
    result = common._config_handlers(handler)
    assert result == [handler]


def test_config_handlers_iterable_user_handlers():
    handlers = [lambda d: None, lambda d: None]
    result = common._config_handlers(handlers)
    assert result == handlers


def test_config_handlers_with_logger_and_no_user_handlers():
    def default_handler(details, logger, log_level):
        pass

    result = common._config_handlers(
        None, default_handler=default_handler, logger="x", log_level=logging.INFO
    )
    assert len(result) == 1
    assert isinstance(result[0], common._LazyLogHandler)


def test_config_handlers_with_logger_and_user_handlers_iterable():
    def default_handler(details, logger, log_level):
        pass

    uh1 = lambda d: None
    uh2 = lambda d: None
    result = common._config_handlers(
        [uh1, uh2],
        default_handler=default_handler,
        logger="x",
        log_level=logging.INFO,
    )
    assert len(result) == 3
    assert result[1] is uh1
    assert result[2] is uh2


def test_config_handlers_with_logger_and_single_user_handler():
    def default_handler(details, logger, log_level):
        pass

    uh = lambda d: None
    result = common._config_handlers(
        uh, default_handler=default_handler, logger="x", log_level=logging.INFO
    )
    assert len(result) == 2
    assert result[1] is uh


# ---------------------------------------------------------------------------
# _LazyLogHandler
# ---------------------------------------------------------------------------

def test_lazy_log_handler_defers_factory():
    calls = []

    def factory_handler(details, logger, log_level):
        calls.append((details, logger, log_level))

    lazy = common._LazyLogHandler(factory_handler, "logname", logging.INFO)
    assert lazy._handler is None
    lazy({"x": 1})
    assert lazy._handler is not None
    assert calls == [({"x": 1}, "logname", logging.INFO)]
    # second call should not re-create
    first = lazy._handler
    lazy({"x": 2})
    assert lazy._handler is first
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# _log_backoff / _log_giveup
# ---------------------------------------------------------------------------

class _RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg, args))


def test_log_backoff_with_exception():
    logger = _RecordingLogger()

    def target():
        pass

    exc = ValueError("boom")
    details = {"exception": exc, "target": target, "wait": 1.5}
    common._log_backoff(details, logger, logging.WARNING)
    level, msg, args = logger.records[0]
    assert level == logging.WARNING
    assert msg == "Backing off %s(...) for %.1fs (%s)"
    assert args == (target.__name__, 1.5, "ValueError: boom")


def test_log_backoff_without_exception():
    logger = _RecordingLogger()

    def target():
        pass

    details = {"target": target, "wait": 2.0, "value": "someval"}
    common._log_backoff(details, logger, logging.INFO)
    _, _, args = logger.records[0]
    assert args == (target.__name__, 2.0, "someval")


def test_log_backoff_without_exception_or_value():
    logger = _RecordingLogger()

    def target():
        pass

    details = {"target": target, "wait": 2.0}
    common._log_backoff(details, logger, logging.INFO)
    _, _, args = logger.records[0]
    assert args == (target.__name__, 2.0, "unknown")


def test_log_giveup_with_exception():
    logger = _RecordingLogger()

    def target():
        pass

    exc = RuntimeError("fail")
    details = {"exception": exc, "target": target, "tries": 3}
    common._log_giveup(details, logger, logging.ERROR)
    level, msg, args = logger.records[0]
    assert level == logging.ERROR
    assert msg == "Giving up %s(...) after %d tries (%s)"
    assert args == (target.__name__, 3, "RuntimeError: fail")


def test_log_giveup_without_exception():
    logger = _RecordingLogger()

    def target():
        pass

    details = {"target": target, "tries": 5, "value": "ret-val"}
    common._log_giveup(details, logger, logging.ERROR)
    _, _, args = logger.records[0]
    assert args == (target.__name__, 5, "ret-val")


def test_log_giveup_without_exception_or_value():
    logger = _RecordingLogger()

    def target():
        pass

    details = {"target": target, "tries": 5}
    common._log_giveup(details, logger, logging.ERROR)
    _, _, args = logger.records[0]
    assert args == (target.__name__, 5, "unknown")


# ---------------------------------------------------------------------------
# _now / _elapsed
# ---------------------------------------------------------------------------

def test_now_and_elapsed(monkeypatch):
    monkeypatch.setattr(common, "_now", lambda: 100.0)
    assert common._now() == 100.0
    assert common._elapsed(90.0) == 10.0


# ---------------------------------------------------------------------------
# _is_custom_wait
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [1, 0, 2.5, -3.0])
def test_is_custom_wait_true(value):
    assert common._is_custom_wait(value) is True


@pytest.mark.parametrize("value", [True, False, "abc", None, [1], (1, 2)])
def test_is_custom_wait_false(value):
    assert common._is_custom_wait(value) is False


# ---------------------------------------------------------------------------
# _check_hot_loop
# ---------------------------------------------------------------------------

@pytest.fixture
def reset_hot_loop():
    original = dict(common._hot_loop_data)
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0
    yield
    common._hot_loop_data.update(original)


def test_check_hot_loop_first_call_no_warning(reset_hot_loop, caplog):
    common._check_hot_loop()
    assert common._hot_loop_data["count"] == 0
    assert "Hot loop detected" not in caplog.text


def test_check_hot_loop_increments_within_interval(reset_hot_loop, monkeypatch):
    times = iter([100.0, 100.02, 100.04])

    class _FakeTime:
        @staticmethod
        def monotonic_ns():
            return int(next(times) * 1_000_000_000)

    monkeypatch.setattr(common, "time_module", _FakeTime)

    common._check_hot_loop()  # prev = 0 -> count stays 0, last_retry=100
    common._check_hot_loop()  # within 0.1 -> count = 1
    common._check_hot_loop()  # within 0.1 -> count = 2
    assert common._hot_loop_data["count"] == 2


def test_check_hot_loop_resets_when_gap_large(reset_hot_loop, monkeypatch):
    times = iter([100.0, 200.0])

    class _FakeTime:
        @staticmethod
        def monotonic_ns():
            return int(next(times) * 1_000_000_000)

    monkeypatch.setattr(common, "time_module", _FakeTime)

    common._check_hot_loop()  # prev=0 -> count 0
    common._check_hot_loop()  # gap large -> count reset to 0
    assert common._hot_loop_data["count"] == 0


def test_check_hot_loop_warns_at_threshold(reset_hot_loop, monkeypatch, caplog):
    # call1: prev=0 -> count=0
    # call2: within -> 1
    # call3: within -> 2
    # call4: within -> 3
    # call5: within -> 4
    # call6: within -> 5 -> warning, reset to 0
    times = iter([100.0 + i * 0.01 for i in range(6)])

    class _FakeTime:
        @staticmethod
        def monotonic_ns():
            return int(next(times) * 1_000_000_000)

    monkeypatch.setattr(common, "time_module", _FakeTime)

    with caplog.at_level(logging.WARNING, logger="backon"):
        for _ in range(6):
            common._check_hot_loop()

    assert "Hot loop detected" in caplog.text
    assert common._hot_loop_data["count"] == 0


def test_check_hot_loop_lock_exists():
    # sanity: lock is a threading.Lock
    assert hasattr(common._hot_loop_lock, "acquire")
    assert hasattr(common._hot_loop_lock, "release")
