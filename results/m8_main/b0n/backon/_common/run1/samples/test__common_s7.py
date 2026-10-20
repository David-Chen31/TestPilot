import logging

import pytest

import backon._common as common


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Ensure global enabled flag is restored after each test."""
    original = common._GLOBAL_ENABLED
    yield
    common._GLOBAL_ENABLED = original


# ---------------------------------------------------------------------------
# Public API: disable / enable / is_enabled
# ---------------------------------------------------------------------------

class TestIsEnabled:
    def test_default_is_enabled(self):
        common.enable()
        assert common.is_enabled() is True

    def test_returns_bool_true(self):
        common.enable()
        result = common.is_enabled()
        assert result is True
        assert isinstance(result, bool)

    def test_returns_bool_false(self):
        common.disable()
        result = common.is_enabled()
        assert result is False
        assert isinstance(result, bool)


class TestDisable:
    def test_disable_sets_global_false(self):
        common.enable()
        assert common._GLOBAL_ENABLED is True
        common.disable()
        assert common._GLOBAL_ENABLED is False
        assert common.is_enabled() is False

    def test_disable_returns_none(self):
        assert common.disable() is None

    def test_disable_is_idempotent(self):
        common.disable()
        common.disable()
        assert common._GLOBAL_ENABLED is False


class TestEnable:
    def test_enable_sets_global_true(self):
        common.disable()
        assert common._GLOBAL_ENABLED is False
        common.enable()
        assert common._GLOBAL_ENABLED is True
        assert common.is_enabled() is True

    def test_enable_returns_none(self):
        assert common.enable() is None

    def test_enable_is_idempotent(self):
        common.enable()
        common.enable()
        assert common._GLOBAL_ENABLED is True


class TestToggleRoundTrip:
    def test_toggle_sequence(self):
        common.enable()
        assert common.is_enabled() is True
        common.disable()
        assert common.is_enabled() is False
        common.enable()
        assert common.is_enabled() is True
        common.disable()
        assert common.is_enabled() is False


# ---------------------------------------------------------------------------
# Supporting helpers exercised indirectly / directly
# ---------------------------------------------------------------------------

class TestMaybeCall:
    def test_callable_called_with_args(self):
        assert common._maybe_call(lambda x: x + 1, 5) == 6

    def test_callable_with_kwargs(self):
        assert common._maybe_call(lambda x, y=0: x + y, 1, y=2) == 3

    def test_non_callable_returned_as_is(self):
        assert common._maybe_call(42) == 42
        sentinel = object()
        assert common._maybe_call(sentinel) is sentinel

    def test_callable_missing_required_arg_returns_callable(self):
        def needs_arg(a, b):
            return a + b

        result = common._maybe_call(needs_arg, 1)
        assert result is needs_arg

    def test_legitimate_type_error_propagates(self):
        def raises_type_error(a):
            raise TypeError("boom")

        with pytest.raises(TypeError, match="boom"):
            common._maybe_call(raises_type_error, 1)

    def test_correct_call_proceeds(self):
        def add(a, b):
            return a + b

        assert common._maybe_call(add, 2, 3) == 5


class TestInitWaitGen:
    def test_passes_through_kwargs(self):
        def gen(**kwargs):
            return kwargs

        result = common._init_wait_gen(gen, {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_calls_callable_kwarg(self):
        def gen(**kwargs):
            return kwargs

        result = common._init_wait_gen(gen, {"a": lambda: 5})
        assert result == {"a": 5}

    def test_non_callable_kwarg_preserved(self):
        def gen(**kwargs):
            return kwargs

        result = common._init_wait_gen(gen, {"x": "value"})
        assert result == {"x": "value"}


class TestNextWait:
    def test_no_jitter_no_max_time(self):
        class Wait:
            def next(self, value):
                return 4

        result = common._next_wait(Wait(), None, None, 0, None)
        # backoff_multiplier default is 1.0
        assert result == 4

    def test_applies_jitter(self):
        class Wait:
            def next(self, value):
                return 4

        result = common._next_wait(Wait(), None, lambda v: v * 2, 0, None)
        assert result == 8

    def test_applies_backoff_multiplier(self, monkeypatch):
        class Wait:
            def next(self, value):
                return 4

        monkeypatch.setitem(common._TEST_CONFIG, "backoff_multiplier", 0.5)
        result = common._next_wait(Wait(), None, None, 0, None)
        assert result == 2

    def test_max_time_caps_value(self):
        class Wait:
            def next(self, value):
                return 10

        result = common._next_wait(Wait(), None, None, 3, 5)
        # max_time - elapsed = 5 - 3 = 2, so min(10, 2) = 2
        assert result == 2

    def test_send_value_passed_through(self):
        class Wait:
            def next(self, value):
                return value

        result = common._next_wait(Wait(), 7, None, 0, None)
        assert result == 7


class TestApplyTestOverrides:
    def test_no_override(self, monkeypatch):
        monkeypatch.setitem(common._TEST_CONFIG, "max_retries", None)
        max_tries, max_time = common._apply_test_overrides(max_tries=3, max_time=10)
        assert max_tries == 3
        assert max_time == 10

    def test_override_max_retries(self, monkeypatch):
        monkeypatch.setitem(common._TEST_CONFIG, "max_retries", 5)
        max_tries, max_time = common._apply_test_overrides(max_tries=3, max_time=10)
        assert max_tries == 5
        assert max_time == 10


class TestPrepareLogger:
    def test_string_returns_logger(self):
        logger = common._prepare_logger("my.test.logger")
        assert logger is logging.getLogger("my.test.logger")

    def test_logger_passthrough(self):
        logger = logging.getLogger("some.logger")
        assert common._prepare_logger(logger) is logger


class TestLazyLogHandler:
    def test_defers_until_called(self):
        calls = []

        def handler(details, *, logger, log_level):
            calls.append((details, logger, log_level))

        lazy = common._LazyLogHandler(handler, "log.name", logging.INFO)
        assert lazy._handler is None
        assert calls == []
        lazy({"k": "v"})
        assert len(calls) == 1
        assert calls[0][0] == {"k": "v"}
        assert calls[0][2] == logging.INFO
        assert lazy._handler is not None

    def test_reuses_handler(self):
        calls = []

        def handler(details, *, logger, log_level):
            calls.append(details)

        lazy = common._LazyLogHandler(handler, "log.name", logging.INFO)
        lazy({"a": 1})
        first = lazy._handler
        lazy({"b": 2})
        assert lazy._handler is first
        assert calls == [{"a": 1}, {"b": 2}]


class TestConfigHandlers:
    def test_list_and_no_logger_returns_same_list(self):
        handlers = [1, 2, 3]
        result = common._config_handlers(handlers)
        assert result is handlers

    def test_logger_none_user_handlers_none_returns_empty(self):
        assert common._config_handlers(None) == []

    def test_logger_and_none_user_handlers_uses_lazy(self):
        def handler(details, *, logger, log_level):
            return details

        result = common._config_handlers(
            None, default_handler=handler, logger="x", log_level=logging.INFO
        )
        assert len(result) == 1
        assert isinstance(result[0], common._LazyLogHandler)

    def test_logger_and_single_handler(self):
        def handler(details, *, logger, log_level):
            return details

        def user(details):
            return details

        result = common._config_handlers(
            user, default_handler=handler, logger="x", log_level=logging.INFO
        )
        assert len(result) == 2

    def test_logger_and_iterable_handlers(self):
        def handler(details, *, logger, log_level):
            return details

        users = [lambda d: 1, lambda d: 2]
        result = common._config_handlers(
            users, default_handler=handler, logger="x", log_level=logging.INFO
        )
        assert len(result) == 3

    def test_user_handlers_only_single(self):
        user = lambda d: d
        result = common._config_handlers(user)
        assert result == [user]

    def test_user_handlers_only_iterable(self):
        users = [lambda d: 1, lambda d: 2]
        result = common._config_handlers(users)
        assert result == users

    def test_user_handlers_none_no_logger_empty(self):
        assert common._config_handlers(None) == []


class TestLogBackoff:
    def test_logs_with_exception(self, caplog):
        def target():
            pass

        details = {"target": target, "wait": 1.23, "exception": ValueError("bad")}
        logger = logging.getLogger("backon._common.test_backoff")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            common._log_backoff(details, logger, logging.WARNING)
        assert "Backing off" in caplog.text
        assert "ValueError: bad" in caplog.text

    def test_logs_without_exception(self, caplog):
        def target():
            pass

        details = {"target": target, "wait": 0.5, "value": "someval"}
        logger = logging.getLogger("backon._common.test_backoff2")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            common._log_backoff(details, logger, logging.WARNING)
        assert "Backing off" in caplog.text
        assert "someval" in caplog.text

    def test_unknown_value_default(self, caplog):
        def target():
            pass

        details = {"target": target, "wait": 0.1}
        logger = logging.getLogger("backon._common.test_backoff3")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            common._log_backoff(details, logger, logging.WARNING)
        assert "unknown" in caplog.text


class TestLogGiveup:
    def test_logs_with_exception(self, caplog):
        def target():
            pass

        details = {"target": target, "tries": 3, "exception": KeyError("k")}
        logger = logging.getLogger("backon._common.test_giveup")
        with caplog.at_level(logging.ERROR, logger=logger.name):
            common._log_giveup(details, logger, logging.ERROR)
        assert "Giving up" in caplog.text
        assert "KeyError" in caplog.text

    def test_logs_without_exception(self, caplog):
        def target():
            pass

        details = {"target": target, "tries": 5, "value": "retval"}
        logger = logging.getLogger("backon._common.test_giveup2")
        with caplog.at_level(logging.ERROR, logger=logger.name):
            common._log_giveup(details, logger, logging.ERROR)
        assert "Giving up" in caplog.text
        assert "retval" in caplog.text


class TestNowElapsed:
    def test_now_returns_float(self):
        result = common._now()
        assert isinstance(result, float)

    def test_elapsed_start_is_now(self):
        start = common._now()
        result = common._elapsed(start)
        assert isinstance(result, float)
        assert result >= 0.0


class TestIsCustomWait:
    def test_int(self):
        assert common._is_custom_wait(5) is True

    def test_float(self):
        assert common._is_custom_wait(1.5) is True

    def test_bool_false(self):
        assert common._is_custom_wait(True) is False
        assert common._is_custom_wait(False) is False

    def test_non_numeric(self):
        assert common._is_custom_wait("1") is False
        assert common._is_custom_wait(None) is False
        assert common._is_custom_wait([1]) is False


class TestCheckHotLoop:
    def test_no_warning_on_first_call(self, monkeypatch, caplog):
        monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 1_000_000_000)
        common._hot_loop_data["last_retry"] = 0.0
        common._hot_loop_data["count"] = 0
        logger = logging.getLogger("backon")
        with caplog.at_level(logging.WARNING, logger="backon"):
            common._check_hot_loop()
        assert common._hot_loop_data["count"] == 0
        # no hot loop warning
        assert "Hot loop detected" not in caplog.text

    def test_detects_hot_loop(self, caplog):
        # Simulate rapid successive retries via monotonic_ns calls
        times = [0, 1_000_000_000] + [1_000_000_000 + i * 10_000_000 for i in range(1, 7)]
        idx = {"i": 0}

        def fake_ns():
            iv = idx["i"]
            if iv < len(times):
                idx["i"] += 1
                return times[iv]
            return times[-1]

        original = common.time_module.monotonic_ns
        common.time_module.monotonic_ns = fake_ns
        try:
            common._hot_loop_data["last_retry"] = 0.0
            common._hot_loop_data["count"] = 0
            with caplog.at_level(logging.WARNING, logger="backon"):
                for _ in range(10):
                    common._check_hot_loop()
        finally:
            common.time_module.monotonic_ns = original
            common._hot_loop_data["last_retry"] = 0.0
            common._hot_loop_data["count"] = 0

        assert "Hot loop detected" in caplog.text
