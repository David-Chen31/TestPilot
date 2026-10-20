from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from claude_converter.converter import Converter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mocks(monkeypatch):
    """Patch every underlying function used by Converter and return them."""
    import claude_converter.converter as converter_module

    claude_load = MagicMock(name="load_session_claude")
    claude_records = MagicMock(name="records_to_messages_claude")
    claude_session = MagicMock(name="session_to_messages_claude")
    claude_inspect = MagicMock(name="inspect_session_claude")

    codex_load = MagicMock(name="load_session_codex")
    codex_records = MagicMock(name="records_to_messages_codex")
    codex_session = MagicMock(name="session_to_messages_codex")
    codex_inspect = MagicMock(name="inspect_session_codex")

    pi_load = MagicMock(name="load_session_pi")
    pi_records = MagicMock(name="records_to_messages_pi")
    pi_session = MagicMock(name="session_to_messages_pi")
    pi_inspect = MagicMock(name="inspect_session_pi")

    monkeypatch.setattr(converter_module, "load_session_claude", claude_load)
    monkeypatch.setattr(converter_module, "records_to_messages_claude", claude_records)
    monkeypatch.setattr(converter_module, "session_to_messages_claude", claude_session)
    monkeypatch.setattr(converter_module, "inspect_session_claude", claude_inspect)

    monkeypatch.setattr(converter_module, "load_session_codex", codex_load)
    monkeypatch.setattr(converter_module, "records_to_messages_codex", codex_records)
    monkeypatch.setattr(converter_module, "session_to_messages_codex", codex_session)
    monkeypatch.setattr(converter_module, "inspect_session_codex", codex_inspect)

    monkeypatch.setattr(converter_module, "load_session_pi", pi_load)
    monkeypatch.setattr(converter_module, "records_to_messages_pi", pi_records)
    monkeypatch.setattr(converter_module, "session_to_messages_pi", pi_session)
    monkeypatch.setattr(converter_module, "inspect_session_pi", pi_inspect)

    return {
        "claude": {
            "load": claude_load,
            "records": claude_records,
            "session": claude_session,
            "inspect": claude_inspect,
        },
        "codex": {
            "load": codex_load,
            "records": codex_records,
            "session": codex_session,
            "inspect": codex_inspect,
        },
        "pi": {
            "load": pi_load,
            "records": pi_records,
            "session": pi_session,
            "inspect": pi_inspect,
        },
    }


CONVERTERS = ["claude-code", "codex", "pi"]


def _backend(name: str) -> str:
    return {"claude-code": "claude", "codex": "codex", "pi": "pi"}[name]


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_converter_is_claude_code(self):
        c = Converter()
        assert c.converter == "claude-code"

    def test_explicit_converter_name(self):
        assert Converter("codex").converter == "codex"
        assert Converter("pi").converter == "pi"
        assert Converter("claude-code").converter == "claude-code"

    def test_stores_attribute_directly(self):
        # Not validated/coerced at construction time.
        c = Converter("weird")
        assert c.converter == "weird"


# ---------------------------------------------------------------------------
# load_session
# ---------------------------------------------------------------------------


class TestLoadSession:
    @pytest.mark.parametrize("name", CONVERTERS)
    def test_dispatch_and_return_value(self, name, mocks, tmp_path):
        backend = _backend(name)
        sentinel = [{"r": name}]
        mocks[backend]["load"].return_value = sentinel
        path = str(tmp_path / "session.jsonl")

        result = Converter(name).load_session(path)

        assert result is sentinel
        mocks[backend]["load"].assert_called_once_with(path)

    @pytest.mark.parametrize("name", CONVERTERS)
    def test_path_object_is_passed_through(self, name, mocks, tmp_path):
        backend = _backend(name)
        p = tmp_path / "session.jsonl"

        Converter(name).load_session(p)

        mocks[backend]["load"].assert_called_once_with(p)

    def test_unknown_converter_raises_value_error(self):
        c = Converter("bogus")
        with pytest.raises(ValueError) as excinfo:
            c.load_session("x.jsonl")
        assert "Unknown converter" in str(excinfo.value)
        assert "bogus" in str(excinfo.value)

    def test_empty_converter_raises_value_error(self):
        with pytest.raises(ValueError):
            Converter("").load_session("x.jsonl")


# ---------------------------------------------------------------------------
# records_to_messages
# ---------------------------------------------------------------------------


class TestRecordsToMessages:
    @pytest.mark.parametrize("name", CONVERTERS)
    def test_dispatch_and_return_value(self, name, mocks):
        backend = _backend(name)
        records = [{"type": "user"}, {"type": "assistant"}]
        sentinel = [{"role": "user", "content": "hi"}]
        mocks[backend]["records"].return_value = sentinel

        result = Converter(name).records_to_messages(records)

        assert result is sentinel
        mocks[backend]["records"].assert_called_once_with(records)

    @pytest.mark.parametrize("name", CONVERTERS)
    def test_empty_records_still_data(self, name, mocks):
        backend = _backend(name)
        mocks[backend]["records"].return_value = []

        result = Converter(name).records_to_messages([])

        assert result == []
        mocks[backend]["records"].assert_called_once_with([])

    def test_unknown_converter_raises_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            Converter("nope").records_to_messages([])
        assert "Unknown converter" in str(excinfo.value)
        assert "nope" in str(excinfo.value)


# ---------------------------------------------------------------------------
# session_to_messages
# ---------------------------------------------------------------------------


class TestSessionToMessages:
    @pytest.mark.parametrize("name", CONVERTERS)
    def test_dispatch_default_output(self, name, mocks, tmp_path):
        backend = _backend(name)
        sentinel = [{"role": "user", "content": "x"}]
        mocks[backend]["session"].return_value = sentinel
        path = str(tmp_path / "session.jsonl")

        result = Converter(name).session_to_messages(path)

        assert result is sentinel
        mocks[backend]["session"].assert_called_once_with(path, output=None)

    @pytest.mark.parametrize("name", CONVERTERS)
    def test_dispatch_with_output(self, name, mocks, tmp_path):
        backend = _backend(name)
        sentinel = [{"role": "assistant", "content": "y"}]
        mocks[backend]["session"].return_value = sentinel
        path = tmp_path / "session.jsonl"
        out = tmp_path / "out" / "messages.json"

        result = Converter(name).session_to_messages(path, output=out)

        assert result is sentinel
        mocks[backend]["session"].assert_called_once_with(path, output=out)

    def test_dispatch_with_output_string(self, mocks, tmp_path):
        sentinel = []
        mocks["claude"]["session"].return_value = sentinel
        path = str(tmp_path / "s.jsonl")
        out = str(tmp_path / "messages.json")

        result = Converter("claude-code").session_to_messages(path, output=out)

        assert result is sentinel
        mocks["claude"]["session"].assert_called_once_with(path, output=out)

    def test_unknown_converter_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            Converter("zzz").session_to_messages(tmp_path / "x.jsonl")
        assert "Unknown converter" in str(excinfo.value)
        assert "zzz" in str(excinfo.value)


# ---------------------------------------------------------------------------
# inspect_session
# ---------------------------------------------------------------------------


class TestInspectSession:
    @pytest.mark.parametrize("name", CONVERTERS)
    def test_dispatch_defaults(self, name, mocks, tmp_path):
        backend = _backend(name)
        path = str(tmp_path / "session.jsonl")

        result = Converter(name).inspect_session(path)

        assert result is None
        mocks[backend]["inspect"].assert_called_once_with(
            path, show_flow=False, show_blocks=False, show_raw=False
        )

    @pytest.mark.parametrize("name", CONVERTERS)
    def test_dispatch_all_flags(self, name, mocks, tmp_path):
        backend = _backend(name)
        path = tmp_path / "session.jsonl"

        result = Converter(name).inspect_session(
            path, show_flow=True, show_blocks=True, show_raw=True
        )

        assert result is None
        mocks[backend]["inspect"].assert_called_once_with(
            path, show_flow=True, show_blocks=True, show_raw=True
        )

    @pytest.mark.parametrize("name", CONVERTERS)
    def test_return_value_is_forwarded(self, name, mocks, tmp_path):
        # Even though the docstring promises None, the dispatch `return`s
        # whatever the backend returns.
        backend = _backend(name)
        mocks[backend]["inspect"].return_value = "backend-sentinel"

        result = Converter(name).inspect_session(tmp_path / "s.jsonl")

        assert result == "backend-sentinel"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"show_flow": True},
            {"show_blocks": True},
            {"show_raw": True},
            {"show_flow": True, "show_blocks": True},
        ],
    )
    def test_partial_flags(self, kwargs, mocks, tmp_path):
        path = str(tmp_path / "s.jsonl")
        full = {"show_flow": False, "show_blocks": False, "show_raw": False}
        full.update(kwargs)

        Converter("claude-code").inspect_session(path, **kwargs)

        mocks["claude"]["inspect"].assert_called_once_with(path, **full)

    def test_unknown_converter_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            Converter("unknown").inspect_session(tmp_path / "s.jsonl")
        assert "Unknown converter" in str(excinfo.value)
        assert "unknown" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Cross-cutting: dispatch isolation
# ---------------------------------------------------------------------------


class TestDispatchIsolation:
    def test_each_converter_only_calls_its_backend(self, mocks, tmp_path):
        path = str(tmp_path / "s.jsonl")
        records = []

        Converter("codex").load_session(path)
        Converter("codex").records_to_messages(records)
        Converter("codex").session_to_messages(path)
        Converter("codex").inspect_session(path)

        assert mocks["codex"]["load"].call_count == 1
        assert mocks["codex"]["records"].call_count == 1
        assert mocks["codex"]["session"].call_count == 1
        assert mocks["codex"]["inspect"].call_count == 1

        for backend in ("claude", "pi"):
            assert mocks[backend]["load"].call_count == 0
            assert mocks[backend]["records"].call_count == 0
            assert mocks[backend]["session"].call_count == 0
            assert mocks[backend]["inspect"].call_count == 0

    def test_repeated_calls_accumulate(self, mocks, tmp_path):
        c = Converter("pi")
        path = str(tmp_path / "s.jsonl")

        c.load_session(path)
        c.load_session(path)
        c.load_session(path)

        assert mocks["pi"]["load"].call_count == 3
