from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_converter.converter import Converter, ConverterName


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

ALL_CONVERTERS = ["claude-code", "codex", "pi"]


@pytest.fixture
def patched_functions():
    """Patch every per-tool function imported into claude_converter.converter.

    Yields a dict keyed by attribute name in the converter module -> MagicMock.
    Each mock returns a unique, identifiable value so we can assert which
    dispatcher branch was exercised.
    """
    names = [
        "load_session_claude",
        "load_session_codex",
        "load_session_pi",
        "records_to_messages_claude",
        "records_to_messages_codex",
        "records_to_messages_pi",
        "session_to_messages_claude",
        "session_to_messages_codex",
        "session_to_messages_pi",
        "inspect_session_claude",
        "inspect_session_codex",
        "inspect_session_pi",
    ]
    patches = {}
    mocks = {}
    for name in names:
        p = patch(f"claude_converter.converter.{name}")
        patches[name] = p
        mocks[name] = p.start()
    # Give each a distinct return value to distinguish them.
    for name, mock in mocks.items():
        mock.return_value = f"result::{name}"
    try:
        yield mocks
    finally:
        for p in patches.values():
            p.stop()


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_converter_is_claude_code(self):
        c = Converter()
        assert c.converter == "claude-code"

    @pytest.mark.parametrize("name", ALL_CONVERTERS)
    def test_explicit_converter_stored(self, name):
        c = Converter(name)
        assert c.converter == name

    def test_unknown_converter_name_stored_without_validation(self):
        # __init__ does not validate; it just stores the value.
        c = Converter("unknown")  # type: ignore[arg-type]
        assert c.converter == "unknown"


# ---------------------------------------------------------------------------
# load_session dispatch
# ---------------------------------------------------------------------------

class TestLoadSession:
    @pytest.mark.parametrize(
        "converter,expected_attr",
        [
            ("claude-code", "load_session_claude"),
            ("codex", "load_session_codex"),
            ("pi", "load_session_pi"),
        ],
    )
    def test_dispatches_to_correct_function(self, patched_functions, converter, expected_attr):
        c = Converter(converter)  # type: ignore[arg-type]
        result = c.load_session("/some/path.jsonl")

        expected_mock = patched_functions[expected_attr]
        expected_mock.assert_called_once_with("/some/path.jsonl")
        # No other load_session_* should have been called.
        for name, mock in patched_functions.items():
            if name.startswith("load_session_") and name != expected_attr:
                mock.assert_not_called()
        assert result == f"result::{expected_attr}"

    def test_accepts_pathlib_path(self, patched_functions):
        c = Converter("claude-code")
        p = Path("/tmp/session.jsonl")
        c.load_session(p)
        patched_functions["load_session_claude"].assert_called_once_with(p)

    def test_unknown_converter_raises(self):
        c = Converter("bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown converter"):
            c.load_session("/x.jsonl")

    def test_unknown_converter_error_mentions_value(self):
        c = Converter("bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError) as exc_info:
            c.load_session("/x.jsonl")
        assert "'bogus'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# records_to_messages dispatch
# ---------------------------------------------------------------------------

class TestRecordsToMessages:
    @pytest.mark.parametrize(
        "converter,expected_attr",
        [
            ("claude-code", "records_to_messages_claude"),
            ("codex", "records_to_messages_codex"),
            ("pi", "records_to_messages_pi"),
        ],
    )
    def test_dispatches_to_correct_function(self, patched_functions, converter, expected_attr):
        c = Converter(converter)  # type: ignore[arg-type]
        records = [{"type": "user"}, {"type": "assistant"}]

        result = c.records_to_messages(records)

        expected_mock = patched_functions[expected_attr]
        expected_mock.assert_called_once_with(records)
        for name, mock in patched_functions.items():
            if name.startswith("records_to_messages_") and name != expected_attr:
                mock.assert_not_called()
        assert result == f"result::{expected_attr}"

    def test_passes_empty_list_through(self, patched_functions):
        c = Converter("codex")
        records: list[dict] = []
        c.records_to_messages(records)
        patched_functions["records_to_messages_codex"].assert_called_once_with(records)

    def test_unknown_converter_raises(self):
        c = Converter("nope")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown converter"):
            c.records_to_messages([])


# ---------------------------------------------------------------------------
# session_to_messages dispatch
# ---------------------------------------------------------------------------

class TestSessionToMessages:
    @pytest.mark.parametrize(
        "converter,expected_attr",
        [
            ("claude-code", "session_to_messages_claude"),
            ("codex", "session_to_messages_codex"),
            ("pi", "session_to_messages_pi"),
        ],
    )
    def test_dispatches_with_default_output_none(
        self, patched_functions, converter, expected_attr
    ):
        c = Converter(converter)  # type: ignore[arg-type]
        result = c.session_to_messages("/in.jsonl")

        expected_mock = patched_functions[expected_attr]
        expected_mock.assert_called_once_with("/in.jsonl", output=None)
        for name, mock in patched_functions.items():
            if name.startswith("session_to_messages_") and name != expected_attr:
                mock.assert_not_called()
        assert result == f"result::{expected_attr}"

    @pytest.mark.parametrize(
        "converter,expected_attr",
        [
            ("claude-code", "session_to_messages_claude"),
            ("codex", "session_to_messages_codex"),
            ("pi", "session_to_messages_pi"),
        ],
    )
    def test_dispatches_with_output_provided(
        self, patched_functions, converter, expected_attr
    ):
        c = Converter(converter)  # type: ignore[arg-type]
        result = c.session_to_messages("/in.jsonl", output="/out.json")

        expected_mock = patched_functions[expected_attr]
        expected_mock.assert_called_once_with("/in.jsonl", output="/out.json")
        assert result == f"result::{expected_attr}"

    def test_output_pathlib_passed_through(self, patched_functions):
        c = Converter("codex")
        out = Path("/tmp/out.json")
        c.session_to_messages(Path("/in.jsonl"), output=out)
        patched_functions["session_to_messages_codex"].assert_called_once_with(
            Path("/in.jsonl"), output=out
        )

    def test_unknown_converter_raises(self):
        c = Converter("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown converter"):
            c.session_to_messages("/in.jsonl")


# ---------------------------------------------------------------------------
# inspect_session dispatch
# ---------------------------------------------------------------------------

class TestInspectSession:
    @pytest.mark.parametrize(
        "converter,expected_attr",
        [
            ("claude-code", "inspect_session_claude"),
            ("codex", "inspect_session_codex"),
            ("pi", "inspect_session_pi"),
        ],
    )
    def test_dispatches_with_default_flags(
        self, patched_functions, converter, expected_attr
    ):
        c = Converter(converter)  # type: ignore[arg-type]
        result = c.inspect_session("/sess.jsonl")

        expected_mock = patched_functions[expected_attr]
        expected_mock.assert_called_once_with(
            "/sess.jsonl", show_flow=False, show_blocks=False, show_raw=False
        )
        for name, mock in patched_functions.items():
            if name.startswith("inspect_session_") and name != expected_attr:
                mock.assert_not_called()
        assert result == f"result::{expected_attr}"

    @pytest.mark.parametrize(
        "converter,expected_attr",
        [
            ("claude-code", "inspect_session_claude"),
            ("codex", "inspect_session_codex"),
            ("pi", "inspect_session_pi"),
        ],
    )
    def test_dispatches_with_all_flags_true(
        self, patched_functions, converter, expected_attr
    ):
        c = Converter(converter)  # type: ignore[arg-type]
        c.inspect_session("/sess.jsonl", show_flow=True, show_blocks=True, show_raw=True)

        patched_functions[expected_attr].assert_called_once_with(
            "/sess.jsonl", show_flow=True, show_blocks=True, show_raw=True
        )

    def test_mixed_flags_passed_through(self, patched_functions):
        c = Converter("codex")
        c.inspect_session("/s.jsonl", show_flow=True, show_blocks=False, show_raw=True)
        patched_functions["inspect_session_codex"].assert_called_once_with(
            "/s.jsonl", show_flow=True, show_blocks=False, show_raw=True
        )

    def test_returns_none(self, patched_functions):
        c = Converter("pi")
        assert c.inspect_session("/s.jsonl") is None

    def test_unknown_converter_raises(self):
        c = Converter("mystery")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown converter"):
            c.inspect_session("/s.jsonl")


# ---------------------------------------------------------------------------
# Type alias sanity
# ---------------------------------------------------------------------------

def test_converter_name_literal_options():
    # ConverterName is a typing.Literal; check its allowed values.
    assert set(ConverterName.__args__) == {"claude-code", "codex", "pi"}
