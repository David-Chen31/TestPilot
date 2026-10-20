import pytest

from claude_converter.converter import Converter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_converter_is_claude_code(self):
        conv = Converter()
        assert conv.converter == "claude-code"

    @pytest.mark.parametrize("name", ["claude-code", "codex", "pi"])
    def test_stores_converter_name(self, name):
        conv = Converter(name)
        assert conv.converter == name

    def test_accepts_arbitrary_string(self):
        # The type is only a Literal hint; runtime allows anything and
        # validates at dispatch time.
        conv = Converter("not-a-real-converter")
        assert conv.converter == "not-a-real-converter"


# ---------------------------------------------------------------------------
# load_session dispatch
# ---------------------------------------------------------------------------

class TestLoadSession:
    @pytest.mark.parametrize(
        "name,target",
        [
            ("claude-code", "load_session_claude"),
            ("codex", "load_session_codex"),
            ("pi", "load_session_pi"),
        ],
    )
    def test_dispatches_to_correct_function(self, monkeypatch, name, target):
        import claude_converter.converter as mod

        sentinel = [{"marker": target}]
        calls = []

        def fake(path):
            calls.append(path)
            return sentinel

        monkeypatch.setattr(mod, target, fake)

        conv = Converter(name)
        result = conv.load_session("/some/path.jsonl")

        assert result is sentinel
        assert calls == ["/some/path.jsonl"]

    def test_accepts_pathlib_path(self, monkeypatch, tmp_path):
        import claude_converter.converter as mod

        seen = {}

        def fake(path):
            seen["path"] = path
            return []

        monkeypatch.setattr(mod, "load_session_claude", fake)

        conv = Converter("claude-code")
        p = tmp_path / "s.jsonl"
        result = conv.load_session(p)

        assert result == []
        assert seen["path"] is p

    def test_unknown_converter_raises(self):
        conv = Converter("bogus")
        with pytest.raises(ValueError, match="Unknown converter: 'bogus'"):
            conv.load_session("whatever.jsonl")


# ---------------------------------------------------------------------------
# records_to_messages dispatch
# ---------------------------------------------------------------------------

class TestRecordsToMessages:
    @pytest.mark.parametrize(
        "name,target",
        [
            ("claude-code", "records_to_messages_claude"),
            ("codex", "records_to_messages_codex"),
            ("pi", "records_to_messages_pi"),
        ],
    )
    def test_dispatches_to_correct_function(self, monkeypatch, name, target):
        import claude_converter.converter as mod

        sentinel = [{"role": "user", "content": "hi"}]
        records_in = [{"type": "user"}]
        calls = []

        def fake(records):
            calls.append(records)
            return sentinel

        monkeypatch.setattr(mod, target, fake)

        conv = Converter(name)
        result = conv.records_to_messages(records_in)

        assert result is sentinel
        assert calls == [records_in]

    def test_unknown_converter_raises(self):
        conv = Converter("bogus")
        with pytest.raises(ValueError, match="Unknown converter: 'bogus'"):
            conv.records_to_messages([])


# ---------------------------------------------------------------------------
# session_to_messages dispatch
# ---------------------------------------------------------------------------

class TestSessionToMessages:
    @pytest.mark.parametrize(
        "name,target",
        [
            ("claude-code", "session_to_messages_claude"),
            ("codex", "session_to_messages_codex"),
            ("pi", "session_to_messages_pi"),
        ],
    )
    def test_dispatches_with_output_none(self, monkeypatch, name, target):
        import claude_converter.converter as mod

        sentinel = [{"role": "user", "content": "hi"}]
        calls = []

        def fake(path, output=None):
            calls.append((path, output))
            return sentinel

        monkeypatch.setattr(mod, target, fake)

        conv = Converter(name)
        result = conv.session_to_messages("/p.jsonl")

        assert result is sentinel
        assert calls == [("/p.jsonl", None)]

    @pytest.mark.parametrize(
        "name,target",
        [
            ("claude-code", "session_to_messages_claude"),
            ("codex", "session_to_messages_codex"),
            ("pi", "session_to_messages_pi"),
        ],
    )
    def test_dispatches_with_output_path(self, monkeypatch, name, target):
        import claude_converter.converter as mod

        sentinel = []
        calls = []

        def fake(path, output=None):
            calls.append((path, output))
            return sentinel

        monkeypatch.setattr(mod, target, fake)

        conv = Converter(name)
        result = conv.session_to_messages("/p.jsonl", output="/out.json")

        assert result is sentinel
        assert calls == [("/p.jsonl", "/out.json")]

    def test_unknown_converter_raises(self):
        conv = Converter("bogus")
        with pytest.raises(ValueError, match="Unknown converter: 'bogus'"):
            conv.session_to_messages("/p.jsonl")


# ---------------------------------------------------------------------------
# inspect_session dispatch
# ---------------------------------------------------------------------------

class TestInspectSession:
    @pytest.mark.parametrize(
        "name,target",
        [
            ("claude-code", "inspect_session_claude"),
            ("codex", "inspect_session_codex"),
            ("pi", "inspect_session_pi"),
        ],
    )
    def test_dispatches_default_flags(self, monkeypatch, name, target):
        import claude_converter.converter as mod

        calls = []

        def fake(path, show_flow=False, show_blocks=False, show_raw=False):
            calls.append(
                {
                    "path": path,
                    "show_flow": show_flow,
                    "show_blocks": show_blocks,
                    "show_raw": show_raw,
                }
            )
            return None

        monkeypatch.setattr(mod, target, fake)

        conv = Converter(name)
        result = conv.inspect_session("/p.jsonl")

        assert result is None
        assert calls == [
            {
                "path": "/p.jsonl",
                "show_flow": False,
                "show_blocks": False,
                "show_raw": False,
            }
        ]

    @pytest.mark.parametrize(
        "name,target",
        [
            ("claude-code", "inspect_session_claude"),
            ("codex", "inspect_session_codex"),
            ("pi", "inspect_session_pi"),
        ],
    )
    def test_dispatches_all_flags(self, monkeypatch, name, target):
        import claude_converter.converter as mod

        calls = []

        def fake(path, show_flow=False, show_blocks=False, show_raw=False):
            calls.append(
                {
                    "path": path,
                    "show_flow": show_flow,
                    "show_blocks": show_blocks,
                    "show_raw": show_raw,
                }
            )

        monkeypatch.setattr(mod, target, fake)

        conv = Converter(name)
        conv.inspect_session(
            "/p.jsonl", show_flow=True, show_blocks=True, show_raw=True
        )

        assert calls == [
            {
                "path": "/p.jsonl",
                "show_flow": True,
                "show_blocks": True,
                "show_raw": True,
            }
        ]

    def test_unknown_converter_raises(self):
        conv = Converter("bogus")
        with pytest.raises(ValueError, match="Unknown converter: 'bogus'"):
            conv.inspect_session("/p.jsonl")


# ---------------------------------------------------------------------------
# The dispatcher should not invoke the other tools' functions.
# ---------------------------------------------------------------------------

class TestExclusiveDispatch:
    def test_only_selected_converter_called(self, monkeypatch):
        import claude_converter.converter as mod

        called = []

        def make(name):
            def fake(*args, **kwargs):
                called.append(name)
                return []

            return fake

        for target in (
            "load_session_claude",
            "load_session_codex",
            "load_session_pi",
        ):
            monkeypatch.setattr(mod, target, make(target))

        conv = Converter("codex")
        conv.load_session("/p.jsonl")

        assert called == ["load_session_codex"]
