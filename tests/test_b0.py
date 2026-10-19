from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from testpilot.eval import b0
from testpilot.sandbox import Workspace


@pytest.mark.parametrize(
    "target,paths,expected",
    [
        ("toolz/dicttoolz.py", (".",), "toolz.dicttoolz"),
        ("src/pkg/mod.py", ("src",), "pkg.mod"),
        ("pkg/__init__.py", (".",), "pkg"),
    ],
)
def test_module_name(target, paths, expected):
    assert b0.module_name(target, paths) == expected


def test_module_name_outside_paths():
    with pytest.raises(ValueError):
        b0.module_name("other/mod.py", ("src",))


def test_extract_code_picks_longest_block():
    text = "Here:\n```python\nx = 1\n```\nand the file:\n```python\nimport pytest\n\ndef test_a():\n    pass\n```\n"
    assert b0.extract_code(text) == "import pytest\n\ndef test_a():\n    pass\n"


def test_extract_code_without_fence():
    assert b0.extract_code("def test_a():\n    pass") == "def test_a():\n    pass\n"


def test_prompt_contains_source_and_requirements():
    messages = b0.build_messages("toolz/dicttoolz.py", "def merge(): ...", "toolz.dicttoolz")
    user = messages[1].content
    assert "def merge(): ..." in user
    assert "from toolz.dicttoolz import" in user
    assert "CURRENT implementation" in user


def test_generate_writes_test_file(sample_repo: Path, tmp_path: Path):
    reply = "```python\nfrom sample.calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n```"
    model = FakeListChatModel(responses=[reply])
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        res = b0.generate(model, ws.path, "sample/calc.py")
        assert res.test_files == ["tests_testpilot/test_calc.py"]
        assert ws.read_text("tests_testpilot/test_calc.py").startswith("from sample.calc import add")
    assert res.usage.calls == 1
    assert res.trace[0]["node"] == "generate"
    assert "def clamp" in res.trace[0]["messages"][1]["content"]
