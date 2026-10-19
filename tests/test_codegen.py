import pytest

from testpilot.agent import codegen


@pytest.mark.parametrize(
    "target,paths,expected",
    [
        ("toolz/dicttoolz.py", (".",), "toolz.dicttoolz"),
        ("src/pkg/mod.py", ("src",), "pkg.mod"),
        ("pkg/__init__.py", (".",), "pkg"),
    ],
)
def test_module_name(target, paths, expected):
    assert codegen.module_name(target, paths) == expected


def test_module_name_outside_paths():
    with pytest.raises(ValueError):
        codegen.module_name("other/mod.py", ("src",))


def test_generated_test_path():
    assert codegen.generated_test_path("toolz/functoolz.py") == "tests_testpilot/test_functoolz.py"


def test_extract_code_picks_longest_block():
    text = "Here:\n```python\nx = 1\n```\nand the file:\n```python\nimport pytest\n\ndef test_a():\n    pass\n```\n"
    assert codegen.extract_code(text) == "import pytest\n\ndef test_a():\n    pass\n"


def test_extract_code_without_fence_is_empty():
    # 没有代码块就是生成失败，不把解释性文字当代码写进测试文件
    assert codegen.extract_code("def test_a():\n    pass") == ""
