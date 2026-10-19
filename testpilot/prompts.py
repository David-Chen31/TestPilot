"""给 LLM 的提示词。各方法共享同一份测试要求，方法之间的差异只来自流程（一次生成 / 迭代 / 变异反馈），
而不是措辞——否则 B0 和 B1 的对比就混进了 prompt 工程的影响。
"""

SYSTEM = "You are an expert Python developer who writes thorough, correct pytest unit tests."

TEST_REQUIREMENTS = """\
Requirements for the tests:
- Write a single, self-contained pytest test file.
- Import the code under test by its absolute module path (e.g. `from {module} import ...`).
- The tests must pass on the CURRENT implementation: assert the behavior the code actually has.
- Cover the public functions and classes, including edge cases, branches and error handling (`pytest.raises`).
- Use precise assertions on return values and side effects; avoid weak checks such as `is not None`.
- Do not access the network, do not write files outside pytest's `tmp_path`, and do not depend on randomness or the current time."""

B0_USER = """\
Write pytest unit tests for the Python module `{module}` (file `{target}`).

{requirements}

Return the complete test file in one ```python code block.

Source of `{target}`:
```python
{source}
```"""
