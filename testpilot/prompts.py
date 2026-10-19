"""给 LLM 的提示词。

- 各方法共享同一份测试要求和上下文：B0 的提示词就是 Agent 第一次写测试的提示词，
  B1 / B2 与 B0 的差异只来自之后的反馈循环，不来自措辞或上下文。
- 每次请求的前缀（系统提示、源码、上下文、要求）在迭代之间完全相同，只有末尾的
  “当前测试 + 反馈 + 任务”会变：DeepSeek 按前缀缓存，重复部分按缓存命中价计费（约 1/50）。
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

PREFIX = """\
Module under test: `{module}` (file `{target}`).

Source of `{target}` (line numbers added for reference; they are not part of the code):
```
{numbered_source}
```

{context}

Public API to cover: {public_symbols}

{requirements}"""

TASK_FIRST = """\
## Task
Write the pytest test file for `{module}`. Return the complete test file in one ```python code block."""

TASK_REVISE = """\
## Current test file (`{test_file}`)
```python
{test_code}
```

## Feedback
{feedback}

## Task
Revise the test file according to the feedback. The implementation is the source of truth: when a test \
fails, fix the test so that it asserts what the code actually does — never assume the code is wrong. \
Keep the tests that already pass. Return the complete revised test file in one ```python code block."""

FEEDBACK_NO_CODE = "Your previous response did not contain a ```python code block, so no tests were written."

FEEDBACK_PYTEST_HEADER = (
    "Running pytest on the current implementation: {passed} passed, {failed} failed, {errors} errors."
)
