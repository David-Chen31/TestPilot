# 作废：m3_b0 第一批（2026-10-19）

评测器 bug：变异测试的 `also_copy` 传入了子目录中的单个测试文件，mutmut 复制时不会创建父目录，
导致 `FileNotFoundError`，3 次运行的变异测试全部没有执行（`mutants_total = 0`，`mutation_score = None`）。

- 生成的测试、有效性、覆盖率数据不受该 bug 影响，但为了让同一批结果用同一版评测器得出，整批重跑。
- 修复与回归测试：`tests/test_mutation.py::test_tests_given_as_nodeids`。
- 正式结果见 `results/m3_b0/`。
