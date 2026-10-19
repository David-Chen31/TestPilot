# m5_b2_rerun

均值 [最小, 最大]；费用按高峰价估算（上限）。

| 设置 | 运行次数 | 有效测试 | 生成测试 | 有效率 | 行覆盖 | 分支覆盖 | 变异得分 | 生成耗时(s) | tokens | 费用上限($) |
|---|---|---|---|---|---|---|---|---|---|---|
| b2 / toolz / toolz/itertoolz.py | 2/3 | 137 [133, 141] | 137 [133, 141] | 100.0% [100.0%, 100.0%] | 97.9% [97.5%, 98.3%] | 97.4% [97.1%, 97.6%] | 93.0% [92.7%, 93.3%] | 206.6 [195.7, 217.5] | 122663 [122451, 122875] | 0.0465 [0.0462, 0.0468] |
| b2 / toolz / toolz/functoolz.py | 0/3 | — | — | — | — | — | — | — | — | — |

## 原始日志

- b2 / toolz / toolz/itertoolz.py run1: `results/m5_b2_rerun/b2/toolz/itertoolz/run1`
- b2 / toolz / toolz/itertoolz.py run2: `results/m5_b2_rerun/b2/toolz/itertoolz/run2`
- b2 / toolz / toolz/itertoolz.py run3: `results/m5_b2_rerun/b2/toolz/itertoolz/run3` — 失败: OpenAIConnectionError: Connection error.
- b2 / toolz / toolz/functoolz.py run1: `results/m5_b2_rerun/b2/toolz/functoolz/run1` — 失败: OpenAIConnectionError: Connection error.
- b2 / toolz / toolz/functoolz.py run2: `results/m5_b2_rerun/b2/toolz/functoolz/run2` — 失败: OpenAIConnectionError: Connection error.
- b2 / toolz / toolz/functoolz.py run3: `results/m5_b2_rerun/b2/toolz/functoolz/run3` — 失败: OpenAIConnectionError: Connection error.
