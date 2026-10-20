# TestPilot

面向 Python 代码仓库的单元测试生成 Agent，研究**用变异测试作为可验证的奖励信号**：它作为反馈、作为选择器、作为评价指标时各自有多大用，以及它会被怎样“钻空子”。

> 当前进度：M8 完成（11 个仓库、66 个目标的主评测）。设计决策与全部实验记录见 [docs/design.md](docs/design.md)。

## 做了什么

- **Agent**：LangGraph 显式状态机（plan → retrieve_context → write_tests → run_tests → analyze → decide），工具通过 MCP server 提供（读代码、写测试、运行 pytest、覆盖率、变异测试）。拓扑固定、终止条件显式，对比实验里只改变反馈信号。
- **方法**（同一张图的不同配置）：
  - B0：一次生成；B1：pytest 报错反馈；
  - **B2：变异测试反馈**（把存活变异体“第 N 行：原代码 → 变异代码”反馈给模型）；
  - BC：覆盖率反馈（对照：只换信号）；B0×N：等成本独立采样（对照：只多花 token）。
- **有效性规则**：测试必须在原代码上连续 3 次通过才计分，覆盖率和变异得分只在有效测试上算——否则在原代码上就失败的测试会让变异得分虚高。
- **评测**：固定 commit 的 11 个纯 Python 仓库，含 4 个晚于模型训练截止创建的仓库作为数据污染对照；按目标配对的 bootstrap 置信区间与置换检验（`testpilot/eval/stats.py`）。每个数字都来自 `results/` 下的原始日志。

## 主要结果（M8：11 个仓库、66 个目标，[docs/design.md](docs/design.md) §7）

| 方法 | 变异得分 | 与 B2 的配对差值 [95% CI] | 有效测试数 | tokens |
|---|---|---|---|---|
| B0 一次生成 | 69.2% | −15.1 [−21.7, −9.1] | 55 | 9.6K |
| B1 pytest 反馈 | 77.9% | −6.3 [−10.0, −3.3] | 61 | 35K |
| BC 覆盖率反馈 | 80.7% | −3.4 [−6.0, −1.1] | 69 | 61K |
| B0×8 按覆盖率挑一份 | 80.4% | −3.8 [−5.5, −2.2] | 68 | 78K |
| **B2 变异反馈** | **84.2%** | — | 76 | 70K |
| B0×8 有效测试并集 | 85.8% | +1.6 [−0.1, +3.2] | 445 | 78K |
| 仓库自带的人写测试（42 个目标） | 68.3% | B2 高 13.0 [7.5, 18.3] | — | — |

- 变异反馈显著优于 pytest 反馈和覆盖率反馈，也优于花更多 token 的“采样 8 次按覆盖率挑一份”；等成本的“8 次采样取并集”与它打平，但测试数是它的约 6 倍。
- 在训练截止后创建的仓库上，B1 比在老仓库上低 16 个点，B2 只低 4 个点：陌生代码上执行反馈更重要（两组难度不同，只是相关）。
- 覆盖率信号会饱和：覆盖率反馈常在行 / 分支覆盖 100% 时停下，此时仍有存活变异体。
- 奖励欺骗：B2 曾用 monkeypatch 替换被测模块内部函数来“杀死”等价变异体，提示词明确禁止也没挡住（§5.6）。
- 评测中发现并修正了一个会让结论偏向自己方法的 bug（§7.5）。

## 快速开始

```bash
docker compose up --build          # 启动 app + redis，http://localhost:8000/health
docker compose run --rm app pytest # 在容器内运行 TestPilot 自身的测试
```

## 运行评测

```bash
cp .env.example .env               # 填写 LLM_API_KEY（DeepSeek，OpenAI 兼容接口）

# 预检（不调用 LLM）：每个目标能否导入、mutmut 能否跑通；顺便得到仓库自带测试的参照得分
docker compose run --rm app python -m testpilot.eval.preflight --tag m8_preflight

# 主评测：多个方法在同一批里交错运行，4 路并发；原始日志写入 results/m8_main/
scripts/eval.sh --method b1 --method bc --method b2 --method b0n --runs 1 --concurrency 4 --tag m8_main

# 汇总：各方法与 B2 按目标配对比较（全部 / old 组 / new 组）
python -m testpilot.eval.compare results/m8_main --reference b2 --out m8_compare
```

变异测试（mutmut）需要 Linux，评测在 Docker 中运行。Windows + Docker Desktop 上的两个坑（WSL 崩溃转储、pagefile 膨胀写满 C 盘）及处理见 docs/design.md §5.7、§6.4。

## 单独调用 MCP 工具

```bash
# 演示全部工具（复制样例仓库到临时副本，依次调用；变异测试需要在容器内运行）
docker compose run --rm app python -m testpilot.tools.client --demo tests/fixtures/sample_repo

# 在某个目录上调用单个工具
python -m testpilot.tools.client --root <目录> list_symbols '{"path": "."}'

# 用 MCP Inspector 交互调试
npx @modelcontextprotocol/inspector python -m testpilot.tools.server --root <目录>
```

## 目录

```
testpilot/
  agent/      LangGraph 状态机（graph.py）、独立采样（sampling.py）、MCP 客户端
  tools/      MCP server 与工具实现
  sandbox/    工作副本、带超时的进程执行、pytest / coverage / mutmut 封装
  retrieval/  AST 符号与依赖上下文
  eval/       评测仓库清单、有效性与指标、预检、批量评测、统计与汇总比较
docs/         设计决策（design.md）、各里程碑面试题
results/      每次评测的原始日志（run.json、trace.json、evaluation.json、生成的测试）
```
