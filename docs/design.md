# TestPilot 设计决策

每个关键决策记录：做了什么、为什么、考虑过的备选方案。随里程碑推进补充。

## 0. 范围与里程碑（2026-10-19 调整）

### 0.1 为什么调整

目标岗位是大模型算法岗。原计划偏 Agent 工程（服务化、缓存、部署），缺少“提出假设 → 实验验证”的部分，也没有碰奖励 / 验证器。没有 GPU，所以不做大规模训练，把重心放到**奖励信号与验证器的设计和评估**上，这些只用 API 就能完成。

### 0.2 里程碑

| # | 内容 | 状态 |
|---|---|---|
| M1 | 项目骨架、Docker、沙箱执行器 | 完成 |
| M2 | MCP server 与全部工具 | 完成 |
| M3 | 基线 B0：一次性生成 | |
| M4 | LangGraph Agent（pytest 报错反馈）= B1 | |
| M5 | 变异测试反馈 + 有效性规则 = B2 | |
| M6 | 检索：AST 依赖 vs AST + Faiss | |
| M7 | 服务化**最小可用版**：LLM 重试 / 降级（实验也依赖）；FastAPI + SSE + Redis 只做最小实现 | |
| M8 | 主评测 B0 / B1 / B2 / 消融，扩大到约 100+ 个目标函数，报告置信区间与显著性检验 | |
| M9 | 研究实验（见 0.3） | |
| M10 | （可选）Kaggle 免费 GPU 上的 LoRA 拒绝采样微调 | |
| M11 | README、设计文档、面试讲解、简历条目 | |

### 0.3 研究实验（M9）

1. **奖励欺骗分析**：去掉“原代码上必须通过”的约束，变异得分虚高多少？虚高来自哪些测试（原代码上失败 / flaky / 断言了错误行为）？对应作者此前 SWE-bench 复现测试审计的发现。
2. **测试时扩展**：每个目标采样 N ∈ {1, 2, 4, 8} 份测试，分别按 随机 / LLM 评委打分 / 变异得分 选择，画出效果随 N 变化的曲线。
3. **代理指标可靠性**：以变异得分为参照，覆盖率、LLM 评委打分与之的相关性。回答“哪个信号适合当奖励”。

### 0.4 模型选择

- 默认 `deepseek-flash`（最便宜的 DeepSeek 模型，2026-10 时由 DeepSeek-V4.1-Flash-1017 提供服务），通过 `.env` 可切换。
- **可复现性风险**：该别名无法锁定到日期版本，DeepSeek 升级后台模型会改变结果。应对：每次运行在日志中记录调用日期和响应返回的模型信息；同一组对比实验在尽量短的时间窗口内跑完。
- 正式评测前根据 M3 实测 token 用量估算总费用，确认后再大规模运行。

## 1. 沙箱（M1）

代码：`testpilot/sandbox/`

### 1.1 每个任务一份独立工作副本（`workspace.py`）

- **做法**：把仓库 `copytree` 到 `tempfile.mkdtemp()` 下，生成的测试、变异体都只写进副本，任务结束删除。
- **为什么**：Agent 会写测试文件，变异测试会改源码。直接在原仓库上操作，一次中断就可能留下被改坏的源码；多个任务并发时还会互相覆盖。
- **不复制** `.git`、`__pycache__`、虚拟环境、旧覆盖率数据：既慢，又可能让旧 `.pyc` / `.coverage` 干扰结果。commit hash 从原仓库读取，不需要副本里有 `.git`。
- **`symlinks=True`**：符号链接按链接复制，不跟随到仓库外，避免把仓库外的文件拷进来。
- **`resolve()` 路径检查**：所有读写都先解析到副本内，`../`、绝对路径、链接逃逸一律拒绝。之后 LLM 可调用的 `read_file` 等工具都经过这一层。
- **备选**：`git worktree`（要求是 git 仓库，且共享对象库）；overlay 文件系统 / 容器卷快照（更快，但依赖 Linux 特性，不便于本地开发）。仓库都是小型库，复制的成本可以忽略，选最简单的。

### 1.2 带超时、终止整棵进程树的子进程执行（`executor.py`）

- **问题**：`subprocess.run(timeout=...)` 超时后只 kill 直接子进程。如果被测代码起了孙进程，孙进程继续运行，并持有 stdout/stderr 管道，父进程读管道会一直阻塞，超时失效。
- **做法**：子进程用 `start_new_session=True` 自成会话（Windows 用 `CREATE_NEW_PROCESS_GROUP` + `taskkill /T`），超时时 `killpg(SIGKILL)` 整组终止。正常结束后也清理一次，防止测试留下的后台进程活过本次执行。
- **验证**：`tests/test_executor.py::test_timeout_kills_grandchild`。把终止进程树的逻辑替换成空操作后，这个测试在 Windows 和 Linux 容器里都会失败，说明它确实测到了这个行为。
- **已知漏洞**：孙进程如果自己 `setsid` 脱离进程组，`killpg` 杀不到。真正的隔离要靠容器 / cgroup（见 1.5）。

### 1.3 输出截断保留首尾

喂给 LLM 的报错按字符数截断，保留开头和结尾、丢中间：开头是命令 / 收集阶段的错误，结尾是最内层 traceback 和 pytest 汇总，中间多是重复的栈帧。截断处会注明省略了多少字符，避免 LLM 误以为输出是完整的。

### 1.4 pytest 结果用 JUnit XML 解析（`pytest_runner.py`）

- 终端输出格式随 pytest 版本和插件变化，XML 是稳定的机器接口，可以拿到逐用例的 通过 / 失败 / 错误 / 跳过 和失败信息。后续“有效测试 = 原代码上连续 3 次通过”需要逐用例结果。
- `-o addopts=` 忽略被测仓库配置里的 `addopts`（例如 `--cov`、`--doctest-modules`），避免因为缺插件导致跑不起来；`-p no:cacheprovider` 不在副本里写缓存。
- **没收集到测试不算通过**（`all_passed` 要求至少一个用例）。否则一个空测试文件会被当成“全部通过”。

### 1.5 生产环境还缺什么（本项目不做，但要知道）

当前的隔离只有“独立目录 + 进程超时 + 运行在 Docker 容器里”。被测代码和 LLM 生成的测试本质上是不可信代码，生产环境还需要：

- **每次执行一个一次性容器**（或 gVisor / Firecracker 这类更强的隔离），而不是在服务进程所在的容器里起子进程；
- **网络隔离**：`--network none`，防止测试代码外连或攻击内网（包括 Redis）；
- **seccomp / AppArmor** 限制系统调用，**非 root 用户**运行，**只读根文件系统**；
- **cgroup 资源限制**：CPU、内存、进程数（防 fork 炸弹）、磁盘写入量；
- 密钥隔离：当前 `sandbox_env()` 按变量名（含 KEY / TOKEN / SECRET / PASSWORD，以及 `REDIS_URL`）过滤掉子进程的环境变量，这是黑名单，容易漏；生产上应该用白名单，只传 `PATH`、`PYTHONPATH` 等必要变量，而且执行环境里根本不存在密钥。

### 1.6 Docker 构建的镜像源

宿主机走代理时，构建容器访问不到 `deb.debian.org`。Dockerfile 提供可选构建参数 `APT_MIRROR` / `PIP_MIRROR`（默认官方源），在 `.env` 里设置即可，不把某个国内镜像写死。

## 2. MCP 工具（M2）

代码：`testpilot/tools/`、`testpilot/sandbox/{coverage_runner,mutation,_mutmut_export}.py`、`testpilot/retrieval/ast_index.py`

### 2.1 每个任务一个 MCP server 进程（stdio），绑定一个工作副本

- **做法**：`python -m testpilot.tools.server --root <副本>`，Agent 端通过 stdio 连接。
- **为什么**：工具天然只能看到这个任务的副本，任务之间隔离；工具参数里不需要传仓库路径，LLM 也就没机会传错或越权。
- **备选**：一个常驻的 HTTP（streamable-http）server 服务所有任务，每次调用带 `job_id`。省掉进程启动开销，但要自己做任务到目录的映射和权限检查。任务是分钟级的，进程启动的几百毫秒可以忽略。
- **逻辑与协议分离**：工具逻辑写在普通类 `RepoTools` 里，`server.py` 只负责注册。单元测试直接测 `RepoTools`，另有一个端到端测试真的启动 server 子进程，走 MCP 协议调用工具。

### 2.2 工具集

| 工具 | 作用 | 输出控制 |
|---|---|---|
| `read_file` | 带行号读文件（可指定行范围） | 每次最多 400 行 |
| `list_symbols` | AST 列出函数 / 类 / 方法、行号、签名、调用的名字 | — |
| `search_code` | 字面量 / 正则搜索 .py 文件，找用法 | 最多 50 条 |
| `write_test_file` | 写生成的测试 | **只允许 `tests_testpilot/test_*.py`** |
| `run_pytest` | 运行测试，逐用例结果 | 只列出未通过的用例，信息首尾截断 |
| `coverage_report` | 目标文件的行 / 分支覆盖率、未覆盖行和分支 | — |
| `run_mutation` | 变异得分与存活变异体（行号、原代码、变异后代码） | 最多 20 个存活变异体 |

- **`write_test_file` 是规格外新增的**：让 Agent 的所有副作用都经过工具。只允许写进生成目录，因为 Agent 一旦能改源码，“测试在原代码上通过”就失去意义了。检查的是**解析后的真实路径**：`tests_testpilot/../sample/test_x.py` 字符串前缀合法，但实际会写进源码目录。单元测试抓到过这个 bug。
- 工具描述（docstring）用英文写，因为它是给 LLM 看的；代码注释用中文。

### 2.3 覆盖率

`coverage run --branch --include=<目标文件> -m pytest ...`，再用 `coverage json` 读取。行覆盖率 = 已执行语句 / 语句数，分支覆盖率 = 已走分支 / 分支数。覆盖率本身不区分测试是否通过，所以只对有效测试统计（由 M5 的有效性规则保证）。

### 2.4 变异测试（mutmut 3.8）

- **集成方式**：在副本的 `pyproject.toml` 末尾追加 `[tool.mutmut]`。`source_paths` 是目标所在的顶层包（mutmut 要把整个包复制到 `mutants/`，才能 import），`only_mutate` 把变异范围限制在目标文件，`also_copy` 声明生成的测试目录（mutmut 默认只复制 `tests/`）。每次运行前删除 `mutants/`，避免复用旧缓存。
- **结果导出**：mutmut 没有稳定的结构化输出，所以在副本目录里起一个子进程调用它的内部 API（`_mutmut_export.py`）。因此 mutmut 版本锁死为 3.8.0。
- **行号换算**：mutmut 的 diff 行号相对于函数开头。先用 AST 找到函数在原文件中的起始行（mutmut 把方法名改写成 `xǁ类名ǁ方法名`），再加上偏移；如果偏移不准（例如有装饰器），就在函数范围内按内容查找。单测验证了每个变异体的行号都能对上原文件内容。
- **得分口径**：`score = killed / (total − timeout − 无法编译)`。
  - `no tests`（没有任何测试执行到该变异体）算存活，不排除，否则不调用某个函数反而不扣分；
  - mutmut 用 libcst 生成变异体，语法总是合法的，所以“无法编译”恒为 0；
  - 其他非常规状态（suspicious、segfault 等）也排除出分母、单独计数，**不算作杀死**，避免得分虚高。
- **等价变异体**：样例 `clamp` 中，`if x < low: return low` 变成 `if x <= low` 后行为完全相同（`x == low` 时返回 `low` 等于返回 `x`），任何测试都杀不死。因此变异得分的上限低于 100%。单测 `test_strong_tests_kill_more` 明确断言了这两个存活者。评测时比较的是**同一目标上不同方法的相对得分**，等价变异体对各方法的影响相同。
- **规则**：变异测试的前提是测试在原代码上通过。mutmut 自己也会先在原代码上跑一遍，失败就拒绝继续，工具会把这种情况报成 error。M5 再加上“连续 3 次通过”的防 flaky 规则。
- **平台**：mutmut 依赖 `fork`，原生 Windows 上直接返回明确的错误；`pyproject.toml` 里按平台条件安装。

### 2.5 示例：覆盖率高 ≠ 能抓 bug

对样例 `sample/calc.py`，只断言 `clamp(5, 0, 10) is not None` 的弱测试：行覆盖率 70%，变异得分 1/7。补上边界值和错误信息断言后：行 / 分支覆盖率 100%，变异得分 5/7，剩下两个是等价变异体。见 `tests/test_repo_tools.py` 与 `tests/test_mutation.py`。
