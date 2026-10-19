# TestPilot 设计决策

每个关键决策记录：做了什么、为什么、考虑过的备选方案。随里程碑推进补充。

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
