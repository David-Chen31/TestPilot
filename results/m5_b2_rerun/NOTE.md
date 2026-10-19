# m5_b2_rerun 说明

- itertoolz run1、run2 有效。
- itertoolz run3 与 functoolz 的 3 次运行因 `OpenAIConnectionError` 失败（0 token）：14:36 UTC 起 Docker Desktop
  的 Windows 侧后端停止响应，容器 DNS 查询超时。第一次（m5_b2）的同类故障与 Windows 进入新型待机的时间吻合，
  这一次没有待机事件，确切原因未查明。
- 应对：eval.sh 在评测期间阻止系统待机（scripts/keep_awake.ps1）；连续 2 次连接失败时中止整批，避免在重试中空耗时间。
- itertoolz 与 functoolz 各 3 次的完整结果见 `../m5_b2_c/`。这里 itertoolz 的 2 次有效运行作为补充数据保留，不并入主表。

## 更正（2026-10-20）：根因是 WSL 崩溃转储，不是待机

上面“与待机吻合”的判断是错的。之后定位到真正的原因：mutmut 用 CPU 时间上限杀超时的变异体（SIGXCPU），
进程 core dump，Docker Desktop 的 WSL 把每个 dump 写到 Windows 的 `%TEMP%\wsl-crashes`（单个 0.1–1.9 GB）。
写 dump 时 Docker 的 WSL 虚拟机卡住，容器网络中断；dump 还写满了 C 盘。证据：
- 两次故障分别从 13:11 和 14:37（UTC）开始，恰好都是 B2 开始跑 itertoolz（大量变异体导致无限迭代、超时）的时刻；
- `wsl-crashes` 中的 10 个 dump 全部是 `python3.11` 被信号 24（SIGXCPU）终止产生的；
- 复现：在容器里让一个 Python 进程触发 SIGXCPU，Docker 立刻卡死，必须重启。
修复：镜像中所有 Python 进程启动时设置 PR_SET_DUMPABLE=0（`docker/no_core_dump.pth`），验证后不再产生 dump、Docker 不再卡死。
`ulimit -c 0` 无效（WSL 用管道收集 core，内核不检查 RLIMIT_CORE），实测仍产生了 dump。
