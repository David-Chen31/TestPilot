# m5_b2 说明

- dicttoolz 的 3 次运行有效。
- itertoolz、functoolz 的 6 次运行全部因 `OpenAIConnectionError` 失败（0 token，与方法无关）：
  13:11 起 Docker Desktop 的容器网络失效，随后 Docker 守护进程无响应，重启 Docker Desktop 后恢复。
  这 6 次运行在 `../m5_b2_rerun/` 用同一版代码重跑。
