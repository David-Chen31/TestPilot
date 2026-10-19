# m5_b2_rerun 说明

- itertoolz run1、run2 有效。
- itertoolz run3 与 functoolz 的 3 次运行因 `OpenAIConnectionError` 失败（0 token）：14:36 UTC 起 Docker Desktop
  的 Windows 侧后端停止响应，容器 DNS 查询超时。第一次（m5_b2）的同类故障与 Windows 进入新型待机的时间吻合，
  这一次没有待机事件，确切原因未查明。
- 应对：eval.sh 在评测期间阻止系统待机（scripts/keep_awake.ps1）；连续 2 次连接失败时中止整批，避免在重试中空耗时间。
- itertoolz 与 functoolz 各 3 次的完整结果见 `../m5_b2_c/`。这里 itertoolz 的 2 次有效运行作为补充数据保留，不并入主表。
