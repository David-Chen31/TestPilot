# 作废：m4_b0 第一批、m4_b1 部分运行（2026-10-19）

基础设施 bug：评测脚本每次运行单独调用 `asyncio.run()`，而 langchain-openai 在进程级缓存异步 HTTP 客户端。
后续运行复用了绑定在已关闭事件循环上的连接，间歇性抛出 `RuntimeError: Event loop is closed`。
m4_b0 9 次运行中有 4 次因此失败（与方法本身无关）；m4_b1 在发现问题后被手动中止（`../m4_b1_partial/`）。

修复：整批运行共用一个事件循环（`testpilot/eval/run.py::_run_batch`），整批重跑。
成功的那 5 次 B0 运行数据本身有效，但为了让同一批结果出自同一版代码，不与重跑结果混用。
