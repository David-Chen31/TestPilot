# TestPilot

面向 Python 代码仓库的单元测试生成 Agent：用 Agent 循环（读代码 → 写测试 → 运行 → 看报错 → 修改）生成能跑通的测试，再用变异测试检验并反馈“测试能不能抓 bug”。

> 开发中。当前进度：M1（项目骨架、Docker 环境、沙箱执行器）。

## 快速开始

```bash
docker compose up --build          # 启动 app + redis，http://localhost:8000/health
docker compose run --rm app pytest # 在容器内运行 TestPilot 自身的测试
```
