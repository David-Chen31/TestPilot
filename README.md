# TestPilot

面向 Python 代码仓库的单元测试生成 Agent：用 Agent 循环（读代码 → 写测试 → 运行 → 看报错 → 修改）生成能跑通的测试，再用变异测试检验并反馈“测试能不能抓 bug”。

> 开发中。当前进度：M4（LangGraph Agent，B1）。

## 快速开始

```bash
docker compose up --build          # 启动 app + redis，http://localhost:8000/health
docker compose run --rm app pytest # 在容器内运行 TestPilot 自身的测试
```

## 单独调用 MCP 工具

```bash
# 演示全部工具（复制样例仓库到临时副本，依次调用；变异测试需要在容器内运行）
docker compose run --rm app python -m testpilot.tools.client --demo tests/fixtures/sample_repo

# 在某个目录上调用单个工具
python -m testpilot.tools.client --root <目录> list_symbols '{"path": "."}'

# 用 MCP Inspector 交互调试
npx @modelcontextprotocol/inspector python -m testpilot.tools.server --root <目录>
```

## 运行评测

```bash
cp .env.example .env               # 填写 LLM_API_KEY
scripts/eval.sh --method b0 --repo toolz --runs 3 --tag m3_b0   # 在 Docker 中运行，原始日志写入 results/m3_b0/
```
