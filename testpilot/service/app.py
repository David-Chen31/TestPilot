"""HTTP 服务入口。M1 只有健康检查；任务接口与 SSE 在 M7 实现。"""

from fastapi import FastAPI

app = FastAPI(title="TestPilot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
