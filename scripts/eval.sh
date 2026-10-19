#!/usr/bin/env sh
# 在 Docker 中运行评测（变异测试需要 Linux），记录当前代码版本。
# 用法：scripts/eval.sh --method b0 --repo toolz --runs 3 --tag m3_b0
#
# Windows 上评测期间阻止系统进入睡眠 / 新型待机：待机会切断 Docker Desktop 的容器网络，
# 正在跑的批次会全部以连接错误失败（M5 中真实发生过）。只在本次评测期间生效。
set -e
cd "$(dirname "$0")/.."

KEEP_AWAKE_PID=""
if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/keep_awake.ps1 &
    KEEP_AWAKE_PID=$!
    trap 'kill $KEEP_AWAKE_PID 2>/dev/null' EXIT INT TERM
fi

docker compose run --rm -e TESTPILOT_COMMIT="$(git describe --always --dirty)" app python -m testpilot.eval.run "$@"
