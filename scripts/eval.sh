#!/usr/bin/env sh
# 在 Docker 中运行评测（变异测试需要 Linux），记录当前代码版本。
# 用法：scripts/eval.sh --method b0 --repo toolz --runs 3 --tag m3_b0
set -e
cd "$(dirname "$0")/.."
docker compose run --rm -e TESTPILOT_COMMIT="$(git describe --always --dirty)" app python -m testpilot.eval.run "$@"
