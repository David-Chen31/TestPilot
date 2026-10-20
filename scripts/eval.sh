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
    # 传入本脚本的 Windows 进程号：脚本被强制结束、trap 来不及执行时，keep_awake 也会自己退出
    PARENT_WINPID=$(cat /proc/$$/winpid 2>/dev/null || echo $$)
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/keep_awake.ps1 -ParentPid "$PARENT_WINPID" &
    KEEP_AWAKE_PID=$!
    trap 'kill $KEEP_AWAKE_PID $WATCHDOG_PID 2>/dev/null' EXIT INT TERM
fi

# Windows 上报告 C 盘剩余空间与 WSL 崩溃转储数量：容器内进程 core dump 会被 WSL 写到
# C 盘的 %TEMP%\wsl-crashes（M5 中写满过 C 盘）。镜像已禁止 Python 进程 core dump，这里再兜底检查。
CRASH_DIR=""
if command -v cygpath >/dev/null 2>&1 && [ -n "$TEMP" ]; then
    CRASH_DIR="$(cygpath -u "$TEMP")/wsl-crashes"
fi
disk_report() {
    [ -n "$CRASH_DIR" ] || return 0
    # pagefile.sys 是隐藏的系统文件，ls -l 能读到大小；读不到时显示 ?
    PAGEFILE=$(ls -l /c/pagefile.sys 2>/dev/null | awk '{printf "%.1fG", $5/2^30}')
    echo "[eval.sh] $1: C 盘剩余 $(df -h /c | awk 'NR==2{print $4}')，pagefile.sys ${PAGEFILE:-?}，WSL 崩溃转储 $(ls "$CRASH_DIR" 2>/dev/null | wc -l) 个"
}
DUMPS_BEFORE=$( [ -n "$CRASH_DIR" ] && ls "$CRASH_DIR" 2>/dev/null | wc -l || echo 0 )
disk_report "开始"

# 看门狗：评测期间 C 盘剩余低于 MIN_FREE_GB 时停掉评测容器。容器内存上限管不住 WSL 虚拟机向
# Windows 要的总内存，长批次里 pagefile.sys 仍会变大（M8 试跑中 4.5G → 10.5G）。
# 每次运行的结果单独落盘，停下后已完成的部分仍可用 compare 汇总。
CONTAINER="testpilot-eval-$$"
WATCHDOG_PID=""
if [ -n "$CRASH_DIR" ]; then
    MIN_FREE_GB=${MIN_FREE_GB:-5}
    (
        while sleep 60; do
            free_gb=$(df -BG /c | awk 'NR==2{gsub("G","",$4); print $4}')
            if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
                echo "[eval.sh] C 盘剩余 ${free_gb}G < ${MIN_FREE_GB}G，停止评测容器 $CONTAINER" >&2
                docker kill "$CONTAINER" >/dev/null 2>&1
                exit 0
            fi
        done
    ) &
    WATCHDOG_PID=$!
fi

status=0
docker compose run --rm --name "$CONTAINER" -e TESTPILOT_COMMIT="$(git describe --always --dirty)" app python -m testpilot.eval.run "$@" || status=$?
[ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null

disk_report "结束"
DUMPS_AFTER=$( [ -n "$CRASH_DIR" ] && ls "$CRASH_DIR" 2>/dev/null | wc -l || echo 0 )
if [ "$DUMPS_AFTER" -gt "$DUMPS_BEFORE" ]; then
    echo "[eval.sh] 警告：评测期间新增了 $((DUMPS_AFTER - DUMPS_BEFORE)) 个 WSL 崩溃转储（$CRASH_DIR），会占用 C 盘，请检查并删除" >&2
fi
exit $status
