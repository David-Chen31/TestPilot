"""在子进程中执行命令：强制超时、超时后终止整个进程树、输出截断。

为什么不直接用 ``subprocess.run(timeout=...)``：它超时后只 kill 直接子进程。
被测代码（或 pytest 插件）如果再起了孙进程，孙进程会继续运行，并且一直持有
stdout/stderr 管道，导致父进程读管道时卡住，超时形同虚设。所以这里让子进程
自成一个进程组 / 会话，超时时按组整体终止。
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

# 终止进程树后再等待管道关闭的时间。正常情况下立即返回；
# 只有孙进程脱离了进程组（例如自己 setsid）时才会用满。
_DRAIN_TIMEOUT = 5.0


@dataclass(frozen=True)
class ExecResult:
    cmd: list[str]
    returncode: int | None  # 超时被终止时为 None
    stdout: str
    stderr: str
    timed_out: bool
    duration: float  # 秒

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def truncate(text: str, limit: int) -> str:
    """把过长的输出截成 ``limit`` 个字符，保留首尾。

    报错信息的开头通常是命令 / 收集阶段的错误，结尾是 traceback 最内层和
    pytest 的汇总行，两头都有用，中间最可以丢。
    """
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n... [省略 {omitted} 个字符] ...\n{text[-tail:]}"


def run(
    cmd: Sequence[str],
    cwd: str | Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
    max_output_chars: int | None = None,
) -> ExecResult:
    """执行 ``cmd``，最多运行 ``timeout`` 秒。

    ``max_output_chars`` 不为 None 时，stdout / stderr 各自按首尾截断。
    """
    cmd = [str(c) for c in cmd]
    if os.name == "posix":
        group_kwargs: dict = {"start_new_session": True}
    else:
        group_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **group_kwargs,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=_DRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
    finally:
        # 正常结束时也清理一次：测试里留下的后台进程不能活过本次执行
        _kill_tree(proc)
    duration = time.monotonic() - start

    if max_output_chars is not None:
        stdout = truncate(stdout, max_output_chars)
        stderr = truncate(stderr, max_output_chars)
    return ExecResult(
        cmd=cmd,
        returncode=None if timed_out else proc.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration=duration,
    )


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "posix":
        # 子进程是会话首进程，pgid == pid；即使它已退出，组内残留进程仍可被杀
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elif proc.poll() is None:
        # Windows 没有进程组信号，taskkill /T 按父子关系杀整棵树
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
