import sys
import time
from pathlib import Path

from testpilot.sandbox.executor import run, truncate

PY = sys.executable


def test_run_captures_output_and_returncode(tmp_path: Path):
    res = run([PY, "-c", "import sys; print('hi'); sys.exit(3)"], cwd=tmp_path, timeout=10)
    assert res.stdout.strip() == "hi"
    assert res.returncode == 3
    assert not res.timed_out
    assert not res.ok


def test_run_times_out_and_terminates(tmp_path: Path):
    start = time.monotonic()
    res = run([PY, "-c", "import time; time.sleep(60)"], cwd=tmp_path, timeout=1)
    assert res.timed_out
    assert res.returncode is None
    assert time.monotonic() - start < 10


def test_timeout_kills_grandchild(tmp_path: Path):
    """子进程再起一个孙进程：超时后孙进程也必须被终止，不能留下来继续写文件。"""
    marker = tmp_path / "grandchild_survived"
    grandchild = f"import time, pathlib; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('x')"
    child = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "time.sleep(60)"
    )
    res = run([PY, "-c", child], cwd=tmp_path, timeout=1)
    assert res.timed_out
    time.sleep(4)  # 孙进程如果还活着，此时已经写出 marker
    assert not marker.exists()


def test_run_truncates_long_output(tmp_path: Path):
    res = run([PY, "-c", "print('a' * 5000 + 'END')"], cwd=tmp_path, timeout=10, max_output_chars=100)
    assert "省略" in res.stdout
    assert res.stdout.rstrip().endswith("END")


def test_truncate_keeps_head_and_tail():
    text = "HEAD" + "x" * 1000 + "TAIL"
    out = truncate(text, 20)
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "省略 988 个字符" in out


def test_truncate_short_text_unchanged():
    assert truncate("short", 100) == "short"
