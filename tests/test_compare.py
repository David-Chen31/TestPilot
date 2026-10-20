"""汇总比较：读 run.json、拆变体、失败运行单独列出。"""

import json
from pathlib import Path

from testpilot.eval.compare import load_runs, report


def _summary(score, valid=10, total=10, error=""):
    return {"tests_total": total, "tests_valid": valid, "line_rate": 0.9, "branch_rate": 0.8,
            "mutation_score": score, "mutation_error": error}  # fmt: skip


def _write(d: Path, rel: str, data: dict) -> None:
    (d / rel).mkdir(parents=True)
    (d / rel / "run.json").write_text(json.dumps(data), encoding="utf-8")


def test_load_runs_and_report(tmp_path: Path):
    usage = {"total_tokens": 100, "cost_usd_peak": 0.01}
    for i, (t, a, b) in enumerate([("t1.py", 0.5, 0.7), ("t2.py", 0.6, 0.6)]):
        _write(tmp_path, f"b1/toolz/{i}/run1", {"method": "b1", "repo": "toolz", "target": t, "summary": _summary(a), "usage": usage})
        _write(tmp_path, f"b2/toolz/{i}/run1", {"method": "b2", "repo": "toolz", "target": t, "summary": _summary(b), "usage": usage})
    _write(tmp_path, "b0n/toolz/0/run1", {
        "method": "b0n", "repo": "toolz", "target": "t1.py", "samples": 8, "usage": {"total_tokens": 800},
        "variants": {"first": _summary(0.4), "union": _summary(0.8)},
        "variant_usage": {"first": {"total_tokens": 100}, "union": {"total_tokens": 800}},
    })  # fmt: skip
    _write(tmp_path, "b2/toolz/9/run1", {"method": "b2", "repo": "toolz", "target": "t3.py", "error": "APIError: boom"})
    # 有有效测试但变异测试出错：不计入变异得分（基础设施问题，不是方法的 0 分）
    _write(tmp_path, "b1/toolz/7/run1", {"method": "b1", "repo": "toolz", "target": "t4.py", "summary": _summary(None, error="mutmut 超时"), "usage": usage})

    runs, failed = load_runs([tmp_path])
    assert {r["method"] for r in runs} == {"b1", "b2", "b0n8-first", "b0n8-union"}
    assert [f["error"] for f in failed] == ["APIError: boom"]
    assert next(r for r in runs if r["method"] == "b0n8-first")["total_tokens"] == 100

    text, data = report(runs, failed, "b2")
    c = data["comparisons"]["b1 vs b2 (all)"]
    assert c["n_targets"] == 2 and round(c["mean_diff"], 6) == -0.1
    assert data["methods"]["b1"]["targets"] == 2  # t4 的变异得分不计入
    assert "失败的运行" in text and "APIError: boom" in text
