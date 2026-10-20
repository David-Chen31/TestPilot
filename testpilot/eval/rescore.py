"""离线重新评价独立采样（B0×N）的已保存样本，不调用 LLM。

    python -m testpilot.eval.rescore results/m8_main --tag m8_union_v2 [--workers 4]

用途：M8 第一版的 union 把收集失败的样本也放进同一次 pytest 运行，导致整个会话中断
（docs/design.md §7.5）。这里按修正后的规则（selection.union_files）重新评价 union，
写成新的结果目录，原始日志不动。输出的 run.json 里方法名为 b0n，变体名 union_v2，
token / 费用沿用原运行（样本是同一批）。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from testpilot.config import PROJECT_ROOT
from testpilot.eval.metrics import evaluate
from testpilot.eval.repos import eval_workspace, load_manifest
from testpilot.eval.run import _testpilot_commit
from testpilot.eval.selection import union_files


def rescore(run_dir: Path, out_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    screening = json.loads((run_dir / "screening.json").read_text(encoding="utf-8"))
    spec = load_manifest()[meta["repo"]]
    files = union_files(screening["samples"])
    with eval_workspace(spec) as ws:
        for f in sorted((run_dir / "samples").glob("*.py")):
            ws.write_text(f"tests_testpilot/{f.name}", f.read_text(encoding="utf-8"))
        ev = evaluate(ws.path, meta["target"], files, spec.python_paths) if files else None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "method": meta["method"],
        "repo": meta["repo"],
        "target": meta["target"],
        "samples": meta["samples"],
        "rescored_from": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "testpilot_commit": _testpilot_commit(),
        "union_files": files,
        "usage": meta["usage"],
        "variants": {"union_v2": ev.summary() if ev else {
            "tests_total": 0, "tests_valid": 0, "line_rate": 0.0, "branch_rate": 0.0,
            "mutation_score": 0.0, "mutants_total": None, "mutation_error": "没有有效测试"}},  # fmt: skip
        "variant_usage": {"union_v2": meta["usage"]},
    }
    if ev:
        (out_dir / "evaluation_union_v2.json").write_text(json.dumps(ev.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "run.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    src = args.src.resolve()
    runs = sorted(p.parent for p in src.glob("b0n/*/*/run*/screening.json"))
    dst = PROJECT_ROOT / "results" / args.tag

    def job(run_dir: Path) -> None:
        out = rescore(run_dir, dst / run_dir.relative_to(src))
        s = out["variants"]["union_v2"]
        print(f"{out['repo']}/{out['target']}: files={len(out['union_files'])} valid={s['tests_valid']}/{s['tests_total']}"
              f" mutation={s['mutation_score']} {s['mutation_error'][-80:]}", flush=True)  # fmt: skip

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(job, runs))


if __name__ == "__main__":
    main()
