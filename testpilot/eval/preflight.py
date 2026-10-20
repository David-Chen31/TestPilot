"""评测前的预检（不调用 LLM）：每个目标能否导入、mutmut 能否跑通、有多少变异体、要多久。

    python -m testpilot.eval.preflight --tag m8_preflight [--repo boltons] [--workers 4]

用仓库**自带的测试**对每个目标走一遍完整的评价流程（有效性 → 覆盖率 → 变异测试）：
- 导入失败、变异测试出错的目标要在正式评测前修正或剔除，而不是在花了钱之后才发现；
- 顺便得到“开发者手写测试”的覆盖率和变异得分，作为生成测试的参照。注意它只包含在评测环境里
  能跑的那部分测试（依赖 hypothesis 等未安装插件的测试文件收集失败，按有效性规则不计入）。

只写一个 ``import`` 的测试不行：它不执行任何函数，mutmut 会因为“没有测试覆盖到变异代码”直接报错。
没有自带测试的仓库只检查能否导入。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from testpilot.agent.codegen import module_name
from testpilot.config import PROJECT_ROOT
from testpilot.eval.metrics import evaluate
from testpilot.eval.repos import RepoSpec, eval_workspace, load_manifest, snapshot_commit

IMPORT_TEST = "import {module}\n\n\ndef test_import():\n    assert {module}\n"


def check_target(spec: RepoSpec, target: str) -> dict:
    module = module_name(target, spec.python_paths)
    row: dict = {"repo": spec.name, "group": spec.group, "target": target, "module": module}
    with eval_workspace(spec, keep_tests=True) as ws:
        assert snapshot_commit(spec.path) == spec.commit
        row["lines"] = (ws.path / target).read_text(encoding="utf-8").count("\n")
        import_file = "tests_testpilot/test_preflight_import.py"
        ws.write_text(import_file, IMPORT_TEST.format(module=module))
        imported = evaluate(ws.path, target, [import_file], spec.python_paths, with_mutation=False)
        ev = evaluate(ws.path, target, [spec.tests], spec.python_paths) if spec.tests else None
    s = ev.summary() if ev else {}
    m = (ev.mutation if ev else None) or {}
    row.update(
        importable=len(imported.validity.valid) == 1,
        import_error=next(iter(imported.validity.invalid.values()), ""),
        human_tests_total=s.get("tests_total"),
        human_tests_valid=s.get("tests_valid"),
        human_line_rate=s.get("line_rate"),
        human_mutation_score=s.get("mutation_score"),
        mutants=m.get("total"),
        status_counts=m.get("status_counts"),
        mutation_error=s.get("mutation_error", ""),
        durations=ev.durations if ev else {},
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", action="append", help="默认检查清单中的全部仓库")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = load_manifest()
    jobs = [(manifest[r], t) for r in (args.repo or manifest) for t in manifest[r].targets]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda job: check_target(*job), jobs))

    out = PROJECT_ROOT / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in rows:
        flag = "OK " if r["importable"] and not r["mutation_error"] else "BAD"
        score = r["human_mutation_score"]
        print(
            f"{flag} {r['repo']:16s} {r['target']:42s} lines={r['lines']:5d} mutants={r['mutants']}"
            f" human={r['human_tests_valid']}/{r['human_tests_total']}"
            f" score={'-' if score is None else f'{score:.1%}'} t_mut={r['durations'].get('mutation')}"
            f" {r['import_error'][:80]} {r['mutation_error'][-120:]}"
        )
    bad = [r for r in rows if not r["importable"] or r["mutation_error"]]
    print(f"{len(rows) - len(bad)}/{len(rows)} 个目标通过预检；变异体合计 {sum(r['mutants'] or 0 for r in rows)}")


if __name__ == "__main__":
    main()
