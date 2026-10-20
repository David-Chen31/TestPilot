"""汇总若干批评测结果，做方法间的配对比较（M8）。

    python -m testpilot.eval.compare results/m8_main [results/...] --reference b2 --out m8_compare

- 读入各目录下所有 run.json；独立采样方法按变体（first / best / union）拆成单独的“方法”；
- 每个方法：目标级均值（先在目标内对多次运行取均值，再对目标取均值），以及有效率、覆盖率、token；
- 每个方法和参照方法做按目标配对的比较（eval/stats.py）：差值均值、95% bootstrap 置信区间、
  符号翻转检验 p 值、胜 / 负 / 平的目标数。全部目标、old 组、new 组各算一遍；
- 运行失败（API 错误等）的运行不计入，并在报告里列出数量，不悄悄丢掉。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from testpilot.config import PROJECT_ROOT
from testpilot.eval.repos import load_manifest
from testpilot.eval.stats import compare

METRICS = ("mutation_score", "line_rate", "branch_rate", "valid_ratio", "tests_valid", "total_tokens", "cost_usd_peak")


def load_runs(dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    """返回 (成功的运行, 失败的运行)。每条成功运行展开成 {method, repo, target, run, 指标...}。"""
    ok, failed = [], []
    for d in dirs:
        for f in sorted(d.rglob("run.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            base = {"repo": r["repo"], "target": r["target"], "run": f.parent.name, "path": f.parent.as_posix()}
            if "error" in r:
                failed.append({**base, "method": r["method"], "error": r["error"]})
                continue
            if "variants" in r:
                n = r.get("samples", 8)
                for variant, summary in r["variants"].items():
                    usage = r.get("variant_usage", {}).get(variant, r["usage"])
                    ok.append({**base, "method": f"{r['method']}{n}-{variant}", **_row(summary, usage)})
            else:
                ok.append({**base, "method": r["method"], **_row(r["summary"], r["usage"])})
    return ok, failed


def load_human(preflight: Path) -> list[dict]:
    """预检里仓库自带测试的得分，作为方法 "human" 参与比较（只取变异测试跑通的目标）。"""
    rows = json.loads(preflight.read_text(encoding="utf-8"))
    return [
        {"repo": r["repo"], "target": r["target"], "run": "preflight", "path": preflight.as_posix(), "method": "human",
         "mutation_score": r["human_mutation_score"], "mutation_error": "", "line_rate": r["human_line_rate"],
         "branch_rate": None, "tests_valid": r["human_tests_valid"],
         "valid_ratio": r["human_tests_valid"] / r["human_tests_total"], "total_tokens": 0, "cost_usd_peak": 0.0}
        for r in rows
        if r["human_mutation_score"] is not None and not r["mutation_error"]
    ]  # fmt: skip


def _row(summary: dict, usage: dict) -> dict:
    total = summary["tests_total"]
    return {
        "mutation_score": summary["mutation_score"],
        "mutation_error": summary.get("mutation_error", ""),
        "line_rate": summary["line_rate"],
        "branch_rate": summary["branch_rate"],
        "tests_valid": summary["tests_valid"],
        "valid_ratio": summary["tests_valid"] / total if total else 0.0,
        "total_tokens": usage["total_tokens"],
        "cost_usd_peak": usage.get("cost_usd_peak") or 0.0,
    }


def per_target(runs: list[dict], metric: str) -> dict[str, dict[str, list[float]]]:
    """method -> target -> 各次运行的指标值（变异测试出错的运行不计入变异得分）。"""
    table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        value = r[metric]
        if value is None or (metric == "mutation_score" and r["mutation_error"] and r["tests_valid"]):
            continue
        table[r["method"]][f"{r['repo']}/{r['target']}"].append(value)
    return table


def report(runs: list[dict], failed: list[dict], reference: str) -> tuple[str, dict]:
    groups = {name: spec.group for name, spec in load_manifest().items()}
    methods = sorted({r["method"] for r in runs})
    tables = {m: per_target(runs, m) for m in METRICS}
    lines = ["# 方法比较", ""]
    lines.append(f"成功运行 {len(runs)} 条，失败 {len(failed)} 条（失败的不计入，列在文末）。")
    lines.append("目标级均值：先在目标内对多次运行取均值，再对目标取均值。")
    lines.append("")
    lines.append("| 方法 | 目标数 | 变异得分 | 行覆盖 | 分支覆盖 | 有效率 | 有效测试数 | tokens | 费用上限($) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    data: dict = {"methods": {}, "comparisons": {}}
    for m in methods:
        means = {}
        for metric in METRICS:
            per = [statistics.fmean(v) for v in tables[metric][m].values() if v]
            means[metric] = statistics.fmean(per) if per else None
        n = len(tables["mutation_score"][m])
        data["methods"][m] = {"targets": n, **means}
        fmt = lambda v, f: "—" if v is None else f.format(v)  # noqa: E731
        lines.append(
            f"| {m} | {n} | {fmt(means['mutation_score'], '{:.1%}')} | {fmt(means['line_rate'], '{:.1%}')} | "
            f"{fmt(means['branch_rate'], '{:.1%}')} | {fmt(means['valid_ratio'], '{:.1%}')} | "
            f"{fmt(means['tests_valid'], '{:.0f}')} | {fmt(means['total_tokens'], '{:.0f}')} | "
            f"{fmt(means['cost_usd_peak'], '{:.4f}')} |"
        )

    scores = tables["mutation_score"]
    if reference in scores:
        for subset in ("all", "old", "new"):
            def pick(table: dict[str, list[float]]) -> dict[str, list[float]]:
                return {t: v for t, v in table.items() if subset == "all" or groups.get(t.split("/", 1)[0]) == subset}

            rows = []
            for m in methods:
                if m == reference:
                    continue
                try:
                    c = compare(pick(scores[reference]), pick(scores[m]))
                except ValueError:
                    continue
                data["comparisons"][f"{m} vs {reference} ({subset})"] = c.to_dict()
                rows.append(
                    f"| {m} | {c.n_targets} | {c.mean_b:.1%} | {c.mean_a:.1%} | {c.mean_diff:+.1%} | "
                    f"[{c.ci_low:+.1%}, {c.ci_high:+.1%}] | {c.p_value:.3g} | {c.wins} / {c.losses} / {c.ties} |"
                )
            if rows:
                lines += ["", f"## 变异得分：各方法 − {reference}（{subset} 目标，按目标配对）", ""]
                lines.append("| 方法 | 配对目标数 | 方法均值 | 参照均值 | 差值 | 95% CI | p（双侧） | 胜 / 负 / 平 |")
                lines.append("|---|---|---|---|---|---|---|---|")
                lines += rows
    if failed:
        lines += ["", "## 失败的运行", ""]
        lines += [f"- {f['method']} / {f['repo']} / {f['target']} {f['run']}: {f['error'][:200]}" for f in failed]
    return "\n".join(lines) + "\n", data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--reference", default="b2")
    parser.add_argument("--out", required=True, help="results/ 下的输出目录名")
    parser.add_argument("--human", type=Path, help="预检结果 preflight.json：加入仓库自带测试作为方法 human")
    parser.add_argument("--exclude", action="append", default=[], help="不参与比较的方法名（可重复）")
    args = parser.parse_args()
    runs, failed = load_runs(args.dirs)
    if args.human:
        runs += load_human(args.human)
    runs = [r for r in runs if r["method"] not in args.exclude]
    text, data = report(runs, failed, args.reference)
    out = PROJECT_ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.md").write_text(text, encoding="utf-8")
    (out / "compare.json").write_text(json.dumps({**data, "inputs": [d.as_posix() for d in args.dirs], "human": args.human and args.human.as_posix(), "excluded": args.exclude}, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 管道默认 GBK
    print(text)


if __name__ == "__main__":
    main()
