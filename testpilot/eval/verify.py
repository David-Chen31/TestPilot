"""抽查结果的真实性：把结果目录里保存的测试文件放回固定 commit 的仓库，重新评价，与记录的分数逐项比较。

    python -m testpilot.eval.verify results/m8_main --sample 8 --seed 20261020 --tag verify_m8

不调用 LLM。分数不是从日志里读的，而是对保存的测试文件重新运行 pytest / coverage / mutmut 得到的。
LLM 生成是随机的、不可重放；可以重放的是“这些测试文件 → 这些分数”这一步。
抽样用固定种子，抽到哪些就报告哪些，不挑。
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from testpilot.agent.codegen import generated_test_path
from testpilot.config import PROJECT_ROOT
from testpilot.eval.metrics import evaluate
from testpilot.eval.repos import eval_workspace, load_manifest

FIELDS = ("tests_total", "tests_valid", "line_rate", "branch_rate", "mutation_score", "mutants_total")


def candidates(src: Path) -> list[dict]:
    """可以抽查的评价：单文件方法的 evaluation.json，以及独立采样的 first / best 变体。"""
    out = []
    for run_json in sorted(src.rglob("run.json")):
        run_dir = run_json.parent
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        if "error" in meta:
            continue
        base = {"repo": meta["repo"], "target": meta["target"], "run_dir": run_dir}
        if "variants" in meta:
            screening = json.loads((run_dir / "screening.json").read_text(encoding="utf-8"))
            stem = Path(generated_test_path(meta["target"])).stem
            for variant, index in (("first", 0), ("best", screening["selected"])):
                out.append({**base, "label": f"{meta['method']}-{variant}",
                            "files": [run_dir / "samples" / f"{stem}_s{index + 1}.py"],
                            "recorded": run_dir / f"evaluation_{variant}.json"})  # fmt: skip
        else:
            [test_file] = [p for p in run_dir.glob("test_*.py")]
            out.append({**base, "label": meta["method"], "files": [test_file], "recorded": run_dir / "evaluation.json"})
    return out


def summarize(ev: dict) -> dict:
    m = ev.get("mutation") or {}
    return {
        "tests_total": ev["validity"]["total"],
        "tests_valid": ev["validity"]["valid"],
        "line_rate": ev["line_rate"],
        "branch_rate": ev["branch_rate"],
        "mutation_score": m.get("score"),
        "mutants_total": m.get("total"),
    }


def recheck(c: dict) -> dict:
    spec = load_manifest()[c["repo"]]
    with eval_workspace(spec) as ws:
        rels = []
        for f in c["files"]:
            rel = f"tests_testpilot/{f.name}"
            ws.write_text(rel, f.read_text(encoding="utf-8"))
            rels.append(rel)
        ev = evaluate(ws.path, c["target"], rels, spec.python_paths)
    recorded = summarize(json.loads(c["recorded"].read_text(encoding="utf-8")))
    again = summarize(ev.to_dict())
    return {
        "label": c["label"],
        "repo": c["repo"],
        "target": c["target"],
        "run_dir": c["run_dir"].relative_to(PROJECT_ROOT).as_posix(),
        "recorded": recorded,
        "recomputed": again,
        "identical": recorded == again,
        "mutation_error": (ev.mutation or {}).get("error", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("--sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    pool_ = candidates(args.src.resolve())
    chosen = random.Random(args.seed).sample(pool_, min(args.sample, len(pool_)))
    print(f"候选 {len(pool_)} 个，种子 {args.seed}，抽到 {len(chosen)} 个", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(recheck, chosen))

    out = PROJECT_ROOT / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify.json").write_text(
        json.dumps({"src": args.src.as_posix(), "seed": args.seed, "candidates": len(pool_), "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )  # fmt: skip
    for r in rows:
        print(f"{'一致' if r['identical'] else '不同'}  {r['label']:10s} {r['repo']}/{r['target']}")
        for k in FIELDS:
            a, b = r["recorded"][k], r["recomputed"][k]
            print(f"      {k:15s} 记录 {a!s:22s} 重算 {b!s:22s}{'' if a == b else '  <- 不同'}")


if __name__ == "__main__":
    main()
