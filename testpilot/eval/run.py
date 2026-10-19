"""评测入口：对一个仓库的目标模块，用指定方法生成测试并评价，原始日志写入 results/。

    python -m testpilot.eval.run --method b0 --repo toolz --runs 3 --tag m3_b0

需要变异测试时在 Docker 中运行（见 scripts/eval.sh）。输出目录结构::

    results/<tag>/<method>/<repo>/<target>/run<k>/
        run.json         元信息：模型、温度、fingerprint、token、费用、耗时、commit
        trace.json       结构化轨迹（每一步的 prompt / 回复 / 耗时）
        evaluation.json  有效性明细、覆盖率、变异测试（含全部存活变异体）
        test_*.py        生成的测试文件
    results/<tag>/summary.json, summary.md   汇总
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import asyncio

from testpilot.agent.graph import B0, B1, B2, AgentConfig, run_agent
from testpilot.config import PROJECT_ROOT, Settings
from testpilot.eval.metrics import evaluate
from testpilot.eval.repos import RepoSpec, eval_workspace, load_manifest, snapshot_commit
from testpilot.llm import make_chat_model

log = logging.getLogger("testpilot.eval")

# 所有方法都是同一张 Agent 图的不同配置
METHODS: dict[str, AgentConfig] = {"b0": B0, "b1": B1, "b2": B2}
RESULTS_DIR = PROJECT_ROOT / "results"


async def run_one(
    method: str, spec: RepoSpec, target: str, out_dir: Path, settings: Settings, with_mutation: bool
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "method": method,
        "repo": spec.name,
        "repo_commit": spec.commit,
        "target": target,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "thinking": settings.llm_thinking,
        "max_tokens": settings.llm_max_tokens,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "testpilot_commit": _testpilot_commit(),
        "python": platform.python_version(),
    }
    model = make_chat_model(settings)
    with eval_workspace(spec) as ws:
        assert snapshot_commit(spec.path) == spec.commit
        t0 = time.monotonic()
        try:
            gen = await run_agent(model, ws.path, target, spec.python_paths, METHODS[method])
        except Exception as e:  # API 失败等：记录下来，不中断整批
            log.exception("生成失败")
            meta["error"] = _describe(e)
            _dump(out_dir / "run.json", meta)
            return meta
        meta["generation_seconds"] = round(time.monotonic() - t0, 2)
        meta["usage"] = gen.usage.to_dict(settings.llm_model)
        meta["agent"] = {**gen.info, "config": _config_dict(METHODS[method])}
        _dump(out_dir / "trace.json", gen.trace)
        for f in gen.test_files:
            shutil.copy(ws.path / f, out_dir / Path(f).name)

        t0 = time.monotonic()
        # 评测是阻塞的子进程调用，放到线程里
        ev = await asyncio.to_thread(
            evaluate, ws.path, target, gen.test_files, spec.python_paths, with_mutation=with_mutation
        )
        meta["evaluation_seconds"] = round(time.monotonic() - t0, 2)
    _dump(out_dir / "evaluation.json", ev.to_dict())
    meta["summary"] = ev.summary()
    _dump(out_dir / "run.json", meta)
    return meta


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if "summary" in r]
    rows = [
        {
            "tests_total": r["summary"]["tests_total"],
            "tests_valid": r["summary"]["tests_valid"],
            "valid_ratio": r["summary"]["tests_valid"] / r["summary"]["tests_total"] if r["summary"]["tests_total"] else 0.0,
            "line_rate": r["summary"]["line_rate"],
            "branch_rate": r["summary"]["branch_rate"],
            "mutation_score": r["summary"]["mutation_score"],
            "generation_seconds": r["generation_seconds"],
            "total_tokens": r["usage"]["total_tokens"],
            "cost_usd_peak": r["usage"]["cost_usd_peak"],
        }
        for r in ok
    ]
    agg = {}
    for key in rows[0] if rows else []:
        values = [row[key] for row in rows if row[key] is not None]
        if values:
            agg[key] = {"mean": statistics.fmean(values), "min": min(values), "max": max(values), "n": len(values)}
    return {"runs": len(records), "failed_runs": len(records) - len(ok), "per_run": rows, "aggregate": agg}


def write_markdown(path: Path, title: str, groups: dict[str, dict], records: dict[str, list[dict]]) -> None:
    cols = [
        ("tests_valid", "有效测试", "{:.0f}"),
        ("tests_total", "生成测试", "{:.0f}"),
        ("valid_ratio", "有效率", "{:.1%}"),
        ("line_rate", "行覆盖", "{:.1%}"),
        ("branch_rate", "分支覆盖", "{:.1%}"),
        ("mutation_score", "变异得分", "{:.1%}"),
        ("generation_seconds", "生成耗时(s)", "{:.1f}"),
        ("total_tokens", "tokens", "{:.0f}"),
        ("cost_usd_peak", "费用上限($)", "{:.4f}"),
    ]
    lines = [f"# {title}", "", "均值 [最小, 最大]；费用按高峰价估算（上限）。", ""]
    lines.append("| 设置 | 运行次数 | " + " | ".join(c[1] for c in cols) + " |")
    lines.append("|---" * (len(cols) + 2) + "|")
    for key, s in groups.items():
        cells = []
        for col, _, fmt in cols:
            a = s["aggregate"].get(col)
            cells.append(f"{fmt.format(a['mean'])} [{fmt.format(a['min'])}, {fmt.format(a['max'])}]" if a else "—")
        lines.append(f"| {key} | {s['runs'] - s['failed_runs']}/{s['runs']} | " + " | ".join(cells) + " |")
    lines += ["", "## 原始日志", ""]
    for key, recs in records.items():
        for r in recs:
            lines.append(f"- {key} run{r['run_index']}: `{r['out_dir']}`" + (f" — 失败: {r['error']}" if "error" in r else ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--target", action="append", help="默认使用清单中的全部 targets")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tag", required=True, help="本批实验的名字，作为 results/ 下的目录名")
    parser.add_argument("--no-mutation", action="store_true")
    args = parser.parse_args()

    tag_dir = RESULTS_DIR / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(tag_dir / "run.log", encoding="utf-8")],
    )
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 整批运行共用一个事件循环：langchain-openai 在进程级缓存异步 HTTP 客户端，
    # 每次 asyncio.run() 新建循环会让缓存的连接绑在已关闭的循环上（“Event loop is closed”）
    asyncio.run(_run_batch(args, tag_dir))


async def _run_batch(args: argparse.Namespace, tag_dir: Path) -> None:
    settings = Settings.from_env()
    spec = load_manifest()[args.repo]
    targets = args.target or list(spec.targets)
    groups: dict[str, dict] = {}
    all_records: dict[str, list[dict]] = {}
    connection_failures, aborted = 0, False
    for target in targets:
        key = f"{args.method} / {spec.name} / {target}"
        records = []
        for k in range(1, args.runs + 1):
            out_dir = tag_dir / args.method / spec.name / Path(target).stem / f"run{k}"
            log.info("%s run %d/%d -> %s", key, k, args.runs, out_dir)
            rec = await run_one(args.method, spec, target, out_dir, settings, with_mutation=not args.no_mutation)
            rec["run_index"] = k
            rec["out_dir"] = out_dir.relative_to(PROJECT_ROOT).as_posix()
            log.info("%s run %d 结果: %s", key, k, rec.get("summary") or rec.get("error"))
            records.append(rec)
            # 连续两次连接失败说明是网络 / 基础设施问题，继续跑只会在重试里耗时间，直接中止整批
            connection_failures = connection_failures + 1 if _is_connection_error(rec) else 0
            if connection_failures >= MAX_CONSECUTIVE_CONNECTION_FAILURES:
                log.error("连续 %d 次连接失败，中止本批（网络问题，与方法无关）", connection_failures)
                aborted = True
                break
        groups[key] = summarize(records)
        all_records[key] = records
        if aborted:
            break

    _dump(tag_dir / "summary.json", {"groups": groups, "records": all_records, "aborted": aborted})
    write_markdown(tag_dir / "summary.md", f"{args.tag}", groups, all_records)
    log.info("汇总写入 %s", tag_dir / "summary.md")


MAX_CONSECUTIVE_CONNECTION_FAILURES = 2


def _is_connection_error(rec: dict) -> bool:
    return "ConnectionError" in rec.get("error", "") or "Connection error" in rec.get("error", "")


def _describe(e: BaseException) -> str:
    """异常的简短描述；ExceptionGroup（TaskGroup 里抛出）展开到最内层，否则只剩一句 “unhandled errors”。"""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}"


def _testpilot_commit() -> str:
    if os.environ.get("TESTPILOT_COMMIT"):
        return os.environ["TESTPILOT_COMMIT"]
    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "describe", "--always", "--dirty"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _config_dict(cfg: AgentConfig) -> dict:
    return {
        "feedback": sorted(cfg.feedback),
        "max_iterations": cfg.max_iterations,
        "patience": cfg.patience,
        "token_budget": cfg.token_budget,
    }


def _dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
