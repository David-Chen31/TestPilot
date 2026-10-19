"""TestPilot Agent：用 LangGraph StateGraph 显式定义的状态机。

    plan → retrieve_context → write_tests → run_tests → analyze → decide ─┬→ write_tests（带反馈）
                                                                         └→ finalize → END

和标准 ReAct 的区别：ReAct 由 LLM 在每一步自己决定调用哪个工具；这里流程是固定的图，
工具由节点确定性地调用，LLM 只负责“写测试”这一步。拓扑固定、终止条件显式、
每一步可复现，对比实验里变化的只有反馈内容。

B0 / B1 / B2 是同一张图的三种配置：
- B0：feedback 为空，只写一次（B0 的提示词就是 B1/B2 第一轮的提示词）；
- B1：pytest 失败信息反馈；
- B2：测试全部通过后，在有效测试上做变异测试，把存活变异体反馈回去。
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from testpilot import prompts
from testpilot.agent.codegen import extract_code, generated_test_path, module_name
from testpilot.agent.tool_client import ToolClient, mcp_tool_client
from testpilot.llm import Usage, ainvoke
from testpilot.retrieval.context import build_context

MAX_FEEDBACK_ITEMS = 15
MAX_MESSAGE_CHARS = 800
MAX_SURVIVORS_FETCHED = 200  # 从工具取回的存活变异体上限；反馈里只放 MAX_FEEDBACK_ITEMS 个


@dataclass(frozen=True)
class AgentConfig:
    name: str
    feedback: frozenset[str] = frozenset()  # "pytest"、"mutation"
    # 终止条件（满足任意一条即停止）：
    max_iterations: int = 5  # write_tests 最多调用次数
    patience: int = 2  # 连续多少轮进度分数没有提升
    token_budget: int = 300_000  # 累计 token（输入 + 输出）上限


B0 = AgentConfig("b0", feedback=frozenset(), max_iterations=1)
B1 = AgentConfig("b1", feedback=frozenset({"pytest"}))
B2 = AgentConfig("b2", feedback=frozenset({"pytest", "mutation"}))


class AgentState(TypedDict, total=False):
    # 任务
    target: str
    module: str
    test_file: str
    python_paths: list[str]
    # 计划与上下文
    public_symbols: list[str]
    prompt_prefix: str
    # 当前一轮
    iteration: int  # 已调用 write_tests 的次数
    test_code: str
    run: dict  # 最近一次 pytest 结果
    coverage: dict | None
    mutation: dict | None
    feedback: str  # 给下一轮 write_tests 的反馈；为空表示没有可改进之处
    shown_mutants: list[str]  # 已经在反馈中展示过的变异体（行号 + 变异后代码）
    # 进度与终止
    score: list  # 进度分数（按字典序比较），见 progress_score
    best: dict  # 分数最高的一轮：code、score、iteration
    stale_rounds: int
    tokens: int
    next: str
    stop_reason: str
    # 结构化轨迹，每个节点追加一条
    trace: Annotated[list[dict], operator.add]


def progress_score(run: dict, coverage: dict | None, mutation: dict | None) -> list:
    """按字典序比较：全部通过 > 变异得分 > 行覆盖率 > 通过的测试数。"""
    return [
        int(bool(run.get("all_passed"))),
        (mutation or {}).get("score") or 0.0,
        (coverage or {}).get("line_rate") or 0.0,
        run.get("passed", 0),
    ]


def _mutant_key(m: dict) -> str:
    return f"{m['line']}|{m['mutated']}"


EventHook = Callable[[dict], Awaitable[None]]


class TestPilotAgent:
    __test__ = False  # 名字以 Test 开头，告诉 pytest 这不是测试类

    def __init__(self, model: BaseChatModel, tools: ToolClient, config: AgentConfig, on_event: EventHook | None = None):
        self.model = model
        self.tools = tools
        self.config = config
        self.usage = Usage()
        self.on_event = on_event
        self._calls_seen = 0
        self.graph = self._build()

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self.plan)
        g.add_node("retrieve_context", self.retrieve_context)
        g.add_node("write_tests", self.write_tests)
        g.add_node("run_tests", self.run_tests)
        g.add_node("analyze", self.analyze)
        g.add_node("decide", self.decide)
        g.add_node("finalize", self.finalize)
        g.add_edge(START, "plan")
        g.add_edge("plan", "retrieve_context")
        g.add_edge("retrieve_context", "write_tests")
        g.add_edge("write_tests", "run_tests")
        g.add_edge("run_tests", "analyze")
        g.add_edge("analyze", "decide")
        g.add_conditional_edges("decide", lambda s: s["next"], {"write_tests": "write_tests", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    # ---------------- 节点 ----------------

    async def plan(self, state: AgentState) -> dict:
        """确定要覆盖的公开 API（AST，不用 LLM：确定性的事不交给模型）。"""
        t0 = time.monotonic()
        symbols = (await self.tools.call("list_symbols", {"path": state["target"]}))["symbols"]
        public = [s["qualname"] for s in symbols if s["kind"] != "method" and not s["name"].startswith("_")]
        return {
            "public_symbols": public,
            "iteration": 0,
            "stale_rounds": 0,
            "tokens": 0,
            "trace": [await self._event("plan", state, t0, f"{len(public)} 个公开符号", public_symbols=public)],
        }

    async def retrieve_context(self, state: AgentState) -> dict:
        t0 = time.monotonic()
        ctx = await build_context(
            self.tools, state["target"], state["module"], state["python_paths"], public=state["public_symbols"]
        )
        prefix = prompts.PREFIX.format(
            module=state["module"],
            target=state["target"],
            numbered_source=ctx.numbered_source,
            context=ctx.render(),
            public_symbols=", ".join(f"`{s}`" for s in state["public_symbols"]),
            requirements=prompts.TEST_REQUIREMENTS.format(module=state["module"]),
        )
        summary = f"依赖定义 {len(ctx.dependencies)} 个，调用处 {len(ctx.usages)} 个，提示词前缀 {len(prefix)} 字符"
        return {"prompt_prefix": prefix, "trace": [await self._event("retrieve_context", state, t0, summary, prompt_prefix=prefix)]}

    async def write_tests(self, state: AgentState) -> dict:
        t0 = time.monotonic()
        iteration = state["iteration"] + 1
        if iteration == 1:
            task = prompts.TASK_FIRST.format(module=state["module"])
        else:
            task = prompts.TASK_REVISE.format(
                test_file=state["test_file"], test_code=state.get("test_code", ""), feedback=state["feedback"]
            )
        messages = [SystemMessage(prompts.SYSTEM), HumanMessage(f"{state['prompt_prefix']}\n\n{task}")]
        call = await ainvoke(self.model, messages)
        self.usage.add(call)
        code = extract_code(call.content)
        if code:
            await self.tools.call("write_test_file", {"path": state["test_file"], "content": code})
        summary = f"第 {iteration} 轮：生成 {code.count(chr(10))} 行测试" if code else f"第 {iteration} 轮：回复中没有代码块"
        event = await self._event(
            "write_tests",
            {**state, "iteration": iteration},
            t0,
            summary,
            llm={
                "task": task,  # 前缀在 retrieve_context 事件里，各轮相同，不重复记录
                "response": call.content,
                "finish_reason": call.finish_reason,
                "prompt_tokens": call.prompt_tokens,
                "cache_hit_tokens": call.cache_hit_tokens,
                "completion_tokens": call.completion_tokens,
                "reasoning_tokens": call.reasoning_tokens,
                "latency": round(call.latency, 2),
            },
        )
        return {
            "iteration": iteration,
            "test_code": code,
            "tokens": self.usage.total_tokens,
            "trace": [event],
        }

    async def run_tests(self, state: AgentState) -> dict:
        t0 = time.monotonic()
        if not state["test_code"]:
            run = {"no_code": True, "all_passed": False, "passed": 0, "failed": 0, "errors": 0, "problems": []}
            summary = "没有测试代码，跳过"
        else:
            run = await self.tools.call("run_pytest", {"tests": [state["test_file"]]})
            summary = f"通过 {run['passed']}，失败 {run['failed']}，错误 {run['errors']}" + ("，超时" if run["timed_out"] else "")
        return {"run": run, "trace": [await self._event("run_tests", state, t0, summary)]}

    async def analyze(self, state: AgentState) -> dict:
        """计算进度分数、更新最好的一轮，并准备给下一轮的反馈。

        启用变异反馈（B2）且测试全部通过时：先在原代码上跑 3 次确定有效测试，
        覆盖率和变异测试都**只用有效测试**——在原代码上不稳定的测试会把变异得分虚高。
        """
        t0 = time.monotonic()
        run = state["run"]
        target = state["target"]
        coverage = mutation = validity = None
        shown = list(state.get("shown_mutants", []))
        if run["all_passed"] and "mutation" in self.config.feedback:
            validity = await self.tools.call("check_validity", {"tests": [state["test_file"]]})
            if validity["valid_ids"]:
                tests = validity["valid_ids"]
                coverage = await self.tools.call("coverage_report", {"target": target, "tests": tests})
                mutation = await self.tools.call(
                    "run_mutation", {"target": target, "tests": tests, "max_survivors": MAX_SURVIVORS_FETCHED}
                )
        elif run["all_passed"]:
            coverage = await self.tools.call("coverage_report", {"target": target, "tests": [state["test_file"]]})

        score = progress_score(run, coverage, mutation)
        best = state.get("best")
        improved = best is None or score > best["score"]
        if improved:
            best = {"code": state["test_code"], "score": score, "iteration": state["iteration"]}
        stale = 0 if improved else state["stale_rounds"] + 1

        feedback = ""
        if not run["all_passed"]:
            feedback = self._pytest_feedback(run) if "pytest" in self.config.feedback else ""
        elif "mutation" in self.config.feedback:
            feedback, shown = self._mutation_feedback(target, run, validity, mutation, shown)

        cov = f"，行覆盖 {coverage['line_rate']:.1%}，分支覆盖 {coverage['branch_rate'] or 0:.1%}" if coverage else ""
        mut = f"，变异得分 {mutation['score']:.1%}（存活 {mutation['survived']}）" if mutation and mutation["score"] is not None else ""
        summary = f"分数 {score}{'（提升）' if improved else f'（连续 {stale} 轮无提升）'}{cov}{mut}"
        mutation_brief = {k: v for k, v in (mutation or {}).items() if k != "survivors"} or None
        validity_brief = {k: v for k, v in (validity or {}).items() if k != "valid_ids"} or None
        return {
            "coverage": coverage,
            "mutation": mutation_brief,
            "score": score,
            "best": best,
            "stale_rounds": stale,
            "feedback": feedback,
            "shown_mutants": shown,
            "trace": [
                await self._event(
                    "analyze", state, t0, summary,
                    score=score, coverage=coverage, validity=validity_brief, mutation=mutation_brief,
                )  # fmt: skip
            ],
        }

    async def decide(self, state: AgentState) -> dict:
        """终止条件按顺序检查，满足任意一条就结束。"""
        t0 = time.monotonic()
        cfg = self.config
        stop = None
        mutation = state.get("mutation") or {}
        if not cfg.feedback:
            stop = "单次生成（无反馈）"
        elif not state["feedback"]:
            # 没有可反馈的问题了
            if "mutation" not in cfg.feedback:
                stop = "全部测试通过，pytest 反馈已无可改进之处"
            elif mutation.get("error"):
                stop = f"变异测试出错，无法继续反馈：{mutation['error'][:200]}"
            else:
                stop = "全部测试有效，且没有存活的变异体"
        elif state["iteration"] >= cfg.max_iterations:
            stop = f"达到最大迭代数 {cfg.max_iterations}"
        elif state["tokens"] >= cfg.token_budget:
            stop = f"token 预算用完（{state['tokens']} ≥ {cfg.token_budget}）"
        elif state["stale_rounds"] >= cfg.patience:
            stop = f"连续 {state['stale_rounds']} 轮没有提升"
        nxt = "finalize" if stop else "write_tests"
        return {
            "next": nxt,
            "stop_reason": stop or "",
            "trace": [await self._event("decide", state, t0, stop or "继续：带反馈重写测试")],
        }

    async def finalize(self, state: AgentState) -> dict:
        """把分数最高的一轮写回测试文件（最后一轮不一定最好）。"""
        t0 = time.monotonic()
        best = state["best"]
        if best["code"] and best["code"] != state["test_code"]:
            await self.tools.call("write_test_file", {"path": state["test_file"], "content": best["code"]})
        elif not best["code"]:
            # 一轮都没生成出代码：仍然留下（空的）测试文件，评测时记为 0 个测试，而不是报错
            await self.tools.call("write_test_file", {"path": state["test_file"], "content": "# no tests generated" + chr(10)})
        summary = f"停止原因：{state['stop_reason']}；采用第 {best['iteration']} 轮，分数 {best['score']}"
        return {"trace": [await self._event("finalize", state, t0, summary, best_iteration=best["iteration"])]}

    # ---------------- 辅助 ----------------

    def _mutation_feedback(
        self, target: str, run: dict, validity: dict, mutation: dict | None, shown: list[str]
    ) -> tuple[str, list[str]]:
        """把不稳定的测试和存活变异体整理成反馈。返回 (反馈文本, 更新后的“已展示变异体”列表)。"""
        parts = []
        if validity["invalid"]:
            items = "\n".join(f"- `{t}`: {r}" for t, r in list(validity["invalid"].items())[:MAX_FEEDBACK_ITEMS])
            parts.append(prompts.FEEDBACK_FLAKY.format(n=len(validity["invalid"]), items=items))
        if mutation and not mutation["error"] and mutation["survivors"]:
            # 排序：没展示过的优先（展示过还活着的，多半是等价变异体，不该一直占着反馈）；
            # 其次是没有任何测试执行到的；最后按行号
            def key(m: dict) -> tuple:
                return (_mutant_key(m) in shown, m["status"] != "no tests", m["line"] or 0)

            chosen = sorted(mutation["survivors"], key=key)[:MAX_FEEDBACK_ITEMS]
            items = "\n".join(
                f"- line {m['line']}: `{m['original']}` → `{m['mutated']}`"
                + (" (no test executes this line)" if m["status"] == "no tests" else "")
                for m in chosen
            )
            more = mutation["survived"] - len(chosen)
            parts.append(
                prompts.FEEDBACK_MUTATION.format(
                    passed=run["passed"],
                    target=target,
                    killed=mutation["killed"],
                    scorable=mutation["killed"] + mutation["survived"],
                    score=mutation["score"] or 0.0,
                    items=items + (f"\n... and {more} more surviving mutants not shown." if more > 0 else ""),
                )
            )
            shown = shown + [k for k in map(_mutant_key, chosen) if k not in shown]
        return "\n\n".join(parts), shown

    def _pytest_feedback(self, run: dict) -> str:
        if run.get("no_code"):
            return prompts.FEEDBACK_NO_CODE
        if run["all_passed"]:
            return ""
        lines = [prompts.FEEDBACK_PYTEST_HEADER.format(passed=run["passed"], failed=run["failed"], errors=run["errors"])]
        if run.get("timed_out"):
            lines.append("The whole pytest run timed out; a test probably loops forever or blocks.")
        for p in run["problems"][:MAX_FEEDBACK_ITEMS]:
            lines.append(f"- `{p['test']}` {p['outcome']}:\n```\n{p['message'][:MAX_MESSAGE_CHARS]}\n```")
        if len(run["problems"]) > MAX_FEEDBACK_ITEMS:
            lines.append(f"... and {len(run['problems']) - MAX_FEEDBACK_ITEMS} more failing tests.")
        if not run["problems"] and run.get("output"):
            lines.append(f"pytest output:\n```\n{run['output'][-3000:]}\n```")  # 收集失败时没有逐用例结果
        return "\n".join(lines)

    async def _event(self, node: str, state: AgentState, t0: float, summary: str, **extra: Any) -> dict:
        calls = self.tools.calls[self._calls_seen :]
        self._calls_seen = len(self.tools.calls)
        event = {
            "node": node,
            "iteration": state.get("iteration", 0),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration": round(time.monotonic() - t0, 2),
            "summary": summary,
            "tool_calls": [c.to_dict() for c in calls],
            "tokens_total": self.usage.total_tokens,
            **extra,
        }
        if self.on_event is not None:
            await self.on_event(event)
        return event


@dataclass
class GenerationResult:
    test_files: list[str]
    usage: Usage
    trace: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)  # 停止原因、迭代次数等


async def run_agent(
    model: BaseChatModel,
    workdir: str | Path,
    target: str,
    python_paths: Sequence[str],
    config: AgentConfig,
    tools: ToolClient | None = None,
    on_event: EventHook | None = None,
) -> GenerationResult:
    """在工作副本上运行 Agent。默认启动绑定到该副本的 MCP server；测试时可注入 tools。"""
    if tools is None:
        async with mcp_tool_client(workdir, python_paths) as mcp_tools:
            return await run_agent(model, workdir, target, python_paths, config, mcp_tools, on_event)

    agent = TestPilotAgent(model, tools, config, on_event)
    initial: AgentState = {
        "target": target,
        "module": module_name(target, python_paths),
        "test_file": generated_test_path(target),
        "python_paths": list(python_paths),
        "trace": [],
    }
    final = await agent.graph.ainvoke(initial, {"recursion_limit": 10 * config.max_iterations + 20})
    return GenerationResult(
        test_files=[final["test_file"]],
        usage=agent.usage,
        trace=final["trace"],
        info={
            "stop_reason": final["stop_reason"],
            "iterations": final["iteration"],
            "best_iteration": final["best"]["iteration"],
            "final_score": final["best"]["score"],
        },
    )
