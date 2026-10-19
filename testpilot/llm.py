"""LLM 调用：ChatOpenAI（OpenAI 兼容接口）+ token 用量与费用记录。

M7 会在这里加上指数退避重试日志和降级策略；目前重试交给 openai SDK 自带的 max_retries。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from testpilot.config import Settings

# 单价：美元 / 百万 token，取**高峰时段**价格，所以费用估算是上限（非高峰时段为一半）。
# 来源：https://api-docs.deepseek.com/quick_start/pricing ，2026-10-19 查询。
PRICES_USD_PER_M: dict[str, dict[str, float]] = {
    "deepseek-flash": {"input_cache_hit": 0.006, "input_cache_miss": 0.3, "output": 1.2},
    "deepseek-v4-pro": {"input_cache_hit": 0.044, "input_cache_miss": 1.32, "output": 3.96},
}


def make_chat_model(settings: Settings) -> ChatOpenAI:
    # langchain 会把 max_tokens 改名成 max_completion_tokens 发送，而 DeepSeek 会忽略后者
    # （实测：输出上限完全不生效）。所以通过 extra_body 直接发送 max_tokens。
    extra: dict = {"max_tokens": settings.llm_max_tokens}
    if settings.llm_thinking != "auto":
        extra["thinking"] = {"type": settings.llm_thinking}
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        extra_body=extra,
    )


@dataclass
class LLMCall:
    content: str
    prompt_tokens: int
    completion_tokens: int  # 含思考 token，按输出价计费
    reasoning_tokens: int
    cache_hit_tokens: int
    latency: float  # 秒
    model: str
    fingerprint: str  # system_fingerprint：模型别名不可锁定版本，用它记录后台实际版本
    finish_reason: str


@dataclass
class Usage:
    """一次任务（或一组调用）累计的 token 用量。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cache_hit_tokens: int = 0
    latency: float = 0.0
    fingerprints: list[str] = field(default_factory=list)

    def add(self, call: LLMCall) -> None:
        self.calls += 1
        self.prompt_tokens += call.prompt_tokens
        self.completion_tokens += call.completion_tokens
        self.reasoning_tokens += call.reasoning_tokens
        self.cache_hit_tokens += call.cache_hit_tokens
        self.latency += call.latency
        if call.fingerprint and call.fingerprint not in self.fingerprints:
            self.fingerprints.append(call.fingerprint)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self, model: str) -> float | None:
        price = PRICES_USD_PER_M.get(model)
        if price is None:
            return None
        miss = self.prompt_tokens - self.cache_hit_tokens
        return (
            self.cache_hit_tokens * price["input_cache_hit"]
            + miss * price["input_cache_miss"]
            + self.completion_tokens * price["output"]
        ) / 1e6

    def to_dict(self, model: str) -> dict:
        return {**asdict(self), "total_tokens": self.total_tokens, "cost_usd_peak": self.cost_usd(model)}


def invoke(model: BaseChatModel, messages: list[BaseMessage]) -> LLMCall:
    start = time.monotonic()
    msg = model.invoke(messages)
    return _to_call(msg, time.monotonic() - start, model)


async def ainvoke(model: BaseChatModel, messages: list[BaseMessage]) -> LLMCall:
    start = time.monotonic()
    msg = await model.ainvoke(messages)
    return _to_call(msg, time.monotonic() - start, model)


def _to_call(msg: BaseMessage, latency: float, model: BaseChatModel) -> LLMCall:
    meta = msg.response_metadata or {}
    usage = meta.get("token_usage") or {}
    return LLMCall(
        content=msg.content if isinstance(msg.content, str) else str(msg.content),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
        cache_hit_tokens=usage.get("prompt_cache_hit_tokens", 0),
        latency=latency,
        model=meta.get("model_name", getattr(model, "model_name", "")),
        fingerprint=meta.get("system_fingerprint") or "",
        finish_reason=meta.get("finish_reason", ""),
    )
