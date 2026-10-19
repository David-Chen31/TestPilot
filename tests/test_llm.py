from dataclasses import replace

import pytest
from langchain_core.messages import HumanMessage

from testpilot.config import Settings
from testpilot.llm import LLMCall, Usage, make_chat_model

BASE = Settings(llm_api_key="sk-test", llm_max_tokens=1234)


def _payload(settings: Settings) -> dict:
    payload = make_chat_model(settings)._get_request_payload([HumanMessage("x")])
    payload.pop("messages")
    return payload


def test_max_tokens_sent_under_name_deepseek_honors():
    # DeepSeek 忽略 max_completion_tokens（langchain 默认用的名字），必须发 max_tokens
    p = _payload(BASE)
    assert p["extra_body"]["max_tokens"] == 1234
    assert "max_completion_tokens" not in p


@pytest.mark.parametrize("mode", ["enabled", "disabled"])
def test_thinking_mode_sent(mode):
    assert _payload(replace(BASE, llm_thinking=mode))["extra_body"]["thinking"] == {"type": mode}


def test_thinking_auto_sends_nothing():
    assert "thinking" not in _payload(replace(BASE, llm_thinking="auto"))["extra_body"]


def test_thinking_disabled_by_default():
    assert Settings(llm_api_key="k").llm_thinking == "disabled"


def test_invalid_thinking_mode():
    with pytest.raises(ValueError):
        replace(BASE, llm_thinking="maybe")


def test_usage_cost_peak_price():
    u = Usage()
    u.add(LLMCall("", 1_000_000, 1_000_000, 400_000, 200_000, 1.0, "deepseek-flash", "fp", "stop"))
    # 80 万未命中缓存输入 × 0.3 + 20 万命中 × 0.006 + 100 万输出 × 1.2（美元 / 百万 token）
    assert u.cost_usd("deepseek-flash") == pytest.approx(0.8 * 0.3 + 0.2 * 0.006 + 1.2)
    assert u.reasoning_tokens == 400_000
    assert u.cost_usd("unknown-model") is None
