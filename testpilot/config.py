"""配置：全部从环境变量 / .env 读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-flash"
    # 取 API 默认值 1.0：评测要求每个设置跑多次看方差，测试时扩展实验也需要采样多样性。
    # 所有方法用同一温度，比较才公平。
    llm_temperature: float = 1.0
    # 输出上限，**包含思考 token**。DeepSeek 只认 max_tokens（不认 max_completion_tokens）
    llm_max_tokens: int = 32000
    # 思考模式：auto（不传，模型自适应决定是否思考）/ enabled / disabled
    llm_thinking: str = "auto"
    llm_timeout: float = 180.0  # 单次请求超时（秒）
    llm_max_retries: int = 3

    def __post_init__(self) -> None:
        if self.llm_thinking not in ("auto", "enabled", "disabled"):
            raise ValueError(f"LLM_THINKING 必须是 auto / enabled / disabled，而不是 {self.llm_thinking!r}")

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env")
        key = os.environ.get("LLM_API_KEY", "")
        if not key:
            raise RuntimeError("缺少 LLM_API_KEY：请在 .env 中设置（参考 .env.example）")
        env = os.environ.get
        return cls(
            llm_api_key=key,
            llm_base_url=env("LLM_BASE_URL", cls.llm_base_url),
            llm_model=env("LLM_MODEL", cls.llm_model),
            llm_temperature=float(env("LLM_TEMPERATURE", cls.llm_temperature)),
            llm_max_tokens=int(env("LLM_MAX_TOKENS", cls.llm_max_tokens)),
            llm_thinking=env("LLM_THINKING", cls.llm_thinking),
            llm_timeout=float(env("LLM_TIMEOUT", cls.llm_timeout)),
            llm_max_retries=int(env("LLM_MAX_RETRIES", cls.llm_max_retries)),
        )
