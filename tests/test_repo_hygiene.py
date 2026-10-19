"""仓库卫生：被提交的文件里不能有密钥。"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_env_example_has_no_secrets():
    example = ROOT / ".env.example"
    if not example.exists():
        pytest.skip("镜像里不包含 .env.example，只在仓库检出目录中检查")
    for line in example.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if "KEY" in key.upper() or "SECRET" in key.upper() or "TOKEN" in key.upper():
            assert value.strip() == "", f".env.example 中 {key} 应留空，真实值写在 .env"
