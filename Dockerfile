FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 可选镜像源（国内网络用），留空则用官方源。例：
#   APT_MIRROR=mirrors.aliyun.com
#   PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple
ARG APT_MIRROR=
ARG PIP_MIRROR=

# git：克隆评测仓库、读取 commit hash（缓存 key 需要）
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY testpilot ./testpilot
RUN pip install ${PIP_MIRROR:+-i "$PIP_MIRROR"} -e .

COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "testpilot.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
