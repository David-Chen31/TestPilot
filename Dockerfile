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

# 先只用 pyproject 安装依赖（包目录用空壳占位），改代码时这一层可以复用缓存
COPY pyproject.toml README.md ./
RUN mkdir testpilot && touch testpilot/__init__.py     && pip install ${PIP_MIRROR:+-i "$PIP_MIRROR"} -e .

# 禁止容器内所有 Python 进程产生 core dump（PR_SET_DUMPABLE = 0，fork 出的子进程继承）。
# mutmut 用 CPU 时间上限杀超时的变异体（SIGXCPU，默认会 core dump）；Docker Desktop 的 WSL
# 会把每个 core dump 写到 Windows 的 %TEMP%\wsl-crashes（单个 0.1–1.9 GB），写满 C 盘并让 Docker
# 卡死。ulimit -c 0 拦不住（WSL 用管道收集，内核不检查 RLIMIT_CORE），所以在进程级关闭。
COPY docker/no_core_dump.pth /tmp/
RUN cp /tmp/no_core_dump.pth "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/"
COPY testpilot ./testpilot

COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "testpilot.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
