"""评测仓库：读取清单、按固定 commit 获取源码快照、准备去掉自带测试的工作副本。

获取方式是下载 GitHub 的 commit 快照（codeload tar.gz），而不是 git clone：
commit 写在 URL 里，内容是确定的；保留上游原始换行符（Windows 上 clone 会受 autocrlf 影响）；
不依赖 git 和 github.com 的可达性。快照不含 .gitattributes 里标记 export-ignore 的文件。
"""

from __future__ import annotations

import re
import shutil
import tarfile
import tempfile
import tomllib
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from testpilot.config import PROJECT_ROOT
from testpilot.sandbox.workspace import Workspace

MANIFEST = Path(__file__).with_name("repos.toml")
DATA_DIR = PROJECT_ROOT / "data" / "repos"
COMMIT_MARKER = ".testpilot-commit"
_GITHUB = re.compile(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")  # 仓库名可以带点（jmespath.py）


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    commit: str
    python_paths: tuple[str, ...] = (".",)
    remove: tuple[str, ...] = ()
    targets: tuple[str, ...] = field(default_factory=tuple)
    group: str = "old"  # old：可能在训练数据里的流行库；new：晚于训练截止创建的仓库（污染对照）
    tests: str = ""  # 仓库自带测试的位置（预检时用作参照，评测时删除）
    stub_files: tuple[tuple[str, str], ...] = ()  # 构建时才生成、源码快照里没有的文件：(路径, 内容)

    @property
    def path(self) -> Path:
        return DATA_DIR / self.name

    @property
    def tarball_url(self) -> str:
        m = _GITHUB.match(self.url)
        if not m:
            raise ValueError(f"只支持 GitHub 仓库: {self.url}")
        return f"https://codeload.github.com/{m.group(1)}/{m.group(2)}/tar.gz/{self.commit}"


def load_manifest(path: Path = MANIFEST) -> dict[str, RepoSpec]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        name: RepoSpec(
            name=name,
            url=cfg["url"],
            commit=cfg["commit"],
            python_paths=tuple(cfg.get("python_paths", ["."])),
            remove=tuple(cfg.get("remove", [])),
            targets=tuple(cfg.get("targets", [])),
            group=cfg.get("group", "old"),
            tests=cfg.get("tests", ""),
            stub_files=tuple(cfg.get("stub_files", {}).items()),
        )
        for name, cfg in raw.items()
    }


def snapshot_commit(path: Path) -> str | None:
    marker = path / COMMIT_MARKER
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else None


def ensure_repo(spec: RepoSpec) -> Path:
    """下载（如需要）固定 commit 的源码快照，返回本地路径。"""
    if snapshot_commit(spec.path) == spec.commit:
        return spec.path
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "src.tar.gz"
        with urllib.request.urlopen(spec.tarball_url, timeout=120) as resp, archive.open("wb") as f:
            shutil.copyfileobj(resp, f)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")  # filter="data"：拒绝绝对路径、../ 和设备文件
        [top] = [p for p in Path(tmp).iterdir() if p.is_dir()]  # 快照的顶层目录 <repo>-<commit>/
        if spec.path.exists():
            shutil.rmtree(spec.path)
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(top), spec.path)
    (spec.path / COMMIT_MARKER).write_text(spec.commit + "\n", encoding="utf-8")
    return spec.path


@contextmanager
def eval_workspace(spec: RepoSpec, keep: bool = False, keep_tests: bool = False) -> Iterator[Workspace]:
    """固定 commit 的工作副本，已删除清单中的自带测试（``keep_tests`` 时保留，预检用）。"""
    source = ensure_repo(spec)
    with Workspace(source, keep=keep) as ws:
        for rel, content in spec.stub_files:
            ws.write_text(rel, content)
        for rel in spec.remove:
            if keep_tests and rel == spec.tests:
                continue
            target = ws.resolve(rel)
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        yield ws
