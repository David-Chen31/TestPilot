from pathlib import Path

import pytest

from testpilot.eval import repos
from testpilot.eval.repos import COMMIT_MARKER, RepoSpec, eval_workspace, load_manifest


def test_manifest_pins_commits():
    manifest = load_manifest()
    assert "toolz" in manifest
    for spec in manifest.values():
        assert len(spec.commit) == 40  # 完整 commit hash，不用分支名
        assert spec.targets


def test_tarball_url():
    spec = RepoSpec("toolz", "https://github.com/pytoolz/toolz.git", "a" * 40)
    assert spec.tarball_url == f"https://codeload.github.com/pytoolz/toolz/tar.gz/{'a' * 40}"
    dotted = RepoSpec("jmespath", "https://github.com/jmespath/jmespath.py.git", "a" * 40)
    assert dotted.tarball_url == f"https://codeload.github.com/jmespath/jmespath.py/tar.gz/{'a' * 40}"
    with pytest.raises(ValueError):
        _ = RepoSpec("x", "https://gitlab.com/a/b", "a" * 40).tarball_url


def test_eval_workspace_removes_bundled_tests(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(repos, "DATA_DIR", tmp_path / "repos")
    spec = RepoSpec("demo", "https://github.com/x/demo", "c" * 40, remove=("pkg/tests", "bench"))
    # 已有快照（标记的 commit 一致）时不会下载
    (spec.path / "pkg" / "tests").mkdir(parents=True)
    (spec.path / "pkg" / "mod.py").write_text("X = 1\n")
    (spec.path / "pkg" / "tests" / "test_mod.py").write_text("def test(): pass\n")
    (spec.path / COMMIT_MARKER).write_text("c" * 40 + "\n")

    with eval_workspace(spec) as ws:
        assert (ws.path / "pkg" / "mod.py").exists()
        assert not (ws.path / "pkg" / "tests").exists()
    assert (spec.path / "pkg" / "tests" / "test_mod.py").exists()  # 原快照不受影响


def test_keep_tests_and_stub_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(repos, "DATA_DIR", tmp_path / "repos")
    spec = RepoSpec(
        "demo", "https://github.com/x/demo", "c" * 40, tests="pkg/tests", remove=("pkg/tests", "bench"),
        stub_files=(("pkg/_version.py", "__version__ = '0'\n"),),
    )  # fmt: skip
    (spec.path / "pkg" / "tests").mkdir(parents=True)
    (spec.path / "bench").mkdir()
    (spec.path / COMMIT_MARKER).write_text("c" * 40 + "\n")
    with eval_workspace(spec, keep_tests=True) as ws:
        assert (ws.path / "pkg" / "tests").exists()  # 预检保留自带测试
        assert not (ws.path / "bench").exists()  # 其他要删的照样删
        assert ws.read_text("pkg/_version.py") == "__version__ = '0'\n"
    assert not (spec.path / "pkg" / "_version.py").exists()  # 只写进工作副本


def test_manifest_targets_have_distinct_result_dirs():
    # 结果目录按 <repo>/<文件名去掉 .py> 命名，同一仓库内不能重名
    for spec in load_manifest().values():
        stems = [Path(t).stem for t in spec.targets]
        assert len(stems) == len(set(stems)), spec.name
        assert spec.group in ("old", "new")
