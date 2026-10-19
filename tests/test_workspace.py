from pathlib import Path

import pytest

from testpilot.sandbox.workspace import Workspace


def _make_repo(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / "pkg" / "__pycache__").mkdir()
    (root / "pkg" / "__pycache__" / "mod.cpython-311.pyc").write_bytes(b"\0")
    return root


def test_copy_excludes_vcs_and_caches(tmp_path: Path):
    repo = _make_repo(tmp_path / "repo")
    with Workspace(repo, base_dir=tmp_path) as ws:
        assert (ws.path / "pkg" / "mod.py").exists()
        assert not (ws.path / ".git").exists()
        assert not (ws.path / "pkg" / "__pycache__").exists()


def test_writes_do_not_touch_source(tmp_path: Path):
    repo = _make_repo(tmp_path / "repo")
    with Workspace(repo, base_dir=tmp_path) as ws:
        ws.write_text("pkg/mod.py", "X = 2\n")
        ws.write_text("tests/test_new.py", "def test_x(): pass\n")
        assert ws.read_text("pkg/mod.py") == "X = 2\n"
    assert (repo / "pkg" / "mod.py").read_text(encoding="utf-8") == "X = 1\n"
    assert not (repo / "tests").exists()


def test_cleanup_removes_temp_dir(tmp_path: Path):
    repo = _make_repo(tmp_path / "repo")
    with Workspace(repo, base_dir=tmp_path) as ws:
        copy = ws.path
    assert not copy.exists()


def test_keep_preserves_temp_dir(tmp_path: Path):
    repo = _make_repo(tmp_path / "repo")
    with Workspace(repo, base_dir=tmp_path, keep=True) as ws:
        copy = ws.path
    assert copy.exists()


@pytest.mark.parametrize("bad", ["../outside.py", "pkg/../../outside.py"])
def test_resolve_rejects_escape(tmp_path: Path, bad: str):
    repo = _make_repo(tmp_path / "repo")
    with Workspace(repo, base_dir=tmp_path) as ws:
        with pytest.raises(ValueError):
            ws.resolve(bad)


def test_resolve_rejects_absolute_path(tmp_path: Path):
    repo = _make_repo(tmp_path / "repo")
    with Workspace(repo, base_dir=tmp_path) as ws:
        with pytest.raises(ValueError):
            ws.resolve(tmp_path / "elsewhere.py")


def test_source_must_be_directory(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        Workspace(tmp_path / "missing")
