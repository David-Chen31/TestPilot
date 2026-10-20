import errno
import os
import stat
import sys

import pytest

from boltons.fileutils import (
    mkdir_p,
    FilePerms,
    atomic_save,
    path_to_unicode,
    AtomicSaver,
    iter_find_files,
    copy_tree,
    copytree,
    DummyFile,
    rotate_file,
)


# ---------------------------------------------------------------------------
# mkdir_p
# ---------------------------------------------------------------------------

def test_mkdir_p_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_existing_is_noop(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    assert mkdir_p(str(target)) is None
    assert target.is_dir()


def test_mkdir_p_raises_on_file(tmp_path):
    target = tmp_path / "afile"
    target.write_text("x")
    with pytest.raises(OSError):
        mkdir_p(str(target))


def test_mkdir_p_raises_when_parent_is_file(tmp_path):
    parent = tmp_path / "parent"
    parent.write_text("x")
    with pytest.raises(OSError):
        mkdir_p(str(parent / "child"))


# ---------------------------------------------------------------------------
# FilePerms
# ---------------------------------------------------------------------------

def test_fileperms_basic_rwx():
    fp = FilePerms(user="rwx", group="rwx", other="rwx")
    assert fp.user == "rwx"
    assert fp.group == "rwx"
    assert fp.other == "rwx"
    assert int(fp) == 0o777


def test_fileperms_character_order_normalized():
    fp = FilePerms(user="xrw", group="xwr", other="wxr")
    assert fp.user == "rwx"
    assert fp.group == "rwx"
    assert fp.other == "rwx"


def test_fileperms_individual():
    fp = FilePerms("r", "r", "")
    assert int(fp) == 0o440
    assert oct(int(fp)) == "0o440"


def test_fileperms_empty():
    fp = FilePerms()
    assert int(fp) == 0
    assert fp.user == ""
    assert fp.group == ""
    assert fp.other == ""


def test_fileperms_from_int():
    fp = FilePerms.from_int(0o644)
    assert fp.user == "rw"
    assert fp.group == "r"
    assert fp.other == "r"
    assert int(fp) == 0o644


def test_fileperms_from_int_zero():
    fp = FilePerms.from_int(0)
    assert int(fp) == 0
    assert fp.user == ""
    assert fp.group == ""
    assert fp.other == ""


def test_fileperms_from_int_masks_high_bits():
    fp = FilePerms.from_int(0o777777)
    assert int(fp) == 0o777


def test_fileperms_from_path(tmp_path):
    p = tmp_path / "f"
    p.write_text("x")
    os.chmod(str(p), 0o640)
    fp = FilePerms.from_path(str(p))
    assert int(fp) == 0o640
    assert fp.user == "rw"
    assert fp.group == "r"
    assert fp.other == ""


def test_fileperms_setter_updates_integer():
    fp = FilePerms()
    fp.user = "r"
    assert fp.user == "r"
    assert int(fp) == 0o400
    fp.group = "w"
    assert int(fp) == 0o420
    fp.other = "x"
    assert int(fp) == 0o421


def test_fileperms_setter_same_value_noop():
    fp = FilePerms(user="r")
    before = int(fp)
    fp.user = "r"
    assert int(fp) == before


def test_fileperms_invalid_char():
    with pytest.raises(ValueError):
        FilePerms(user="rz")


def test_fileperms_non_string():
    fp = FilePerms()
    with pytest.raises(TypeError):
        fp.user = 123


def test_fileperms_repr():
    fp = FilePerms(user="rwx", group="r", other="")
    assert repr(fp) == "FilePerms(user='rwx', group='r', other='')"


def test_fileperms_roundtrip_int():
    fp = FilePerms.from_int(0o751)
    assert int(fp) == 0o751


# ---------------------------------------------------------------------------
# atomic_save / AtomicSaver
# ---------------------------------------------------------------------------

def test_atomic_save_binary(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as fo:
        fo.write(b"hello")
    assert dest.read_bytes() == b"hello"
    assert not (tmp_path / "out.bin.part").exists()


def test_atomic_save_text(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("hello")
    assert dest.read_text() == "hello"


def test_atomic_save_no_overwrite_existing(tmp_path):
    dest = tmp_path / "exists.txt"
    dest.write_text("orig")
    with pytest.raises(OSError) as exc_info:
        with atomic_save(str(dest), overwrite=False) as fo:
            fo.write(b"new")
    assert exc_info.value.errno == errno.EEXIST
    assert dest.read_text() == "orig"


def test_atomic_save_exception_removes_part(tmp_path):
    dest = tmp_path / "err.bin"
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest)) as fo:
            fo.write(b"bye")
            1 / 0
    assert not dest.exists()
    assert not (tmp_path / "err.bin.part").exists()


def test_atomic_save_exception_keeps_part(tmp_path):
    dest = tmp_path / "err2.bin"
    with pytest.raises(RuntimeError):
        with atomic_save(str(dest), rm_part_on_exc=False) as fo:
            fo.write(b"data")
            raise RuntimeError("boom")
    assert not dest.exists()
    part = tmp_path / "err2.bin.part"
    assert part.exists()
    assert part.read_bytes() == b"data"


def test_atomic_save_overwrites_existing(tmp_path):
    dest = tmp_path / "overwrite.bin"
    dest.write_bytes(b"old")
    with atomic_save(str(dest)) as fo:
        fo.write(b"new")
    assert dest.read_bytes() == b"new"


def test_atomic_save_preserves_perms(tmp_path):
    dest = tmp_path / "perms.bin"
    dest.write_bytes(b"old")
    os.chmod(str(dest), 0o600)
    with atomic_save(str(dest)) as fo:
        fo.write(b"new")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o600


def test_atomic_save_custom_file_perms(tmp_path):
    dest = tmp_path / "custom.bin"
    with atomic_save(str(dest), file_perms=0o640) as fo:
        fo.write(b"x")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o640


def test_atomic_saver_unknown_kwarg(tmp_path):
    with pytest.raises(TypeError):
        AtomicSaver(str(tmp_path / "f"), bogus=1)


def test_atomic_saver_part_file_conflict(tmp_path):
    dest = tmp_path / "conflict.bin"
    part = tmp_path / "conflict.bin.part"
    part.write_bytes(b"existing")
    with pytest.raises(OSError):
        with atomic_save(str(dest)) as fo:
            fo.write(b"x")


def test_atomic_saver_overwrite_part(tmp_path):
    dest = tmp_path / "owp.bin"
    part = tmp_path / "owp.bin.part"
    part.write_bytes(b"existing")
    with atomic_save(str(dest), overwrite_part=True) as fo:
        fo.write(b"fresh")
    assert dest.read_bytes() == b"fresh"
    assert not part.exists()


def test_atomic_saver_custom_part_file(tmp_path):
    dest = tmp_path / "custom_part.bin"
    with atomic_save(str(dest), part_file="mypart.tmp") as fo:
        fo.write(b"data")
        assert (tmp_path / "mypart.tmp").exists()
    assert dest.read_bytes() == b"data"
    assert not (tmp_path / "mypart.tmp").exists()


def test_atomic_saver_explicit_setup(tmp_path):
    dest = tmp_path / "manual.bin"
    saver = AtomicSaver(str(dest))
    saver.setup()
    saver.part_file.write(b"manual")
    saver.part_file.flush()
    saver.part_file.close()
    from boltons.fileutils import atomic_rename
    atomic_rename(saver.part_path, saver.dest_path, overwrite=True)
    assert dest.read_bytes() == b"manual"


def test_atomic_saver_accepts_pathlike(tmp_path):
    dest = tmp_path / "pathlike.bin"
    with atomic_save(dest) as fo:
        fo.write(b"pl")
    assert dest.read_bytes() == b"pl"


# ---------------------------------------------------------------------------
# path_to_unicode
# ---------------------------------------------------------------------------

def test_path_to_unicode_str():
    assert path_to_unicode("hello") == "hello"


def test_path_to_unicode_bytes():
    res = path_to_unicode(b"hello")
    assert res == "hello"
    assert isinstance(res, str)


# ---------------------------------------------------------------------------
# iter_find_files
# ---------------------------------------------------------------------------

def _make_tree(root):
    (root / "a.txt").write_text("a")
    (root / "b.py").write_text("b")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    (sub / "d.py").write_text("d")
    deep = sub / "deep"
    deep.mkdir()
    (deep / "e.txt").write_text("e")


def test_iter_find_files_single_pattern(tmp_path):
    _make_tree(tmp_path)
    res = sorted(os.path.basename(p) for p in iter_find_files(str(tmp_path), "*.txt"))
    assert res == ["a.txt", "c.txt", "e.txt"]


def test_iter_find_files_multiple_patterns(tmp_path):
    _make_tree(tmp_path)
    res = sorted(os.path.basename(p) for p in
                 iter_find_files(str(tmp_path), ["*.txt", "*.py"]))
    assert res == ["a.txt", "b.py", "c.txt", "d.py", "e.txt"]


def test_iter_find_files_ignored(tmp_path):
    _make_tree(tmp_path)
    res = sorted(os.path.basename(p) for p in
                 iter_find_files(str(tmp_path), "*.txt", ignored="a*"))
    assert res == ["c.txt", "e.txt"]


def test_iter_find_files_ignored_list(tmp_path):
    _make_tree(tmp_path)
    res = sorted(os.path.basename(p) for p in
                 iter_find_files(str(tmp_path), "*", ignored=["*.py", "a.txt"]))
    assert res == ["c.txt", "e.txt"]


def test_iter_find_files_include_dirs(tmp_path):
    _make_tree(tmp_path)
    res = sorted(os.path.basename(p) for p in
                 iter_find_files(str(tmp_path), "sub", include_dirs=True))
    assert res == ["sub"]


def test_iter_find_files_dirs_excluded_by_default(tmp_path):
    _make_tree(tmp_path)
    res = list(iter_find_files(str(tmp_path), "sub"))
    assert res == []


def test_iter_find_files_max_depth_zero(tmp_path):
    _make_tree(tmp_path)
    res = sorted(os.path.basename(p) for p in
                 iter_find_files(str(tmp_path), "*", max_depth=0))
    assert res == ["a.txt", "b.py"]


def test_iter_find_files_max_depth_one(tmp_path):
    _make_tree(tmp_path)
    res = sorted(os.path.basename(p) for p in
                 iter_find_files(str(tmp_path), "*.txt", max_depth=1))
    assert res == ["a.txt", "c.txt"]


def test_iter_find_files_accepted_pathlike(tmp_path):
    _make_tree(tmp_path)
    res = sorted(os.path.basename(p) for p in iter_find_files(tmp_path, "*.txt"))
    assert res == ["a.txt", "c.txt", "e.txt"]


# ---------------------------------------------------------------------------
# copy_tree / copytree
# ---------------------------------------------------------------------------

def test_copy_tree_basic(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "one.txt").write_text("1")
    (src / "two.txt").write_text("2")
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))
    assert (dst / "one.txt").read_text() == "1"
    assert (dst / "two.txt").read_text() == "2"


def test_copy_tree_nested(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    sub = src / "sub"
    sub.mkdir()
    (sub / "n.txt").write_text("nested")
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))
    assert (dst / "sub" / "n.txt").read_text() == "nested"


def test_copy_tree_existing_dst_ok(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("content")
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "existing.txt").write_text("stay")
    copy_tree(str(src), str(dst))
    assert (dst / "f.txt").read_text() == "content"
    assert (dst / "existing.txt").read_text() == "stay"


def test_copy_tree_ignore(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("k")
    (src / "skip.txt").write_text("s")
    dst = tmp_path / "dst"

    def ignore(path, names):
        return {"skip.txt"}

    copy_tree(str(src), str(dst), ignore=ignore)
    assert (dst / "keep.txt").exists()
    assert not (dst / "skip.txt").exists()


def test_copy_tree_symlinks(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    target = tmp_path / "targetfile"
    target.write_text("orig")
    os.symlink(str(target), str(src / "link"))
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst), symlinks=True)
    assert os.path.islink(str(dst / "link"))


def test_copytree_alias():
    assert copytree is copy_tree


# ---------------------------------------------------------------------------
# DummyFile
# ---------------------------------------------------------------------------

def test_dummyfile_basic():
    d = DummyFile("/dev/null")
    assert d.name == "/dev/null"
    assert d.mode == "r"
    assert d.closed is False
    assert d.fileno() == -1
    assert d.read() == ""
    assert d.readline() == ""
    assert d.readlines() == []
    assert d.tell() == 0
    assert d.seek() is None
    assert d.truncate() is None
    assert d.write("x") is None
    assert d.writelines(["a"]) is None
    assert d.flush() is None


def test_dummyfile_close_and_errors():
    d = DummyFile("/dev/null")
    d.close()
    assert d.closed is True
    with pytest.raises(ValueError):
        d.flush()
    with pytest.raises(ValueError):
        d.read()
    with pytest.raises(ValueError):
        d.write("x")
    with pytest.raises(ValueError):
        d.seek()
    with pytest.raises(ValueError):
        d.tell()
    with pytest.raises(ValueError):
        d.truncate()
    with pytest.raises(ValueError):
        d.readlines()


def test_dummyfile_context_manager():
    d = DummyFile("/dev/null")
    assert d.__enter__() is None
    assert d.__exit__(None, None, None) is None
    d.close()
    with pytest.raises(ValueError):
        d.__enter__()


def test_dummyfile_next():
    d = DummyFile("/dev/null")
    with pytest.raises(StopIteration):
        next(d)
    with pytest.raises(StopIteration):
        d.next()


# ---------------------------------------------------------------------------
# rotate_file
# ---------------------------------------------------------------------------

def test_rotate_file_no_existing(tmp_path):
    target = tmp_path / "log.txt"
    assert rotate_file(str(target)) is None
    assert not target.exists()


def test_rotate_file_basic(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("current")
    rotate_file(str(base), keep=3)
    assert not base.exists()
    assert (tmp_path / "log.1.txt").read_text() == "current"


def test_rotate_file_chain(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("v0")
    rotate_file(str(base), keep=3)
    (tmp_path / "log.txt").write_text("v1")
    rotate_file(str(base), keep=3)
    (tmp_path / "log.txt").write_text("v2")
    rotate_file(str(base), keep=3)
    assert (tmp_path / "log.1.txt").read_text() == "v2"
    assert (tmp_path / "log.2.txt").read_text() == "v1"
    assert (tmp_path / "log.3.txt").read_text() == "v0"


def test_rotate_file_keep_limit_drops_old(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("v0")
    rotate_file(str(base), keep=2)
    (tmp_path / "log.txt").write_text("v1")
    rotate_file(str(base), keep=2)
    (tmp_path / "log.txt").write_text("v2")
    rotate_file(str(base), keep=2)
    assert (tmp_path / "log.1.txt").read_text() == "v2"
    assert (tmp_path / "log.2.txt").read_text() == "v1"
    assert not (tmp_path / "log.3.txt").exists()


def test_rotate_file_no_extension(tmp_path):
    base = tmp_path / "plainlog"
    base.write_text("data")
    rotate_file(str(base), keep=2)
    assert not base.exists()
    assert (tmp_path / "plainlog.1").read_text() == "data"


def test_rotate_file_keep_invalid(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("x")
    with pytest.raises(ValueError):
        rotate_file(str(base), keep=0)


def test_rotate_file_not_a_file(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d))
