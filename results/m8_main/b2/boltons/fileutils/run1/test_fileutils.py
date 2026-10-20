import os
import sys
import stat
import errno
import shutil
import builtins

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
    replace,
    atomic_rename,
    _atomic_rename,
    FULL_PERMS,
    RW_PERMS,
)


# ---------------------------------------------------------------------------
# mkdir_p
# ---------------------------------------------------------------------------

def test_mkdir_p_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_existing_is_noop(tmp_path):
    target = tmp_path / "exists"
    target.mkdir()
    assert mkdir_p(str(target)) is None
    assert target.is_dir()


def test_mkdir_p_existing_file_raises(tmp_path):
    target = tmp_path / "afile"
    target.write_text("hi")
    with pytest.raises(OSError) as excinfo:
        mkdir_p(str(target))
    assert excinfo.value.errno == errno.EEXIST


# ---------------------------------------------------------------------------
# path_to_unicode
# ---------------------------------------------------------------------------

def test_path_to_unicode_str_passthrough():
    s = "some/path"
    assert path_to_unicode(s) is s


def test_path_to_unicode_bytes():
    b = b"some/bytes"
    result = path_to_unicode(b)
    assert result == "some/bytes"
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# FilePerms
# ---------------------------------------------------------------------------

def test_fileperms_basic():
    fp = FilePerms(user="rwx", group="rw", other="r")
    assert fp.user == "rwx"
    assert fp.group == "rw"
    assert fp.other == "r"
    assert int(fp) == 0o764


def test_fileperms_repr():
    fp = FilePerms("r", "r", "")
    assert repr(fp) == "FilePerms(user='r', group='r', other='')"


def test_fileperms_char_order_normalized():
    fp = FilePerms(user="xrw", group="wxr", other="rxw")
    assert fp.user == "rwx"
    assert fp.group == "rwx"
    assert fp.other == "rwx"


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


def test_fileperms_from_int_masks_full_perms():
    fp = FilePerms.from_int(0o777 | 0o777000)
    assert int(fp) == 0o777


def test_fileperms_from_path(tmp_path):
    p = tmp_path / "f"
    p.write_text("x")
    os.chmod(str(p), 0o640)
    fp = FilePerms.from_path(str(p))
    assert int(fp) == 0o640


def test_fileperms_invalid_chars():
    with pytest.raises(ValueError):
        FilePerms(user="rz")


def test_fileperms_int_value_raises_value_error():
    # Implementation converts the value with str() first, then validates
    # the characters; integer invalid chars lead to ValueError, not TypeError.
    with pytest.raises(ValueError):
        FilePerms(user=123)


def test_fileperms_set_same_value_noop():
    fp = FilePerms(user="r")
    fp.user = "r"
    assert fp.user == "r"
    assert int(fp) == 0o400


def test_fileperms_property_access_on_class():
    prop = FilePerms.user
    assert prop.attribute == "_user"


def test_fileperms_empty():
    fp = FilePerms()
    assert int(fp) == 0
    assert repr(fp) == "FilePerms(user='', group='', other='')"


# ---------------------------------------------------------------------------
# DummyFile
# ---------------------------------------------------------------------------

def test_dummyfile_basic():
    df = DummyFile("somepath")
    assert df.name == "somepath"
    assert df.mode == "r"
    assert df.fileno() == -1
    assert df.read() == ""
    assert df.readline() == ""
    assert df.readlines() == []
    assert df.tell() == 0
    df.seek()
    df.truncate()
    df.write("x")
    df.writelines(["a"])
    df.flush()
    assert df.closed is False
    df.close()
    assert df.closed is True


def test_dummyfile_closed_raises():
    df = DummyFile("p")
    df.close()
    # Methods taking no args
    for meth in ("flush", "read", "readline", "readlines",
                 "seek", "tell", "truncate"):
        with pytest.raises(ValueError):
            getattr(df, meth)()
    # Methods requiring arguments need to be called with args
    with pytest.raises(ValueError):
        df.write("x")
    with pytest.raises(ValueError):
        df.writelines(["a"])


def test_dummyfile_next():
    df = DummyFile("p")
    with pytest.raises(StopIteration):
        next(df)
    with pytest.raises(StopIteration):
        df.next()
    with pytest.raises(StopIteration):
        df.__next__()


def test_dummyfile_context():
    df = DummyFile("p")
    with df as f:
        assert f is None
    df.close()
    with pytest.raises(ValueError):
        with df:
            pass


def test_dummyfile_attributes():
    df = DummyFile("p", mode="w", buffering=0)
    assert df.errors is None
    assert df.isatty is False
    assert df.encoding is None
    assert df.newlines is None
    assert df.softspace == 0


# ---------------------------------------------------------------------------
# AtomicSaver / atomic_save
# ---------------------------------------------------------------------------

def test_atomic_save_basic(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("hello")
    assert dest.read_text() == "hello"
    assert not os.path.exists(str(dest) + ".part")


def test_atomic_save_binary(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as fo:
        fo.write(b"\x00\x01")
    assert dest.read_bytes() == b"\x00\x01"


def test_atomic_saver_overwrite_false_existing(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("orig")
    saver = AtomicSaver(str(dest), overwrite=False)
    with pytest.raises(OSError) as excinfo:
        saver.setup()
    assert excinfo.value.errno == errno.EEXIST


def test_atomic_saver_exception_removes_part(tmp_path):
    dest = tmp_path / "out.txt"
    part = str(dest) + ".part"
    with pytest.raises(ValueError):
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write("data")
            raise ValueError("boom")
    assert not os.path.exists(part)
    assert not dest.exists()


def test_atomic_saver_exception_keeps_part(tmp_path):
    dest = tmp_path / "out.txt"
    part = str(dest) + ".part"
    with pytest.raises(ValueError):
        with atomic_save(str(dest), text_mode=True, rm_part_on_exc=False) as fo:
            fo.write("data")
            raise ValueError("boom")
    assert os.path.exists(part)
    assert not dest.exists()


def test_atomic_saver_missing_dir_raises(tmp_path):
    dest = tmp_path / "nodir" / "out.txt"
    with pytest.raises(OSError):
        with atomic_save(str(dest)) as fo:
            fo.write(b"x")


def test_atomic_saver_overwrite_part(tmp_path):
    dest = tmp_path / "out.txt"
    part = str(dest) + ".part"
    with open(part, "w") as f:
        f.write("stale")
    with atomic_save(str(dest), text_mode=True, overwrite_part=True) as fo:
        fo.write("new")
    assert dest.read_text() == "new"


def test_atomic_saver_part_file_name(tmp_path):
    # Note: part files are opened via os.fdopen, whose .name attribute
    # is the integer file descriptor, not a path string. We verify the
    # actual behavior: part_path uses the custom filename.
    dest = tmp_path / "out.txt"
    saver = AtomicSaver(str(dest), text_mode=True, part_file="custom.part")
    assert os.path.basename(saver.part_path) == "custom.part"
    with atomic_save(str(dest), text_mode=True, part_file="custom.part") as fo:
        # fdopen's name is the fd (int)
        assert isinstance(fo.name, int)
        fo.write("x")
    assert dest.read_text() == "x"
    assert not os.path.exists(str(tmp_path / "custom.part"))


def test_atomic_saver_unknown_kwarg():
    with pytest.raises(TypeError):
        AtomicSaver("/tmp/foo", not_a_kwarg=1)


def test_atomic_saver_file_perms(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True, file_perms=0o600) as fo:
        fo.write("x")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o600


def test_atomic_saver_copies_existing_perms(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    os.chmod(str(dest), 0o640)
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("new")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o640
    assert dest.read_text() == "new"


def test_atomic_saver_setup_manual(tmp_path):
    dest = tmp_path / "out.txt"
    saver = AtomicSaver(str(dest), text_mode=True)
    saver.setup()
    saver.part_file.write("manual")
    saver.part_file.flush()
    saver.part_file.close()
    atomic_rename(saver.part_path, saver.dest_path, overwrite=True)
    assert dest.read_text() == "manual"


# ---------------------------------------------------------------------------
# replace / atomic_rename
# ---------------------------------------------------------------------------

def test_replace_moves_file(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_text("content")
    replace(str(src), str(dst))
    assert not src.exists()
    assert dst.read_text() == "content"


def test_atomic_rename_no_overwrite(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_text("a")
    atomic_rename(str(src), str(dst))
    assert dst.read_text() == "a"
    assert not src.exists()


def test_atomic_rename_overwrite(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_text("new")
    dst.write_text("old")
    atomic_rename(str(src), str(dst), overwrite=True)
    assert dst.read_text() == "new"
    assert not src.exists()


def test_atomic_rename_no_overwrite_existing_raises(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_text("new")
    dst.write_text("old")
    with pytest.raises(OSError):
        atomic_rename(str(src), str(dst))
    assert dst.read_text() == "old"


def test_atomic_rename_alias():
    assert _atomic_rename is atomic_rename


# ---------------------------------------------------------------------------
# iter_find_files
# ---------------------------------------------------------------------------

@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.py").write_text("c")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.txt").write_text("d")
    (sub / "e.py").write_text("e")
    subsub = sub / "deeper"
    subsub.mkdir()
    (subsub / "f.txt").write_text("f")
    return tmp_path


def test_iter_find_files_pattern(tree):
    found = sorted(os.path.basename(p) for p in iter_find_files(str(tree), "*.txt"))
    assert found == ["a.txt", "b.txt", "d.txt", "f.txt"]


def test_iter_find_files_list_pattern(tree):
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tree), ["*.txt", "*.py"]))
    assert sorted(found) == sorted(["a.txt", "b.txt", "c.py",
                                    "d.txt", "e.py", "f.txt"])


def test_iter_find_files_ignored(tree):
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tree), "*.txt", ignored="a.*"))
    assert "a.txt" not in found
    assert "b.txt" in found


def test_iter_find_files_ignored_list(tree):
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tree), "*.txt",
                                            ignored=["a.*", "b.*"]))
    assert "a.txt" not in found
    assert "b.txt" not in found
    assert "d.txt" in found


def test_iter_find_files_include_dirs(tree):
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tree), "*",
                                            include_dirs=True))
    assert "sub" in found
    assert "deeper" in found


def test_iter_find_files_max_depth(tree):
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tree), "*.txt", max_depth=1))
    assert found == ["a.txt", "b.txt", "d.txt"]


def test_iter_find_files_max_depth_zero(tree):
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tree), "*.txt", max_depth=0))
    assert found == ["a.txt", "b.txt"]


def test_iter_find_files_returns_generator(tree):
    gen = iter_find_files(str(tree), "*.txt")
    assert hasattr(gen, "__iter__")
    results = list(gen)
    assert len(results) == 4


# ---------------------------------------------------------------------------
# copy_tree
# ---------------------------------------------------------------------------

def test_copy_tree_basic(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")
    (src / "sub").mkdir()
    (src / "sub" / "b.txt").write_text("b")
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))
    assert (dst / "a.txt").read_text() == "a"
    assert (dst / "sub" / "b.txt").read_text() == "b"


def test_copy_tree_into_existing(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "existing.txt").write_text("existing")
    copy_tree(str(src), str(dst))
    assert (dst / "a.txt").read_text() == "a"
    assert (dst / "existing.txt").read_text() == "existing"


def test_copy_tree_with_ignore(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("k")
    (src / "skip.txt").write_text("s")
    dst = tmp_path / "dst"

    def ignore_func(path, names):
        return [n for n in names if n == "skip.txt"]

    copy_tree(str(src), str(dst), ignore=ignore_func)
    assert (dst / "keep.txt").exists()
    assert not (dst / "skip.txt").exists()


def test_copy_tree_symlinks(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    target = src / "real.txt"
    target.write_text("real")
    link = src / "link.txt"
    os.symlink(str(target), str(link))
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst), symlinks=True)
    assert os.path.islink(str(dst / "link.txt"))


def test_copytree_alias():
    assert copytree is copy_tree


# ---------------------------------------------------------------------------
# rotate_file
# ---------------------------------------------------------------------------

def test_rotate_file_basic(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("v1")
    rotate_file(str(f))
    assert not f.exists()
    assert (tmp_path / "log.1.txt").read_text() == "v1"


def test_rotate_file_existing_rotation(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("v2")
    (tmp_path / "log.1.txt").write_text("v1")
    rotate_file(str(f))
    assert (tmp_path / "log.1.txt").read_text() == "v2"
    assert (tmp_path / "log.2.txt").read_text() == "v1"


def test_rotate_file_keep_drops_oldest(tmp_path):
    # With keep=2, log.2.txt is created as the last kept name, and then
    # removed by the final os.remove() call. So only log.1.txt survives.
    f = tmp_path / "log.txt"
    f.write_text("v3")
    (tmp_path / "log.1.txt").write_text("v2")
    (tmp_path / "log.2.txt").write_text("v1")
    rotate_file(str(f), keep=2)
    assert (tmp_path / "log.1.txt").read_text() == "v3"
    assert not (tmp_path / "log.2.txt").exists()


def test_rotate_file_no_ext(tmp_path):
    f = tmp_path / "log"
    f.write_text("v1")
    rotate_file(str(f))
    assert (tmp_path / "log.1").read_text() == "v1"


def test_rotate_file_nonexistent(tmp_path):
    assert rotate_file(str(tmp_path / "missing.txt")) is None


def test_rotate_file_keep_invalid(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        rotate_file(str(f), keep=0)


def test_rotate_file_not_a_file(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d))


def test_rotate_file_multiple_keeps(tmp_path):
    # With keep=3, log.3.txt ends up as the "last kept" name and is removed.
    f = tmp_path / "log.txt"
    f.write_text("v4")
    (tmp_path / "log.1.txt").write_text("v3")
    (tmp_path / "log.2.txt").write_text("v2")
    (tmp_path / "log.3.txt").write_text("v1")
    rotate_file(str(f), keep=3)
    assert (tmp_path / "log.1.txt").read_text() == "v4"
    assert (tmp_path / "log.2.txt").read_text() == "v3"
    assert not (tmp_path / "log.3.txt").exists()
    assert not (tmp_path / "log.4.txt").exists()
