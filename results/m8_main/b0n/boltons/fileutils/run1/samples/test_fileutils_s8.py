import os
import sys
import errno
import stat
import shutil

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


# ------------------------- mkdir_p -------------------------

def test_mkdir_p_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_existing_is_ok(tmp_path):
    target = tmp_path / "a" / "b"
    mkdir_p(str(target))
    assert target.is_dir()
    # calling again should not raise
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_existing_file_raises(tmp_path):
    p = tmp_path / "afile"
    p.write_text("x")
    with pytest.raises(OSError):
        mkdir_p(str(p))


# ------------------------- FilePerms -------------------------

def test_fileperms_basic_construction():
    fp = FilePerms(user="rwx", group="xrw", other="wxr")
    assert fp.user == "rwx"
    assert fp.group == "rwx"
    assert fp.other == "rwx"


def test_fileperms_int():
    fp = FilePerms("r", "r", "")
    assert int(fp) == 288
    assert oct(288)[-3:] == "440"


def test_fileperms_empty():
    fp = FilePerms()
    assert fp.user == ""
    assert fp.group == ""
    assert fp.other == ""
    assert int(fp) == 0


def test_fileperms_from_int():
    fp = FilePerms.from_int(0o644)
    assert fp.user == "rw"
    assert fp.group == "r"
    assert fp.other == "r"
    assert int(fp) == 0o644


def test_fileperms_from_int_zero():
    fp = FilePerms.from_int(0)
    assert fp.user == ""
    assert fp.group == ""
    assert fp.other == ""
    assert int(fp) == 0


def test_fileperms_repr():
    fp = FilePerms("rw", "r", "")
    assert repr(fp) == "FilePerms(user='rw', group='r', other='')"


def test_fileperms_from_path(tmp_path):
    target = tmp_path / "f"
    target.write_text("x")
    os.chmod(str(target), 0o640)
    fp = FilePerms.from_path(str(target))
    assert fp.user == "rw"
    assert fp.group == "r"
    assert fp.other == ""


def test_fileperms_invalid_chars():
    with pytest.raises(ValueError):
        FilePerms(user="rz")


def test_fileperms_setter_sorts_and_updates_int():
    fp = FilePerms()
    fp.user = "xrw"
    assert fp.user == "rwx"
    assert int(fp) == (0o7 << 6)


def test_fileperms_setter_same_value_noop():
    fp = FilePerms(user="rw")
    fp.user = "rw"
    assert fp.user == "rw"


def test_fileperms_setter_replacement_updates_int():
    fp = FilePerms(user="rwx")
    fp.user = "r"
    assert fp.user == "r"
    # other fields stay untouched; group/other are empty
    assert int(fp) == (0o4 << 6)


# ------------------------- path_to_unicode -------------------------

def test_path_to_unicode_str():
    assert path_to_unicode("abc") == "abc"


def test_path_to_unicode_bytes():
    enc = sys.getfilesystemencoding() or sys.getdefaultencoding()
    b = "héllo".encode(enc)
    assert path_to_unicode(b) == "héllo"


# ------------------------- atomic_save / AtomicSaver -------------------------

def test_atomic_save_basic(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("hello")
    assert dest.read_text() == "hello"


def test_atomic_save_no_part_left(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("data")
    assert not os.path.exists(str(dest) + ".part")


def test_atomic_save_exception_leaves_original(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("original")
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write("new")
            1 / 0
    assert dest.read_text() == "original"
    assert not os.path.exists(str(dest) + ".part")


def test_atomic_save_exception_rm_part_false(tmp_path):
    dest = tmp_path / "out.txt"
    with pytest.raises(ValueError):
        with atomic_save(str(dest), text_mode=True, rm_part_on_exc=False) as fo:
            fo.write("partial")
            raise ValueError("boom")
    assert os.path.exists(str(dest) + ".part")


def test_atomic_saver_custom_part_file(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True, part_file="mypart") as fo:
        fo.write("hi")
    assert dest.read_text() == "hi"
    assert not os.path.exists(str(tmp_path / "mypart"))


def test_atomic_saver_overwrite_false_existing(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("original")
    saver = AtomicSaver(str(dest), overwrite=False, text_mode=True)
    with pytest.raises(OSError) as excinfo:
        saver.setup()
    assert excinfo.value.errno == errno.EEXIST
    assert dest.read_text() == "original"


def test_atomic_saver_preexisting_part_raises(tmp_path):
    dest = tmp_path / "out.txt"
    part = str(dest) + ".part"
    with open(part, "w") as f:
        f.write("stale")
    with pytest.raises(OSError):
        with atomic_save(str(dest), text_mode=True):
            pass


def test_atomic_saver_overwrite_part_true(tmp_path):
    dest = tmp_path / "out.txt"
    part = str(dest) + ".part"
    with open(part, "w") as f:
        f.write("stale")
    with atomic_save(str(dest), text_mode=True, overwrite_part=True) as fo:
        fo.write("fresh")
    assert dest.read_text() == "fresh"


def test_atomic_saver_bad_kwarg(tmp_path):
    dest = tmp_path / "out.txt"
    with pytest.raises(TypeError):
        AtomicSaver(str(dest), not_a_kwarg=1)


def test_atomic_saver_explicit_file_perms(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True, file_perms=0o600) as fo:
        fo.write("x")
    mode = stat.S_IMODE(os.stat(str(dest)).st_mode)
    assert mode == 0o600


def test_atomic_saver_copies_existing_perms(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("orig")
    os.chmod(str(dest), 0o640)
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("new")
    mode = stat.S_IMODE(os.stat(str(dest)).st_mode)
    assert mode == 0o640


def test_atomic_saver_binary_mode(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as fo:
        fo.write(b"\x00\x01\x02")
    assert dest.read_bytes() == b"\x00\x01\x02"


def test_atomic_saver_default_part_path(tmp_path):
    dest = tmp_path / "out.txt"
    saver = AtomicSaver(str(dest))
    assert saver.part_path == str(dest) + ".part"
    assert saver.dest_dir == str(tmp_path)


def test_atomic_saver_custom_part_path(tmp_path):
    dest = tmp_path / "out.txt"
    saver = AtomicSaver(str(dest), part_file="foo.part")
    assert saver.part_path == str(tmp_path / "foo.part")


# ------------------------- iter_find_files -------------------------

def _make_tree(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.py").write_text("b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    (sub / "d.py").write_text("d")
    deeper = sub / "deeper"
    deeper.mkdir()
    (deeper / "e.txt").write_text("e")


def test_iter_find_files_single_pattern(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p) for p in iter_find_files(str(tmp_path), "*.txt"))
    assert found == ["a.txt", "c.txt", "e.txt"]


def test_iter_find_files_list_patterns(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p) for p in iter_find_files(str(tmp_path), ["*.txt", "*.py"]))
    assert found == ["a.txt", "b.py", "c.txt", "d.py", "e.txt"]


def test_iter_find_files_ignored(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p) for p in iter_find_files(str(tmp_path), "*.txt", ignored="c*"))
    assert found == ["a.txt", "e.txt"]


def test_iter_find_files_include_dirs(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p) for p in iter_find_files(str(tmp_path), "sub*", include_dirs=True))
    assert "sub" in found


def test_iter_find_files_max_depth_zero(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p) for p in iter_find_files(str(tmp_path), "*.txt", max_depth=0))
    assert found == ["a.txt"]


def test_iter_find_files_max_depth_one(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p) for p in iter_find_files(str(tmp_path), "*.txt", max_depth=1))
    assert found == ["a.txt", "c.txt"]


def test_iter_find_files_returns_generator(tmp_path):
    gen = iter_find_files(str(tmp_path), "*.none")
    assert list(gen) == []


# ------------------------- copy_tree / copytree -------------------------

def test_copy_tree_basic(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f1.txt").write_text("one")
    sub = src / "sub"
    sub.mkdir()
    (sub / "f2.txt").write_text("two")
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))
    assert (dst / "f1.txt").read_text() == "one"
    assert (dst / "sub" / "f2.txt").read_text() == "two"


def test_copy_tree_existing_dst(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f1.txt").write_text("one")
    dst = tmp_path / "dst"
    dst.mkdir()
    copy_tree(str(src), str(dst))
    assert (dst / "f1.txt").read_text() == "one"


def test_copy_tree_ignore(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("k")
    (src / "skip.txt").write_text("s")
    dst = tmp_path / "dst"

    def ignore_func(directory, names):
        return [n for n in names if n == "skip.txt"]

    copy_tree(str(src), str(dst), ignore=ignore_func)
    assert (dst / "keep.txt").exists()
    assert not (dst / "skip.txt").exists()


def test_copytree_alias_is_copy_tree():
    assert copytree is copy_tree


# ------------------------- DummyFile -------------------------

def test_dummyfile_basic():
    df = DummyFile("/nonexistent")
    assert df.name == "/nonexistent"
    assert df.mode == "r"
    assert df.closed is False
    assert df.fileno() == -1


def test_dummyfile_reads():
    df = DummyFile("x")
    assert df.read() == ""
    assert df.readline() == ""
    assert df.readlines() == []
    assert df.tell() == 0


def test_dummyfile_write_and_flush():
    df = DummyFile("x")
    assert df.write("abc") is None
    assert df.flush() is None
    assert df.seek() is None
    assert df.truncate() is None
    assert df.writelines(["a", "b"]) is None


def test_dummyfile_next():
    df = DummyFile("x")
    with pytest.raises(StopIteration):
        df.next()
    with pytest.raises(StopIteration):
        next(df)


def test_dummyfile_close_and_closed_errors():
    df = DummyFile("x")
    df.close()
    assert df.closed is True
    for method in (df.read, df.readline, df.readlines, df.flush,
                   df.seek, df.tell, df.truncate, df.write, df.writelines):
        with pytest.raises(ValueError):
            method()


def test_dummyfile_context_manager():
    df = DummyFile("x")
    with df as result:
        assert result is None
    # closed flag isn't set by context manager exit in current impl
    assert df.closed is False


# ------------------------- rotate_file -------------------------

def test_rotate_file_keep_too_small(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        rotate_file(str(f), keep=0)


def test_rotate_file_nonexistent(tmp_path):
    f = tmp_path / "missing.txt"
    # no error, no-op
    assert rotate_file(str(f)) is None
    assert not f.exists()


def test_rotate_file_not_a_file(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d))


def test_rotate_file_basic(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("current")
    rotate_file(str(f), keep=3)
    assert not f.exists()
    assert (tmp_path / "log.1.txt").read_text() == "current"


def test_rotate_file_shifts_existing(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("v0")
    (tmp_path / "log.1.txt").write_text("v1")
    (tmp_path / "log.2.txt").write_text("v2")
    rotate_file(str(base), keep=3)
    assert not base.exists()
    assert (tmp_path / "log.1.txt").read_text() == "v0"
    assert (tmp_path / "log.2.txt").read_text() == "v1"
    assert (tmp_path / "log.3.txt").read_text() == "v2"


def test_rotate_file_drops_beyond_keep(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("v0")
    (tmp_path / "log.1.txt").write_text("v1")
    (tmp_path / "log.2.txt").write_text("v2")
    rotate_file(str(base), keep=2)
    assert (tmp_path / "log.1.txt").read_text() == "v0"
    assert (tmp_path / "log.2.txt").read_text() == "v1"
    # log.3 should not exist
    assert not (tmp_path / "log.3.txt").exists()


def test_rotate_file_no_extension(tmp_path):
    base = tmp_path / "myfile"
    base.write_text("v0")
    rotate_file(str(base), keep=2)
    assert not base.exists()
    assert (tmp_path / "myfile.1").read_text() == "v0"
