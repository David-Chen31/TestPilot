import errno
import os
import stat

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


# ---------------------------------------------------------------- mkdir_p

def test_mkdir_p_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_existing_dir_no_error(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_reraises_when_path_is_file(tmp_path):
    target = tmp_path / "afile"
    target.write_text("x")
    with pytest.raises(OSError) as excinfo:
        mkdir_p(str(target))
    assert excinfo.value.errno == errno.EEXIST


# ---------------------------------------------------------------- FilePerms

def test_fileperms_basic_construction():
    fp = FilePerms(user="rwx", group="xrw", other="wxr")
    assert fp.user == "rwx"
    assert fp.group == "rwx"
    assert fp.other == "rwx"


def test_fileperms_int_value():
    fp = FilePerms("r", "r", "")
    assert int(fp) == 288
    assert oct(288)[-3:] == "440"


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


def test_fileperms_from_int_masks_full_perms():
    fp = FilePerms.from_int(0o7777)
    assert int(fp) == 0o777


def test_fileperms_repr():
    fp = FilePerms.from_int(0o644)
    assert repr(fp) == "FilePerms(user='rw', group='r', other='r')"


def test_fileperms_invalid_chars():
    with pytest.raises(ValueError):
        FilePerms(user="abc")


def test_fileperms_invalid_type():
    # Non-string values are stringified, yielding a ValueError on invalid chars
    with pytest.raises(ValueError):
        FilePerms(user=123)


def test_fileperms_setter_updates_to_sorted():
    fp = FilePerms()
    fp.user = "xrw"
    assert fp.user == "rwx"
    assert int(fp) == 0o700


def test_fileperms_setter_same_value_noop():
    fp = FilePerms(user="rwx")
    before = int(fp)
    fp.user = "rwx"
    assert int(fp) == before


def test_fileperms_setter_clearing():
    # The setter only ORs bits into the integer; it never clears them.
    fp = FilePerms(user="rwx")
    fp.user = ""
    assert fp.user == ""
    # integer is not reset by clearing the string attribute
    assert int(fp) == 0o700


def test_fileperms_from_path(tmp_path):
    target = tmp_path / "permfile"
    target.write_text("hi")
    os.chmod(str(target), 0o640)
    fp = FilePerms.from_path(str(target))
    assert fp.user == "rw"
    assert fp.group == "r"
    assert fp.other == ""
    assert int(fp) == stat.S_IMODE(os.stat(str(target)).st_mode)


# ---------------------------------------------------------------- path_to_unicode

def test_path_to_unicode_str():
    assert path_to_unicode("hello") == "hello"


def test_path_to_unicode_bytes():
    result = path_to_unicode(b"hello")
    assert result == "hello"


# ---------------------------------------------------------------- AtomicSaver / atomic_save

def test_atomic_save_writes_file(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("content")
    assert dest.read_text() == "content"
    assert not os.path.exists(str(dest) + ".part")


def test_atomic_save_exception_leaves_no_file(tmp_path):
    dest = tmp_path / "out.txt"
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write("bye")
            1 / 0
    assert not dest.exists()
    assert not os.path.exists(str(dest) + ".part")


def test_atomic_saver_binary_mode(tmp_path):
    dest = tmp_path / "out.bin"
    saver = AtomicSaver(str(dest))
    with saver as fo:
        fo.write(b"binary")
    assert dest.read_bytes() == b"binary"


def test_atomic_saver_overwrite_copies_perms(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    os.chmod(str(dest), 0o600)
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("new")
    assert dest.read_text() == "new"
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o600


def test_atomic_saver_explicit_file_perms(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True, file_perms=0o644) as fo:
        fo.write("x")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o644


def test_atomic_saver_overwrite_disabled(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    with pytest.raises(OSError) as excinfo:
        with atomic_save(str(dest), text_mode=True, overwrite=False):
            pass
    assert excinfo.value.errno == errno.EEXIST
    assert dest.read_text() == "old"


def test_atomic_saver_overwrite_true(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("new")
    assert dest.read_text() == "new"


def test_atomic_saver_preexisting_part_file(tmp_path):
    dest = tmp_path / "out.txt"
    part = str(dest) + ".part"
    open(part, "wb").close()
    with pytest.raises(OSError) as excinfo:
        with atomic_save(str(dest), text_mode=True):
            pass
    assert excinfo.value.errno == errno.EEXIST


def test_atomic_saver_overwrite_part(tmp_path):
    dest = tmp_path / "out.txt"
    part = str(dest) + ".part"
    open(part, "wb").close()
    with atomic_save(str(dest), text_mode=True, overwrite_part=True) as fo:
        fo.write("new")
    assert dest.read_text() == "new"
    assert not os.path.exists(part)


def test_atomic_saver_custom_part_filename(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True, part_file="my.part") as fo:
        fo.write("data")
    assert dest.read_text() == "data"
    assert not (tmp_path / "my.part").exists()


def test_atomic_saver_rm_part_on_exc_false(tmp_path):
    dest = tmp_path / "out.txt"
    part = str(dest) + ".part"
    with pytest.raises(ValueError):
        with atomic_save(str(dest), text_mode=True, rm_part_on_exc=False) as fo:
            fo.write("partial")
            raise ValueError("boom")
    assert os.path.exists(part)
    assert not dest.exists()


def test_atomic_saver_unexpected_kwargs(tmp_path):
    with pytest.raises(TypeError):
        AtomicSaver(str(tmp_path / "x"), bogus_arg=1)


def test_atomic_saver_setup_manual(tmp_path):
    dest = tmp_path / "out.txt"
    saver = AtomicSaver(str(dest), text_mode=True)
    saver.setup()
    try:
        saver.part_file.write("manual")
    finally:
        saver.part_file.flush()
        saver.part_file.close()
    assert os.path.exists(str(dest) + ".part")
    assert not dest.exists()


# ---------------------------------------------------------------- iter_find_files

def _make_tree(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.py").write_text("b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    (sub / "d.py").write_text("d")
    return tmp_path


def test_iter_find_files_single_pattern(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "*.txt"))
    expected = sorted([str(tmp_path / "a.txt"), str(tmp_path / "sub" / "c.txt")])
    assert found == expected


def test_iter_find_files_list_patterns(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), ["*.txt", "*.py"]))
    assert len(found) == 4


def test_iter_find_files_ignored(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "*.txt", ignored="a.txt"))
    assert found == [str(tmp_path / "sub" / "c.txt")]


def test_iter_find_files_ignored_list(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "*.txt", ignored=["a.txt", "c.txt"]))
    assert found == []


def test_iter_find_files_include_dirs(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "sub", include_dirs=True))
    assert found == [str(tmp_path / "sub")]


def test_iter_find_files_include_dirs_ignored(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "sub", ignored="sub", include_dirs=True))
    assert found == []


def test_iter_find_files_max_depth_zero(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "*.txt", max_depth=0))
    assert found == [str(tmp_path / "a.txt")]


def test_iter_find_files_max_depth_one(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "*.txt", max_depth=1))
    assert found == sorted([str(tmp_path / "a.txt"), str(tmp_path / "sub" / "c.txt")])


def test_iter_find_files_no_match(tmp_path):
    _make_tree(tmp_path)
    found = list(iter_find_files(str(tmp_path), "*.nonexistent"))
    assert found == []


# ---------------------------------------------------------------- copy_tree

def test_copy_tree_basic(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f1.txt").write_text("one")
    (src / "sub").mkdir()
    (src / "sub" / "f2.txt").write_text("two")

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))

    assert (dst / "f1.txt").read_text() == "one"
    assert (dst / "sub" / "f2.txt").read_text() == "two"


def test_copy_tree_existing_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f1.txt").write_text("one")
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "existing.txt").write_text("keep")

    copy_tree(str(src), str(dst))
    assert (dst / "f1.txt").read_text() == "one"
    assert (dst / "existing.txt").read_text() == "keep"


def test_copy_tree_ignore(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("keep")
    (src / "skip.txt").write_text("skip")

    def ignore_func(path, names):
        return [n for n in names if n == "skip.txt"]

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst), ignore=ignore_func)
    assert (dst / "keep.txt").exists()
    assert not (dst / "skip.txt").exists()


def test_copy_tree_symlinks(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "real.txt").write_text("real")
    os.symlink(str(src / "real.txt"), str(src / "link.txt"))

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst), symlinks=True)
    assert os.path.islink(str(dst / "link.txt"))


def test_copytree_alias():
    assert copytree is copy_tree


# ---------------------------------------------------------------- DummyFile

def test_dummyfile_basic():
    df = DummyFile("/dev/null")
    assert df.name == "/dev/null"
    assert df.mode == "r"
    assert df.closed is False
    assert df.fileno() == -1
    assert df.read() == ""
    assert df.readline() == ""
    assert df.readlines() == []
    assert df.tell() == 0


def test_dummyfile_write_returns_none():
    df = DummyFile("x")
    assert df.write("hello") is None
    assert df.writelines(["a", "b"]) is None


def test_dummyfile_close():
    df = DummyFile("x")
    df.close()
    assert df.closed is True


def test_dummyfile_operations_after_close():
    df = DummyFile("x")
    df.close()
    for meth in [df.flush, df.read, df.readline, df.readlines,
                 df.seek, df.tell, df.truncate]:
        with pytest.raises(ValueError):
            meth()
    with pytest.raises(ValueError):
        df.write("x")
    with pytest.raises(ValueError):
        df.writelines(["x"])
    with pytest.raises(ValueError):
        df.__enter__()


def test_dummyfile_next():
    df = DummyFile("x")
    with pytest.raises(StopIteration):
        next(df)
    with pytest.raises(StopIteration):
        df.next()


def test_dummyfile_context_manager():
    df = DummyFile("x")
    assert df.__enter__() is None
    assert df.__exit__(None, None, None) is None


# ---------------------------------------------------------------- rotate_file

def test_rotate_file_no_existing(tmp_path):
    fname = str(tmp_path / "log.txt")
    rotate_file(fname)
    assert not os.path.exists(fname)


def test_rotate_file_keep_invalid(tmp_path):
    fname = str(tmp_path / "log.txt")
    with pytest.raises(ValueError):
        rotate_file(fname, keep=0)


def test_rotate_file_dir_raises(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d))


def test_rotate_file_basic_rotation(tmp_path):
    # With keep=1, the single rotated name gets removed at the end,
    # so nothing remains.
    fname = str(tmp_path / "log.txt")
    with open(fname, "w") as f:
        f.write("original")

    rotate_file(fname)

    assert not os.path.exists(fname)
    # keep defaults to 5 -> kept_names = log.1.txt .. log.5.txt, last removed
    assert (tmp_path / "log.1.txt").read_text() == "original"
    assert not (tmp_path / "log.5.txt").exists()


def test_rotate_file_multiple(tmp_path):
    root = tmp_path / "log"
    (root).write_text("v0")
    (tmp_path / "log.1").write_text("v1")
    (tmp_path / "log.2").write_text("v2")

    rotate_file(str(root), keep=3)

    assert not os.path.exists(str(root))
    # the last kept name (log.3) is removed at the end
    assert (tmp_path / "log.1").read_text() == "v0"
    assert (tmp_path / "log.2").read_text() == "v1"
    assert not (tmp_path / "log.3").exists()


def test_rotate_file_drops_beyond_keep(tmp_path):
    root = tmp_path / "log.txt"
    (root).write_text("v0")
    (tmp_path / "log.1.txt").write_text("v1")
    (tmp_path / "log.2.txt").write_text("v2")

    rotate_file(str(root), keep=1)

    # With keep=1, the only kept name (log.1.txt) is renamed then removed,
    # so nothing remains.
    assert not os.path.exists(str(root))
    assert not (tmp_path / "log.1.txt").exists()
