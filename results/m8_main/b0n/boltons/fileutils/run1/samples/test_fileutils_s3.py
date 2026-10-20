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
    replace,
    atomic_rename,
    set_cloexec,
)


# --------------------------------------------------------------------------
# mkdir_p
# --------------------------------------------------------------------------

def test_mkdir_p_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_existing_ok(tmp_path):
    target = tmp_path / "exists"
    target.mkdir()
    assert mkdir_p(str(target)) is None


def test_mkdir_p_raises_on_file_conflict(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(OSError) as ei:
        mkdir_p(str(f))
    assert ei.value.errno == errno.EEXIST


# --------------------------------------------------------------------------
# FilePerms
# --------------------------------------------------------------------------

def test_fileperms_basic():
    fp = FilePerms("rwx", "r", "rx")
    assert fp.user == "rwx"
    assert fp.group == "r"
    assert fp.other == "rx"
    assert int(fp) == 0o754


def test_fileperms_defaults_empty():
    fp = FilePerms()
    assert fp.user == ""
    assert fp.group == ""
    assert fp.other == ""
    assert int(fp) == 0


def test_fileperms_normalizes_ordering():
    fp = FilePerms(user="rwx", group="xrw", other="wxr")
    assert fp.user == "rwx"
    assert fp.group == "rwx"
    assert fp.other == "rwx"
    assert int(fp) == 0o777


def test_fileperms_repr():
    fp = FilePerms("rw", "r", "r")
    assert repr(fp) == "FilePerms(user='rw', group='r', other='r')"


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


def test_fileperms_from_int_masks_extra_bits():
    fp = FilePerms.from_int(0o7777)
    assert int(fp) == 0o777
    assert fp.user == "rwx"


def test_fileperms_invalid_chars():
    with pytest.raises(ValueError):
        FilePerms(user="rxz")
    with pytest.raises(ValueError):
        FilePerms(group="q")


def test_fileperms_duplicate_chars_collapsed():
    fp = FilePerms("rrwwxx", "", "")
    assert fp.user == "rwx"
    assert int(fp) == 0o700


def test_fileperms_from_path(tmp_path):
    f = tmp_path / "afile"
    f.write_text("data")
    os.chmod(str(f), 0o640)
    fp = FilePerms.from_path(str(f))
    assert fp.user == "rw"
    assert fp.group == "r"
    assert fp.other == ""
    assert int(fp) == 0o640


def test_fileperms_setter_update_integer():
    fp = FilePerms()
    fp.user = "rw"
    assert int(fp) == 0o600
    fp.group = "r"
    assert int(fp) == 0o640


def test_fileperms_setter_same_value_noop():
    fp = FilePerms("rwx", "", "")
    fp.user = "rwx"
    assert fp.user == "rwx"
    assert int(fp) == 0o700


def test_fileperms_class_level_property_access():
    prop = FilePerms.user
    assert prop.attribute == "_user"
    assert prop.offset == 2


# --------------------------------------------------------------------------
# path_to_unicode
# --------------------------------------------------------------------------

def test_path_to_unicode_str():
    assert path_to_unicode("hello") == "hello"


def test_path_to_unicode_bytes():
    s = "hello"
    result = path_to_unicode(s.encode(sys.getfilesystemencoding()))
    assert result == "hello"
    assert isinstance(result, str)


# --------------------------------------------------------------------------
# AtomicSaver / atomic_save
# --------------------------------------------------------------------------

def test_atomic_save_basic(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as f:
        f.write("hello")
    assert dest.read_text() == "hello"
    assert not (tmp_path / "out.txt.part").exists()


def test_atomic_save_binary(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as f:
        f.write(b"\x00\x01\x02")
    assert dest.read_bytes() == b"\x00\x01\x02"


def test_atomic_save_error_preserves_old(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), text_mode=True) as f:
            f.write("new")
            raise ZeroDivisionError()
    assert dest.read_text() == "old"
    assert not (tmp_path / "out.txt.part").exists()


def test_atomic_save_error_keep_part(tmp_path):
    dest = tmp_path / "out.txt"
    with pytest.raises(RuntimeError):
        with atomic_save(str(dest), text_mode=True,
                         rm_part_on_exc=False) as f:
            f.write("partial")
            raise RuntimeError()
    part = tmp_path / "out.txt.part"
    assert part.exists()
    assert part.read_text() == "partial"


def test_atomic_saver_overwrite_false_existing(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    saver = AtomicSaver(str(dest), overwrite=False)
    with pytest.raises(OSError) as ei:
        with saver:
            pass
    assert ei.value.errno == errno.EEXIST


def test_atomic_saver_overwrite_false_no_existing(tmp_path):
    dest = tmp_path / "out.txt"
    with AtomicSaver(str(dest), overwrite=False, text_mode=True) as f:
        f.write("new")
    assert dest.read_text() == "new"


def test_atomic_saver_existing_part_raises(tmp_path):
    dest = tmp_path / "out.txt"
    part = tmp_path / "out.txt.part"
    part.write_text("stale")
    saver = AtomicSaver(str(dest), text_mode=True)
    with pytest.raises(OSError) as ei:
        saver.setup()
    assert ei.value.errno == errno.EEXIST


def test_atomic_saver_overwrite_part(tmp_path):
    dest = tmp_path / "out.txt"
    part = tmp_path / "out.txt.part"
    part.write_text("stale")
    with AtomicSaver(str(dest), text_mode=True,
                     overwrite_part=True) as f:
        f.write("new")
    assert dest.read_text() == "new"
    assert not part.exists()


def test_atomic_saver_custom_part_file(tmp_path):
    dest = tmp_path / "out.txt"
    with AtomicSaver(str(dest), text_mode=True,
                     part_file="mypart.tmp") as f:
        f.write("x")
    assert dest.read_text() == "x"
    assert not (tmp_path / "mypart.tmp").exists()


def test_atomic_saver_file_perms(tmp_path):
    dest = tmp_path / "out.txt"
    with AtomicSaver(str(dest), text_mode=True,
                     file_perms=0o600) as f:
        f.write("x")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o600


def test_atomic_saver_copies_existing_perms(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    os.chmod(str(dest), 0o640)
    with AtomicSaver(str(dest), text_mode=True) as f:
        f.write("new")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o640


def test_atomic_saver_unexpected_kwargs(tmp_path):
    with pytest.raises(TypeError):
        AtomicSaver(str(tmp_path / "x"), bogus=True)


def test_atomic_saver_explicit_setup_and_manual(tmp_path):
    dest = tmp_path / "out.txt"
    saver = AtomicSaver(str(dest), text_mode=True)
    saver.setup()
    saver.part_file.write("manual")
    saver.__exit__(None, None, None)
    assert dest.read_text() == "manual"


# --------------------------------------------------------------------------
# iter_find_files
# --------------------------------------------------------------------------

def _make_tree(root):
    (root / "a.py").write_text("")
    (root / "b.txt").write_text("")
    (root / "sub").mkdir()
    (root / "sub" / "c.py").write_text("")
    (root / "sub" / "deep").mkdir()
    (root / "sub" / "deep" / "d.py").write_text("")
    (root / ".hidden.py").write_text("")


def test_iter_find_files_single_pattern(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tmp_path), "*.py"))
    assert found == [".hidden.py", "a.py", "c.py", "d.py"]


def test_iter_find_files_list_patterns(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tmp_path),
                                            ["*.py", "*.txt"]))
    assert found == [".hidden.py", "a.py", "b.txt", "c.py", "d.py"]


def test_iter_find_files_ignored(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tmp_path), "*.py",
                                            ignored=".*"))
    assert found == ["a.py", "c.py", "d.py"]


def test_iter_find_files_ignored_list(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tmp_path), "*.py",
                                            ignored=[".*", "c.*"]))
    assert found == ["a.py", "d.py"]


def test_iter_find_files_include_dirs(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "*",
                                   include_dirs=True))
    names = sorted(os.path.basename(p) for p in found)
    assert "sub" in names
    assert "deep" in names


def test_iter_find_files_max_depth_zero(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tmp_path), "*.py",
                                            max_depth=0))
    assert found == [".hidden.py", "a.py"]


def test_iter_find_files_max_depth_one(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(p)
                   for p in iter_find_files(str(tmp_path), "*.py",
                                            max_depth=1))
    assert found == [".hidden.py", "a.py", "c.py"]


def test_iter_find_files_includes_directory_prefix(tmp_path):
    _make_tree(tmp_path)
    for p in iter_find_files(str(tmp_path), "*.py"):
        assert p.startswith(str(tmp_path))


# --------------------------------------------------------------------------
# copy_tree / copytree
# --------------------------------------------------------------------------

def test_copy_tree_basic(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("alpha")
    (src / "sub").mkdir()
    (src / "sub" / "b.txt").write_text("beta")
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))
    assert (dst / "a.txt").read_text() == "alpha"
    assert (dst / "sub" / "b.txt").read_text() == "beta"


def test_copy_tree_existing_dst_ok(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("alpha")
    dst = tmp_path / "dst"
    dst.mkdir()
    copy_tree(str(src), str(dst))
    assert (dst / "a.txt").read_text() == "alpha"


def test_copy_tree_ignore(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")
    (src / "b.txt").write_text("b")
    dst = tmp_path / "dst"

    def ignore_func(path, names):
        return [n for n in names if n == "b.txt"]

    copy_tree(str(src), str(dst), ignore=ignore_func)
    assert (dst / "a.txt").exists()
    assert not (dst / "b.txt").exists()


def test_copytree_alias():
    assert copytree is copy_tree


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


# --------------------------------------------------------------------------
# DummyFile
# --------------------------------------------------------------------------

def test_dummyfile_basic_attrs():
    d = DummyFile("somepath", mode="w")
    assert d.name == "somepath"
    assert d.mode == "w"
    assert d.closed is False
    assert d.fileno() == -1
    assert d.tell() == 0


def test_dummyfile_reads():
    d = DummyFile("p")
    assert d.read() == ""
    assert d.readline() == ""
    assert d.readlines() == []


def test_dummyfile_writes_noop():
    d = DummyFile("p")
    assert d.write("x") is None
    assert d.writelines(["a", "b"]) is None
    assert d.flush() is None
    assert d.seek() is None
    assert d.truncate() is None


def test_dummyfile_next():
    d = DummyFile("p")
    with pytest.raises(StopIteration):
        next(d)
    with pytest.raises(StopIteration):
        d.next()


def test_dummyfile_close():
    d = DummyFile("p")
    d.close()
    assert d.closed is True


def test_dummyfile_operations_on_closed():
    d = DummyFile("p")
    d.close()
    for meth in ("read", "readline", "readlines", "seek", "tell",
                 "truncate", "flush"):
        with pytest.raises(ValueError):
            getattr(d, meth)()
    with pytest.raises(ValueError):
        d.write("x")
    with pytest.raises(ValueError):
        d.writelines(["x"])
    with pytest.raises(ValueError):
        d.__enter__()


def test_dummyfile_context_manager():
    d = DummyFile("p")
    assert d.__enter__() is None
    assert d.__exit__(None, None, None) is None


# --------------------------------------------------------------------------
# rotate_file
# --------------------------------------------------------------------------

def test_rotate_file_no_exist(tmp_path):
    target = tmp_path / "log.txt"
    assert rotate_file(str(target)) is None
    assert not target.exists()


def test_rotate_file_basic(tmp_path):
    target = tmp_path / "log.txt"
    target.write_text("current")
    rotate_file(str(target))
    assert not target.exists()
    assert (tmp_path / "log.1.txt").read_text() == "current"


def test_rotate_file_chain(tmp_path):
    target = tmp_path / "log.txt"
    (tmp_path / "log.1.txt").write_text("one")
    (tmp_path / "log.2.txt").write_text("two")
    target.write_text("current")
    rotate_file(str(target))
    assert (tmp_path / "log.1.txt").read_text() == "current"
    assert (tmp_path / "log.2.txt").read_text() == "one"
    # original .2 dropped if beyond keep
    assert not (tmp_path / "log.3.txt").exists()


def test_rotate_file_keep_limit(tmp_path):
    target = tmp_path / "log.txt"
    target.write_text("current")
    for i in range(1, 4):
        (tmp_path / f"log.{i}.txt").write_text(f"v{i}")
    rotate_file(str(target), keep=2)
    assert (tmp_path / "log.1.txt").read_text() == "current"
    assert (tmp_path / "log.2.txt").read_text() == "v1"
    assert not (tmp_path / "log.3.txt").exists()


def test_rotate_file_no_ext(tmp_path):
    target = tmp_path / "log"
    target.write_text("current")
    rotate_file(str(target))
    assert (tmp_path / "log.1").read_text() == "current"


def test_rotate_file_keep_invalid(tmp_path):
    target = tmp_path / "log.txt"
    target.write_text("x")
    with pytest.raises(ValueError):
        rotate_file(str(target), keep=0)
    with pytest.raises(ValueError):
        rotate_file(str(target), keep=-1)


def test_rotate_file_not_a_file(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d))


# --------------------------------------------------------------------------
# replace / atomic_rename helpers
# --------------------------------------------------------------------------

def test_replace_moves_file(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello")
    replace(str(src), str(dst))
    assert not src.exists()
    assert dst.read_text() == "hello"


def test_atomic_rename_overwrite_true(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("new")
    dst.write_text("old")
    atomic_rename(str(src), str(dst), overwrite=True)
    assert dst.read_text() == "new"
    assert not src.exists()


def test_atomic_rename_no_overwrite(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("new")
    atomic_rename(str(src), str(dst), overwrite=False)
    assert dst.read_text() == "new"
    assert not src.exists()


def test_set_cloexec_does_not_raise(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    fd = os.open(str(f), os.O_RDONLY)
    try:
        assert set_cloexec(fd) is None
    finally:
        os.close(fd)
