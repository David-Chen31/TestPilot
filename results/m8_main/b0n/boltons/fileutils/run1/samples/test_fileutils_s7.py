import os
import stat
import errno
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


# ---------------------------------------------------------------------------
# mkdir_p
# ---------------------------------------------------------------------------

def test_mkdir_p_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_existing_ok(tmp_path):
    target = tmp_path / "exists"
    target.mkdir()
    # Should not raise
    assert mkdir_p(str(target)) is None


def test_mkdir_p_raises_on_file(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(OSError):
        mkdir_p(str(f))


def test_mkdir_p_returns_none(tmp_path):
    assert mkdir_p(str(tmp_path / "newdir")) is None


# ---------------------------------------------------------------------------
# FilePerms
# ---------------------------------------------------------------------------

def test_fileperms_default_empty():
    fp = FilePerms()
    assert fp.user == ''
    assert fp.group == ''
    assert fp.other == ''
    assert int(fp) == 0


def test_fileperms_basic():
    fp = FilePerms(user='rwx', group='rw', other='r')
    assert fp.user == 'rwx'
    assert fp.group == 'rw'
    assert fp.other == 'r'
    assert int(fp) == 0o744


def test_fileperms_char_order_normalized():
    fp = FilePerms(user='xrw', group='wxr', other='rxw')
    assert fp.user == 'rwx'
    assert fp.group == 'rwx'
    assert fp.other == 'rwx'


def test_fileperms_int():
    assert int(FilePerms('r', 'r', '')) == 0o440


def test_fileperms_repr():
    fp = FilePerms('rwx', 'rw', 'r')
    assert repr(fp) == "FilePerms(user='rwx', group='rw', other='r')"


def test_fileperms_invalid_chars():
    with pytest.raises(ValueError):
        FilePerms(user='q')


def test_fileperms_invalid_type():
    with pytest.raises(TypeError):
        FilePerms(user=123)


def test_fileperms_setter_same_value_noop():
    fp = FilePerms(user='rw')
    fp.user = 'rw'
    assert fp.user == 'rw'


def test_fileperms_setter_updates_integer():
    fp = FilePerms()
    fp.user = 'r'
    assert int(fp) == 0o400


def test_fileperms_from_int():
    fp = FilePerms.from_int(0o644)
    assert fp.user == 'rw'
    assert fp.group == 'r'
    assert fp.other == 'r'
    assert int(fp) == 0o644


def test_fileperms_from_int_zero():
    fp = FilePerms.from_int(0)
    assert int(fp) == 0


def test_fileperms_from_int_masks():
    fp = FilePerms.from_int(0o7777)
    assert int(fp) == 0o777
    assert fp.user == 'rwx'
    assert fp.group == 'rwx'
    assert fp.other == 'rwx'


def test_fileperms_from_path(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    os.chmod(str(f), 0o640)
    fp = FilePerms.from_path(str(f))
    assert int(fp) == 0o640


def test_fileperms_descriptor_access_on_class():
    prop = FilePerms.__dict__['user']
    assert prop.__get__(None, FilePerms) is prop


# ---------------------------------------------------------------------------
# path_to_unicode
# ---------------------------------------------------------------------------

def test_path_to_unicode_str():
    assert path_to_unicode("hello") == "hello"


def test_path_to_unicode_bytes():
    assert path_to_unicode(b"hello") == "hello"


# ---------------------------------------------------------------------------
# atomic_save / AtomicSaver
# ---------------------------------------------------------------------------

def test_atomic_save_returns_saver(tmp_path):
    dest = tmp_path / "file.txt"
    saver = atomic_save(str(dest))
    assert isinstance(saver, AtomicSaver)


def test_atomic_saver_writes_file(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("hello")
    assert dest.read_text() == "hello"
    assert not os.path.exists(str(dest) + ".part")


def test_atomic_saver_binary(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as fo:
        fo.write(b"\x00\x01\x02")
    assert dest.read_bytes() == b"\x00\x01\x02"


def test_atomic_saver_exception_keeps_original(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("original")
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write("new")
            1 / 0
    assert dest.read_text() == "original"
    assert not os.path.exists(str(dest) + ".part")


def test_atomic_saver_overwrite_false_raises(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("original")
    with pytest.raises(OSError) as excinfo:
        with atomic_save(str(dest), text_mode=True, overwrite=False) as fo:
            fo.write("new")
    assert excinfo.value.errno == errno.EEXIST
    assert dest.read_text() == "original"


def test_atomic_saver_overwrite_true(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("original")
    with atomic_save(str(dest), text_mode=True, overwrite=True) as fo:
        fo.write("new")
    assert dest.read_text() == "new"


def test_atomic_saver_custom_part_file(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True, part_file="mypart") as fo:
        fo.write("data")
        assert os.path.exists(str(tmp_path / "mypart"))
    assert dest.read_text() == "data"
    assert not os.path.exists(str(tmp_path / "mypart"))


def test_atomic_saver_part_exists_raises(tmp_path):
    dest = tmp_path / "out.txt"
    part = tmp_path / "out.txt.part"
    part.write_text("stale")
    with pytest.raises(OSError):
        with atomic_save(str(dest), text_mode=True):
            pass


def test_atomic_saver_overwrite_part(tmp_path):
    dest = tmp_path / "out.txt"
    part = tmp_path / "out.txt.part"
    part.write_text("stale")
    with atomic_save(str(dest), text_mode=True, overwrite_part=True) as fo:
        fo.write("new")
    assert dest.read_text() == "new"


def test_atomic_saver_rm_part_on_exc_false(tmp_path):
    dest = tmp_path / "out.txt"
    part = tmp_path / "out.txt.part"
    with pytest.raises(ValueError):
        with atomic_save(str(dest), text_mode=True,
                         rm_part_on_exc=False) as fo:
            fo.write("partial")
            raise ValueError("boom")
    assert part.exists()
    assert part.read_text() == "partial"
    assert not dest.exists()


def test_atomic_saver_unexpected_kwarg(tmp_path):
    with pytest.raises(TypeError):
        AtomicSaver(str(tmp_path / "x"), bogus=True)


def test_atomic_saver_file_perms(tmp_path):
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


def test_atomic_saver_setup_method(tmp_path):
    dest = tmp_path / "out.txt"
    saver = AtomicSaver(str(dest), text_mode=True)
    saver.setup()
    try:
        assert os.path.exists(saver.part_path)
        saver.part_file.write("direct")
    finally:
        saver.part_file.close()
    assert saver.part_file is not None


# ---------------------------------------------------------------------------
# iter_find_files
# ---------------------------------------------------------------------------

def _make_tree(root):
    (root / "a.py").write_text("")
    (root / "b.txt").write_text("")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("")
    (sub / "d.txt").write_text("")
    deeper = sub / "deeper"
    deeper.mkdir()
    (deeper / "e.py").write_text("")


def test_iter_find_files_single_pattern(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "*.py"))
    names = sorted(os.path.basename(p) for p in found)
    assert names == ["a.py", "c.py", "e.py"]


def test_iter_find_files_list_patterns(tmp_path):
    _make_tree(tmp_path)
    found = list(iter_find_files(str(tmp_path), ["*.py", "*.txt"]))
    assert len(found) == 5


def test_iter_find_files_ignored(tmp_path):
    _make_tree(tmp_path)
    found = list(iter_find_files(str(tmp_path), "*.py", ignored="a.py"))
    names = sorted(os.path.basename(p) for p in found)
    assert names == ["c.py", "e.py"]


def test_iter_find_files_ignored_list(tmp_path):
    _make_tree(tmp_path)
    found = list(iter_find_files(str(tmp_path), "*.py",
                                 ignored=["a.py", "c.py"]))
    names = sorted(os.path.basename(p) for p in found)
    assert names == ["e.py"]


def test_iter_find_files_include_dirs(tmp_path):
    _make_tree(tmp_path)
    found = list(iter_find_files(str(tmp_path), "sub", include_dirs=True))
    assert len(found) == 1
    assert found[0].endswith("sub")


def test_iter_find_files_max_depth_zero(tmp_path):
    _make_tree(tmp_path)
    found = list(iter_find_files(str(tmp_path), "*", max_depth=0))
    names = sorted(os.path.basename(p) for p in found)
    assert names == ["a.py", "b.txt"]


def test_iter_find_files_max_depth_one(tmp_path):
    _make_tree(tmp_path)
    found = list(iter_find_files(str(tmp_path), "*", max_depth=1))
    names = sorted(os.path.basename(p) for p in found)
    assert names == ["a.py", "b.txt", "c.py", "d.txt"]


def test_iter_find_files_yields_paths_with_prefix(tmp_path):
    _make_tree(tmp_path)
    found = list(iter_find_files(str(tmp_path), "a.py"))
    assert found == [str(tmp_path / "a.py")]


def test_iter_find_files_empty(tmp_path):
    assert list(iter_find_files(str(tmp_path), "*.py")) == []


def test_iter_find_files_returns_generator(tmp_path):
    gen = iter_find_files(str(tmp_path), "*")
    assert hasattr(gen, "__next__")


# ---------------------------------------------------------------------------
# copy_tree / copytree
# ---------------------------------------------------------------------------

def test_copy_tree_basic(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f1.txt").write_text("one")
    (src / "f2.txt").write_text("two")
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))
    assert (dst / "f1.txt").read_text() == "one"
    assert (dst / "f2.txt").read_text() == "two"


def test_copy_tree_into_existing(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f1.txt").write_text("one")
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "existing.txt").write_text("keep")
    copy_tree(str(src), str(dst))
    assert (dst / "f1.txt").read_text() == "one"
    assert (dst / "existing.txt").read_text() == "keep"


def test_copy_tree_nested(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    sub = src / "sub"
    sub.mkdir()
    (sub / "n.txt").write_text("nested")
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))
    assert (dst / "sub" / "n.txt").read_text() == "nested"


def test_copy_tree_ignore(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("k")
    (src / "skip.txt").write_text("s")
    dst = tmp_path / "dst"

    def ignore_func(path, names):
        return {"skip.txt"}

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
# DummyFile
# ---------------------------------------------------------------------------

def test_dummyfile_init():
    d = DummyFile("path", mode="w")
    assert d.name == "path"
    assert d.mode == "w"
    assert d.closed is False
    assert d.fileno() == -1


def test_dummyfile_read():
    d = DummyFile("p")
    assert d.read() == ""
    assert d.readline() == ""
    assert d.readlines() == []
    assert d.tell() == 0


def test_dummyfile_write_returns_none():
    d = DummyFile("p")
    assert d.write("x") is None
    assert d.writelines(["a"]) is None


def test_dummyfile_flush_noop():
    d = DummyFile("p")
    assert d.flush() is None


def test_dummyfile_close_and_ops_raise():
    d = DummyFile("p")
    d.close()
    assert d.closed is True
    for meth in ("read", "readline", "readlines", "flush",
                 "write", "writelines", "seek", "tell", "truncate"):
        with pytest.raises(ValueError):
            getattr(d, meth)()


def test_dummyfile_next():
    d = DummyFile("p")
    with pytest.raises(StopIteration):
        next(d)
    with pytest.raises(StopIteration):
        d.next()


def test_dummyfile_context_manager():
    d = DummyFile("p")
    assert d.__enter__() is None
    assert d.__exit__(None, None, None) is None


def test_dummyfile_context_manager_closed():
    d = DummyFile("p")
    d.close()
    with pytest.raises(ValueError):
        d.__enter__()


# ---------------------------------------------------------------------------
# rotate_file
# ---------------------------------------------------------------------------

def test_rotate_file_nonexistent(tmp_path):
    f = tmp_path / "never.txt"
    assert rotate_file(str(f)) is None
    assert not f.exists()


def test_rotate_file_invalid_keep(tmp_path):
    with pytest.raises(ValueError):
        rotate_file(str(tmp_path / "x"), keep=0)


def test_rotate_file_dir_raises(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d))


def test_rotate_file_basic(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("current")
    rotate_file(str(f), keep=3)
    assert not f.exists()
    assert (tmp_path / "log.1.txt").read_text() == "current"


def test_rotate_file_sequence(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("v3")
    (tmp_path / "log.1.txt").write_text("v2")
    (tmp_path / "log.2.txt").write_text("v1")
    rotate_file(str(f), keep=3)
    assert not f.exists()
    assert (tmp_path / "log.1.txt").read_text() == "v3"
    assert (tmp_path / "log.2.txt").read_text() == "v2"
    assert (tmp_path / "log.3.txt").read_text() == "v1"


def test_rotate_file_drops_beyond_keep(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("new")
    (tmp_path / "log.1.txt").write_text("old1")
    (tmp_path / "log.2.txt").write_text("old2")
    rotate_file(str(f), keep=2)
    assert not f.exists()
    assert (tmp_path / "log.1.txt").read_text() == "new"
    assert (tmp_path / "log.2.txt").read_text() == "old1"
    # log.3 should have been dropped
    assert not (tmp_path / "log.3.txt").exists()


def test_rotate_file_no_extension(tmp_path):
    f = tmp_path / "log"
    f.write_text("data")
    rotate_file(str(f), keep=2)
    assert not f.exists()
    assert (tmp_path / "log.1").read_text() == "data"


def test_rotate_file_keep_one(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("new")
    (tmp_path / "log.1.txt").write_text("old")
    rotate_file(str(f), keep=1)
    assert not f.exists()
    assert (tmp_path / "log.1.txt").read_text() == "new"
