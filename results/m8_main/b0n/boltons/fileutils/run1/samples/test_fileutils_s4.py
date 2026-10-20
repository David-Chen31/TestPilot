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


def test_mkdir_p_existing_returns_none(tmp_path):
    target = tmp_path / "exists"
    target.mkdir()
    assert mkdir_p(str(target)) is None
    assert target.is_dir()


def test_mkdir_p_existing_deep(tmp_path):
    target = tmp_path / "x" / "y"
    mkdir_p(str(target))
    # second call on existing directory should not raise
    assert mkdir_p(str(target)) is None
    assert target.is_dir()


def test_mkdir_p_raises_when_path_is_file(tmp_path):
    f = tmp_path / "afile"
    f.write_text("hi")
    with pytest.raises(OSError):
        mkdir_p(str(f))


# ---------------------------------------------------------------------------
# FilePerms
# ---------------------------------------------------------------------------

def test_fileperms_default_empty():
    fp = FilePerms()
    assert fp.user == ""
    assert fp.group == ""
    assert fp.other == ""
    assert int(fp) == 0


def test_fileperms_basic_and_int():
    fp = FilePerms('r', 'r', '')
    assert fp.user == 'r'
    assert fp.group == 'r'
    assert fp.other == ''
    assert int(fp) == 288
    assert oct(288)[-3:] == '440'


def test_fileperms_sorting():
    fp = FilePerms(user='rwx', group='xrw', other='wxr')
    assert fp.user == 'rwx'
    assert fp.group == 'rwx'
    assert fp.other == 'rwx'
    assert int(fp) == 0o777


def test_fileperms_repr():
    fp = FilePerms.from_int(0o644)
    assert repr(fp) == "FilePerms(user='rw', group='r', other='r')"


def test_fileperms_from_int_full():
    fp = FilePerms.from_int(0o777)
    assert (fp.user, fp.group, fp.other) == ('rwx', 'rwx', 'rwx')
    assert int(fp) == 0o777


def test_fileperms_from_int_masks_high_bits():
    # Only bottom 9 bits considered.
    fp = FilePerms.from_int(0o7777)
    assert int(fp) == 0o777


def test_fileperms_from_int_zero():
    fp = FilePerms.from_int(0)
    assert (fp.user, fp.group, fp.other) == ('', '', '')
    assert int(fp) == 0


def test_fileperms_from_path(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello")
    os.chmod(str(f), 0o640)
    fp = FilePerms.from_path(str(f))
    assert fp.user == 'rw'
    assert fp.group == 'r'
    assert fp.other == ''
    assert int(fp) == 0o640


def test_fileperms_invalid_char():
    with pytest.raises(ValueError):
        FilePerms(user='rwq')


def test_fileperms_invalid_type():
    with pytest.raises(TypeError):
        FilePerms(user=5)


def test_fileperms_setter_updates_int():
    fp = FilePerms()
    fp.user = 'rw'
    assert int(fp) == 0o600
    fp.group = 'r'
    assert int(fp) == 0o640
    fp.other = 'x'
    assert int(fp) == 0o641


def test_fileperms_setter_same_value_noop():
    fp = FilePerms(user='rw')
    before = int(fp)
    fp.user = 'rw'
    assert int(fp) == before


def test_fileperms_set_other_after_same():
    fp = FilePerms()
    fp.other = 'r'
    fp.other = 'r'
    assert int(fp) == 0o004


# ---------------------------------------------------------------------------
# path_to_unicode
# ---------------------------------------------------------------------------

def test_path_to_unicode_str():
    assert path_to_unicode("hello") == "hello"


def test_path_to_unicode_bytes():
    encoding = sys.getfilesystemencoding() or sys.getdefaultencoding()
    assert path_to_unicode("hé".encode(encoding)) == "hé"


# ---------------------------------------------------------------------------
# AtomicSaver / atomic_save
# ---------------------------------------------------------------------------

def test_atomic_save_writes_file(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("hello")
    assert dest.read_text() == "hello"
    assert not (tmp_path / "out.txt.part").exists()


def test_atomic_save_binary_default(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as fo:
        fo.write(b"bytes")
    assert dest.read_bytes() == b"bytes"


def test_atomic_save_exception_removes_part(tmp_path):
    dest = tmp_path / "out.txt"
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write("bye")
            1 / 0
    assert not dest.exists()
    assert not (tmp_path / "out.txt.part").exists()


def test_atomic_save_exception_no_rm_part(tmp_path):
    dest = tmp_path / "out.txt"
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), text_mode=True, rm_part_on_exc=False) as fo:
            fo.write("bye")
            1 / 0
    assert (tmp_path / "out.txt.part").exists()


def test_atomic_saver_overwrite_false_raises(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("existing")
    with pytest.raises(OSError) as exc_info:
        with atomic_save(str(dest), text_mode=True, overwrite=False) as fo:
            fo.write("new")
    assert exc_info.value.errno == errno.EEXIST
    assert dest.read_text() == "existing"


def test_atomic_saver_overwrite_true(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("existing")
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("new")
    assert dest.read_text() == "new"


def test_atomic_saver_custom_part_file(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True, part_file="custom.part") as fo:
        fo.write("data")
    assert dest.read_text() == "data"
    assert not (tmp_path / "custom.part").exists()


def test_atomic_saver_part_exists_no_overwrite_part(tmp_path):
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
        fo.write("fresh")
    assert dest.read_text() == "fresh"


def test_atomic_saver_unexpected_kwargs(tmp_path):
    with pytest.raises(TypeError):
        AtomicSaver(str(tmp_path / "x"), bogus_flag=True)


def test_atomic_saver_explicit_file_perms(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True, file_perms=0o600) as fo:
        fo.write("perm")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o600


def test_atomic_saver_preserves_existing_perms(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    os.chmod(str(dest), 0o640)
    with atomic_save(str(dest), text_mode=True) as fo:
        fo.write("new content")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o640


def test_atomic_saver_manual_setup(tmp_path):
    dest = tmp_path / "manual.txt"
    saver = AtomicSaver(str(dest), text_mode=True)
    saver.setup()
    saver.part_file.write("manual")
    saver.part_file.close()
    # Not using __exit__: emulate by calling it explicitly
    saver.part_file = None
    saver.__exit__(None, None, None)
    assert dest.read_text() == "manual"


# ---------------------------------------------------------------------------
# iter_find_files
# ---------------------------------------------------------------------------

def _build_tree(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.py").write_text("b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    (sub / "d.py").write_text("d")
    deep = sub / "deep"
    deep.mkdir()
    (deep / "e.txt").write_text("e")
    return tmp_path


def test_iter_find_files_single_pattern(tmp_path):
    root = _build_tree(tmp_path)
    results = sorted(os.path.basename(p) for p in iter_find_files(str(root), "*.txt"))
    assert results == ["a.txt", "c.txt", "e.txt"]


def test_iter_find_files_list_patterns(tmp_path):
    root = _build_tree(tmp_path)
    results = set(os.path.basename(p) for p in iter_find_files(str(root), ["*.py", "*.txt"]))
    assert results == {"a.txt", "b.py", "c.txt", "d.py", "e.txt"}


def test_iter_find_files_ignored(tmp_path):
    root = _build_tree(tmp_path)
    results = sorted(
        os.path.basename(p) for p in iter_find_files(str(root), "*.txt", ignored="c*")
    )
    assert results == ["a.txt", "e.txt"]


def test_iter_find_files_ignored_list(tmp_path):
    root = _build_tree(tmp_path)
    results = sorted(
        os.path.basename(p)
        for p in iter_find_files(str(root), "*.txt", ignored=["c*", "e*"])
    )
    assert results == ["a.txt"]


def test_iter_find_files_include_dirs(tmp_path):
    root = _build_tree(tmp_path)
    results = set(os.path.basename(p) for p in iter_find_files(str(root), "*sub*", include_dirs=True))
    assert results == {"sub"}


def test_iter_find_files_dirs_excluded_by_default(tmp_path):
    root = _build_tree(tmp_path)
    results = list(iter_find_files(str(root), "sub"))
    assert results == []


def test_iter_find_files_max_depth_zero(tmp_path):
    root = _build_tree(tmp_path)
    results = sorted(os.path.basename(p) for p in iter_find_files(str(root), "*.txt", max_depth=0))
    assert results == ["a.txt"]


def test_iter_find_files_max_depth_one(tmp_path):
    root = _build_tree(tmp_path)
    results = sorted(os.path.basename(p) for p in iter_find_files(str(root), "*.txt", max_depth=1))
    assert results == ["a.txt", "c.txt"]


def test_iter_find_files_yields_full_paths(tmp_path):
    root = _build_tree(tmp_path)
    results = list(iter_find_files(str(root), "a.txt"))
    assert len(results) == 1
    assert results[0] == str(root / "a.txt")


# ---------------------------------------------------------------------------
# copy_tree
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


def test_copy_tree_into_existing(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("data")
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "existing.txt").write_text("keep")
    copy_tree(str(src), str(dst))
    assert (dst / "f.txt").read_text() == "data"
    assert (dst / "existing.txt").read_text() == "keep"


def test_copy_tree_alias():
    assert copytree is copy_tree


def test_copy_tree_recursive(tmp_path):
    src = tmp_path / "src"
    sub = src / "sub"
    sub.mkdir(parents=True)
    (src / "top.txt").write_text("t")
    (sub / "nested.txt").write_text("n")
    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))
    assert (dst / "top.txt").read_text() == "t"
    assert (dst / "sub" / "nested.txt").read_text() == "n"


def test_copy_tree_ignore(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("k")
    (src / "skip.txt").write_text("s")

    def ignore(path, names):
        return {"skip.txt"}

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst), ignore=ignore)
    assert (dst / "keep.txt").exists()
    assert not (dst / "skip.txt").exists()


# ---------------------------------------------------------------------------
# DummyFile
# ---------------------------------------------------------------------------

def test_dummyfile_attributes():
    df = DummyFile("/dev/null", mode='w')
    assert df.name == "/dev/null"
    assert df.mode == 'w'
    assert df.closed is False
    assert df.isatty is False
    assert df.fileno() == -1


def test_dummyfile_reads_return_empty():
    df = DummyFile("x")
    assert df.read() == ''
    assert df.read(10) == ''
    assert df.readline() == ''
    assert df.readlines() == []
    assert df.tell() == 0


def test_dummyfile_write_and_seek():
    df = DummyFile("x", mode='w')
    assert df.write("abc") is None
    assert df.writelines(["a", "b"]) is None
    assert df.seek() is None
    assert df.truncate() is None
    df.flush()


def test_dummyfile_close_then_raises():
    df = DummyFile("x")
    df.close()
    assert df.closed is True
    for meth in (df.flush, df.read, df.readline, df.readlines,
                 df.seek, df.tell, df.truncate,
                 lambda: df.write("x"), lambda: df.writelines([])):
        with pytest.raises(ValueError):
            meth()


def test_dummyfile_iteration():
    df = DummyFile("x")
    with pytest.raises(StopIteration):
        next(df)
    with pytest.raises(StopIteration):
        df.__next__()


def test_dummyfile_context_manager():
    df = DummyFile("x")
    assert df.__enter__() is None
    assert df.__exit__(None, None, None) is None
    df.close()
    with pytest.raises(ValueError):
        df.__enter__()


# ---------------------------------------------------------------------------
# rotate_file
# ---------------------------------------------------------------------------

def test_rotate_file_no_existing_file(tmp_path):
    target = tmp_path / "log.txt"
    rotate_file(str(target))
    assert not target.exists()
    assert not (tmp_path / "log.1.txt").exists()


def test_rotate_file_keep_must_be_positive(tmp_path):
    target = tmp_path / "log.txt"
    with pytest.raises(ValueError):
        rotate_file(str(target), keep=0)
    with pytest.raises(ValueError):
        rotate_file(str(target), keep=-1)


def test_rotate_file_not_a_file(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d))


def test_rotate_file_basic(tmp_path):
    target = tmp_path / "log.txt"
    target.write_text("content")
    rotate_file(str(target))
    assert not target.exists()
    assert (tmp_path / "log.1.txt").read_text() == "content"


def test_rotate_file_shifts_up(tmp_path):
    target = tmp_path / "log.txt"
    target.write_text("new")
    (tmp_path / "log.1.txt").write_text("old1")
    (tmp_path / "log.2.txt").write_text("old2")
    rotate_file(str(target), keep=3)
    assert not target.exists()
    assert (tmp_path / "log.1.txt").read_text() == "new"
    assert (tmp_path / "log.2.txt").read_text() == "old1"
    assert (tmp_path / "log.3.txt").read_text() == "old2"


def test_rotate_file_drops_beyond_keep(tmp_path):
    target = tmp_path / "log.txt"
    target.write_text("new")
    (tmp_path / "log.1.txt").write_text("old1")
    (tmp_path / "log.2.txt").write_text("old2")
    (tmp_path / "log.3.txt").write_text("old3")
    rotate_file(str(target), keep=2)
    assert not target.exists()
    assert (tmp_path / "log.1.txt").read_text() == "new"
    assert (tmp_path / "log.2.txt").read_text() == "old1"
    # keep=2 means only indices 1,2 kept; original log.3 dropped
    assert not (tmp_path / "log.3.txt").exists()


def test_rotate_file_no_extension(tmp_path):
    target = tmp_path / "logfile"
    target.write_text("data")
    rotate_file(str(target), keep=1)
    assert not target.exists()
    assert (tmp_path / "logfile.1").read_text() == "data"


def test_rotate_file_keeps_only_last_when_keep_one(tmp_path):
    target = tmp_path / "log.txt"
    target.write_text("newest")
    (tmp_path / "log.1.txt").write_text("older")
    rotate_file(str(target), keep=1)
    assert not target.exists()
    assert (tmp_path / "log.1.txt").read_text() == "newest"
