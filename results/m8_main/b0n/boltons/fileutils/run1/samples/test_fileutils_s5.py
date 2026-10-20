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

def test_mkdir_p_new(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    mkdir_p(str(target))
    assert os.path.isdir(str(target))


def test_mkdir_p_existing(tmp_path):
    target = tmp_path / "a" / "b"
    mkdir_p(str(target))
    # calling again should not raise
    mkdir_p(str(target))
    assert os.path.isdir(str(target))


def test_mkdir_p_existing_file_raises(tmp_path):
    target = tmp_path / "file"
    target.write_text("hello")
    with pytest.raises(OSError) as ei:
        mkdir_p(str(target))
    assert ei.value.errno == errno.EEXIST


# ---------------------------------------------------------------------------
# FilePerms
# ---------------------------------------------------------------------------

def test_fileperms_basic():
    fp = FilePerms(user='rwx', group='xrw', other='wxr')
    assert fp.user == 'rwx'
    assert fp.group == 'rwx'
    assert fp.other == 'rwx'
    assert repr(fp) == "FilePerms(user='rwx', group='rwx', other='rwx')"


def test_fileperms_int():
    assert int(FilePerms('r', 'r', '')) == 288
    assert int(FilePerms('rwx', 'rwx', 'rwx')) == 0o777
    assert int(FilePerms()) == 0


def test_fileperms_from_int():
    fp = FilePerms.from_int(0o644)
    assert fp.user == 'rw'
    assert fp.group == 'r'
    assert fp.other == 'r'
    assert int(fp) == 0o644


def test_fileperms_from_int_zero():
    fp = FilePerms.from_int(0)
    assert fp.user == ''
    assert fp.group == ''
    assert fp.other == ''
    assert int(fp) == 0


def test_fileperms_from_int_masks_full_perms():
    fp = FilePerms.from_int(0o1777)
    assert int(fp) == 0o777


def test_fileperms_from_path(tmp_path):
    p = tmp_path / "f"
    p.write_text("x")
    os.chmod(str(p), 0o640)
    fp = FilePerms.from_path(str(p))
    assert fp.user == 'rw'
    assert fp.group == 'r'
    assert fp.other == ''
    assert int(fp) == 0o640


def test_fileperms_invalid_char():
    with pytest.raises(ValueError):
        FilePerms(user='z')


def test_fileperms_invalid_type():
    with pytest.raises(TypeError):
        FilePerms(user=123)


def test_fileperms_set_same_value_noop():
    fp = FilePerms(user='r')
    int_before = int(fp)
    fp.user = 'r'
    assert int(fp) == int_before
    assert fp.user == 'r'


def test_fileperms_update_after_init():
    fp = FilePerms()
    fp.user = 'rw'
    assert fp.user == 'rw'
    assert int(fp) == 0o600


# ---------------------------------------------------------------------------
# path_to_unicode
# ---------------------------------------------------------------------------

def test_path_to_unicode_str():
    assert path_to_unicode("hello") == "hello"


def test_path_to_unicode_bytes():
    b = b"somepath"
    result = path_to_unicode(b)
    assert isinstance(result, str)
    assert result == "somepath"


# ---------------------------------------------------------------------------
# AtomicSaver / atomic_save
# ---------------------------------------------------------------------------

def test_atomic_save_basic(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as fo:
        fo.write(b"hello world")
    assert dest.read_bytes() == b"hello world"
    assert not os.path.exists(str(dest) + ".part")


def test_atomic_save_text_mode(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as fo:
        _ = fo.write("text data")
    assert dest.read_text() == "text data"


def test_atomic_save_exception_leaves_no_dest(tmp_path):
    dest = tmp_path / "out.bin"
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest)) as fo:
            fo.write(b"partial")
            1 / 0
    assert not dest.exists()
    assert not os.path.exists(str(dest) + ".part")


def test_atomic_save_no_overwrite_existing(tmp_path):
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"original")
    with pytest.raises(OSError) as ei:
        with atomic_save(str(dest), overwrite=False) as fo:
            fo.write(b"new")
    assert ei.value.errno == errno.EEXIST
    assert dest.read_bytes() == b"original"


def test_atomic_save_overwrite_existing(tmp_path):
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"original")
    with atomic_save(str(dest)) as fo:
        fo.write(b"new content")
    assert dest.read_bytes() == b"new content"


def test_atomicsaver_unexpected_kwarg(tmp_path):
    with pytest.raises(TypeError):
        atomic_save(str(tmp_path / "x"), bogus_kwarg=1)


def test_atomicsaver_custom_part_file(tmp_path):
    dest = tmp_path / "out.bin"
    saver = AtomicSaver(str(dest), part_file="my.part")
    assert saver.part_path == str(tmp_path / "my.part")
    with saver as fo:
        fo.write(b"data")
    assert dest.read_bytes() == b"data"
    assert not (tmp_path / "my.part").exists()


def test_atomicsaver_preexisting_part_raises(tmp_path):
    dest = tmp_path / "out.bin"
    part = tmp_path / "out.bin.part"
    part.write_bytes(b"stale")
    with pytest.raises(OSError):
        with atomic_save(str(dest)) as fo:
            fo.write(b"x")


def test_atomicsaver_overwrite_part(tmp_path):
    dest = tmp_path / "out.bin"
    part = tmp_path / "out.bin.part"
    part.write_bytes(b"stale")
    with atomic_save(str(dest), overwrite_part=True) as fo:
        fo.write(b"fresh")
    assert dest.read_bytes() == b"fresh"


def test_atomicsaver_keeps_part_on_exception(tmp_path):
    dest = tmp_path / "out.bin"
    part_path = str(dest) + ".part"
    with pytest.raises(RuntimeError):
        with atomic_save(str(dest), rm_part_on_exc=False) as fo:
            fo.write(b"partial")
            raise RuntimeError("boom")
    assert os.path.exists(part_path)
    assert dest.read_bytes() == b"partial"


def test_atomicsaver_file_perms(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest), file_perms=0o600) as fo:
        fo.write(b"x")
    mode = stat.S_IMODE(os.stat(str(dest)).st_mode)
    assert mode == 0o600


def test_atomicsaver_setup_explicit(tmp_path):
    dest = tmp_path / "out.bin"
    saver = AtomicSaver(str(dest))
    saver.setup()
    saver.part_file.write(b"manual")
    saver.__exit__(None, None, None)
    assert dest.read_bytes() == b"manual"


# ---------------------------------------------------------------------------
# iter_find_files
# ---------------------------------------------------------------------------

def _build_tree(root):
    (root / "a.txt").write_text("a")
    (root / "b.py").write_text("b")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    (sub / "d.py").write_text("d")
    sub2 = sub / "deep"
    sub2.mkdir()
    (sub2 / "e.txt").write_text("e")
    return root


def test_iter_find_files_single_pattern(tmp_path):
    _build_tree(tmp_path)
    found = sorted(
        os.path.basename(p) for p in iter_find_files(str(tmp_path), "*.txt")
    )
    assert found == ["a.txt", "c.txt", "e.txt"]


def test_iter_find_files_pattern_list(tmp_path):
    _build_tree(tmp_path)
    found = sorted(
        os.path.basename(p)
        for p in iter_find_files(str(tmp_path), ["*.txt", "*.py"])
    )
    assert found == ["a.txt", "b.py", "c.txt", "d.py", "e.txt"]


def test_iter_find_files_ignored(tmp_path):
    _build_tree(tmp_path)
    found = sorted(
        os.path.basename(p)
        for p in iter_find_files(str(tmp_path), "*.txt", ignored="c.txt")
    )
    assert found == ["a.txt", "e.txt"]


def test_iter_find_files_ignored_as_list(tmp_path):
    _build_tree(tmp_path)
    found = sorted(
        os.path.basename(p)
        for p in iter_find_files(str(tmp_path), "*.txt", ignored=["a.txt", "c.txt"])
    )
    assert found == ["e.txt"]


def test_iter_find_files_include_dirs(tmp_path):
    _build_tree(tmp_path)
    found = sorted(
        os.path.basename(p)
        for p in iter_find_files(str(tmp_path), "s*", include_dirs=True)
    )
    assert found == ["sub"]


def test_iter_find_files_max_depth(tmp_path):
    _build_tree(tmp_path)
    found = sorted(
        os.path.basename(p)
        for p in iter_find_files(str(tmp_path), "*.txt", max_depth=0)
    )
    assert found == ["a.txt"]

    found = sorted(
        os.path.basename(p)
        for p in iter_find_files(str(tmp_path), "*.txt", max_depth=1)
    )
    assert found == ["a.txt", "c.txt"]


# ---------------------------------------------------------------------------
# copy_tree / copytree
# ---------------------------------------------------------------------------

def test_copy_tree_into_new_dir(tmp_path):
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


def test_copy_tree_existing_dest_no_error(tmp_path):
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
    (src / "keep.txt").write_text("keep")
    (src / "skip.txt").write_text("skip")

    def ignore(path, names):
        return {n for n in names if n == "skip.txt"}

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst), ignore=ignore)
    assert (dst / "keep.txt").exists()
    assert not (dst / "skip.txt").exists()


def test_copytree_alias():
    assert copytree is copy_tree


# ---------------------------------------------------------------------------
# DummyFile
# ---------------------------------------------------------------------------

def test_dummyfile_attributes():
    df = DummyFile("somepath", mode="r")
    assert df.name == "somepath"
    assert df.mode == "r"
    assert df.closed is False
    assert df.fileno() == -1
    assert df.tell() == 0
    assert df.read() == ''
    assert df.readline() == ''
    assert df.readlines() == []
    assert df.encoding is None


def test_dummyfile_reads_and_writes():
    df = DummyFile("p")
    df.write("x")
    df.writelines(["a", "b"])
    df.flush()
    df.seek()
    df.truncate()
    assert df.read(5) == ''
    assert df.readline(5) == ''


def test_dummyfile_close_and_errors():
    df = DummyFile("p")
    df.close()
    assert df.closed is True
    for meth in (df.read, df.readline, df.readlines, df.flush, df.tell,
                 df.seek, df.truncate, df.writelines):
        with pytest.raises(ValueError):
            if meth == df.writelines:
                meth(["x"])
            else:
                meth()
    with pytest.raises(ValueError):
        df.write("x")


def test_dummyfile_iteration_and_context():
    df = DummyFile("p")
    with pytest.raises(StopIteration):
        next(df)
    with pytest.raises(StopIteration):
        df.next()
    df2 = DummyFile("p")
    result = df2.__enter__()
    assert result is None
    assert df2.__exit__(None, None, None) is None


# ---------------------------------------------------------------------------
# rotate_file
# ---------------------------------------------------------------------------

def test_rotate_file_keep_one(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("v0")
    rotate_file(str(base), keep=1)
    assert not base.exists()
    assert (tmp_path / "log.1.txt").read_text() == "v0"
    assert not (tmp_path / "log.2.txt").exists()


def test_rotate_file_multiple(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("v0")
    (tmp_path / "log.1.txt").write_text("v1")
    rotate_file(str(base), keep=3)
    assert not base.exists()
    assert (tmp_path / "log.1.txt").read_text() == "v0"
    assert (tmp_path / "log.2.txt").read_text() == "v1"
    assert not (tmp_path / "log.3.txt").exists()


def test_rotate_file_no_extension(tmp_path):
    base = tmp_path / "log"
    base.write_text("v0")
    rotate_file(str(base), keep=2)
    assert not base.exists()
    assert (tmp_path / "log.1").read_text() == "v0"


def test_rotate_file_nonexistent(tmp_path):
    base = tmp_path / "missing.txt"
    # Should be a no-op
    rotate_file(str(base), keep=3)
    assert not (tmp_path / "missing.1.txt").exists()


def test_rotate_file_invalid_keep(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("v0")
    with pytest.raises(ValueError):
        rotate_file(str(base), keep=0)


def test_rotate_file_on_directory(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d), keep=1)


def test_rotate_file_drops_beyond_keep(tmp_path):
    base = tmp_path / "log.txt"
    base.write_text("v0")
    (tmp_path / "log.1.txt").write_text("v1")
    (tmp_path / "log.2.txt").write_text("v2")
    (tmp_path / "log.3.txt").write_text("v3")
    rotate_file(str(base), keep=2)
    assert (tmp_path / "log.1.txt").read_text() == "v0"
    assert (tmp_path / "log.2.txt").read_text() == "v1"
    assert not (tmp_path / "log.3.txt").exists()
