import errno
import os
import stat
import sys

import pytest

from boltons.fileutils import (
    FilePerms,
    DummyFile,
    atomic_save,
    mkdir_p,
    iter_find_files,
    copy_tree,
    copytree,
    path_to_unicode,
    rotate_file,
    AtomicSaver,
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
    # Should not raise
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_raises_on_file_conflict(tmp_path):
    target = tmp_path / "afile"
    target.write_text("data")
    with pytest.raises(OSError) as exc_info:
        mkdir_p(str(target))
    assert exc_info.value.errno == errno.EEXIST


# ---------------------------------------------------------------------------
# FilePerms
# ---------------------------------------------------------------------------

def test_fileperms_basic_int():
    assert int(FilePerms("r", "r", "")) == 288  # 0o440


def test_fileperms_normalizes_order():
    fp = FilePerms(user="rwx", group="xrw", other="wxr")
    assert fp.user == "rwx"
    assert fp.group == "rwx"
    assert fp.other == "rwx"


def test_fileperms_empty_init():
    fp = FilePerms()
    assert fp.user == ""
    assert fp.group == ""
    assert fp.other == ""
    assert int(fp) == 0


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
    assert int(fp) == 0


def test_fileperms_from_int_masks():
    fp = FilePerms.from_int(0o7777)
    assert fp.user == "rwx"
    assert fp.group == "rwx"
    assert fp.other == "rwx"


def test_fileperms_from_path(tmp_path):
    p = tmp_path / "permfile"
    p.write_text("x")
    os.chmod(str(p), 0o640)
    fp = FilePerms.from_path(str(p))
    assert fp.user == "rw"
    assert fp.group == "r"
    assert fp.other == ""


def test_fileperms_setter_updates_integer():
    fp = FilePerms()
    fp.user = "rwx"
    assert fp.user == "rwx"
    assert int(fp) == 0o700


def test_fileperms_setter_invalid_chars():
    fp = FilePerms()
    with pytest.raises(ValueError):
        fp.user = "rz"


def test_fileperms_setter_type_error():
    fp = FilePerms()
    with pytest.raises(TypeError):
        fp.user = 5


def test_fileperms_setter_same_value():
    fp = FilePerms("rw")
    before = int(fp)
    fp.user = "rw"
    assert int(fp) == before


def test_fileperms_setter_dedupes_chars():
    fp = FilePerms()
    fp.user = "rrr"
    assert fp.user == "r"


# ---------------------------------------------------------------------------
# path_to_unicode
# ---------------------------------------------------------------------------

def test_path_to_unicode_str():
    assert path_to_unicode("hello") == "hello"


def test_path_to_unicode_bytes():
    b = "héllo".encode(sys.getfilesystemencoding() or sys.getdefaultencoding())
    assert path_to_unicode(b) == "héllo"


# ---------------------------------------------------------------------------
# atomic_save returns AtomicSaver
# ---------------------------------------------------------------------------

def test_atomic_save_returns_saver(tmp_path):
    dest = tmp_path / "file.txt"
    saver = atomic_save(str(dest))
    assert isinstance(saver, AtomicSaver)
    assert saver.dest_path == os.path.abspath(str(dest))


# ---------------------------------------------------------------------------
# AtomicSaver
# ---------------------------------------------------------------------------

def test_atomic_saver_writes_file(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as f:
        f.write(b"hello")
    assert dest.read_bytes() == b"hello"


def test_atomic_saver_text_mode(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as f:
        f.write("hello")
    assert dest.read_text() == "hello"


def test_atomic_saver_no_part_file_left(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as f:
        f.write(b"x")
    assert not (tmp_path / "out.bin.part").exists()


def test_atomic_saver_exception_removes_part(tmp_path):
    dest = tmp_path / "out.bin"
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest)) as f:
            f.write(b"partial")
            1 / 0
    assert not dest.exists()
    assert not (tmp_path / "out.bin.part").exists()


def test_atomic_saver_exception_keeps_part_when_configured(tmp_path):
    dest = tmp_path / "out.bin"
    with pytest.raises(ZeroDivisionError):
        with atomic_save(str(dest), rm_part_on_exc=False) as f:
            f.write(b"partial")
            1 / 0
    assert not dest.exists()
    assert (tmp_path / "out.bin.part").exists()


def test_atomic_saver_overwrite_false_existing(tmp_path):
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"orig")
    saver = AtomicSaver(str(dest), overwrite=False)
    with pytest.raises(OSError) as exc_info:
        saver.setup()
    assert exc_info.value.errno == errno.EEXIST


def test_atomic_saver_overwrite_true_existing(tmp_path):
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"orig")
    with atomic_save(str(dest)) as f:
        f.write(b"new")
    assert dest.read_bytes() == b"new"


def test_atomic_saver_custom_part_file(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest), part_file="my.part") as f:
        f.write(b"data")
    assert dest.read_bytes() == b"data"
    assert not (tmp_path / "my.part").exists()


def test_atomic_saver_overwrite_part(tmp_path):
    dest = tmp_path / "out.bin"
    part = tmp_path / "out.bin.part"
    part.write_bytes(b"leftover")
    with atomic_save(str(dest), overwrite_part=True) as f:
        f.write(b"data")
    assert dest.read_bytes() == b"data"


def test_atomic_saver_existing_part_raises(tmp_path):
    dest = tmp_path / "out.bin"
    part = tmp_path / "out.bin.part"
    part.write_bytes(b"leftover")
    saver = AtomicSaver(str(dest))
    with pytest.raises(OSError) as exc_info:
        saver.setup()
    assert exc_info.value.errno == errno.EEXIST


def test_atomic_saver_unexpected_kwarg(tmp_path):
    with pytest.raises(TypeError):
        AtomicSaver(str(tmp_path / "f.bin"), bogus=1)


def test_atomic_saver_copies_existing_perms(tmp_path):
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"orig")
    os.chmod(str(dest), 0o640)
    with atomic_save(str(dest)) as f:
        f.write(b"new")
    mode = stat.S_IMODE(os.stat(str(dest)).st_mode)
    assert mode == 0o640


def test_atomic_saver_explicit_file_perms(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest), file_perms=0o600) as f:
        f.write(b"new")
    mode = stat.S_IMODE(os.stat(str(dest)).st_mode)
    assert mode == 0o600


def test_atomic_saver_context_manager_returns_file(tmp_path):
    dest = tmp_path / "out.bin"
    saver = AtomicSaver(str(dest))
    with saver as f:
        assert f is saver.part_file
        assert not f.closed


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
    deeper = sub / "deep"
    deeper.mkdir()
    (deeper / "e.txt").write_text("e")


def test_iter_find_files_single_pattern(tmp_path):
    _make_tree(tmp_path)
    found = sorted(iter_find_files(str(tmp_path), "*.txt"))
    expected = sorted([
        str(tmp_path / "a.txt"),
        str(tmp_path / "sub" / "c.txt"),
        str(tmp_path / "sub" / "deep" / "e.txt"),
    ])
    assert found == expected


def test_iter_find_files_multiple_patterns(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(x) for x in
                   iter_find_files(str(tmp_path), ["*.txt", "*.py"]))
    assert found == ["a.txt", "b.py", "c.txt", "d.py", "e.txt"]


def test_iter_find_files_ignored(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(x) for x in
                   iter_find_files(str(tmp_path), "*.txt", ignored="c.txt"))
    assert found == ["a.txt", "e.txt"]


def test_iter_find_files_ignored_list(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(x) for x in
                   iter_find_files(str(tmp_path), "*.txt",
                                   ignored=["c.txt", "e.txt"]))
    assert found == ["a.txt"]


def test_iter_find_files_include_dirs(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(x) for x in
                   iter_find_files(str(tmp_path), "*", include_dirs=True))
    # dirs: sub, deep; files: a.txt, b.py, c.txt, d.py, e.txt
    assert "sub" in found
    assert "deep" in found
    assert "a.txt" in found


def test_iter_find_files_max_depth_zero(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(x) for x in
                   iter_find_files(str(tmp_path), "*.txt", max_depth=0))
    assert found == ["a.txt"]


def test_iter_find_files_max_depth_one(tmp_path):
    _make_tree(tmp_path)
    found = sorted(os.path.basename(x) for x in
                   iter_find_files(str(tmp_path), "*.txt", max_depth=1))
    assert found == ["a.txt", "c.txt"]


# ---------------------------------------------------------------------------
# copy_tree
# ---------------------------------------------------------------------------

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


def test_copy_tree_existing_dest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f1.txt").write_text("one")

    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "existing.txt").write_text("keep")

    copy_tree(str(src), str(dst))

    assert (dst / "f1.txt").read_text() == "one"
    assert (dst / "existing.txt").read_text() == "keep"


def test_copy_tree_with_ignore(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("k")
    (src / "skip.txt").write_text("s")

    def ignore(src_path, names):
        return [n for n in names if n == "skip.txt"]

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst), ignore=ignore)

    assert (dst / "keep.txt").exists()
    assert not (dst / "skip.txt").exists()


def test_copytree_alias():
    assert copytree is copy_tree


def test_copy_tree_copy_preserves_content(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "data.bin").write_bytes(b"\x00\x01\x02")

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))

    assert (dst / "data.bin").read_bytes() == b"\x00\x01\x02"


# ---------------------------------------------------------------------------
# DummyFile
# ---------------------------------------------------------------------------

def test_dummyfile_attributes():
    df = DummyFile("/dev/null")
    assert df.name == "/dev/null"
    assert df.mode == "r"
    assert df.closed is False
    assert df.fileno() == -1


def test_dummyfile_read_write():
    df = DummyFile("x")
    assert df.read() == ""
    assert df.readline() == ""
    assert df.readlines() == []
    df.write("abc")
    df.writelines(["a", "b"])
    df.flush()
    df.seek()
    assert df.tell() == 0
    df.truncate()


def test_dummyfile_next():
    df = DummyFile("x")
    with pytest.raises(StopIteration):
        next(df)
    with pytest.raises(StopIteration):
        df.next()


def test_dummyfile_close_and_errors():
    df = DummyFile("x")
    df.close()
    assert df.closed is True
    with pytest.raises(ValueError):
        df.flush()
    with pytest.raises(ValueError):
        df.read()
    with pytest.raises(ValueError):
        df.write("x")
    with pytest.raises(ValueError):
        df.__enter__()


def test_dummyfile_context_manager():
    df = DummyFile("x")
    with df as f:
        assert f is None


# ---------------------------------------------------------------------------
# rotate_file
# ---------------------------------------------------------------------------

def test_rotate_file_no_file(tmp_path):
    target = tmp_path / "missing.txt"
    # No-op when file doesn't exist
    rotate_file(str(target))
    assert not target.exists()


def test_rotate_file_basic(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("v1")
    rotate_file(str(f))
    assert not f.exists()
    assert (tmp_path / "log.1.txt").read_text() == "v1"


def test_rotate_file_multiple(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("v1")
    rotate_file(str(f))
    f.write_text("v2")
    rotate_file(str(f))
    f.write_text("v3")
    rotate_file(str(f))

    assert (tmp_path / "log.1.txt").read_text() == "v3"
    assert (tmp_path / "log.2.txt").read_text() == "v2"
    assert (tmp_path / "log.3.txt").read_text() == "v1"


def test_rotate_file_no_extension(tmp_path):
    f = tmp_path / "log"
    f.write_text("v1")
    rotate_file(str(f))
    assert not f.exists()
    assert (tmp_path / "log.1").read_text() == "v1"


def test_rotate_file_keep_limit(tmp_path):
    f = tmp_path / "log.txt"
    # Create 6 generations
    for i in range(6):
        f.write_text(f"v{i}")
        rotate_file(str(f), keep=3)
    # keep files 1..3
    assert (tmp_path / "log.1.txt").exists()
    assert (tmp_path / "log.2.txt").exists()
    assert (tmp_path / "log.3.txt").exists()
    # Anything beyond keep should be dropped
    assert not (tmp_path / "log.4.txt").exists()


def test_rotate_file_keep_invalid(tmp_path):
    f = tmp_path / "log.txt"
    with pytest.raises(ValueError):
        rotate_file(str(f), keep=0)
    with pytest.raises(ValueError):
        rotate_file(str(f), keep=-1)


def test_rotate_file_directory_raises(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d))
