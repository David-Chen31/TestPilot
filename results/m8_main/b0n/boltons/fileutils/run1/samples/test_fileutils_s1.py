import os
import stat
import errno
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
    target = tmp_path / "a" / "b"
    target.mkdir(parents=True)
    mkdir_p(str(target))
    assert target.is_dir()


def test_mkdir_p_existing_file_raises(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(OSError):
        mkdir_p(str(f))


# ---------------------------------------------------------------------------
# FilePerms
# ---------------------------------------------------------------------------

def test_fileperms_defaults_empty():
    fp = FilePerms()
    assert fp.user == ""
    assert fp.group == ""
    assert fp.other == ""
    assert int(fp) == 0


def test_fileperms_char_order_normalized():
    fp = FilePerms(user="rwx", group="xrw", other="wxr")
    assert (fp.user, fp.group, fp.other) == ("rwx", "rwx", "rwx")
    assert int(fp) == FULL_PERMS


def test_fileperms_int_and_repr():
    fp = FilePerms("r", "r", "")
    assert int(fp) == 288
    assert oct(288)[-3:] == "440"
    assert repr(fp) == "FilePerms(user='r', group='r', other='')"


def test_fileperms_setter_updates_integer():
    fp = FilePerms()
    fp.other = "r"
    assert fp.other == "r"
    assert int(fp) == 0o004


def test_fileperms_setter_same_value_noop():
    fp = FilePerms(user="r")
    before = int(fp)
    fp.user = "r"
    assert int(fp) == before


def test_fileperms_setter_invalid_chars():
    fp = FilePerms()
    with pytest.raises(ValueError):
        fp.user = "rz"


def test_fileperms_setter_non_string():
    fp = FilePerms()
    with pytest.raises(TypeError):
        fp.user = 123


def test_fileperms_from_int():
    fp = FilePerms.from_int(0o644)
    assert fp.user == "rw"
    assert fp.group == "r"
    assert fp.other == "r"
    assert repr(fp) == "FilePerms(user='rw', group='r', other='r')"


def test_fileperms_from_int_zero():
    fp = FilePerms.from_int(0)
    assert (fp.user, fp.group, fp.other) == ("", "", "")
    assert int(fp) == 0


def test_fileperms_from_int_full():
    fp = FilePerms.from_int(0o777)
    assert int(fp) == 0o777


def test_fileperms_from_path(tmp_path):
    f = tmp_path / "permfile"
    f.write_text("data")
    os.chmod(str(f), 0o640)
    fp = FilePerms.from_path(str(f))
    assert int(fp) == 0o640
    assert fp.user == "rw"
    assert fp.group == "r"
    assert fp.other == ""


# ---------------------------------------------------------------------------
# path_to_unicode
# ---------------------------------------------------------------------------

def test_path_to_unicode_str():
    assert path_to_unicode("abc") == "abc"


def test_path_to_unicode_bytes():
    raw = "abc".encode("ascii")
    assert path_to_unicode(raw) == "abc"


# ---------------------------------------------------------------------------
# AtomicSaver / atomic_save
# ---------------------------------------------------------------------------

def test_atomic_saver_basic_text(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True) as f:
        f.write("hello")
    assert dest.read_text() == "hello"
    assert not (tmp_path / "out.txt.part").exists()


def test_atomic_saver_basic_binary(tmp_path):
    dest = tmp_path / "out.bin"
    with atomic_save(str(dest)) as f:
        f.write(b"bytes!")
    assert dest.read_bytes() == b"bytes!"


def test_atomic_saver_part_file_name(tmp_path):
    dest = tmp_path / "out.txt"
    with AtomicSaver(str(dest), text_mode=True, part_file="custom.part") as f:
        assert os.path.basename(f.name) == "custom.part"
        f.write("x")
    assert dest.read_text() == "x"
    assert not (tmp_path / "custom.part").exists()


def test_atomic_saver_exception_removes_part(tmp_path):
    dest = tmp_path / "out.txt"
    with pytest.raises(RuntimeError):
        with atomic_save(str(dest), text_mode=True) as f:
            f.write("partial")
            raise RuntimeError("boom")
    assert not dest.exists()
    assert not (tmp_path / "out.txt.part").exists()


def test_atomic_saver_exception_keeps_part(tmp_path):
    dest = tmp_path / "out.txt"
    with pytest.raises(RuntimeError):
        with AtomicSaver(str(dest), text_mode=True, rm_part_on_exc=False) as f:
            f.write("partial")
            raise RuntimeError("boom")
    assert (tmp_path / "out.txt.part").exists()


def test_atomic_saver_no_overwrite_raises(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("original")
    saver = AtomicSaver(str(dest), overwrite=False, text_mode=True)
    with pytest.raises(OSError) as ei:
        saver.setup()
    assert ei.value.errno == errno.EEXIST
    assert dest.read_text() == "original"


def test_atomic_saver_overwrite_existing(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    with atomic_save(str(dest), text_mode=True) as f:
        f.write("new")
    assert dest.read_text() == "new"


def test_atomic_saver_existing_part_raises(tmp_path):
    dest = tmp_path / "out.txt"
    part = tmp_path / "out.txt.part"
    part.write_text("stale")
    saver = AtomicSaver(str(dest), text_mode=True)
    with pytest.raises(OSError):
        saver.setup()


def test_atomic_saver_overwrite_part(tmp_path):
    dest = tmp_path / "out.txt"
    part = tmp_path / "out.txt.part"
    part.write_text("stale")
    with atomic_save(str(dest), text_mode=True, overwrite_part=True) as f:
        f.write("fresh")
    assert dest.read_text() == "fresh"
    assert not part.exists()


def test_atomic_saver_copies_existing_perms(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old")
    os.chmod(str(dest), 0o600)
    with atomic_save(str(dest), text_mode=True) as f:
        f.write("new")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o600


def test_atomic_saver_explicit_file_perms(tmp_path):
    dest = tmp_path / "out.txt"
    with atomic_save(str(dest), text_mode=True, file_perms=0o640) as f:
        f.write("perm")
    assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o640


def test_atomic_saver_unexpected_kwargs():
    with pytest.raises(TypeError):
        AtomicSaver("/tmp/whatever", bogus=True)


# ---------------------------------------------------------------------------
# iter_find_files
# ---------------------------------------------------------------------------

@pytest.fixture
def find_tree(tmp_path):
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


def test_iter_find_files_single_pattern(find_tree):
    results = sorted(iter_find_files(str(find_tree), "*.txt"))
    names = sorted(os.path.basename(r) for r in results)
    assert names == ["a.txt", "c.txt", "e.txt"]


def test_iter_find_files_list_patterns(find_tree):
    results = list(iter_find_files(str(find_tree), ["*.txt", "*.py"]))
    names = sorted(os.path.basename(r) for r in results)
    assert names == ["a.txt", "b.py", "c.txt", "d.py", "e.txt"]


def test_iter_find_files_ignored(find_tree):
    results = sorted(iter_find_files(str(find_tree), "*.txt", ignored="c.*"))
    names = sorted(os.path.basename(r) for r in results)
    assert names == ["a.txt", "e.txt"]


def test_iter_find_files_include_dirs(find_tree):
    results = list(iter_find_files(str(find_tree), "sub", include_dirs=True))
    assert len(results) == 1
    assert os.path.basename(results[0]) == "sub"


def test_iter_find_files_include_dirs_ignored(find_tree):
    results = list(iter_find_files(
        str(find_tree), "sub", ignored="sub", include_dirs=True))
    assert results == []


def test_iter_find_files_max_depth_zero(find_tree):
    results = sorted(iter_find_files(str(find_tree), "*.txt", max_depth=0))
    names = sorted(os.path.basename(r) for r in results)
    assert names == ["a.txt"]


def test_iter_find_files_max_depth_one(find_tree):
    results = sorted(iter_find_files(str(find_tree), "*.txt", max_depth=1))
    names = sorted(os.path.basename(r) for r in results)
    assert names == ["a.txt", "c.txt"]


def test_iter_find_files_no_matches(find_tree):
    results = list(iter_find_files(str(find_tree), "*.nope"))
    assert results == []


# ---------------------------------------------------------------------------
# copy_tree / copytree
# ---------------------------------------------------------------------------

def test_copy_tree_basic(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "one.txt").write_text("1")
    sub = src / "sub"
    sub.mkdir()
    (sub / "two.txt").write_text("2")

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst))

    assert (dst / "one.txt").read_text() == "1"
    assert (dst / "sub" / "two.txt").read_text() == "2"


def test_copy_tree_existing_dest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "one.txt").write_text("1")
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "existing.txt").write_text("keep")

    copy_tree(str(src), str(dst))

    assert (dst / "one.txt").read_text() == "1"
    assert (dst / "existing.txt").read_text() == "keep"


def test_copy_tree_ignore(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("k")
    (src / "skip.txt").write_text("s")

    def ignore(path, names):
        return [n for n in names if n == "skip.txt"]

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst), ignore=ignore)

    assert (dst / "keep.txt").exists()
    assert not (dst / "skip.txt").exists()


def test_copy_tree_symlinks(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    real = src / "real.txt"
    real.write_text("real")
    link = src / "link.txt"
    try:
        os.symlink(str(real), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported")

    dst = tmp_path / "dst"
    copy_tree(str(src), str(dst), symlinks=True)
    assert os.path.islink(str(dst / "link.txt"))


def test_copytree_alias_same_object():
    assert copytree is copy_tree


# ---------------------------------------------------------------------------
# DummyFile
# ---------------------------------------------------------------------------

def test_dummyfile_attributes():
    df = DummyFile("nope", mode="r")
    assert df.name == "nope"
    assert df.mode == "r"
    assert df.closed is False
    assert df.fileno() == -1
    assert df.isatty is False


def test_dummyfile_reads():
    df = DummyFile("x")
    assert df.read() == ""
    assert df.readline() == ""
    assert df.readlines() == []
    assert df.tell() == 0
    df.flush()
    df.seek()
    df.truncate()
    df.write("abc")
    df.writelines(["a", "b"])


def test_dummyfile_iteration():
    df = DummyFile("x")
    with pytest.raises(StopIteration):
        next(df)
    with pytest.raises(StopIteration):
        df.next()


def test_dummyfile_closed_raises():
    df = DummyFile("x")
    df.close()
    assert df.closed is True
    for method in ("flush", "read", "readline", "readlines", "seek",
                   "tell", "truncate", "write", "writelines"):
        with pytest.raises(ValueError):
            getattr(df, method)("x") if method in ("write",) else getattr(df, method)()
    with pytest.raises(ValueError):
        df.__enter__()
    with pytest.raises(StopIteration):
        next(df)


def test_dummyfile_context_manager():
    df = DummyFile("x")
    with df as entered:
        assert entered is None


# ---------------------------------------------------------------------------
# rotate_file
# ---------------------------------------------------------------------------

def test_rotate_file_keep_invalid():
    with pytest.raises(ValueError):
        rotate_file("whatever.txt", keep=0)


def test_rotate_file_missing_file(tmp_path):
    target = tmp_path / "missing.log"
    rotate_file(str(target))
    assert list(tmp_path.iterdir()) == []


def test_rotate_file_is_directory(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        rotate_file(str(d))


def test_rotate_file_basic(tmp_path):
    f = tmp_path / "app.log"
    f.write_text("current")
    rotate_file(str(f))
    assert not f.exists()
    assert (tmp_path / "app.1.log").read_text() == "current"


def test_rotate_file_rotation_chain(tmp_path):
    f = tmp_path / "app.log"
    f.write_text("current")
    (tmp_path / "app.1.log").write_text("one")
    (tmp_path / "app.2.log").write_text("two")

    rotate_file(str(f), keep=3)

    assert not f.exists()
    assert (tmp_path / "app.1.log").read_text() == "current"
    assert (tmp_path / "app.2.log").read_text() == "one"
    assert (tmp_path / "app.3.log").read_text() == "two"


def test_rotate_file_drops_beyond_keep(tmp_path):
    f = tmp_path / "app.log"
    f.write_text("current")
    (tmp_path / "app.1.log").write_text("one")
    (tmp_path / "app.2.log").write_text("two")

    rotate_file(str(f), keep=2)

    assert not f.exists()
    assert (tmp_path / "app.1.log").read_text() == "current"
    assert (tmp_path / "app.2.log").read_text() == "one"


def test_rotate_file_no_extension(tmp_path):
    f = tmp_path / "logfile"
    f.write_text("current")
    rotate_file(str(f))
    assert not f.exists()
    assert (tmp_path / "logfile.1").read_text() == "current"
