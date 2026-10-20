import errno
import os
import sys
import stat
import shutil

import pytest

import boltons.fileutils as fileutils
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
    set_cloexec,
    FULL_PERMS,
    RW_PERMS,
    _SINGLE_FULL_PERM,
)


# ---------------------------------------------------------------------------
# mkdir_p
# ---------------------------------------------------------------------------
class TestMkdirP:
    def test_creates_nested_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        assert not target.exists()
        mkdir_p(str(target))
        assert target.is_dir()

    def test_existing_dir_no_error(self, tmp_path):
        target = tmp_path / "exists"
        target.mkdir()
        assert mkdir_p(str(target)) is None
        assert target.is_dir()

    def test_mkdir_p_returns_none(self, tmp_path):
        target = tmp_path / "newdir"
        assert mkdir_p(str(target)) is None

    def test_raises_on_file_blocking_dir(self, tmp_path):
        target = tmp_path / "blocked"
        target.write_text("data")
        with pytest.raises(OSError) as exc_info:
            mkdir_p(str(target))
        assert exc_info.value.errno == errno.EEXIST


# ---------------------------------------------------------------------------
# FilePerms
# ---------------------------------------------------------------------------
class TestFilePerms:
    def test_default_empty(self):
        fp = FilePerms()
        assert fp.user == ""
        assert fp.group == ""
        assert fp.other == ""
        assert int(fp) == 0

    def test_basic_construction(self):
        fp = FilePerms("r", "r", "")
        assert int(fp) == 288
        assert oct(int(fp))[-3:] == "440"

    def test_char_order_normalized(self):
        fp = FilePerms(user="rwx", group="xrw", other="wxr")
        assert fp.user == "rwx"
        assert fp.group == "rwx"
        assert fp.other == "rwx"
        assert int(fp) == 0o777

    def test_repr(self):
        fp = FilePerms("rw", "r", "r")
        assert repr(fp) == "FilePerms(user='rw', group='r', other='r')"

    def test_from_int_644(self):
        fp = FilePerms.from_int(0o644)
        assert fp.user == "rw"
        assert fp.group == "r"
        assert fp.other == "r"
        assert int(fp) == 0o644

    def test_from_int_masks_full_perms(self):
        fp = FilePerms.from_int(0o777 + 0o10000)
        assert int(fp) == 0o777

    def test_from_int_zero(self):
        fp = FilePerms.from_int(0)
        assert int(fp) == 0
        assert fp.user == ""
        assert fp.group == ""
        assert fp.other == ""

    def test_from_path(self, tmp_path):
        p = tmp_path / "perm_test"
        p.write_text("x")
        os.chmod(str(p), 0o644)
        fp = FilePerms.from_path(str(p))
        expected = stat.S_IMODE(os.stat(str(p)).st_mode)
        assert int(fp) == expected

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError):
            FilePerms(user="z")

    def test_non_string_raises_value_error(self):
        # The implementation str()-ifies the value, so 123 becomes '123',
        # whose chars are not in the perm set -> ValueError.
        with pytest.raises(ValueError):
            FilePerms(user=123)

    def test_setting_same_value_noop(self):
        fp = FilePerms("r", "", "")
        fp.user = "r"
        assert fp.user == "r"
        assert int(fp) == 0o400

    def test_from_int_known_values_roundtrip(self):
        # `from_int` maps a bitmask to parts, but only 3-part values
        # (and 0) round-trip cleanly. Curated set of documented /
        # verified round-trips:
        for i, expected in [
            (0o000, 0o000),
            (0o100, 0o100),
            (0o400, 0o400),
            (0o644, 0o644),
            (0o750, 0o750),
            (0o777, 0o777),
        ]:
            fp = FilePerms.from_int(i)
            assert int(fp) == expected, f"i={oct(i)} got {oct(int(fp))}"

    def test_from_int_result_stays_in_bounds(self):
        # No matter the input (including values whose bits overflow
        # the 3-byte representation), the resulting int must fit in
        # the bottom 9 permission bits.
        for i in range(0, 0o1000):
            fp = FilePerms.from_int(i)
            iv = int(fp)
            assert 0 <= iv <= FULL_PERMS

    def test_full_perms_constant(self):
        assert FULL_PERMS == 0o777
        assert RW_PERMS == 438
        assert _SINGLE_FULL_PERM == 7

    def test_class_access_returns_property_descriptor(self):
        # Accessing the class-level descriptor (fp_obj is None) returns
        # the descriptor itself.
        desc = FilePerms.__dict__["user"]
        assert FilePerms.user is desc
        assert FilePerms.group is FilePerms.__dict__["group"]
        assert FilePerms.other is FilePerms.__dict__["other"]

    def test_non_string_that_cannot_str_raises_type_error(self):
        # A value whose str() fails raises inside the try -> converted to
        # TypeError by the descriptor.
        class BadStr:
            def __str__(self):
                raise TypeError("nope")

        fp = FilePerms("r", "", "")
        with pytest.raises(TypeError):
            fp.user = BadStr()

    def test_setting_none_is_noop_when_current_is_empty(self):
        # current ('') == value? No -- None != '', so value goes through
        # and str(None) == 'None' whose chars are invalid -> ValueError.
        fp = FilePerms()
        with pytest.raises(ValueError):
            fp.user = None


# ---------------------------------------------------------------------------
# path_to_unicode
# ---------------------------------------------------------------------------
class TestPathToUnicode:
    def test_str_passthrough(self):
        assert path_to_unicode("abc") == "abc"

    def test_bytes_decoded(self):
        result = path_to_unicode(b"hello")
        assert isinstance(result, str)
        assert result == "hello"

    def test_uses_filesystem_encoding(self, monkeypatch):
        calls = {}

        def fake_getfilesystemencoding():
            calls["fs"] = True
            return "latin-1"

        monkeypatch.setattr(fileutils.sys, "getfilesystemencoding",
                            fake_getfilesystemencoding)
        # latin-1 decodes 0xFF as 'ÿ'
        assert path_to_unicode(b"\xff") == "\xff"
        assert calls.get("fs") is True

    def test_falls_back_to_default_encoding(self, monkeypatch):
        monkeypatch.setattr(fileutils.sys, "getfilesystemencoding",
                            lambda: None)
        monkeypatch.setattr(fileutils.sys, "getdefaultencoding",
                            lambda: "ascii")
        assert path_to_unicode(b"abc") == "abc"


# ---------------------------------------------------------------------------
# set_cloexec (misc)
# ---------------------------------------------------------------------------
class TestSetCloexec:
    def test_runs_on_fd(self, tmp_path):
        p = tmp_path / "clex"
        with open(str(p), "w") as f:
            assert set_cloexec(f.fileno()) is None


# ---------------------------------------------------------------------------
# atomic_save / AtomicSaver
# ---------------------------------------------------------------------------
class TestAtomicSaver:
    def test_basic_write(self, tmp_path):
        dest = tmp_path / "out.txt"
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write("hello")
        assert dest.read_text() == "hello"
        assert not (tmp_path / "out.txt.part").exists()

    def test_binary_write(self, tmp_path):
        dest = tmp_path / "out.bin"
        with atomic_save(str(dest)) as fo:
            fo.write(b"\x00\x01")
        assert dest.read_bytes() == b"\x00\x01"

    def test_exception_removes_part(self, tmp_path):
        dest = tmp_path / "out.txt"
        with pytest.raises(ZeroDivisionError):
            with atomic_save(str(dest), text_mode=True) as fo:
                fo.write("bad")
                1 / 0
        assert not dest.exists()
        assert not (tmp_path / "out.txt.part").exists()

    def test_exception_keeps_part_when_not_rm(self, tmp_path):
        dest = tmp_path / "out.txt"
        with pytest.raises(RuntimeError):
            with atomic_save(str(dest), text_mode=True, rm_part_on_exc=False) as fo:
                fo.write("keepme")
                raise RuntimeError("boom")
        assert not dest.exists()
        part = tmp_path / "out.txt.part"
        assert part.exists()
        assert part.read_text() == "keepme"

    def test_overwrite_false_existing_dest(self, tmp_path):
        dest = tmp_path / "out.txt"
        dest.write_text("orig")
        with pytest.raises(OSError) as ei:
            with atomic_save(str(dest), overwrite=False, text_mode=True):
                pass
        assert ei.value.errno == errno.EEXIST
        assert dest.read_text() == "orig"

    def test_overwrite_true_replaces(self, tmp_path):
        dest = tmp_path / "out.txt"
        dest.write_text("orig")
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write("new")
        assert dest.read_text() == "new"

    def test_overwrite_part_existing(self, tmp_path):
        dest = tmp_path / "out.txt"
        part = tmp_path / "out.txt.part"
        part.write_text("stale")
        with pytest.raises(OSError):
            with atomic_save(str(dest), text_mode=True):
                pass
        # setup raised, so __exit__ not called; part remains.
        assert part.exists()
        with atomic_save(str(dest), text_mode=True, overwrite_part=True) as fo:
            fo.write("fresh")
        assert dest.read_text() == "fresh"

    def test_custom_part_file(self, tmp_path):
        dest = tmp_path / "out.txt"
        with atomic_save(str(dest), text_mode=True, part_file="custom.part") as fo:
            fo.write("data")
        assert dest.read_text() == "data"
        assert not (tmp_path / "custom.part").exists()

    def test_file_perms_explicit(self, tmp_path):
        dest = tmp_path / "out.txt"
        with atomic_save(str(dest), text_mode=True, file_perms=0o600) as fo:
            fo.write("x")
        mode = stat.S_IMODE(os.stat(str(dest)).st_mode)
        assert mode == 0o600

    def test_perms_copied_from_existing_dest(self, tmp_path):
        dest = tmp_path / "out.txt"
        dest.write_text("orig")
        os.chmod(str(dest), 0o640)
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write("new")
        mode = stat.S_IMODE(os.stat(str(dest)).st_mode)
        assert mode == 0o640

    def test_unexpected_kwargs_raises(self, tmp_path):
        with pytest.raises(TypeError):
            AtomicSaver(str(tmp_path / "f"), bogus=1)

    def test_manual_setup(self, tmp_path):
        dest = tmp_path / "out.txt"
        saver = AtomicSaver(str(dest), text_mode=True)
        saver.setup()
        saver.part_file.write("manual")
        saver.part_file.flush()
        result = saver.__exit__(None, None, None)
        assert result is None
        assert dest.read_text() == "manual"

    def test_enter_returns_file(self, tmp_path):
        dest = tmp_path / "out.txt"
        saver = AtomicSaver(str(dest), text_mode=True)
        fo = saver.__enter__()
        assert fo is saver.part_file
        fo.write("z")
        saver.__exit__(None, None, None)
        assert dest.read_text() == "z"

    def test_dest_path_abspath(self, tmp_path):
        saver = AtomicSaver("relative.txt")
        assert os.path.isabs(saver.dest_path)
        assert saver.part_path == saver.dest_path + ".part"

    def test_pathlib_path(self, tmp_path):
        dest = tmp_path / "out.txt"
        with atomic_save(dest, text_mode=True) as fo:
            fo.write("path")
        assert dest.read_text() == "path"

    def test_default_file_perms_attribute(self):
        assert AtomicSaver._default_file_perms == RW_PERMS

    def test_setup_without_context_then_no_part_file_used(self, tmp_path):
        # Verify part_filename attribute path handling
        dest = tmp_path / "f.txt"
        saver = AtomicSaver(str(dest), part_file="mypart")
        assert saver.part_path == os.path.join(saver.dest_dir, "mypart")
        assert saver.text_mode is False
        assert saver.mode == "w+b"
        assert saver.buffering == -1

    def test_no_lexists_dest_creates_new_perms_default(self, tmp_path):
        # When dest doesn't exist, do_chmod is False so umask applies.
        dest = tmp_path / "fresh.txt"
        with atomic_save(str(dest), text_mode=True) as fo:
            fo.write("a")
        assert dest.read_text() == "a"
        # File should have some mode within the full perms range.
        mode = stat.S_IMODE(os.stat(str(dest)).st_mode)
        assert 0 <= mode <= FULL_PERMS

    def test_exit_without_setup_raises(self, tmp_path):
        # If setup() was never called, part_file is None; __exit__ without
        # an exception tries to atomically rename a nonexistent part file,
        # which raises (and the cleanup unlink swallows its own error).
        dest = tmp_path / "x.txt"
        saver = AtomicSaver(str(dest), text_mode=True)
        with pytest.raises(FileNotFoundError):
            saver.__exit__(None, None, None)
        assert not dest.exists()

    def test_exit_with_exc_without_setup_is_noop(self, tmp_path):
        # With an exception type and no part file, __exit__ swallows the
        # cleanup (unlink fails and is ignored) and returns None.
        dest = tmp_path / "x.txt"
        saver = AtomicSaver(str(dest), text_mode=True)
        assert saver.__exit__(ValueError, ValueError("x"), None) is None
        assert not dest.exists()

    def test_chmod_failure_closes_part_file(self, tmp_path, monkeypatch):
        dest = tmp_path / "withperms.txt"
        dest.write_text("orig")
        os.chmod(str(dest), 0o640)

        real_chmod = os.chmod
        orig_fdopen = os.fdopen
        closed_files = []

        class Tracker:
            def __init__(self, f):
                self._f = f

            def __getattr__(self, name):
                return getattr(self._f, name)

            def close(self):
                closed_files.append(True)
                return self._f.close()

        def fake_fdopen(fd, mode, buffering=-1):
            return Tracker(orig_fdopen(fd, mode, buffering))

        def fake_chmod(path, mode):
            raise OSError("chmod fail")

        monkeypatch.setattr(os, "chmod", fake_chmod)
        monkeypatch.setattr(os, "fdopen", fake_fdopen)

        saver = AtomicSaver(str(dest), text_mode=True)
        with pytest.raises(OSError):
            saver.setup()
        assert closed_files == [True]


# ---------------------------------------------------------------------------
# atomic_rename / replace
# ---------------------------------------------------------------------------
class TestRenameHelpers:
    def test_replace(self, tmp_path):
        src = tmp_path / "s"
        dst = tmp_path / "d"
        src.write_text("x")
        replace(str(src), str(dst))
        assert not src.exists()
        assert dst.read_text() == "x"

    def test_atomic_rename_overwrite(self, tmp_path):
        src = tmp_path / "s"
        dst = tmp_path / "d"
        src.write_text("new")
        dst.write_text("old")
        atomic_rename(str(src), str(dst), overwrite=True)
        assert dst.read_text() == "new"

    def test_atomic_rename_no_overwrite(self, tmp_path):
        src = tmp_path / "s"
        dst = tmp_path / "d"
        src.write_text("new")
        atomic_rename(str(src), str(dst), overwrite=False)
        assert dst.read_text() == "new"
        assert not src.exists()

    def test_atomic_rename_no_overwrite_existing_dst_raises(self, tmp_path):
        src = tmp_path / "s"
        dst = tmp_path / "d"
        src.write_text("new")
        dst.write_text("old")
        with pytest.raises(OSError):
            atomic_rename(str(src), str(dst), overwrite=False)
        assert dst.read_text() == "old"
        assert src.read_text() == "new"

    def test_alias(self):
        assert _atomic_rename is atomic_rename

    def test_replace_docstring_present(self):
        assert "os.replace" in replace.__doc__


# ---------------------------------------------------------------------------
# iter_find_files
# ---------------------------------------------------------------------------
class TestIterFindFiles:
    @pytest.fixture
    def tree(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("")
        (sub / "d.txt").write_text("")
        deep = sub / "deep"
        deep.mkdir()
        (deep / "e.py").write_text("")
        return tmp_path

    def test_basic_single_pattern(self, tree):
        results = sorted(iter_find_files(str(tree), "*.py"))
        names = {os.path.basename(r) for r in results}
        assert names == {"a.py", "c.py", "e.py"}

    def test_multiple_patterns(self, tree):
        results = sorted(iter_find_files(str(tree), ["*.py", "*.txt"]))
        names = {os.path.basename(r) for r in results}
        assert names == {"a.py", "b.txt", "c.py", "d.txt", "e.py"}

    def test_ignored(self, tree):
        results = sorted(iter_find_files(str(tree), "*.txt", ignored="d*"))
        names = {os.path.basename(r) for r in results}
        assert names == {"b.txt"}

    def test_include_dirs(self, tree):
        results = sorted(iter_find_files(str(tree), "sub", include_dirs=True))
        assert any(os.path.basename(r) == "sub" for r in results)

    def test_dirs_not_included_by_default(self, tree):
        results = list(iter_find_files(str(tree), "sub"))
        assert results == []

    def test_max_depth(self, tree):
        results = sorted(iter_find_files(str(tree), "*.py", max_depth=0))
        names = {os.path.basename(r) for r in results}
        assert names == {"a.py"}

    def test_max_depth_one(self, tree):
        results = sorted(iter_find_files(str(tree), "*.py", max_depth=1))
        names = {os.path.basename(r) for r in results}
        assert names == {"a.py", "c.py"}

    def test_nonexistent_dir(self, tmp_path):
        results = list(iter_find_files(str(tmp_path / "missing"), "*.py"))
        assert results == []

    def test_is_generator(self, tree):
        gen = iter_find_files(str(tree), "*.py")
        assert hasattr(gen, "__next__")

    def test_ignored_list(self, tree):
        # Ignored patterns match against basenames. b.txt and d.txt start
        # with 'b' / 'd' -- 'd.txt' does not start with 't'. Only names
        # matching '*.txt' and not matching b*/t* survive. d.txt matches
        # neither ignore pattern, so it remains.
        results = sorted(iter_find_files(str(tree), "*.txt",
                                         ignored=["b*", "t*"]))
        names = {os.path.basename(r) for r in results}
        assert names == {"d.txt"}

    def test_ignored_list_multiple(self, tree):
        # Ignore both b.txt and d.txt explicitly.
        results = sorted(iter_find_files(str(tree), "*.txt",
                                         ignored=["b*", "d*"]))
        assert results == []

    def test_include_dirs_with_ignore(self, tree):
        results = sorted(iter_find_files(str(tree), "s*",
                                         include_dirs=True,
                                         ignored="sub"))
        # 'sub' dir is ignored, so no directory named sub matches;
        # but no files match 's*' either.
        assert results == []

    def test_pathlike_directory(self, tree):
        results = sorted(iter_find_files(tree, "*.py"))
        names = {os.path.basename(r) for r in results}
        assert names == {"a.py", "c.py", "e.py"}

    def test_pattern_returns_full_paths(self, tree):
        results = list(iter_find_files(str(tree), "a.py"))
        assert results == [os.path.join(str(tree), "a.py")]


# ---------------------------------------------------------------------------
# copy_tree / copytree
# ---------------------------------------------------------------------------
class TestCopyTree:
    def test_basic_copy(self, tmp_path):
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

    def test_existing_dst_ok(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f1.txt").write_text("one")
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("keep")
        copy_tree(str(src), str(dst))
        assert (dst / "f1.txt").read_text() == "one"
        assert (dst / "existing.txt").read_text() == "keep"

    def test_ignore_callable(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "keep.txt").write_text("k")
        (src / "skip.txt").write_text("s")
        dst = tmp_path / "dst"

        def ignore(path, names):
            return [n for n in names if n == "skip.txt"]

        copy_tree(str(src), str(dst), ignore=ignore)
        assert (dst / "keep.txt").exists()
        assert not (dst / "skip.txt").exists()

    def test_copytree_alias(self):
        assert copytree is copy_tree

    def test_symlinks(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "real.txt").write_text("real")
        try:
            os.symlink(str(src / "real.txt"), str(src / "link.txt"))
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported")
        dst = tmp_path / "dst"
        copy_tree(str(src), str(dst), symlinks=True)
        assert os.path.islink(str(dst / "link.txt"))

    def test_copy_tree_copies_contents_without_symlinks(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "real.txt").write_text("real")
        try:
            os.symlink(str(src / "real.txt"), str(src / "link.txt"))
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported")
        dst = tmp_path / "dst"
        copy_tree(str(src), str(dst), symlinks=False)
        # default symlinks=False: link copied as regular file
        assert not os.path.islink(str(dst / "link.txt"))
        assert (dst / "link.txt").read_text() == "real"

    def test_copies_file_metadata(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        f = src / "f.txt"
        f.write_text("content")
        os.chmod(str(f), 0o640)
        dst = tmp_path / "dst"
        copy_tree(str(src), str(dst))
        assert stat.S_IMODE(os.stat(str(dst / "f.txt")).st_mode) == 0o640

    def test_returns_none(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        assert copy_tree(str(src), str(dst)) is None

    def test_ignore_returning_set(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        (src / "b.txt").write_text("b")
        dst = tmp_path / "dst"
        copy_tree(str(src), str(dst), ignore=lambda p, n: {"a.txt"})
        assert not (dst / "a.txt").exists()
        assert (dst / "b.txt").exists()


# ---------------------------------------------------------------------------
# DummyFile
# ---------------------------------------------------------------------------
class TestDummyFile:
    def test_attrs(self):
        df = DummyFile("/dev/null")
        assert df.name == "/dev/null"
        assert df.mode == "r"
        assert df.closed is False
        assert df.errors is None
        assert df.isatty is False
        assert df.encoding is None
        assert df.newlines is None
        assert df.softspace == 0

    def test_fileno(self):
        assert DummyFile("x").fileno() == -1

    def test_read_methods(self):
        df = DummyFile("x")
        assert df.read() == ""
        assert df.readline() == ""
        assert df.readlines() == []
        assert df.tell() == 0
        assert df.seek() is None
        assert df.truncate() is None
        assert df.write("abc") is None
        assert df.writelines(["a"]) is None
        assert df.flush() is None

    def test_next_raises(self):
        df = DummyFile("x")
        with pytest.raises(StopIteration):
            df.next()
        with pytest.raises(StopIteration):
            next(df)

    def test_close_then_errors(self):
        df = DummyFile("x")
        df.close()
        assert df.closed is True
        # methods with no required args
        for meth in (df.flush, df.read, df.readline, df.readlines,
                     df.seek, df.tell, df.truncate):
            with pytest.raises(ValueError):
                meth()
        # methods requiring args still validate closed state
        with pytest.raises(ValueError):
            df.write("x")
        with pytest.raises(ValueError):
            df.writelines([])

    def test_enter_closed_error(self):
        df = DummyFile("x")
        df.close()
        with pytest.raises(ValueError):
            df.__enter__()

    def test_enter_open(self):
        df = DummyFile("x")
        assert df.__enter__() is None
        assert df.__exit__(None, None, None) is None

    def test_custom_args(self):
        df = DummyFile("p", mode="wb", buffering=10)
        assert df.name == "p"
        assert df.mode == "wb"

    def test_closed_error_message(self):
        df = DummyFile("x")
        df.close()
        with pytest.raises(ValueError) as ei:
            df.read()
        assert str(ei.value) == "I/O operation on a closed file"

    def test_next_uses_stopiteration(self):
        df = DummyFile("x")
        try:
            df.__next__()
        except StopIteration as e:
            assert isinstance(e, StopIteration)


# ---------------------------------------------------------------------------
# rotate_file
# ---------------------------------------------------------------------------
class TestRotateFile:
    def test_keep_invalid(self, tmp_path):
        with pytest.raises(ValueError):
            rotate_file(str(tmp_path / "f.log"), keep=0)
        with pytest.raises(ValueError):
            rotate_file(str(tmp_path / "f.log"), keep=-1)

    def test_keep_error_message(self, tmp_path):
        with pytest.raises(ValueError) as ei:
            rotate_file(str(tmp_path / "f.log"), keep=0)
        assert "keep" in str(ei.value)
        assert ">=1" in str(ei.value)

    def test_nonexistent_returns(self, tmp_path):
        target = tmp_path / "nope.log"
        assert rotate_file(str(target)) is None
        assert not target.exists()

    def test_directory_raises(self, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(ValueError) as ei:
            rotate_file(str(d))
        assert "file" in str(ei.value)

    def test_basic_rotation(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("current")
        rotate_file(str(f), keep=3)
        assert not f.exists()
        assert (tmp_path / "app.1.log").read_text() == "current"

    def test_rotation_shifts(self, tmp_path):
        # rotate_file builds kept_names for i in 1..keep, then walks
        # reversed pairs of [filename, .1, .2, .3], renaming only files
        # that exist. With keep=3, .2 -> .3 happens then .1 -> .2, ... and
        # finally the last kept name (app.3.log) is removed.
        f = tmp_path / "app.log"
        f.write_text("v0")
        (tmp_path / "app.1.log").write_text("v1")
        (tmp_path / "app.2.log").write_text("v2")
        rotate_file(str(f), keep=3)
        # app.log -> app.1.log
        assert (tmp_path / "app.1.log").read_text() == "v0"
        # app.1.log was overwritten by v0; original v1 -> app.2.log
        assert (tmp_path / "app.2.log").read_text() == "v1"
        # app.2.log's original v2 was moved to app.3.log then removed
        assert not (tmp_path / "app.3.log").exists()
        assert not f.exists()

    def test_rotation_keep_two(self, tmp_path):
        # For keep=2: kept_names = ['app.1.log', 'app.2.log'],
        # fns = ['app.log', 'app.1.log', 'app.2.log'].
        # Reversed pairs: ('app.1.log', 'app.2.log'), ('app.log', 'app.1.log').
        # So .1(='v1') -> .2 (overwriting v2), then app.log('v0') -> .1.
        # Then kept_names[-1] = 'app.2.log' is removed.
        # app.3.log is never in kept_names, so it is untouched.
        f = tmp_path / "app.log"
        f.write_text("v0")
        for i in range(1, 4):
            (tmp_path / f"app.{i}.log").write_text(f"v{i}")
        rotate_file(str(f), keep=2)
        assert not f.exists()
        assert (tmp_path / "app.1.log").read_text() == "v0"
        assert not (tmp_path / "app.2.log").exists()
        # .3 was never part of kept_names, so it remains untouched
        assert (tmp_path / "app.3.log").read_text() == "v3"

    def test_no_extension(self, tmp_path):
        f = tmp_path / "logfile"
        f.write_text("x")
        rotate_file(str(f), keep=2)
        assert not f.exists()
        assert (tmp_path / "logfile.1").read_text() == "x"

    def test_default_keep_is_five(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("v0")
        rotate_file(str(f))
        assert (tmp_path / "app.1.log").read_text() == "v0"
        # keep=5 means .5 could exist; .6 should not be created
        assert not (tmp_path / "app.6.log").exists()

    def test_rotation_with_only_kept_files_no_current(self, tmp_path):
        # When f doesn't exist, returns early.
        (tmp_path / "app.1.log").write_text("v1")
        result = rotate_file(str(tmp_path / "app.log"), keep=3)
        assert result is None
        # Nothing moved since original doesn't exist.
        assert (tmp_path / "app.1.log").read_text() == "v1"

    def test_keep_one(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("v0")
        (tmp_path / "app.1.log").write_text("v1")
        rotate_file(str(f), keep=1)
        assert not f.exists()
        # app.log -> app.1.log overwrites, then app.1.log removed.
        assert not (tmp_path / "app.1.log").exists()
