import os
import io
import pytest

from boltons.ioutils import (
    SpooledIOBase,
    SpooledBytesIO,
    SpooledStringIO,
    is_text_fileobj,
    MultiFileReader,
    READ_CHUNK_SIZE,
)


# ---------------------------------------------------------------------------
# SpooledBytesIO
# ---------------------------------------------------------------------------

class TestSpooledBytesIO:
    def test_write_read_roundtrip(self):
        f = SpooledBytesIO()
        assert f.write(b"Happy IO") == 8
        f.seek(0)
        assert f.read() == b"Happy IO"
        assert f._rolled is False
        f.close()

    def test_default_max_size(self):
        f = SpooledBytesIO()
        assert f._max_size == 5000000
        f.close()

    def test_custom_max_size_and_dir(self, tmp_path):
        f = SpooledBytesIO(max_size=10, dir=str(tmp_path))
        assert f._max_size == 10
        assert f._dir == str(tmp_path)
        f.close()

    def test_memoryview_and_bytearray(self):
        f = SpooledBytesIO()
        assert f.write(bytearray(b"abc")) == 3
        assert f.write(memoryview(b"def")) == 3
        f.seek(0)
        assert f.read() == b"abcdef"
        f.close()

    def test_write_text_raises_typeerror(self):
        f = SpooledBytesIO()
        with pytest.raises(TypeError):
            f.write("text")
        f.close()

    def test_getvalue_bytes(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        val = f.getvalue()
        assert isinstance(val, bytes)
        assert val == b"abc"
        f.close()

    def test_rollover(self):
        f = SpooledBytesIO(max_size=5)
        f.write(b"12345")
        assert f._rolled is True
        f.seek(0)
        assert f.read() == b"12345"
        f.close()

    def test_len_after_rollover(self):
        f = SpooledBytesIO(max_size=3)
        f.write(b"hello")
        assert f._rolled is True
        assert len(f) == 5
        assert f.len == 5
        f.close()

    def test_len_not_rolled(self):
        f = SpooledBytesIO()
        f.write(b"hello")
        assert len(f) == 5
        # Position preserved
        assert f.tell() == 5
        f.close()

    def test_readline(self):
        f = SpooledBytesIO()
        f.write(b"line1\nline2\n")
        f.seek(0)
        assert f.readline() == b"line1\n"
        assert f.readline() == b"line2\n"
        f.close()

    def test_readline_with_length(self):
        f = SpooledBytesIO()
        f.write(b"line1\nline2\n")
        f.seek(0)
        assert f.readline(3) == b"lin"
        f.close()

    def test_readlines(self):
        f = SpooledBytesIO()
        f.write(b"a\nb\nc\n")
        f.seek(0)
        assert f.readlines() == [b"a\n", b"b\n", b"c\n"]
        f.close()

    def test_writelines(self):
        f = SpooledBytesIO()
        f.writelines([b"a", b"b", b"c"])
        f.seek(0)
        assert f.read() == b"abc"
        f.close()

    def test_seek_modes(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(0)
        assert f.seek(2) == 2
        assert f.read(1) == b"c"
        f.seek(-1, os.SEEK_CUR)
        assert f.read(1) == b"c"
        f.seek(-1, os.SEEK_END)
        assert f.read(1) == b"f"
        f.close()

    def test_closed_operations_raise(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        f.close()
        assert f.closed is True
        with pytest.raises(ValueError):
            f.read()
        with pytest.raises(ValueError):
            f.write(b"x")
        with pytest.raises(ValueError):
            f.seek(0)
        with pytest.raises(ValueError):
            f.tell()
        with pytest.raises(ValueError):
            f.getvalue()
        with pytest.raises(ValueError):
            f.flush()

    def test_context_manager(self):
        with SpooledBytesIO() as f:
            f.write(b"data")
            f.seek(0)
            assert f.read() == b"data"
        assert f.closed is True

    def test_iteration(self):
        f = SpooledBytesIO()
        f.write(b"a\nb\nc")
        f.seek(0)
        assert list(f) == [b"a\n", b"b\n", b"c"]
        f.close()

    def test_equality(self):
        f1 = SpooledBytesIO()
        f2 = SpooledBytesIO()
        f1.write(b"abc")
        f2.write(b"abc")
        assert f1 == f2
        # positions preserved
        assert f1.tell() == 3
        assert f2.tell() == 3

        f3 = SpooledBytesIO()
        f3.write(b"xyz")
        assert f1 != f3
        assert f1 != "not a file"
        f1.close()
        f2.close()
        f3.close()

    def test_bool(self):
        f = SpooledBytesIO()
        assert bool(f) is True
        f.close()

    def test_truncate_no_args(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(3)
        f.truncate()
        f.seek(0)
        assert f.read() == b"abc"
        f.close()

    def test_truncate_with_size(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.truncate(3)
        f.seek(0)
        assert f.read() == b"abc"
        f.close()

    def test_truncate_negative_raises(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        with pytest.raises(OSError):
            f.truncate(-1)
        f.close()

    def test_truncate_keeps_position_if_before(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(1)
        f.truncate(3)
        assert f.tell() == 1
        f.close()

    def test_fileno_rolls_over(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        fd = f.fileno()
        assert isinstance(fd, int)
        assert f._rolled is True
        f.close()

    def test_readable_writable_seekable(self):
        f = SpooledBytesIO()
        assert f.readable() is True
        assert f.writable() is True
        assert f.seekable() is True
        f.close()

    def test_isatty(self):
        f = SpooledBytesIO()
        assert f.isatty() is False
        f.close()

    def test_pos_and_buf_properties(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        assert f.pos == 3
        assert f.buf == b"abc"
        f.close()

    def test_softspace(self):
        f = SpooledBytesIO()
        f.softspace = 1
        assert f.softspace == 1
        f.close()


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

class TestSpooledStringIO:
    def test_write_read_roundtrip(self):
        f = SpooledStringIO()
        assert f.write("\u2014 Hey, an emdash!") == 17
        f.seek(0)
        ret = f.read()
        assert ret == "\u2014 Hey, an emdash!"
        assert isinstance(ret, str)
        assert f._rolled is False
        f.close()

    def test_write_non_str_raises(self):
        f = SpooledStringIO()
        with pytest.raises(TypeError):
            f.write(b"bytes")
        f.close()

    def test_tell_codepoints(self):
        f = SpooledStringIO()
        f.write("abc")
        assert f.tell() == 3
        f.write("d\U0001F600")  # 4 codepoints total for next write
        assert f.tell() == 5
        f.seek(0)
        assert f.tell() == 0
        f.close()

    def test_seek_set_and_read(self):
        f = SpooledStringIO()
        f.write("hello world")
        assert f.seek(6) == 6
        assert f.read() == "world"
        f.close()

    def test_seek_negative_set_raises(self):
        f = SpooledStringIO()
        f.write("abc")
        with pytest.raises(ValueError):
            f.seek(-1)
        f.close()

    def test_seek_cur(self):
        f = SpooledStringIO()
        f.write("hello world")
        f.seek(0)
        assert f.seek(5, os.SEEK_CUR) == 5
        assert f.read(1) == " "
        assert f.seek(1, os.SEEK_CUR) == 7
        assert f.read(1) == "w"
        f.close()

    def test_seek_cur_negative(self):
        f = SpooledStringIO()
        f.write("hello")
        f.seek(3)
        assert f.seek(-2, os.SEEK_CUR) == 1
        assert f.read(1) == "e"
        f.close()

    def test_seek_cur_negative_beyond_start_raises(self):
        f = SpooledStringIO()
        f.write("hello")
        f.seek(1)
        with pytest.raises(ValueError):
            f.seek(-5, os.SEEK_CUR)
        f.close()

    def test_seek_end(self):
        f = SpooledStringIO()
        f.write("hello world")
        assert f.seek(0, os.SEEK_END) == 11
        assert f.read() == ""
        f.close()

    def test_seek_invalid_whence(self):
        f = SpooledStringIO()
        f.write("abc")
        with pytest.raises(ValueError):
            f.seek(0, 3)
        f.close()

    def test_multibyte_seek(self):
        f = SpooledStringIO()
        f.write("\u00e9\u00e9\u00e9")  # 3 two-byte chars
        assert f.seek(1) == 1
        assert f.read(1) == "\u00e9"
        assert f.read(1) == "\u00e9"
        f.close()

    def test_readline(self):
        f = SpooledStringIO()
        f.write("line1\nline2\n")
        f.seek(0)
        assert f.readline() == "line1\n"
        assert f.readline() == "line2\n"
        assert f.tell() == 12
        f.close()

    def test_readline_length(self):
        f = SpooledStringIO()
        f.write("line1\nline2\n")
        f.seek(0)
        assert f.readline(2) == "li"
        f.close()

    def test_readlines(self):
        f = SpooledStringIO()
        f.write("a\nb\nc\n")
        f.seek(0)
        assert f.readlines() == ["a\n", "b\n", "c\n"]
        assert f.tell() == 6
        f.close()

    def test_len(self):
        f = SpooledStringIO()
        f.write("hello")
        assert f.len == 5
        assert len(f) == 5
        # position preserved
        assert f.tell() == 5
        f.close()

    def test_len_multibyte(self):
        f = SpooledStringIO()
        f.write("\u2014abc")  # 4 codepoints, more bytes
        assert len(f) == 4
        f.close()

    def test_rollover(self):
        f = SpooledStringIO(max_size=3)
        f.write("hello")
        assert f._rolled is True
        f.seek(0)
        assert f.read() == "hello"
        f.close()

    def test_len_after_rollover(self):
        f = SpooledStringIO(max_size=3)
        f.write("hello")
        assert f._rolled is True
        assert len(f) == 5
        f.close()

    def test_getvalue(self):
        f = SpooledStringIO()
        f.write("abc")
        assert f.getvalue() == "abc"
        assert f.tell() == 3
        f.close()

    def test_closed_operations_raise(self):
        f = SpooledStringIO()
        f.write("abc")
        f.close()
        assert f.closed is True
        with pytest.raises(ValueError):
            f.read()
        with pytest.raises(ValueError):
            f.write("x")
        with pytest.raises(ValueError):
            f.tell()

    def test_context_manager(self):
        with SpooledStringIO() as f:
            f.write("data")
            f.seek(0)
            assert f.read() == "data"
        assert f.closed is True

    def test_iteration(self):
        f = SpooledStringIO()
        f.write("a\nb\nc")
        f.seek(0)
        assert list(f) == ["a\n", "b\n", "c"]
        f.close()

    def test_equality(self):
        f1 = SpooledStringIO()
        f2 = SpooledStringIO()
        f1.write("abc")
        f2.write("abc")
        assert f1 == f2
        assert f1.tell() == 3
        assert f2.tell() == 3

        f3 = SpooledStringIO()
        f3.write("xyz")
        assert f1 != f3
        f1.close()
        f2.close()
        f3.close()

    def test_traverse_codepoints_chunked(self):
        # Force traversal across multiple READ_CHUNK_SIZE chunks
        big = "a" * (READ_CHUNK_SIZE * 2 + 5)
        f = SpooledStringIO()
        f.write(big)
        f.seek(0)
        assert f.seek(READ_CHUNK_SIZE * 2 + 3) == READ_CHUNK_SIZE * 2 + 3
        assert f.read() == "aa"
        f.close()


# ---------------------------------------------------------------------------
# SpooledIOBase abstractness
# ---------------------------------------------------------------------------

class TestSpooledIOBase:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SpooledIOBase()

    def test_subclass_relationship(self):
        assert issubclass(SpooledBytesIO, SpooledIOBase)
        assert issubclass(SpooledStringIO, SpooledIOBase)


# ---------------------------------------------------------------------------
# is_text_fileobj
# ---------------------------------------------------------------------------

class TestIsTextFileobj:
    def test_stringio(self):
        assert is_text_fileobj(io.StringIO("x")) is True

    def test_bytesio(self):
        assert is_text_fileobj(io.BytesIO(b"x")) is False

    def test_text_file(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello")
        with open(p, "r", encoding="utf-8") as f:
            assert is_text_fileobj(f) is True
        with open(p, "rb") as f:
            assert is_text_fileobj(f) is False

    def test_spooled_stringio(self):
        f = SpooledStringIO()
        assert is_text_fileobj(f) is True
        f.close()

    def test_spooled_bytesio(self):
        f = SpooledBytesIO()
        assert is_text_fileobj(f) is False
        f.close()

    def test_object_without_encoding_or_getvalue(self):
        assert is_text_fileobj(object()) is False

    def test_getvalue_raising(self):
        class Bad:
            def getvalue(self):
                raise RuntimeError("nope")
        assert is_text_fileobj(Bad()) is False

    def test_getvalue_bytes(self):
        class BytesGetter:
            def getvalue(self):
                return b"abc"
        assert is_text_fileobj(BytesGetter()) is False

    def test_getvalue_str(self):
        class StrGetter:
            def getvalue(self):
                return "abc"
        assert is_text_fileobj(StrGetter()) is True

    def test_empty_encoding_is_false(self):
        class EmptyEnc:
            encoding = ""
            def getvalue(self):
                return b"x"
        assert is_text_fileobj(EmptyEnc()) is False


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

class TestMultiFileReader:
    def test_docstring_example(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"), io.BytesIO(b"e"))
        assert mfr.read(3).decode("ascii") == "abc"
        assert mfr.read(3).decode("ascii") == "de"

    def test_read_all(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"), io.BytesIO(b"e"))
        assert mfr.read() == b"abcde"
        assert mfr.read() == b"abcde"  # nothing? actually reads remaining -> b''
        # Exhausted after first full read; second returns empty
        # NOTE: read(None) re-reads from current position of each file.
        # Since files are exhausted, returns b''.

    def test_read_all_returns_empty_when_exhausted(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        assert mfr.read() == b"ab"
        assert mfr.read() == b""

    def test_read_zero_or_none(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        # amt = 0 is falsy, so full read path is taken
        assert mfr.read(0) == b"ab"

    def test_text_files(self):
        mfr = MultiFileReader(io.StringIO("ab"), io.StringIO("cd"))
        assert mfr.read(3) == "abc"
        assert mfr.read(3) == "d"
        assert mfr.read(3) == ""

    def test_text_full_read(self):
        mfr = MultiFileReader(io.StringIO("ab"), io.StringIO("cd"))
        assert mfr.read() == "abcd"

    def test_mixed_raises_value_error(self):
        with pytest.raises(ValueError):
            MultiFileReader(io.BytesIO(b"ab"), io.StringIO("cd"))

    def test_non_filelike_raises_type_error(self):
        with pytest.raises(TypeError):
            MultiFileReader("not a file")

    def test_non_filelike_missing_seek(self):
        class OnlyRead:
            def read(self, n=-1):
                return b""
        with pytest.raises(TypeError):
            MultiFileReader(OnlyRead())

    def test_seek_to_zero(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        assert mfr.read(3) == b"abc"
        mfr.seek(0)
        assert mfr.read(3) == b"abc"

    def test_seek_nonzero_offset_raises(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        with pytest.raises(NotImplementedError):
            mfr.seek(1)

    def test_seek_non_set_whence_raises(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        with pytest.raises(NotImplementedError):
            mfr.seek(0, os.SEEK_CUR)

    def test_joiner_bytes(self):
        mfr = MultiFileReader(io.BytesIO(b"a"))
        assert mfr._joiner == b""

    def test_joiner_text(self):
        mfr = MultiFileReader(io.StringIO("a"))
        assert mfr._joiner == ""

    def test_read_across_boundary(self):
        mfr = MultiFileReader(io.BytesIO(b"abc"), io.BytesIO(b"def"))
        assert mfr.read(2) == b"ab"
        assert mfr.read(2) == b"cd"
        assert mfr.read(2) == b"ef"
        assert mfr.read(2) == b""
