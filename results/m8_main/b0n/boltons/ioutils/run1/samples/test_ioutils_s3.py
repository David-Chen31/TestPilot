import os
import io
import codecs
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
    def test_write_read_basic(self):
        f = SpooledBytesIO()
        assert f.write(b"Happy IO") == 8
        f.seek(0)
        assert f.read() == b"Happy IO"
        assert isinstance(f.getvalue(), bytes)

    def test_context_manager(self):
        with SpooledBytesIO() as f:
            f.write(b"abc")
            f.seek(0)
            assert f.read() == b"abc"

    def test_write_text_raises_type_error(self):
        f = SpooledBytesIO()
        with pytest.raises(TypeError):
            f.write("text")

    def test_write_bytearray_and_memoryview(self):
        f = SpooledBytesIO()
        assert f.write(bytearray(b"abc")) == 3
        assert f.write(memoryview(b"def")) == 3
        assert f.getvalue() == b"abcdef"

    def test_tell_and_pos(self):
        f = SpooledBytesIO()
        assert f.tell() == 0
        f.write(b"hello")
        assert f.tell() == 5
        assert f.pos == 5

    def test_len_and_len_magic(self):
        f = SpooledBytesIO()
        f.write(b"hello")
        assert f.len == 5
        assert len(f) == 5
        # len preserves position
        pos = f.tell()
        _ = f.len
        assert f.tell() == pos

    def test_readline_and_readlines(self):
        f = SpooledBytesIO()
        f.write(b"line1\nline2\nline3")
        f.seek(0)
        assert f.readline() == b"line1\n"
        assert f.readline(3) == b"lin"
        f.seek(0)
        assert f.readlines() == [b"line1\n", b"line2\n", b"line3"]
        f.seek(0)
        assert f.readlines(6) == [b"line1\n"]

    def test_seek_modes(self):
        f = SpooledBytesIO()
        f.write(b"0123456789")
        f.seek(0)
        assert f.read(3) == b"012"
        f.seek(1, os.SEEK_CUR)
        assert f.tell() == 4
        f.seek(-1, os.SEEK_END)
        assert f.read() == b"9"

    def test_writelines(self):
        f = SpooledBytesIO()
        f.writelines([b"a", b"b", b"c"])
        assert f.getvalue() == b"abc"

    def test_getvalue_preserves_position(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(2)
        assert f.getvalue() == b"abcdef"
        assert f.tell() == 2

    def test_truncate_none(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(3)
        f.truncate()
        assert f.getvalue() == b"abc"

    def test_truncate_size(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(0)
        f.truncate(3)
        assert f.getvalue() == b"abc"

    def test_truncate_negative_raises(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        with pytest.raises(OSError):
            f.truncate(-1)

    def test_truncate_preserves_pos_before_size(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(2)
        f.truncate(4)
        assert f.tell() == 2
        assert f.getvalue() == b"abcd"

    def test_closed_raises(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        f.close()
        with pytest.raises(ValueError):
            f.read()
        with pytest.raises(ValueError):
            f.write(b"x")

    def test_seekable_readable_writable(self):
        f = SpooledBytesIO()
        assert f.seekable() is True
        assert f.readable() is True
        assert f.writable() is True

    def test_bool(self):
        f = SpooledBytesIO()
        assert bool(f) is True

    def test_iter(self):
        f = SpooledBytesIO()
        f.write(b"a\nb\nc\n")
        f.seek(0)
        assert list(f) == [b"a\n", b"b\n", b"c\n"]

    def test_eq(self):
        a = SpooledBytesIO()
        a.write(b"a\nb\n")
        b = SpooledBytesIO()
        b.write(b"a\nb\n")
        c = SpooledBytesIO()
        c.write(b"a\nc\n")
        assert a == b
        assert a != c
        assert (a == "notafile") is False
        # positions preserved
        a.seek(1)
        b.seek(1)
        _ = (a == b)
        assert a.tell() == 1
        assert b.tell() == 1

    def test_rollover(self):
        f = SpooledBytesIO()
        assert f._rolled is False
        f.rollover()
        assert f._rolled is True
        assert f._rolled is True  # idempotent
        f.write(b"data")
        f.seek(0)
        assert f.read() == b"data"
        assert f.len == 4

    def test_auto_rollover_on_write(self):
        f = SpooledBytesIO(max_size=10)
        assert f._rolled is False
        f.write(b"0123456789")
        assert f._rolled is True
        assert f.getvalue() == b"0123456789"

    def test_fileno_rolls_over(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        fd = f.fileno()
        assert isinstance(fd, int)
        assert f._rolled is True

    def test_buf_and_file(self):
        f = SpooledBytesIO()
        f.write(b"xyz")
        assert f.buf == b"xyz"
        assert f._file is f.buffer

    def test_flush_isatty(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        f.flush()  # no error
        assert f.isatty() is False

    def test_softspace(self):
        f = SpooledBytesIO()
        f.softspace = 1
        assert f.softspace == 1


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

class TestSpooledStringIO:
    def test_write_read_basic(self):
        f = SpooledStringIO()
        assert f.write("\u2014 Hey, an emdash!") == len("\u2014 Hey, an emdash!")
        f.seek(0)
        val = f.read()
        assert isinstance(val, str)
        assert val == "\u2014 Hey, an emdash!"

    def test_write_bytes_raises_type_error(self):
        f = SpooledStringIO()
        with pytest.raises(TypeError) as exc:
            f.write(b"bytes")
        assert "str expected" in str(exc.value)

    def test_tell_codepoints(self):
        f = SpooledStringIO()
        f.write("\u2014ab")  # emdash is 3 bytes, 1 codepoint
        assert f.tell() == 3
        f.seek(0)
        assert f.read(1) == "\u2014"
        assert f.tell() == 1

    def test_len(self):
        f = SpooledStringIO()
        f.write("abc\u2014")
        assert f.len == 4
        assert len(f) == 4

    def test_getvalue(self):
        f = SpooledStringIO()
        f.write("hello w\u00f6rld")
        assert f.getvalue() == "hello w\u00f6rld"

    def test_seek_set(self):
        f = SpooledStringIO()
        f.write("abcdef")
        assert f.seek(2) == 2
        assert f.read() == "cdef"

    def test_seek_negative_set_raises(self):
        f = SpooledStringIO()
        f.write("abc")
        with pytest.raises(ValueError):
            f.seek(-1)

    def test_seek_cur(self):
        f = SpooledStringIO()
        f.write("abcdef")
        f.seek(1)
        f.seek(2, os.SEEK_CUR)
        assert f.tell() == 3
        assert f.read() == "def"

    def test_seek_cur_negative(self):
        f = SpooledStringIO()
        f.write("abcdef")
        f.seek(4)
        f.seek(-2, os.SEEK_CUR)
        assert f.tell() == 2
        assert f.read() == "cdef"

    def test_seek_cur_negative_result_raises(self):
        f = SpooledStringIO()
        f.write("abc")
        f.seek(1)
        with pytest.raises(ValueError):
            f.seek(-5, os.SEEK_CUR)

    def test_seek_end(self):
        f = SpooledStringIO()
        f.write("abcdef")
        f.seek(0, os.SEEK_END)
        assert f.tell() == 6
        f.seek(0)

    def test_seek_invalid_whence(self):
        f = SpooledStringIO()
        f.write("abc")
        with pytest.raises(ValueError):
            f.seek(0, 5)

    def test_multibyte_seek(self):
        f = SpooledStringIO()
        f.write("\u2014\u2014\u2014")  # three emdashes
        f.seek(1)
        assert f.read(1) == "\u2014"
        assert f.tell() == 2

    def test_readline(self):
        f = SpooledStringIO()
        f.write("line1\nline2\nline3")
        f.seek(0)
        assert f.readline() == "line1\n"
        assert f.readline(3) == "lin"
        assert f.tell() == 9

    def test_readlines(self):
        f = SpooledStringIO()
        f.write("a\nb\nc")
        f.seek(0)
        assert f.readlines() == ["a\n", "b\n", "c"]
        assert f.tell() == 5

    def test_writelines(self):
        f = SpooledStringIO()
        f.writelines(["a", "b", "c"])
        assert f.getvalue() == "abc"

    def test_iter(self):
        f = SpooledStringIO()
        f.write("a\nb\n")
        f.seek(0)
        assert list(f) == ["a\n", "b\n"]

    def test_rollover(self):
        f = SpooledStringIO()
        assert f._rolled is False
        f.write("hello")
        f.rollover()
        assert f._rolled is True
        assert f.getvalue() == "hello"

    def test_auto_rollover(self):
        f = SpooledStringIO(max_size=5)
        assert f._rolled is False
        f.write("123456")  # 6 bytes >= 5
        assert f._rolled is True
        f.seek(0)
        assert f.read() == "123456"

    def test_closed_raises(self):
        f = SpooledStringIO()
        f.write("abc")
        f.close()
        with pytest.raises(ValueError):
            f.read()
        with pytest.raises(ValueError):
            f.tell()

    def test_eq(self):
        a = SpooledStringIO()
        a.write("a\nb\n")
        b = SpooledStringIO()
        b.write("a\nb\n")
        c = SpooledStringIO()
        c.write("diff")
        assert a == b
        assert a != c
        assert (a == "x") is False


# ---------------------------------------------------------------------------
# SpooledIOBase
# ---------------------------------------------------------------------------

class TestSpooledIOBase:
    def test_is_base_of_implementations(self):
        assert issubclass(SpooledBytesIO, SpooledIOBase)
        assert issubclass(SpooledStringIO, SpooledIOBase)

    def test_check_closed_custom_msg(self):
        f = SpooledBytesIO()
        f.close()
        with pytest.raises(ValueError) as exc:
            f._checkClosed("custom message")
        assert "custom message" in str(exc.value)

    def test_default_max_size(self):
        f = SpooledBytesIO()
        assert f._max_size == 5000000
        assert f._dir is None


# ---------------------------------------------------------------------------
# is_text_fileobj
# ---------------------------------------------------------------------------

class TestIsTextFileobj:
    def test_string_io(self):
        assert is_text_fileobj(io.StringIO()) is True

    def test_bytes_io(self):
        assert is_text_fileobj(io.BytesIO()) is False

    def test_file_object(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello", encoding="utf-8")
        with open(str(p), encoding="utf-8") as f:
            assert is_text_fileobj(f) is True

    def test_binary_file_object(self, tmp_path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello")
        with open(str(p), "rb") as f:
            assert is_text_fileobj(f) is False

    def test_spooled_string(self):
        assert is_text_fileobj(SpooledStringIO()) is True

    def test_spooled_bytes(self):
        assert is_text_fileobj(SpooledBytesIO()) is False

    def test_no_read_attr(self):
        assert is_text_fileobj(object()) is False


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

class TestMultiFileReader:
    def test_bytes_read_across_files(self):
        mfr = MultiFileReader(
            io.BytesIO(b"ab"), io.BytesIO(b"cd"), io.BytesIO(b"e"))
        assert mfr.read(3) == b"abc"
        assert mfr.read(3) == b"de"

    def test_text_read_across_files(self):
        mfr = MultiFileReader(
            io.StringIO("ab"), io.StringIO("cd"), io.StringIO("e"))
        assert mfr.read(3) == "abc"
        assert mfr.read(3) == "de"

    def test_read_all_when_no_amount(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        assert mfr.read() == b"abcd"

    def test_read_returns_empty_when_exhausted(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        assert mfr.read(10) == b"ab"
        assert mfr.read(10) == b""

    def test_read_amt_falsy_zero(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        assert mfr.read(0) == b"abcd"

    def test_mixed_types_raises_value_error(self):
        with pytest.raises(ValueError):
            MultiFileReader(io.BytesIO(b"a"), io.StringIO("b"))

    def test_non_file_like_raises_type_error(self):
        with pytest.raises(TypeError):
            MultiFileReader(object())

    def test_missing_seek_raises_type_error(self):
        class OnlyRead:
            def read(self, amt=None):
                return b""

        with pytest.raises(TypeError):
            MultiFileReader(OnlyRead())

    def test_seek_to_start(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        assert mfr.read(2) == b"ab"
        mfr.seek(0)
        assert mfr.read(2) == b"ab"

    def test_seek_nonzero_raises(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        with pytest.raises(NotImplementedError):
            mfr.seek(1)

    def test_seek_nonzero_whence_raises(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        with pytest.raises(NotImplementedError):
            mfr.seek(0, os.SEEK_CUR)

    def test_empty_fileobjs(self):
        mfr = MultiFileReader()
        assert mfr.read() == b""
        assert mfr.read(5) == b""

    def test_joiner_type_bytes(self):
        mfr = MultiFileReader(io.BytesIO(b"a"))
        assert mfr._joiner == b""

    def test_joiner_type_text(self):
        mfr = MultiFileReader(io.StringIO("a"))
        assert mfr._joiner == ""
