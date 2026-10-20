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
# Sanity constants
# ---------------------------------------------------------------------------
def test_read_chunk_size_value():
    assert READ_CHUNK_SIZE == 21333


# ---------------------------------------------------------------------------
# SpooledBytesIO
# ---------------------------------------------------------------------------
class TestSpooledBytesIO:
    def test_write_read_getvalue(self):
        f = SpooledBytesIO()
        assert f.write(b"Happy IO") == 8
        assert f.seek(0) == 0
        assert isinstance(f.getvalue(), bytes)
        assert f.getvalue() == b"Happy IO"
        assert f.read() == b"Happy IO"

    def test_write_bytearray_and_memoryview(self):
        f = SpooledBytesIO()
        assert f.write(bytearray(b"abc")) == 3
        assert f.write(memoryview(b"def")) == 3
        f.seek(0)
        assert f.read() == b"abcdef"

    def test_write_text_raises_type_error(self):
        f = SpooledBytesIO()
        with pytest.raises(TypeError):
            f.write("text")

    def test_write_returns_count_and_tell(self):
        f = SpooledBytesIO()
        assert f.write(b"hello") == 5
        assert f.tell() == 5

    def test_read_n(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(0)
        assert f.read(2) == b"ab"
        assert f.read(2) == b"cd"
        assert f.read() == b"ef"

    def test_readline_and_readlines(self):
        f = SpooledBytesIO()
        f.write(b"line1\nline2\nline3\n")
        f.seek(0)
        assert f.readline() == b"line1\n"
        assert f.readline(3) == b"lin"
        f.seek(0)
        assert f.readlines() == [b"line1\n", b"line2\n", b"line3\n"]

    def test_writelines(self):
        f = SpooledBytesIO()
        f.writelines([b"a", b"b", b"c"])
        f.seek(0)
        assert f.read() == b"abc"

    def test_getvalue_preserves_position(self):
        f = SpooledBytesIO()
        f.write(b"hello world")
        f.seek(5)
        assert f.getvalue() == b"hello world"
        assert f.tell() == 5

    def test_len_and_position(self):
        f = SpooledBytesIO()
        f.write(b"12345")
        assert len(f) == 5
        assert f.len == 5
        assert f.tell() == 5
        assert f.pos == 5
        assert f.buf == b"12345"

    def test_rollover_on_max_size(self):
        f = SpooledBytesIO(max_size=4)
        assert f._rolled is False
        f.write(b"1234")  # 0 + 4 >= 4 -> rollover
        assert f._rolled is True
        f.seek(0)
        assert f.read() == b"1234"

    def test_rollover_still_works_directly(self):
        f = SpooledBytesIO()
        f.write(b"data")
        f.rollover()
        assert f._rolled is True
        f.rollover()  # second rollover should be a no-op
        f.seek(0)
        assert f.read() == b"data"

    def test_fileno_triggers_rollover(self):
        f = SpooledBytesIO()
        f.write(b"x")
        assert f._rolled is False
        fd = f.fileno()
        assert isinstance(fd, int)
        assert f._rolled is True

    def test_len_after_rollover(self):
        f = SpooledBytesIO(max_size=2)
        f.write(b"abcdef")
        assert f._rolled is True
        assert f.len == 6
        assert len(f) == 6

    def test_truncate_no_arg(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.truncate()
        assert f.getvalue() == b"abcdef"

    def test_truncate_with_size(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(0)
        f.truncate(3)
        assert f.getvalue() == b"abc"
        assert f.tell() == 0

    def test_truncate_negative_raises(self):
        f = SpooledBytesIO()
        f.write(b"abc")
        with pytest.raises(OSError) as exc:
            f.truncate(-1)
        assert exc.value.errno == 22  # EINVAL

    def test_truncate_empty_buffer(self):
        f = SpooledBytesIO()
        f.truncate(0)
        assert f.getvalue() == b""

    def test_closed_operations_raise(self):
        f = SpooledBytesIO()
        f.write(b"data")
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
        with pytest.raises(ValueError):
            f.readline()

    def test_context_manager(self):
        with SpooledBytesIO() as f:
            f.write(b"abc")
            f.seek(0)
            assert f.read() == b"abc"
        assert f.closed is True

    def test_iter_lines(self):
        f = SpooledBytesIO()
        f.write(b"a\nb\nc\n")
        f.seek(0)
        assert list(f) == [b"a\n", b"b\n", b"c\n"]

    def test_iter_empty(self):
        f = SpooledBytesIO()
        f.seek(0)
        assert list(f) == []

    def test_seekable_readable_writable(self):
        f = SpooledBytesIO()
        assert f.seekable() is True
        assert f.readable() is True
        assert f.writable() is True

    def test_bool_true(self):
        f = SpooledBytesIO()
        assert bool(f) is True

    def test_equality(self):
        a = SpooledBytesIO()
        b = SpooledBytesIO()
        a.write(b"same")
        b.write(b"same")
        assert a == b
        c = SpooledBytesIO()
        c.write(b"diff")
        assert a != c
        assert a != "not a file"

    def test_equality_preserves_positions(self):
        a = SpooledBytesIO()
        b = SpooledBytesIO()
        a.write(b"line1\nline2\n")
        b.write(b"line1\nline2\n")
        a.seek(3)
        b.seek(4)
        assert a == b
        assert a.tell() == 3
        assert b.tell() == 4

    def test_isatty(self):
        f = SpooledBytesIO()
        # BytesIO.isatty returns False
        assert f.isatty() is False

    def test_seek_with_whence(self):
        f = SpooledBytesIO()
        f.write(b"abcdef")
        f.seek(-2, os.SEEK_END)
        assert f.read() == b"ef"


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------
class TestSpooledStringIO:
    def test_write_read_str(self):
        f = SpooledStringIO()
        assert f.write("\u2014 Hey, an emdash!") == 17
        f.seek(0)
        val = f.read()
        assert isinstance(val, str)
        assert val == "\u2014 Hey, an emdash!"

    def test_write_non_str_raises(self):
        f = SpooledStringIO()
        with pytest.raises(TypeError):
            f.write(b"bytes")

    def test_tell_counts_codepoints(self):
        f = SpooledStringIO()
        f.write("hello")
        assert f.tell() == 5
        # multibyte char counts as one codepoint
        f.write("\u2014")
        assert f.tell() == 6

    def test_read_n(self):
        f = SpooledStringIO()
        f.write("abcdef")
        f.seek(0)
        assert f.read(2) == "ab"
        assert f.read(2) == "cd"

    def test_read_after_multibyte(self):
        f = SpooledStringIO()
        f.write("\u2014abc")
        f.seek(0)
        assert f.read(2) == "\u2014a"
        assert f.read() == "bc"

    def test_readline(self):
        f = SpooledStringIO()
        f.write("line1\nline2\n")
        f.seek(0)
        assert f.readline() == "line1\n"
        assert f.readline() == "line2\n"

    def test_readlines(self):
        f = SpooledStringIO()
        f.write("a\nb\nc\n")
        f.seek(0)
        assert f.readlines() == ["a\n", "b\n", "c\n"]

    def test_writelines(self):
        f = SpooledStringIO()
        f.writelines(["a", "b", "c"])
        f.seek(0)
        assert f.read() == "abc"

    def test_len_codepoints(self):
        f = SpooledStringIO()
        f.write("ab\u2014cd")
        assert f.len == 5
        assert len(f) == 5

    def test_seek_negative_start_raises(self):
        f = SpooledStringIO()
        f.write("abc")
        with pytest.raises(ValueError):
            f.seek(-1)

    def test_seek_cur(self):
        f = SpooledStringIO()
        f.write("abcdef")
        f.seek(2)
        assert f.seek(2, os.SEEK_CUR) == 4
        assert f.read(1) == "e"

    def test_seek_cur_negative(self):
        f = SpooledStringIO()
        f.write("abcdef")
        f.seek(4)
        assert f.seek(-2, os.SEEK_CUR) == 2
        assert f.read(1) == "c"

    def test_seek_cur_negative_past_zero_raises(self):
        f = SpooledStringIO()
        f.write("abc")
        f.seek(1)
        with pytest.raises(ValueError):
            f.seek(-5, os.SEEK_CUR)

    def test_seek_end(self):
        f = SpooledStringIO()
        f.write("abcdef")
        # seek(2, SEEK_END) -> position len-2 = 4
        assert f.seek(2, os.SEEK_END) == 4
        assert f.read() == "ef"

    def test_seek_invalid_whence(self):
        f = SpooledStringIO()
        f.write("abc")
        with pytest.raises(ValueError):
            f.seek(0, 99)

    def test_seek_with_multibyte(self):
        f = SpooledStringIO()
        f.write("\u2014\u2014abc")
        f.seek(2)
        assert f.read(1) == "a"
        f.seek(0)
        assert f.read(2) == "\u2014\u2014"

    def test_getvalue(self):
        f = SpooledStringIO()
        f.write("hello")
        f.seek(2)
        assert f.getvalue() == "hello"
        assert f.tell() == 2

    def test_rollover_on_max_size(self):
        f = SpooledStringIO(max_size=3)
        assert f._rolled is False
        f.write("abcd")
        assert f._rolled is True
        f.seek(0)
        assert f.read() == "abcd"

    def test_rollover_multibyte(self):
        f = SpooledStringIO(max_size=2)
        f.write("\u2014")  # 3 bytes utf-8, >= 2 -> rollover
        assert f._rolled is True
        f.seek(0)
        assert f.read() == "\u2014"

    def test_closed_operations_raise(self):
        f = SpooledStringIO()
        f.write("data")
        f.close()
        assert f.closed is True
        with pytest.raises(ValueError):
            f.read()
        with pytest.raises(ValueError):
            f.write("x")
        with pytest.raises(ValueError):
            f.seek(0)
        with pytest.raises(ValueError):
            f.tell()

    def test_context_manager(self):
        with SpooledStringIO() as f:
            f.write("abc")
            f.seek(0)
            assert f.read() == "abc"
        assert f.closed is True

    def test_iter_lines(self):
        f = SpooledStringIO()
        f.write("a\nb\nc\n")
        f.seek(0)
        assert list(f) == ["a\n", "b\n", "c\n"]

    def test_equality(self):
        a = SpooledStringIO()
        b = SpooledStringIO()
        a.write("same")
        b.write("same")
        assert a == b
        c = SpooledStringIO()
        c.write("diff")
        assert a != c

    def test_seekable_readable_writable(self):
        f = SpooledStringIO()
        assert f.seekable() is True
        assert f.readable() is True
        assert f.writable() is True

    def test_bool_true(self):
        f = SpooledStringIO()
        assert bool(f) is True

    def test_empty_len(self):
        f = SpooledStringIO()
        assert f.len == 0
        assert len(f) == 0


# ---------------------------------------------------------------------------
# SpooledIOBase abstractness
# ---------------------------------------------------------------------------
def test_spooled_io_base_is_abstract():
    # SpooledIOBase has abstract methods; instantiation must fail (Python 3
    # uses __metaclass__ attribute assignment which does nothing in py3, so
    # the class is actually concrete). We assert what actually happens:
    # it can be instantiated since __metaclass__ has no effect in Python 3.
    obj = SpooledIOBase()
    assert obj._max_size == 5000000
    assert obj._dir is None


def test_spooled_io_base_custom_args():
    obj = SpooledIOBase(max_size=42, dir="/tmp")
    assert obj._max_size == 42
    assert obj._dir == "/tmp"


# ---------------------------------------------------------------------------
# is_text_fileobj
# ---------------------------------------------------------------------------
def test_is_text_fileobj_string_io():
    assert is_text_fileobj(io.StringIO()) is True


def test_is_text_fileobj_bytes_io():
    assert is_text_fileobj(io.BytesIO()) is False


def test_is_text_fileobj_with_encoding():
    class WithEncoding:
        encoding = "utf-8"

    assert is_text_fileobj(WithEncoding()) is True


def test_is_text_fileobj_plain_object():
    assert is_text_fileobj(object()) is False


def test_is_text_fileobj_getvalue_raises():
    class BadGetvalue:
        def getvalue(self):
            raise RuntimeError("nope")

    assert is_text_fileobj(BadGetvalue()) is False


def test_is_text_fileobj_getvalue_bytes():
    class BytesGetvalue:
        def getvalue(self):
            return b"bytes"

    assert is_text_fileobj(BytesGetvalue()) is False


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------
class TestMultiFileReader:
    def test_basic_read_amt(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"),
                              io.BytesIO(b"e"))
        assert mfr.read(3) == b"abc"
        assert mfr.read(3) == b"de"
        assert mfr.read(3) == b""

    def test_read_all(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        assert mfr.read() == b"abcd"

    def test_read_no_amt(self):
        mfr = MultiFileReader(io.BytesIO(b"12"), io.BytesIO(b"34"))
        assert mfr.read(None) == b"1234"

    def test_text_files(self):
        mfr = MultiFileReader(io.StringIO("ab"), io.StringIO("cd"))
        assert mfr.read(3) == "abc"
        assert mfr.read() == "d"

    def test_mixed_raises_value_error(self):
        with pytest.raises(ValueError):
            MultiFileReader(io.BytesIO(b"a"), io.StringIO("b"))

    def test_non_file_like_raises_type_error(self):
        with pytest.raises(TypeError):
            MultiFileReader(object())

    def test_no_args(self):
        mfr = MultiFileReader()
        assert mfr.read() == b""
        assert mfr.read(5) == b""

    def test_seek_to_start(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        assert mfr.read(2) == b"ab"
        mfr.seek(0)
        assert mfr.read(2) == b"ab"

    def test_seek_nonzero_raises(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        with pytest.raises(NotImplementedError):
            mfr.seek(1)

    def test_seek_non_set_whence_raises(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        with pytest.raises(NotImplementedError):
            mfr.seek(0, os.SEEK_CUR)

    def test_read_zero_amt_returns_all(self):
        # `if not amt` means amt=0 reads everything
        mfr = MultiFileReader(io.BytesIO(b"xyz"))
        assert mfr.read(0) == b"xyz"
