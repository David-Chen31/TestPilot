import os
import io
from io import BytesIO, StringIO

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
# is_text_fileobj
# ---------------------------------------------------------------------------

def test_read_chunk_size_value():
    assert READ_CHUNK_SIZE == 21333


def test_is_text_fileobj_stringio():
    assert is_text_fileobj(StringIO("abc")) is True


def test_is_text_fileobj_bytesio():
    assert is_text_fileobj(BytesIO(b"abc")) is False


def test_is_text_fileobj_encoding_attr():
    class WithEncoding:
        encoding = 'utf-8'

    assert is_text_fileobj(WithEncoding()) is True


def test_is_text_fileobj_getvalue_raises():
    class BadGetvalue:
        def getvalue(self):
            raise RuntimeError("boom")

    assert is_text_fileobj(BadGetvalue()) is False


def test_is_text_fileobj_getvalue_bytes():
    class BytesGetvalue:
        def getvalue(self):
            return b"x"

    assert is_text_fileobj(BytesGetvalue()) is False


def test_is_text_fileobj_plain_object():
    assert is_text_fileobj(object()) is False


# ---------------------------------------------------------------------------
# SpooledIOBase
# ---------------------------------------------------------------------------

def test_spooled_io_base_instantiation():
    # SpooledIOBase uses the __metaclass__ attribute, which is not
    # honored in Python 3, so it can be instantiated.
    base = SpooledIOBase()
    assert base._max_size == 5000000
    assert base._dir is None
    assert base.seekable() is True
    assert base.readable() is True
    assert base.writable() is True


def test_spooled_io_base_stores_dir():
    base = SpooledIOBase(max_size=10, dir="/tmp/somedir")
    assert base._max_size == 10
    assert base._dir == "/tmp/somedir"


def test_check_closed_default_message():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError) as exc:
        f.read()
    assert str(exc.value) == "I/O operation on closed file."


def test_check_closed_custom_message():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError) as exc:
        f._checkClosed("custom msg here")
    assert str(exc.value) == "custom msg here"


# ---------------------------------------------------------------------------
# SpooledBytesIO
# ---------------------------------------------------------------------------

def test_spooled_bytesio_write_read_roundtrip():
    f = SpooledBytesIO()
    assert f.write(b"Happy IO") == 8
    f.seek(0)
    assert f.read() == b"Happy IO"
    assert isinstance(f.getvalue(), bytes)
    f.close()


def test_spooled_bytesio_write_bytearray_and_memoryview():
    f = SpooledBytesIO()
    assert f.write(bytearray(b"abc")) == 3
    assert f.write(memoryview(b"def")) == 3
    f.seek(0)
    assert f.read() == b"abcdef"
    f.close()


def test_spooled_bytesio_write_text_raises():
    f = SpooledBytesIO()
    with pytest.raises(TypeError):
        f.write("text")
    f.close()


def test_spooled_bytesio_not_rolled_by_default():
    f = SpooledBytesIO()
    f.write(b"abc")
    assert f._rolled is False
    f.close()


def test_spooled_bytesio_rolls_over(tmp_path):
    f = SpooledBytesIO(max_size=5, dir=str(tmp_path))
    f.write(b"abcdef")  # exceeds max_size
    assert f._rolled is True
    f.seek(0)
    assert f.read() == b"abcdef"
    f.close()


def test_spooled_bytesio_rolls_over_at_boundary(tmp_path):
    # Condition is tell() + nbytes >= max_size
    f = SpooledBytesIO(max_size=5, dir=str(tmp_path))
    f.write(b"abcde")  # exactly max_size -> rollover
    assert f._rolled is True
    f.seek(0)
    assert f.read() == b"abcde"
    f.close()


def test_spooled_bytesio_no_rollover_just_below(tmp_path):
    f = SpooledBytesIO(max_size=6, dir=str(tmp_path))
    f.write(b"abcde")  # 5 < 6 -> no rollover
    assert f._rolled is False
    f.close()


def test_spooled_bytesio_rollover_into_dir(tmp_path):
    # Verify rollover actually uses the dir argument by ensuring the
    # temporary file lives in tmp_path.
    sub = tmp_path / "subdir"
    sub.mkdir()
    f = SpooledBytesIO(max_size=2, dir=str(sub))
    f.write(b"abcdef")
    assert f._rolled is True
    # The rolled temp file's name (if any) should be inside sub.
    name = getattr(f.buffer, "name", None)
    if name is not None and isinstance(name, str):
        assert os.path.dirname(os.path.abspath(name)) == os.path.abspath(str(sub))
    f.close()


def test_spooled_bytesio_len_not_rolled():
    f = SpooledBytesIO()
    f.write(b"hello")
    assert f.len == 5
    assert len(f) == 5
    f.close()


def test_spooled_bytesio_len_rolled(tmp_path):
    f = SpooledBytesIO(max_size=2, dir=str(tmp_path))
    f.write(b"abcdef")
    assert f._rolled is True
    assert f.len == 6
    f.close()


def test_spooled_bytesio_len_preserves_position():
    f = SpooledBytesIO()
    f.write(b"hello")
    f.seek(2)
    assert f.len == 5
    assert f.tell() == 2
    f.close()


def test_spooled_bytesio_readline():
    f = SpooledBytesIO()
    f.write(b"line1\nline2\n")
    f.seek(0)
    assert f.readline() == b"line1\n"
    assert f.readline(3) == b"lin"
    f.close()


def test_spooled_bytesio_read_empty():
    f = SpooledBytesIO()
    f.write(b"abc")
    f.seek(0)
    # read() with default n uses buffer.read(-1) => whole content
    assert f.read() == b"abc"
    f.close()


def test_spooled_bytesio_readline_exact():
    f = SpooledBytesIO()
    f.write(b"line1\nline2\n")
    f.seek(0)
    assert f.readline(6) == b"line1\n"
    f.close()


def test_spooled_bytesio_readlines():
    f = SpooledBytesIO()
    f.write(b"a\nb\nc\n")
    f.seek(0)
    assert f.readlines() == [b"a\n", b"b\n", b"c\n"]
    f.close()


def test_spooled_bytesio_readlines_sizehint():
    f = SpooledBytesIO()
    f.write(b"a\nb\nc\n")
    f.seek(0)
    # sizehint=2 should read roughly one line
    result = f.readlines(2)
    assert result == [b"a\n"]
    f.close()


def test_spooled_bytesio_writelines():
    f = SpooledBytesIO()
    f.writelines([b"ab", b"cd"])
    f.seek(0)
    assert f.read() == b"abcd"
    f.close()


def test_spooled_bytesio_writelines_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.writelines([b"ab"])


def test_spooled_bytesio_truncate_no_arg():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(3)
    f.truncate()
    f.seek(0)
    assert f.read() == b"abc"
    f.close()


def test_spooled_bytesio_truncate_size():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(2)
    f.truncate(4)
    f.seek(0)
    assert f.read() == b"abcd"
    f.close()


def test_spooled_bytesio_truncate_size_zero():
    # size=0 is not negative, so it should truncate to empty, not raise.
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.truncate(0)
    f.seek(0)
    assert f.read() == b""
    f.close()


def test_spooled_bytesio_truncate_negative_size():
    f = SpooledBytesIO()
    f.write(b"abc")
    with pytest.raises(OSError) as exc:
        f.truncate(-1)
    assert exc.value.errno == 22  # EINVAL
    assert exc.value.strerror == "Negative size not allowed"
    f.close()


def test_spooled_bytesio_truncate_pos_after_size():
    # When the current position is beyond the truncation point,
    # truncate() leaves the cursor at the new (truncated) position.
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(5)
    f.truncate(3)
    assert f.tell() == 3
    f.close()


def test_spooled_bytesio_truncate_pos_before_size():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(1)
    f.truncate(4)
    assert f.tell() == 1
    f.seek(0)
    assert f.read() == b"abcd"
    f.close()


def test_spooled_bytesio_truncate_at_size():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(4)
    f.truncate(4)
    assert f.tell() == 4
    f.close()


def test_spooled_bytesio_truncate_pos_before_size_total_len():
    # Distinguish pos < size from pos <= size: when pos == size, the
    # behaviour is identical, but let's ensure pos < size restores.
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(2)
    f.truncate(4)
    assert f.tell() == 2
    f.seek(0)
    assert f.read() == b"abcd"
    f.close()


def test_spooled_bytesio_truncate_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.truncate()


def test_spooled_bytesio_getvalue_preserves_position():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(2)
    assert f.getvalue() == b"abcdef"
    assert f.tell() == 2
    f.close()


def test_spooled_bytesio_closed_operations():
    f = SpooledBytesIO()
    f.close()
    assert f.closed is True
    with pytest.raises(ValueError):
        f.read()
    with pytest.raises(ValueError):
        f.write(b"x")
    with pytest.raises(ValueError):
        f.seek(0)
    with pytest.raises(ValueError):
        f.getvalue()
    with pytest.raises(ValueError):
        f.flush()
    with pytest.raises(ValueError):
        f.tell()
    with pytest.raises(ValueError):
        f.readline()
    with pytest.raises(ValueError):
        next(f)
    with pytest.raises(ValueError):
        f.isatty()


def test_spooled_bytesio_pos_and_buf():
    f = SpooledBytesIO()
    f.write(b"abc")
    assert f.pos == 3
    assert f.buf == b"abc"
    f.close()


def test_spooled_bytesio_seekable_readable_writable():
    f = SpooledBytesIO()
    assert f.seekable() is True
    assert f.readable() is True
    assert f.writable() is True
    f.close()


def test_spooled_bytesio_iteration():
    f = SpooledBytesIO()
    f.write(b"a\nb\nc")
    f.seek(0)
    lines = list(f)
    assert lines == [b"a\n", b"b\n", b"c"]
    f.close()


def test_spooled_bytesio_iter_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        iter(f)


def test_spooled_bytesio_next_after_reading_end_returns_line():
    # When readline returns a line, __next__ returns it. But when readline
    # returns b'' at the end, __next__ seeks and checks position. The
    # `self.buffer.seek(pos)` in the else-branch shouldn't hard-fail.
    f = SpooledBytesIO()
    f.write(b"line1\n")
    f.seek(0)
    assert next(f) == b"line1\n"
    with pytest.raises(StopIteration):
        next(f)
    f.close()


def test_spooled_bytesio_not_equal_other_type():
    f = SpooledBytesIO()
    f.write(b"abc")
    f.seek(0)
    assert (f == "abc") is False
    f.close()


def test_spooled_bytesio_equality():
    f1 = SpooledBytesIO()
    f1.write(b"abc\ndef\n")
    f1.seek(2)
    f2 = SpooledBytesIO()
    f2.write(b"abc\ndef\n")
    assert f1 == f2
    assert not (f1 != f2)
    assert f1.tell() == 2


def test_spooled_bytesio_inequality():
    f1 = SpooledBytesIO()
    f1.write(b"abc")
    f2 = SpooledBytesIO()
    f2.write(b"abd")
    assert (f1 == f2) is False
    assert f1 != f2


def test_spooled_bytesio_equality_restores_positions():
    f1 = SpooledBytesIO()
    f1.write(b"abc")
    f2 = SpooledBytesIO()
    f2.write(b"abc")
    f1.seek(1)
    f2.seek(2)
    assert f1 == f2
    assert f1.tell() == 1
    assert f2.tell() == 2


def test_spooled_bytesio_equality_restores_both_positions():
    # Equality comparison must restore both files' positions, not just one.
    f1 = SpooledBytesIO()
    f1.write(b"abc")
    f2 = SpooledBytesIO()
    f2.write(b"abc")
    f1.seek(1)
    f2.seek(2)
    assert f1 == f2
    assert f1.tell() == 1
    assert f2.tell() == 2
    assert f1.getvalue() == b"abc"
    assert f2.getvalue() == b"abc"
    f1.close()
    f2.close()


def test_spooled_bytesio_equality_exception_restores_positions():
    # Force an exception during comparison by closing one inside __eq__.
    # Simpler: compare files where one is closed -> accessing it raises.
    f1 = SpooledBytesIO()
    f1.write(b"abc")
    f1.seek(1)
    f2 = SpooledBytesIO()
    f2.write(b"abc")
    f2.close()
    # __eq__ should raise a ValueError during iteration; both positions
    # must be restored (best effort). Verify f1 position was restored.
    with pytest.raises(ValueError):
        f1 == f2
    assert f1.tell() == 1
    f1.close()


def test_spooled_bytesio_bool():
    f = SpooledBytesIO()
    assert bool(f) is True
    f.close()


def test_spooled_bytesio_context_manager():
    with SpooledBytesIO() as f:
        f.write(b"abc")
        f.seek(0)
        assert f.read() == b"abc"
    assert f.closed is True


def test_spooled_bytesio_enter_closed_raises():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        with f:
            pass


def test_spooled_bytesio_fileno_rolls_over(tmp_path):
    f = SpooledBytesIO(dir=str(tmp_path))
    f.write(b"abc")
    assert f._rolled is False
    fd = f.fileno()
    assert isinstance(fd, int)
    assert f._rolled is True
    f.close()


def test_spooled_bytesio_next_at_end():
    f = SpooledBytesIO()
    f.write(b"")
    f.seek(0)
    with pytest.raises(StopIteration):
        next(f)
    f.close()


def test_spooled_bytesio_isatty():
    f = SpooledBytesIO()
    assert f.isatty() is False
    f.close()


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

def test_spooled_stringio_write_read_roundtrip():
    f = SpooledStringIO()
    # write() returns the number of codepoints in the input string.
    assert f.write("\u2014 Hey, an emdash!") == 17
    f.seek(0)
    result = f.read()
    assert isinstance(result, str)
    assert result == "\u2014 Hey, an emdash!"
    f.close()


def test_spooled_stringio_write_bytes_raises():
    f = SpooledStringIO()
    with pytest.raises(TypeError):
        f.write(b"bytes")
    f.close()


def test_spooled_stringio_write_bytes_error_message():
    f = SpooledStringIO()
    with pytest.raises(TypeError) as exc:
        f.write(b"bytes")
    assert str(exc.value) == "str expected, got bytes"
    f.close()


def test_spooled_stringio_write_int_error_message():
    f = SpooledStringIO()
    with pytest.raises(TypeError) as exc:
        f.write(123)
    assert str(exc.value) == "str expected, got int"
    f.close()


def test_spooled_stringio_tell_after_write():
    f = SpooledStringIO()
    f.write("abc")
    assert f.tell() == 3
    f.write("de")
    assert f.tell() == 5
    f.close()


def test_spooled_stringio_read_updates_tell():
    f = SpooledStringIO()
    f.write("hello")
    f.seek(0)
    assert f.tell() == 0
    assert f.read(2) == "he"
    assert f.tell() == 2
    assert f.read(2) == "ll"
    assert f.tell() == 4
    f.close()


def test_spooled_stringio_read_all_with_multibyte():
    f = SpooledStringIO()
    f.write("\u2014\u2014")
    f.seek(0)
    assert f.read() == "\u2014\u2014"
    assert f.tell() == 2
    f.close()


def test_spooled_stringio_seek_set():
    f = SpooledStringIO()
    f.write("abcdef")
    assert f.seek(3) == 3
    assert f.tell() == 3
    assert f.read(2) == "de"
    f.close()


def test_spooled_stringio_seek_negative_set():
    f = SpooledStringIO()
    f.write("abc")
    with pytest.raises(ValueError) as exc:
        f.seek(-1, os.SEEK_SET)
    assert str(exc.value) == "negative seek position"
    f.close()


def test_spooled_stringio_seek_cur_positive():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(1)
    assert f.seek(2, os.SEEK_CUR) == 3
    assert f.tell() == 3
    f.close()


def test_spooled_stringio_seek_cur_negative():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(4)
    assert f.seek(-2, os.SEEK_CUR) == 2
    assert f.tell() == 2
    f.close()


def test_spooled_stringio_seek_cur_negative_invalid():
    f = SpooledStringIO()
    f.write("abc")
    f.seek(1)
    with pytest.raises(ValueError) as exc:
        f.seek(-2, os.SEEK_CUR)
    assert str(exc.value) == "negative seek position"
    f.close()


def test_spooled_stringio_seek_end():
    f = SpooledStringIO()
    f.write("abcdef")
    assert f.seek(2, os.SEEK_END) == 4
    assert f.tell() == 4
    f.close()


def test_spooled_stringio_seek_invalid_whence():
    f = SpooledStringIO()
    f.write("abc")
    with pytest.raises(ValueError) as exc:
        f.seek(1, 99)
    assert "Invalid whence" in str(exc.value)
    f.close()


def test_spooled_stringio_seek_multibyte():
    f = SpooledStringIO()
    f.write("\u2014\u2014\u2014")
    assert f.seek(2) == 2
    assert f.tell() == 2
    assert f.read(1) == "\u2014"
    f.close()


def test_spooled_stringio_seek_to_exact_end():
    # Exercises _traverse_codepoints where current_position == dest (break)
    f = SpooledStringIO()
    f.write("abc")
    assert f.seek(3) == 3
    assert f.tell() == 3
    assert f.read() == ""
    f.close()


def test_spooled_stringio_large_traversal():
    # Forces multiple READ_CHUNK_SIZE iterations in _traverse_codepoints.
    big = "a" * (READ_CHUNK_SIZE + 100)
    f = SpooledStringIO()
    f.write(big)
    assert f.seek(READ_CHUNK_SIZE + 50) == READ_CHUNK_SIZE + 50
    assert f.tell() == READ_CHUNK_SIZE + 50
    assert f.read(1) == "a"
    f.close()


def test_spooled_stringio_len():
    f = SpooledStringIO()
    f.write("\u2014abc")
    assert f.len == 4
    f.close()


def test_spooled_stringio_readline():
    f = SpooledStringIO()
    f.write("line1\nline2\n")
    f.seek(0)
    assert f.readline() == "line1\n"
    assert f.tell() == 6
    f.close()


def test_spooled_stringio_readline_length():
    f = SpooledStringIO()
    f.write("line1\nline2\n")
    f.seek(0)
    assert f.readline(3) == "lin"
    assert f.tell() == 3
    f.close()


def test_spooled_stringio_readlines():
    f = SpooledStringIO()
    f.write("a\nb\nc\n")
    f.seek(0)
    lines = f.readlines()
    assert lines == ["a\n", "b\n", "c\n"]
    assert f.tell() == 6
    f.close()


def test_spooled_stringio_writelines():
    f = SpooledStringIO()
    f.writelines(["ab", "cd"])
    f.seek(0)
    assert f.read() == "abcd"
    f.close()


def test_spooled_stringio_getvalue():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(2)
    assert f.getvalue() == "abcdef"
    assert f.tell() == 2
    f.close()


def test_spooled_stringio_not_rolled_by_default():
    f = SpooledStringIO()
    f.write("abc")
    assert f._rolled is False
    f.close()


def test_spooled_stringio_rollover(tmp_path):
    f = SpooledStringIO(max_size=2, dir=str(tmp_path))
    f.write("abcdef")
    assert f._rolled is True
    f.seek(0)
    assert f.read() == "abcdef"
    f.close()


def test_spooled_stringio_rollover_into_dir(tmp_path):
    sub = tmp_path / "sub2"
    sub.mkdir()
    f = SpooledStringIO(max_size=2, dir=str(sub))
    f.write("abcdef")
    assert f._rolled is True
    name = getattr(f.buffer.stream, "name", None)
    if name is not None and isinstance(name, str):
        assert os.path.dirname(os.path.abspath(name)) == os.path.abspath(str(sub))
    f.close()


def test_spooled_stringio_rollover_at_boundary(tmp_path):
    # Condition is buffer.tell() + len(utf8) >= max_size
    f = SpooledStringIO(max_size=6, dir=str(tmp_path))
    f.write("abcdef")  # exactly 6 -> rollover
    assert f._rolled is True
    f.close()


def test_spooled_stringio_close():
    f = SpooledStringIO()
    f.write("abc")
    f.close()
    assert f.closed is True
    with pytest.raises(ValueError):
        f.read()
    with pytest.raises(ValueError):
        f.tell()
    with pytest.raises(ValueError):
        f.write("x")


def test_spooled_stringio_context_manager():
    with SpooledStringIO() as f:
        f.write("hello")
        f.seek(0)
        assert f.read() == "hello"
    assert f.closed is True


def test_spooled_stringio_truncate():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(2)
    f.truncate(4)
    f.seek(0)
    assert f.read() == "abcd"
    f.close()


def test_spooled_stringio_truncate_negative():
    f = SpooledStringIO()
    f.write("abcdef")
    with pytest.raises(OSError) as exc:
        f.truncate(-1)
    assert exc.value.errno == 22
    assert exc.value.strerror == "Negative size not allowed"
    f.close()


def test_spooled_stringio_equality():
    f1 = SpooledStringIO()
    f1.write("abc\ndef\n")
    f2 = SpooledStringIO()
    f2.write("abc\ndef\n")
    assert f1 == f2


def test_spooled_stringio_inequality():
    f1 = SpooledStringIO()
    f1.write("abc")
    f2 = SpooledStringIO()
    f2.write("abd")
    assert f1 != f2


def test_spooled_stringio_traverse_codepoints_beyond_end():
    f = SpooledStringIO()
    f.write("ab")
    assert f.seek(10) == 10
    assert f.tell() == 10
    f.close()


def test_spooled_stringio_read_none():
    # read() with no arg should work via read(-1, -1)
    f = SpooledStringIO()
    f.write("hello")
    f.seek(0)
    assert f.read() == "hello"
    f.close()


def test_spooled_stringio_read_exact_remaining():
    f = SpooledStringIO()
    f.write("hello")
    f.seek(2)
    assert f.read(3) == "llo"
    assert f.tell() == 5
    f.close()


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

def test_multifilereader_bytes():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"), BytesIO(b"e"))
    assert mfr.read(3) == b"abc"
    assert mfr.read(3) == b"de"
    assert mfr.read(3) == b""


def test_multifilereader_text():
    mfr = MultiFileReader(StringIO("ab"), StringIO("cd"), StringIO("e"))
    assert mfr.read(3) == "abc"
    assert mfr.read(3) == "de"


def test_multifilereader_read_all():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"))
    assert mfr.read() == b"abcd"


def test_multifilereader_read_all_text():
    mfr = MultiFileReader(StringIO("ab"), StringIO("cd"))
    assert mfr.read() == "abcd"


def test_multifilereader_non_filelike_raises():
    with pytest.raises(TypeError) as exc:
        MultiFileReader(object())
    assert "expected file-like" in str(exc.value)


def test_multifilereader_missing_seek_raises():
    class OnlyRead:
        def read(self, *a):
            return b""

    with pytest.raises(TypeError):
        MultiFileReader(OnlyRead())


def test_multifilereader_mixed_types_raises():
    with pytest.raises(ValueError) as exc:
        MultiFileReader(BytesIO(b"a"), StringIO("b"))
    assert "not a mix" in str(exc.value)


def test_multifilereader_seek():
    # seek(0) only rewinds each underlying file; the internal index is
    # not reset, so after reading from the second file, reads continue
    # from the current index.
    f1 = BytesIO(b"ab")
    f2 = BytesIO(b"cd")
    mfr = MultiFileReader(f1, f2)
    assert mfr.read(3) == b"abc"
    mfr.seek(0)
    # index is now 1, so read resumes from f2 (rewound to start)
    assert mfr.read(2) == b"cd"


def test_multifilereader_seek_resets_files():
    f1 = BytesIO(b"ab")
    f2 = BytesIO(b"cd")
    mfr = MultiFileReader(f1, f2)
    mfr.read(2)
    mfr.seek(0)
    assert f1.tell() == 0
    assert f2.tell() == 0


def test_multifilereader_seek_nonzero_raises():
    mfr = MultiFileReader(BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(1)


def test_multifilereader_seek_whence_raises():
    mfr = MultiFileReader(BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(0, os.SEEK_CUR)


def test_multifilereader_empty():
    # With no file objects, the fallback joiner is '' (text), so reads
    # return empty strings.
    mfr = MultiFileReader()
    assert mfr.read() == ""
    assert mfr.read(5) == ""
