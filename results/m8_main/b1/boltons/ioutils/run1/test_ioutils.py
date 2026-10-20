import os
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

def test_is_text_fileobj_stringio():
    assert is_text_fileobj(StringIO("abc")) is True


def test_is_text_fileobj_bytesio():
    assert is_text_fileobj(BytesIO(b"abc")) is False


def test_is_text_fileobj_encoding_attr():
    class FakeObj:
        encoding = "utf-8"

    assert is_text_fileobj(FakeObj()) is True


def test_is_text_fileobj_encoding_false():
    class FakeObj:
        encoding = ""

    assert is_text_fileobj(FakeObj()) is False


def test_is_text_fileobj_getvalue_raises():
    class FakeObj:
        def getvalue(self):
            raise RuntimeError("boom")

    assert is_text_fileobj(FakeObj()) is False


def test_is_text_fileobj_plain_object():
    assert is_text_fileobj(object()) is False


# ---------------------------------------------------------------------------
# SpooledBytesIO
# ---------------------------------------------------------------------------

def test_bytesio_basic_read_write():
    f = SpooledBytesIO()
    assert f.write(b"hello") == 5
    assert f.seek(0) == 0
    assert f.read() == b"hello"
    assert f.getvalue() == b"hello"
    f.close()


def test_bytesio_read_partial_and_readline():
    f = SpooledBytesIO()
    f.write(b"line1\nline2\nline3")
    f.seek(0)
    assert f.readline() == b"line1\n"
    assert f.readline(3) == b"lin"
    f.seek(0)
    assert f.readlines() == [b"line1\n", b"line2\n", b"line3"]
    f.close()


def test_bytesio_accepts_bytearray_and_memoryview():
    f = SpooledBytesIO()
    f.write(bytearray(b"ab"))
    f.write(memoryview(b"cd"))
    assert f.getvalue() == b"abcd"
    f.close()


def test_bytesio_write_text_raises_typeerror():
    f = SpooledBytesIO()
    with pytest.raises(TypeError):
        f.write("text")
    f.close()


def test_bytesio_len():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    assert len(f) == 6
    f.close()


def test_bytesio_tell_pos_buf():
    f = SpooledBytesIO()
    f.write(b"abc")
    assert f.tell() == 3
    assert f.pos == 3
    assert f.buf == b"abc"
    f.close()


def test_bytesio_writelines():
    f = SpooledBytesIO()
    f.writelines([b"a", b"b", b"c"])
    assert f.getvalue() == b"abc"
    f.close()


def test_bytesio_rollover_exceeding_max_size(tmp_path):
    f = SpooledBytesIO(max_size=4, dir=str(tmp_path))
    assert f._rolled is False
    f.write(b"12345")
    assert f._rolled is True
    f.seek(0)
    assert f.read() == b"12345"
    assert len(f) == 5
    f.close()


def test_bytesio_rollover_explicit():
    f = SpooledBytesIO()
    f.write(b"data")
    f.rollover()
    assert f._rolled is True
    # second rollover is a no-op
    buf = f.buffer
    f.rollover()
    assert f.buffer is buf
    f.seek(0)
    assert f.read() == b"data"
    f.close()


def test_bytesio_fileno_after_rollover():
    f = SpooledBytesIO()
    f.write(b"abc")
    fileno = f.fileno()
    assert isinstance(fileno, int)
    assert f._rolled is True
    f.close()


def test_bytesio_closed_rules():
    f = SpooledBytesIO()
    f.write(b"abc")
    assert f.closed is False
    f.close()
    assert f.closed is True
    with pytest.raises(ValueError):
        f.read()
    with pytest.raises(ValueError):
        f.write(b"x")
    with pytest.raises(ValueError):
        f.tell()
    with pytest.raises(ValueError):
        f.seek(0)
    with pytest.raises(ValueError):
        f.read()
    with pytest.raises(ValueError):
        f.flush()
    with pytest.raises(ValueError):
        f.isatty()
    with pytest.raises(ValueError):
        f.truncate()
    with pytest.raises(ValueError):
        f.getvalue()


def test_bytesio_truncate_no_arg():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(3)
    f.truncate()
    assert f.getvalue() == b"abc"
    f.close()


def test_bytesio_truncate_with_size():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(0)
    f.truncate(2)
    assert f.getvalue() == b"ab"
    # position preserved when pos < size
    f.seek(1)
    f.truncate(2)
    assert f.tell() == 1
    f.close()


def test_bytesio_truncate_negative():
    f = SpooledBytesIO()
    f.write(b"abc")
    with pytest.raises(OSError):
        f.truncate(-1)
    f.close()


def test_bytesio_seekable_readable_writable():
    f = SpooledBytesIO()
    assert f.seekable() is True
    assert f.readable() is True
    assert f.writable() is True
    f.close()


def test_bytesio_iteration():
    f = SpooledBytesIO()
    f.write(b"a\nb\nc")
    f.seek(0)
    assert list(f) == [b"a\n", b"b\n", b"c"]
    f.close()


def test_bytesio_next_stop_iteration():
    f = SpooledBytesIO()
    f.write(b"abc")
    f.seek(0)
    assert next(f) == b"abc"
    with pytest.raises(StopIteration):
        next(f)
    f.close()


def test_bytesio_context_manager():
    with SpooledBytesIO() as f:
        f.write(b"abc")
        assert f.getvalue() == b"abc"
    assert f.closed is True


def test_bytesio_eq():
    f1 = SpooledBytesIO()
    f1.write(b"abc")
    f1.seek(0)
    f2 = SpooledBytesIO()
    f2.write(b"abc")
    f2.seek(0)
    assert f1 == f2
    assert not (f1 != f2)
    f3 = SpooledBytesIO()
    f3.write(b"xyz")
    f3.seek(0)
    assert f1 != f3
    # positions preserved
    f1.seek(2)
    pos = f1.tell()
    f1 == f2
    assert f1.tell() == pos
    f1.close()
    f2.close()
    f3.close()


def test_bytesio_eq_other_type():
    f = SpooledBytesIO()
    assert (f == b"abc") is False
    assert (f != b"abc") is True
    f.close()


def test_bytesio_bool():
    f = SpooledBytesIO()
    assert bool(f) is True
    f.close()


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

def test_stringio_basic_read_write():
    f = SpooledStringIO()
    assert f.write("hello") == 5
    assert f.seek(0) == 0
    assert f.read() == "hello"
    f.write.__self__  # keep flake happy
    f.close()


def test_stringio_getvalue_via_read():
    f = SpooledStringIO()
    f.write("abc")
    f.seek(0)
    assert f.getvalue() == "abc"
    # getvalue() restores the codepoint position to the start of the
    # buffer, so a subsequent write overwrites from the beginning
    # rather than appending.
    f.seek(0)
    f.write("def")
    f.seek(0)
    assert f.getvalue() == "def"
    f.close()


def test_stringio_write_non_str_raises():
    f = SpooledStringIO()
    with pytest.raises(TypeError):
        f.write(b"bytes")
    f.close()


def test_stringio_unicode():
    f = SpooledStringIO()
    f.write(u"\u2014 emdash")
    f.seek(0)
    assert f.read() == u"\u2014 emdash"
    f.close()


def test_stringio_tell_and_pos():
    f = SpooledStringIO()
    f.write("abc")
    assert f.tell() == 3
    f.seek(0)
    assert f.pos == 0
    f.close()


def test_stringio_readline():
    f = SpooledStringIO()
    f.write("line1\nline2\n")
    f.seek(0)
    assert f.readline() == "line1\n"
    assert f.readline() == "line2\n"
    f.close()


def test_stringio_readline_length():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(0)
    assert f.readline(3) == "abc"
    f.close()


def test_stringio_readlines():
    f = SpooledStringIO()
    f.write("a\nb\nc")
    f.seek(0)
    assert f.readlines() == ["a\n", "b\n", "c"]
    f.close()


def test_stringio_len():
    f = SpooledStringIO()
    f.write("abcdef")
    assert len(f) == 6
    f.write("gh")
    assert len(f) == 8
    f.close()


def test_stringio_seek_cur():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(2)
    assert f.tell() == 2
    assert f.seek(2, os.SEEK_CUR) == 4
    # seek_cur traverses forward from the current position, then the
    # read continues from the newly reached buffer offset.
    assert f.read(2) == "ef"
    f.close()


def test_stringio_seek_cur_negative():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(4)
    assert f.seek(-2, os.SEEK_CUR) == 2
    assert f.read(2) == "cd"
    f.close()


def test_stringio_seek_cur_negative_too_far():
    f = SpooledStringIO()
    f.write("abcdef")
    with pytest.raises(ValueError):
        f.seek(-10, os.SEEK_CUR)
    f.close()


def test_stringio_seek_end():
    f = SpooledStringIO()
    f.write("abcdef")
    # seek to end: dest_position = len - pos = 6 - 0 = 6
    assert f.seek(0, os.SEEK_END) == 6
    assert f.read() == ""
    f.close()


def test_stringio_seek_negative_start():
    f = SpooledStringIO()
    f.write("abc")
    with pytest.raises(ValueError):
        f.seek(-1)
    f.close()


def test_stringio_seek_invalid_whence():
    f = SpooledStringIO()
    f.write("abc")
    with pytest.raises(ValueError):
        f.seek(0, 99)
    f.close()


def test_stringio_rollover(tmp_path):
    f = SpooledStringIO(max_size=4, dir=str(tmp_path))
    assert f._rolled is False
    f.write("12345")
    assert f._rolled is True
    f.seek(0)
    assert f.read() == "12345"
    f.close()


def test_stringio_explicit_rollover():
    f = SpooledStringIO()
    f.write("data")
    f.rollover()
    assert f._rolled is True
    buf = f._buffer
    f.rollover()
    assert f._buffer is buf
    f.seek(0)
    assert f.read() == "data"
    f.close()


def test_stringio_closed_rules():
    f = SpooledStringIO()
    f.write("abc")
    f.close()
    with pytest.raises(ValueError):
        f.read()
    with pytest.raises(ValueError):
        f.write("x")
    with pytest.raises(ValueError):
        f.tell()
    with pytest.raises(ValueError):
        f.seek(0)
    with pytest.raises(ValueError):
        f.readline()
    with pytest.raises(ValueError):
        f.getvalue()


def test_stringio_writelines():
    f = SpooledStringIO()
    f.writelines(["a", "b", "c"])
    f.seek(0)
    assert f.read() == "abc"
    f.close()


def test_stringio_len_after_rollover(tmp_path):
    f = SpooledStringIO(max_size=2, dir=str(tmp_path))
    f.write("abcd")
    assert f._rolled is True
    assert len(f) == 4
    f.close()


def test_stringio_context_manager():
    with SpooledStringIO() as f:
        f.write("abc")
    assert f.closed is True


def test_stringio_iteration():
    f = SpooledStringIO()
    f.write("a\nb\nc")
    f.seek(0)
    assert list(f) == ["a\n", "b\n", "c"]
    f.close()


def test_stringio_truncate():
    f = SpooledStringIO()
    f.write("abcdef")
    f.truncate(3)
    f.seek(0)
    assert f.read() == "abc"
    f.close()


# ---------------------------------------------------------------------------
# SpooledIOBase metadata
# ---------------------------------------------------------------------------

def test_spooled_io_base_defaults():
    f = SpooledBytesIO()
    assert f._max_size == 5000000
    assert f._dir is None
    f.close()


def test_spooled_io_base_custom_size():
    f = SpooledBytesIO(max_size=100, dir="/tmp")
    assert f._max_size == 100
    assert f._dir == "/tmp"
    f.close()


def test_read_chunk_size_value():
    assert READ_CHUNK_SIZE == 21333


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

def test_multifile_reader_bytes_read():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"), BytesIO(b"e"))
    assert mfr.read(3) == b"abc"
    assert mfr.read(3) == b"de"
    assert mfr.read(3) == b""
    assert mfr.read(3) == b""


def test_multifile_reader_read_all():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"))
    assert mfr.read() == b"abcd"


def test_multifile_reader_text():
    mfr = MultiFileReader(StringIO("ab"), StringIO("cd"))
    assert mfr.read(3) == "abc"
    assert mfr.read() == "d"


def test_multifile_reader_mixed_raises():
    with pytest.raises(ValueError):
        MultiFileReader(BytesIO(b"ab"), StringIO("cd"))


def test_multifile_reader_non_filelike_raises():
    with pytest.raises(TypeError):
        MultiFileReader("not a file")
    with pytest.raises(TypeError):
        MultiFileReader(object())


def test_multifile_reader_seek():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"))
    assert mfr.read(2) == b"ab"
    mfr.seek(0)
    assert mfr.read(2) == b"ab"


def test_multifile_reader_seek_nonzero_raises():
    mfr = MultiFileReader(BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(1)


def test_multifile_reader_seek_whence_raises():
    mfr = MultiFileReader(BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(0, os.SEEK_CUR)


def test_multifile_reader_empty():
    mfr = MultiFileReader()
    # With no file objects, ``all([])`` is True, so the text joiner
    # (``''``) is selected and empty reads return a str.
    assert mfr.read() == ""
    assert mfr.read(5) == ""
