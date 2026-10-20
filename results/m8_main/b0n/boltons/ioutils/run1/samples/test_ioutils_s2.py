import os
from io import BytesIO, StringIO, TextIOWrapper

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
    assert is_text_fileobj(StringIO("hi")) is True


def test_is_text_fileobj_bytesio():
    assert is_text_fileobj(BytesIO(b"hi")) is False


def test_is_text_fileobj_textwrapper():
    tf = TextIOWrapper(BytesIO(b"hi"))
    try:
        assert is_text_fileobj(tf) is True
    finally:
        tf.detach()


def test_is_text_fileobj_plain_object():
    assert is_text_fileobj(object()) is False


def test_is_text_fileobj_getvalue_raises():
    class Weird:
        def getvalue(self):
            raise RuntimeError("boom")
    assert is_text_fileobj(Weird()) is False


def test_is_text_fileobj_getvalue_bytes():
    class B:
        def getvalue(self):
            return b"bytes"
    assert is_text_fileobj(B()) is False


# ---------------------------------------------------------------------------
# SpooledBytesIO
# ---------------------------------------------------------------------------

def test_bytesio_basic_roundtrip():
    f = SpooledBytesIO()
    assert f.write(b"Happy IO") == 8
    assert f.tell() == 8
    f.seek(0)
    assert f.read() == b"Happy IO"
    assert f.getvalue() == b"Happy IO"
    f.close()
    assert f.closed is True


def test_bytesio_write_bytearray_and_memoryview():
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


def test_bytesio_not_rolled_by_default():
    f = SpooledBytesIO()
    f.write(b"x")
    assert f._rolled is False
    f.close()


def test_bytesio_rollover_on_max_size():
    f = SpooledBytesIO(max_size=5)
    f.write(b"12345")
    assert f._rolled is True
    f.seek(0)
    assert f.read() == b"12345"
    f.close()


def test_bytesio_len():
    f = SpooledBytesIO()
    f.write(b"hello")
    assert len(f) == 5
    assert f.len == 5
    # length stable after move
    f.seek(0)
    assert f.len == 5
    assert f.tell() == 0
    f.close()


def test_bytesio_len_after_rollover(tmp_path):
    f = SpooledBytesIO(max_size=3, dir=str(tmp_path))
    f.write(b"abcdef")
    assert f._rolled is True
    assert f.len == 6
    f.close()


def test_bytesio_readline():
    f = SpooledBytesIO()
    f.write(b"a\nb\nc")
    f.seek(0)
    assert f.readline() == b"a\n"
    assert f.readline(1) == b"b"
    f.close()


def test_bytesio_readlines():
    f = SpooledBytesIO()
    f.write(b"a\nb\nc")
    f.seek(0)
    assert f.readlines() == [b"a\n", b"b\n", b"c"]
    f.close()


def test_bytesio_iteration():
    f = SpooledBytesIO()
    f.write(b"a\nb\n")
    f.seek(0)
    assert list(f) == [b"a\n", b"b\n"]
    f.close()


def test_bytesio_writelines_no_separators():
    f = SpooledBytesIO()
    f.writelines([b"a", b"b", b"c"])
    assert f.getvalue() == b"abc"
    f.close()


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
    f.truncate(4)
    assert f.getvalue() == b"abcd"
    assert f.tell() == 0
    f.close()


def test_bytesio_truncate_size_larger_than_pos():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(2)
    f.truncate(4)
    assert f.getvalue() == b"abcd"
    assert f.tell() == 2
    f.close()


def test_bytesio_truncate_negative_raises():
    f = SpooledBytesIO()
    with pytest.raises(OSError):
        f.truncate(-1)
    f.close()


def test_bytesio_closed_operations_raise():
    f = SpooledBytesIO()
    f.write(b"x")
    f.close()
    with pytest.raises(ValueError):
        f.read()
    with pytest.raises(ValueError):
        f.write(b"y")
    with pytest.raises(ValueError):
        f.seek(0)
    with pytest.raises(ValueError):
        f.tell()
    with pytest.raises(ValueError):
        f.getvalue()


def test_bytesio_softspace_property():
    f = SpooledBytesIO()
    f.softspace = 1
    assert f.softspace == 1
    f.close()


def test_bytesio_pos_buf_properties():
    f = SpooledBytesIO()
    f.write(b"xyz")
    assert f.pos == 3
    assert f.buf == b"xyz"
    f.close()


def test_bytesio_bool_true():
    f = SpooledBytesIO()
    assert bool(f) is True
    f.close()


def test_bytesio_context_manager():
    with SpooledBytesIO() as f:
        f.write(b"data")
        assert f.getvalue() == b"data"
    assert f.closed is True


def test_bytesio_equality():
    a = SpooledBytesIO()
    b = SpooledBytesIO()
    a.write(b"hello\nworld")
    b.write(b"hello\nworld")
    a.seek(2)
    b.seek(4)
    assert a == b
    # positions restored
    assert a.tell() == 2
    assert b.tell() == 4
    a.close()
    b.close()


def test_bytesio_inequality():
    a = SpooledBytesIO()
    b = SpooledBytesIO()
    a.write(b"hello")
    b.write(b"world")
    assert a != b
    assert (a == b) is False
    a.close()
    b.close()


def test_bytesio_equality_other_type():
    a = SpooledBytesIO()
    assert (a == b"data") is False
    assert (a != b"data") is True
    a.close()


def test_bytesio_fileno_rolls_over(tmp_path):
    f = SpooledBytesIO(dir=str(tmp_path))
    f.write(b"abc")
    fd = f.fileno()
    assert isinstance(fd, int)
    assert f._rolled is True
    f.close()


def test_bytesio_seekable_readable_writable():
    f = SpooledBytesIO()
    assert f.seekable() is True
    assert f.readable() is True
    assert f.writable() is True
    f.close()


def test_bytesio_isatty():
    f = SpooledBytesIO()
    assert f.isatty() is False
    f.close()


def test_bytesio_flush():
    f = SpooledBytesIO()
    f.write(b"abc")
    f.flush()  # should not raise
    f.close()


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

def test_stringio_basic_roundtrip():
    f = SpooledStringIO()
    assert f.write(u"\u2014 Hey, an emdash!") == 17
    f.seek(0)
    assert f.read() == u"\u2014 Hey, an emdash!"
    f.close()


def test_stringio_write_non_str_raises():
    f = SpooledStringIO()
    with pytest.raises(TypeError):
        f.write(b"bytes")
    f.close()


def test_stringio_tell_in_codepoints():
    f = SpooledStringIO()
    f.write(u"abcde")
    assert f.tell() == 5
    f.seek(2)
    assert f.tell() == 2
    assert f.read(2) == u"cd"
    assert f.tell() == 4
    f.close()


def test_stringio_len_codepoints():
    f = SpooledStringIO()
    f.write(u"\u2014abc")
    assert f.len == 4
    assert len(f) == 4
    f.close()


def test_stringio_seek_set_negative_raises():
    f = SpooledStringIO()
    f.write(u"abc")
    with pytest.raises(ValueError):
        f.seek(-1)
    f.close()


def test_stringio_seek_cur():
    f = SpooledStringIO()
    f.write(u"abcdef")
    f.seek(1)
    f.seek(2, os.SEEK_CUR)
    assert f.tell() == 3
    assert f.read(2) == u"de"
    f.close()


def test_stringio_seek_cur_negative_backwards():
    f = SpooledStringIO()
    f.write(u"abcdef")
    f.seek(4)
    f.seek(-2, os.SEEK_CUR)
    assert f.tell() == 2
    assert f.read(2) == u"cd"
    f.close()


def test_stringio_seek_cur_before_start_raises():
    f = SpooledStringIO()
    f.write(u"abc")
    f.seek(1)
    with pytest.raises(ValueError):
        f.seek(-5, os.SEEK_CUR)
    f.close()


def test_stringio_seek_end():
    f = SpooledStringIO()
    f.write(u"abcdef")
    f.seek(0)
    f.seek(2, os.SEEK_END)
    assert f.tell() == 4
    assert f.read() == u"ef"
    f.close()


def test_stringio_seek_invalid_whence():
    f = SpooledStringIO()
    f.write(u"abc")
    with pytest.raises(ValueError):
        f.seek(0, 99)
    f.close()


def test_stringio_readline_and_readlines():
    f = SpooledStringIO()
    f.write(u"a\nb\nc")
    f.seek(0)
    assert f.readline() == u"a\n"
    f.seek(0)
    assert f.readlines() == [u"a\n", u"b\n", u"c"]
    f.close()


def test_stringio_rollover_on_max_size(tmp_path):
    f = SpooledStringIO(max_size=3, dir=str(tmp_path))
    f.write(u"abcdef")
    assert f._rolled is True
    f.seek(0)
    assert f.read() == u"abcdef"
    f.close()


def test_stringio_not_rolled_default():
    f = SpooledStringIO()
    f.write(u"x")
    assert f._rolled is False
    f.close()


def test_stringio_truncate():
    f = SpooledStringIO()
    f.write(u"abcdef")
    f.seek(0)
    f.truncate(3)
    assert f.getvalue() == u"abc"
    f.close()


def test_stringio_getvalue():
    f = SpooledStringIO()
    f.write(u"hello")
    f.seek(2)
    assert f.getvalue() == u"hello"
    assert f.tell() == 2
    f.close()


def test_stringio_closed_raises():
    f = SpooledStringIO()
    f.write(u"x")
    f.close()
    with pytest.raises(ValueError):
        f.read()
    with pytest.raises(ValueError):
        f.tell()


def test_stringio_equality():
    a = SpooledStringIO()
    b = SpooledStringIO()
    a.write(u"line1\nline2")
    b.write(u"line1\nline2")
    assert a == b
    a.close()
    b.close()


def test_stringio_inequality():
    a = SpooledStringIO()
    b = SpooledStringIO()
    a.write(u"foo")
    b.write(u"bar")
    assert a != b
    a.close()
    b.close()


def test_stringio_context_manager():
    with SpooledStringIO() as f:
        f.write(u"ctx")
        assert f.getvalue() == u"ctx"
    assert f.closed is True


def test_io_base_is_abstract():
    # SpooledIOBase should not be directly instantiable due to abstractmethods
    with pytest.raises(TypeError):
        SpooledIOBase()


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

def test_mfr_bytes_read():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"), BytesIO(b"e"))
    assert mfr.read(3) == b"abc"
    assert mfr.read(3) == b"de"
    assert mfr.read(3) == b""


def test_mfr_bytes_read_all():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"))
    assert mfr.read() == b"abcd"
    # non-positive amt reads all
    mfr.seek(0)
    assert mfr.read(0) == b"abcd"
    # exhausted files
    assert mfr.read() == b"cd"


def test_mfr_text_read():
    mfr = MultiFileReader(StringIO("ab"), StringIO("cd"))
    assert mfr.read(1) == "a"
    assert mfr.read() == "bcd"


def test_mfr_empty():
    mfr = MultiFileReader()
    assert mfr.read() == b""
    assert mfr.read(5) == b""


def test_mfr_non_filelike_raises():
    with pytest.raises(TypeError):
        MultiFileReader(object())


def test_mfr_missing_seek_raises():
    class ReadOnly:
        def read(self, n=-1):
            return b""
    with pytest.raises(TypeError):
        MultiFileReader(ReadOnly())


def test_mfr_mixed_types_raises():
    with pytest.raises(ValueError):
        MultiFileReader(BytesIO(b"ab"), StringIO("cd"))


def test_mfr_seek_to_start():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"))
    assert mfr.read(2) == b"ab"
    mfr.seek(0)
    assert mfr.read(2) == b"ab"


def test_mfr_seek_non_start_raises():
    mfr = MultiFileReader(BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(1)


def test_mfr_seek_invalid_whence_raises():
    mfr = MultiFileReader(BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(0, os.SEEK_END)
