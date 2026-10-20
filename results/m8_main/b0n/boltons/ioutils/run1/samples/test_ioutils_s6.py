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
# READ_CHUNK_SIZE constant
# ---------------------------------------------------------------------------

def test_read_chunk_size_value():
    assert READ_CHUNK_SIZE == 21333


# ---------------------------------------------------------------------------
# SpooledBytesIO
# ---------------------------------------------------------------------------

def test_bytesio_write_read_getvalue():
    with SpooledBytesIO() as f:
        assert f.write(b"Happy IO") == 8
        f.seek(0)
        val = f.getvalue()
    assert isinstance(val, bytes)
    assert val == b"Happy IO"


def test_bytesio_accepts_bytearray_and_memoryview():
    with SpooledBytesIO() as f:
        assert f.write(bytearray(b"abc")) == 3
        assert f.write(memoryview(b"def")) == 3
        f.seek(0)
        assert f.getvalue() == b"abcdef"


def test_bytesio_write_text_raises_typeerror():
    with SpooledBytesIO() as f:
        with pytest.raises(TypeError):
            f.write("a string")


def test_bytesio_read_and_readline():
    with SpooledBytesIO() as f:
        f.write(b"line1\nline2\n")
        f.seek(0)
        assert f.readline() == b"line1\n"
        assert f.read() == b"line2\n"


def test_bytesio_readline_length():
    with SpooledBytesIO() as f:
        f.write(b"abcdef")
        f.seek(0)
        assert f.readline(3) == b"abc"


def test_bytesio_readlines():
    with SpooledBytesIO() as f:
        f.write(b"a\nb\nc\n")
        f.seek(0)
        assert f.readlines() == [b"a\n", b"b\n", b"c\n"]


def test_bytesio_tell_seek():
    with SpooledBytesIO() as f:
        f.write(b"hello")
        assert f.tell() == 5
        assert f.seek(2) == 2
        assert f.tell() == 2
        f.seek(0)
        assert f.read() == b"hello"


def test_bytesio_seek_whence_end():
    with SpooledBytesIO() as f:
        f.write(b"hello")
        f.seek(-2, os.SEEK_END)
        assert f.read() == b"lo"


def test_bytesio_len():
    with SpooledBytesIO() as f:
        f.write(b"12345")
        assert f.len == 5
        assert len(f) == 5


def test_bytesio_rollover_to_tempfile():
    with SpooledBytesIO(max_size=4) as f:
        assert f._rolled is False
        f.write(b"abcdef")  # exceeds max_size, triggers rollover
        assert f._rolled is True
        f.seek(0)
        assert f.getvalue() == b"abcdef"


def test_bytesio_rollover_explicit():
    with SpooledBytesIO() as f:
        f.write(b"data")
        f.rollover()
        assert f._rolled is True
        f.seek(0)
        assert f.getvalue() == b"data"


def test_bytesio_rollover_idempotent():
    with SpooledBytesIO() as f:
        f.write(b"data")
        f.rollover()
        buf_after_first = f.buffer
        f.rollover()
        assert f.buffer is buf_after_first


def test_bytesio_getvalue_preserves_position():
    with SpooledBytesIO() as f:
        f.write(b"hello")
        f.seek(2)
        assert f.getvalue() == b"hello"
        assert f.tell() == 2


def test_bytesio_writelines():
    with SpooledBytesIO() as f:
        f.writelines([b"a", b"b", b"c"])
        f.seek(0)
        assert f.getvalue() == b"abc"


def test_bytesio_truncate_no_arg():
    with SpooledBytesIO() as f:
        f.write(b"hello")
        f.truncate()
        f.seek(0)
        assert f.getvalue() == b""


def test_bytesio_truncate_to_size():
    with SpooledBytesIO() as f:
        f.write(b"hello")
        f.truncate(3)
        f.seek(0)
        assert f.getvalue() == b"hel"


def test_bytesio_truncate_negative_raises():
    with SpooledBytesIO() as f:
        f.write(b"hello")
        with pytest.raises(OSError):
            f.truncate(-1)


def test_bytesio_truncate_restores_pos_when_less():
    with SpooledBytesIO() as f:
        f.write(b"hello")
        f.seek(1)
        f.truncate(4)
        assert f.tell() == 1


def test_bytesio_fileno():
    with SpooledBytesIO() as f:
        f.write(b"data")
        # fileno triggers rollover and gives a real fd
        fd = f.fileno()
        assert isinstance(fd, int)


def test_bytesio_closed_raises_on_read():
    f = SpooledBytesIO()
    f.write(b"data")
    f.close()
    with pytest.raises(ValueError):
        f.read()


def test_bytesio_closed_property():
    f = SpooledBytesIO()
    assert f.closed is False
    f.close()
    assert f.closed is True


def test_bytesio_seekable_readable_writable():
    with SpooledBytesIO() as f:
        assert f.seekable() is True
        assert f.readable() is True
        assert f.writable() is True


def test_bytesio_iteration():
    with SpooledBytesIO() as f:
        f.write(b"a\nb\nc\n")
        f.seek(0)
        assert list(f) == [b"a\n", b"b\n", b"c\n"]


def test_bytesio_iteration_at_eof_stops():
    with SpooledBytesIO() as f:
        f.write(b"a\n")
        f.seek(0)
        lines = list(f)
        assert lines == [b"a\n"]


def test_bytesio_eq():
    with SpooledBytesIO() as a, SpooledBytesIO() as b:
        a.write(b"same\n")
        b.write(b"same\n")
        assert a == b
        assert not (a != b)


def test_bytesio_ne():
    with SpooledBytesIO() as a, SpooledBytesIO() as b:
        a.write(b"one\n")
        b.write(b"two\n")
        assert a != b
        assert not (a == b)


def test_bytesio_eq_non_spooled_returns_false():
    with SpooledBytesIO() as a:
        a.write(b"x")
        assert (a == object()) is False


def test_bytesio_eq_restores_positions():
    with SpooledBytesIO() as a, SpooledBytesIO() as b:
        a.write(b"line\n")
        b.write(b"line\n")
        a.seek(2)
        b.seek(1)
        a == b
        assert a.tell() == 2
        assert b.tell() == 1


def test_bytesio_pos_and_buf_properties():
    with SpooledBytesIO() as f:
        f.write(b"hello")
        assert f.pos == 5
        assert f.buf == b"hello"


def test_bytesio_bool():
    with SpooledBytesIO() as f:
        assert bool(f) is True


def test_bytesio_context_manager_closes():
    with SpooledBytesIO() as f:
        f.write(b"x")
    assert f.closed is True


def test_bytesio_softspace():
    with SpooledBytesIO() as f:
        f.write(b"x")
        # BytesIO supports softspace attribute
        f.softspace = 1
        assert f.softspace == 1


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

def test_stringio_write_read():
    with SpooledStringIO() as f:
        assert f.write("Hello") == 5
        f.seek(0)
        assert f.read() == "Hello"


def test_stringio_unicode():
    with SpooledStringIO() as f:
        f.write("\u2014 Hey")
        f.seek(0)
        val = f.read()
    assert isinstance(val, str)
    assert val == "\u2014 Hey"


def test_stringio_write_bytes_raises():
    with SpooledStringIO() as f:
        with pytest.raises(TypeError):
            f.write(b"bytes")


def test_stringio_tell_is_codepoints():
    with SpooledStringIO() as f:
        f.write("abc")
        assert f.tell() == 3
        f.write("\u00e9")  # e-acute is 2 utf-8 bytes, 1 codepoint
        assert f.tell() == 4


def test_stringio_len_codepoints():
    with SpooledStringIO() as f:
        f.write("a\u00e9b")  # 3 codepoints, 4 utf-8 bytes
        assert f.len == 3
        assert len(f) == 3


def test_stringio_seek_set():
    with SpooledStringIO() as f:
        f.write("hello")
        assert f.seek(2) == 2
        assert f.read() == "llo"


def test_stringio_seek_set_negative_raises():
    with SpooledStringIO() as f:
        f.write("hello")
        with pytest.raises(ValueError):
            f.seek(-1)


def test_stringio_seek_cur_positive():
    with SpooledStringIO() as f:
        f.write("abcdef")
        f.seek(2)
        assert f.seek(2, os.SEEK_CUR) == 4
        assert f.read() == "ef"


def test_stringio_seek_cur_negative():
    with SpooledStringIO() as f:
        f.write("abcdef")
        f.seek(4)
        assert f.seek(-2, os.SEEK_CUR) == 2
        assert f.read() == "cdef"


def test_stringio_seek_cur_negative_out_of_range():
    with SpooledStringIO() as f:
        f.write("abc")
        f.seek(1)
        with pytest.raises(ValueError):
            f.seek(-5, os.SEEK_CUR)


def test_stringio_seek_from_end():
    with SpooledStringIO() as f:
        f.write("abcdef")
        # Note: implementation uses len - pos for SEEK_END
        f.seek(2, os.SEEK_END)
        assert f.read() == "ef"


def test_stringio_seek_invalid_whence():
    with SpooledStringIO() as f:
        f.write("abc")
        with pytest.raises(ValueError):
            f.seek(0, 5)


def test_stringio_read_n():
    with SpooledStringIO() as f:
        f.write("abcdef")
        f.seek(0)
        assert f.read(3) == "abc"
        assert f.read(3) == "def"
        assert f.read() == ""


def test_stringio_readline():
    with SpooledStringIO() as f:
        f.write("a\nb\n")
        f.seek(0)
        assert f.readline() == "a\n"
        assert f.readline() == "b\n"


def test_stringio_readlines():
    with SpooledStringIO() as f:
        f.write("a\nb\nc\n")
        f.seek(0)
        assert f.readlines() == ["a\n", "b\n", "c\n"]


def test_stringio_getvalue():
    with SpooledStringIO() as f:
        f.write("hello")
        f.seek(1)
        assert f.getvalue() == "hello"
        assert f.tell() == 1


def test_stringio_rollover_to_tempfile():
    with SpooledStringIO(max_size=4) as f:
        assert f._rolled is False
        f.write("abcdef")
        assert f._rolled is True
        f.seek(0)
        assert f.getvalue() == "abcdef"


def test_stringio_iteration():
    with SpooledStringIO() as f:
        f.write("a\nb\n")
        f.seek(0)
        assert list(f) == ["a\n", "b\n"]


def test_stringio_writelines():
    with SpooledStringIO() as f:
        f.writelines(["a\n", "b\n"])
        f.seek(0)
        assert f.getvalue() == "a\nb\n"


def test_stringio_truncate():
    with SpooledStringIO() as f:
        f.write("hello")
        f.truncate(3)
        f.seek(0)
        assert f.getvalue() == "hel"


def test_stringio_closed_raises():
    f = SpooledStringIO()
    f.write("data")
    f.close()
    with pytest.raises(ValueError):
        f.read()


def test_stringio_eq():
    with SpooledStringIO() as a, SpooledStringIO() as b:
        a.write("same\n")
        b.write("same\n")
        assert a == b
        assert not (a != b)


def test_stringio_ne():
    with SpooledStringIO() as a, SpooledStringIO() as b:
        a.write("one\n")
        b.write("two\n")
        assert a != b


def test_stringio_multibyte_seek():
    with SpooledStringIO() as f:
        f.write("a\u00e9b\u00e9c")  # codepoints: a é b é c
        f.seek(2)
        assert f.read() == "b\u00e9c"
        f.seek(0)
        assert f.read() == "a\u00e9b\u00e9c"


# ---------------------------------------------------------------------------
# SpooledIOBase
# ---------------------------------------------------------------------------

def test_spooled_io_base_is_base():
    assert issubclass(SpooledBytesIO, SpooledIOBase)
    assert issubclass(SpooledStringIO, SpooledIOBase)


def test_spooled_io_base_defaults():
    b = SpooledBytesIO()
    assert b._max_size == 5000000
    assert b._dir is None
    b.close()


# ---------------------------------------------------------------------------
# is_text_fileobj
# ---------------------------------------------------------------------------

def test_is_text_fileobj_stringio():
    assert is_text_fileobj(StringIO("hi")) is True


def test_is_text_fileobj_bytesio():
    assert is_text_fileobj(BytesIO(b"hi")) is False


def test_is_text_fileobj_encoding():
    class WithEncoding:
        encoding = 'utf-8'
    assert is_text_fileobj(WithEncoding()) is True


def test_is_text_fileobj_getvalue_str():
    class WithGetValue:
        def getvalue(self):
            return "text"
    assert is_text_fileobj(WithGetValue()) is True


def test_is_text_fileobj_getvalue_bytes():
    class WithGetValue:
        def getvalue(self):
            return b"bytes"
    assert is_text_fileobj(WithGetValue()) is False


def test_is_text_fileobj_getvalue_raises():
    class WithGetValue:
        def getvalue(self):
            raise RuntimeError("boom")
    assert is_text_fileobj(WithGetValue()) is False


def test_is_text_fileobj_plain_object():
    assert is_text_fileobj(object()) is False


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

def test_mfr_read_basic():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"), BytesIO(b"e"))
    assert mfr.read(3) == b"abc"
    assert mfr.read(3) == b"de"
    assert mfr.read(3) == b""


def test_mfr_read_all():
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"), BytesIO(b"e"))
    assert mfr.read() == b"abcde"


def test_mfr_read_none_returns_join():
    mfr = MultiFileReader(BytesIO(b"x"), BytesIO(b"y"))
    assert mfr.read(None) == b"xy"


def test_mfr_read_zero_reads_all():
    mfr = MultiFileReader(BytesIO(b"x"), BytesIO(b"y"))
    assert mfr.read(0) == b"xy"


def test_mfr_text_mode():
    mfr = MultiFileReader(StringIO("ab"), StringIO("cd"))
    assert mfr.read(3) == "abc"
    assert mfr.read(3) == "d"


def test_mfr_mixed_raises_valueerror():
    with pytest.raises(ValueError):
        MultiFileReader(BytesIO(b"a"), StringIO("b"))


def test_mfr_non_fileobj_raises_typeerror():
    with pytest.raises(TypeError):
        MultiFileReader(object())


def test_mfr_missing_read_raises_typeerror():
    class OnlySeek:
        def seek(self, *a):
            pass
    with pytest.raises(TypeError):
        MultiFileReader(OnlySeek())


def test_mfr_empty():
    mfr = MultiFileReader()
    assert mfr.read() == b""
    assert mfr.read(5) == b""


def test_mfr_seek_start():
    f1, f2 = BytesIO(b"abc"), BytesIO(b"def")
    mfr = MultiFileReader(f1, f2)
    assert mfr.read(2) == b"ab"
    mfr.seek(0)
    assert f1.tell() == 0
    assert f2.tell() == 0


def test_mfr_seek_nonzero_raises():
    mfr = MultiFileReader(BytesIO(b"abc"))
    with pytest.raises(NotImplementedError):
        mfr.seek(1)


def test_mfr_seek_nonzero_whence_raises():
    mfr = MultiFileReader(BytesIO(b"abc"))
    with pytest.raises(NotImplementedError):
        mfr.seek(0, os.SEEK_CUR)


def test_mfr_joiner_bytes():
    mfr = MultiFileReader(BytesIO(b"a"))
    assert isinstance(mfr._joiner, bytes)


def test_mfr_joiner_text():
    mfr = MultiFileReader(StringIO("a"))
    assert isinstance(mfr._joiner, str)
