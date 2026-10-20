import os
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

def test_bytesio_basic_read_write():
    f = SpooledBytesIO()
    assert f.write(b"hello") == 5
    assert f.tell() == 5
    assert f.seek(0) == 0
    assert f.read() == b"hello"


def test_bytesio_partial_read_and_readline():
    f = SpooledBytesIO()
    f.write(b"line1\nline2\nline3")
    f.seek(0)
    assert f.read(3) == b"lin"
    assert f.readline() == b"e1\n"
    f.seek(0)
    assert f.readlines() == [b"line1\n", b"line2\n", b"line3"]


def test_bytesio_write_textraises():
    f = SpooledBytesIO()
    with pytest.raises(TypeError):
        f.write("text")


def test_bytesio_accepts_bytearray_and_memoryview():
    f = SpooledBytesIO()
    assert f.write(bytearray(b"ab")) == 2
    assert f.write(memoryview(b"cd")) == 2
    f.seek(0)
    assert f.read() == b"abcd"


def test_bytesio_getvalue_preserves_position():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(2)
    assert f.getvalue() == b"abcdef"
    assert f.tell() == 2


def test_bytesio_len():
    f = SpooledBytesIO()
    f.write(b"12345")
    assert len(f) == 5
    assert f.len == 5


def test_bytesio_rollover_on_exceed():
    f = SpooledBytesIO(max_size=4)
    assert not f._rolled
    f.write(b"12345")  # 5 >= 4 => rollover
    assert f._rolled
    f.seek(0)
    assert f.read() == b"12345"
    assert f.len == 5


def test_bytesio_rollover_manual():
    f = SpooledBytesIO()
    f.write(b"data")
    assert not f._rolled
    f.rollover()
    assert f._rolled
    f.seek(0)
    assert f.read() == b"data"
    # second rollover no-op
    f.rollover()
    assert f._rolled


def test_bytesio_truncate_noarg():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(3)
    f.truncate()
    f.seek(0)
    assert f.read() == b"abc"


def test_bytesio_truncate_with_size():
    f = SpooledBytesIO()
    f.write(b"abcdef")
    f.seek(0)
    f.truncate(2)
    f.seek(0)
    assert f.read() == b"ab"


def test_bytesio_truncate_negative():
    f = SpooledBytesIO()
    f.write(b"abc")
    with pytest.raises(OSError):
        f.truncate(-1)


def test_bytesio_closed_operations():
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
        f.seek(0)
    with pytest.raises(ValueError):
        f.tell()
    with pytest.raises(ValueError):
        f.getvalue()


def test_bytesio_context_manager():
    with SpooledBytesIO() as f:
        f.write(b"ctx")
        f.seek(0)
        assert f.read() == b"ctx"
    assert f.closed


def test_bytesio_iteration():
    f = SpooledBytesIO()
    f.write(b"a\nb\nc\n")
    f.seek(0)
    assert list(f) == [b"a\n", b"b\n", b"c\n"]


def test_bytesio_writelines():
    f = SpooledBytesIO()
    f.writelines([b"x", b"y", b"z"])
    f.seek(0)
    assert f.read() == b"xyz"


def test_bytesio_equality():
    f1 = SpooledBytesIO()
    f1.write(b"same")
    f2 = SpooledBytesIO()
    f2.write(b"same")
    assert f1 == f2
    assert not (f1 != f2)

    f3 = SpooledBytesIO()
    f3.write(b"different")
    assert f1 != f3
    assert not (f1 == f3)

    # Not equal to a different class
    assert f1 != b"same"
    assert (f1 == b"same") is False


def test_bytesio_bool_and_iter():
    f = SpooledBytesIO()
    assert bool(f) is True
    f.write(b"x")
    f.seek(0)
    assert iter(f) is f


def test_bytesio_fileno_rolls():
    f = SpooledBytesIO()
    f.write(b"fileno")
    fd = f.fileno()
    assert isinstance(fd, int)
    assert f._rolled


def test_bytesio_properties():
    f = SpooledBytesIO()
    f.write(b"props")
    assert f.pos == f.tell()
    assert f.buf == b"props"
    assert f.seekable() is True
    assert f.readable() is True
    assert f.writable() is True
    assert f.isatty() is False


def test_bytesio_flush():
    f = SpooledBytesIO()
    f.write(b"x")
    f.flush()  # should not raise


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

def test_stringio_basic():
    f = SpooledStringIO()
    assert f.write("hello") == 5
    assert f.tell() == 5
    assert f.seek(0) == 0
    assert f.read() == "hello"


def test_stringio_write_typeerror():
    f = SpooledStringIO()
    with pytest.raises(TypeError):
        f.write(b"bytes")


def test_stringio_unicode_emdash():
    f = SpooledStringIO()
    f.write("\u2014 Hi")
    f.seek(0)
    assert f.read() == "\u2014 Hi"


def test_stringio_readline():
    f = SpooledStringIO()
    f.write("a\nb\nc")
    f.seek(0)
    assert f.readline() == "a\n"
    assert f.readline() == "b\n"
    assert f.readline() == "c"


def test_stringio_readlines():
    f = SpooledStringIO()
    f.write("a\nb\nc")
    f.seek(0)
    assert f.readlines() == ["a\n", "b\n", "c"]


def test_stringio_seek_cur():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(2)
    assert f.seek(1, os.SEEK_CUR) == 3
    assert f.read(1) == "d"


def test_stringio_seek_negative_cur():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(4)
    assert f.seek(-2, os.SEEK_CUR) == 2
    assert f.read(1) == "c"


def test_stringio_seek_cur_negative_below_zero():
    f = SpooledStringIO()
    f.write("abc")
    f.seek(1)
    with pytest.raises(ValueError):
        f.seek(-5, os.SEEK_CUR)


def test_stringio_seek_negative_set():
    f = SpooledStringIO()
    f.write("abc")
    with pytest.raises(ValueError):
        f.seek(-1, os.SEEK_SET)


def test_stringio_seek_end():
    f = SpooledStringIO()
    f.write("abcdef")
    # SEEK_END with pos=0 -> end position (6)
    assert f.seek(0, os.SEEK_END) == 6


def test_stringio_seek_invalid_whence():
    f = SpooledStringIO()
    f.write("abc")
    with pytest.raises(ValueError):
        f.seek(0, 99)


def test_stringio_len():
    f = SpooledStringIO()
    f.write("hello")
    assert f.len == 5
    assert len(f) == 5


def test_stringio_len_multibyte():
    f = SpooledStringIO()
    f.write("\u2014")  # single codepoint, 3 utf-8 bytes
    assert len(f) == 1


def test_stringio_rollover_on_exceed():
    f = SpooledStringIO(max_size=4)
    assert not f._rolled
    f.write("12345")  # 5 bytes >= 4 => rollover
    assert f._rolled
    f.seek(0)
    assert f.read() == "12345"


def test_stringio_getvalue():
    f = SpooledStringIO()
    f.write("value")
    f.seek(1)
    assert f.getvalue() == "value"
    assert f.tell() == 1


def test_stringio_iteration():
    f = SpooledStringIO()
    f.write("x\ny\nz\n")
    f.seek(0)
    assert list(f) == ["x\n", "y\n", "z\n"]


def test_stringio_truncate():
    f = SpooledStringIO()
    f.write("abcdef")
    f.seek(0)
    f.truncate(2)
    f.seek(0)
    assert f.read() == "ab"


def test_stringio_closed():
    f = SpooledStringIO()
    f.write("abc")
    f.close()
    assert f.closed
    with pytest.raises(ValueError):
        f.read()
    with pytest.raises(ValueError):
        f.write("x")


def test_stringio_equality():
    f1 = SpooledStringIO()
    f1.write("same")
    f2 = SpooledStringIO()
    f2.write("same")
    assert f1 == f2

    f3 = SpooledStringIO()
    f3.write("other")
    assert f1 != f3


def test_stringio_multibyte_traverse():
    # exercise _traverse_codepoints with large reads
    f = SpooledStringIO()
    s = "a\u2014b" * 100
    f.write(s)
    f.seek(0)
    assert f.read() == s
    assert f.seek(1) == 1
    assert f.read(2) == "\u2014b"


# ---------------------------------------------------------------------------
# SpooledIOBase abstractness
# ---------------------------------------------------------------------------

def test_spooled_io_base_is_abstract():
    with pytest.raises(TypeError):
        SpooledIOBase()


# ---------------------------------------------------------------------------
# is_text_fileobj
# ---------------------------------------------------------------------------

def test_is_text_fileobj_str_stream():
    from io import StringIO, BytesIO
    assert is_text_fileobj(StringIO("x")) is True
    assert is_text_fileobj(BytesIO(b"x")) is False


def test_is_text_fileobj_encoding_attr():
    class FakeText:
        encoding = "utf-8"

    class FakeBytes:
        @property
        def getvalue(self):
            return b"abc"

    assert is_text_fileobj(FakeText()) is True
    assert is_text_fileobj(FakeBytes()) is False


def test_is_text_fileobj_no_attrs():
    class NoAttrs:
        pass

    assert is_text_fileobj(NoAttrs()) is False


def test_is_text_fileobj_getvalue_raises():
    class BadGetvalue:
        def getvalue(self):
            raise RuntimeError("boom")

    assert is_text_fileobj(BadGetvalue()) is False


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

def test_mfr_basic_read():
    from io import BytesIO
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"), BytesIO(b"e"))
    assert mfr.read(3) == b"abc"
    assert mfr.read(3) == b"de"
    assert mfr.read(3) == b""


def test_mfr_read_all():
    from io import BytesIO
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"))
    assert mfr.read() == b"abcd"


def test_mfr_text_joiner():
    from io import StringIO
    mfr = MultiFileReader(StringIO("ab"), StringIO("cd"))
    assert mfr.read() == "abcd"
    assert mfr.seek(0) is None


def test_mfr_mixed_raises():
    from io import BytesIO, StringIO
    with pytest.raises(ValueError):
        MultiFileReader(BytesIO(b"a"), StringIO("b"))


def test_mfr_non_filelike_raises():
    with pytest.raises(TypeError):
        MultiFileReader(object())


def test_mfr_empty():
    mfr = MultiFileReader()
    assert mfr.read() == b""
    assert mfr.read(5) == b""


def test_mfr_seek_to_start():
    from io import BytesIO
    mfr = MultiFileReader(BytesIO(b"ab"), BytesIO(b"cd"))
    mfr.read(2)
    mfr.seek(0)
    assert mfr.read(4) == b"abcd"


def test_mfr_seek_nonzero_raises():
    from io import BytesIO
    mfr = MultiFileReader(BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(1)


def test_mfr_seek_non_set_whence_raises():
    from io import BytesIO
    mfr = MultiFileReader(BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(0, os.SEEK_CUR)


def test_mfr_read_exact_across_boundaries():
    from io import BytesIO
    mfr = MultiFileReader(BytesIO(b"12345"), BytesIO(b"6789"))
    assert mfr.read(5) == b"12345"
    assert mfr.read(4) == b"6789"
    assert mfr.read(1) == b""


def test_read_chunk_size_constant():
    assert isinstance(READ_CHUNK_SIZE, int)
    assert READ_CHUNK_SIZE > 0
