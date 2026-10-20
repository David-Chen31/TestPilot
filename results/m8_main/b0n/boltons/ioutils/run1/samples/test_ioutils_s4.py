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
# SpooledIOBase
# ---------------------------------------------------------------------------

def test_spooled_base_is_abstract():
    with pytest.raises(TypeError):
        SpooledIOBase()


def test_predicates_true():
    f = SpooledBytesIO()
    try:
        assert f.seekable() is True
        assert f.readable() is True
        assert f.writable() is True
    finally:
        f.close()


def test_bool_true():
    f = SpooledBytesIO()
    try:
        assert bool(f) is True
    finally:
        f.close()


def test_spooled_base_writelines():
    f = SpooledBytesIO()
    try:
        f.writelines([b"ab", b"cd", b"ef"])
        assert f.getvalue() == b"abcdef"
    finally:
        f.close()


def test_writelines_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.writelines([b"x"])


def test_spooled_base_pos_and_buf():
    f = SpooledBytesIO()
    try:
        f.write(b"hello")
        assert f.pos == 5
        assert f.buf == b"hello"
    finally:
        f.close()


def test_getvalue_preserves_position():
    f = SpooledBytesIO()
    try:
        f.write(b"abcdef")
        f.seek(2)
        assert f.getvalue() == b"abcdef"
        assert f.tell() == 2
    finally:
        f.close()


def test_getvalue_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.getvalue()


def test_truncate_none():
    f = SpooledBytesIO()
    try:
        f.write(b"abcdef")
        f.truncate()
        assert f.getvalue() == b"abcdef"
    finally:
        f.close()


def test_truncate_size():
    f = SpooledBytesIO()
    try:
        f.write(b"abcdef")
        f.truncate(3)
        assert f.getvalue() == b"abc"
    finally:
        f.close()


def test_truncate_size_restores_pos_when_smaller():
    f = SpooledBytesIO()
    try:
        f.write(b"abcdef")
        f.seek(2)
        f.truncate(4)
        assert f.getvalue() == b"abcd"
        assert f.tell() == 2
    finally:
        f.close()


def test_truncate_negative():
    f = SpooledBytesIO()
    try:
        f.write(b"abc")
        with pytest.raises(OSError) as excinfo:
            f.truncate(-1)
        assert excinfo.value.errno == 22  # EINVAL
    finally:
        f.close()


def test_truncate_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.truncate()


def test_len_uses_property():
    f = SpooledBytesIO()
    try:
        f.write(b"xyz")
        assert len(f) == 3
        assert f.len == 3
    finally:
        f.close()


def test_flush_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.flush()


def test_isatty_returns_bool():
    f = SpooledBytesIO()
    try:
        assert f.isatty() is False
    finally:
        f.close()


def test_isatty_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.isatty()


def test_context_manager():
    with SpooledBytesIO() as f:
        f.write(b"hi")
        assert f.closed is False
    assert f.closed is True


def test_iter_returns_self():
    f = SpooledBytesIO()
    try:
        assert iter(f) is f
    finally:
        f.close()


def test_iter_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        iter(f)


def test_next_iteration():
    f = SpooledBytesIO()
    try:
        f.write(b"a\nb\nc")
        f.seek(0)
        assert list(f) == [b"a\n", b"b\n", b"c"]
    finally:
        f.close()


def test_eq_same_class():
    a = SpooledBytesIO()
    b = SpooledBytesIO()
    try:
        a.write(b"abc\ndef\n")
        b.write(b"abc\ndef\n")
        assert a == b
        assert not (a != b)
    finally:
        a.close()
        b.close()


def test_eq_different_content():
    a = SpooledBytesIO()
    b = SpooledBytesIO()
    try:
        a.write(b"abc")
        b.write(b"xyz")
        assert a != b
        assert not (a == b)
    finally:
        a.close()
        b.close()


def test_eq_different_class_false():
    a = SpooledBytesIO()
    b = SpooledStringIO()
    try:
        a.write(b"abc")
        b.write("abc")
        assert (a == b) is False
        assert (a != b) is True
    finally:
        a.close()
        b.close()


def test_eq_restores_positions():
    a = SpooledBytesIO()
    b = SpooledBytesIO()
    try:
        a.write(b"abc\ndef\n")
        b.write(b"abc\ndef\n")
        a.seek(1)
        b.seek(4)
        assert a == b
        assert a.tell() == 1
        assert b.tell() == 4
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# SpooledBytesIO
# ---------------------------------------------------------------------------

def test_bytesio_basic_roundtrip():
    with SpooledBytesIO() as f:
        assert f.write(b"Happy IO") == 8
        f.seek(0)
        assert isinstance(f.getvalue(), bytes)
        assert f.read() == b"Happy IO"


def test_bytesio_write_accepts_bytearray_and_memoryview():
    with SpooledBytesIO() as f:
        assert f.write(bytearray(b"abc")) == 3
        assert f.write(memoryview(b"def")) == 3
        assert f.getvalue() == b"abcdef"


def test_bytesio_write_text_raises():
    f = SpooledBytesIO()
    try:
        with pytest.raises(TypeError):
            f.write("text")
    finally:
        f.close()


def test_bytesio_read_n():
    with SpooledBytesIO() as f:
        f.write(b"abcdef")
        f.seek(0)
        assert f.read(2) == b"ab"
        assert f.read(2) == b"cd"


def test_bytesio_write_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.write(b"x")


def test_bytesio_read_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.read()


def test_bytesio_seek_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.seek(0)


def test_bytesio_tell_closed():
    f = SpooledBytesIO()
    f.close()
    with pytest.raises(ValueError):
        f.tell()


def test_bytesio_readline():
    with SpooledBytesIO() as f:
        f.write(b"line1\nline2\n")
        f.seek(0)
        assert f.readline() == b"line1\n"
        assert f.readline(3) == b"lin"


def test_bytesio_readlines():
    with SpooledBytesIO() as f:
        f.write(b"a\nb\nc")
        f.seek(0)
        assert f.readlines() == [b"a\n", b"b\n", b"c"]


def test_bytesio_len_not_rolled():
    with SpooledBytesIO() as f:
        f.write(b"12345")
        assert f.len == 5
        assert f._rolled is False


def test_bytesio_rollover_on_size():
    with SpooledBytesIO(max_size=5) as f:
        f.write(b"123456")
        assert f._rolled is True
        assert f.getvalue() == b"123456"
        assert f.len == 6


def test_bytesio_rollover_preserves_position():
    f = SpooledBytesIO(max_size=10)
    try:
        f.write(b"abc")
        f.seek(1)
        f.rollover()
        assert f._rolled is True
        assert f.tell() == 1
        assert f.getvalue() == b"abc"
    finally:
        f.close()


def test_bytesio_fileno_rolls_over():
    f = SpooledBytesIO()
    try:
        f.write(b"abc")
        fd = f.fileno()
        assert isinstance(fd, int)
        assert f._rolled is True
    finally:
        f.close()


def test_bytesio_rollover_idempotent():
    f = SpooledBytesIO(max_size=2)
    try:
        f.write(b"abc")
        assert f._rolled is True
        buf = f.buffer
        f.rollover()
        assert f.buffer is buf
    finally:
        f.close()


def test_bytesio_rollover_dir(tmp_path):
    f = SpooledBytesIO(max_size=2, dir=str(tmp_path))
    try:
        f.write(b"abc")
        assert f._rolled is True
        assert f.getvalue() == b"abc"
    finally:
        f.close()


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

def test_stringio_basic_roundtrip():
    with SpooledStringIO() as f:
        assert f.write(u"\u2014 Hey, an emdash!") == 17
        f.seek(0)
        assert isinstance(f.read(), str)
        assert f.read() == ''


def test_stringio_write_non_str_raises():
    f = SpooledStringIO()
    try:
        with pytest.raises(TypeError):
            f.write(b"bytes")
    finally:
        f.close()


def test_stringio_reads_text():
    with SpooledStringIO() as f:
        f.write(u"h\u00e9llo")
        f.seek(0)
        assert f.read() == u"h\u00e9llo"


def test_stringio_tell_codepoints():
    with SpooledStringIO() as f:
        f.write(u"h\u00e9llo")
        assert f.tell() == 5
        assert len(f) == 5


def test_stringio_seek_set():
    with SpooledStringIO() as f:
        f.write(u"abcdef")
        f.seek(0)
        assert f.tell() == 0
        f.seek(3)
        assert f.tell() == 3
        assert f.read() == "def"


def test_stringio_seek_negative_raises():
    with SpooledStringIO() as f:
        f.write(u"abc")
        f.seek(0)
        with pytest.raises(ValueError):
            f.seek(-1)


def test_stringio_seek_cur_forward():
    with SpooledStringIO() as f:
        f.write(u"abcdef")
        f.seek(1)
        f.seek(2, os.SEEK_CUR)
        assert f.tell() == 3
        assert f.read(1) == "d"


def test_stringio_seek_cur_backward():
    with SpooledStringIO() as f:
        f.write(u"abcdef")
        f.seek(4)
        f.seek(-2, os.SEEK_CUR)
        assert f.tell() == 2
        assert f.read(1) == "c"


def test_stringio_seek_cur_negative_raises():
    with SpooledStringIO() as f:
        f.write(u"abc")
        f.seek(0)
        with pytest.raises(ValueError):
            f.seek(-1, os.SEEK_CUR)


def test_stringio_seek_invalid_whence():
    with SpooledStringIO() as f:
        f.write(u"abc")
        with pytest.raises(ValueError):
            f.seek(0, 5)


def test_stringio_seek_end():
    with SpooledStringIO() as f:
        f.write(u"abcdef")
        f.seek(0, os.SEEK_END)
        assert f.tell() == 6


def test_stringio_readline():
    with SpooledStringIO() as f:
        f.write(u"line1\nline2\n")
        f.seek(0)
        assert f.readline() == u"line1\n"
        assert f.readline() == u"line2\n"


def test_stringio_readlines():
    with SpooledStringIO() as f:
        f.write(u"a\nb\nc")
        f.seek(0)
        assert f.readlines() == [u"a\n", u"b\n", u"c"]


def test_stringio_multibyte_seek():
    with SpooledStringIO() as f:
        f.write(u"h\u00e9llo w\u00f6rld")
        f.seek(0)
        assert f.read(2) == u"h\u00e9"
        assert f.tell() == 2
        f.seek(0)
        assert f.read() == u"h\u00e9llo w\u00f6rld"


def test_stringio_seek_beyond_end():
    with SpooledStringIO() as f:
        f.write(u"abc")
        f.seek(10)
        assert f.read() == ''
        assert f.tell() == 10


def test_stringio_rollover():
    f = SpooledStringIO(max_size=4)
    try:
        f.write(u"abcdef")
        assert f._rolled is True
        assert f.getvalue() == u"abcdef"
        f.seek(0)
        assert f.read() == u"abcdef"
    finally:
        f.close()


def test_stringio_rollover_dir(tmp_path):
    f = SpooledStringIO(max_size=2, dir=str(tmp_path))
    try:
        f.write(u"abc")
        assert f._rolled is True
        assert f.len == 3
    finally:
        f.close()


def test_stringio_len_not_rolled():
    with SpooledStringIO() as f:
        f.write(u"abcdef")
        assert f.len == 6
        assert f._rolled is False


# ---------------------------------------------------------------------------
# is_text_fileobj
# ---------------------------------------------------------------------------

def test_is_text_fileobj_stringio():
    assert is_text_fileobj(io.StringIO()) is True


def test_is_text_fileobj_bytesio():
    assert is_text_fileobj(io.BytesIO()) is False


def test_is_text_fileobj_text_file(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    with open(str(p), "r") as f:
        assert is_text_fileobj(f) is True


def test_is_text_fileobj_binary_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    with open(str(p), "rb") as f:
        assert is_text_fileobj(f) is False


def test_is_text_fileobj_encoding_attr():
    class Dummy:
        encoding = "utf-8"
    assert is_text_fileobj(Dummy()) is True


def test_is_text_fileobj_getvalue_bytes():
    class Dummy:
        def getvalue(self):
            return b"bytes"
    assert is_text_fileobj(Dummy()) is False


def test_is_text_fileobj_getvalue_raises():
    class Dummy:
        def getvalue(self):
            raise RuntimeError("nope")
    assert is_text_fileobj(Dummy()) is False


def test_is_text_fileobj_plain_object():
    assert is_text_fileobj(object()) is False


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

def test_mfr_read_across_files():
    mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"), io.BytesIO(b"e"))
    assert mfr.read(3) == b"abc"
    assert mfr.read(3) == b"de"


def test_mfr_read_all():
    mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"), io.BytesIO(b"e"))
    assert mfr.read() == b"abcde"


def test_mfr_read_exhausted():
    mfr = MultiFileReader(io.BytesIO(b"ab"))
    assert mfr.read(10) == b"ab"
    assert mfr.read(10) == b""


def test_mfr_text_files():
    mfr = MultiFileReader(io.StringIO(u"ab"), io.StringIO(u"cd"))
    assert mfr.read(3) == u"abc"
    assert mfr.read() == u"d"


def test_mfr_mixed_raises():
    with pytest.raises(ValueError):
        MultiFileReader(io.BytesIO(b"ab"), io.StringIO(u"cd"))


def test_mfr_non_filelike_raises():
    with pytest.raises(TypeError):
        MultiFileReader("not a file")


def test_mfr_missing_seek_raises():
    class OnlyRead:
        def read(self, *a):
            return b""
    with pytest.raises(TypeError):
        MultiFileReader(OnlyRead())


def test_mfr_seek_zero():
    b1 = io.BytesIO(b"ab")
    b2 = io.BytesIO(b"cd")
    mfr = MultiFileReader(b1, b2)
    mfr.read(3)
    mfr.seek(0)
    assert b1.tell() == 0
    assert b2.tell() == 0


def test_mfr_seek_nonzero_raises():
    mfr = MultiFileReader(io.BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(1)


def test_mfr_seek_non_set_whence_raises():
    mfr = MultiFileReader(io.BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(0, os.SEEK_CUR)
