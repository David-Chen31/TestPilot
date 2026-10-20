import io
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

def test_bytesio_basic_write_read():
    f = SpooledBytesIO()
    try:
        assert f.write(b"hello") == 5
        assert f.tell() == 5
        f.seek(0)
        assert f.read() == b"hello"
        assert f.read() == b""
    finally:
        f.close()


def test_bytesio_write_returns_count_and_is_bytes():
    f = SpooledBytesIO()
    try:
        assert f.write(b"abc") == 3
        f.seek(0)
        assert isinstance(f.getvalue(), bytes)
        assert f.getvalue() == b"abc"
    finally:
        f.close()


def test_bytesio_write_text_raises_typeerror():
    f = SpooledBytesIO()
    try:
        with pytest.raises(TypeError):
            f.write("text")
    finally:
        f.close()


def test_bytesio_write_memoryview_and_bytearray():
    f = SpooledBytesIO()
    try:
        f.write(bytearray(b"ab"))
        f.write(memoryview(b"cd"))
        f.seek(0)
        assert f.read() == b"abcd"
    finally:
        f.close()


def test_bytesio_rollover(tmp_path):
    f = SpooledBytesIO(max_size=4, dir=str(tmp_path))
    try:
        assert f._rolled is False
        f.write(b"abcdef")
        assert f._rolled is True
        f.seek(0)
        assert f.read() == b"abcdef"
        assert f.len == 6
    finally:
        f.close()


def test_bytesio_rolled_len_and_fileno(tmp_path):
    f = SpooledBytesIO(max_size=2, dir=str(tmp_path))
    try:
        f.write(b"abcdef")
        assert f._rolled is True
        assert f.len == 6
        fd = f.fileno()
        assert isinstance(fd, int)
    finally:
        f.close()


def test_bytesio_rollover_idempotent(tmp_path):
    f = SpooledBytesIO(max_size=2, dir=str(tmp_path))
    try:
        f.write(b"xxx")
        assert f._rolled is True
        buf = f.buffer
        f.rollover()
        assert f.buffer is buf
    finally:
        f.close()


def test_bytesio_len_unrolled():
    f = SpooledBytesIO()
    try:
        f.write(b"12345")
        assert f.len == 5
        assert len(f) == 5
    finally:
        f.close()


def test_bytesio_len_preserves_position():
    f = SpooledBytesIO()
    try:
        f.write(b"hello")
        f.seek(2)
        assert f.len == 5
        assert f.tell() == 2
    finally:
        f.close()


def test_bytesio_readline_and_readlines():
    f = SpooledBytesIO()
    try:
        f.write(b"a\nb\nc")
        f.seek(0)
        assert f.readline() == b"a\n"
        assert f.readline(2) == b"b\n"
        f.seek(0)
        assert f.readlines() == [b"a\n", b"b\n", b"c"]
    finally:
        f.close()


def test_bytesio_writelines():
    f = SpooledBytesIO()
    try:
        f.writelines([b"a", b"b", b"c"])
        f.seek(0)
        assert f.read() == b"abc"
    finally:
        f.close()


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

def test_stringio_basic_write_read():
    f = SpooledStringIO()
    try:
        assert f.write("hello") == 5
        assert f.tell() == 5
        f.seek(0)
        assert f.read() == "hello"
    finally:
        f.close()


def test_stringio_unicode_emdash():
    f = SpooledStringIO()
    try:
        s = "\u2014 Hey, an emdash!"
        assert f.write(s) == len(s)
        f.seek(0)
        got = f.read()
        assert isinstance(got, str)
        assert got == s
    finally:
        f.close()


def test_stringio_write_bytes_raises_typeerror():
    f = SpooledStringIO()
    try:
        with pytest.raises(TypeError):
            f.write(b"bytes")
    finally:
        f.close()


def test_stringio_seek_set_cur_end():
    f = SpooledStringIO()
    try:
        f.write("abcdef")
        f.seek(2)
        assert f.tell() == 2
        assert f.read(2) == "cd"
        assert f.tell() == 4
        f.seek(0)
        f.seek(1, os.SEEK_CUR)
        assert f.tell() == 1
        f.seek(0, os.SEEK_END)
        assert f.tell() == 6
    finally:
        f.close()


def test_stringio_seek_invalid_whence():
    f = SpooledStringIO()
    try:
        f.write("abc")
        with pytest.raises(ValueError):
            f.seek(0, 99)
    finally:
        f.close()


def test_stringio_negative_seek_raises():
    f = SpooledStringIO()
    try:
        f.write("abc")
        with pytest.raises(ValueError):
            f.seek(-1)
    finally:
        f.close()


def test_stringio_seek_cur_negative_backwards():
    f = SpooledStringIO()
    try:
        f.write("abcdef")
        f.seek(4)
        f.seek(-2, os.SEEK_CUR)
        assert f.tell() == 2
    finally:
        f.close()


def test_stringio_seek_cur_negative_past_start_raises():
    f = SpooledStringIO()
    try:
        f.write("abc")
        f.seek(1)
        with pytest.raises(ValueError):
            f.seek(-5, os.SEEK_CUR)
    finally:
        f.close()


def test_stringio_len_codepoints():
    f = SpooledStringIO()
    try:
        f.write("abc")
        f.write("\u2014")
        assert f.len == 4
        assert len(f) == 4
    finally:
        f.close()


def test_stringio_readline_readlines():
    f = SpooledStringIO()
    try:
        f.write("a\nb\nc")
        f.seek(0)
        assert f.readline() == "a\n"
        f.seek(0)
        assert f.readlines() == ["a\n", "b\n", "c"]
    finally:
        f.close()


def test_stringio_rollover(tmp_path):
    f = SpooledStringIO(max_size=4, dir=str(tmp_path))
    try:
        assert f._rolled is False
        f.write("abcdef")
        assert f._rolled is True
        f.seek(0)
        assert f.read() == "abcdef"
        assert f.len == 6
    finally:
        f.close()


def test_stringio_getvalue():
    f = SpooledStringIO()
    try:
        f.write("hello")
        f.seek(2)
        assert f.getvalue() == "hello"
        assert f.tell() == 2
    finally:
        f.close()


def test_stringio_unicode_rollover_len(tmp_path):
    f = SpooledStringIO(max_size=2, dir=str(tmp_path))
    try:
        f.write("\u2014\u2014\u2014")
        assert f._rolled is True
        assert f.len == 3
        f.seek(0)
        assert f.read() == "\u2014\u2014\u2014"
    finally:
        f.close()


# ---------------------------------------------------------------------------
# SpooledIOBase shared behavior
# ---------------------------------------------------------------------------

def test_softspace_property():
    f = SpooledBytesIO()
    try:
        f.softspace = 1
        assert f.softspace == 1
    finally:
        f.close()


def test_file_property():
    f = SpooledBytesIO()
    try:
        assert f._file is f.buffer
    finally:
        f.close()


def test_pos_and_buf_properties():
    f = SpooledBytesIO()
    try:
        f.write(b"abc")
        assert f.pos == 3
        assert f.buf == b"abc"
    finally:
        f.close()


def test_flush_isatty_seekable_readable_writable():
    f = SpooledBytesIO()
    try:
        f.write(b"x")
        f.flush()
        assert f.isatty() is False
        assert f.seekable() is True
        assert f.readable() is True
        assert f.writable() is True
    finally:
        f.close()


def test_truncate_no_args():
    f = SpooledBytesIO()
    try:
        f.write(b"abcdef")
        f.seek(3)
        f.truncate()
        f.seek(0)
        assert f.read() == b"abc"
    finally:
        f.close()


def test_truncate_with_size():
    f = SpooledBytesIO()
    try:
        f.write(b"abcdef")
        f.seek(5)
        f.truncate(2)
        assert f.tell() == 5
        f.seek(0)
        assert f.read() == b"ab"
    finally:
        f.close()


def test_truncate_negative_size_raises():
    f = SpooledBytesIO()
    try:
        f.write(b"abc")
        with pytest.raises(OSError):
            f.truncate(-1)
    finally:
        f.close()


def test_truncate_position_before_size():
    f = SpooledBytesIO()
    try:
        f.write(b"abcdef")
        f.seek(1)
        f.truncate(4)
        assert f.tell() == 1
        f.seek(0)
        assert f.read() == b"abcd"
    finally:
        f.close()


def test_closed_behavior():
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


def test_context_manager():
    with SpooledBytesIO() as f:
        f.write(b"hi")
        f.seek(0)
        assert f.read() == b"hi"
    assert f.closed is True


def test_iteration():
    f = SpooledStringIO()
    try:
        f.write("a\nb\nc\n")
        f.seek(0)
        assert list(f) == ["a\n", "b\n", "c\n"]
    finally:
        f.close()


def test_eq_and_ne():
    a = SpooledBytesIO()
    b = SpooledBytesIO()
    try:
        a.write(b"same")
        b.write(b"same")
        assert a == b
        assert not (a != b)
        b.seek(0)
        b.write(b"diff")
        b.truncate()
        b.seek(0)
        b.write(b"samy")
        b.truncate()
        assert a != b
        assert not (a == b)
    finally:
        a.close()
        b.close()


def test_eq_different_type_returns_false():
    a = SpooledBytesIO()
    try:
        a.write(b"x")
        assert (a == "not a file") is False
        assert (a != "not a file") is True
    finally:
        a.close()


def test_bool_true():
    f = SpooledBytesIO()
    try:
        assert bool(f) is True
    finally:
        f.close()


def test_abstract_base_cannot_instantiate():
    with pytest.raises(TypeError):
        SpooledIOBase()


# ---------------------------------------------------------------------------
# is_text_fileobj
# ---------------------------------------------------------------------------

def test_is_text_fileobj_stringio():
    assert is_text_fileobj(io.StringIO("abc")) is True


def test_is_text_fileobj_bytesio():
    assert is_text_fileobj(io.BytesIO(b"abc")) is False


def test_is_text_fileobj_with_encoding():
    class Enc:
        encoding = "utf-8"

    assert is_text_fileobj(Enc()) is True


def test_is_text_fileobj_getvalue_bytes():
    class FakeBytes:
        def getvalue(self):
            return b"abc"

    assert is_text_fileobj(FakeBytes()) is False


def test_is_text_fileobj_getvalue_raises():
    class Bad:
        def getvalue(self):
            raise RuntimeError("nope")

    assert is_text_fileobj(Bad()) is False


def test_is_text_fileobj_plain_object():
    assert is_text_fileobj(object()) is False


def test_is_text_fileobj_spooled_stringio():
    f = SpooledStringIO()
    try:
        assert is_text_fileobj(f) is False  # no .getvalue? actually has getvalue -> str
    finally:
        f.close()


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

def test_mfr_read_bytes():
    mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"), io.BytesIO(b"e"))
    assert mfr.read(3) == b"abc"
    assert mfr.read(3) == b"de"
    assert mfr.read(3) == b""
    assert mfr.read() == b""


def test_mfr_read_all():
    mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
    assert mfr.read() == b"abcd"


def test_mfr_read_text():
    mfr = MultiFileReader(io.StringIO("ab"), io.StringIO("cd"))
    assert mfr.read(3) == "abc"
    assert mfr.read() == "d"


def test_mfr_mixed_types_raises_valueerror():
    with pytest.raises(ValueError):
        MultiFileReader(io.BytesIO(b"a"), io.StringIO("b"))


def test_mfr_non_fileobj_raises_typeerror():
    with pytest.raises(TypeError):
        MultiFileReader("not a file")


def test_mfr_no_args_reads_empty():
    mfr = MultiFileReader()
    assert mfr.read() == b""
    assert mfr.read(5) == b""


def test_mfr_seek_to_zero():
    mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
    assert mfr.read() == b"abcd"
    mfr.seek(0)
    assert mfr.read() == b"abcd"


def test_mfr_seek_nonzero_raises():
    mfr = MultiFileReader(io.BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(1)


def test_mfr_seek_bad_whence_raises():
    mfr = MultiFileReader(io.BytesIO(b"ab"))
    with pytest.raises(NotImplementedError):
        mfr.seek(0, os.SEEK_END)


def test_read_chunk_size_constant():
    assert READ_CHUNK_SIZE == 21333
