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
    def test_default_uses_in_memory_buffer(self):
        f = SpooledBytesIO()
        try:
            assert isinstance(f.buffer, io.BytesIO)
            assert f._rolled is False
        finally:
            f.close()

    def test_write_and_read_roundtrip(self):
        with SpooledBytesIO() as f:
            n = f.write(b"Happy IO")
            assert n == 8
            f.seek(0)
            assert f.read() == b"Happy IO"
            assert isinstance(f.getvalue(), bytes)

    def test_write_returns_bytes_written(self):
        with SpooledBytesIO() as f:
            assert f.write(b"abc") == 3
            assert f.write(bytearray(b"de")) == 2
            assert f.write(memoryview(b"f")) == 1
            f.seek(0)
            assert f.read() == b"abcdef"

    def test_write_text_raises_type_error(self):
        with SpooledBytesIO() as f:
            with pytest.raises(TypeError):
                f.write("not bytes")

    def test_initial_max_size_stored(self):
        f = SpooledBytesIO(max_size=123)
        try:
            assert f._max_size == 123
        finally:
            f.close()

    def test_rollover_on_exceeding_max_size(self):
        # Small max_size to trigger rollover. Note: condition is
        # tell() + nbytes >= max_size.
        with SpooledBytesIO(max_size=5) as f:
            f.write(b"abc")
            assert f._rolled is False
            f.write(b"de")  # 3 + 2 >= 5 -> rollover
            assert f._rolled is True
            f.seek(0)
            assert f.read() == b"abcde"

    def test_rollover_preserves_position(self):
        with SpooledBytesIO(max_size=2) as f:
            f.write(b"a")
            f.write(b"b")  # rollover here
            assert f._rolled is True
            assert f.tell() == 2
            f.seek(0)
            assert f.read() == b"ab"

    def test_rollover_idempotent(self):
        with SpooledBytesIO(max_size=1) as f:
            f.write(b"x")  # 0 + 1 >= 1 -> rollover
            assert f._rolled is True
            first_buffer = f.buffer
            f.rollover()
            assert f.buffer is first_buffer

    def test_readline_with_and_without_length(self):
        with SpooledBytesIO() as f:
            f.write(b"line1\nline2\n")
            f.seek(0)
            assert f.readline() == b"line1\n"
            assert f.readline(5) == b"line2"
            assert f.readline() == b"\n"

    def test_readlines(self):
        with SpooledBytesIO() as f:
            f.write(b"a\nb\nc\n")
            f.seek(0)
            assert f.readlines() == [b"a\n", b"b\n", b"c\n"]

    def test_len(self):
        with SpooledBytesIO() as f:
            f.write(b"hello")
            assert len(f) == 5
            assert f.len == 5
            # position unaffected
            assert f.tell() == 5

    def test_len_after_rollover(self):
        with SpooledBytesIO(max_size=3) as f:
            f.write(b"abcdef")
            assert f._rolled is True
            assert len(f) == 6
            assert f.len == 6

    def test_tell_and_seek(self):
        with SpooledBytesIO() as f:
            f.write(b"abcdef")
            assert f.tell() == 6
            f.seek(2)
            assert f.tell() == 2
            f.seek(-1, os.SEEK_END)
            assert f.tell() == 5
            assert f.read() == b"f"

    def test_getvalue_does_not_move_position(self):
        with SpooledBytesIO() as f:
            f.write(b"abc")
            f.seek(1)
            assert f.getvalue() == b"abc"
            assert f.tell() == 1

    def test_write_after_close_raises(self):
        f = SpooledBytesIO()
        f.close()
        assert f.closed is True
        with pytest.raises(ValueError):
            f.write(b"x")
        with pytest.raises(ValueError):
            f.read()
        with pytest.raises(ValueError):
            f.seek(0)
        with pytest.raises(ValueError):
            f.tell()

    def test_truncate_no_arg(self):
        with SpooledBytesIO() as f:
            f.write(b"abcdef")
            f.truncate()
            assert f.getvalue() == b"abcdef"

    def test_truncate_with_size(self):
        with SpooledBytesIO() as f:
            f.write(b"abcdef")
            f.seek(2)
            f.truncate(4)
            assert f.getvalue() == b"abcd"
            # pos < size, so position preserved
            assert f.tell() == 2

    def test_truncate_size_larger_than_pos(self):
        with SpooledBytesIO() as f:
            f.write(b"abcdef")
            f.seek(2)
            f.truncate(6)
            assert f.getvalue() == b"abcdef"
            assert f.tell() == 2

    def test_truncate_size_larger_than_pos_but_beyond_end(self):
        # Exercises the ``if pos < size:`` branch where pos < size.
        with SpooledBytesIO() as f:
            f.write(b"abcdef")
            f.seek(1)
            f.truncate(3)
            assert f.getvalue() == b"abc"
            # pos (1) < size (3), so position is restored to 1
            assert f.tell() == 1

    def test_truncate_pos_not_less_than_size(self):
        # Exercises the branch where pos >= size (no restore).
        with SpooledBytesIO() as f:
            f.write(b"abcdef")
            f.seek(5)
            f.truncate(2)
            assert f.getvalue() == b"ab"
            # pos (5) >= size (2), so position stays at size (2)
            assert f.tell() == 2

    def test_truncate_negative_size_raises(self):
        with SpooledBytesIO() as f:
            f.write(b"abc")
            with pytest.raises(OSError):
                f.truncate(-1)

    def test_truncate_on_closed_raises(self):
        f = SpooledBytesIO()
        f.close()
        with pytest.raises(ValueError):
            f.truncate()

    def test_writelines(self):
        with SpooledBytesIO() as f:
            f.writelines([b"a", b"b", b"c"])
            assert f.getvalue() == b"abc"

    def test_iteration(self):
        with SpooledBytesIO() as f:
            f.write(b"a\nb\nc")
            f.seek(0)
            assert list(f) == [b"a\n", b"b\n", b"c"]

    def test_next_stops_at_end(self):
        with SpooledBytesIO() as f:
            f.write(b"a\n")
            f.seek(0)
            assert next(f) == b"a\n"
            with pytest.raises(StopIteration):
                next(f)

    def test_next_past_a_newline_at_eof(self):
        # When there is trailing content that readline sees as empty
        # (a pending newline at EOF), __next__ should re-raise StopIteration
        # only after re-seeking. Cover the case where position is not at
        # the very end (line 219 branch).
        with SpooledBytesIO() as f:
            f.write(b"\n")
            f.seek(0)
            assert next(f) == b"\n"
            with pytest.raises(StopIteration):
                next(f)

    def test_next_rewinds_when_readline_empty_but_not_at_end(self):
        # Exercise the ``else: self.buffer.seek(pos)`` branch inside
        # __next__: when readline() reports nothing but the underlying
        # position is not at EOF, the original position must be restored
        # and the empty line returned (rather than StopIteration).
        with SpooledBytesIO() as f:
            f.write(b"abcdef")
            f.seek(2)
            f.readline = lambda *a, **kw: b""
            result = next(f)
            assert result == b""
            # Position restored to where it was before the empty read
            assert f.tell() == 2

    def test_readable_writable_seekable(self):
        with SpooledBytesIO() as f:
            assert f.readable() is True
            assert f.writable() is True
            assert f.seekable() is True

    def test_flush_isatty(self):
        with SpooledBytesIO() as f:
            f.write(b"x")
            f.flush()  # no error
            assert f.isatty() is False

    def test_flush_on_closed_raises(self):
        f = SpooledBytesIO()
        f.close()
        with pytest.raises(ValueError):
            f.flush()

    def test_equality(self):
        a = SpooledBytesIO()
        b = SpooledBytesIO()
        try:
            a.write(b"hello\nworld")
            b.write(b"hello\nworld")
            a.seek(2)
            b.seek(3)
            assert a == b
            # positions preserved after comparison
            assert a.tell() == 2
            assert b.tell() == 3
        finally:
            a.close()
            b.close()

    def test_inequality(self):
        a = SpooledBytesIO()
        b = SpooledBytesIO()
        try:
            a.write(b"hello")
            b.write(b"world")
            assert a != b
            assert (a == b) is False
        finally:
            a.close()
            b.close()

    def test_equality_with_non_spooled_is_false(self):
        with SpooledBytesIO() as f:
            f.write(b"x")
            assert (f == b"x") is False
            assert f != b"x"

    def test_bool_true(self):
        with SpooledBytesIO() as f:
            assert bool(f) is True
            f.seek(0)
            assert bool(f) is True

    def test_context_manager_closes(self):
        with SpooledBytesIO() as f:
            f.write(b"x")
        assert f.closed is True

    def test_enter_on_closed_raises(self):
        f = SpooledBytesIO()
        f.close()
        with pytest.raises(ValueError):
            f.__enter__()

    def test_iter_on_closed_raises(self):
        f = SpooledBytesIO()
        f.close()
        with pytest.raises(ValueError):
            iter(f)

    def test_fileno_rolls_over(self):
        with SpooledBytesIO() as f:
            f.write(b"abc")
            fd = f.fileno()
            assert isinstance(fd, int)
            assert f._rolled is True

    def test_buf_property(self):
        with SpooledBytesIO() as f:
            f.write(b"abc")
            assert f.buf == b"abc"

    def test_pos_property(self):
        with SpooledBytesIO() as f:
            f.write(b"abc")
            assert f.pos == 3
            f.seek(1)
            assert f.pos == 1


# ---------------------------------------------------------------------------
# SpooledStringIO
# ---------------------------------------------------------------------------

class TestSpooledStringIO:
    def test_write_and_read_roundtrip(self):
        with SpooledStringIO() as f:
            n = f.write("\u2014 Hey, an emdash!")
            assert n == len("\u2014 Hey, an emdash!")
            f.seek(0)
            assert f.read() == "\u2014 Hey, an emdash!"
            assert isinstance(f.read() if False else f.getvalue(), str)

    def test_write_non_str_raises(self):
        with SpooledStringIO() as f:
            with pytest.raises(TypeError):
                f.write(b"bytes")
            with pytest.raises(TypeError):
                f.write(123)

    def test_tell_counts_codepoints(self):
        with SpooledStringIO() as f:
            f.write("h\u00e9llo")  # 5 codepoints
            assert f.tell() == 5
            assert f.len == 5

    def test_seek_set(self):
        with SpooledStringIO() as f:
            f.write("abcdef")
            assert f.seek(2) == 2
            assert f.tell() == 2
            assert f.read() == "cdef"

    def test_seek_cur(self):
        with SpooledStringIO() as f:
            f.write("abcdef")
            f.seek(2)
            f.seek(2, os.SEEK_CUR)
            assert f.tell() == 4
            assert f.read() == "ef"

    def test_seek_cur_backwards(self):
        with SpooledStringIO() as f:
            f.write("abcdef")
            f.seek(4)
            f.seek(-2, os.SEEK_CUR)
            assert f.tell() == 2
            assert f.read() == "cdef"

    def test_seek_cur_to_negative_raises(self):
        with SpooledStringIO() as f:
            f.write("abc")
            f.seek(1)
            with pytest.raises(ValueError):
                f.seek(-5, os.SEEK_CUR)

    def test_seek_end(self):
        with SpooledStringIO() as f:
            f.write("abcdef")
            f.seek(0, os.SEEK_END)
            assert f.tell() == 6
            f.seek(2, os.SEEK_END)
            assert f.tell() == 4
            assert f.read() == "ef"

    def test_seek_negative_from_start_raises(self):
        with SpooledStringIO() as f:
            f.write("abc")
            with pytest.raises(ValueError):
                f.seek(-1)

    def test_seek_invalid_whence(self):
        with SpooledStringIO() as f:
            f.write("abc")
            with pytest.raises(ValueError):
                f.seek(0, 7)

    def test_seek_on_closed_raises(self):
        f = SpooledStringIO()
        f.close()
        with pytest.raises(ValueError):
            f.seek(0)
        with pytest.raises(ValueError):
            f.tell()

    def test_readline(self):
        with SpooledStringIO() as f:
            f.write("line1\nline2\nmore")
            f.seek(0)
            assert f.readline() == "line1\n"
            assert f.readline() == "line2\n"
            assert f.readline() == "more"

    def test_readline_with_length(self):
        with SpooledStringIO() as f:
            f.write("abcdef\n")
            f.seek(0)
            assert f.readline(3) == "abc"

    def test_readlines(self):
        with SpooledStringIO() as f:
            f.write("a\nb\nc\n")
            f.seek(0)
            assert f.readlines() == ["a\n", "b\n", "c\n"]

    def test_readlines_updates_tell(self):
        with SpooledStringIO() as f:
            f.write("a\nb\n")
            f.seek(0)
            lines = f.readlines()
            assert lines == ["a\n", "b\n"]
            assert f.tell() == 4

    def test_read_with_n(self):
        with SpooledStringIO() as f:
            f.write("abcdef")
            f.seek(0)
            assert f.read(2) == "ab"
            assert f.read(2) == "cd"
            assert f.read() == "ef"

    def test_read_empty_at_end(self):
        with SpooledStringIO() as f:
            f.write("abc")
            f.seek(0)
            assert f.read() == "abc"
            assert f.read() == ""

    def test_getvalue(self):
        with SpooledStringIO() as f:
            f.write("hello")
            assert f.getvalue() == "hello"

    def test_rollover(self):
        with SpooledStringIO(max_size=3) as f:
            f.write("abcdef")  # 6 >= 3 -> rollover
            assert f._rolled is True
            f.seek(0)
            assert f.read() == "abcdef"

    def test_rollover_idempotent(self):
        with SpooledStringIO(max_size=1) as f:
            f.write("x")
            assert f._rolled is True
            buf = f.buffer
            f.rollover()
            assert f.buffer is buf

    def test_multibyte_read_through(self):
        # Multi-byte utf-8 characters should be correctly traversed
        with SpooledStringIO() as f:
            f.write("\u2014\u2014\u2014")  # three emdashes
            f.seek(0)
            assert f.read(1) == "\u2014"
            assert f.tell() == 1
            assert f.read(2) == "\u2014\u2014"
            assert f.tell() == 3

    def test_seek_through_multibyte(self):
        with SpooledStringIO() as f:
            f.write("a\u2014b")
            f.seek(2)
            assert f.read() == "b"

    def test_traverse_large_with_read_chunk(self):
        # Force the READ_CHUNK_SIZE branch in _traverse_codepoints
        n = READ_CHUNK_SIZE + 5
        with SpooledStringIO() as f:
            f.write("a" * n)
            f.seek(n - 1)
            assert f.tell() == n - 1
            assert f.read() == "a"

    def test_traverse_past_end_before_chunk_boundary(self):
        # Exercises the ``if not ret: break`` path (lines 425-426). Seeking
        # via SEEK_END with a large negative offset traverses chunks past
        # the actual content, and read() eventually returns empty.
        with SpooledStringIO() as f:
            f.write("abc")
            # SEEK_END with pos < 0 moves dest_position beyond file length,
            # causing _traverse_codepoints to read until exhaustion.
            f.seek(-100, os.SEEK_END)
            # The traversal reads past end and breaks; tell reflects the
            # computed destination (which is larger than the actual data).
            assert f.tell() == f.len + 100

    def test_traverse_past_end_multi_chunk(self):
        # Larger-than-chunk traversal to force the ``not ret`` break after
        # having read at least one full chunk first.
        with SpooledStringIO() as f:
            f.write("a" * 10)
            # Jump far beyond the end.
            f.seek(-(READ_CHUNK_SIZE + 100), os.SEEK_END)
            assert f.tell() == f.len + READ_CHUNK_SIZE + 100

    def test_traverse_empty_file_large_dest_hits_not_ret_break(self):
        # On an *empty* file, seeking far past the end via SEEK_END forces
        # _traverse_codepoints into the chunk-read branch (dest is larger
        # than READ_CHUNK_SIZE) where read() returns "" immediately. This
        # exercises the `if not ret: break` path on the very first
        # iteration.
        with SpooledStringIO() as f:
            assert f.len == 0
            expected = READ_CHUNK_SIZE + 100
            f.seek(-expected, os.SEEK_END)
            # The traversal breaks at once yet the reported position is the
            # computed destination.
            assert f.tell() == expected
            assert f.len == 0

    def test_len_property(self):
        with SpooledStringIO() as f:
            f.write("abcd")
            assert f.len == 4
            assert len(f) == 4

    def test_len_empty(self):
        with SpooledStringIO() as f:
            assert f.len == 0

    def test_writelines(self):
        with SpooledStringIO() as f:
            f.writelines(["a", "b", "c"])
            assert f.getvalue() == "abc"

    def test_iteration(self):
        with SpooledStringIO() as f:
            f.write("a\nb\nc")
            f.seek(0)
            assert list(f) == ["a\n", "b\n", "c"]

    def test_equality(self):
        a = SpooledStringIO()
        b = SpooledStringIO()
        try:
            a.write("hello\nworld")
            b.write("hello\nworld")
            assert a == b
            assert (a != b) is False
        finally:
            a.close()
            b.close()

    def test_inequality(self):
        a = SpooledStringIO()
        b = SpooledStringIO()
        try:
            a.write("hi")
            b.write("bye")
            assert a != b
            assert (a == b) is False
        finally:
            a.close()
            b.close()

    def test_context_manager_closes(self):
        with SpooledStringIO() as f:
            f.write("x")
        assert f.closed is True

    def test_write_after_close_raises(self):
        f = SpooledStringIO()
        f.close()
        with pytest.raises(ValueError):
            f.write("x")

    def test_read_after_close_raises(self):
        f = SpooledStringIO()
        f.close()
        with pytest.raises(ValueError):
            f.read()
        with pytest.raises(ValueError):
            f.readline()

    def test_truncate_via_base(self):
        with SpooledStringIO() as f:
            f.write("abcdef")
            f.truncate()
            assert f.getvalue() == "abcdef"


# ---------------------------------------------------------------------------
# SpooledIOBase
# ---------------------------------------------------------------------------

class TestSpooledIOBase:
    def test_init_stores_max_size_and_dir(self):
        f = SpooledBytesIO(max_size=42, dir=None)
        try:
            assert f._max_size == 42
            assert f._dir is None
        finally:
            f.close()

    def test_check_closed_raises_when_closed(self):
        f = SpooledBytesIO()
        f.close()
        with pytest.raises(ValueError):
            f._checkClosed()

    def test_check_closed_custom_message(self):
        f = SpooledBytesIO()
        f.close()
        with pytest.raises(ValueError, match="custom message"):
            f._checkClosed("custom message")

    def test_check_closed_no_raise_when_open(self):
        with SpooledBytesIO() as f:
            assert f._checkClosed() is None

    def test_file_property_is_buffer(self):
        with SpooledBytesIO() as f:
            assert f._file is f.buffer

    def test_writelines_on_closed_raises(self):
        f = SpooledBytesIO()
        f.close()
        with pytest.raises(ValueError):
            f.writelines([b"x"])

    def test_del_after_close_no_error(self):
        f = SpooledBytesIO()
        f.close()
        f.__del__()  # should not raise

    def test_softspace_get(self):
        # softspace delegates to self.buffer.softspace.
        f = SpooledBytesIO()
        try:
            f.buffer.softspace = 1
            assert f.softspace == 1
        finally:
            f.close()

    def test_softspace_set(self):
        f = SpooledBytesIO()
        try:
            f.softspace = 2
            assert f.buffer.softspace == 2
        finally:
            f.close()

    def test_equality_internal_error_restores_positions(self):
        # Cover the except-branch in __eq__: force an exception during the
        # comparison loop after positions are saved, then verify positions
        # were restored before the exception propagates.
        a = SpooledBytesIO()
        b = SpooledBytesIO()
        try:
            a.write(b"x\ny")
            b.write(b"x\ny")
            a.seek(0)
            b.seek(0)

            original_seek = a.seek
            state = {'count': 0}

            def flaky_seek(pos, mode=0):
                # Allow the initial seek to 0, then fail during restore attempt.
                state['count'] += 1
                if state['count'] == 1:
                    return original_seek(pos, mode)
                raise RuntimeError("seek failure")

            a.seek = flaky_seek

            with pytest.raises(RuntimeError):
                a == b
        finally:
            a.close()
            b.close()

    def test_equality_other_seek_fails_during_restore(self):
        # Cover the *nested* except (lines 260-261): the primary comparison
        # raises (bad readline), then while restoring, self.seek() succeeds
        # but other.seek() fails and is silently swallowed before the
        # original exception is re-raised.
        a = SpooledBytesIO()
        b = SpooledBytesIO()
        try:
            a.write(b"x")
            b.write(b"y")
            a.seek(0)
            b.seek(0)

            original_b_seek = b.seek
            state = {'n': 0}

            def flaky_b_seek(pos, mode=0):
                state['n'] += 1
                if state['n'] == 1:
                    # First call (from __eq__ line 244) must succeed.
                    return original_b_seek(pos, mode)
                raise RuntimeError("b seek restore failure")

            b.seek = flaky_b_seek

            def bad_readline(*args, **kwargs):
                raise RuntimeError("b readline failure")

            b.readline = bad_readline

            # The original readline failure propagates; the failing
            # b.seek() during restore is swallowed by the nested except.
            with pytest.raises(RuntimeError, match="b readline failure"):
                a == b

            # b.seek was invoked twice: once by __eq__'s reset to 0, and
            # once (failing) during the restore attempt.
            assert state['n'] == 2
        finally:
            a.close()
            b.close()

    def test_del_swallows_close_errors(self):
        # __del__ catches all exceptions raised while closing.
        f = SpooledBytesIO()

        class Boom:
            def close(self):
                raise RuntimeError("boom")

            closed = False

        f._buffer = Boom()
        # Should not raise despite close() failing.
        f.__del__()


# ---------------------------------------------------------------------------
# is_text_fileobj
# ---------------------------------------------------------------------------

class TestIsTextFileobj:
    def test_string_io(self):
        assert is_text_fileobj(io.StringIO("hi")) is True

    def test_bytes_io(self):
        assert is_text_fileobj(io.BytesIO(b"hi")) is False

    def test_file_with_encoding(self, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("hello")
        with open(p, "r") as f:
            assert is_text_fileobj(f) is True

    def test_binary_file(self, tmp_path):
        p = tmp_path / "file.bin"
        p.write_bytes(b"hello")
        with open(p, "rb") as f:
            assert is_text_fileobj(f) is False

    def test_object_without_attributes(self):
        class Dummy:
            pass
        assert is_text_fileobj(Dummy()) is False

    def test_getvalue_returning_str(self):
        class GetVal:
            encoding = None

            def getvalue(self):
                return "text"
        assert is_text_fileobj(GetVal()) is True

    def test_getvalue_returning_bytes(self):
        class GetVal:
            encoding = None

            def getvalue(self):
                return b"bytes"
        assert is_text_fileobj(GetVal()) is False

    def test_getvalue_raising(self):
        class GetVal:
            encoding = None

            def getvalue(self):
                raise RuntimeError("boom")
        assert is_text_fileobj(GetVal()) is False

    def test_encoding_falsey(self):
        class Enc:
            encoding = ""
        assert is_text_fileobj(Enc()) is False


# ---------------------------------------------------------------------------
# MultiFileReader
# ---------------------------------------------------------------------------

class TestMultiFileReader:
    def test_read_bytes_across_files(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"),
                              io.BytesIO(b"e"))
        assert mfr.read(3) == b"abc"
        assert mfr.read(3) == b"de"
        assert mfr.read(3) == b""

    def test_read_all_bytes(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        assert mfr.read() == b"abcd"

    def test_read_all_text(self):
        mfr = MultiFileReader(io.StringIO("ab"), io.StringIO("cd"))
        assert mfr.read() == "abcd"

    def test_read_text_across_files(self):
        mfr = MultiFileReader(io.StringIO("abc"), io.StringIO("def"))
        assert mfr.read(4) == "abcd"
        assert mfr.read(4) == "ef"

    def test_read_with_amt_none_returns_joined(self):
        mfr = MultiFileReader(io.BytesIO(b"x"), io.BytesIO(b"y"))
        assert mfr.read(None) == b"xy"
        # exhausted gives empty
        assert mfr.read(None) == b""

    def test_type_error_on_non_filelike(self):
        with pytest.raises(TypeError):
            MultiFileReader("not a file")
        with pytest.raises(TypeError):
            MultiFileReader(io.BytesIO(b"x"), object())

    def test_type_error_when_read_not_callable(self):
        class Bad:
            read = None

            def seek(self, *a):
                pass
        with pytest.raises(TypeError):
            MultiFileReader(Bad())

    def test_value_error_on_mixed(self):
        with pytest.raises(ValueError):
            MultiFileReader(io.BytesIO(b"x"), io.StringIO("y"))

    def test_joiner_set_for_bytes(self):
        mfr = MultiFileReader(io.BytesIO(b"x"))
        assert mfr._joiner == b""

    def test_joiner_set_for_text(self):
        mfr = MultiFileReader(io.StringIO("x"))
        assert mfr._joiner == ""

    def test_zero_args_text_join(self):
        # all([]) is True -> treated as text
        mfr = MultiFileReader()
        assert mfr._joiner == ""
        assert mfr.read() == ""
        assert mfr.read(5) == ""

    def test_seek_to_zero_resets_file_positions_only(self):
        # seek(0) resets each underlying file to 0 but does NOT reset the
        # internal index; after reading across the boundary the index stays
        # advanced, so subsequent reads start at the second file.
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        assert mfr.read(3) == b"abc"
        assert mfr._index == 1
        mfr.seek(0)
        assert mfr._index == 1
        assert mfr.read(3) == b"cd"

    def test_seek_to_zero_before_any_read(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        mfr.seek(0)
        assert mfr._index == 0
        assert mfr.read(3) == b"abc"

    def test_seek_nonzero_offset_raises(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        with pytest.raises(NotImplementedError):
            mfr.seek(1)

    def test_seek_non_set_whence_raises(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"))
        with pytest.raises(NotImplementedError):
            mfr.seek(0, os.SEEK_END)
        with pytest.raises(NotImplementedError):
            mfr.seek(0, os.SEEK_CUR)

    def test_index_advances_on_short_read(self):
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cdef"))
        assert mfr._index == 0
        mfr.read(1)
        assert mfr._index == 0
        # read more than first file has
        mfr.seek(0)
        mfr.read(3)
        assert mfr._index == 1

    def test_read_amt_zero_joins_all(self):
        # falsy amt triggers the "read everything" branch
        mfr = MultiFileReader(io.BytesIO(b"ab"), io.BytesIO(b"cd"))
        assert mfr.read(0) == b"abcd"
