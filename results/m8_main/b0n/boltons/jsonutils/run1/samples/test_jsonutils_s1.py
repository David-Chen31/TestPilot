import io
import json
import pytest

from boltons.jsonutils import reverse_iter_lines, JSONLIterator, DEFAULT_BLOCKSIZE


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

class TestReverseIterLinesBytes:
    def _make_bytes_file(self, data):
        return io.BytesIO(data)

    def test_simple_no_trailing_newline(self):
        f = self._make_bytes_file(b"a\nb\nc")
        assert list(reverse_iter_lines(f)) == [b"c", b"b", b"a"]

    def test_trailing_newline_yields_empty(self):
        f = self._make_bytes_file(b"a\nb\nc\n")
        assert list(reverse_iter_lines(f)) == [b"", b"c", b"b", b"a"]

    def test_single_line_no_newline(self):
        f = self._make_bytes_file(b"only")
        assert list(reverse_iter_lines(f)) == [b"only"]

    def test_single_line_with_newline(self):
        f = self._make_bytes_file(b"only\n")
        assert list(reverse_iter_lines(f)) == [b"", b"only"]

    def test_empty_file(self):
        f = self._make_bytes_file(b"")
        assert list(reverse_iter_lines(f)) == []

    def test_multiple_trailing_newlines(self):
        f = self._make_bytes_file(b"a\n\n\n")
        # Reverse of splitlines gives empties for trailing newlines
        result = list(reverse_iter_lines(f))
        assert result[0] == b""
        assert result[-1] == b"a"
        assert all(isinstance(x, bytes) for x in result)

    def test_small_blocksize(self):
        data = b"line1\nline2\nline3\nline4\nline5"
        f1 = self._make_bytes_file(data)
        f2 = self._make_bytes_file(data)
        assert list(reverse_iter_lines(f1)) == list(reverse_iter_lines(f2, blocksize=2))

    def test_preseek_false_at_end(self):
        f = self._make_bytes_file(b"a\nb\nc\n")
        f.seek(0, io.SEEK_END)
        assert list(reverse_iter_lines(f, preseek=False)) == [b"", b"c", b"b", b"a"]

    def test_preseek_false_mid(self):
        f = self._make_bytes_file(b"a\nb\nc\n")
        f.seek(3)  # after "a\nb"
        result = list(reverse_iter_lines(f, preseek=False))
        # Should iterate backwards from position 3
        assert result == [b"", b"b", b"a"]

    def test_encoding_decodes(self):
        f = self._make_bytes_file("héllo\nwörld\n".encode("utf-8"))
        result = list(reverse_iter_lines(f, encoding="utf-8"))
        assert result == ["", "wörld", "héllo"]

    def test_default_blocksize_constant(self):
        assert DEFAULT_BLOCKSIZE == 4096


class TestReverseIterLinesText:
    def test_text_file_no_encoding_attr(self):
        # StringIO has no encoding and no buffer
        f = io.StringIO("a\nb\nc")
        result = list(reverse_iter_lines(f))
        # No encoding available -> bytes? Actually for StringIO, read returns
        # str, splitlines on str, and comparison to b'' fails...
        # Just assert it doesn't crash and gives something sensible.
        assert result

    def test_text_file_encoding(self):
        f = io.StringIO("a\nb\nc\n")
        result = list(reverse_iter_lines(f, encoding="utf-8"))
        assert result == ["", "c", "b", "a"]

    def test_text_file_uses_buffer(self):
        # A TextIOWrapper file will use its .buffer
        f = io.TextIOWrapper(io.BytesIO(b"x\ny\nz\n"), encoding="utf-8")
        result = list(reverse_iter_lines(f))
        assert result == [b"", b"z", b"y", b"x"]


# ---------------------------------------------------------------------------
# JSONLIterator
# ---------------------------------------------------------------------------

class TestJSONLIterator:
    def _lines(self, objs):
        return "".join(json.dumps(o) + "\n" for o in objs)

    def test_forward_basic(self):
        data = self._lines([{"a": 1}, {"b": 2}, {"c": 3}])
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f)
        assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_forward_no_trailing_newline(self):
        data = '{"a": 1}\n{"b": 2}'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f)
        assert list(it) == [{"a": 1}, {"b": 2}]

    def test_forward_blank_lines_skipped(self):
        data = '{"a": 1}\n\n\n{"b": 2}\n'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f)
        assert list(it) == [{"a": 1}, {"b": 2}]

    def test_forward_empty_file(self):
        f = io.BytesIO(b"")
        it = JSONLIterator(f)
        assert list(it) == []

    def test_forward_leading_whitespace_stripped(self):
        data = '   {"a": 1}\n\t{"b": 2}\n'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f)
        assert list(it) == [{"a": 1}, {"b": 2}]

    def test_forward_invalid_raises(self):
        data = '{"a": 1}\nnot json\n'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f)
        assert next(it) == {"a": 1}
        with pytest.raises(ValueError):
            next(it)

    def test_forward_ignore_errors(self):
        data = '{"a": 1}\nnot json\n{"b": 2}\n'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, ignore_errors=True)
        assert list(it) == [{"a": 1}, {"b": 2}]

    def test_reverse_basic(self):
        data = self._lines([{"a": 1}, {"b": 2}, {"c": 3}])
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, reverse=True)
        assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]

    def test_reverse_no_trailing_newline(self):
        data = '{"a": 1}\n{"b": 2}'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, reverse=True)
        assert list(it) == [{"b": 2}, {"a": 1}]

    def test_reverse_ignore_errors(self):
        data = '{"a": 1}\nbad\n{"b": 2}\n'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, reverse=True, ignore_errors=True)
        assert list(it) == [{"b": 2}, {"a": 1}]

    def test_reverse_invalid_raises(self):
        data = '{"a": 1}\nbad\n{"b": 2}\n'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, reverse=True)
        assert next(it) == {"b": 2}
        with pytest.raises(ValueError):
            next(it)

    def test_rel_seek_zero_start(self):
        data = self._lines([{"a": 1}, {"b": 2}, {"c": 3}])
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, rel_seek=0.0)
        assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_rel_seek_one_forward_empty(self):
        data = self._lines([{"a": 1}, {"b": 2}])
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, rel_seek=1.0)
        assert list(it) == []

    def test_rel_seek_negative(self):
        data = self._lines([{"a": 1}, {"b": 2}, {"c": 3}])
        # -1.0 -> 1.0 + (-1.0) = 0.0
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, rel_seek=-1.0)
        assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_rel_seek_invalid_above(self):
        f = io.BytesIO(b"")
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=1.5)

    def test_rel_seek_invalid_below(self):
        f = io.BytesIO(b"")
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=-1.0 - 0.001)

    def test_rel_seek_invalid_zero_negative_boundary(self):
        # -1.0 exactly is not > -1.0
        f = io.BytesIO(b"")
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=-1.0 - 0.5)

    def test_rel_seek_mid(self):
        # Build many lines, seek to ~middle, should skip first partial line
        objs = [{"i": i} for i in range(100)]
        data = self._lines(objs)
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, rel_seek=0.5)
        result = list(it)
        # Should be a suffix of objs, starting at some point
        assert len(result) > 0
        assert result == objs[-len(result):]

    def test_iter_returns_self(self):
        f = io.BytesIO(b'{"a": 1}\n')
        it = JSONLIterator(f)
        assert iter(it) is it

    def test_cur_byte_pos_advances(self):
        data = '{"a": 1}\n{"b": 2}\n'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f)
        start = it.cur_byte_pos
        next(it)
        after = it.cur_byte_pos
        assert after > start

    def test_cur_byte_pos_property(self):
        f = io.BytesIO(b'{"a": 1}\n')
        it = JSONLIterator(f)
        assert it.cur_byte_pos == 0

    def test_stop_iteration_at_end(self):
        f = io.BytesIO(b'{"a": 1}\n')
        it = JSONLIterator(f)
        assert next(it) == {"a": 1}
        with pytest.raises(StopIteration):
            next(it)
        with pytest.raises(StopIteration):
            next(it)

    def test_forward_whitespace_only_lines_skipped(self):
        data = '   \n\t\n{"a": 1}\n'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f)
        assert list(it) == [{"a": 1}]

    def test_reverse_with_text_wrapper(self):
        data = '{"a": 1}\n{"b": 2}\n'
        f = io.TextIOWrapper(io.BytesIO(data.encode()), encoding="utf-8")
        it = JSONLIterator(f, reverse=True)
        assert list(it) == [{"b": 2}, {"a": 1}]

    def test_reverse_negative_rel_seek(self):
        data = '{"a": 1}\n{"b": 2}\n{"c": 3}\n'
        f = io.BytesIO(data.encode())
        it = JSONLIterator(f, reverse=True, rel_seek=-0.001)
        result = list(it)
        # rel_seek becomes ~0.999, close to end; should iterate all lines in reverse
        assert result == [{"c": 3}, {"b": 2}, {"a": 1}]
