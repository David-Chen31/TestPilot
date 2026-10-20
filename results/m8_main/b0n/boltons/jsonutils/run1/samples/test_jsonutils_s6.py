import io
import json
import os

import pytest

from boltons.jsonutils import (
    JSONLIterator,
    reverse_iter_lines,
    DEFAULT_BLOCKSIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bytes_io(data: bytes) -> io.BytesIO:
    return io.BytesIO(data)


def make_text_io(data: str, encoding="utf-8") -> io.StringIO:
    sio = io.StringIO(data)
    # StringIO doesn't have .encoding by default in older versions; set it.
    try:
        sio.encoding = encoding
    except (AttributeError, TypeError):
        pass
    return sio


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

class TestReverseIterLinesBytes:
    def test_basic(self):
        f = make_bytes_io(b"one\ntwo\nthree\n")
        assert list(reverse_iter_lines(f)) == [b"three", b"two", b"one"]

    def test_no_trailing_newline(self):
        f = make_bytes_io(b"one\ntwo\nthree")
        assert list(reverse_iter_lines(f)) == [b"three", b"two", b"one"]

    def test_single_line_no_newline(self):
        f = make_bytes_io(b"only")
        assert list(reverse_iter_lines(f)) == [b"only"]

    def test_single_line_with_newline(self):
        f = make_bytes_io(b"only\n")
        # trailing newline yields an empty first entry in reverse
        assert list(reverse_iter_lines(f)) == [b"", b"only"]

    def test_empty_file(self):
        f = make_bytes_io(b"")
        assert list(reverse_iter_lines(f)) == []

    def test_only_newline(self):
        f = make_bytes_io(b"\n")
        assert list(reverse_iter_lines(f)) == [b"", b""]

    def test_small_blocksize(self):
        data = b"line1\nline2\nline3\nline4\n"
        f = make_bytes_io(data)
        result = list(reverse_iter_lines(f, blocksize=4))
        assert result == [b"", b"line4", b"line3", b"line2", b"line1"]

    def test_blocksize_one(self):
        data = b"a\nb\nc\n"
        f = make_bytes_io(data)
        result = list(reverse_iter_lines(f, blocksize=1))
        assert result == [b"", b"c", b"b", b"a"]

    def test_preseek_false_position_at_end(self):
        f = make_bytes_io(b"a\nb\nc\n")
        f.seek(0, os.SEEK_END)
        assert list(reverse_iter_lines(f, preseek=False)) == [b"", b"c", b"b", b"a"]

    def test_preseek_false_middle(self):
        # position after "a\n"
        f = make_bytes_io(b"a\nb\nc\n")
        f.seek(2)
        result = list(reverse_iter_lines(f, preseek=False))
        # Only lines fully before the cursor position are produced.
        assert result == [b"a"]

    def test_encoding_decodes(self):
        f = make_bytes_io("héllo\nwörld\n".encode("utf-8"))
        result = list(reverse_iter_lines(f, encoding="utf-8"))
        assert result == ["wörld", "héllo"]

    def test_follows_file_encoding_attribute(self):
        raw = io.BytesIO("héllo\nwörld\n".encode("latin-1"))
        # TextIOWrapper exposes .buffer and .encoding
        tw = io.TextIOWrapper(raw, encoding="latin-1")
        result = list(reverse_iter_lines(tw))
        assert result == ["wörld", "héllo"]
        # ensure original still usable
        assert raw.closed is False


class TestReverseIterLinesText:
    def test_basic(self):
        f = make_text_io("a\nb\nc\n")
        result = list(reverse_iter_lines(f, encoding="utf-8"))
        assert result == ["", "c", "b", "a"]

    def test_uses_text_buffer_without_encoding(self):
        # TextIOWrapper has buffer + encoding attribute
        raw = io.BytesIO(b"x\ny\n")
        tw = io.TextIOWrapper(raw, encoding="utf-8")
        result = list(reverse_iter_lines(tw))
        assert result == ["y", "x"]

    def test_text_stream_no_encoding_attr_raises(self):
        # A plain StringIO-like used as text but not decodable; here we just
        # test that supplying encoding makes it work.
        f = io.StringIO("a\nb\n")
        # StringIO has no .buffer, and its .encoding attribute may be None;
        # feeding encoding explicitly should produce decoded str lines.
        result = list(reverse_iter_lines(f, encoding="utf-8"))
        assert result == ["b", "a"]


# ---------------------------------------------------------------------------
# JSONLIterator
# ---------------------------------------------------------------------------

class TestJSONLIteratorForward:
    def test_basic(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f)
        assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_no_trailing_newline(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}')
        assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}]

    def test_skips_blank_lines(self):
        f = make_bytes_io(b'\n{"a": 1}\n\n\n{"b": 2}\n\n')
        assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}]

    def test_leading_whitespace_stripped(self):
        f = make_bytes_io(b'   {"a": 1}\n\t{"b": 2}\n')
        assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}]

    def test_ignore_errors(self):
        f = make_bytes_io(b'{"a": 1}\nnot json\n{"b": 2}\n')
        it = JSONLIterator(f, ignore_errors=True)
        assert list(it) == [{"a": 1}, {"b": 2}]

    def test_raises_on_bad_line(self):
        f = make_bytes_io(b'{"a": 1}\nnot json\n{"b": 2}\n')
        it = JSONLIterator(f)
        assert next(it) == {"a": 1}
        with pytest.raises(ValueError):
            next(it)

    def test_empty_file(self):
        f = make_bytes_io(b"")
        assert list(JSONLIterator(f)) == []

    def test_text_file(self):
        f = make_text_io('{"a": 1}\n{"b": 2}\n')
        assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}]

    def test_iter_returns_self(self):
        f = make_bytes_io(b'{"a": 1}\n')
        it = JSONLIterator(f)
        assert iter(it) is it

    def test_cur_byte_pos_advances(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}\n')
        it = JSONLIterator(f)
        start = it.cur_byte_pos
        next(it)
        end = it.cur_byte_pos
        assert end > start


class TestJSONLIteratorReverse:
    def test_basic_reverse(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, reverse=True)
        assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]

    def test_reverse_no_trailing_newline(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}')
        it = JSONLIterator(f, reverse=True)
        assert list(it) == [{"b": 2}, {"a": 1}]

    def test_reverse_skips_blank_lines(self):
        f = make_bytes_io(b'{"a": 1}\n\n\n{"b": 2}\n\n')
        it = JSONLIterator(f, reverse=True)
        assert list(it) == [{"b": 2}, {"a": 1}]

    def test_reverse_ignore_errors(self):
        f = make_bytes_io(b'{"a": 1}\nnot json\n{"b": 2}\n')
        it = JSONLIterator(f, reverse=True, ignore_errors=True)
        assert list(it) == [{"b": 2}, {"a": 1}]

    def test_reverse_raises_on_bad_line(self):
        f = make_bytes_io(b'{"a": 1}\nnot json\n{"b": 2}\n')
        it = JSONLIterator(f, reverse=True)
        assert next(it) == {"b": 2}
        with pytest.raises(ValueError):
            next(it)

    def test_reverse_empty_file(self):
        f = make_bytes_io(b"")
        assert list(JSONLIterator(f, reverse=True)) == []


class TestJSONLIteratorRelSeek:
    def test_rel_seek_zero_forward(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, rel_seek=0.0)
        assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_rel_seek_one_forward(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, rel_seek=1.0)
        assert list(it) == []

    def test_rel_seek_negative(self):
        # -1.0 maps to 0.0 (start)
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, rel_seek=-1.0)
        assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_rel_seek_out_of_range_high(self):
        f = make_bytes_io(b'{"a": 1}\n')
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=1.5)

    def test_rel_seek_out_of_range_low(self):
        f = make_bytes_io(b'{"a": 1}\n')
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=-1.5)

    def test_rel_seek_mid_forward_aligns_to_newline(self):
        data = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'
        f = make_bytes_io(data)
        # seek approximately to middle; iterator aligns to next newline
        it = JSONLIterator(f, rel_seek=0.5)
        # After aligning, we start at a line boundary; we should get at least
        # some subset of objects starting from a boundary.
        objs = list(it)
        # Every yielded object must be one of the known ones, in order.
        known = [{"a": 1}, {"b": 2}, {"c": 3}]
        # objs is a suffix of known
        assert objs == known[len(known) - len(objs):]

    def test_rel_seek_reverse_one(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, reverse=True)
        assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]

    def test_rel_seek_reverse_zero(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, reverse=True, rel_seek=0.0)
        # Starting at byte 0 in reverse yields nothing
        assert list(it) == []


class TestJSONLIteratorWithTempFile:
    def test_tmp_file_forward_and_reverse(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_bytes(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')

        with open(p, "rb") as f:
            assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}, {"c": 3}]

        with open(p, "rb") as f:
            assert list(JSONLIterator(f, reverse=True)) == [
                {"c": 3}, {"b": 2}, {"a": 1}
            ]

    def test_tmp_text_file(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        with open(p, "r", encoding="utf-8") as f:
            assert list(JSONLIterator(f, reverse=True)) == [{"b": 2}, {"a": 1}]


class TestJSONLIteratorNextMethod:
    def test_next_alias(self):
        f = make_bytes_io(b'{"a": 1}\n')
        it = JSONLIterator(f)
        assert it.next() == {"a": 1}
        with pytest.raises(StopIteration):
            it.next()

    def test_dunder_next(self):
        f = make_bytes_io(b'{"a": 1}\n')
        it = JSONLIterator(f)
        assert it.__next__() == {"a": 1}
        with pytest.raises(StopIteration):
            it.__next__()

    def test_builtin_next(self):
        f = make_bytes_io(b'{"a": 1}\n{"b": 2}\n')
        it = JSONLIterator(f)
        assert next(it) == {"a": 1}
        assert next(it) == {"b": 2}
        with pytest.raises(StopIteration):
            next(it)


class TestJSONLValues:
    def test_various_json_types(self):
        lines = [
            '1',
            '"hello"',
            'true',
            'null',
            '[1, 2, 3]',
            '{"nested": {"x": [1, 2]}}',
        ]
        data = ("\n".join(lines) + "\n").encode("utf-8")
        f = make_bytes_io(data)
        result = list(JSONLIterator(f))
        assert result == [1, "hello", True, None, [1, 2, 3],
                          {"nested": {"x": [1, 2]}}]

    def test_bad_line_no_ignore_middle(self):
        # json.JSONDecodeError is a subclass of ValueError
        f = make_bytes_io(b'[1, 2\n')
        it = JSONLIterator(f)
        with pytest.raises(ValueError):
            next(it)

    def test_ignore_errors_only_bad(self):
        f = make_bytes_io(b'bad\nbad2\n')
        it = JSONLIterator(f, ignore_errors=True)
        assert list(it) == []
