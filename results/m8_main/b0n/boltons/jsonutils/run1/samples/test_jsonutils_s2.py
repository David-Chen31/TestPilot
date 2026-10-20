import io
import json
import os

import pytest

from boltons.jsonutils import reverse_iter_lines, JSONLIterator, DEFAULT_BLOCKSIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_io(data):
    return io.BytesIO(data)


def _text_io(text):
    return io.StringIO(text)


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

class TestReverseIterLines:

    def test_default_blocksize(self):
        assert DEFAULT_BLOCKSIZE == 4096

    def test_simple_bytes(self):
        f = _bytes_io(b"a\nb\nc\n")
        assert list(reverse_iter_lines(f)) == [b"c", b"b", b"a"]

    def test_simple_text(self):
        f = _text_io("a\nb\nc\n")
        # StringIO has no encoding attribute -> yields str? Actually StringIO
        # has an .encoding of None, so encoding stays None -> bytes are
        # produced? No: StringIO returns str already, and decode is only
        # called when encoding truthy. So lines are str.
        assert list(reverse_iter_lines(f)) == ["c", "b", "a"]

    def test_no_trailing_newline_bytes(self):
        f = _bytes_io(b"a\nb\nc")
        assert list(reverse_iter_lines(f)) == [b"c", b"b", b"a"]

    def test_single_line_no_newline(self):
        f = _bytes_io(b"onlyline")
        assert list(reverse_iter_lines(f)) == [b"onlyline"]

    def test_single_line_with_newline(self):
        f = _bytes_io(b"onlyline\n")
        # trailing newline logic yields "" then the line
        assert list(reverse_iter_lines(f)) == [b"", b"onlyline"]

    def test_empty_file(self):
        f = _bytes_io(b"")
        assert list(reverse_iter_lines(f)) == []

    def test_blank_line_in_middle(self):
        f = _bytes_io(b"a\n\nb\n")
        assert list(reverse_iter_lines(f)) == [b"b", b"", b"a"]

    def test_blank_line_at_start(self):
        f = _bytes_io(b"\na\n")
        assert list(reverse_iter_lines(f)) == [b"a", b""]

    def test_small_blocksize(self):
        data = b"first\nsecond\nthird\n"
        f = _bytes_io(data)
        result = list(reverse_iter_lines(f, blocksize=2))
        assert result == [b"third", b"second", b"first"]

    def test_blocksize_one(self):
        f = _bytes_io(b"x\ny\n")
        assert list(reverse_iter_lines(f, blocksize=1)) == [b"y", b"x"]

    def test_preseek_false_from_end(self):
        f = _bytes_io(b"a\nb\nc\n")
        f.seek(0, os.SEEK_END)
        assert list(reverse_iter_lines(f, preseek=False)) == [b"c", b"b", b"a"]

    def test_preseek_false_mid_file(self):
        f = _bytes_io(b"a\nb\nc\nd\n")
        # position after "b\n" i.e. at index 4
        f.seek(4)
        assert list(reverse_iter_lines(f, preseek=False)) == [b"b", b"a"]

    def test_preseek_false_at_start(self):
        f = _bytes_io(b"a\nb\n")
        assert list(reverse_iter_lines(f, preseek=False)) == []

    def test_encoding_explicit_bytes(self):
        f = _bytes_io("héllo\nwörld\n".encode("utf-8"))
        result = list(reverse_iter_lines(f, encoding="utf-8"))
        assert result == ["wörld", "héllo"]

    def test_encoding_explicit_with_trailing_newline(self):
        f = _bytes_io("a\nb\n".encode("utf-8"))
        result = list(reverse_iter_lines(f, encoding="utf-8"))
        # trailing newline yields empty string
        assert result == ["b", "a", ""]

    def test_text_file_with_encoding_attr(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        with open(p, "r", encoding="utf-8") as f:
            result = list(reverse_iter_lines(f))
        assert result == ["gamma", "beta", "alpha"]

    def test_binary_file_default_bytes(self, tmp_path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"one\ntwo\nthree\n")
        with open(p, "rb") as f:
            result = list(reverse_iter_lines(f))
        assert result == [b"three", b"two", b"one"]

    def test_returns_iterator(self):
        f = _bytes_io(b"a\n")
        it = reverse_iter_lines(f)
        assert iter(it) is it

    def test_blocksize_larger_than_file(self):
        f = _bytes_io(b"one\ntwo\n")
        assert list(reverse_iter_lines(f, blocksize=100000)) == [b"two", b"one"]

    def test_crlf_lines(self):
        f = _bytes_io(b"a\r\nb\r\n")
        # splitlines handles \r\n
        result = list(reverse_iter_lines(f))
        assert result == [b"b", b"a"]


# ---------------------------------------------------------------------------
# JSONLIterator
# ---------------------------------------------------------------------------

class TestJSONLIterator:

    def test_basic_forward(self):
        f = _bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f)
        assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_basic_forward_no_trailing_newline(self):
        f = _bytes_io(b'{"a": 1}\n{"b": 2}')
        it = JSONLIterator(f)
        assert list(it) == [{"a": 1}, {"b": 2}]

    def test_empty_file(self):
        f = _bytes_io(b"")
        assert list(JSONLIterator(f)) == []

    def test_blank_lines_skipped(self):
        f = _bytes_io(b'{"a": 1}\n\n\n{"b": 2}\n')
        assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}]

    def test_leading_whitespace_stripped(self):
        f = _bytes_io(b'   {"a": 1}\n\t{"b": 2}\n')
        assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}]

    def test_list_values(self):
        f = _bytes_io(b'[1, 2]\n[3]\n"str"\n42\n')
        assert list(JSONLIterator(f)) == [[1, 2], [3], "str", 42]

    def test_ignore_errors_false_raises(self):
        f = _bytes_io(b'{"a": 1}\nnot json\n{"b": 2}\n')
        it = JSONLIterator(f)
        assert next(it) == {"a": 1}
        with pytest.raises(ValueError):
            next(it)

    def test_ignore_errors_true_skips(self):
        f = _bytes_io(b'{"a": 1}\nnot json\n{"b": 2}\n')
        it = JSONLIterator(f, ignore_errors=True)
        assert list(it) == [{"a": 1}, {"b": 2}]

    def test_ignore_errors_true_all_bad(self):
        f = _bytes_io(b"bad\nalso bad\n")
        it = JSONLIterator(f, ignore_errors=True)
        assert list(it) == []

    def test_stop_iteration(self):
        f = _bytes_io(b'{"a": 1}\n')
        it = JSONLIterator(f)
        assert next(it) == {"a": 1}
        with pytest.raises(StopIteration):
            next(it)

    def test_iter_returns_self(self):
        f = _bytes_io(b'{"a": 1}\n')
        it = JSONLIterator(f)
        assert iter(it) is it

    def test_reverse(self):
        f = _bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, reverse=True)
        assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]

    def test_reverse_no_trailing_newline(self):
        f = _bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}')
        it = JSONLIterator(f, reverse=True)
        assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]

    def test_reverse_empty(self):
        f = _bytes_io(b"")
        it = JSONLIterator(f, reverse=True)
        assert list(it) == []

    def test_reverse_with_ignore_errors(self):
        f = _bytes_io(b'{"a": 1}\nbad\n{"b": 2}\n')
        it = JSONLIterator(f, reverse=True, ignore_errors=True)
        assert list(it) == [{"b": 2}, {"a": 1}]

    def test_rel_seek_zero(self):
        f = _bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, rel_seek=0.0)
        assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_rel_seek_one(self):
        f = _bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, rel_seek=1.0)
        assert list(it) == []

    def test_rel_seek_middle_forward(self):
        data = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n{"d": 4}\n'
        f = _bytes_io(data)
        it = JSONLIterator(f, rel_seek=0.5)
        result = list(it)
        # Should start at a line boundary after the middle; must be a suffix
        assert result == [{"c": 3}, {"d": 4}]

    def test_rel_seek_negative(self):
        data = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n{"d": 4}\n'
        f = _bytes_io(data)
        # rel_seek=-0.5 -> 0.5
        it = JSONLIterator(f, rel_seek=-0.5)
        assert list(it) == [{"c": 3}, {"d": 4}]

    def test_rel_seek_negative_one_invalid(self):
        # -1.0 not strictly greater than -1.0
        f = _bytes_io(b'{"a": 1}\n')
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=-1.0)

    def test_rel_seek_too_large_invalid(self):
        f = _bytes_io(b'{"a": 1}\n')
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=1.5)

    def test_rel_seek_too_small_invalid(self):
        f = _bytes_io(b'{"a": 1}\n')
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=-2.0)

    def test_reverse_with_rel_seek_zero(self):
        f = _bytes_io(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        it = JSONLIterator(f, reverse=True, rel_seek=0.0)
        # reverse starts at position 0 -> nothing to read
        assert list(it) == []

    def test_reverse_with_rel_seek_middle(self):
        data = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n{"d": 4}\n'
        f = _bytes_io(data)
        it = JSONLIterator(f, reverse=True, rel_seek=0.5)
        result = list(it)
        # reverse reading from mid-point back to start
        assert result == [{"b": 2}, {"a": 1}]

    def test_cur_byte_pos_forward(self):
        f = _bytes_io(b'{"a": 1}\n{"b": 2}\n')
        it = JSONLIterator(f)
        assert it.cur_byte_pos == 0
        next(it)
        assert it.cur_byte_pos == len(b'{"a": 1}\n')

    def test_cur_byte_pos_reverse(self):
        f = _bytes_io(b'{"a": 1}\n{"b": 2}\n')
        it = JSONLIterator(f, reverse=True)
        # reverse seeks to end during init
        assert it.cur_byte_pos == len(b'{"a": 1}\n{"b": 2}\n')

    def test_text_mode_forward(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        with open(p, "r", encoding="utf-8") as f:
            it = JSONLIterator(f)
            assert list(it) == [{"a": 1}, {"b": 2}]

    def test_text_mode_reverse(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n', encoding="utf-8")
        with open(p, "r", encoding="utf-8") as f:
            it = JSONLIterator(f, reverse=True)
            assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]

    def test_unicode(self):
        data = '{"name": "héllo"}\n{"name": "wörld"}\n'.encode("utf-8")
        f = _bytes_io(data)
        assert list(JSONLIterator(f)) == [
            {"name": "héllo"}, {"name": "wörld"}
        ]

    def test_nested_structures(self):
        obj1 = {"a": [1, 2, {"b": None}], "c": True}
        obj2 = {"x": {"y": [3, 4]}}
        data = (json.dumps(obj1) + "\n" + json.dumps(obj2) + "\n").encode()
        f = _bytes_io(data)
        assert list(JSONLIterator(f)) == [obj1, obj2]

    def test_reverse_ignore_errors_all_bad(self):
        f = _bytes_io(b"bad\nalso bad\n")
        it = JSONLIterator(f, reverse=True, ignore_errors=True)
        assert list(it) == []

    def test_blocksize_attribute(self):
        f = _bytes_io(b'{"a": 1}\n')
        it = JSONLIterator(f)
        assert it._blocksize == 4096
