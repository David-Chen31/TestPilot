import io
import json
import os

import pytest

from boltons.jsonutils import reverse_iter_lines, JSONLIterator


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

class TestReverseIterLines:

    def test_basic_bytes_file(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"a\nb\nc\n")
        with open(p, "rb") as f:
            lines = list(reverse_iter_lines(f))
        assert lines == [b"c", b"b", b"a"]

    def test_no_trailing_newline_bytes(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"a\nb\nc")
        with open(p, "rb") as f:
            lines = list(reverse_iter_lines(f))
        # last line "c" yielded first, then b, then a
        assert lines == [b"c", b"b", b"a"]

    def test_text_file_via_io(self):
        f = io.StringIO("x\ny\nz\n")
        lines = list(reverse_iter_lines(f))
        assert lines == ["z", "y", "x"]

    def test_text_no_trailing_newline(self):
        f = io.StringIO("x\ny\nz")
        lines = list(reverse_iter_lines(f))
        assert lines == ["z", "y", "x"]

    def test_blank_lines_are_preserved(self):
        f = io.StringIO("a\n\nb\n")
        lines = list(reverse_iter_lines(f))
        # trailing newline yields an empty line, then b, then empty, then a
        assert lines == ["", "b", "", "a"]

    def test_empty_file(self):
        f = io.StringIO("")
        assert list(reverse_iter_lines(f)) == []

    def test_single_line_with_trailing_newline(self):
        f = io.StringIO("only\n")
        assert list(reverse_iter_lines(f)) == ["only"]

    def test_preseek_false_uses_current_position(self):
        f = io.StringIO("aaa\nbbb\nccc\n")
        # position at end of "bbb\n" chunk: we seek to offset 8 (after bbb\n)
        f.seek(8)
        lines = list(reverse_iter_lines(f, preseek=False))
        assert lines == ["bbb", "aaa"]

    def test_small_blocksize(self):
        content = "".join("%d\n" % i for i in range(20))
        f = io.StringIO(content)
        lines = list(reverse_iter_lines(f, blocksize=2))
        assert lines == [str(i) for i in range(19, -1, -1)]

    def test_explicit_encoding(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes("hé\nlo\n".encode("utf-8"))
        with open(p, "rb") as f:
            lines = list(reverse_iter_lines(f, encoding="utf-8"))
        assert lines == ["lo", "hé"]

    def test_bytes_file_without_encoding(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"one\ntwo\n")
        with open(p, "rb") as f:
            lines = list(reverse_iter_lines(f))
        assert all(isinstance(l, bytes) for l in lines)
        assert lines == [b"two", b"one"]

    def test_text_file_auto_encoding_detection(self, tmp_path):
        # Open in text mode; encoding attribute present -> strings out
        p = tmp_path / "f.txt"
        p.write_text("a\nb\n", encoding="utf-8")
        with open(p, "r", encoding="utf-8") as f:
            lines = list(reverse_iter_lines(f))
        assert lines == ["b", "a"]

    def test_iterator_is_lazy(self):
        f = io.StringIO("a\nb\n")
        it = reverse_iter_lines(f)
        assert iter(it) is it
        assert next(it) == "b"
        assert next(it) == "a"
        with pytest.raises(StopIteration):
            next(it)


# ---------------------------------------------------------------------------
# JSONLIterator - forward
# ---------------------------------------------------------------------------

def _write_jsonl_bytes(tmp_path, objs, name="data.jsonl"):
    p = tmp_path / name
    text = "".join(json.dumps(o) + "\n" for o in objs)
    p.write_bytes(text.encode("utf-8"))
    return p


class TestJSONLIteratorForward:

    def test_basic_forward_bytes(self, tmp_path):
        p = _write_jsonl_bytes(tmp_path, [{"a": 1}, {"b": 2}, [1, 2, 3]])
        with open(p, "rb") as f:
            items = list(JSONLIterator(f))
        assert items == [{"a": 1}, {"b": 2}, [1, 2, 3]]

    def test_basic_forward_text(self):
        f = io.StringIO('{"a": 1}\n{"b": 2}\n')
        assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}]

    def test_iter_returns_self(self):
        f = io.StringIO("1\n")
        it = JSONLIterator(f)
        assert iter(it) is it

    def test_next_alias(self):
        f = io.StringIO("1\n2\n")
        it = JSONLIterator(f)
        assert it.next() == 1
        assert it.__next__() == 2
        with pytest.raises(StopIteration):
            next(it)

    def test_blank_lines_skipped(self):
        f = io.StringIO("\n\n1\n\n2\n\n")
        assert list(JSONLIterator(f)) == [1, 2]

    def test_whitespace_lines_skipped(self):
        f = io.StringIO("   \n1\n\t\n2\n")
        assert list(JSONLIterator(f)) == [1, 2]

    def test_leading_whitespace_stripped(self):
        f = io.StringIO("   1\n   2\n")
        assert list(JSONLIterator(f)) == [1, 2]

    def test_invalid_json_raises(self):
        f = io.StringIO("not json\n1\n")
        with pytest.raises(Exception):
            list(JSONLIterator(f))

    def test_invalid_json_ignore_errors(self):
        f = io.StringIO("not json\n1\n{bad\n2\n")
        assert list(JSONLIterator(f, ignore_errors=True)) == [1, 2]

    def test_empty_file(self):
        f = io.StringIO("")
        assert list(JSONLIterator(f)) == []

    def test_cur_byte_pos_advances(self, tmp_path):
        p = _write_jsonl_bytes(tmp_path, [{"a": 1}, {"b": 2}])
        with open(p, "rb") as f:
            it = JSONLIterator(f)
            assert it.cur_byte_pos == 0
            next(it)
            pos_after_first = it.cur_byte_pos
            assert pos_after_first > 0
            next(it)
            assert it.cur_byte_pos >= pos_after_first

    def test_reverse_false_explicit(self):
        f = io.StringIO("1\n2\n3\n")
        assert list(JSONLIterator(f, reverse=False)) == [1, 2, 3]


# ---------------------------------------------------------------------------
# JSONLIterator - reverse
# ---------------------------------------------------------------------------

class TestJSONLIteratorReverse:

    def test_basic_reverse_bytes(self, tmp_path):
        p = _write_jsonl_bytes(tmp_path, [{"a": 1}, {"a": 2}, {"a": 3}])
        with open(p, "rb") as f:
            items = list(JSONLIterator(f, reverse=True))
        assert items == [{"a": 3}, {"a": 2}, {"a": 1}]

    def test_basic_reverse_text(self):
        f = io.StringIO("1\n2\n3\n")
        assert list(JSONLIterator(f, reverse=True)) == [3, 2, 1]

    def test_reverse_no_trailing_newline(self):
        f = io.StringIO("1\n2\n3")
        assert list(JSONLIterator(f, reverse=True)) == [3, 2, 1]

    def test_reverse_empty(self):
        f = io.StringIO("")
        assert list(JSONLIterator(f, reverse=True)) == []

    def test_reverse_ignore_errors(self):
        f = io.StringIO("1\nnope\n2\n")
        assert list(JSONLIterator(f, reverse=True, ignore_errors=True)) == [2, 1]

    def test_reverse_invalid_raises(self):
        f = io.StringIO("1\nnope\n2\n")
        with pytest.raises(Exception):
            list(JSONLIterator(f, reverse=True))

    def test_reverse_single_line(self):
        f = io.StringIO("42\n")
        assert list(JSONLIterator(f, reverse=True)) == [42]


# ---------------------------------------------------------------------------
# rel_seek
# ---------------------------------------------------------------------------

class TestRelSeek:

    def test_rel_seek_zero_start(self, tmp_path):
        p = _write_jsonl_bytes(tmp_path, [1, 2, 3, 4, 5])
        with open(p, "rb") as f:
            items = list(JSONLIterator(f, rel_seek=0.0))
        assert items == [1, 2, 3, 4, 5]

    def test_rel_seek_one_is_end_forward(self, tmp_path):
        p = _write_jsonl_bytes(tmp_path, [1, 2, 3])
        with open(p, "rb") as f:
            items = list(JSONLIterator(f, rel_seek=1.0))
        # at very end, forward iteration yields nothing
        assert items == []

    def test_rel_seek_negative(self, tmp_path):
        p = _write_jsonl_bytes(tmp_path, [1, 2, 3])
        with open(p, "rb") as f:
            items = list(JSONLIterator(f, rel_seek=-1.0 + 0.0))
        # -1.0 -> 0.0
        assert items == [1, 2, 3]

    def test_rel_seek_invalid_raises(self, tmp_path):
        p = _write_jsonl_bytes(tmp_path, [1])
        with open(p, "rb") as f:
            with pytest.raises(ValueError):
                JSONLIterator(f, rel_seek=1.5)
        with open(p, "rb") as f:
            with pytest.raises(ValueError):
                JSONLIterator(f, rel_seek=-1.0)

    def test_rel_seek_reverse_default_is_end(self, tmp_path):
        p = _write_jsonl_bytes(tmp_path, [1, 2, 3, 4])
        with open(p, "rb") as f:
            # reverse with no rel_seek => starts at end
            items = list(JSONLIterator(f, reverse=True))
        assert items == [4, 3, 2, 1]

    def test_rel_seek_to_middle_aligns_to_newline(self, tmp_path):
        lines = ["%d\n" % i for i in range(10)]
        p = tmp_path / "data.jsonl"
        p.write_text("".join(lines), encoding="utf-8")
        total = sum(len(l) for l in lines)
        with open(p, "rb") as f:
            # choose a rel_seek that likely lands mid-line
            items = list(JSONLIterator(f, rel_seek=0.5))
        # should be some suffix of the original sequence, in order
        assert items == list(range(10 - len(items), 10))

    def test_no_rel_seek_starts_at_zero(self, tmp_path):
        p = _write_jsonl_bytes(tmp_path, [7, 8, 9])
        with open(p, "rb") as f:
            it = JSONLIterator(f)
            assert it.cur_byte_pos == 0
            assert next(it) == 7


# ---------------------------------------------------------------------------
# _align_to_newline / edge cases at EOF
# ---------------------------------------------------------------------------

class TestAlignToNewline:

    def test_align_no_newline_goes_to_eof(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text("1\n2\n3", encoding="utf-8")  # no trailing newline
        with open(p, "rb") as f:
            f.seek(3)  # inside "2\n..." region, mid content
            it = JSONLIterator.__new__(JSONLIterator)
            it._file_obj = f
            it._blocksize = 4096
            it._align_to_newline()
            f.tell() == len("1\n2\n3")

    def test_align_finds_newline(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text("aaa\nbbb\nccc\n", encoding="utf-8")
        with open(p, "rb") as f:
            f.seek(1)  # mid first line
            it = JSONLIterator.__new__(JSONLIterator)
            it._file_obj = f
            it._blocksize = 4096
            it._align_to_newline()
            # should point right after the first newline
            assert f.tell() == 4


# ---------------------------------------------------------------------------
# TextWrapper / buffer handling
# ---------------------------------------------------------------------------

class TestBufferHandling:

    def test_text_wrapper_uses_buffer(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"1\n2\n3\n")
        with open(p, "r", encoding="utf-8") as f:
            items = list(JSONLIterator(f, reverse=True))
        assert items == [3, 2, 1]

    def test_stringio_reverse(self):
        f = io.StringIO("a\nb\nc\n")
        assert list(reverse_iter_lines(f)) == ["c", "b", "a"]
