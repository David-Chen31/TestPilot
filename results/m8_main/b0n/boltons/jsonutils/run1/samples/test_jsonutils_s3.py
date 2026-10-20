import io
import os
import json

import pytest

from boltons.jsonutils import reverse_iter_lines, JSONLIterator, DEFAULT_BLOCKSIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bytes_file(content, encoding=None):
    return io.BytesIO(content)


def _make_text_file(content, encoding='utf-8'):
    return io.StringIO(content)


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

def test_reverse_iter_lines_bytes():
    f = io.BytesIO(b"line1\nline2\nline3\n")
    lines = list(reverse_iter_lines(f))
    assert lines == [b"line3", b"line2", b"line1"]


def test_reverse_iter_lines_no_trailing_newline_bytes():
    f = io.BytesIO(b"line1\nline2\nline3")
    lines = list(reverse_iter_lines(f))
    assert lines == [b"line3", b"line2", b"line1"]


def test_reverse_iter_lines_text_stream():
    f = io.StringIO("line1\nline2\nline3\n")
    lines = list(reverse_iter_lines(f))
    assert lines == ["line3", "line2", "line1"]


def test_reverse_iter_lines_text_explicit_encoding():
    f = io.BytesIO("é\nü\n".encode("utf-8"))
    lines = list(reverse_iter_lines(f, encoding="utf-8"))
    assert lines == ["ü", "é"]


def test_reverse_iter_lines_empty_file():
    f = io.BytesIO(b"")
    assert list(reverse_iter_lines(f)) == []


def test_reverse_iter_lines_empty_text_file():
    f = io.StringIO("")
    assert list(reverse_iter_lines(f)) == []


def test_reverse_iter_lines_single_line_bytes():
    f = io.BytesIO(b"only")
    assert list(reverse_iter_lines(f)) == [b"only"]


def test_reverse_iter_lines_single_line_with_newline():
    f = io.BytesIO(b"only\n")
    assert list(reverse_iter_lines(f)) == [b"only"]


def test_reverse_iter_lines_small_blocksize():
    f = io.BytesIO(b"a\nbb\nccc\ndddd\n")
    lines = list(reverse_iter_lines(f, blocksize=2))
    assert lines == [b"dddd", b"ccc", b"bb", b"a"]


def test_reverse_iter_lines_blocksize_one():
    f = io.BytesIO(b"a\nb\nc\n")
    lines = list(reverse_iter_lines(f, blocksize=1))
    assert lines == [b"c", b"b", b"a"]


def test_reverse_iter_lines_preseek_false_from_middle():
    f = io.BytesIO(b"line1\nline2\nline3\nline4\n")
    # Position cursor to just after "line2\n" (byte 12)
    f.seek(12)
    lines = list(reverse_iter_lines(f, preseek=False))
    assert lines == [b"line2", b"line1"]


def test_reverse_iter_lines_preseek_false_at_end():
    f = io.BytesIO(b"a\nb\nc")
    f.seek(0, os.SEEK_END)
    lines = list(reverse_iter_lines(f, preseek=False))
    assert lines == [b"c", b"b", b"a"]


def test_reverse_iter_lines_text_stream_preseek_false():
    f = io.StringIO("x\ny\nz\n")
    f.seek(4)  # just after "y\n"
    lines = list(reverse_iter_lines(f, preseek=False))
    assert lines == ["y", "x"]


def test_reverse_iter_lines_blank_lines():
    f = io.BytesIO(b"a\n\nb\n\n")
    lines = list(reverse_iter_lines(f))
    assert lines == [b"b", b"a"]


def test_reverse_iter_lines_only_newlines():
    f = io.BytesIO(b"\n\n\n")
    lines = list(reverse_iter_lines(f))
    assert lines == []


def test_reverse_iter_lines_returns_iterator():
    f = io.BytesIO(b"a\n")
    it = reverse_iter_lines(f)
    # It's a generator
    assert hasattr(it, "__next__")
    next(it)
    it.close()


def test_reverse_iter_lines_blocksize_default():
    assert DEFAULT_BLOCKSIZE == 4096


def test_reverse_iter_lines_windows_newlines():
    f = io.BytesIO(b"a\r\nb\r\nc\r\n")
    lines = list(reverse_iter_lines(f))
    assert lines == [b"c", b"b", b"a"]


def test_reverse_iter_lines_large_content_multiple_blocks():
    data = b"\n".join(b"item%d" % i for i in range(1000)) + b"\n"
    f = io.BytesIO(data)
    lines = list(reverse_iter_lines(f, blocksize=64))
    expected = [b"item%d" % i for i in range(999, -1, -1)]
    assert lines == expected


# ---------------------------------------------------------------------------
# JSONLIterator - forward iteration
# ---------------------------------------------------------------------------

def test_jsonl_iterator_forward_bytes():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_jsonl_iterator_forward_text():
    f = io.StringIO('{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_forward_no_trailing_newline():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}')
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_empty_file():
    f = io.BytesIO(b"")
    it = JSONLIterator(f)
    assert list(it) == []


def test_jsonl_iterator_blank_lines_skipped():
    f = io.BytesIO(b'{"a": 1}\n\n   \n{"b": 2}\n')
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_is_iter_returns_self():
    f = io.BytesIO(b'{"a": 1}\n')
    it = JSONLIterator(f)
    assert iter(it) is it


def test_jsonl_iterator_next_method():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f)
    assert it.next() == {"a": 1}
    assert next(it) == {"b": 2}
    with pytest.raises(StopIteration):
        next(it)


def test_jsonl_iterator_leading_whitespace_stripped():
    f = io.BytesIO(b'   {"a": 1}\n\t{"b": 2}\n')
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_parses_various_types():
    f = io.BytesIO(b'1\n"two"\n[3, 4]\ntrue\nnull\n')
    it = JSONLIterator(f)
    assert list(it) == [1, "two", [3, 4], True, None]


# ---------------------------------------------------------------------------
# JSONLIterator - error handling
# ---------------------------------------------------------------------------

def test_jsonl_iterator_invalid_line_raises():
    f = io.BytesIO(b'{"a": 1}\nnot json\n{"c": 3}\n')
    it = JSONLIterator(f)
    assert next(it) == {"a": 1}
    with pytest.raises(json.JSONDecodeError):
        next(it)


def test_jsonl_iterator_ignore_errors():
    f = io.BytesIO(b'{"a": 1}\nnot json\n{"c": 3}\n')
    it = JSONLIterator(f, ignore_errors=True)
    assert list(it) == [{"a": 1}, {"c": 3}]


def test_jsonl_iterator_ignore_errors_all_bad():
    f = io.BytesIO(b'bad\nalso bad\n')
    it = JSONLIterator(f, ignore_errors=True)
    assert list(it) == []


# ---------------------------------------------------------------------------
# JSONLIterator - reverse iteration
# ---------------------------------------------------------------------------

def test_jsonl_iterator_reverse_bytes():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
    it = JSONLIterator(f, reverse=True)
    assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]


def test_jsonl_iterator_reverse_text():
    f = io.StringIO('{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f, reverse=True)
    assert list(it) == [{"b": 2}, {"a": 1}]


def test_jsonl_iterator_reverse_no_trailing_newline():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}')
    it = JSONLIterator(f, reverse=True)
    assert list(it) == [{"b": 2}, {"a": 1}]


def test_jsonl_iterator_reverse_empty():
    f = io.BytesIO(b"")
    it = JSONLIterator(f, reverse=True)
    assert list(it) == []


def test_jsonl_iterator_reverse_with_errors():
    f = io.BytesIO(b'{"a": 1}\nbad\n{"c": 3}\n')
    it = JSONLIterator(f, reverse=True, ignore_errors=True)
    assert list(it) == [{"c": 3}, {"a": 1}]


def test_jsonl_iterator_reverse_does_not_raise_on_good_lines():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
    it = JSONLIterator(f, reverse=True)
    assert it.next() == {"c": 3}
    assert it.next() == {"b": 2}
    assert it.next() == {"a": 1}
    with pytest.raises(StopIteration):
        next(it)


# ---------------------------------------------------------------------------
# JSONLIterator - rel_seek
# ---------------------------------------------------------------------------

def test_jsonl_iterator_rel_seek_zero_starts_at_beginning():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f, rel_seek=0.0)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_rel_seek_one_starts_at_end():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f, rel_seek=1.0)
    assert list(it) == []


def test_jsonl_iterator_rel_seek_invalid_raises():
    f = io.BytesIO(b'{"a": 1}\n')
    with pytest.raises(ValueError):
        JSONLIterator(f, rel_seek=1.5)
    with pytest.raises(ValueError):
        JSONLIterator(f, rel_seek=-1.0)


def test_jsonl_iterator_rel_seek_negative_transparent():
    # -0.5 -> 0.5 relative
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n{"d": 4}\n'
    f = io.BytesIO(content)
    it = JSONLIterator(f, rel_seek=-0.5)
    results = list(it)
    all_objects = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
    # Everything yielded must be a subset of the valid objects, in order
    assert results == all_objects[len(all_objects) - len(results):]


def test_jsonl_iterator_rel_seek_mid_file():
    lines = [b'{"n": %d}\n' % i for i in range(100)]
    content = b"".join(lines)
    f = io.BytesIO(content)
    it = JSONLIterator(f, rel_seek=0.5)
    results = list(it)
    # Should start somewhere in the middle, ending at last object
    assert results[-1] == {"n": 99}
    # All yielded objects must be valid JSON dicts from the set
    valid = [{"n": i} for i in range(100)]
    assert results == valid[len(valid) - len(results):]


def test_jsonl_iterator_rel_seek_reverse_unset_end():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f, reverse=True)
    # reverse with no rel_seek -> rel_seek defaults to 1.0 (end)
    assert it._rel_seek == 1.0
    assert list(it) == [{"b": 2}, {"a": 1}]


def test_jsonl_iterator_rel_seek_zero_reverse():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f, reverse=True, rel_seek=0.0)
    assert list(it) == []


def test_jsonl_iterator_rel_seek_stored_value():
    f = io.BytesIO(b'{"a": 1}\n')
    it = JSONLIterator(f, rel_seek=0.5)
    assert it._rel_seek == 0.5


def test_jsonl_iterator_rel_seek_negative_stored_value():
    f = io.BytesIO(b'{"a": 1}\n')
    it = JSONLIterator(f, rel_seek=-0.25)
    assert it._rel_seek == 0.75


# ---------------------------------------------------------------------------
# JSONLIterator - cur_byte_pos
# ---------------------------------------------------------------------------

def test_jsonl_iterator_cur_byte_pos_forward():
    content = b'{"a": 1}\n{"b": 2}\n'
    f = io.BytesIO(content)
    it = JSONLIterator(f)
    assert it.cur_byte_pos == 0
    next(it)
    # After reading first line, position should be at end of first line
    assert it.cur_byte_pos > 0


def test_jsonl_iterator_cur_byte_pos_after_full_read():
    content = b'{"a": 1}\n{"b": 2}\n'
    f = io.BytesIO(content)
    it = JSONLIterator(f)
    list(it)
    assert it.cur_byte_pos == len(content)


# ---------------------------------------------------------------------------
# JSONLIterator - attributes
# ---------------------------------------------------------------------------

def test_jsonl_iterator_ignore_errors_attribute():
    f = io.BytesIO(b"")
    it = JSONLIterator(f, ignore_errors=True)
    assert it.ignore_errors is True
    it2 = JSONLIterator(f, ignore_errors=False)
    assert it2.ignore_errors is False


def test_jsonl_iterator_reverse_bool_coerced():
    f = io.BytesIO(b'{"a": 1}\n')
    it = JSONLIterator(f, reverse=0)
    assert it._reverse is False
    it2 = JSONLIterator(f, reverse=1)
    assert it2._reverse is True
</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</invoke>
