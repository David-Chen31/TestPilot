import io
import json
import os

import pytest

from boltons.jsonutils import reverse_iter_lines, JSONLIterator, DEFAULT_BLOCKSIZE


# --------------------------------------------------------------------------
# reverse_iter_lines
# --------------------------------------------------------------------------

def test_reverse_iter_lines_returns_lines_reversed():
    data = b"line1\nline2\nline3\n"
    f = io.BytesIO(data)
    lines = list(reverse_iter_lines(f))
    # Note: due to the trailing newline, an extra empty bytes line is yielded
    # first.  Then the lines reverse order, preceded by nothing extra.
    assert lines == [b"", b"line3", b"line2", b"line1"]


def test_reverse_iter_lines_no_trailing_newline():
    data = b"line1\nline2\nline3"
    f = io.BytesIO(data)
    lines = list(reverse_iter_lines(f))
    assert lines == [b"line3", b"line2", b"line1"]


def test_reverse_iter_lines_single_line():
    f = io.BytesIO(b"onlyline")
    assert list(reverse_iter_lines(f)) == [b"onlyline"]


def test_reverse_iter_lines_single_line_with_newline():
    f = io.BytesIO(b"onlyline\n")
    assert list(reverse_iter_lines(f)) == [b"", b"onlyline"]


def test_reverse_iter_lines_empty():
    f = io.BytesIO(b"")
    assert list(reverse_iter_lines(f)) == []


def test_reverse_iter_lines_encoding():
    data = "héllo\nwörld\n".encode("utf-8")
    f = io.BytesIO(data)
    lines = list(reverse_iter_lines(f, encoding="utf-8"))
    assert lines == ["", "wörld", "héllo"]


def test_reverse_iter_lines_encoding_from_file():
    data = "a\nb\nc".encode("utf-8")
    f = io.BytesIO(data)
    # BytesIO has no encoding attribute, so encoding stays None
    assert getattr(f, "encoding", None) is None
    lines = list(reverse_iter_lines(f))
    assert lines == [b"c", b"b", b"a"]


def test_reverse_iter_lines_small_blocksize():
    data = b"one\ntwo\nthree\nfour\n"
    f = io.BytesIO(data)
    lines = list(reverse_iter_lines(f, blocksize=2))
    assert lines == [b"", b"four", b"three", b"two", b"one"]


def test_reverse_iter_lines_preseek_false():
    data = b"aaa\nbbb\nccc\n"
    f = io.BytesIO(data)
    # Move cursor to just after "aaa\n" -> only lines before that position considered
    f.seek(4)
    lines = list(reverse_iter_lines(f, preseek=False))
    assert lines == [b"aaa"]


def test_reverse_iter_lines_preseek_false_at_end():
    data = b"aaa\nbbb\nccc\n"
    f = io.BytesIO(data)
    f.seek(0, os.SEEK_END)
    lines = list(reverse_iter_lines(f, preseek=False))
    assert lines == [b"", b"ccc", b"bbb", b"aaa"]


def test_reverse_iter_lines_text_wrapper():
    sio = io.StringIO("x\ny\nz\n")
    lines = list(reverse_iter_lines(sio, encoding="utf-8"))
    assert lines == ["", "z", "y", "x"]


def test_reverse_iter_lines_text_wrapper_default_encoding():
    sio = io.StringIO("x\ny\nz")
    lines = list(reverse_iter_lines(sio))
    # StringIO has no .buffer; no encoding attribute either, so yields bytes?
    # Actually StringIO has no encoding attr -> encoding None, no .buffer.
    # read returns str, so buff.splitlines gives str lines
    assert lines == ["z", "y", "x"]


# --------------------------------------------------------------------------
# JSONLIterator
# --------------------------------------------------------------------------

def _make_jsonl_bytes(objs):
    return b"".join(json.dumps(o).encode("utf-8") + b"\n" for o in objs)


def test_jsonliterator_forward_basic():
    data = _make_jsonl_bytes([{"a": 1}, {"b": 2}, {"c": 3}])
    f = io.BytesIO(data)
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_jsonliterator_forward_empty():
    f = io.BytesIO(b"")
    it = JSONLIterator(f)
    assert list(it) == []


def test_jsonliterator_forward_skips_blank_lines():
    lines = [b'{"a": 1}', b"", b"   ", b'{"b": 2}']
    data = b"\n".join(lines) + b"\n"
    f = io.BytesIO(data)
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonliterator_forward_error_raises():
    data = b'{"a": 1}\nnot json\n{"c": 3}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f)
    assert next(it) == {"a": 1}
    with pytest.raises(Exception):
        next(it)


def test_jsonliterator_ignore_errors():
    data = b'{"a": 1}\nnot json\n{"c": 3}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f, ignore_errors=True)
    assert list(it) == [{"a": 1}, {"c": 3}]


def test_jsonliterator_reverse_basic():
    data = _make_jsonl_bytes([{"a": 1}, {"b": 2}, {"c": 3}])
    f = io.BytesIO(data)
    it = JSONLIterator(f, reverse=True)
    assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]


def test_jsonliterator_reverse_empty():
    f = io.BytesIO(b"")
    it = JSONLIterator(f, reverse=True)
    assert list(it) == []


def test_jsonliterator_reverse_with_blank_lines():
    data = b'{"a": 1}\n\n{"b": 2}\n\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f, reverse=True)
    assert list(it) == [{"b": 2}, {"a": 1}]


def test_jsonliterator_reverse_ignore_errors():
    data = b'{"a": 1}\nnot json\n{"c": 3}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f, reverse=True, ignore_errors=True)
    assert list(it) == [{"c": 3}, {"a": 1}]


def test_jsonliterator_reverse_error_raises():
    data = b'{"a": 1}\nnot json\n{"c": 3}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f, reverse=True)
    results = []
    with pytest.raises(Exception):
        for obj in it:
            results.append(obj)
    assert results == [{"c": 3}]


def test_rel_seek_invalid():
    f = io.BytesIO(b"")
    with pytest.raises(ValueError):
        JSONLIterator(f, rel_seek=1.5)
    f2 = io.BytesIO(b"")
    with pytest.raises(ValueError):
        JSONLIterator(f2, rel_seek=-1.5)


def test_rel_seek_zero_starts_at_beginning():
    data = _make_jsonl_bytes([{"a": 1}, {"b": 2}, {"c": 3}])
    f = io.BytesIO(data)
    it = JSONLIterator(f, rel_seek=0.0)
    assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_rel_seek_negative_translated():
    # -1.0 not allowed but values in (-1.0, 0) are translated to 1.0 + rel_seek
    data = _make_jsonl_bytes([{"a": 1}, {"b": 2}, {"c": 3}])
    f = io.BytesIO(data)
    it = JSONLIterator(f, rel_seek=-0.5)
    # 1.0 + (-0.5) = 0.5 -> should align to a newline past the middle
    result = list(it)
    assert result  # should start somewhere in the middle/end
    assert len(result) < 3


def test_rel_seek_one_at_end():
    data = _make_jsonl_bytes([{"a": 1}, {"b": 2}])
    f = io.BytesIO(data)
    it = JSONLIterator(f, rel_seek=1.0)
    assert list(it) == []


def test_rel_seek_partial_forward():
    # Build many lines to make positive rel_seek skip some
    objs = [{"i": i} for i in range(100)]
    data = _make_jsonl_bytes(objs)
    f = io.BytesIO(data)
    it = JSONLIterator(f, rel_seek=0.9)
    result = list(it)
    # Should skip roughly the first 90% of the file, aligning to a newline
    assert len(result) < 100
    assert all(isinstance(o, dict) for o in result)
    # Last object must be present since rel_seek < 1.0 ends near EOF
    assert result[-1] == {"i": 99}


def test_cur_byte_pos_forward():
    data = _make_jsonl_bytes([{"a": 1}, {"b": 2}])
    f = io.BytesIO(data)
    it = JSONLIterator(f)
    assert it.cur_byte_pos == 0
    next(it)
    pos_after_first = it.cur_byte_pos
    assert pos_after_first > 0
    assert isinstance(pos_after_first, int)


def test_cur_byte_pos_reverse():
    data = _make_jsonl_bytes([{"a": 1}, {"b": 2}])
    f = io.BytesIO(data)
    it = JSONLIterator(f, reverse=True)
    first = next(it)
    assert first == {"b": 2}


def test_next_alias_and_iter():
    data = b'{"a": 1}\n{"b": 2}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f)
    assert iter(it) is it
    assert next(it) == {"a": 1}
    assert it.__next__() == {"b": 2}
    with pytest.raises(StopIteration):
        next(it)


def test_reverse_property_true():
    data = b'{"a": 1}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f, reverse=1)
    assert it._reverse is True


def test_forward_property_false():
    data = b'{"a": 1}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f)
    assert it._reverse is False


# --------------------------------------------------------------------------
# _align_to_newline (via rel_seek) behavior
# --------------------------------------------------------------------------

def test_rel_seek_midline_alignment():
    # first line is short, second is long; rel_seek landing in middle of an
    # object line should be realigned to next newline.
    data = b'{"a": 1}\n{"longkey": "a very long value that spans"}\n'
    f = io.BytesIO(data)
    # target int(size*0.4) should land mid first or second line
    it = JSONLIterator(f, rel_seek=0.4)
    result = list(it)
    # Any yielded items must be complete dicts (aligned at newline)
    for obj in result:
        assert isinstance(obj, dict)


def test_rel_seek_align_past_eof_sets_end():
    # very tiny file so rel_seek lands mid first line, then align scans to EOF
    data = b'{"a": 12}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f, rel_seek=0.5)
    # Since align reaches EOF without newline after target, iterator gets nothing
    assert list(it) == []


def test_default_blocksize_value():
    assert DEFAULT_BLOCKSIZE == 4096


def test_jsonliterator_cur_byte_pos_after_reverse_full_consumption():
    data = _make_jsonl_bytes([{"a": 1}, {"b": 2}])
    f = io.BytesIO(data)
    it = JSONLIterator(f, reverse=True)
    list(it)
    # After consuming everything, position should be near the beginning
    assert it.cur_byte_pos == 0
