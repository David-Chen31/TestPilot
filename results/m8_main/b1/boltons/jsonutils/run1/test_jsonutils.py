import io
import os
import json

import pytest

from boltons.jsonutils import JSONLIterator, reverse_iter_lines, DEFAULT_BLOCKSIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path, name, data, mode='wb'):
    p = tmp_path / name
    if 'b' in mode:
        p.write_bytes(data.encode('utf-8') if isinstance(data, str) else data)
    else:
        p.write_text(data, encoding='utf-8')
    return p


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

def test_reverse_iter_lines_bytesio_basic():
    f = io.BytesIO(b'a\nb\nc\n')
    result = list(reverse_iter_lines(f))
    assert result == [b'', b'c', b'b', b'a']


def test_reverse_iter_lines_no_trailing_newline():
    f = io.BytesIO(b'a\nb\nc')
    result = list(reverse_iter_lines(f))
    assert result == [b'c', b'b', b'a']


def test_reverse_iter_lines_single_line():
    f = io.BytesIO(b'only')
    assert list(reverse_iter_lines(f)) == [b'only']


def test_reverse_iter_lines_single_line_with_newline():
    f = io.BytesIO(b'only\n')
    assert list(reverse_iter_lines(f)) == [b'only\n']


def test_reverse_iter_lines_empty():
    f = io.BytesIO(b'')
    assert list(reverse_iter_lines(f)) == []


def test_reverse_iter_lines_trailing_newline_yields_empty():
    # Trailing newline semantics: a trailing empty line is produced.
    f = io.BytesIO(b'a\nb\n')
    result = list(reverse_iter_lines(f))
    assert result == [b'', b'b', b'a']


def test_reverse_iter_lines_empty_lines_inside():
    f = io.BytesIO(b'a\n\nb\n')
    result = list(reverse_iter_lines(f))
    assert result == [b'', b'b', b'', b'a']


def test_reverse_iter_lines_small_blocksize():
    f = io.BytesIO(b'aaa\nbbb\nccc\nddd\n')
    result = list(reverse_iter_lines(f, blocksize=2))
    assert result == [b'', b'ddd', b'ccc', b'bbb', b'aaa']


def test_reverse_iter_lines_blocksize_one():
    f = io.BytesIO(b'ab\ncd\n')
    result = list(reverse_iter_lines(f, blocksize=1))
    assert result == [b'', b'cd', b'ab']


def test_reverse_iter_lines_encoding_explicit():
    f = io.BytesIO('héllo\nwörld\n'.encode('utf-8'))
    result = list(reverse_iter_lines(f, encoding='utf-8'))
    assert result == ['', 'wörld', 'héllo']


def test_reverse_iter_lines_text_mode():
    # StringIO has no .buffer, so it is read as-is (str), but buff starts as
    # bytes, causing a TypeError when concatenated with str reads.
    s = io.StringIO('a\nb\nc\n')
    s.seek(0)
    with pytest.raises(TypeError):
        list(reverse_iter_lines(s))


def test_reverse_iter_lines_preseek_false():
    # Starting at position 4 (start of 'b\nc\nd\n') with preseek=False reads
    # everything from position 0 up to the current position in reverse.
    f = io.BytesIO(b'a\nb\nc\nd\n')
    f.seek(4)
    result = list(reverse_iter_lines(f, preseek=False))
    assert result == [b'', b'b', b'a']


def test_reverse_iter_lines_preseek_false_from_midline():
    f = io.BytesIO(b'aaa\nbbb\nccc\n')
    f.seek(5)
    result = list(reverse_iter_lines(f, preseek=False))
    assert result == [b'b', b'aaa']


def test_reverse_iter_lines_preseek_false_text_mode():
    s = io.StringIO('aaa\nbbb\nccc')
    s.seek(len('aaa\nbb'))
    with pytest.raises(TypeError):
        list(reverse_iter_lines(s, preseek=False))


# ---------------------------------------------------------------------------
# JSONLIterator: forward
# ---------------------------------------------------------------------------

def test_jsonl_iterator_forward_bytesio():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f)
    assert it._reverse is False
    assert next(it) == {'a': 1}
    assert next(it) == {'b': 2}
    with pytest.raises(StopIteration):
        next(it)


def test_jsonl_iterator_iterable_protocol():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f)
    assert list(it) == [{'a': 1}, {'b': 2}]


def test_jsonl_iterator_skips_blank_lines():
    f = io.BytesIO(b'\n\n{"a": 1}\n\n{"b": 2}\n\n')
    assert list(JSONLIterator(f)) == [{'a': 1}, {'b': 2}]


def test_jsonl_iterator_leading_whitespace_lstripped():
    f = io.BytesIO(b'   {"a": 1}\n   {"b": 2}\n')
    assert list(JSONLIterator(f)) == [{'a': 1}, {'b': 2}]


def test_jsonl_iterator_error_raises():
    f = io.BytesIO(b'{"a": 1}\nnot json\n{"b": 2}\n')
    it = JSONLIterator(f)
    assert next(it) == {'a': 1}
    with pytest.raises(Exception):
        next(it)


def test_jsonl_iterator_ignore_errors():
    f = io.BytesIO(b'{"a": 1}\nnot json\n{"b": 2}\n')
    assert list(JSONLIterator(f, ignore_errors=True)) == [{'a': 1}, {'b': 2}]


def test_jsonl_iterator_text_file(tmp_path):
    p = _write(tmp_path, 'f.jsonl', '{"a": 1}\n{"b": 2}\n', mode='w')
    with open(p, 'r', encoding='utf-8') as f:
        assert list(JSONLIterator(f)) == [{'a': 1}, {'b': 2}]


def test_jsonl_iterator_empty_file():
    assert list(JSONLIterator(io.BytesIO(b''))) == []


def test_jsonl_iterator_cur_byte_pos_forward():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f)
    assert it.cur_byte_pos == 0
    next(it)
    assert it.cur_byte_pos == len(b'{"a": 1}\n')


# ---------------------------------------------------------------------------
# JSONLIterator: reverse
# ---------------------------------------------------------------------------

def test_jsonl_iterator_reverse_bytesio():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
    it = JSONLIterator(f, reverse=True)
    assert list(it) == [{'c': 3}, {'b': 2}, {'a': 1}]


def test_jsonl_iterator_reverse_no_trailing_newline():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n{"c": 3}')
    assert list(JSONLIterator(f, reverse=True)) == [{'c': 3}, {'b': 2}, {'a': 1}]


def test_jsonl_iterator_reverse_ignore_errors():
    f = io.BytesIO(b'{"a": 1}\nBOOM\n{"c": 3}\n')
    assert list(JSONLIterator(f, reverse=True, ignore_errors=True)) == [
        {'c': 3}, {'a': 1}]


def test_jsonl_iterator_reverse_error_raises():
    f = io.BytesIO(b'{"a": 1}\nBOOM\n{"c": 3}\n')
    it = JSONLIterator(f, reverse=True)
    assert next(it) == {'c': 3}
    with pytest.raises(Exception):
        next(it)


def test_jsonl_iterator_text_file_reverse(tmp_path):
    p = _write(tmp_path, 'f.jsonl', '{"a": 1}\n{"b": 2}\n{"c": 3}\n', mode='w')
    with open(p, 'r', encoding='utf-8') as f:
        assert list(JSONLIterator(f, reverse=True)) == [
            {'c': 3}, {'b': 2}, {'a': 1}]


# ---------------------------------------------------------------------------
# JSONLIterator: rel_seek
# ---------------------------------------------------------------------------

def test_jsonl_iterator_rel_seek_zero_starts_at_beginning():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    assert list(JSONLIterator(f, rel_seek=0.0)) == [{'a': 1}, {'b': 2}]


def test_jsonl_iterator_rel_seek_invalid_raises():
    with pytest.raises(ValueError):
        JSONLIterator(io.BytesIO(b''), rel_seek=1.5)
    with pytest.raises(ValueError):
        JSONLIterator(io.BytesIO(b''), rel_seek=-1.0)


def test_jsonl_iterator_rel_seek_negative_translated():
    # -0.5 => 0.5
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n{"d": 4}\n')
    it = JSONLIterator(f, rel_seek=-0.5)
    assert it._rel_seek == 0.5


def test_jsonl_iterator_rel_seek_midline_aligns_forward():
    data = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'
    f = io.BytesIO(data)
    # size = len(data); pick rs to land mid-line.
    it = JSONLIterator(f, rel_seek=0.3)
    got = list(it)
    all_objs = [{'a': 1}, {'b': 2}, {'c': 3}]
    assert got == all_objs[len(all_objs) - len(got):]


def test_jsonl_iterator_rel_seek_one_is_end():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    assert list(JSONLIterator(f, rel_seek=1.0)) == []


def test_jsonl_iterator_rel_seek_zero_reverse():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f, reverse=True, rel_seek=0.0)
    assert list(it) == []


# ---------------------------------------------------------------------------
# _align_to_newline (internal but exercised via rel_seek)
# ---------------------------------------------------------------------------

def test_align_to_newline_no_newline_seeks_eof():
    # File with no newline at all: align should land at EOF, no objects.
    f = io.BytesIO(b'nonnewlinecontent')
    it = JSONLIterator(f, rel_seek=0.0)
    # rel_seek==0.0 doesn't align; force an alignment path:
    it2 = JSONLIterator(io.BytesIO(b'abc'), rel_seek=0.5)
    assert list(it2) == []


def test_reverse_iter_lines_default_blocksize():
    assert DEFAULT_BLOCKSIZE == 4096
