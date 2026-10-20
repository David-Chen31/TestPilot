import io
import json
import os

import pytest

from boltons.jsonutils import JSONLIterator, reverse_iter_lines, DEFAULT_BLOCKSIZE


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

def test_reverse_iter_lines_bytes_basic():
    data = b"line1\nline2\nline3\n"
    f = io.BytesIO(data)
    result = list(reverse_iter_lines(f))
    assert result == [b"line3", b"line2", b"line1"]


def test_reverse_iter_lines_bytes_no_trailing_newline():
    data = b"alpha\nbeta\ngamma"
    f = io.BytesIO(data)
    result = list(reverse_iter_lines(f))
    assert result == [b"gamma", b"beta", b"alpha"]


def test_reverse_iter_lines_text_basic():
    f = io.StringIO("a\nb\nc\n")
    result = list(reverse_iter_lines(f))
    assert result == ["c", "b", "a"]


def test_reverse_iter_lines_text_no_trailing():
    f = io.StringIO("one\ntwo\nthree")
    result = list(reverse_iter_lines(f))
    assert result == ["three", "two", "one"]


def test_reverse_iter_lines_empty_file():
    f = io.BytesIO(b"")
    assert list(reverse_iter_lines(f)) == []


def test_reverse_iter_lines_empty_text_file():
    f = io.StringIO("")
    assert list(reverse_iter_lines(f)) == []


def test_reverse_iter_lines_trailing_newline_yields_empty_first():
    # A file ending with a newline means the first logical line (from the end)
    # is empty, represented as b'' / ''.
    f = io.BytesIO(b"x\ny\n")
    result = list(reverse_iter_lines(f))
    assert result == [b"", b"y", b"x"]


def test_reverse_iter_lines_trailing_newline_text():
    f = io.StringIO("x\ny\n")
    result = list(reverse_iter_lines(f))
    assert result == ["", "y", "x"]


def test_reverse_iter_lines_blocksize_small():
    data = b"".join(b"line%d\n" % i for i in range(50))
    f = io.BytesIO(data)
    result = list(reverse_iter_lines(f, blocksize=8))
    expected = [b"line%d" % i for i in range(49, -1, -1)]
    assert result == expected


def test_reverse_iter_lines_preseek_false():
    # Without preseek, iteration starts at the current cursor position.
    f = io.BytesIO(b"a\nb\nc\nd\n")
    f.seek(0)  # position at start -> nothing to read
    result = list(reverse_iter_lines(f, preseek=False))
    assert result == []


def test_reverse_iter_lines_preseek_false_mid():
    data = b"a\nb\nc\nd\n"
    f = io.BytesIO(data)
    f.seek(4)  # after "a\nb\n"
    result = list(reverse_iter_lines(f, preseek=False))
    assert result == [b"b", b"a"]


def test_reverse_iter_lines_encoding():
    f = io.BytesIO("héllo\nwörld\n".encode("utf-8"))
    result = list(reverse_iter_lines(f, encoding="utf-8"))
    assert result == ["wörld", "héllo"]


def test_reverse_iter_lines_explicit_blocksize_constant():
    assert isinstance(DEFAULT_BLOCKSIZE, int)
    f = io.BytesIO(b"a\nb\n")
    result = list(reverse_iter_lines(f, blocksize=DEFAULT_BLOCKSIZE))
    assert result == [b"b", b"a"]


def test_reverse_iter_lines_single_line():
    f = io.BytesIO(b"only\n")
    assert list(reverse_iter_lines(f)) == [b"only"]


def test_reverse_iter_lines_single_line_no_newline():
    f = io.BytesIO(b"only")
    assert list(reverse_iter_lines(f)) == [b"only"]


# ---------------------------------------------------------------------------
# JSONLIterator
# ---------------------------------------------------------------------------

def _bytes_obj(objs):
    return b"".join((json.dumps(o) + "\n").encode("utf-8") for o in objs)


def _text_obj(objs):
    return "".join(json.dumps(o) + "\n" for o in objs)


def test_jsonl_iterator_forward_bytes():
    objs = [{"a": 1}, [1, 2], "str", 42, None]
    f = io.BytesIO(_bytes_obj(objs))
    it = JSONLIterator(f)
    assert list(it) == objs


def test_jsonl_iterator_forward_text():
    objs = [{"a": 1}, {"b": 2}]
    f = io.StringIO(_text_obj(objs))
    it = JSONLIterator(f)
    assert list(it) == objs


def test_jsonl_iterator_reverse_bytes():
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    f = io.BytesIO(_bytes_obj(objs))
    it = JSONLIterator(f, reverse=True)
    assert list(it) == list(reversed(objs))


def test_jsonl_iterator_reverse_text():
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    f = io.StringIO(_text_obj(objs))
    it = JSONLIterator(f, reverse=True)
    assert list(it) == list(reversed(objs))


def test_jsonl_iterator_ignore_errors():
    data = b'{"a": 1}\nnot json\n{"b": 2}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f, ignore_errors=True)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_ignore_errors_reverse():
    data = b'{"a": 1}\nnot json\n{"b": 2}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f, ignore_errors=True, reverse=True)
    assert list(it) == [{"b": 2}, {"a": 1}]


def test_jsonl_iterator_raises_on_error():
    data = b'{"a": 1}\nnot json\n{"b": 2}\n'
    f = io.BytesIO(data)
    it = JSONLIterator(f)
    values = []
    with pytest.raises(json.JSONDecodeError):
        for obj in it:
            values.append(obj)
    assert values == [{"a": 1}]


def test_jsonl_iterator_skips_blank_lines():
    data = b'{"a": 1}\n\n\n{"b": 2}\n'
    f = io.BytesIO(data)
    assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_lstrip_whitespace():
    data = b'   {"a": 1}\n\t  {"b": 2}\n'
    f = io.BytesIO(data)
    assert list(JSONLIterator(f)) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_rel_seek_zero():
    objs = [{"n": i} for i in range(5)]
    f = io.BytesIO(_bytes_obj(objs))
    it = JSONLIterator(f, rel_seek=0.0)
    assert list(it) == objs


def test_jsonl_iterator_rel_seek_one_forward():
    objs = [{"n": i} for i in range(5)]
    f = io.BytesIO(_bytes_obj(objs))
    it = JSONLIterator(f, rel_seek=1.0)
    assert list(it) == []


def test_jsonl_iterator_rel_seek_negative():
    objs = [{"n": i} for i in range(5)]
    f = io.BytesIO(_bytes_obj(objs))
    # -1.0 is not allowed (must be > -1.0), but -0.5 maps to 0.5
    it = JSONLIterator(f, rel_seek=-0.5)
    # just ensure it iterates without error and is a subset of objs
    result = list(it)
    assert all(o in objs for o in result)


def test_jsonl_iterator_rel_seek_invalid_too_large():
    f = io.BytesIO(b'{"a": 1}\n')
    with pytest.raises(ValueError):
        JSONLIterator(f, rel_seek=1.5)


def test_jsonl_iterator_rel_seek_invalid_zero_negative():
    f = io.BytesIO(b'{"a": 1}\n')
    with pytest.raises(ValueError):
        JSONLIterator(f, rel_seek=-1.0)


def test_jsonl_iterator_rel_seek_invalid_too_negative():
    f = io.BytesIO(b'{"a": 1}\n')
    with pytest.raises(ValueError):
        JSONLIterator(f, rel_seek=-2.0)


def test_jsonl_iterator_cur_byte_pos_forward():
    data = _bytes_obj([{"a": 1}, {"b": 2}])
    f = io.BytesIO(data)
    it = JSONLIterator(f)
    assert it.cur_byte_pos == 0
    next(it)
    first_len = len(json.dumps({"a": 1}) + "\n")
    assert it.cur_byte_pos == first_len


def test_jsonl_iterator_cur_byte_pos_reverse_preseek():
    data = _bytes_obj([{"a": 1}, {"b": 2}])
    f = io.BytesIO(data)
    it = JSONLIterator(f, reverse=True)
    # reverse with default rel_seek=1.0 seeks to end of file
    assert it.cur_byte_pos == len(data)


def test_jsonl_iterator_iter_returns_self():
    f = io.BytesIO(b'{"a": 1}\n')
    it = JSONLIterator(f)
    assert iter(it) is it


def test_jsonl_iterator_empty_file():
    f = io.BytesIO(b"")
    assert list(JSONLIterator(f)) == []


def test_jsonl_iterator_empty_reverse():
    f = io.BytesIO(b"")
    assert list(JSONLIterator(f, reverse=True)) == []


def test_jsonl_iterator_rel_seek_on_text_stream():
    objs = [{"n": i} for i in range(10)]
    f = io.StringIO(_text_obj(objs))
    it = JSONLIterator(f, rel_seek=0.0)
    assert list(it) == objs


def test_jsonl_iterator_reverse_with_rel_seek_partial():
    objs = [{"n": i} for i in range(10)]
    f = io.BytesIO(_bytes_obj(objs))
    it = JSONLIterator(f, reverse=True, rel_seek=0.5)
    result = list(it)
    # result is a reversed subset of the objects; each must be valid
    assert all(o in objs for o in result)
    # order within result should be descending by n
    ns = [o["n"] for o in result]
    assert ns == sorted(ns, reverse=True)


def test_jsonl_iterator_explicit_rel_seek_forward_partial():
    objs = [{"n": i} for i in range(10)]
    f = io.BytesIO(_bytes_obj(objs))
    it = JSONLIterator(f, rel_seek=0.5)
    result = list(it)
    assert all(o in objs for o in result)
    # forward partial should be contiguous tail of the objects
    if result:
        start = result[0]["n"]
        assert result == objs[start:]
