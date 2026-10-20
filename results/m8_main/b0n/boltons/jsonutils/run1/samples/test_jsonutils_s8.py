import io
import json
import os

import pytest

from boltons.jsonutils import reverse_iter_lines, JSONLIterator


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

def test_reverse_iter_lines_bytes_simple():
    f = io.BytesIO(b"a\nb\nc\n")
    result = list(reverse_iter_lines(f))
    assert result == [b"c", b"b", b"a"]


def test_reverse_iter_lines_bytes_no_trailing_newline():
    f = io.BytesIO(b"a\nb\nc")
    result = list(reverse_iter_lines(f))
    assert result == [b"c", b"b", b"a"]


def test_reverse_iter_lines_bytes_single_line_no_newline():
    f = io.BytesIO(b"onlyline")
    assert list(reverse_iter_lines(f)) == [b"onlyline"]


def test_reverse_iter_lines_bytes_single_line_with_newline():
    f = io.BytesIO(b"onlyline\n")
    result = list(reverse_iter_lines(f))
    # trailing newline produces an empty line first
    assert result == [b"", b"onlyline"]


def test_reverse_iter_lines_empty():
    f = io.BytesIO(b"")
    assert list(reverse_iter_lines(f)) == []


def test_reverse_iter_lines_small_blocksize():
    data = b"\n".join([b"line%d" % i for i in range(20)]) + b"\n"
    f = io.BytesIO(data)
    result = list(reverse_iter_lines(f, blocksize=16))
    expected = [b"line%d" % i for i in range(19, -1, -1)]
    assert result == expected


def test_reverse_iter_lines_preseek_false():
    f = io.BytesIO(b"a\nb\nc\n")
    f.seek(0, os.SEEK_END)
    result = list(reverse_iter_lines(f, preseek=False))
    assert result == [b"c", b"b", b"a"]


def test_reverse_iter_lines_preseek_false_mid_file():
    f = io.BytesIO(b"a\nb\nc\nd\n")
    # position right after "b\n"
    f.seek(4)
    result = list(reverse_iter_lines(f, preseek=False))
    assert result == [b"b", b"a"]


def test_reverse_iter_lines_encoding():
    f = io.BytesIO(b"a\nb\nc\n")
    result = list(reverse_iter_lines(f, encoding="utf-8"))
    assert result == ["c", "b", "a"]


def test_reverse_iter_lines_text_file():
    f = io.StringIO("a\nb\nc\n")
    result = list(reverse_iter_lines(f))
    assert result == ["c", "b", "a"]


def test_reverse_iter_lines_text_file_no_trailing_newline():
    f = io.StringIO("a\nb\nc")
    result = list(reverse_iter_lines(f))
    assert result == ["c", "b", "a"]


def test_reverse_iter_lines_real_file_bytes(tmp_path):
    p = tmp_path / "data.txt"
    p.write_bytes(b"x\ny\nz\n")
    with open(str(p), "rb") as f:
        assert list(reverse_iter_lines(f)) == [b"z", b"y", b"x"]


def test_reverse_iter_lines_real_file_text(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("x\ny\nz\n")
    with open(str(p), "r") as f:
        assert list(reverse_iter_lines(f)) == ["z", "y", "x"]


def test_reverse_iter_lines_text_wrapper_over_utf8(tmp_path):
    p = tmp_path / "data.txt"
    p.write_bytes("é\nü\n".encode("utf-8"))
    with open(str(p), "r", encoding="utf-8") as f:
        assert list(reverse_iter_lines(f)) == ["ü", "é"]


# ---------------------------------------------------------------------------
# JSONLIterator: forward iteration
# ---------------------------------------------------------------------------

def _jsonl_bytes(objs):
    return b"".join(json.dumps(o).encode("utf-8") + b"\n" for o in objs)


def test_jsonl_iterator_forward():
    objs = [{"a": 1}, {"b": 2}, [1, 2, 3], "hello", 42]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f)
    assert list(it) == objs


def test_jsonl_iterator_forward_no_trailing_newline():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}')
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_skips_blank_lines():
    f = io.BytesIO(b'\n\n{"a": 1}\n\n\n{"b": 2}\n\n')
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_lstrip_whitespace():
    f = io.BytesIO(b'   {"a": 1}\n\t{"b": 2}\n')
    it = JSONLIterator(f)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_empty_file():
    f = io.BytesIO(b"")
    assert list(JSONLIterator(f)) == []


def test_jsonl_iterator_is_its_own_iterator():
    f = io.BytesIO(b'{"a": 1}\n')
    it = JSONLIterator(f)
    assert iter(it) is it


def test_jsonl_iterator_next_and_cur_byte_pos():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n')
    it = JSONLIterator(f)
    assert it.cur_byte_pos == 0
    assert next(it) == {"a": 1}
    assert it.cur_byte_pos > 0
    assert next(it) == {"b": 2}


def test_jsonl_iterator_ignore_errors_true():
    f = io.BytesIO(b'{"a": 1}\nnot valid json\n{"b": 2}\n')
    it = JSONLIterator(f, ignore_errors=True)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonl_iterator_ignore_errors_false():
    f = io.BytesIO(b'{"a": 1}\nnot valid json\n{"b": 2}\n')
    it = JSONLIterator(f, ignore_errors=False)
    assert next(it) == {"a": 1}
    with pytest.raises(ValueError):
        next(it)


# ---------------------------------------------------------------------------
# JSONLIterator: reverse iteration
# ---------------------------------------------------------------------------

def test_jsonl_iterator_reverse():
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f, reverse=True)
    assert list(it) == list(reversed(objs))


def test_jsonl_iterator_reverse_no_trailing_newline():
    f = io.BytesIO(b'{"a": 1}\n{"b": 2}\n{"c": 3}')
    it = JSONLIterator(f, reverse=True)
    assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]


def test_jsonl_iterator_reverse_ignore_errors():
    f = io.BytesIO(b'{"a": 1}\nbad json\n{"c": 3}\n')
    it = JSONLIterator(f, reverse=True, ignore_errors=True)
    assert list(it) == [{"c": 3}, {"a": 1}]


def test_jsonl_iterator_reverse_ignore_errors_false():
    f = io.BytesIO(b'{"a": 1}\nbad json\n{"c": 3}\n')
    it = JSONLIterator(f, reverse=True, ignore_errors=False)
    assert next(it) == {"c": 3}
    with pytest.raises(ValueError):
        next(it)


# ---------------------------------------------------------------------------
# JSONLIterator: rel_seek
# ---------------------------------------------------------------------------

def test_jsonl_iterator_rel_seek_zero():
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f, rel_seek=0.0)
    assert list(it) == objs


def test_jsonl_iterator_rel_seek_one():
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f, rel_seek=1.0)
    assert list(it) == []


def test_jsonl_iterator_rel_seek_middle():
    objs = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f, rel_seek=0.5)
    # Should start at or after the first line's boundary; last object present.
    result = list(it)
    assert result[-1] == {"d": 4}
    assert len(result) <= len(objs)


def test_jsonl_iterator_rel_seek_negative():
    objs = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f, rel_seek=-0.5)
    assert it._rel_seek == pytest.approx(0.5)
    result = list(it)
    assert result[-1] == {"d": 4}


def test_jsonl_iterator_rel_seek_negative_one():
    objs = [{"a": 1}, {"b": 2}]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f, rel_seek=-1.0)
    assert it._rel_seek == pytest.approx(0.0)
    assert list(it) == objs


def test_jsonl_iterator_rel_seek_out_of_range():
    f = io.BytesIO(b'{"a": 1}\n')
    with pytest.raises(ValueError):
        JSONLIterator(f, rel_seek=1.5)
    f2 = io.BytesIO(b'{"a": 1}\n')
    with pytest.raises(ValueError):
        JSONLIterator(f2, rel_seek=-1.5)


def test_jsonl_iterator_reverse_with_rel_seek_zero():
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f, reverse=True, rel_seek=0.0)
    assert list(it) == []


def test_jsonl_iterator_reverse_with_rel_seek_one():
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f, reverse=True, rel_seek=1.0)
    assert list(it) == list(reversed(objs))


def test_jsonl_iterator_reverse_default_rel_seek():
    objs = [{"a": 1}, {"b": 2}]
    f = io.BytesIO(_jsonl_bytes(objs))
    it = JSONLIterator(f, reverse=True)
    assert it._rel_seek == 1.0
    assert list(it) == list(reversed(objs))


# ---------------------------------------------------------------------------
# JSONLIterator: real file objects
# ---------------------------------------------------------------------------

def test_jsonl_iterator_real_file_bytes(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    p.write_bytes(_jsonl_bytes(objs))
    with open(str(p), "rb") as f:
        it = JSONLIterator(f)
        assert list(it) == objs


def test_jsonl_iterator_real_file_text(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}]
    p.write_text("".join(json.dumps(o) + "\n" for o in objs))
    with open(str(p), "r") as f:
        it = JSONLIterator(f)
        assert list(it) == objs


def test_jsonl_iterator_real_file_reverse(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    p.write_bytes(_jsonl_bytes(objs))
    with open(str(p), "rb") as f:
        it = JSONLIterator(f, reverse=True)
        assert list(it) == list(reversed(objs))


def test_jsonl_iterator_bool_attrs():
    f = io.BytesIO(b"")
    it = JSONLIterator(f, reverse=1)
    assert it._reverse is True
    assert it.ignore_errors is False
