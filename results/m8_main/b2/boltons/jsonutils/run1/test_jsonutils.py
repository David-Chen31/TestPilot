import io
import os
import json

import pytest

from boltons.jsonutils import reverse_iter_lines, JSONLIterator, DEFAULT_BLOCKSIZE


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

def _collect_reverse(file_obj, **kwargs):
    return list(reverse_iter_lines(file_obj, **kwargs))


def test_reverse_iter_lines_bytes_basic(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"a\nb\nc\n")
    with open(p, "rb") as f:
        result = _collect_reverse(f)
    # last line first; a trailing newline emits an initial empty entry
    assert result == [b"", b"c", b"b", b"a"]


def test_reverse_iter_lines_bytes_no_trailing_newline(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"a\nb\nc")
    with open(p, "rb") as f:
        result = _collect_reverse(f)
    assert result == [b"c", b"b", b"a"]


def test_reverse_iter_lines_empty_file(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"")
    with open(p, "rb") as f:
        result = _collect_reverse(f)
    assert result == []


def test_reverse_iter_lines_single_line(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"only")
    with open(p, "rb") as f:
        result = _collect_reverse(f)
    assert result == [b"only"]


def test_reverse_iter_lines_single_line_with_newline(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"only\n")
    with open(p, "rb") as f:
        result = _collect_reverse(f)
    # small file: everything fits in one buffer, so it yields the buffer
    # wholesale without splitting off the trailing newline
    assert result == [b"only\n"]


def test_reverse_iter_lines_text_mode(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    with open(p, "r", encoding="utf-8") as f:
        result = _collect_reverse(f)
    assert result == ["", "c", "b", "a"]


def test_reverse_iter_lines_encoding_override(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"a\nb\nc\n")
    with open(p, "rb") as f:
        result = _collect_reverse(f, encoding="utf-8")
    assert result == ["", "c", "b", "a"]


def test_reverse_iter_lines_small_blocksize(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"line1\nline2\nline3\n")
    with open(p, "rb") as f:
        result = _collect_reverse(f, blocksize=2)
    assert result == [b"", b"line3", b"line2", b"line1"]


def test_reverse_iter_lines_preseek_false_from_end(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"a\nb\nc\n")
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        result = _collect_reverse(f, preseek=False)
    assert result == [b"", b"c", b"b", b"a"]


def test_reverse_iter_lines_preseek_false_partial(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"a\nb\nc\n")
    with open(p, "rb") as f:
        # position in middle of file; read backwards from here
        f.seek(3, os.SEEK_SET)
        result = _collect_reverse(f, preseek=False)
    # reading backwards from position 3 covers "a\nb\n" (0..3).
    # The buffer "a\nb\n" splits into ["a", "b"] (splitlines drops the
    # trailing newline), so the loop yields b"b" and keeps b"a" as buff,
    # then finally yields b"a". No empty entry is emitted because the last
    # byte read is a newline only if buff after the final segment triggers
    # the special-case branch, which does not happen mid-read here.
    assert result == [b"b", b"a"]


def test_reverse_iter_lines_stringio_raises_typeerror():
    # StringIO without encoding causes str/bytes concatenation error
    sio = io.StringIO("a\nb\nc\n")
    with pytest.raises(TypeError):
        list(reverse_iter_lines(sio))


def test_reverse_iter_lines_bytesio():
    bio = io.BytesIO(b"a\nb\nc\n")
    result = list(reverse_iter_lines(bio))
    # BytesIO: seeks to end, reads all in one block, splits lines and yields
    # them reversed, with an empty entry for the trailing newline
    assert result == [b"", b"c", b"b", b"a"]


def test_reverse_iter_lines_trailing_newline_multiple(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"a\n\n\nb\n")
    with open(p, "rb") as f:
        result = _collect_reverse(f)
    assert result == [b"", b"b", b"", b"", b"a"]


# ---------------------------------------------------------------------------
# JSONLIterator - forward mode
# ---------------------------------------------------------------------------

def _write_jsonl(path, objs):
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o) + "\n")


def test_jsonliterator_forward_text(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": [1, 2]}, {"c": "x"}]
    _write_jsonl(p, objs)
    with open(p, "r", encoding="utf-8") as f:
        it = JSONLIterator(f)
        assert it.ignore_errors is False
        result = list(it)
    assert result == objs


def test_jsonliterator_forward_bytes(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}]
    with open(p, "wb") as f:
        for o in objs:
            f.write((json.dumps(o) + "\n").encode("utf-8"))
    with open(p, "rb") as f:
        result = list(JSONLIterator(f))
    assert result == objs


def test_jsonliterator_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    with open(p, "wb") as f:
        f.write(b'{"a": 1}\n\n   \n{"b": 2}\n')
    with open(p, "rb") as f:
        result = list(JSONLIterator(f))
    assert result == [{"a": 1}, {"b": 2}]


def test_jsonliterator_whitespace_lstrip(tmp_path):
    p = tmp_path / "data.jsonl"
    with open(p, "wb") as f:
        f.write(b'   {"a": 1}\n\t{"b": 2}\n')
    with open(p, "rb") as f:
        result = list(JSONLIterator(f))
    assert result == [{"a": 1}, {"b": 2}]


def test_jsonliterator_ignore_errors(tmp_path):
    p = tmp_path / "data.jsonl"
    with open(p, "wb") as f:
        f.write(b'{"a": 1}\nnot json\n{"b": 2}\n')
    with open(p, "rb") as f:
        result = list(JSONLIterator(f, ignore_errors=True))
    assert result == [{"a": 1}, {"b": 2}]


def test_jsonliterator_raises_without_ignore_errors(tmp_path):
    p = tmp_path / "data.jsonl"
    with open(p, "wb") as f:
        f.write(b'{"a": 1}\nnot json\n')
    with open(p, "rb") as f:
        it = JSONLIterator(f)
        assert next(it) == {"a": 1}
        with pytest.raises(ValueError):
            next(it)


def test_jsonliterator_next_and_iter(tmp_path):
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"x": 1}, {"y": 2}])
    with open(p, "r", encoding="utf-8") as f:
        it = JSONLIterator(f)
        assert iter(it) is it
        assert next(it) == {"x": 1}
        assert it.__next__() == {"y": 2}
        with pytest.raises(StopIteration):
            next(it)


def test_jsonliterator_empty_file(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_bytes(b"")
    with open(p, "rb") as f:
        assert list(JSONLIterator(f)) == []


def test_jsonliterator_cur_byte_pos(tmp_path):
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"a": 1}, {"b": 2}])
    with open(p, "rb") as f:
        it = JSONLIterator(f)
        next(it)
        pos = it.cur_byte_pos
        assert pos > 0
        assert pos == f.tell()


# ---------------------------------------------------------------------------
# JSONLIterator - reverse mode
# ---------------------------------------------------------------------------

def test_jsonliterator_reverse_text(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    _write_jsonl(p, objs)
    with open(p, "r", encoding="utf-8") as f:
        result = list(JSONLIterator(f, reverse=True))
    assert result == list(reversed(objs))


def test_jsonliterator_reverse_bytes(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    with open(p, "wb") as f:
        for o in objs:
            f.write((json.dumps(o) + "\n").encode())
    with open(p, "rb") as f:
        result = list(JSONLIterator(f, reverse=True))
    assert result == list(reversed(objs))


def test_jsonliterator_reverse_ignore_errors(tmp_path):
    p = tmp_path / "data.jsonl"
    with open(p, "wb") as f:
        f.write(b'{"a": 1}\nbad json\n{"c": 3}\n')
    with open(p, "rb") as f:
        result = list(JSONLIterator(f, reverse=True, ignore_errors=True))
    assert result == [{"c": 3}, {"a": 1}]


# ---------------------------------------------------------------------------
# JSONLIterator - rel_seek
# ---------------------------------------------------------------------------

def test_jsonliterator_rel_seek_zero(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}]
    _write_jsonl(p, objs)
    with open(p, "rb") as f:
        result = list(JSONLIterator(f, rel_seek=0.0))
    assert result == objs


def test_jsonliterator_rel_seek_one(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}]
    _write_jsonl(p, objs)
    with open(p, "rb") as f:
        # rel_seek=1.0, forward: starts at end -> empty
        result = list(JSONLIterator(f, rel_seek=1.0))
    assert result == []


def test_jsonliterator_rel_seek_one_reverse(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}]
    _write_jsonl(p, objs)
    with open(p, "rb") as f:
        result = list(JSONLIterator(f, rel_seek=1.0, reverse=True))
    assert result == list(reversed(objs))


def test_jsonliterator_rel_seek_invalid_low(tmp_path):
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"a": 1}])
    with open(p, "rb") as f:
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=-1.0)


def test_jsonliterator_rel_seek_invalid_high(tmp_path):
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"a": 1}])
    with open(p, "rb") as f:
        with pytest.raises(ValueError):
            JSONLIterator(f, rel_seek=1.5)


def test_jsonliterator_rel_seek_negative(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    _write_jsonl(p, objs)
    with open(p, "rb") as f:
        result = list(JSONLIterator(f, rel_seek=-0.5))
    # -0.5 maps to 0.5 -> target near middle; in this short file, aligns
    # to the last line, yielding only the final object
    assert result == [{"c": 3}]


def test_jsonliterator_rel_seek_forward_mid(tmp_path):
    p = tmp_path / "data.jsonl"
    objs = [{"a": 1}, {"b": 2}, {"c": 3}]
    _write_jsonl(p, objs)
    with open(p, "rb") as f:
        result = list(JSONLIterator(f, rel_seek=0.5))
    # target lands in the last line, aligned past it -> only last object
    assert result == [{"c": 3}]


# ---------------------------------------------------------------------------
# defaults / module constants
# ---------------------------------------------------------------------------

def test_default_blocksize_value():
    assert DEFAULT_BLOCKSIZE == 4096
