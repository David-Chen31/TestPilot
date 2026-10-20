import io
import json
import runpy
import sys

import pytest

from boltons.jsonutils import JSONLIterator, reverse_iter_lines, DEFAULT_BLOCKSIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lines_of(data):
    if isinstance(data, str):
        return data.splitlines()
    return data.splitlines()


# ---------------------------------------------------------------------------
# reverse_iter_lines
# ---------------------------------------------------------------------------

def test_reverse_iter_lines_basic_bytes():
    # A trailing newline causes an initial empty element to be yielded.
    data = b"one\ntwo\nthree\n"
    fo = io.BytesIO(data)
    result = list(reverse_iter_lines(fo))
    assert result == [b"", b"three", b"two", b"one"]


def test_reverse_iter_lines_no_trailing_newline_bytes():
    data = b"one\ntwo\nthree"
    fo = io.BytesIO(data)
    result = list(reverse_iter_lines(fo))
    assert result == [b"three", b"two", b"one"]


def test_reverse_iter_lines_empty():
    fo = io.BytesIO(b"")
    assert list(reverse_iter_lines(fo)) == []


def test_reverse_iter_lines_single_line():
    fo = io.BytesIO(b"only")
    assert list(reverse_iter_lines(fo)) == [b"only"]


def test_reverse_iter_lines_trailing_blank_line():
    # "a\n\n" -> trailing newline yields b"", and the blank line itself
    # yields another b"", then b"a".
    fo = io.BytesIO(b"a\n\n")
    result = list(reverse_iter_lines(fo))
    assert result == [b"", b"", b"a"]


def test_reverse_iter_lines_small_blocksize():
    data = b"alpha\nbeta\ngamma\ndelta\n"
    small = list(reverse_iter_lines(io.BytesIO(data), blocksize=4))
    large = list(reverse_iter_lines(io.BytesIO(data), blocksize=DEFAULT_BLOCKSIZE))
    expected = [b"", b"delta", b"gamma", b"beta", b"alpha"]
    assert small == large == expected


def test_reverse_iter_lines_text_stream_with_encoding():
    # io.StringIO has no .buffer and returns str, which this
    # implementation does not support (it mixes str with bytes).
    fo = io.StringIO("a\nb\nc\n")
    with pytest.raises(TypeError):
        list(reverse_iter_lines(fo))


def test_reverse_iter_lines_text_stream_explicit_encoding():
    # Same limitation: StringIO is not supported by reverse_iter_lines.
    fo = io.StringIO("x\ny\n")
    with pytest.raises(TypeError):
        list(reverse_iter_lines(fo, encoding="utf-8"))


def test_reverse_iter_lines_utf8_bytes_with_encoding():
    data = "héllo\nwörld\n".encode("utf-8")
    fo = io.BytesIO(data)
    result = list(reverse_iter_lines(fo, encoding="utf-8"))
    assert result == ["", "wörld", "héllo"]


def test_reverse_iter_lines_preseek_false():
    fo = io.BytesIO(b"a\nb\nc\n")
    # move to middle position manually (end of 'b\n')
    fo.seek(0, io.SEEK_END)
    fo.seek(4)
    result = list(reverse_iter_lines(fo, preseek=False))
    assert result == [b"", b"b", b"a"]


def test_reverse_iter_lines_preseek_true_default_seeks_to_end():
    fo = io.BytesIO(b"a\nb\n")
    fo.seek(0)
    result = list(reverse_iter_lines(fo))
    # starting cursor is ignored when preseek=True; trailing newline
    # produces a leading empty element.
    assert result == [b"", b"b", b"a"]


# ---------------------------------------------------------------------------
# JSONLIterator - forward
# ---------------------------------------------------------------------------

def test_jsonliter_forward_basic():
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'
    fo = io.BytesIO(content)
    it = JSONLIterator(fo)
    assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_jsonliter_forward_empty():
    it = JSONLIterator(io.BytesIO(b""))
    assert list(it) == []


def test_jsonliter_forward_skips_blank_lines():
    content = b'{"a": 1}\n\n\n{"b": 2}\n'
    it = JSONLIterator(io.BytesIO(content))
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonliter_forward_non_dicts():
    content = b'[1, 2]\n"string"\n42\n'
    it = JSONLIterator(io.BytesIO(content))
    assert list(it) == [[1, 2], "string", 42]


def test_jsonliter_forward_error_raises_by_default():
    content = b'{"a": 1}\nnot json\n{"b": 2}\n'
    it = JSONLIterator(io.BytesIO(content))
    assert next(it) == {"a": 1}
    with pytest.raises(ValueError):
        next(it)


def test_jsonliter_forward_ignore_errors():
    content = b'{"a": 1}\nnot json\n{"b": 2}\n'
    it = JSONLIterator(io.BytesIO(content), ignore_errors=True)
    assert list(it) == [{"a": 1}, {"b": 2}]


def test_jsonliter_next_and_iter():
    it = JSONLIterator(io.BytesIO(b'{"a": 1}\n'))
    assert iter(it) is it
    assert it.next() == {"a": 1}
    with pytest.raises(StopIteration):
        next(it)


def test_jsonliter_cur_byte_pos():
    content = b'{"a": 1}\n{"b": 2}\n'
    fo = io.BytesIO(content)
    it = JSONLIterator(fo)
    assert it.cur_byte_pos == 0
    next(it)
    assert it.cur_byte_pos == len(b'{"a": 1}\n')


# ---------------------------------------------------------------------------
# JSONLIterator - reverse
# ---------------------------------------------------------------------------

def test_jsonliter_reverse_basic():
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'
    it = JSONLIterator(io.BytesIO(content), reverse=True)
    assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]


def test_jsonliter_reverse_no_trailing_newline():
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}'
    it = JSONLIterator(io.BytesIO(content), reverse=True)
    assert list(it) == [{"c": 3}, {"b": 2}, {"a": 1}]


def test_jsonliter_reverse_ignore_errors():
    content = b'{"a": 1}\nbad line\n{"b": 2}\n'
    it = JSONLIterator(io.BytesIO(content), reverse=True, ignore_errors=True)
    assert list(it) == [{"b": 2}, {"a": 1}]


def test_jsonliter_reverse_errors_raise_by_default():
    content = b'{"a": 1}\nbad line\n{"b": 2}\n'
    it = JSONLIterator(io.BytesIO(content), reverse=True)
    assert next(it) == {"b": 2}
    with pytest.raises(ValueError):
        next(it)


# ---------------------------------------------------------------------------
# JSONLIterator - rel_seek
# ---------------------------------------------------------------------------

def test_jsonliter_rel_seek_zero_is_start():
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'
    it = JSONLIterator(io.BytesIO(content), rel_seek=0.0)
    assert list(it) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_jsonliter_rel_seek_one_is_end():
    content = b'{"a": 1}\n{"b": 2}\n'
    it = JSONLIterator(io.BytesIO(content), rel_seek=1.0)
    assert list(it) == []


def test_jsonliter_rel_seek_negative():
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n{"d": 4}\n'
    # -0.5 -> 0.5 -> mid file, aligned to next newline.
    it = JSONLIterator(io.BytesIO(content), rel_seek=-0.5)
    objects = list(it)
    assert objects
    # every object returned must be one of the valid ones and in order
    assert objects[-1] == {"d": 4}


def test_jsonliter_rel_seek_invalid_too_large():
    with pytest.raises(ValueError):
        JSONLIterator(io.BytesIO(b""), rel_seek=1.5)


def test_jsonliter_rel_seek_invalid_too_small():
    with pytest.raises(ValueError):
        JSONLIterator(io.BytesIO(b""), rel_seek=-1.5)


def test_jsonliter_rel_seek_invalid_boundary_negative_one():
    # -1.0 is not in (-1.0, 1.0]
    with pytest.raises(ValueError):
        JSONLIterator(io.BytesIO(b""), rel_seek=-1.0)


def test_jsonliter_rel_seek_reverse_combination():
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n{"d": 4}\n'
    it = JSONLIterator(io.BytesIO(content), reverse=True, rel_seek=0.5)
    objects = list(it)
    assert objects
    # rel_seek=0.5 seeks into the middle and aligns forward to a newline,
    # so the first reverse-yielded object is {"c": 3}, and {"d": 4} is
    # skipped because iteration is backwards from the aligned position.
    assert objects[0] == {"c": 3}


def test_jsonliter_reverse_default_rel_seek_is_one():
    content = b'{"a": 1}\n{"b": 2}\n'
    it = JSONLIterator(io.BytesIO(content), reverse=True)
    assert it._rel_seek == 1.0
    assert list(it) == [{"b": 2}, {"a": 1}]


# ---------------------------------------------------------------------------
# Malformed partial-line / alignment edge cases
# ---------------------------------------------------------------------------

def test_jsonliter_rel_seek_aligns_to_newline():
    # Seek to a byte inside the second line; iterator should align forward
    # to the next newline and thus skip the partial line.
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'
    size = len(content)
    # pick a rel_seek targeting inside the second line
    target_byte = content.index(b'{"b"') + 2
    rel_seek = target_byte / size
    it = JSONLIterator(io.BytesIO(content), rel_seek=rel_seek)
    objects = list(it)
    # should start at '"c": 3' line
    assert objects == [{"c": 3}]


def test_jsonliter_ignore_partial_trailing_data():
    content = b'{"a": 1}\n{"b": 2'
    it = JSONLIterator(io.BytesIO(content), ignore_errors=True)
    assert list(it) == [{"a": 1}]


# ---------------------------------------------------------------------------
# Additional coverage: _align_to_newline hitting end-of-file without a newline
# ---------------------------------------------------------------------------

def test_jsonliter_rel_seek_align_hits_eof_no_newline():
    # Choose a rel_seek that lands inside the trailing region which has no
    # following newline character. In that case _align_to_newline should
    # exhaust reads, seek to EOF, and iteration should yield nothing.
    content = b'{"a": 1}\n{"b": 2}'
    size = len(content)
    # target somewhere inside the trailing partial line (no trailing newline)
    target_byte = content.index(b'{"b"') + 3
    rel_seek = target_byte / size
    it = JSONLIterator(io.BytesIO(content), rel_seek=rel_seek)
    objects = list(it)
    # Alignment lands at EOF; no object can be produced.
    assert objects == []


def test_jsonliter_rel_seek_negative_align_hits_eof_no_newline():
    # Same as above but going via the negative relative-seek path.
    content = b'{"a": 1}\n{"b": 2}'
    size = len(content)
    target_byte = content.index(b'{"b"') + 3
    rel_seek = target_byte / size - 1.0  # negative in (-1.0, 0.0]
    assert -1.0 < rel_seek < 0.0
    it = JSONLIterator(io.BytesIO(content), rel_seek=rel_seek)
    assert list(it) == []


def test_jsonliter_rel_seek_align_to_newline_after_partial():
    # A case where alignment succeeds: rel_seek points into line 2, and
    # line 3 follows with a newline, so alignment finds the newline.
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'
    size = len(content)
    target_byte = content.index(b'{"b"') + 4
    rel_seek = target_byte / size
    it = JSONLIterator(io.BytesIO(content), rel_seek=rel_seek)
    assert list(it) == [{"c": 3}]


def test_jsonliter_align_to_newline_empty_after_target():
    # Ensure alignment reading an empty string (EOF) is handled and the
    # iterator is positioned at the end.
    content = b'{"a": 1}\n'
    size = len(content)
    # Target the final newline position precisely, then rel_seek would
    # align at end after reading nothing newline-related in that block.
    it = JSONLIterator(io.BytesIO(content), rel_seek=0.9)
    # rel_seek 0.9 lands near the end of file within or at the newline;
    # result should not include '{"a": 1}' in this forward direction.
    assert list(it) == []


# ---------------------------------------------------------------------------
# __main__ / _main() coverage
# ---------------------------------------------------------------------------

def _run_main(monkeypatch, tmp_path, argv_tail, files=None):
    """Write given files under tmp_path, run boltons.jsonutils.__main__ with
    argv, and return captured stdout."""
    written = []
    if files:
        for name, content in files.items():
            p = tmp_path / name
            p.write_bytes(content)
            written.append(str(p))
    monkeypatch.setattr(sys, "argv", ["jsonutils"] + list(argv_tail) + written)
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    runpy.run_module("boltons.jsonutils", run_name="__main__")
    return captured.getvalue()


def test_main_help(monkeypatch, tmp_path):
    out = _run_main(monkeypatch, tmp_path, ["--help"])
    assert out == "loads one or more JSON Line files for basic validation.\n"


def test_main_help_short(monkeypatch, tmp_path):
    out = _run_main(monkeypatch, tmp_path, ["-h"])
    assert out == "loads one or more JSON Line files for basic validation.\n"


def test_main_valid_file_silent(monkeypatch, tmp_path):
    out = _run_main(
        monkeypatch, tmp_path, [],
        files={"valid.jsonl": b'{"a": 1}\n{"b": 2}\n'},
    )
    assert out == ""


def test_main_verbose_small_file(monkeypatch, tmp_path):
    # Only 2 objects; verbose progress dots appear only every 100 objects,
    # so only summary lines appear.
    out = _run_main(
        monkeypatch, tmp_path, ["-v"],
        files={"small.jsonl": b'{"a": 1}\n{"b": 2}\n'},
    )
    assert out == "files checked: 1\nobjects loaded: 2\n"


def test_main_verbose_short_flag(monkeypatch, tmp_path):
    out = _run_main(
        monkeypatch, tmp_path, ["--verbose"],
        files={"small.jsonl": b'{"a": 1}\n'},
    )
    assert out == "files checked: 1\nobjects loaded: 1\n"


def test_main_verbose_multiple_files(monkeypatch, tmp_path):
    out = _run_main(
        monkeypatch, tmp_path, ["-v"],
        files={
            "a.jsonl": b'{"x": 1}\n',
            "b.jsonl": b'{"y": 2}\n{"z": 3}\n',
        },
    )
    assert out == "files checked: 2\nobjects loaded: 3\n"


def test_main_error_reports_and_returns(monkeypatch, tmp_path):
    out = _run_main(
        monkeypatch, tmp_path, [],
        files={"bad.jsonl": b'{"a": 1}\nnot json\n{"b": 2}\n'},
    )
    # Error report includes object #2; byte position varies but the
    # message format is deterministic for the prefix and file name.
    assert out.startswith("error reading object #2 around byte ")
    assert out.rstrip().endswith("in " + str(tmp_path / "bad.jsonl"))
    assert out.endswith("\n")


def test_main_ignores_flag_like_filenames(monkeypatch, tmp_path):
    # Flags appearing before files get skipped as filenames; a real file
    # afterwards is processed.
    out = _run_main(
        monkeypatch, tmp_path, ["-v"],
        files={"data.jsonl": b'{"a": 1}\n{"b": 2}\n'},
    )
    # The argv tail is ["-v"] and file appended after; verbose summary
    # should still report 1 file / 2 objects.
    assert out == "files checked: 1\nobjects loaded: 2\n"


def test_main_verbose_hundred_objects_progress(monkeypatch, tmp_path):
    # 100 objects trigger the progress dot branch. Note the implementation
    # has a quirk: it writes a '.' and then writes an extra newline only
    # when obj_count % 10000 is truthy.
    content = b"".join(b'{"i": %d}\n' % i for i in range(100))
    out = _run_main(
        monkeypatch, tmp_path, ["-v"],
        files={"hundred.jsonl": content},
    )
    assert out.startswith(".")
    assert out.endswith("files checked: 1\nobjects loaded: 100\n")


def test_main_no_files(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["jsonutils"])
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    runpy.run_module("boltons.jsonutils", run_name="__main__")
    assert captured.getvalue() == ""


def test_main_verbose_multiple_of_ten_thousand_writes_only_dot(monkeypatch, tmp_path):
    # Exercises the `if obj_count % 10000:` *false* branch (line 259): when
    # obj_count reaches a multiple of 10000, only a '.' is written and the
    # numeric count line is skipped.
    n_objs = 10000
    content = b"".join(b'{"i": %d}\n' % i for i in range(n_objs))
    out = _run_main(
        monkeypatch, tmp_path, ["-v"],
        files={"tenk.jsonl": content},
    )
    # For every multiple of 100 below 10000 the count number is written.
    expected_progress = "".join(".%d\n" % n for n in range(100, n_objs, 100))
    # At exactly 10000 objects (a multiple of 10000), only a lone '.' is
    # emitted — the number branch is skipped.
    expected_progress += "."
    expected = expected_progress + (
        "files checked: 1\nobjects loaded: %d\n" % n_objs
    )
    assert out == expected
