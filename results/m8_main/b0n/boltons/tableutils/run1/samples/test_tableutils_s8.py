"""Tests for boltons.tableutils."""

import collections
import types

import pytest

from boltons.tableutils import (
    to_text,
    escape_html,
    UnsupportedData,
    InputType,
    DictInputType,
    ObjectInputType,
    ListInputType,
    TupleInputType,
    NamedTupleInputType,
    Table,
    _MISSING,
)


# ---------------------------------------------------------------------------
# to_text
# ---------------------------------------------------------------------------

def test_to_text_basic():
    assert to_text(123) == "123"
    assert to_text("abc") == "abc"
    assert to_text(None) == "None"
    assert to_text(1.5) == "1.5"


def test_to_text_maxlen():
    assert to_text("abcdefghij", maxlen=5) == "ab..."
    # exactly maxlen: unchanged
    assert to_text("abcde", maxlen=5) == "abcde"
    # shorter than maxlen
    assert to_text("abc", maxlen=5) == "abc"


def test_to_text_no_maxlen():
    assert to_text("abcdefghij") == "abcdefghij"


def test_to_text_evil_str_falls_back_to_repr():
    class Bad:
        def __str__(self):
            raise RuntimeError("nope")

        def __repr__(self):
            return "BADREPR"

    assert to_text(Bad()) == "BADREPR"


def test_to_text_evil_str_and_repr():
    class Bad:
        def __str__(self):
            raise RuntimeError("nope")

        def __repr__(self):
            raise RuntimeError("nope2")

    # falls back to object.__repr__, which contains the class name
    result = to_text(Bad())
    assert "Bad object at" in result


# ---------------------------------------------------------------------------
# escape_html
# ---------------------------------------------------------------------------

def test_escape_html_basic():
    assert escape_html("<b>") == "&lt;b&gt;"
    assert escape_html("a & b") == "a &amp; b"
    assert escape_html('"q"') == "&quot;q&quot;"
    assert escape_html("'s'") == "&#x27;s&#x27;"


def test_escape_html_maxlen():
    # text "<abcdef" truncated with maxlen=6 -> "<ab..."
    assert escape_html("<abcdef", maxlen=6) == "&lt;ab..."


def test_escape_html_non_string():
    assert escape_html(5) == "5"


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

def test_input_type_base():
    it = InputType()
    # base get_entry_seq requires get_entry, which base doesn't define
    with pytest.raises(AttributeError):
        it.get_entry_seq([1, 2], None)


def test_dict_input_type():
    it = DictInputType()
    assert it.check_type({}) is True
    assert it.check_type([]) is False
    assert it.check_type(collections.OrderedDict()) is True
    assert it.guess_headers({"b": 1, "a": 2}) == ["a", "b"]
    assert it.get_entry({"a": 1, "b": 2}, ["a", "b"]) == [1, 2]
    assert it.get_entry({"a": 1}, ["a", "b"]) == [1, None]
    seq = it.get_entry_seq([{"a": 1}, {"b": 2}], ["a", "b"])
    assert seq == [[1, None], [None, 2]]


def test_list_input_type():
    it = ListInputType()
    assert it.check_type([]) is True
    assert it.check_type((1,)) is False
    assert it.check_type({}) is False
    assert it.guess_headers([1, 2]) is None
    data = [1, 2, 3]
    assert it.get_entry(data, None) is data
    seq = [[1, 2], [3, 4]]
    assert it.get_entry_seq(seq, None) is seq


def test_tuple_input_type():
    it = TupleInputType()
    assert it.check_type((1, 2)) is True
    assert it.check_type([1, 2]) is False
    assert it.guess_headers((1, 2)) is None
    assert it.get_entry((1, 2, 3), None) == [1, 2, 3]
    seq = it.get_entry_seq([(1, 2), (3, 4)], None)
    assert seq == [[1, 2], [3, 4]]


def test_named_tuple_input_type():
    NT = collections.namedtuple("NT", ["x", "y"])
    it = NamedTupleInputType()
    n = NT(1, 2)
    assert it.check_type(n) is True
    assert it.check_type((1, 2)) is False
    assert it.check_type([1, 2]) is False
    assert it.guess_headers(n) == ["x", "y"]
    assert it.get_entry(n, ["x", "y"]) == [1, 2]
    # missing header -> None
    assert it.get_entry(n, ["x", "z"]) == [1, None]
    seq = it.get_entry_seq([NT(1, 2), NT(3, 4)], ["x", "y"])
    assert seq == [[1, 2], [3, 4]]


def test_object_input_type():
    class Obj:
        def __init__(self):
            self.a = 1
            self.b = 2

        def method(self):
            return 1

    it = ObjectInputType()
    obj = Obj()
    assert it.check_type(obj) is True
    # DNR types should be rejected
    assert it.check_type(1) is False
    assert it.check_type("x") is False
    assert it.check_type(None) is False
    headers = it.guess_headers(obj)
    assert "a" in headers
    assert "b" in headers
    assert "method" not in headers
    vals = it.get_entry(obj, ["a", "b"])
    assert vals == [1, 2]


def test_object_input_type_getattr_error():
    class Obj:
        a = 1

        @property
        def bad(self):
            raise ValueError("boom")

    it = ObjectInputType()
    obj = Obj()
    headers = it.guess_headers(obj)
    assert "a" in headers
    assert "bad" not in headers
    # get_entry catches errors and appends None
    assert it.get_entry(obj, ["a", "bad"]) == [1, None]


# ---------------------------------------------------------------------------
# Table basics
# ---------------------------------------------------------------------------

def test_table_empty():
    t = Table()
    assert len(t) == 0
    assert t.headers == []
    assert t.metadata == {}
    assert t._width == 0
    assert repr(t) == "Table([])"


def test_table_from_lists_no_headers():
    t = Table([[1, 2], [3, 4]])
    assert len(t) == 2
    assert t.headers == []
    assert t._width == 2
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_table_with_explicit_headers():
    t = Table([[1, 2]], headers=["a", "b"])
    assert t.headers == ["a", "b"]
    assert t._width == 2
    assert t[0] == [1, 2]


def test_table_first_row_becomes_headers():
    t = Table([["h1", "h2"], [1, 2], [3, 4]])
    assert t.headers == ["h1", "h2"]
    assert len(t) == 2
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_table_fills_short_rows():
    t = Table([[1], [2, 3, 4]], headers=["a", "b", "c"])
    assert t._width == 3
    assert t[0] == [1, None, None]
    assert t[1] == [2, 3, 4]


def test_table_repr_with_headers():
    t = Table([[1, 2]], headers=["a", "b"])
    assert repr(t) == "Table(headers=['a', 'b'], data=[[1, 2]])"


def test_table_metadata():
    t = Table([[1]], metadata={"k": "v"})
    assert t.metadata == {"k": "v"}


def test_table_len_and_getitem():
    t = Table([[1, 2], [3, 4]])
    assert len(t) == 2
    assert t[-1] == [3, 4]
    assert t[0:1] == [[1, 2]]


def test_table_extend():
    t = Table([[1, 2]])
    t.extend([[3, 4], [5, 6]])
    assert len(t) == 3
    assert t[2] == [5, 6]


def test_table_extend_empty():
    t = Table([[1, 2]])
    t.extend([])
    assert len(t) == 1


def test_table_from_dict():
    t = Table.from_dict({"a": 1, "b": 2})
    assert sorted(t.headers) == ["a", "b"]
    assert len(t) == 1
    # get_entry preserves header order
    entry = t[0]
    assert entry[t.headers.index("a")] == 1
    assert entry[t.headers.index("b")] == 2


def test_table_from_dict_sequence():
    t = Table.from_dict([{"a": 1}, {"b": 2}])
    assert t.headers == ["a", "b"]
    assert t[0] == [1, None]
    assert t[1] == [None, 2]


def test_table_from_list():
    t = Table.from_list([[1, 2], [3, 4]])
    assert t.headers == []
    assert t[0] == [1, 2]


def test_table_from_object():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    p = Point(1, 2)
    t = Table.from_object(p)
    assert "x" in t.headers
    assert "y" in t.headers
    assert len(t) == 1
    assert t[0][t.headers.index("x")] == 1


def test_table_from_namedtuple():
    NT = collections.namedtuple("NT", ["x", "y"])
    t = Table.from_data(NT(1, 2))
    assert t.headers == ["x", "y"]
    assert t[0] == [1, 2]


def test_table_from_data_sequence_of_namedtuples():
    NT = collections.namedtuple("NT", ["x", "y"])
    t = Table.from_data([NT(1, 2), NT(3, 4)])
    assert t.headers == ["x", "y"]
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_table_from_data_scalar():
    t = Table.from_data(42)
    assert len(t) == 1
    assert t[0] == [42]


def test_table_from_data_empty():
    t = Table.from_data([])
    assert len(t) == 0


def test_table_from_data_unsupported():
    class Opaque:
        __slots__ = ()

    with pytest.raises(UnsupportedData):
        Table.from_data(Opaque())


def test_table_from_data_max_depth_zero():
    t = Table.from_data([1, 2], max_depth=0)
    assert len(t) == 0


def test_table_from_data_nested_max_depth():
    t = Table.from_data([[1, 2], [3, 4]], max_depth=2)
    assert len(t) == 2
    assert isinstance(t[0], Table)
    assert t[0][0] == [1, 2]


def test_table_from_data_explicit_headers():
    t = Table.from_data([[1, 2]], headers=["a", "b"])
    assert t.headers == ["a", "b"]


# ---------------------------------------------------------------------------
# to_text rendering
# ---------------------------------------------------------------------------

def test_table_to_text_basic():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    out = t.to_text()
    lines = out.split("\n")
    assert lines[0].strip().startswith("a")
    assert "b" in lines[0]
    assert set(lines[1]) == {"-", "|"}
    assert lines[2].strip() == "1 | 2"
    assert lines[3].strip() == "3 | 4"


def test_table_to_text_no_headers():
    t = Table([[1, 2]])
    out = t.to_text()
    assert "1 | 2" in out
    # no separator line when there are no headers
    assert "|-" not in out


def test_table_to_text_with_headers_false():
    t = Table([[1, 2]], headers=["a", "b"])
    out = t.to_text(with_headers=False)
    assert "a" not in out
    assert "1 | 2" in out


def test_table_to_text_maxlen():
    t = Table([["abcdefghij"]], headers=["h"])
    out = t.to_text(maxlen=5)
    assert "ab..." in out


def test_table_to_text_empty():
    t = Table()
    assert t.to_text() == ""


# ---------------------------------------------------------------------------
# to_html rendering
# ---------------------------------------------------------------------------

def test_table_to_html_basic():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    out = t.to_html()
    assert out.startswith("<table><thead><tr><th>a</th><th>b</th></tr></thead>")
    assert "<tbody><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></tbody></table>" in out


def test_table_to_html_newlines():
    t = Table([[1]], headers=["h"])
    out = t.to_html(with_newlines=True)
    assert "\n" in out
    assert out.startswith("<table>\n")


def test_table_to_html_unwrapped():
    t = Table([[1, 2]])
    out = t.to_html(wrapped=False)
    assert not out.startswith("<table>")
    assert not out.endswith("</table>")


def test_table_to_html_no_headers():
    t = Table([[1, 2]])
    out = t.to_html(with_headers=False)
    assert "<thead>" not in out


def test_table_to_html_vertical():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    out = t.to_html(orientation="vertical")
    assert "<th>a</th>" in out
    assert "<th>b</th>" in out
    # vertical means one row per column
    assert out.count("<tr>") == 3  # header row + 2 data rows


def test_table_to_html_auto_single_row_vertical():
    t = Table([[1, 2]], headers=["a", "b"])
    out = t.to_html(orientation="auto")
    assert "<th>a</th>" in out


def test_table_to_html_auto_multi_row_horizontal():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    out = t.to_html(orientation="auto")
    assert out.count("<tr>") == 3  # 1 header + 2 data rows


def test_table_to_html_invalid_orientation():
    t = Table([[1]])
    with pytest.raises(ValueError):
        t.to_html(orientation="sideways")


def test_table_to_html_escaping():
    t = Table([["<script>"]])
    out = t.to_html()
    assert "&lt;script&gt;" in out
    assert "<script>" not in out


def test_table_to_html_metadata():
    t = Table([[1]], metadata={"k": "v"})
    out = t.to_html(with_metadata=True)
    assert "k" in out
    assert "v" in out


def test_table_to_html_metadata_bottom():
    t = Table([[1]], metadata={"k": "v"})
    out = t.to_html(with_metadata="bottom")
    # metadata appears after main table, before </table>
    idx_k = out.index("k")
    idx_table_close = out.rindex("</table>")
    assert idx_k < idx_table_close


def test_get_cell_html():
    t = Table()
    assert t.get_cell_html("<b>") == "&lt;b&gt;"


# ---------------------------------------------------------------------------
# Nested tables in HTML
# ---------------------------------------------------------------------------

def test_table_nested_html():
    inner = Table([[1, 2]])
    outer = Table([[inner]], headers=["c"])
    out = outer.to_html(max_depth=2)
    assert "<table>" in out
    assert "<td>1</td>" in out or "<td>2</td>" in out
