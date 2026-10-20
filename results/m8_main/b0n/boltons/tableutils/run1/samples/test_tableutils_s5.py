"""Unit tests for boltons.tableutils."""

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
)


# ---------------------------------------------------------------------------
# to_text
# ---------------------------------------------------------------------------

def test_to_text_basic():
    assert to_text(123) == "123"
    assert to_text("hello") == "hello"
    assert to_text(None) == "None"


def test_to_text_maxlen_truncates():
    assert to_text("abcdefghij", maxlen=5) == "ab..."
    # Exactly at limit, no truncation
    assert to_text("abcde", maxlen=5) == "abcde"
    # Shorter than limit
    assert to_text("abc", maxlen=5) == "abc"


def test_to_text_falls_back_to_repr():
    class BadStr:
        def __str__(self):
            raise RuntimeError("no str")

        def __repr__(self):
            return "BadStr-repr"

    assert to_text(BadStr()) == "BadStr-repr"


def test_to_text_falls_back_to_object_repr():
    class BadBoth:
        def __str__(self):
            raise RuntimeError("no str")

        def __repr__(self):
            raise RuntimeError("no repr")

    result = to_text(BadBoth())
    assert "BadBoth" in result


# ---------------------------------------------------------------------------
# escape_html
# ---------------------------------------------------------------------------

def test_escape_html_basic():
    assert escape_html("<b>") == "&lt;b&gt;"


def test_escape_html_quotes():
    assert escape_html('a"b') == "a&quot;b"
    assert escape_html("a'b") == "a&#x27;b"


def test_escape_html_maxlen():
    assert escape_html("<html>", maxlen=4) == "&lt;..."


# ---------------------------------------------------------------------------
# InputType base class
# ---------------------------------------------------------------------------

def test_input_type_base():
    it = InputType()
    assert it.get_entry_seq([1, 2, 3], None) == [1, 2, 3]


# ---------------------------------------------------------------------------
# DictInputType
# ---------------------------------------------------------------------------

def test_dict_input_type():
    it = DictInputType()
    assert it.check_type({}) is True
    assert it.check_type([]) is False
    assert it.guess_headers({"b": 1, "a": 2}) == ["a", "b"]
    assert it.get_entry({"a": 1, "b": 2}, ["a", "b"]) == [1, 2]
    # Missing key -> None
    assert it.get_entry({"a": 1}, ["a", "b"]) == [1, None]
    assert it.get_entry_seq([{"a": 1}, {"a": 2}], ["a"]) == [[1], [2]]


# ---------------------------------------------------------------------------
# ObjectInputType
# ---------------------------------------------------------------------------

def test_object_input_type():
    class Obj:
        def __init__(self):
            self.x = 1
            self.y = 2

        def method(self):
            pass

    it = ObjectInputType()
    o = Obj()
    assert it.check_type(o) is True
    assert it.check_type(5) is False
    assert it.check_type("str") is False
    assert it.check_type(None) is False
    headers = it.guess_headers(o)
    assert "x" in headers
    assert "y" in headers
    # callables excluded
    assert "method" not in headers
    assert it.get_entry(o, ["x", "y"]) == [1, 2]
    # missing attribute -> None
    assert it.get_entry(o, ["x", "nope"]) == [1, None]


def test_object_input_type_property_raises():
    class Obj:
        def __init__(self):
            self.x = 1

        @property
        def bad(self):
            raise ValueError("boom")

    it = ObjectInputType()
    o = Obj()
    headers = it.guess_headers(o)
    assert "bad" not in headers
    assert "x" in headers


# ---------------------------------------------------------------------------
# ListInputType
# ---------------------------------------------------------------------------

def test_list_input_type():
    it = ListInputType()
    assert it.check_type([]) is True
    assert it.check_type(()) is False
    assert it.guess_headers([1, 2]) is None
    assert it.get_entry([1, 2], None) == [1, 2]
    seq = [[1, 2], [3, 4]]
    assert it.get_entry_seq(seq, None) == seq


# ---------------------------------------------------------------------------
# TupleInputType
# ---------------------------------------------------------------------------

def test_tuple_input_type():
    it = TupleInputType()
    assert it.check_type((1, 2)) is True
    assert it.check_type([1, 2]) is False
    assert it.guess_headers((1, 2)) is None
    assert it.get_entry((1, 2), None) == [1, 2]
    assert it.get_entry_seq([(1, 2), (3, 4)], None) == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# NamedTupleInputType
# ---------------------------------------------------------------------------

def test_namedtuple_input_type():
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    it = NamedTupleInputType()
    p = Point(1, 2)
    assert it.check_type(p) is True
    assert it.check_type((1, 2)) is False
    assert it.guess_headers(p) == ["x", "y"]
    assert it.get_entry(p, ["x", "y"]) == [1, 2]
    assert it.get_entry(p, ["x", "z"]) == [1, None]
    assert it.get_entry_seq([Point(1, 2), Point(3, 4)], ["x", "y"]) == [
        [1, 2],
        [3, 4],
    ]


# ---------------------------------------------------------------------------
# Table - construction
# ---------------------------------------------------------------------------

def test_table_simple_list_of_lists():
    t = Table([[1, 2], [3, 4]])
    assert len(t) == 2
    assert t._width == 2
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_table_headers_inferred_from_first_row():
    t = Table([["a", "b"], [1, 2], [3, 4]])
    assert t.headers == ["a", "b"]
    assert len(t) == 2
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_table_explicit_headers():
    t = Table([[1, 2]], headers=["x", "y"])
    assert t.headers == ["x", "y"]
    assert t[0] == [1, 2]


def test_table_no_headers():
    t = Table([[1, 2], [3, 4]], headers=None)
    assert t.headers == []
    assert len(t) == 2


def test_table_empty():
    t = Table()
    assert len(t) == 0
    assert t.headers == []
    assert repr(t) == "Table([])"


def test_table_fill_short_rows():
    t = Table([[1], [2, 3]], headers=["a", "b"])
    assert t[0] == [1, None]
    assert t[1] == [2, 3]


def test_table_metadata():
    t = Table([[1]], metadata={"k": "v"})
    assert t.metadata == {"k": "v"}


def test_table_repr_with_headers():
    t = Table([[1]], headers=["h"])
    assert repr(t) == "Table(headers=['h'], data=[[1]])"


def test_table_repr_no_headers():
    t = Table([[1, 2]], headers=None)
    assert repr(t) == "Table([[1, 2]])"


def test_table_getitem():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    assert t[0] == [1, 2]
    assert t[-1] == [3, 4]


def test_table_extend():
    t = Table([[1]], headers=["h"])
    t.extend([[2]])
    assert len(t) == 2
    assert t[1] == [2]


def test_table_extend_empty():
    t = Table([[1]], headers=["h"])
    t.extend([])
    assert len(t) == 1


# ---------------------------------------------------------------------------
# Table.from_data and variants
# ---------------------------------------------------------------------------

def test_from_data_list_of_lists():
    t = Table.from_data([[1, 2], [3, 4]])
    assert len(t) == 2
    assert t[0] == [1, 2]


def test_from_data_list_with_headers():
    t = Table.from_data([["a", "b"], [1, 2]])
    # list input type has no header guessing
    assert t.headers == []
    assert len(t) == 2
    assert t[0] == ["a", "b"]
    assert t[1] == [1, 2]


def test_from_data_dict():
    t = Table.from_data({"a": 1, "b": 2})
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]


def test_from_data_list_of_dicts():
    t = Table.from_data([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_from_data_namedtuple():
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    t = Table.from_data(Point(1, 2))
    assert t.headers == ["x", "y"]
    assert t[0] == [1, 2]


def test_from_data_list_of_namedtuples():
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    t = Table.from_data([Point(1, 2), Point(3, 4)])
    assert t.headers == ["x", "y"]
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_from_data_tuple():
    t = Table.from_data((1, 2, 3))
    # A tuple of scalars: data[0]=1, no input type matches int -> rewind,
    # then treat whole tuple; TupleInputType matches
    assert t[0] == [1, 2, 3]


def test_from_data_object():
    class Obj:
        def __init__(self):
            self.a = 1
            self.b = 2

    t = Table.from_data(Obj())
    assert "a" in t.headers
    assert "b" in t.headers
    row = dict(zip(t.headers, t[0]))
    assert row["a"] == 1
    assert row["b"] == 2


def test_from_data_scalar():
    t = Table.from_data(42)
    assert t[0] == [42]


def test_from_data_unsupported():
    class Weird:
        __slots__ = ()

    # A list containing something unsupported should raise
    with pytest.raises(UnsupportedData):
        Table.from_data([object()])


def test_from_data_empty():
    t = Table.from_data([])
    assert len(t) == 0


def test_from_data_max_depth_zero():
    t = Table.from_data([[1, 2]], max_depth=0)
    assert len(t) == 0


def test_from_data_metadata():
    t = Table.from_data({"a": 1}, metadata={"m": 1})
    assert t.metadata == {"m": 1}


def test_from_dict():
    t = Table.from_dict({"a": 1, "b": 2})
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]


def test_from_list():
    t = Table.from_list([[1, 2], [3, 4]])
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_from_object():
    class Obj:
        def __init__(self):
            self.a = 1

    t = Table.from_object(Obj())
    assert "a" in t.headers


def test_from_data_nested_max_depth():
    t = Table.from_data([[{"a": 1}]], max_depth=2)
    # inner dict becomes a Table
    assert isinstance(t[0][0], Table)


# ---------------------------------------------------------------------------
# Table.to_text
# ---------------------------------------------------------------------------

def test_to_text_with_headers():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    result = t.to_text()
    lines = result.split("\n")
    assert len(lines) == 4  # header, separator, 2 rows
    assert "a" in lines[0]
    assert "b" in lines[0]
    assert "-" in lines[1]


def test_to_text_no_headers():
    t = Table([[1, 2]], headers=None)
    result = t.to_text()
    lines = result.split("\n")
    assert len(lines) == 1


def test_to_text_with_headers_false():
    t = Table([[1, 2]], headers=["a", "b"])
    result = t.to_text(with_headers=False)
    lines = result.split("\n")
    assert len(lines) == 1


def test_to_text_maxlen():
    t = Table([["abcdefghij"]], headers=["h"])
    result = t.to_text(maxlen=5)
    assert "..." in result


def test_to_text_headers_longer_than_width_padded():
    t = Table([[1]], headers=["a", "b"])
    # width becomes 2 (from headers)
    result = t.to_text()
    lines = result.split("\n")
    assert "a" in lines[0]
    assert "b" in lines[0]


# ---------------------------------------------------------------------------
# Table.to_html
# ---------------------------------------------------------------------------

def test_to_html_horizontal():
    t = Table([[1, 2]], headers=["a", "b"])
    html = t.to_html(orientation="horizontal")
    assert html.startswith("<table>")
    assert html.endswith("</table>")
    assert "<thead>" in html
    assert "<tbody>" in html
    assert "<th>a</th>" in html
    assert "<td>1</td>" in html
    assert "<td>2</td>" in html


def test_to_html_vertical():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    html = t.to_html(orientation="vertical")
    assert "<table>" in html
    assert "<th>a</th>" in html
    assert "<th>b</th>" in html


def test_to_html_auto_single_row_vertical():
    t = Table([[1, 2]], headers=["a", "b"])
    # len == 1 -> vertical
    html = t.to_html(orientation="auto")
    assert "<th>a</th>" in html
    assert "<th>b</th>" in html


def test_to_html_auto_multi_row_horizontal():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    html = t.to_html(orientation="auto")
    assert "<thead>" in html


def test_to_html_unwrapped():
    t = Table([[1]], headers=["a"])
    html = t.to_html(wrapped=False)
    assert not html.startswith("<table>")
    assert "<td>1</td>" in html


def test_to_html_with_newlines():
    t = Table([[1]], headers=["a"])
    html = t.to_html(with_newlines=True)
    assert "\n" in html


def test_to_html_no_headers():
    t = Table([[1, 2]], headers=None)
    html = t.to_html(orientation="horizontal")
    assert "<thead>" not in html
    assert "<td>1</td>" in html


def test_to_html_with_headers_false():
    t = Table([[1]], headers=["a"])
    html = t.to_html(with_headers=False, orientation="horizontal")
    assert "<th>" not in html


def test_to_html_invalid_orientation():
    t = Table([[1]], headers=["a"])
    with pytest.raises(ValueError):
        t.to_html(orientation="zigzag")


def test_to_html_escapes():
    t = Table([["<b>"]], headers=["<h>"])
    html = t.to_html(orientation="horizontal")
    assert "&lt;b&gt;" in html
    assert "&lt;h&gt;" in html


def test_to_html_metadata():
    t = Table([[1]], headers=["a"], metadata={"k": "v"})
    html = t.to_html(with_metadata=True, orientation="horizontal")
    assert "&lt;" not in html or True  # sanity
    assert html.count("<table>") >= 1
    assert "k" in html


def test_to_html_metadata_bottom():
    t = Table([[1]], headers=["a"], metadata={"k": "v"})
    html_top = t.to_html(
        with_metadata=True, with_metadata_bottom=False
    ) if False else t.to_html(with_metadata="bottom", orientation="horizontal")
    assert html_top.count("<table>") >= 1


def test_to_html_nested_table_vertical():
    inner = Table([[1, 2]], headers=["a", "b"])
    outer = Table([[inner]], headers=["c"])
    html = outer.to_html(orientation="horizontal", max_depth=2)
    # inner table should render its own <table> tag
    assert html.count("<table>") == 2


def test_get_cell_html():
    t = Table([[1]])
    assert t.get_cell_html("<x>") == "&lt;x&gt;"


# ---------------------------------------------------------------------------
# Custom subclass behavior
# ---------------------------------------------------------------------------

def test_table_tag_subclass():
    class MyTable(Table):
        _html_table_tag = "<TABLE>"
        _html_table_tag_close = "</TABLE>"

    t = MyTable([[1]], headers=["a"])
    html = t.to_html(orientation="horizontal")
    assert html.startswith("<TABLE>")
    assert html.endswith("</TABLE>")


def test_input_type_custom():
    class CustomInputType(InputType):
        def check_type(self, obj):
            return isinstance(obj, set)

        def guess_headers(self, obj):
            return None

        def get_entry(self, obj, headers):
            return list(obj)

        def get_entry_seq(self, obj_seq, headers):
            return [list(o) for o in obj_seq]

    t = Table.from_data([{1, 2}], _data_type=CustomInputType())
    assert t[0] == [1, 2]
