import types
from collections import namedtuple, OrderedDict

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


def test_to_text_maxlen_truncates():
    assert to_text("abcdefghij", maxlen=5) == "ab..."
    assert to_text("abcde", maxlen=5) == "abcde"
    assert to_text("abcdef", maxlen=5) == "ab..."


def test_to_text_maxlen_zero_is_falsy():
    # maxlen of 0 is falsy -> no truncation
    assert to_text("abcdef", maxlen=0) == "abcdef"


def test_to_text_broken_str_uses_repr():
    class BadStr:
        def __str__(self):
            raise ValueError("nope")

        def __repr__(self):
            return "BADREPR"

    assert to_text(BadStr()) == "BADREPR"


def test_to_text_broken_str_and_repr():
    class BadBoth:
        def __str__(self):
            raise ValueError("nope")

        def __repr__(self):
            raise ValueError("also nope")

    result = to_text(BadBoth())
    assert result.startswith("<") and "object at" in result


# ---------------------------------------------------------------------------
# escape_html
# ---------------------------------------------------------------------------

def test_escape_html_basic():
    assert escape_html("<a>&\"'") == "&lt;a&gt;&amp;&quot;&#x27;"


def test_escape_html_maxlen():
    assert escape_html("abcdefghij", maxlen=5) == "ab..."

    class Weird:
        def __str__(self):
            return "<x>"

    assert escape_html(Weird(), maxlen=10) == "&lt;x&gt;"


# ---------------------------------------------------------------------------
# InputType subclasses
# ---------------------------------------------------------------------------

def test_input_type_get_entry_seq():
    it = InputType()
    # default get_entry delegates to get_entry
    assert it.get_entry_seq([1, 2, 3], []) == [1, 2, 3]


def test_dict_input_type_check_type():
    dit = DictInputType()
    assert dit.check_type({}) is True
    assert dit.check_type(OrderedDict()) is True
    assert dit.check_type([]) is False
    assert dit.check_type("abc") is False


def test_dict_input_type_guess_headers_sorted():
    dit = DictInputType()
    assert dit.guess_headers({"b": 1, "a": 2}) == ["a", "b"]


def test_dict_input_type_get_entry():
    dit = DictInputType()
    assert dit.get_entry({"a": 1, "b": 2}, ["a", "b"]) == [1, 2]
    # missing keys -> None
    assert dit.get_entry({"a": 1}, ["a", "b"]) == [1, None]


def test_dict_input_type_get_entry_seq():
    dit = DictInputType()
    result = dit.get_entry_seq([{"a": 1, "b": 2}, {"a": 3}], ["a", "b"])
    assert result == [[1, 2], [3, None]]


def test_object_input_type_check_type():
    oit = ObjectInputType()

    class Foo:
        pass

    assert oit.check_type(Foo()) is True
    assert oit.check_type(123) is False
    assert oit.check_type("abc") is False
    assert oit.check_type(None) is False
    assert oit.check_type(True) is False


def test_object_input_type_guess_headers_skips_callables():
    oit = ObjectInputType()

    class Foo:
        x = 1

        def method(self):
            pass

    foo = Foo()
    headers = oit.guess_headers(foo)
    assert "x" in headers
    assert "method" not in headers


def test_object_input_type_guess_headers_property_raises():
    oit = ObjectInputType()

    class Foo:
        x = 1

        @property
        def bad(self):
            raise RuntimeError("boom")

    headers = oit.guess_headers(Foo())
    assert "x" in headers
    assert "bad" not in headers


def test_object_input_type_get_entry_missing_attr():
    oit = ObjectInputType()

    class Foo:
        x = 1

    assert oit.get_entry(Foo(), ["x", "y"]) == [1, None]


def test_list_input_type():
    lit = ListInputType()
    assert lit.check_type([1, 2]) is True
    assert lit.check_type((1, 2)) is False
    assert lit.guess_headers([1, 2]) is None
    assert lit.get_entry([1, 2], None) == [1, 2]
    assert lit.get_entry_seq([[1, 2]], None) == [[1, 2]]


def test_tuple_input_type():
    tit = TupleInputType()
    assert tit.check_type((1, 2)) is True
    assert tit.check_type([1, 2]) is False
    assert tit.guess_headers((1, 2)) is None
    assert tit.get_entry((1, 2), None) == [1, 2]
    assert tit.get_entry_seq([(1, 2), (3, 4)], None) == [[1, 2], [3, 4]]


def test_namedtuple_input_type():
    NT = namedtuple("NT", ["a", "b"])
    nit = NamedTupleInputType()
    nt = NT(1, 2)
    assert nit.check_type(nt) is True
    assert nit.check_type((1, 2)) is False
    assert nit.guess_headers(nt) == ["a", "b"]
    assert nit.get_entry(nt, ["a", "b"]) == [1, 2]
    assert nit.get_entry(nt, ["a", "x"]) == [1, None]
    assert nit.get_entry_seq([nt, NT(3, 4)], ["a", "b"]) == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# Table basics
# ---------------------------------------------------------------------------

def test_table_empty():
    t = Table()
    assert len(t) == 0
    assert t.headers == []
    assert t.metadata == {}
    assert repr(t) == "Table([])"


def test_table_with_headers():
    t = Table(headers=["a", "b"])
    assert t.headers == ["a", "b"]
    assert t._width == 2
    assert len(t) == 0
    assert repr(t) == "Table(headers=['a', 'b'], data=[])"


def test_table_init_from_data_with_headers():
    t = Table([["a", "b"], [1, 2], [3, 4]])
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2], [3, 4]]
    assert len(t) == 2


def test_table_init_with_explicit_headers():
    t = Table([[1, 2]], headers=["x", "y"])
    assert t.headers == ["x", "y"]
    assert t._data == [[1, 2]]


def test_table_getitem():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_table_metadata():
    t = Table(metadata={"k": "v"})
    assert t.metadata == {"k": "v"}
    t2 = Table()
    assert t2.metadata == {}


def test_table_fill_pads_short_rows():
    # width determined by headers
    t = Table([[1]], headers=["a", "b", "c"])
    assert t._data == [[1, None, None]]


def test_table_width_from_data():
    t = Table([[1, 2, 3], [4, 5]])
    assert t._width == 3
    assert t._data == [[1, 2, 3], [4, 5, None]]


def test_table_extend():
    t = Table(headers=["a", "b"])
    t.extend([[1, 2]])
    assert t._data == [[1, 2]]
    t.extend([[3]])
    assert t._data == [[1, 2], [3, None]]


def test_table_extend_empty_noop():
    t = Table(headers=["a"])
    t.extend([])
    assert t._data == []


def test_table_extend_none_noop():
    t = Table(headers=["a"])
    t.extend(None)
    assert t._data == []


# ---------------------------------------------------------------------------
# Table.from_data variants
# ---------------------------------------------------------------------------

def test_from_data_list_of_lists():
    t = Table.from_data([["h1", "h2"], [1, 2]], headers=["h1", "h2"])
    assert t.headers == ["h1", "h2"]
    assert t._data == [[1, 2]]


def test_from_data_list_guess_headers_none():
    t = Table.from_data([[1, 2], [3, 4]])
    assert t.headers == [] or t.headers is None or t.headers == []


def test_from_data_empty_sequence():
    t = Table.from_data([])
    assert len(t) == 0
    assert t.headers == []


def test_from_data_max_depth_zero():
    t = Table.from_data([1, 2, 3], max_depth=0)
    assert len(t) == 0
    assert t.headers == []


def test_from_data_dict():
    t = Table.from_data({"a": 1, "b": 2})
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2]]


def test_from_data_list_of_dicts():
    t = Table.from_data([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2], [3, 4]]


def test_from_data_scalar():
    t = Table.from_data(42)
    assert t._data == [[42]]


def test_from_data_scalar_string():
    t = Table.from_data("hello")
    assert t._data == [["hello"]]


def test_from_data_tuple():
    t = Table.from_data((1, 2, 3))
    assert t._data == [[1, 2, 3]]


def test_from_data_list_of_tuples():
    t = Table.from_data([(1, 2), (3, 4)])
    assert t._data == [[1, 2], [3, 4]]


def test_from_data_namedtuple():
    NT = namedtuple("NT", ["a", "b"])
    t = Table.from_data(NT(1, 2))
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2]]


def test_from_data_list_of_namedtuples():
    NT = namedtuple("NT", ["a", "b"])
    t = Table.from_data([NT(1, 2), NT(3, 4)])
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2], [3, 4]]


def test_from_data_object():
    class Foo:
        def __init__(self):
            self.x = 1
            self.y = 2

    t = Table.from_data(Foo())
    assert sorted(t.headers) == ["x", "y"]
    row = t._data[0]
    assert row[t.headers.index("x")] == 1
    assert row[t.headers.index("y")] == 2


def test_from_data_unsupported():
    class Weird:
        def __iter__(self):
            return iter([])

        __class__ = None  # force nothing to match

    # non-sequence, not in _DNR, no input type matches -> UnsupportedData
    with pytest.raises(UnsupportedData):
        Table.from_data(object.__new__(object))


def test_from_data_nested_max_depth():
    inner = [[1, 2]]
    t = Table.from_data([inner], max_depth=2)
    # inner nested Table should be created
    assert isinstance(t._data[0][0], Table)


def test_from_dict_forced():
    t = Table.from_dict({"a": 1})
    assert t.headers == ["a"]
    assert t._data == [[1]]


def test_from_list_forced():
    t = Table.from_list([[1, 2], [3, 4]])
    assert t._data == [[1, 2], [3, 4]]


def test_from_object_forced():
    class Foo:
        def __init__(self):
            self.a = 5

    t = Table.from_object(Foo())
    assert t.headers == ["a"]
    assert t._data == [[5]]


# ---------------------------------------------------------------------------
# to_text rendering
# ---------------------------------------------------------------------------

def test_to_text_simple():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    out = t.to_text()
    lines = out.split("\n")
    assert len(lines) == 4
    assert "a" in lines[0]
    assert "b" in lines[0]
    assert set(lines[1]) <= {"-", "|"}
    assert "1" in lines[2]
    assert "3" in lines[3]


def test_to_text_without_headers():
    t = Table([[1, 2]], headers=["a", "b"])
    out = t.to_text(with_headers=False)
    lines = out.split("\n")
    assert len(lines) == 1
    assert "1" in lines[0]


def test_to_text_maxlen():
    t = Table([["abcdefghij"]], headers=["h"])
    out = t.to_text(maxlen=5)
    assert "ab..." in out


def test_to_text_empty():
    t = Table()
    assert t.to_text() == ""


# ---------------------------------------------------------------------------
# to_html rendering
# ---------------------------------------------------------------------------

def test_to_html_horizontal():
    t = Table([[1, 2]], headers=["a", "b"])
    out = t.to_html()
    assert out.startswith("<table>")
    assert out.endswith("</table>")
    assert "<thead>" in out
    assert "<th>a</th>" in out
    assert "<th>b</th>" in out
    assert "<td>1</td>" in out
    assert "<td>2</td>" in out


def test_to_html_newlines():
    t = Table([[1]], headers=["a"])
    out = t.to_html(with_newlines=True)
    assert "\n" in out


def test_to_html_no_newlines():
    t = Table([[1]], headers=["a"])
    out = t.to_html(with_newlines=False)
    assert "\n" not in out


def test_to_html_unwrapped():
    t = Table([[1]], headers=["a"])
    out = t.to_html(wrapped=False)
    assert not out.startswith("<table>")
    assert "<td>1</td>" in out


def test_to_html_vertical():
    t = Table([[1, 2]], headers=["a", "b"])
    out = t.to_html(orientation="vertical")
    assert "<th>a</th>" in out
    assert "<th>b</th>" in out
    assert "<td>1</td>" in out
    assert "<td>2</td>" in out


def test_to_html_orientation_auto_single_row():
    t = Table([[1]], headers=["a"])
    # single row -> vertical
    out = t.to_html(orientation="auto")
    assert "<th>a</th>" in out


def test_to_html_orientation_auto_multi_row():
    t = Table([[1]], headers=["a"])
    t.extend([[2]])
    out = t.to_html(orientation="auto")
    assert "<tbody>" in out


def test_to_html_invalid_orientation():
    t = Table([[1]], headers=["a"])
    with pytest.raises(ValueError):
        t.to_html(orientation="zebra")


def test_to_html_escapes():
    t = Table([["<b>"]], headers=["<h>"])
    out = t.to_html()
    assert "<b>" not in out
    assert "&lt;b&gt;" in out
    assert "&lt;h&gt;" in out


def test_to_html_without_headers():
    t = Table([[1]])
    out = t.to_html()
    assert "<thead>" not in out
    assert "<td>1</td>" in out


def test_to_html_with_metadata_top():
    t = Table([[1]], headers=["a"], metadata={"k": "v"})
    out = t.to_html(with_metadata=True, with_newlines=True)
    assert "k" in out
    assert "<br />" in out
    # metadata table comes before main table
    assert out.index("k") < out.index("<th>a</th>")


def test_to_html_with_metadata_bottom():
    t = Table([[1]], headers=["a"], metadata={"k": "v"})
    out = t.to_html(with_metadata="bottom", with_newlines=True)
    assert out.index("<th>a</th>") < out.index("k")


def test_to_html_nested_table():
    inner = Table([[1]], headers=["x"])
    t = Table([[inner]], headers=["col"])
    out = t.to_html(max_depth=2)
    assert "<td>1</td>" in out


def test_to_html_no_recursion_depth_one():
    inner = Table([[1]], headers=["x"])
    t = Table([[inner]], headers=["col"])
    out = t.to_html(max_depth=1)
    # inner is rendered via escape_html -> repr-ish
    assert "<td>1</td>" not in out


def test_get_cell_html_override():
    class MyTable(Table):
        def get_cell_html(self, value):
            return "X"

    t = MyTable([[1]], headers=["a"])
    out = t.to_html()
    assert "<td>X</td>" in out
    assert "<th>X</th>" in out


# ---------------------------------------------------------------------------
# Table internals: _set_width / _fill
# ---------------------------------------------------------------------------

def test_set_width_reset():
    t = Table(headers=["a", "b"])
    assert t._width == 2
    t._width = 0
    t._set_width()
    assert t._width == 2


def test_set_width_from_data_no_headers():
    t = Table()
    t._data = [[1, 2, 3]]
    t._set_width()
    assert t._width == 3


def test_fill_no_width():
    t = Table()
    t._data = [[]]
    t._fill()
    assert t._data == [[]]


def test_fill_pads():
    t = Table()
    t._width = 3
    t._data = [[1]]
    t._fill()
    assert t._data == [[1, None, None]]


def test_fill_no_pad_when_long_enough():
    t = Table()
    t._width = 2
    t._data = [[1, 2, 3]]
    t._fill()
    assert t._data == [[1, 2, 3]]
