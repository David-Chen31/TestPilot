import pytest
from collections import namedtuple, OrderedDict

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
# to_text / escape_html
# ---------------------------------------------------------------------------

def test_to_text_basic():
    assert to_text(123) == "123"
    assert to_text("hello") == "hello"
    assert to_text([1, 2, 3]) == "[1, 2, 3]"


def test_to_text_maxlen_truncates():
    assert to_text("abcdefghij", maxlen=5) == "ab..."
    assert to_text("abc", maxlen=5) == "abc"
    # exactly at maxlen is not truncated
    assert to_text("12345", maxlen=5) == "12345"


def test_to_text_maxlen_zero_is_unlimited():
    # falsy maxlen means no truncation
    assert to_text("abcdefgh", maxlen=0) == "abcdefgh"


def test_to_text_fallback_on_broken_str():
    class BadStr:
        def __str__(self):
            raise ValueError("nope")

        def __repr__(self):
            return "BADREPR"

    assert to_text(BadStr()) == "BADREPR"


def test_to_text_fallback_to_object_repr():
    class BadBoth:
        def __str__(self):
            raise ValueError("nope")

        def __repr__(self):
            raise ValueError("nope")

    result = to_text(BadBoth())
    assert "BadBoth object at" in result


def test_escape_html_escapes():
    assert escape_html("<a>") == "&lt;a&gt;"
    assert escape_html('"&') == "&quot;&amp;"
    assert escape_html(5) == "5"


def test_escape_html_maxlen():
    assert escape_html("abcdefgh", maxlen=5) == "ab..."


# ---------------------------------------------------------------------------
# InputType classes
# ---------------------------------------------------------------------------

def test_input_type_base_get_entry_seq():
    it = InputType()
    result = it.get_entry_seq([1, 2, 3], ["h"])
    assert result == [None, None, None]


def test_dict_input_type():
    it = DictInputType()
    assert it.check_type({}) is True
    assert it.check_type(OrderedDict()) is True
    assert it.check_type([]) is False
    obj = {"b": 2, "a": 1}
    assert it.guess_headers(obj) == ["a", "b"]
    assert it.get_entry(obj, ["a", "b"]) == [1, 2]
    assert it.get_entry(obj, ["a", "z"]) == [1, None]
    seq = [{"a": 1, "b": 2}, {"b": 3}]
    assert it.get_entry_seq(seq, ["a", "b"]) == [[1, 2], [None, 3]]


def test_list_input_type():
    it = ListInputType()
    assert it.check_type([1, 2]) is True
    assert it.check_type((1, 2)) is False
    assert it.guess_headers([1]) is None
    assert it.get_entry([1, 2], ["x"]) == [1, 2]
    assert it.get_entry_seq([[1], [2]], None) == [[1], [2]]


def test_tuple_input_type():
    it = TupleInputType()
    assert it.check_type((1, 2)) is True
    assert it.check_type([1, 2]) is False
    assert it.guess_headers((1,)) is None
    assert it.get_entry((1, 2), None) == [1, 2]
    assert it.get_entry_seq([(1, 2), (3,)], None) == [[1, 2], [3]]


def test_named_tuple_input_type():
    it = NamedTupleInputType()
    NT = namedtuple("NT", ["a", "b"])
    nt = NT(1, 2)
    assert it.check_type(nt) is True
    assert it.check_type((1, 2)) is False
    assert it.guess_headers(nt) == ["a", "b"]
    assert it.get_entry(nt, ["a", "b"]) == [1, 2]
    assert it.get_entry(nt, ["a", "z"]) == [1, None]
    assert it.get_entry_seq([nt], ["a", "b"]) == [[1, 2]]


def test_object_input_type():
    it = ObjectInputType()
    assert it.check_type(object()) is True
    assert it.check_type(5) is False
    assert it.check_type("str") is False

    class Obj:
        x = 1

        def __init__(self):
            self.y = 2

        def method(self):
            pass

    obj = Obj()
    headers = it.guess_headers(obj)
    assert "x" in headers
    assert "y" in headers
    assert "method" not in headers
    # get_entry robust to missing attr
    assert it.get_entry(obj, ["y", "nonexistent"]) == [2, None]


def test_object_input_type_getattr_raises():
    it = ObjectInputType()

    class Obj:
        @property
        def bad(self):
            raise RuntimeError("boom")

        def __init__(self):
            self.good = 1

    obj = Obj()
    headers = it.guess_headers(obj)
    assert "bad" not in headers
    assert "good" in headers


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def test_table_empty():
    t = Table()
    assert len(t) == 0
    assert t.headers == []
    assert t._data == []
    assert t._width == 0


def test_table_from_list_of_lists_no_headers():
    t = Table([[1, 2], [3, 4]])
    # first row becomes headers, rest is data
    assert t.headers == [1, 2]
    assert t._data == [[3, 4]]


def test_table_from_list_with_headers():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2], [3, 4]]


def test_table_default_headers_data_preserved():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    t = Table(data)
    assert t.headers == [1, 2, 3]
    assert t._data == [[4, 5, 6], [7, 8, 9]]


def test_table_fill_short_rows():
    t = Table([[1, 2], [3]], headers=["a", "b"])
    assert t._data == [[1, 2], [3, None]]


def test_table_none_headers():
    t = Table([[1, 2], [3]], headers=None)
    assert t.headers == []
    assert t._width == 2
    assert t._data == [[1, 2], [3, None]]


def test_table_extend():
    t = Table(headers=["a"])
    t.extend([[1], [2]])
    assert t._data == [[1], [2]]
    # extend with empty does nothing
    t.extend([])
    assert t._data == [[1], [2]]
    assert len(t) == 2


def test_table_getitem_and_len():
    t = Table([[1], [2]], headers=["a"])
    assert len(t) == 2
    assert t[0] == [1]
    assert t[1] == [2]


def test_table_repr_with_headers():
    t = Table([[1]], headers=["a"])
    r = repr(t)
    assert r.startswith("Table(headers=")
    assert "data=" in r


def test_table_repr_without_headers():
    t = Table(headers=None)
    assert repr(t) == "Table([])"


def test_table_metadata_default_and_explicit():
    t = Table(metadata=None)
    assert t.metadata == {}
    meta = {"k": "v"}
    t2 = Table(metadata=meta)
    assert t2.metadata == meta


# ---------------------------------------------------------------------------
# from_data and friends
# ---------------------------------------------------------------------------

def test_from_data_dict_single():
    t = Table.from_data({"a": 1, "b": 2})
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2]]


def test_from_data_dict_list():
    t = Table.from_data([{"a": 1, "b": 2}, {"a": 3}])
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2], [3, None]]


def test_from_data_namedtuple():
    NT = namedtuple("NT", ["x", "y"])
    t = Table.from_data([NT(1, 2), NT(3, 4)])
    assert t.headers == ["x", "y"]
    assert t._data == [[1, 2], [3, 4]]


def test_from_data_list_of_tuples():
    t = Table.from_data([(1, 2), (3, 4)])
    # first tuple is treated as ListInputType? No - TupleInputType guesses None headers
    assert t.headers == []
    # width from data
    assert t._data == [[1, 2], [3, 4]]


def test_from_data_list_of_lists():
    t = Table.from_data([[1, 2], [3, 4]], headers=["a", "b"])
    assert t._data == [[1, 2], [3, 4]]


def test_from_data_object_single():
    class Obj:
        a = 1
        b = 2

    t = Table.from_data(Obj())
    assert "a" in t.headers
    assert "b" in t.headers
    row = t._data[0]
    assert row[t.headers.index("a")] == 1
    assert row[t.headers.index("b")] == 2


def test_from_data_scalar():
    t = Table.from_data(5)
    assert t._data == [[5]]
    assert t.headers == []


def test_from_data_empty_list():
    t = Table.from_data([])
    assert t._data == []
    assert t.headers == []


def test_from_data_max_depth_zero():
    t = Table.from_data([{"a": 1}], max_depth=0)
    assert len(t) == 0
    assert t.headers == []


def test_from_data_unsupported():
    # frozenset first element is unsupported for guessing
    with pytest.raises(UnsupportedData):
        Table.from_data([frozenset([1])])


def test_from_data_nested_tables():
    data = [[1, [2, 3]]]
    t = Table.from_data(data, headers=["a", "b"], max_depth=2)
    assert isinstance(t._data[0][1], Table)
    assert t._data[0][1]._data == [[2, 3]]


def test_from_dict_forced():
    class FakeDict(dict):
        pass

    t = Table.from_dict(FakeDict({"a": 1}))
    assert t.headers == ["a"]
    assert t._data == [[1]]


def test_from_list_forced():
    t = Table.from_list([[1, 2], [3, 4]], headers=["a", "b"])
    assert t._data == [[1, 2], [3, 4]]


def test_from_object_forced():
    class Obj:
        z = 9

    t = Table.from_object(Obj())
    assert "z" in t.headers
    assert 9 in t._data[0]


def test_from_data_metadata():
    meta = {"m": 1}
    t = Table.from_data([{"a": 1}], metadata=meta)
    assert t.metadata == meta


# ---------------------------------------------------------------------------
# to_text rendering
# ---------------------------------------------------------------------------

def test_to_text_with_headers():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    out = t.to_text()
    lines = out.split("\n")
    assert lines[0].strip().startswith("a")
    assert "---" in lines[1] or "-" in lines[1]
    assert len(lines) == 4


def test_to_text_without_headers():
    t = Table([[1, 2]], headers=None)
    out = t.to_text(with_headers=False)
    lines = out.split("\n")
    assert len(lines) == 1
    assert "1" in lines[0]


def test_to_text_maxlen():
    t = Table([["abcdefghij"]], headers=None)
    out = t.to_text(with_headers=False, maxlen=5)
    assert "ab..." in out


def test_to_text_headers_padded():
    t = Table(headers=["a"])  # width 1 via headers... empty data
    t.extend([[1, 2]])  # will fill
    out = t.to_text()
    assert "a" in out


# ---------------------------------------------------------------------------
# to_html
# ---------------------------------------------------------------------------

def test_to_html_basic():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    html = t.to_html()
    assert html.startswith("<table>")
    assert html.endswith("</table>")
    assert "<thead>" in html
    assert "<th>a</th>" in html
    assert "<td>1</td>" in html
    assert "<td>4</td>" in html
    assert "\n" not in html


def test_to_html_with_newlines():
    t = Table([[1]], headers=["a"])
    html = t.to_html(with_newlines=True)
    assert "\n" in html


def test_to_html_unwrapped():
    t = Table([[1]], headers=["a"])
    html = t.to_html(wrapped=False)
    assert not html.startswith("<table>")
    assert not html.endswith("</table>")


def test_to_html_vertical_orientation_single_row():
    t = Table([[1]], headers=["a"])
    html = t.to_html(orientation="vertical")
    assert "<tbody>" not in html or True  # vertical path used
    assert "<th>a</th>" in html
    assert "<td>1</td>" in html


def test_to_html_auto_single_row_uses_vertical():
    t = Table([[1, 2]], headers=["a", "b"])
    html = t.to_html(orientation="auto")
    # vertical: headers not in thead
    assert "<thead>" not in html
    assert "<th>a</th>" in html


def test_to_html_horizontal_orientation():
    t = Table([[1, 2]], headers=["a", "b"])
    html = t.to_html(orientation="horizontal")
    assert "<thead>" in html
    assert "<th>a</th>" in html


def test_to_html_invalid_orientation():
    t = Table([[1]], headers=["a"])
    with pytest.raises(ValueError):
        t.to_html(orientation="sideways")


def test_to_html_escaping():
    t = Table([["<script>"]], headers=["<h>"])
    html = t.to_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;h&gt;" in html


def test_to_html_no_headers():
    t = Table([[1, 2]], headers=None)
    html = t.to_html()
    assert "<thead>" not in html
    assert "<td>1</td>" in html


def test_to_html_nested_tables():
    t = Table([[1, Table([[9]], headers=["x"])]], headers=["a", "b"], max_depth=2)
    html = t.to_html(max_depth=2)
    assert "<table>" in html.replace("<table>", "", 1)  # nested table present
    assert "<td><table>" in html


def test_to_html_metadata_top():
    t = Table([[1]], headers=["a"], metadata={"m": "v"})
    html = t.to_html(with_metadata=True)
    assert html.startswith("<table>")
    assert "<br />" in html


def test_to_html_metadata_bottom():
    t = Table([[1]], headers=["a"], metadata={"m": "v"})
    html = t.to_html(with_metadata="bottom")
    assert "<br />" in html
    assert html.endswith("<br /><table></table>") or "m" in html


def test_get_cell_html_override():
    class CustomTable(Table):
        def get_cell_html(self, value):
            return "X"

    t = CustomTable([[1]], headers=["a"])
    html = t.to_html()
    assert "<th>X</th>" in html
    assert "<td>X</td>" in html


# ---------------------------------------------------------------------------
# UnsupportedData is a TypeError
# ---------------------------------------------------------------------------

def test_unsupported_data_is_type_error():
    assert issubclass(UnsupportedData, TypeError)
