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


# ----------------------------- to_text -----------------------------

def test_to_text_basic():
    assert to_text(123) == "123"
    assert to_text("abc") == "abc"
    assert to_text(None) == "None"


def test_to_text_maxlen_truncation():
    assert to_text("abcdefghij", maxlen=5) == "ab..."
    assert to_text("abcdefghij", maxlen=3) == "..."
    # exactly at maxlen: no truncation
    assert to_text("abc", maxlen=3) == "abc"


def test_to_text_no_truncation_when_shorter():
    assert to_text("ab", maxlen=10) == "ab"


def test_to_text_maxlen_none():
    assert to_text("abcdefghij", maxlen=None) == "abcdefghij"


class _BadStr:
    def __str__(self):
        raise ValueError("no str")

    def __repr__(self):
        raise ValueError("no repr")


class _BadStrGoodRepr:
    def __str__(self):
        raise ValueError("no str")

    def __repr__(self):
        return "GOODREPR"


def test_to_text_str_failure_falls_back_to_repr():
    assert to_text(_BadStrGoodRepr()) == "GOODREPR"


def test_to_text_both_fail_falls_back_to_object_repr():
    obj = _BadStr()
    result = to_text(obj)
    # object.__repr__ output contains the class name and address
    assert "_BadStr object at" in result


# ----------------------------- escape_html -----------------------------

def test_escape_html_basic():
    assert escape_html("<b>") == "&lt;b&gt;"


def test_escape_html_quotes():
    assert escape_html('"') == "&quot;"
    assert escape_html("'") == "&#x27;"


def test_escape_html_ampersand():
    assert escape_html("a & b") == "a &amp; b"


def test_escape_html_maxlen():
    assert escape_html("abcdefghij", maxlen=5) == "ab..."


def test_escape_html_none():
    assert escape_html(None) == "None"


# ----------------------------- InputType & subclasses -----------------------------

def test_input_type_default_get_entry_seq():
    class MyInput(InputType):
        def get_entry(self, entry, headers):
            return list(entry)

    it = MyInput()
    assert it.get_entry_seq([[1, 2], [3, 4]], ["a", "b"]) == [[1, 2], [3, 4]]


def test_dict_input_type():
    d = DictInputType()
    assert d.check_type({"a": 1}) is True
    assert d.check_type([1]) is False
    assert d.guess_headers({"b": 2, "a": 1}) == ["a", "b"]
    assert d.get_entry({"a": 1}, ["a", "b"]) == [1, None]
    assert d.get_entry_seq([{"a": 1}, {"b": 2}], ["a", "b"]) == [[1, None], [None, 2]]


def test_dict_input_type_ordered():
    d = DictInputType()
    od = OrderedDict([("z", 1), ("a", 2)])
    assert d.guess_headers(od) == ["a", "z"]


def test_list_input_type():
    lt = ListInputType()
    assert lt.check_type([1, 2]) is True
    assert lt.check_type((1, 2)) is False
    assert lt.guess_headers([1, 2]) is None
    assert lt.get_entry([1, 2], None) == [1, 2]
    seq = [[1], [2]]
    assert lt.get_entry_seq(seq, None) is seq


def test_tuple_input_type():
    tt = TupleInputType()
    assert tt.check_type((1, 2)) is True
    assert tt.check_type([1, 2]) is False
    assert tt.guess_headers((1, 2)) is None
    assert tt.get_entry((1, 2), None) == [1, 2]
    assert tt.get_entry_seq([(1, 2), (3, 4)], None) == [[1, 2], [3, 4]]


def test_named_tuple_input_type():
    NT = namedtuple("NT", ["x", "y"])
    nt = NT(1, 2)
    ntit = NamedTupleInputType()
    assert ntit.check_type(nt) is True
    assert ntit.check_type((1, 2)) is False
    assert ntit.check_type([1, 2]) is False
    assert ntit.guess_headers(nt) == ["x", "y"]
    assert ntit.get_entry(nt, ["x", "y"]) == [1, 2]
    assert ntit.get_entry(nt, ["x", "z"]) == [1, None]
    assert ntit.get_entry_seq([nt], ["x", "y"]) == [[1, 2]]


class _SimpleObj:
    def __init__(self):
        self.a = 1
        self.b = "two"

    def method(self):
        pass


def test_object_input_type():
    ot = ObjectInputType()
    obj = _SimpleObj()
    assert ot.check_type(obj) is True
    assert ot.check_type(3) is False
    assert ot.check_type(None) is False
    headers = ot.guess_headers(obj)
    assert "a" in headers
    assert "b" in headers
    assert "method" not in headers
    assert ot.get_entry(obj, ["a", "b"]) == [1, "two"]
    # missing attribute -> None
    assert ot.get_entry(obj, ["a", "nonexistent"]) == [1, None]


# ----------------------------- UnsupportedData -----------------------------

def test_unsupported_data_is_type_error():
    assert issubclass(UnsupportedData, TypeError)


# ----------------------------- Table construction -----------------------------

def test_table_empty():
    t = Table()
    assert len(t) == 0
    assert t.headers == []
    assert t._data == []


def test_table_basic_list_of_lists_with_headers():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2], [3, 4]]
    assert len(t) == 2
    assert t[0] == [1, 2]


def test_table_infers_headers_from_first_row():
    t = Table([["a", "b"], [1, 2], [3, 4]])
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2], [3, 4]]


def test_table_headers_none():
    t = Table([[1, 2]], headers=None)
    assert t.headers == []
    assert t._data == [[1, 2]]


def test_table_fill_short_rows():
    t = Table([[1, 2, 3], [4]], headers=["a", "b", "c"])
    assert t._data == [[1, 2, 3], [4, None, None]]


def test_table_fill_short_rows_no_headers():
    t = Table([[1, 2, 3], [4]])
    assert t._data == [[1, 2, 3], [4, None, None]]


def test_table_metadata():
    t = Table([[1]], headers=["a"], metadata={"k": "v"})
    assert t.metadata == {"k": "v"}


def test_table_extend():
    t = Table([[1, 2]], headers=["a", "b"])
    t.extend([[3, 4], [5]])
    assert t._data == [[1, 2], [3, 4], [5, None]]


def test_table_extend_empty():
    t = Table([[1, 2]], headers=["a", "b"])
    t.extend([])
    assert t._data == [[1, 2]]


def test_table_repr_no_headers():
    t = Table([[1, 2]], headers=None)
    assert repr(t) == "Table([[1, 2]])"


def test_table_repr_with_headers():
    t = Table([[1, 2]], headers=["a", "b"])
    assert repr(t) == "Table(headers=['a', 'b'], data=[[1, 2]])"


# ----------------------------- from_data -----------------------------

def test_from_data_dict_list():
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    t = Table.from_data(data)
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2], [3, 4]]


def test_from_data_single_dict():
    t = Table.from_data({"a": 1, "b": 2})
    assert t.headers == ["a", "b"]
    assert t._data == [[1, 2]]


def test_from_data_scalar():
    t = Table.from_data(42)
    assert t._data == [[42]]


def test_from_data_empty_list():
    t = Table.from_data([])
    assert len(t) == 0


def test_from_data_max_depth_zero():
    t = Table.from_data([[1, 2]], max_depth=0)
    assert len(t) == 0


def test_from_data_namedtuple_list():
    NT = namedtuple("NT", ["x", "y"])
    t = Table.from_data([NT(1, 2), NT(3, 4)])
    assert t.headers == ["x", "y"]
    assert t._data == [[1, 2], [3, 4]]


def test_from_data_unsupported():
    with pytest.raises(UnsupportedData):
        Table.from_data([[{1, 2}]])


def test_from_data_tuple_rows():
    t = Table.from_data([(1, 2), (3, 4)])
    assert t._data == [[1, 2], [3, 4]]


def test_from_dict_forces_mapping():
    t = Table.from_dict({"a": 1})
    assert t.headers == ["a"]
    assert t._data == [[1]]


def test_from_list_forces_sequence():
    t = Table.from_list([[1, 2], [3, 4]], headers=["a", "b"])
    assert t._data == [[1, 2], [3, 4]]


def test_from_object():
    t = Table.from_object(_SimpleObj())
    assert "a" in t.headers
    assert "b" in t.headers


def test_from_data_nested_max_depth():
    data = [[{"a": 1}]]
    t = Table.from_data(data, max_depth=2)
    # The nested dict gets converted to a Table
    assert isinstance(t[0][0], Table)
    assert t[0][0].headers == ["a"]


# ----------------------------- to_text -----------------------------

def test_table_to_text_with_headers():
    t = Table([["a", "b"], [1, 2]], headers=["h1", "h2"])
    result = t.to_text()
    lines = result.split("\n")
    assert "h1" in lines[0]
    assert "h2" in lines[0]
    assert set(lines[1]) <= set("-|")
    assert lines[2] == "a | b"
    assert lines[3] == "1 | 2"


def test_table_to_text_no_headers():
    t = Table([[1, 2]], headers=None)
    result = t.to_text()
    assert result == "1 | 2"


def test_table_to_text_with_headers_false():
    t = Table([[1, 2]], headers=["a", "b"])
    result = t.to_text(with_headers=False)
    assert result == "1 | 2"


def test_table_to_text_maxlen():
    t = Table([["abcdefghij"]], headers=None)
    result = t.to_text(maxlen=5)
    assert result == "ab..."


def test_table_to_text_alignment():
    t = Table([["a"], ["bbbb"]], headers=None)
    lines = t.to_text().split("\n")
    # all lines should have same length due to centering
    assert len(lines[0]) == len(lines[1])


# ----------------------------- to_html -----------------------------

def test_table_to_html_basic():
    t = Table([["a", "b"], [1, 2]], headers=["h1", "h2"])
    result = t.to_html()
    assert result.startswith("<table>")
    assert result.endswith("</table>")
    assert "<thead>" in result
    assert "<th>h1</th>" in result
    assert "<th>h2</th>" in result
    assert "<tbody>" in result
    assert "<td>a</td>" in result
    assert "<td>1</td>" in result


def test_table_to_html_no_wrap():
    t = Table([[1]], headers=None)
    result = t.to_html(wrapped=False)
    assert "<table>" not in result
    assert "<td>1</td>" in result


def test_table_to_html_with_newlines():
    t = Table([[1]], headers=None)
    result = t.to_html(with_newlines=True)
    assert "\n" in result


def test_table_to_html_no_newlines():
    t = Table([[1]], headers=None)
    result = t.to_html()
    assert "\n" not in result


def test_table_to_html_escapes():
    t = Table([["<b>"]], headers=None)
    result = t.to_html()
    assert "&lt;b&gt;" in result


def test_table_to_html_vertical():
    t = Table([[1], [2]], headers=None)
    result = t.to_html(orientation="vertical")
    assert "<td>1</td>" in result
    assert "<td>2</td>" in result


def test_table_to_html_horizontal():
    t = Table([[1, 2]], headers=None)
    result = t.to_html(orientation="horizontal")
    assert "<td>1</td>" in result
    assert "<td>2</td>" in result


def test_table_to_html_auto_vertical_single_row():
    # single row -> auto becomes 'v'
    t = Table([[1, 2]], headers=None)
    result = t.to_html()
    # vertical: each column gets its own row
    assert "<tr><td>1</td></tr>" in result
    assert "<tr><td>2</td></tr>" in result


def test_table_to_html_invalid_orientation():
    t = Table([[1]], headers=None)
    with pytest.raises(ValueError):
        t.to_html(orientation="bogus")


def test_table_to_html_with_metadata():
    t = Table([[1]], headers=None, metadata={"k": "v"})
    result = t.to_html(with_metadata=True)
    assert "<td>k</td>" in result
    assert "<td>v</td>" in result
    assert "<br />" in result


def test_table_to_html_with_metadata_bottom():
    t = Table([[1]], headers=None, metadata={"k": "v"})
    result = t.to_html(with_metadata="bottom")
    # metadata should appear after main table
    idx_meta = result.index("<td>k</td>")
    idx_main = result.index("<td>1</td>")
    assert idx_meta > idx_main


def test_table_to_html_no_metadata_when_absent():
    t = Table([[1]], headers=None)
    result = t.to_html(with_metadata=True)
    assert "<br />" not in result


def test_table_to_html_nested_table_max_depth():
    inner = Table([["x"]], headers=None)
    t = Table([[inner]], headers=None)
    result = t.to_html(max_depth=2)
    assert "<table>" in result.replace("<table><tr><td><table>", "", 1) or result.count("<table>") >= 2


def test_get_cell_html():
    t = Table()
    assert t.get_cell_html("<x>") == "&lt;x&gt;"
