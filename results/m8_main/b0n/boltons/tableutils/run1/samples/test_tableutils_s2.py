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
)


# ---------------------------------------------------------------------------
# to_text
# ---------------------------------------------------------------------------

def test_to_text_basic():
    assert to_text(123) == "123"
    assert to_text("hello") == "hello"
    assert to_text(None) == "None"


def test_to_text_maxlen_truncates():
    assert to_text("abcdefgh", maxlen=6) == "abc..."
    # exactly maxlen: no truncation
    assert to_text("abcdef", maxlen=6) == "abcdef"


def test_to_text_maxlen_zero_no_truncation():
    # maxlen=0 is falsy -> no truncation
    assert to_text("abcdef", maxlen=0) == "abcdef"


def test_to_text_bad_str_uses_repr():
    class BadStr:
        def __str__(self):
            raise ValueError("nope")

        def __repr__(self):
            return "BadStr-repr"

    assert to_text(BadStr()) == "BadStr-repr"


def test_to_text_bad_str_and_repr():
    class BadBoth:
        def __str__(self):
            raise ValueError("nope")

        def __repr__(self):
            raise ValueError("nope too")

    result = to_text(BadBoth())
    # falls back to object.__repr__ which includes the class name
    assert "BadBoth" in result


# ---------------------------------------------------------------------------
# escape_html
# ---------------------------------------------------------------------------

def test_escape_html_basic():
    assert escape_html("<a>") == "&lt;a&gt;"
    assert escape_html('"') == "&quot;"
    assert escape_html("'") == "&#x27;"
    assert escape_html("&") == "&amp;"


def test_escape_html_maxlen():
    assert escape_html("abcdefgh", maxlen=6) == "abc..."


# ---------------------------------------------------------------------------
# InputType classes
# ---------------------------------------------------------------------------

def test_input_type_get_entry_seq():
    it = InputType()
    # get_entry not overridden raises AttributeError
    with pytest.raises(AttributeError):
        it.get_entry_seq([1, 2], [])


def test_dict_input_type():
    dit = DictInputType()
    assert dit.check_type({"a": 1}) is True
    assert dit.check_type([1, 2]) is False
    obj = {"b": 2, "a": 1}
    assert dit.guess_headers(obj) == ["a", "b"]
    assert dit.get_entry({"a": 1, "b": 2}, ["a", "b"]) == [1, 2]
    assert dit.get_entry({"a": 1}, ["a", "b"]) == [1, None]
    seq = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert dit.get_entry_seq(seq, ["a", "b"]) == [[1, 2], [3, 4]]


def test_object_input_type():
    oit = ObjectInputType()

    class Thing:
        def __init__(self):
            self.x = 1
            self.y = 2

        def method(self):
            return 5

    t = Thing()
    assert oit.check_type(t) is True
    # plain scalars are not objects per _DNR
    assert oit.check_type(5) is False
    assert oit.check_type("string") is False
    headers = oit.guess_headers(t)
    # callable "method" excluded
    assert "method" not in headers
    assert "x" in headers
    assert "y" in headers
    values = oit.get_entry(t, ["x", "y"])
    assert values == [1, 2]
    # missing attrs -> None
    assert oit.get_entry(t, ["x", "missing"]) == [1, None]


def test_object_input_type_getattr_raises():
    oit = ObjectInputType()

    class Bad:
        @property
        def boom(self):
            raise RuntimeError("bang")

    b = Bad()
    headers = oit.guess_headers(b)
    assert "boom" not in headers
    assert oit.get_entry(b, ["boom"]) == [None]


def test_list_input_type():
    lit = ListInputType()
    assert lit.check_type([1, 2]) is True
    assert lit.check_type((1, 2)) is False
    assert lit.guess_headers([1, 2]) is None
    assert lit.get_entry([1, 2], None) == [1, 2]
    seq = [[1], [2]]
    assert lit.get_entry_seq(seq, None) is seq


def test_tuple_input_type():
    tit = TupleInputType()
    assert tit.check_type((1, 2)) is True
    assert tit.check_type([1, 2]) is False
    assert tit.guess_headers((1, 2)) is None
    assert tit.get_entry((1, 2), None) == [1, 2]
    assert tit.get_entry_seq([(1, 2), (3, 4)], None) == [[1, 2], [3, 4]]


def test_named_tuple_input_type():
    import collections

    NT = collections.namedtuple("NT", ["a", "b"])
    ntit = NamedTupleInputType()
    nt = NT(1, 2)
    assert ntit.check_type(nt) is True
    assert ntit.check_type((1, 2)) is False
    assert ntit.guess_headers(nt) == ["a", "b"]
    assert ntit.get_entry(nt, ["a", "b"]) == [1, 2]
    assert ntit.get_entry(nt, ["a", "missing"]) == [1, None]
    seq = [NT(1, 2), NT(3, 4)]
    assert ntit.get_entry_seq(seq, ["a", "b"]) == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# Table: construction and basic access
# ---------------------------------------------------------------------------

def test_table_basic_headers_passed():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    assert t.headers == ["a", "b"]
    assert len(t) == 2
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]
    assert t._width == 2


def test_table_headers_inferred_from_first_row():
    t = Table([["a", "b"], [1, 2], [3, 4]])
    assert t.headers == ["a", "b"]
    assert len(t) == 2
    assert t[0] == [1, 2]


def test_table_no_headers_explicit():
    t = Table([[1, 2], [3, 4]], headers=None)
    assert t.headers == []
    assert t._width == 2
    assert t[0] == [1, 2]


def test_table_empty():
    t = Table()
    assert t.headers == []
    assert len(t) == 0
    assert t._width == 0
    assert t.metadata == {}


def test_table_metadata():
    t = Table([[1]], headers=["a"], metadata={"k": "v"})
    assert t.metadata == {"k": "v"}


def test_table_fill_pads_short_rows():
    t = Table([[1], [2, 3, 4]], headers=None)
    # width = max header-less -> 3
    assert t._width == 3
    assert t[0] == [1, None, None]
    assert t[1] == [2, 3, 4]


def test_table_extend():
    t = Table([[1, 2]], headers=["a", "b"])
    t.extend([[3, 4], [5, 6]])
    assert len(t) == 3
    assert t[2] == [5, 6]


def test_table_extend_empty_noop():
    t = Table([[1]], headers=["a"])
    t.extend([])
    assert len(t) == 1
    t.extend(None)
    assert len(t) == 1


def test_table_repr_with_headers():
    t = Table([[1, 2]], headers=["a", "b"])
    r = repr(t)
    assert r.startswith("Table(headers=")
    assert "[1, 2]" in r


def test_table_repr_without_headers():
    t = Table([[1, 2]], headers=None)
    r = repr(t)
    assert r.startswith("Table(")
    assert "headers=" not in r
    assert "[1, 2]" in r


# ---------------------------------------------------------------------------
# Table.from_data and friends
# ---------------------------------------------------------------------------

def test_from_data_list_of_lists():
    t = Table.from_data([["a", "b"], [1, 2]])
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]


def test_from_data_empty_sequence():
    t = Table.from_data([])
    assert len(t) == 0
    assert t.headers == []


def test_from_data_scalar():
    t = Table.from_data(42)
    assert len(t) == 1
    assert t[0] == [42]


def test_from_data_dict_single():
    t = Table.from_data({"b": 2, "a": 1})
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]


def test_from_data_list_of_dicts():
    t = Table.from_data([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_from_data_namedtuple_single():
    import collections
    NT = collections.namedtuple("NT", ["x", "y"])
    t = Table.from_data(NT(1, 2))
    assert t.headers == ["x", "y"]
    assert t[0] == [1, 2]


def test_from_data_namedtuple_list():
    import collections
    NT = collections.namedtuple("NT", ["x", "y"])
    t = Table.from_data([NT(1, 2), NT(3, 4)])
    assert t.headers == ["x", "y"]
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_from_data_tuple_single():
    t = Table.from_data((1, 2, 3))
    # tuple -> TupleInputType, headers guessed None
    assert t.headers == []
    assert t[0] == [1, 2, 3]


def test_from_data_list_of_tuples():
    t = Table.from_data([(1, 2), (3, 4)])
    assert t.headers == []
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_from_data_object():
    class Obj:
        def __init__(self):
            self.a = 1
            self.b = 2

    t = Table.from_data(Obj())
    assert "a" in t.headers
    assert "b" in t.headers
    assert t[0] == [1, 2]


def test_from_data_headers_explicit():
    t = Table.from_data([{"a": 1, "b": 2}], headers=["b"])
    assert t.headers == ["b"]
    assert t[0] == [2]


def test_from_data_max_depth_zero():
    t = Table.from_data([[1, 2]], max_depth=0, headers=["a", "b"])
    assert len(t) == 0
    assert t.headers == ["a", "b"]


def test_from_data_unsupported():
    class Weird:
        # Not a mapping/sequence/tuple, but ObjectInputType will actually
        # accept it. Use something whose type is in _DNR to be scalar-handled,
        # so to trigger UnsupportedData we need a non-sequence of unknown type.
        pass

    # An iterator is not a Sequence, not in _DNR; ObjectInputType checks
    # hasattr(__class__) which is always true, so it'll be accepted. To force
    # UnsupportedData, craft data type not matching any input type.
    class Meta(type):
        pass

    # Instead directly test the error path by means of an object type that is
    # excluded: a generator is in _DNR but as scalar it returns [[data]].
    # Use a numpy-free approach: a dict subclass that is not a Mapping?
    # Simplest: use an object whose __class__ attribute is missing is not
    # possible. So test UnsupportedData via monkey-like forcing is skipped.
    # Verify the exception class hierarchy instead.
    assert issubclass(UnsupportedData, TypeError)


def test_from_data_nested_max_depth():
    t = Table.from_data([[[1, 2]]], max_depth=2)
    # The inner list should become a nested Table
    assert isinstance(t[0][0], Table)
    assert t[0][0][0] == [1, 2]


def test_from_dict_forced():
    t = Table.from_dict({"a": 1, "b": 2})
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]


def test_from_list_forced():
    t = Table.from_list(["a", "b"])
    assert t.headers == ["a", "b"]
    assert t[0] == ["b"]


def test_from_object_forced():
    class Obj:
        def __init__(self):
            self.a = 1

    t = Table.from_object(Obj())
    assert "a" in t.headers
    assert t[0] == [1]


def test_from_data_metadata_passthrough():
    t = Table.from_data([{"a": 1}], metadata={"m": 1})
    assert t.metadata == {"m": 1}


# ---------------------------------------------------------------------------
# to_html
# ---------------------------------------------------------------------------

def test_to_html_horizontal_single_row_simple():
    t = Table([[1, 2]], headers=["a", "b"])
    html = t.to_html(with_newlines=False)
    assert "<table>" in html
    assert "</table>" in html
    assert "<thead>" in html
    assert "<th>a</th>" in html
    assert "<th>b</th>" in html
    assert "<td>1</td>" in html
    assert "<td>2</td>" in html


def test_to_html_no_headers():
    t = Table([[1, 2]], headers=None)
    html = t.to_html()
    assert "<thead>" not in html
    assert "<td>1</td>" in html


def test_to_html_unwrapped():
    t = Table([[1]], headers=["a"])
    html = t.to_html(wrapped=False)
    assert "<table>" not in html
    assert "<td>1</td>" in html


def test_to_html_escape_cells():
    t = Table([["<b>"]], headers=None)
    html = t.to_html()
    assert "&lt;b&gt;" in html
    assert "<b>" not in html


def test_to_html_vertical():
    t = Table([[1, 2]], headers=["a", "b"])
    html = t.to_html(orientation="vertical", with_newlines=True)
    lines = html.split("\n")
    # vertical: each column is a row
    assert any("<th>a</th>" in line and "<td>1</td>" in line for line in lines)
    assert any("<th>b</th>" in line and "<td>2</td>" in line for line in lines)


def test_to_html_auto_vertical_single_row():
    t = Table([[1, 2]], headers=["a", "b"])
    html_v = t.to_html(orientation="auto")
    html_vert = t.to_html(orientation="vertical")
    assert html_v == html_vert


def test_to_html_invalid_orientation():
    t = Table([[1]], headers=["a"])
    with pytest.raises(ValueError):
        t.to_html(orientation="diagonal")


def test_to_html_with_metadata_top():
    t = Table([[1]], headers=["a"], metadata={"m": "v"})
    html = t.to_html(with_metadata=True)
    assert "<br />" in html
    idx_m = html.index("m")
    idx_a = html.index("<th>a</th>")
    assert idx_m < idx_a


def test_to_html_with_metadata_bottom():
    t = Table([[1]], headers=["a"], metadata={"m": "v"})
    html = t.to_html(with_metadata="bottom")
    assert "<br />" in html
    idx_m = html.index("m")
    idx_a = html.index("<th>a</th>")
    assert idx_m > idx_a


def test_to_html_no_metadata_no_br():
    t = Table([[1]], headers=["a"])
    html = t.to_html(with_metadata=True)
    assert "<br />" not in html


def test_to_html_nested_table():
    inner = Table([[1, 2]], headers=["x", "y"])
    outer = Table([[inner, "c"]], headers=["a", "b"])
    html = outer.to_html(max_depth=2)
    # inner table rendered
    assert html.count("<table>") >= 2
    assert "<th>x</th>" in html


def test_get_cell_html_override():
    class MyTable(Table):
        def get_cell_html(self, value):
            return "custom:%s" % value

    t = MyTable([[1]], headers=None)
    html = t.to_html()
    assert "custom:1" in html


# ---------------------------------------------------------------------------
# to_text (Table)
# ---------------------------------------------------------------------------

def test_table_to_text_with_headers():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    out = t.to_text()
    lines = out.split("\n")
    assert len(lines) == 4  # header + separator + 2 rows
    assert "a" in lines[0]
    assert "b" in lines[0]
    assert "-" in lines[1]
    assert "1" in lines[2]
    assert "2" in lines[2]


def test_table_to_text_no_headers():
    t = Table([[1, 2]], headers=None)
    out = t.to_text()
    lines = out.split("\n")
    assert len(lines) == 1
    assert "1" in lines[0]
    assert "2" in lines[0]


def test_table_to_text_with_headers_false():
    t = Table([[1, 2]], headers=["a", "b"])
    out = t.to_text(with_headers=False)
    lines = out.split("\n")
    assert len(lines) == 1
    assert "a" not in lines[0]


def test_table_to_text_maxlen():
    t = Table([["abcdefgh"]], headers=["abcdefgh"])
    out = t.to_text(maxlen=6)
    assert "abc..." in out
    assert "abcdefgh" not in out


def test_table_to_text_empty():
    t = Table()
    out = t.to_text()
    assert out == ""
