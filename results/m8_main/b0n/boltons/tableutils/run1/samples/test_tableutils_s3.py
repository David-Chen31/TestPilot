"""Tests for boltons.tableutils."""

import collections

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

class TestToText:
    def test_basic(self):
        assert to_text(123) == "123"
        assert to_text("abc") == "abc"
        assert to_text(None) == "None"

    def test_no_maxlen(self):
        assert to_text("abcdefgh", maxlen=None) == "abcdefgh"
        assert to_text("abcdefgh", maxlen=0) == "abcdefgh"

    def test_under_maxlen(self):
        assert to_text("abc", maxlen=10) == "abc"

    def test_exactly_maxlen(self):
        # len == maxlen is not > maxlen, so no truncation
        assert to_text("abcdef", maxlen=6) == "abcdef"

    def test_truncation(self):
        # maxlen=6: text[:3] + '...' == 'abc...'
        assert to_text("abcdefgh", maxlen=6) == "abc..."
        # maxlen=5: text[:2] + '...'
        assert to_text("abcdefgh", maxlen=5) == "ab..."

    def test_str_raises(self):
        class Bad:
            def __str__(self):
                raise RuntimeError("nope")

            def __repr__(self):
                return "BadRepr"

        assert to_text(Bad()) == "BadRepr"

    def test_str_and_repr_raise(self):
        class Bad:
            def __str__(self):
                raise RuntimeError("nope")

            def __repr__(self):
                raise RuntimeError("nope too")

        result = to_text(Bad())
        assert result.startswith("<")
        assert "Bad" in result


# ---------------------------------------------------------------------------
# escape_html
# ---------------------------------------------------------------------------

class TestEscapeHtml:
    def test_basic(self):
        assert escape_html("<b>") == "&lt;b&gt;"

    def test_quotes(self):
        assert escape_html('"a"') == "&quot;a&quot;"
        assert escape_html("'a'") == "&#x27;a&#x27;"

    def test_ampersand(self):
        assert escape_html("a & b") == "a &amp; b"

    def test_maxlen(self):
        assert escape_html("abcdefgh", maxlen=6) == "abc..."


# ---------------------------------------------------------------------------
# UnsupportedData
# ---------------------------------------------------------------------------

def test_unsupported_data_is_type_error():
    assert issubclass(UnsupportedData, TypeError)


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

class TestInputTypes:
    def test_input_type_base(self):
        it = InputType()
        assert it.get_entry_seq([1, 2, 3], ["h"]) is not None
        # base get_entry doesn't exist, so get_entry_seq would fail
        # Actually base InputType has no get_entry; verify it fails.
        with pytest.raises(AttributeError):
            InputType().get_entry_seq([1], [])

    def test_dict_input_type(self):
        it = DictInputType()
        assert it.check_type({"a": 1}) is True
        assert it.check_type([1]) is False
        assert it.guess_headers({"b": 2, "a": 1}) == ["a", "b"]
        assert it.get_entry({"a": 1, "b": 2}, ["b", "a"]) == [2, 1]
        assert it.get_entry({"a": 1}, ["a", "missing"]) == [1, None]
        assert it.get_entry_seq([{"a": 1}, {"a": 2}], ["a"]) == [[1], [2]]

    def test_object_input_type(self):
        class Foo:
            def __init__(self):
                self.x = 1
                self.y = 2

            def method(self):
                return 3

        it = ObjectInputType()
        obj = Foo()
        assert it.check_type(obj) is True
        # ints are in _DNR, so not object type
        assert it.check_type(1) is False
        assert it.check_type([1, 2]) is True  # lists still pass this raw check
        headers = it.guess_headers(obj)
        assert "x" in headers
        assert "y" in headers
        assert "method" not in headers
        assert it.get_entry(obj, ["x", "y"]) == [1, 2]
        assert it.get_entry(obj, ["x", "missing"]) == [1, None]

    def test_list_input_type(self):
        it = ListInputType()
        assert it.check_type([1, 2]) is True
        assert it.check_type((1, 2)) is False
        assert it.guess_headers([1, 2]) is None
        assert it.get_entry([1, 2], None) == [1, 2]
        assert it.get_entry_seq([[1], [2]], None) == [[1], [2]]

    def test_tuple_input_type(self):
        it = TupleInputType()
        assert it.check_type((1, 2)) is True
        assert it.check_type([1, 2]) is False
        assert it.guess_headers((1, 2)) is None
        assert it.get_entry((1, 2), None) == [1, 2]
        assert it.get_entry_seq([(1, 2), (3, 4)], None) == [[1, 2], [3, 4]]

    def test_namedtuple_input_type(self):
        NT = collections.namedtuple("NT", ["a", "b"])
        it = NamedTupleInputType()
        nt = NT(1, 2)
        assert it.check_type(nt) is True
        assert it.check_type((1, 2)) is False
        assert it.guess_headers(nt) == ["a", "b"]
        assert it.get_entry(nt, ["b", "a"]) == [2, 1]
        assert it.get_entry(nt, ["a", "missing"]) == [1, None]
        assert it.get_entry_seq([nt, NT(3, 4)], ["a"]) == [[1], [3]]


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

class TestTableBasic:
    def test_empty(self):
        t = Table()
        assert len(t) == 0
        assert t.headers == []
        assert t.metadata == {}
        assert t._data == []
        assert t._width == 0

    def test_list_of_lists_headers_inferred(self):
        data = [["h1", "h2"], [1, 2], [3, 4]]
        t = Table(data)
        assert t.headers == ["h1", "h2"]
        assert t._data == [[1, 2], [3, 4]]
        assert t._width == 2
        assert len(t) == 2

    def test_explicit_headers(self):
        t = Table([[1, 2], [3, 4]], headers=["a", "b"])
        assert t.headers == ["a", "b"]
        assert t._data == [[1, 2], [3, 4]]

    def test_headers_none(self):
        t = Table([[1, 2]], headers=None)
        assert t.headers == []

    def test_fill_short_rows(self):
        t = Table([[1], [2, 3, 4]], headers=None)
        # width is max row length = 3
        assert t._width == 3
        assert t._data == [[1, None, None], [2, 3, 4]]

    def test_fill_with_headers(self):
        t = Table([[1]], headers=["a", "b", "c"])
        assert t._data == [[1, None, None]]

    def test_metadata(self):
        t = Table([[1]], metadata={"k": "v"})
        assert t.metadata == {"k": "v"}

    def test_metadata_none_becomes_dict(self):
        t = Table([[1]], metadata=None)
        assert t.metadata == {}

    def test_getitem(self):
        t = Table([[1, 2], [3, 4]], headers=["a", "b"])
        assert t[0] == [1, 2]
        assert t[1] == [3, 4]
        assert t[-1] == [3, 4]

    def test_repr_with_headers(self):
        t = Table([[1, 2]], headers=["a", "b"])
        r = repr(t)
        assert r.startswith("Table(headers=")
        assert "'a'" in r
        assert "[[1, 2]]" in r

    def test_repr_without_headers(self):
        t = Table([[1, 2]], headers=None)
        r = repr(t)
        assert r == "Table([[1, 2]])"

    def test_extend(self):
        t = Table([[1, 2]], headers=["a", "b"])
        t.extend([[3, 4], [5, 6]])
        assert t._data == [[1, 2], [3, 4], [5, 6]]

    def test_extend_empty_noop(self):
        t = Table([[1, 2]], headers=["a", "b"])
        t.extend([])
        assert t._data == [[1, 2]]

    def test_extend_empty_when_empty_table(self):
        t = Table()
        t.extend([])
        assert t._data == []
        assert t._width == 0


class TestTableFromData:
    def test_from_list_of_lists(self):
        t = Table.from_data([[1, 2], [3, 4]])
        assert t.headers == [1, 2] or t.headers == [1, 2]
        assert t._data == [[3, 4]]

    def test_from_dict_single(self):
        t = Table.from_dict({"a": 1, "b": 2})
        # dict guessing headers sorted
        assert t.headers == ["a", "b"]
        assert t._data == [[1, 2]]

    def test_from_dict_list(self):
        t = Table.from_dict([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert t.headers == ["a", "b"]
        assert t._data == [[1, 2], [3, 4]]

    def test_from_list(self):
        t = Table.from_list([[1, 2], [3, 4]])
        assert t._data == [[3, 4]]

    def test_from_object(self):
        class Foo:
            def __init__(self):
                self.a = 1
                self.b = 2

        t = Table.from_object(Foo())
        headers = t.headers
        assert "a" in headers
        assert "b" in headers
        row = t._data[0]
        assert row[headers.index("a")] == 1
        assert row[headers.index("b")] == 2

    def test_from_namedtuple(self):
        NT = collections.namedtuple("NT", ["a", "b"])
        t = Table.from_data([NT(1, 2), NT(3, 4)])
        assert t.headers == ["a", "b"]
        assert t._data == [[1, 2], [3, 4]]

    def test_from_single_namedtuple(self):
        NT = collections.namedtuple("NT", ["a", "b"])
        t = Table.from_data(NT(1, 2))
        assert t.headers == ["a", "b"]
        assert t._data == [[1, 2]]

    def test_from_tuple_list(self):
        t = Table.from_data([(1, 2), (3, 4)])
        # first entry is a tuple -> TupleInputType, guessed headers None
        assert t.headers == []
        assert t._data == [[1, 2], [3, 4]]

    def test_from_dict_with_explicit_headers(self):
        t = Table.from_dict({"a": 1, "b": 2}, headers=["b", "a"])
        assert t.headers == ["b", "a"]
        assert t._data == [[2, 1]]

    def test_scalar_data(self):
        t = Table.from_data(5)
        assert t._data == [[5]]

    def test_scalar_string(self):
        t = Table.from_data("hello")
        assert t._data == [["hello"]]

    def test_empty_seq(self):
        t = Table.from_data([])
        assert t._data == []
        assert t.headers == []

    def test_max_depth_zero(self):
        t = Table.from_data([[1, 2]], max_depth=0)
        # max_depth < 1 returns empty table with headers
        assert t._data == []
        assert t.headers == []

    def test_unsupported_data(self):
        class UnsupportedCheck:
            pass

        # An object with __class__ and not in _DNR can be handled by
        # ObjectInputType, so most things are supported. Force a failure
        # by disabling all input types.
        with pytest.raises(UnsupportedData):
            # a generator: not Sequence, not in _DNR, but ObjectInputType
            # accepts almost anything with __class__. Instead check that
            # from_data with a bad _data_type fails naturally. Use
            # an unusual approach: pass a Mapping subclass whose check
            # returns False and force detection through no match.
            Table.from_data(object(), max_depth=1)  # sanity: this works

    def test_unsupported_data_explicit(self):
        class BadInputType(InputType):
            def check_type(self, obj):
                return False

        with pytest.raises(UnsupportedData):
            Table.from_data([1, 2], _data_type=BadInputType())

    def test_max_depth_nesting(self):
        t = Table.from_data([[[1, 2]]], max_depth=2)
        # outer list of one row containing inner list; with max_depth 2
        # inner data becomes a Table.
        assert isinstance(t._data[0][0], Table)

    def test_max_depth_one_no_nesting(self):
        t = Table.from_data([[[1, 2]]], max_depth=1)
        assert t._data == [[[1, 2]]]


class TestTableToText:
    def test_simple_with_headers(self):
        t = Table([[1, 2]], headers=["a", "b"])
        out = t.to_text()
        lines = out.split("\n")
        assert len(lines) == 3
        assert "a" in lines[0]
        assert "|" in lines[0]
        assert set(lines[1].replace(" ", "")) == {"-", "|"}
        assert "1" in lines[2]
        assert "2" in lines[2]

    def test_without_headers(self):
        t = Table([[1, 2]], headers=["a", "b"])
        out = t.to_text(with_headers=False)
        lines = out.split("\n")
        assert len(lines) == 1
        assert "1" in lines[0]
        assert "2" in lines[0]
        assert "a" not in lines[0]

    def test_no_headers_table(self):
        t = Table([[1, 2]], headers=None)
        out = t.to_text()
        lines = out.split("\n")
        assert len(lines) == 1
        assert "1" in lines[0]

    def test_maxlen(self):
        t = Table([["abcdefghij"]], headers=["col"])
        out = t.to_text(maxlen=6)
        assert "abc..." in out
        assert "abcdefghij" not in out

    def test_empty_table(self):
        t = Table()
        out = t.to_text()
        assert out == ""

    def test_column_widths(self):
        t = Table([["a", "bb"], ["ccc", "d"]], headers=["h", "hh"])
        out = t.to_text()
        lines = out.split("\n")
        # First line should have header cell widths matching widest.
        assert "ccc".center(3) in lines[2]


class TestTableToHtml:
    def test_basic_horizontal(self):
        t = Table([[1, 2]], headers=["a", "b"])
        out = t.to_html()
        assert out.startswith("<table>")
        assert out.endswith("</table>")
        assert "<thead>" in out
        assert "<tbody>" in out
        assert "<th>a</th>" in out
        assert "<td>1</td>" in out

    def test_no_wrap(self):
        t = Table([[1, 2]], headers=["a", "b"])
        out = t.to_html(wrapped=False)
        assert "<table>" not in out
        assert "</table>" not in out
        assert "<td>1</td>" in out

    def test_with_newlines(self):
        t = Table([[1]], headers=["a"])
        out = t.to_html(with_newlines=True)
        assert "\n" in out
        out_no = t.to_html(with_newlines=False)
        assert "\n" not in out_no

    def test_no_headers(self):
        t = Table([[1, 2]], headers=None)
        out = t.to_html(with_headers=False)
        assert "<thead>" not in out
        assert "<td>1</td>" in out

    def test_escape_cells(self):
        t = Table([["<b>"]], headers=["<h>"])
        out = t.to_html()
        assert "&lt;b&gt;" in out
        assert "&lt;h&gt;" in out
        assert "<b>" not in out

    def test_vertical_orientation(self):
        t = Table([[1, 2], [3, 4]], headers=["a", "b"])
        out = t.to_html(orientation="vertical")
        assert "<thead>" not in out
        assert "<th>a</th>" in out
        assert "<td>1</td>" in out
        assert "<td>3</td>" in out

    def test_auto_single_row_is_vertical(self):
        t = Table([[1, 2]], headers=["a", "b"])
        out = t.to_html(orientation="auto")
        # single row -> vertical
        assert "<thead>" not in out
        assert "<th>a</th>" in out

    def test_auto_multi_row_horizontal(self):
        t = Table([[1, 2], [3, 4]], headers=["a", "b"])
        out = t.to_html(orientation="auto")
        assert "<thead>" in out

    def test_invalid_orientation(self):
        t = Table([[1]], headers=["a"])
        with pytest.raises(ValueError):
            t.to_html(orientation="diagonal")

    def test_with_metadata_top(self):
        t = Table([[1]], headers=["a"], metadata={"k": "v"})
        out = t.to_html(with_metadata=True)
        assert "<br />" in out
        # metadata table appears before main table
        idx_meta = out.find("<td>k</td>")
        idx_main = out.find("<td>1</td>")
        assert idx_meta != -1
        assert idx_meta < idx_main

    def test_with_metadata_bottom(self):
        t = Table([[1]], headers=["a"], metadata={"k": "v"})
        out = t.to_html(with_metadata="bottom")
        idx_meta = out.find("<td>k</td>")
        idx_main = out.find("<td>1</td>")
        assert idx_meta != -1
        assert idx_main < idx_meta

    def test_no_metadata(self):
        t = Table([[1]], headers=["a"])
        out = t.to_html(with_metadata=True)
        assert "<br />" not in out

    def test_nested_table_html(self):
        inner = Table([[1, 2]], headers=["x", "y"])
        outer = Table([[inner]], headers=["nested"])
        out = outer.to_html(max_depth=2)
        assert "<th>nested</th>" in out
        # inner table rendered
        assert "<th>x</th>" in out
        assert "<td>1</td>" in out

    def test_nested_table_repr_fallback(self):
        inner = Table([[1]], headers=["x"])
        outer = Table([[inner]], headers=["nested"])
        out = outer.to_html(max_depth=1)
        # inner table rendered via repr
        assert "Table(" in out

    def test_get_cell_html_override(self):
        class MyTable(Table):
            def get_cell_html(self, value):
                return "[[%s]]" % value

        t = MyTable([[1]], headers=["a"])
        out = t.to_html()
        assert "[[1]]" in out
        assert "[[a]]" in out

    def test_vertical_nested_table(self):
        inner = Table([[9]], headers=["z"])
        t = Table([[inner]], headers=["a"])
        out = t.to_html(orientation="vertical", max_depth=2)
        assert "<td>9</td>" in out
