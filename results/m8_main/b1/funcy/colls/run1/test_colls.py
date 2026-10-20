import pytest
from collections import defaultdict, Counter, OrderedDict
from collections.abc import Iterator

from funcy.colls import (
    empty, iteritems, itervalues, join, merge, join_with, merge_with,
    walk, walk_keys, walk_values, select, select_keys, select_values,
    split_keys, compact, is_distinct, all, any, none, one, some, zipdict,
    flip, project, omit, zip_values, zip_dicts, get_in, get_lax, set_in,
    update_in, del_in, has_path, lwhere, lpluck, lpluck_attr, linvoke,
    where, pluck, pluck_attr, invoke,
)
from funcy.primitives import EMPTY


# ---------- empty ----------

def test_empty_dict():
    assert empty({1: 2}) == {}

def test_empty_list():
    assert empty([1, 2]) == []

def test_empty_tuple():
    assert empty((1, 2)) == ()

def test_empty_set():
    assert empty({1, 2}) == set()

def test_empty_str():
    assert empty("abc") == ""

def test_empty_bytes():
    assert empty(b"abc") == b""

def test_empty_iterator():
    result = empty(iter([1, 2, 3]))
    assert isinstance(result, Iterator)
    assert list(result) == []

def test_empty_dict_keys():
    d = {1: 2}
    result = empty(d.keys())
    assert isinstance(result, list)
    assert result == []

def test_empty_dict_values():
    d = {1: 2}
    assert empty(d.values()) == []

def test_empty_dict_items():
    d = {1: 2}
    assert empty(d.items()) == []

def test_empty_defaultdict():
    d = defaultdict(list, {1: [2]})
    result = empty(d)
    assert isinstance(result, defaultdict)
    assert result == {}

def test_empty_counter():
    c = Counter({1: 2})
    result = empty(c)
    assert isinstance(result, Counter)
    assert result == {}


# ---------- iteritems / itervalues ----------

def test_iteritems_dict():
    d = {1: 2, 3: 4}
    assert list(iteritems(d)) == [(1, 2), (3, 4)]

def test_iteritems_list_of_pairs():
    data = [(1, 2), (3, 4)]
    assert list(iteritems(data)) == [(1, 2), (3, 4)]

def test_itervalues_dict():
    d = {1: 2, 3: 4}
    assert list(itervalues(d)) == [2, 4]

def test_itervalues_list():
    data = [1, 2, 3]
    assert list(itervalues(data)) == [1, 2, 3]


# ---------- join / merge ----------

def test_join_empty():
    assert join([]) is None

def test_join_strings():
    assert join(["ab", "cd", "ef"]) == "abcdef"

def test_join_bytes():
    assert join([b"ab", b"cd"]) == b"abcd"

def test_join_dicts():
    result = join([{1: 2}, {3: 4}, {1: 5}])
    assert result == {1: 5, 3: 4}

def test_join_sets():
    assert join([{1, 2}, {2, 3}]) == {1, 2, 3}

def test_join_lists():
    assert join([[1, 2], [3, 4]]) == [1, 2, 3, 4]

def test_join_tuples():
    assert join([(1, 2), (3, 4)]) == (1, 2, 3, 4)

def test_join_iterators():
    result = join([iter([1, 2]), iter([3, 4])])
    assert isinstance(result, Iterator)
    assert list(result) == [1, 2, 3, 4]

def test_join_ranges():
    result = join([range(3), range(3, 5)])
    assert list(result) == [0, 1, 2, 3, 4]

def test_join_invalid_type():
    with pytest.raises(TypeError):
        join([1, 2, 3])

def test_merge_dicts():
    assert merge({1: 2}, {3: 4}) == {1: 2, 3: 4}

def test_merge_lists():
    assert merge([1, 2], [3, 4]) == [1, 2, 3, 4]

def test_merge_empty():
    assert merge() is None


# ---------- join_with / merge_with ----------

def test_join_with_empty():
    assert join_with(sum, []) == {}

def test_join_with_single():
    d = {1: 2}
    # not strict, single dict returns that exact dict
    assert join_with(sum, [d]) is d

def test_join_with_single_strict():
    assert join_with(sum, [{1: 2}], strict=True) == {1: 2}

def test_join_with_sum():
    result = join_with(sum, [{1: 2}, {1: 3, 2: 4}, {2: 5}])
    assert result == {1: 5, 2: 9}

def test_join_with_list():
    result = join_with(list, [{1: 2}, {1: 3, 2: 4}])
    assert result == {1: [2, 3], 2: [4]}

def test_merge_with():
    result = merge_with(sum, {1: 2}, {1: 3, 2: 4})
    assert result == {1: 5, 2: 4}


# ---------- walk family ----------

def test_walk_list():
    assert walk(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

def test_walk_dict_uses_items():
    # walk applies f to items() of a dict
    result = walk(lambda pair: (pair[0] + 1, pair[1] + 10), {1: 2, 3: 4})
    assert result == {2: 12, 4: 14}

def test_walk_keys_dict():
    assert walk_keys(lambda k: k * 2, {1: 10, 2: 20}) == {2: 10, 4: 20}

def test_walk_keys_with_funcmaker():
    # a string key acts as a regex finder
    assert walk_keys("a", {"abc": 1, "xyz": 2}) == {"a": 1, None: 2}

def test_walk_values_dict():
    assert walk_values(lambda v: v * 2, {1: 10, 2: 20}) == {1: 20, 2: 40}

def test_walk_values_with_int():
    # int acts as itemgetter
    assert walk_values(0, {1: [10, 20], 2: [30, 40]}) == {1: 10, 2: 30}

def test_walk_values_defaultdict():
    d = defaultdict(list, {1: [1]})
    result = walk_values(lambda v: v * 2, d)
    assert isinstance(result, defaultdict)
    assert result == {1: [1, 1]}

def test_walk_values_counter():
    c = Counter({1: 2})
    result = walk_values(lambda v: v * 2, c)
    assert isinstance(result, Counter)
    assert result[1] == 4


# ---------- select family ----------

def test_select_list():
    assert select(lambda x: x > 1, [1, 2, 3]) == [2, 3]

def test_select_dict():
    result = select(lambda pair: pair[1] > 20, {1: 10, 2: 30})
    assert result == {2: 30}

def test_select_keys_dict():
    assert select_keys(lambda k: k > 1, {1: 10, 2: 20, 3: 30}) == {2: 20, 3: 30}

def test_select_keys_with_str_regex():
    assert select_keys("a", {"abc": 1, "xyz": 2}) == {"abc": 1}

def test_select_values_dict():
    assert select_values(lambda v: v > 15, {1: 10, 2: 20}) == {2: 20}

def test_select_values_with_int():
    # int acts as itemgetter on value; [10, 5][1] == 5, [3, 4][1] == 4, both truthy
    assert select_values(1, {1: [10, 5], 2: [3, 4]}) == {1: [10, 5], 2: [3, 4]}


# ---------- split_keys ----------

def test_split_keys_dict():
    yes, no = split_keys(lambda k: k > 1, {1: 1, 2: 2, 3: 3})
    assert yes == {2: 2, 3: 3}
    assert no == {1: 1}

def test_split_keys_preserves_type():
    d = defaultdict(list, {1: [1], 2: [2]})
    yes, no = split_keys(lambda k: k > 1, d)
    assert isinstance(yes, defaultdict)
    assert isinstance(no, defaultdict)


# ---------- compact ----------

def test_compact_list():
    assert compact([0, 1, 2, None, 3, ""]) == [1, 2, 3]

def test_compact_dict():
    result = compact({1: 0, 2: 5, 3: None, 4: 7})
    assert result == {2: 5, 4: 7}

def test_compact_set():
    assert compact({0, 1, 2}) == {1, 2}


# ---------- content tests ----------

def test_is_distinct():
    assert is_distinct([1, 2, 3]) is True
    assert is_distinct([1, 2, 1]) is False

def test_is_distinct_key():
    data = [{"a": 1}, {"a": 2}]
    assert is_distinct(data, key=lambda x: x["a"]) is True
    assert is_distinct([1, -1], key=abs) is False

def test_all_with_pred():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 0, [1, -2, 3]) is False

def test_all_without_pred():
    assert all([1, 2, 3]) is True
    assert all([1, 0, 3]) is False

def test_any_with_pred():
    assert any(lambda x: x > 2, [1, 2, 3]) is True
    assert any(lambda x: x > 3, [1, 2, 3]) is False

def test_any_without_pred():
    assert any([0, 0, 1]) is True
    assert any([0, 0, 0]) is False

def test_none():
    assert none(lambda x: x > 5, [1, 2, 3]) is True
    assert none(lambda x: x > 2, [1, 2, 3]) is False

def test_one():
    assert one(lambda x: x > 2, [1, 2, 3]) is True
    assert one(lambda x: x > 1, [1, 2, 3]) is False
    assert one(lambda x: x > 5, [1, 2, 3]) is False

def test_one_without_pred():
    assert one([0, 1, 0]) is True
    assert one([1, 1, 0]) is False

def test_some():
    assert some(lambda x: x > 1, [1, 2, 3]) == 2
    assert some(lambda x: x > 5, [1, 2, 3]) is None

def test_some_without_pred():
    assert some([0, 0, 5, 7]) == 5
    assert some([0, 0, 0]) is None


# ---------- zipdict / flip ----------

def test_zipdict():
    assert zipdict([1, 2], [3, 4]) == {1: 3, 2: 4}

def test_zipdict_short():
    assert zipdict([1, 2, 3], [3, 4]) == {1: 3, 2: 4}

def test_flip_dict():
    assert flip({1: 2, 3: 4}) == {2: 1, 4: 3}

def test_flip_pairs():
    # flipping a list of pairs produces a list of swapped pairs
    assert flip([(1, "a"), (2, "b")]) == [("a", 1), ("b", 2)]


# ---------- project / omit ----------

def test_project():
    d = {1: "a", 2: "b", 3: "c"}
    assert project(d, [1, 3, 5]) == {1: "a", 3: "c"}

def test_project_preserves_type():
    o = OrderedDict([(1, "a"), (2, "b")])
    result = project(o, [2])
    assert isinstance(result, OrderedDict)
    assert result == OrderedDict([(2, "b")])

def test_omit():
    d = {1: "a", 2: "b", 3: "c"}
    assert omit(d, [2]) == {1: "a", 3: "c"}

def test_omit_preserves_type():
    o = OrderedDict([(1, "a"), (2, "b")])
    result = omit(o, [1])
    assert isinstance(result, OrderedDict)
    assert result == OrderedDict([(2, "b")])


# ---------- zip_values / zip_dicts ----------

def test_zip_values():
    d1 = {1: "a", 2: "b"}
    d2 = {1: "x", 2: "y", 3: "z"}
    result = set(zip_values(d1, d2))
    assert result == {("a", "x"), ("b", "y")}

def test_zip_values_no_args():
    gen = zip_values()
    with pytest.raises(TypeError):
        list(gen)

def test_zip_dicts():
    d1 = {1: "a", 2: "b"}
    d2 = {1: "x", 2: "y", 3: "z"}
    result = dict(zip_dicts(d1, d2))
    assert result == {1: ("a", "x"), 2: ("b", "y")}

def test_zip_dicts_no_args():
    gen = zip_dicts()
    with pytest.raises(TypeError):
        list(gen)


# ---------- get_in / get_lax ----------

def test_get_in():
    d = {1: {2: {3: "x"}}}
    assert get_in(d, [1, 2, 3]) == "x"

def test_get_in_missing():
    d = {1: {2: {3: "x"}}}
    assert get_in(d, [1, 9]) is None
    assert get_in(d, [1, 9], "default") == "default"

def test_get_in_list():
    data = [[1, 2], [3, 4]]
    assert get_in(data, [1, 0]) == 3

def test_get_in_index_error():
    data = [[1, 2], [3, 4]]
    assert get_in(data, [5], "def") == "def"

def test_get_lax_type_error():
    # get_lax returns default on wrong collection type; get_in raises TypeError
    assert get_lax(5, [0], "def") == "def"
    with pytest.raises(TypeError):
        get_in(5, [0], "def")

def test_get_lax_basic():
    d = {1: [10, 20]}
    assert get_lax(d, [1, 1]) == 20
    assert get_lax(d, [2], "x") == "x"


# ---------- set_in / update_in ----------

def test_set_in_dict():
    d = {1: {2: 3}}
    result = set_in(d, [1, 2], 5)
    assert result == {1: {2: 5}}
    assert d == {1: {2: 3}}  # original unchanged

def test_set_in_list():
    data = [[1, 2], [3, 4]]
    result = set_in(data, [0, 1], 9)
    assert result == [[1, 9], [3, 4]]
    assert data == [[1, 2], [3, 4]]

def test_set_in_creates_dict_path():
    d = {}
    result = set_in(d, [1, 2, 3], "v")
    assert result == {1: {2: {3: "v"}}}

def test_update_in_dict():
    d = {1: {"a": 5}}
    result = update_in(d, [1, "a"], lambda x: x + 1)
    assert result == {1: {"a": 6}}
    assert d == {1: {"a": 5}}

def test_update_in_empty_path():
    assert update_in({1: 2}, [], lambda c: {"new": True}) == {"new": True}

def test_update_in_with_default():
    result = update_in({}, ["a", "b"], lambda x: x + 1, default=0)
    assert result == {"a": {"b": 1}}


# ---------- del_in ----------

def test_del_in_dict():
    d = {1: {2: 3, 4: 5}}
    result = del_in(d, [1, 2])
    assert result == {1: {4: 5}}
    assert d == {1: {2: 3, 4: 5}}

def test_del_in_top_level():
    d = {1: 2, 3: 4}
    assert del_in(d, [1]) == {3: 4}

def test_del_in_list():
    data = [[1, 2], [3, 4]]
    result = del_in(data, [0, 1])
    assert result == [[1], [3, 4]]
    assert data == [[1, 2], [3, 4]]

def test_del_in_missing():
    d = {1: 2}
    assert del_in(d, [5]) == {1: 2}
    # del_in does not guard against non-subscriptable values; raises TypeError
    with pytest.raises(TypeError):
        del_in(d, [1, 2])

def test_del_in_empty_path():
    d = {1: 2}
    assert del_in(d, []) == {1: 2}


# ---------- has_path ----------

def test_has_path():
    d = {1: {2: {3: "x"}}}
    assert has_path(d, [1, 2, 3]) is True
    assert has_path(d, [1, 2, 4]) is False
    assert has_path(d, [1, 9]) is False

def test_has_path_empty():
    assert has_path({}, []) is True

def test_has_path_list():
    data = [[1, 2], [3, 4]]
    assert has_path(data, [1, 0]) is True
    assert has_path(data, [5]) is False


# ---------- where / lwhere ----------

MAPPINGS = [
    {"a": 1, "b": 2},
    {"a": 1, "b": 3},
    {"a": 2, "b": 2},
]


def test_where():
    result = list(where(MAPPINGS, a=1))
    assert result == [{"a": 1, "b": 2}, {"a": 1, "b": 3}]

def test_where_multiple_conditions():
    result = list(where(MAPPINGS, a=1, b=3))
    assert result == [{"a": 1, "b": 3}]

def test_where_no_match():
    assert list(where(MAPPINGS, a=5)) == []

def test_lwhere():
    result = lwhere(MAPPINGS, a=1)
    assert isinstance(result, list)
    assert result == [{"a": 1, "b": 2}, {"a": 1, "b": 3}]


# ---------- pluck / lpluck ----------

def test_pluck():
    assert list(pluck("a", MAPPINGS)) == [1, 1, 2]

def test_pluck_index():
    data = [[10, 20], [30, 40]]
    assert list(pluck(1, data)) == [20, 40]

def test_lpluck():
    assert lpluck("b", MAPPINGS) == [2, 3, 2]


# ---------- pluck_attr / lpluck_attr ----------

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_pluck_attr():
    points = [Point(1, 2), Point(3, 4)]
    assert list(pluck_attr("x", points)) == [1, 3]

def test_lpluck_attr():
    points = [Point(1, 2), Point(3, 4)]
    assert lpluck_attr("y", points) == [2, 4]


# ---------- invoke / linvoke ----------

class Speaker:
    def say(self, word, punct="!"):
        return word + punct


def test_invoke():
    speakers = [Speaker(), Speaker()]
    assert list(invoke(speakers, "say", "hi")) == ["hi!", "hi!"]

def test_invoke_with_kwargs():
    speakers = [Speaker()]
    assert list(invoke(speakers, "say", "hi", punct="?")) == ["hi?"]

def test_linvoke():
    speakers = [Speaker(), Speaker()]
    result = linvoke(speakers, "say", "hi", punct=".")
    assert isinstance(result, list)
    assert result == ["hi.", "hi."]
