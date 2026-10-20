import os

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_identity():
    assert augpath("foo.bar") == "foo.bar"
    assert augpath("foo") == "foo"
    assert augpath("/a/b/c.txt") == "/a/b/c.txt"


def test_augpath_docstring_examples():
    assert augpath("foo.bar", ext=".BAZ") == "foo.BAZ"
    assert augpath("foo.bar", suffix="_") == "foo_.bar"
    assert augpath("foo.bar", prefix="_") == "_foo.bar"
    assert augpath("foo.bar", base="baz") == "baz.bar"
    assert augpath("foo.tar.gz", ext=".zip", multidot=True) == "foo.zip"
    assert augpath("foo.tar.gz", ext=".zip", multidot=False) == "foo.tar.zip"
    assert augpath("foo.tar.gz", suffix="_new", multidot=True) == "foo_new.tar.gz"


def test_augpath_combined_example():
    newpath = augpath("foo.bar", suffix="_suff", prefix="pref_",
                      ext=".baz", base="bar")
    assert newpath == "pref_bar_suff.baz"


def test_augpath_suffix_inserted_between_base_and_ext():
    assert augpath("foo.tar.gz", suffix="_x") == "foo.tar_x.gz"


def test_augpath_prefix_only():
    assert augpath("/some/dir/file.txt", prefix="pre_") == "/some/dir/pre_file.txt"


def test_augpath_ext_replacement():
    assert augpath("file.txt", ext=".md") == "file.md"
    # ext replaces entirely, including the dot
    assert augpath("file.txt", ext="md") == "filemd"
    # no ext on original, adding one
    assert augpath("file", ext=".txt") == "file.txt"
    # replacing with empty string removes extension
    assert augpath("file.txt", ext="") == "file"


def test_augpath_base_replacement():
    assert augpath("file.txt", base="other") == "other.txt"
    assert augpath("file", base="other") == "other"


def test_augpath_dpath_replacement(tmp_path):
    d = str(tmp_path)
    result = augpath("file.txt", dpath=d)
    assert result == os.path.join(d, "file.txt")
    assert os.path.dirname(result) == d


def test_augpath_dpath_kept_by_default():
    result = augpath("/a/b/file.txt", suffix="_s")
    assert result == "/a/b/file_s.txt"


def test_augpath_dpath_pathlike(tmp_path):
    # PathLike dpath
    result = augpath("file.txt", dpath=tmp_path)
    assert os.path.dirname(result) == str(tmp_path)
    assert os.path.basename(result) == "file.txt"


def test_augpath_multidot_no_dot():
    # multidot with no dot in the basename
    assert augpath("foo", ext=".bz", multidot=True) == "foo.bz"
    assert augpath("foo", suffix="_s", multidot=True) == "foo_s"


def test_augpath_multidot_base_replacement():
    # multidot: base is only the first segment before first dot
    assert augpath("foo.tar.gz", base="bar", multidot=True) == "bar.tar.gz"


def test_augpath_multidot_prefix_suffix():
    assert augpath("foo.tar.gz", prefix="p_", multidot=True) == "p_foo.tar.gz"
    assert augpath("foo.tar.gz", suffix="_s", multidot=True) == "foo_s.tar.gz"


def test_augpath_empty_path():
    assert augpath("") == ""


def test_augpath_leading_dot_file():
    # dotfile: splitext treats leading dot name as having no extension
    assert augpath(".bashrc", suffix="_x") == ".bashrc_x"
    assert augpath(".bashrc", ext=".bak") == ".bashrc.bak"


def test_augpath_all_parts_replaced():
    result = augpath("dir/old.txt", suffix="_s", prefix="p_", ext=".md",
                     base="new", dpath="newdir")
    assert result == os.path.join("newdir", "p_new_s.md")


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_home_exact():
    path = os.path.expanduser("~")
    assert shrinkuser(path) == "~"


def test_shrinkuser_home_custom_symbol():
    path = os.path.expanduser("~")
    assert shrinkuser(path, home="$HOME") == "$HOME"
    assert shrinkuser(path, home="%USERPROFILE%") == "%USERPROFILE%"


def test_shrinkuser_under_home():
    path = os.path.expanduser("~")
    sub = os.path.join(path, "1")
    assert shrinkuser(sub) == os.path.join("~", "1")
    assert shrinkuser(sub, home="$HOME") == os.path.join("$HOME", "1")


def test_shrinkuser_nested_under_home():
    path = os.path.expanduser("~")
    sub = os.path.join(path, "a", "b", "c")
    assert shrinkuser(sub) == os.path.join("~", "a", "b", "c")


def test_shrinkuser_sibling_not_shrunk():
    path = os.path.expanduser("~")
    # appending "1" makes it no longer a subdirectory of home
    sibling = path + "1"
    assert shrinkuser(sibling) == sibling


def test_shrinkuser_unrelated_path():
    assert shrinkuser("/definitely/not/home/xyz") == "/definitely/not/home/xyz"


def test_shrinkuser_relative_path():
    assert shrinkuser("relative/path") == os.path.normpath("relative/path")


def test_shrinkuser_normalizes():
    # normpath is applied
    result = shrinkuser("/a//b/./c")
    assert result == os.path.normpath("/a//b/./c")


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv("SPAM", "eggs")
    assert expandpath("~/$SPAM") == os.path.expanduser("~/eggs")
    assert expandpath("$SPAM") == "eggs"
    assert expandpath("/pre/$SPAM/post") == "/pre/eggs/post"


def test_expandpath_plain():
    assert expandpath("foo") == "foo"
    assert expandpath("/a/b/c") == "/a/b/c"


def test_expandpath_tilde():
    assert expandpath("~") == os.path.expanduser("~")
    sub = os.path.expanduser("~/somefile")
    assert expandpath("~/somefile") == sub


def test_expandpath_missing_env_var(monkeypatch):
    monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
    # expandvars leaves unknown $VAR unchanged
    assert expandpath("$NONEXISTENT_VAR_XYZ") == "$NONEXISTENT_VAR_XYZ"


def test_expandpath_combined_env_and_tilde(monkeypatch):
    monkeypatch.setenv("MYDIR", "subdir")
    assert expandpath("~/../$MYDIR") == os.path.expanduser("~/../subdir")


def test_expandpath_empty():
    assert expandpath("") == ""
