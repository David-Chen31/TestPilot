import os
import sys

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_no_changes():
    assert augpath('foo.bar') == 'foo.bar'
    assert augpath('foo') == 'foo'


def test_augpath_docstring_examples():
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'
    assert augpath('foo.bar', prefix='_') == '_foo.bar'
    assert augpath('foo.bar', base='baz') == 'baz.bar'
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_full_example():
    newpath = augpath('foo.bar', suffix='_suff', prefix='pref_', ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_dpath_replacement():
    result = augpath('some/dir/foo.bar', dpath='other/dir')
    assert result == os.path.join('other', 'dir', 'foo.bar')


def test_augpath_keeps_directory():
    expected = os.path.join('some', 'dir', 'foo.baz')
    assert augpath('some/dir/foo.bar', ext='.baz') == expected


def test_augpath_no_extension():
    assert augpath('foo', ext='.txt') == 'foo.txt'
    assert augpath('foo', suffix='_x') == 'foo_x'
    assert augpath('foo', prefix='pre_') == 'pre_foo'


def test_augpath_multidot_suffix_without_ext():
    # multidot=True on a name with no dot -> orig_ext == ''
    assert augpath('foo', suffix='_new', multidot=True) == 'foo_new'


def test_augpath_multidot_leading_dot():
    # A leading dot means the first part is empty and everything after is ext
    assert augpath('.bashrc', ext='.txt', multidot=True) == '.txt'
    assert augpath('.bashrc', multidot=False) == '.bashrc'


def test_augpath_combined():
    result = augpath('a/b/c.tar.gz', prefix='p_', suffix='_s', ext='.zip',
                     base='newname', multidot=True)
    assert result == os.path.join('a', 'b', 'p_newname_s.zip')


def test_augpath_empty_path():
    assert augpath('') == ''


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_home_only():
    home = os.path.expanduser('~')
    assert shrinkuser(home) == '~'


def test_shrinkuser_custom_home_symbol():
    home = os.path.expanduser('~')
    assert shrinkuser(home, home='$HOME') == '$HOME'


def test_shrinkuser_subpath():
    home = os.path.expanduser('~')
    assert shrinkuser(os.path.join(home, '1')) == os.path.join('~', '1')


def test_shrinkuser_subpath_custom_symbol():
    home = os.path.expanduser('~')
    result = shrinkuser(os.path.join(home, '1'), home='$HOME')
    assert result == os.path.join('$HOME', '1')


def test_shrinkuser_not_under_home():
    # A path that simply shares a prefix but is not actually inside home
    home = os.path.expanduser('~')
    weird = home + '1'
    assert shrinkuser(weird) == weird


def test_shrinkuser_relative_path_unchanged():
    assert shrinkuser('foo/bar') == os.path.normpath('foo/bar')
    assert shrinkuser('foo') == 'foo'


def test_shrinkuser_normalizes():
    home = os.path.expanduser('~')
    weird = os.path.join(home, '.', 'x', '..', 'y')
    assert shrinkuser(weird) == os.path.join('~', 'y')


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    result = expandpath('~/$SPAM')
    assert result == os.path.expanduser('~/eggs')


def test_expandpath_plain(monkeypatch):
    # Ensure setting a var doesn't break expansion of a plain path
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('foo') == 'foo'


def test_expandpath_tilde_only():
    assert expandpath('~') == os.path.expanduser('~')


def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv('BOLTONS_TEST_VAR', 'value123')
    assert expandpath('${BOLTONS_TEST_VAR}') == 'value123'
    assert expandpath('$BOLTONS_TEST_VAR') == 'value123'


def test_expandpath_missing_var(monkeypatch):
    monkeypatch.delenv('BOLTONS_MISSING_VAR', raising=False)
    # expandvars leaves unknown variables untouched
    assert expandpath('$BOLTONS_MISSING_VAR') == '$BOLTONS_MISSING_VAR'


def test_expandpath_combines_tilde_and_vars(monkeypatch):
    monkeypatch.setenv('BOLTONS_NAME', 'file.txt')
    result = expandpath('~/dir/$BOLTONS_NAME')
    expected = os.path.expanduser('~/dir/file.txt')
    assert result == expected
