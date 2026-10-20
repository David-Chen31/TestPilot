import os
import os.path
from pathlib import Path

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_no_changes():
    assert augpath('foo.bar') == 'foo.bar'


def test_augpath_docstring_example():
    newpath = augpath('foo.bar', '_suff', 'pref_', ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_replace_ext():
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'


def test_augpath_ext_without_dot():
    # ext is used verbatim by joining; no dot is added
    assert augpath('foo.bar', ext='BAZ') == 'fooBAZ'


def test_augpath_add_suffix():
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'


def test_augpath_add_prefix():
    assert augpath('foo.bar', prefix='_') == '_foo.bar'


def test_augpath_replace_base():
    assert augpath('foo.bar', base='baz') == 'baz.bar'


def test_augpath_multidot_true_ext():
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'


def test_augpath_multidot_false_ext():
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'


def test_augpath_multidot_true_suffix():
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_multidot_false_suffix():
    assert augpath('foo.tar.gz', suffix='_new', multidot=False) == 'foo.tar_new.gz'


def test_augpath_multidot_true_replace_base():
    assert augpath('foo.tar.gz', base='bar', multidot=True) == 'bar.tar.gz'


def test_augpath_no_extension():
    # no dot at all
    assert augpath('foo', ext='.bar') == 'foo.bar'
    assert augpath('foo', suffix='_s') == 'foo_s'
    assert augpath('foo', prefix='p_') == 'p_foo'


def test_augpath_no_extension_multidot():
    assert augpath('foo', ext='.bar', multidot=True) == 'foo.bar'


def test_augpath_with_directory():
    assert augpath(os.path.join('a', 'b', 'foo.bar'), ext='.baz') == \
        os.path.join('a', 'b', 'foo.baz')


def test_augpath_replace_dpath():
    result = augpath('foo.bar', dpath='newdir')
    assert result == os.path.join('newdir', 'foo.bar')


def test_augpath_replace_dpath_empty_string_is_used():
    # dpath='' is not None, so it should be used
    result = augpath('foo.bar', dpath='')
    assert result == 'foo.bar'


def test_augpath_directory_with_multidot_and_combos():
    path = os.path.join('dir', 'foo.tar.gz')
    result = augpath(path, prefix='p_', suffix='_s', ext='.zip', multidot=True,
                     dpath='out')
    assert result == os.path.join('out', 'p_foo_s.zip')


def test_augpath_all_parts_multidot_false():
    path = os.path.join('dir', 'a.b.c')
    result = augpath(path, prefix='P', base='B', suffix='S', ext='E',
                     dpath='D', multidot=False)
    assert result == os.path.join('D', 'PBS' + 'E')


def test_augpath_pathlike_input():
    result = augpath(Path('foo.bar'), ext='.baz')
    assert result == 'foo.baz'


def test_augpath_pathlike_dpath():
    result = augpath('foo.bar', dpath=Path('somedir'))
    assert result == os.path.join('somedir', 'foo.bar')


def test_augpath_hidden_file_multidot_false():
    # splitext treats leading dot specially: '.bashrc' -> ('.bashrc', '')
    assert augpath('.bashrc', suffix='_s') == '.bashrc_s'
    assert augpath('.bashrc', ext='.txt') == '.bashrc.txt'


def test_augpath_hidden_file_multidot_true():
    # split('.', 1) on '.bashrc' -> ['', 'bashrc']
    assert augpath('.bashrc', suffix='_s', multidot=True) == '_s.bashrc'


def test_augpath_empty_path():
    assert augpath('') == ''


def test_augpath_trailing_dot():
    # splitext('foo.') -> ('foo', '.')
    assert augpath('foo.', suffix='_s') == 'foo_s.'
    assert augpath('foo.', ext='.bar') == 'foo.bar'


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_basic():
    home = os.path.expanduser('~')
    assert shrinkuser(home) == '~'


def test_shrinkuser_basic_custom_home():
    home = os.path.expanduser('~')
    assert shrinkuser(home, home='$HOME') == '$HOME'


def test_shrinkuser_subpath():
    home = os.path.expanduser('~')
    assert shrinkuser(os.path.join(home, '1')) == os.path.join('~', '1')


def test_shrinkuser_subpath_custom_home():
    home = os.path.expanduser('~')
    assert shrinkuser(os.path.join(home, '1'), home='$HOME') == \
        os.path.join('$HOME', '1')


def test_shrinkuser_non_home():
    home = os.path.expanduser('~')
    other = home + '1'
    assert shrinkuser(other) == other


def test_shrinkuser_does_not_match_prefix_without_sep():
    home = os.path.expanduser('~')
    other = home + 'extra'
    assert shrinkuser(other) == other


def test_shrinkuser_relative_path_unchanged():
    assert shrinkuser('foo/bar') == os.path.normpath('foo/bar')


def test_shrinkuser_normpath_applied():
    home = os.path.expanduser('~')
    messy = os.path.join(home, 'a', '..', 'b')
    result = shrinkuser(messy)
    assert result == os.path.join('~', 'b')


def test_shrinkuser_pathlike():
    home = os.path.expanduser('~')
    assert shrinkuser(Path(home)) == '~'


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('~/$SPAM') == os.path.expanduser('~/eggs')


def test_expandpath_plain():
    assert expandpath('foo') == 'foo'


def test_expandpath_tilde(monkeypatch):
    for var in ('FOO', 'BAR'):
        monkeypatch.delenv(var, raising=False)
    assert expandpath('~') == os.path.expanduser('~')


def test_expandpath_env_var_only(monkeypatch):
    monkeypatch.setenv('MYVAR', 'value')
    assert expandpath('$MYVAR') == 'value'


def test_expandpath_braced_env_var(monkeypatch):
    monkeypatch.setenv('MYVAR', 'value')
    assert expandpath('${MYVAR}/foo') == 'value/foo'


def test_expandpath_undefined_var(monkeypatch):
    # expandvars leaves unset variables untouched
    monkeypatch.delenv('NONEXISTENT_VAR_XYZ', raising=False)
    assert expandpath('$NONEXISTENT_VAR_XYZ/foo') == '$NONEXISTENT_VAR_XYZ/foo'


def test_expandpath_pathlike(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    result = expandpath(Path('$SPAM'))
    assert result == 'eggs'


def test_expandpath_tilde_and_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('~/$SPAM') == os.path.expanduser('~/eggs')
