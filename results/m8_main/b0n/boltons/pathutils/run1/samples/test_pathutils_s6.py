import os

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_no_changes():
    assert augpath('foo.bar') == 'foo.bar'


def test_augpath_ext_replace():
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'


def test_augpath_suffix():
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'


def test_augpath_prefix():
    assert augpath('foo.bar', prefix='_') == '_foo.bar'


def test_augpath_base_replace():
    assert augpath('foo.bar', base='baz') == 'baz.bar'


def test_augpath_multidot_true_ext():
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'


def test_augpath_multidot_false_ext():
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'


def test_augpath_multidot_true_suffix():
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_docstring_example():
    newpath = augpath('foo.bar', '_suff', 'pref_', ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_no_extension():
    assert augpath('foo', ext='.bar') == 'foo.bar'
    assert augpath('foo', suffix='_x') == 'foo_x'


def test_augpath_multidot_no_extension():
    assert augpath('foo', ext='.bar', multidot=True) == 'foo.bar'
    assert augpath('foo', suffix='_x', multidot=True) == 'foo_x'


def test_augpath_with_directory():
    result = augpath(os.path.join('a', 'b', 'foo.txt'), ext='.md')
    assert result == os.path.join('a', 'b', 'foo.md')


def test_augpath_dpath_override():
    result = augpath(os.path.join('a', 'b', 'foo.txt'), dpath=os.path.join('x', 'y'))
    assert result == os.path.join('x', 'y', 'foo.txt')


def test_augpath_dpath_override_and_augment():
    result = augpath(
        os.path.join('a', 'foo.txt'),
        dpath='z',
        prefix='p_',
        suffix='_s',
        ext='.md',
    )
    assert result == os.path.join('z', 'p_foo_s.md')


def test_augpath_pathlike(tmp_path):
    p = tmp_path / 'foo.txt'
    result = augpath(p, ext='.md')
    assert result == str(tmp_path / 'foo.md')


def test_augpath_multidot_pathlike(tmp_path):
    p = tmp_path / 'foo.tar.gz'
    assert augpath(p, ext='.zip', multidot=True) == str(tmp_path / 'foo.zip')


def test_augpath_hidden_file():
    # ".bashrc" -> splitext gives base='.bashrc', ext=''
    assert augpath('.bashrc', suffix='_bak') == '.bashrc_bak'
    assert augpath('.bashrc', ext='.txt') == '.bashrc.txt'


def test_augpath_multidot_hidden_file():
    # With multidot, leading dot yields base='' and ext='.bashrc'
    assert augpath('.bashrc', suffix='_bak', multidot=True) == '_bak.bashrc'


def test_augpath_multiple_dots_non_multidot():
    assert augpath('a.b.c.d') == 'a.b.c.d'
    assert augpath('a.b.c.d', ext='.e') == 'a.b.c.e'


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_home_exact():
    home = os.path.expanduser('~')
    assert shrinkuser(home) == '~'


def test_shrinkuser_home_custom_symbol():
    home = os.path.expanduser('~')
    assert shrinkuser(home, '$HOME') == '$HOME'


def test_shrinkuser_under_home():
    home = os.path.expanduser('~')
    assert shrinkuser(os.path.join(home, '1')) == os.path.join('~', '1')


def test_shrinkuser_under_home_custom_symbol():
    home = os.path.expanduser('~')
    assert shrinkuser(os.path.join(home, '1'), '$HOME') == os.path.join('$HOME', '1')


def test_shrinkuser_deep_path_under_home():
    home = os.path.expanduser('~')
    nested = os.path.join(home, 'a', 'b', 'c')
    assert shrinkuser(nested) == os.path.join('~', 'a', 'b', 'c')


def test_shrinkuser_sibling_of_home():
    home = os.path.expanduser('~')
    sibling = home + '1'
    # A path that merely starts with the home string but isn't under it
    assert shrinkuser(sibling) == sibling


def test_shrinkuser_unrelated_path(tmp_path):
    p = str(tmp_path)
    assert shrinkuser(p) == os.path.normpath(p)


def test_shrinkuser_normalizes_path():
    home = os.path.expanduser('~')
    # Extra separators get normalized to the home dir
    messy = home + os.path.sep
    assert shrinkuser(messy) == '~'


def test_shrinkuser_pathlike(tmp_path):
    result = shrinkuser(tmp_path)
    assert result == os.path.normpath(str(tmp_path))


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    home = os.path.expanduser('~')
    assert expandpath('~/$SPAM') == os.path.join(home, 'eggs')


def test_expandpath_plain_no_expansion():
    assert expandpath('foo') == 'foo'


def test_expandpath_tilde_only(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    home = os.path.expanduser('~')
    assert expandpath('~') == home


def test_expandpath_env_var_only(monkeypatch):
    monkeypatch.setenv('EGGS', 'ham')
    assert expandpath('$EGGS') == 'ham'


def test_expandpath_undefined_var_stays(monkeypatch):
    monkeypatch.delenv('NOT_A_REAL_VAR_XYZ', raising=False)
    result = expandpath('$NOT_A_REAL_VAR_XYZ')
    # expandvars leaves undefined variables untouched
    assert result == '$NOT_A_REAL_VAR_XYZ'


def test_expandpath_env_var_in_middle(monkeypatch):
    monkeypatch.setenv('DIR', 'sub')
    assert expandpath('a/$DIR/b') == 'a/sub/b'


def test_expandpath_pathlike(tmp_path):
    # PathLike with no special tokens should round-trip to its string form
    assert expandpath(tmp_path) == str(tmp_path)
