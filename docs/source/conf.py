"""Sphinx configuration of the TxCellNet documentation (built on Read the Docs from .readthedocs.yaml).

Local build, from the repository root:
    pip install -r docs/requirements.txt
    sphinx-build -b html docs/source docs/build/html
"""
import inspect
import os
import re
import sys

from sphinx.ext.autodoc.mock import ismock, undecorate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)  # autodoc imports txcellnet from the repository; no installation needed

# -- Project information
project = 'TxCellNet'
author = 'Rui Wang, Hongliang Zhou, Sourav Medya, Hao Chen'
copyright = '2026, Rui Wang'
with open(os.path.join(ROOT, 'pyproject.toml')) as f:
    release = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M).group(1)
version = release

# -- General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'myst_nb',
]
exclude_patterns = ['_build', '**.ipynb_checkpoints']
default_role = 'literal'  # `name` in the package docstrings is code

# Tutorials: notebooks stored with the outputs of a real run, rendered as they are (never executed here)
nb_execution_mode = 'off'
nb_merge_streams = True
myst_enable_extensions = ['dollarmath']

# API: torch and the other heavy imports are mocked, so the documentation builds without them
autodoc_mock_imports = ['torch', 'cv2', 'numpy', 'scipy', 'pandas']
autodoc_member_order = 'bysource'

# -- HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {'navigation_depth': 3}
html_context = {
    'display_github': True,
    'github_user': 'wrab12',
    'github_repo': 'TxCellNet',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}


def plain_docstring(app, what, name, obj, options, lines):
    """The package docstrings are plain text: a name ending in '_' (e.g. the argument C_) is not a link reference."""
    lines[:] = [re.sub(r'(?<=\w)_(?=[\s:,.;)]|$)', r'\\_', line) for line in lines]


def init_signature(app, what, name, obj, options, signature, return_annotation):
    """The networks subclass the mocked torch.nn.Module: take the class signature from the class's own __init__."""
    if what == 'class' and '__init__' in vars(obj):
        sig = inspect.signature(obj.__init__)
        return str(sig.replace(parameters=list(sig.parameters.values())[1:])), return_annotation


def follow_decorated(app, modname, fullname):
    """refine and sample are wrapped by the mocked torch.no_grad: link their source through the wrapped function."""
    obj = getattr(sys.modules.get(modname), fullname, None)
    return getattr(undecorate(obj), '__module__', None) if ismock(obj) else None


def setup(app):
    app.connect('autodoc-process-docstring', plain_docstring)
    app.connect('autodoc-process-signature', init_signature)
    app.connect('viewcode-follow-imported', follow_decorated)
