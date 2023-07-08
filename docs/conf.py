# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from unittest import mock

os.environ["FACESWAP_BACKEND"] = "nvidia"
sys.path.insert(0, os.path.abspath('../'))
sys.setrecursionlimit(1500)


MOCK_MODULES = ["pynvx", "ctypes.windll", "comtypes"]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- Project information -----------------------------------------------------

project = 'faceswap'
copyright = '2022, faceswap.dev'
author = 'faceswap.dev'

# The full version, including alpha/beta/rc tags
release = '0.99'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon', "sphinx.ext.autosummary", ]
napoleon_custom_sections = ['License']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'analytics_id': 'UA-145659566-2',
    'logo_only': True,
    # Toc options
    'navigation_depth': -1,
}
html_logo = '_static/logo.png'
latext_logo = '_static/logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = 'index'

autosummary_generate = True
