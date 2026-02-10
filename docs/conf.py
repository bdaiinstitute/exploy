# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from unittest.mock import MagicMock

# Add the parent directory of exporter packages to the path
# This allows "import exporter" and "import exporter_frameworks" to work
sys.path.insert(0, os.path.abspath("../exporter"))


# Mock isaaclab and isaacsim to avoid import errors
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = [
    "isaaclab",
    "isaacsim",
    "isaaclab.app",
    "isaaclab.envs",
    "isaaclab.sim",
    "isaaclab_tasks",
    "isaaclab_rl",
    "isaaclab_rl.rsl_rl",
    "rsl_rl",
    "rsl_rl.runners",
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Exporter"
copyright = "2026, RAI Institute"
author = "RAI Institute"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "breathe",
]

# Breathe configuration for C++ documentation
breathe_projects = {"Exporter": "_build/doxygen/xml"}
breathe_default_project = "Exporter"
breathe_default_members = ("members", "undoc-members")

# Add external documentation links for Eigen types
breathе_domain_by_extension = {
    "hpp": "cpp",
    "cpp": "cpp",
}

# Custom configuration for linking Eigen documentation
breathе_show_enumvalue_initializer = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Skip isaaclab and isaacsim modules in autodoc
autodoc_mock_imports = ["isaaclab", "isaacsim", "isaaclab_tasks", "isaaclab_rl", "rsl_rl"]


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip documenting isaaclab and isaacsim related members."""
    # Skip if the module is from isaaclab or isaacsim
    if hasattr(obj, "__module__"):
        module = obj.__module__
        if module and any(mod in module for mod in ["isaaclab", "isaacsim"]):
            return True
    return skip


def add_eigen_links(app, doctree, docname):
    """Add hyperlinks to Eigen documentation for common Eigen types."""
    from docutils import nodes

    eigen_base_url = "https://eigen.tuxfamily.org/dox/"

    # Mapping of Eigen types to their documentation URLs
    eigen_types = {
        "Eigen::Vector3d": eigen_base_url + "group__matrixtypedefs.html",
        "Eigen::Quaterniond": eigen_base_url + "classEigen_1_1Quaternion.html",
        "Eigen::VectorXd": eigen_base_url + "group__matrixtypedefs.html",
        "Eigen::MatrixXd": eigen_base_url + "group__matrixtypedefs.html",
        "Eigen::Matrix": eigen_base_url + "classEigen_1_1Matrix.html",
    }

    for node in doctree.traverse(nodes.Text):
        text = node.astext()
        for eigen_type, url in eigen_types.items():
            if eigen_type in text:
                # Replace text node with a reference
                parent = node.parent
                if parent and not isinstance(parent, nodes.reference):
                    parts = text.split(eigen_type)
                    if len(parts) > 1:
                        new_nodes = []
                        for _i, part in enumerate(parts[:-1]):
                            if part:
                                new_nodes.append(nodes.Text(part))
                            ref = nodes.reference("", eigen_type, refuri=url, internal=False)
                            ref += nodes.Text(eigen_type)
                            new_nodes.append(ref)
                        if parts[-1]:
                            new_nodes.append(nodes.Text(parts[-1]))
                        parent.replace(node, new_nodes)
                        break


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("doctree-resolved", add_eigen_links)


# -- Options for napoleon (Google/NumPy docstring style) --------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Note: Eigen uses Doxygen, not Sphinx, so intersphinx won't work.
# For Eigen cross-references, use direct links or configure Doxygen tag files in breathe_doxygen_config_options

# -- GitHub Pages configuration ----------------------------------------------
# For hosting on GitHub Pages, we need to add a .nojekyll file
# This will be handled by adding it to the _build/html directory after build
