# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the parent directory of exporter packages to the path
# This allows "import exploy.exporter.core" and "import exploy.exporter.frameworks" to work
sys.path.insert(0, os.path.abspath("../exploy"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Exploy"
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
    "myst_parser",
]

# Breathe configuration for C++ documentation
breathe_projects = {"Exporter": "_build/doxygen/xml"}
breathe_default_project = "Exporter"
breathe_default_members = ("members", "undoc-members")

# Add external documentation links for Eigen types
breathe_domain_by_extension = {
    "hpp": "cpp",
    "cpp": "cpp",
}

# Custom configuration for linking Eigen documentation
breathe_show_enumvalue_initializer = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82b1ff",
        "color-brand-content": "#82b1ff",
    },
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/bdaiinstitute/exploy",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Mock external dependencies that are not available in the docs environment
autodoc_mock_imports = [
    "gymnasium",
    "isaaclab",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaacsim",
    "rsl_rl",
]


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip inherited members that come from mocked external packages."""
    if hasattr(obj, "__module__"):
        module = obj.__module__ or ""
        # Skip members from external packages, but keep our own isaaclab framework modules
        if not module.startswith("exploy") and any(
            mod in module for mod in ["isaaclab", "isaacsim"]
        ):
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


def fix_readme_doc_links(app, doctree):
    """Strip the leading 'docs/' from doc links included from outside the Sphinx source tree.

    Files like README.md use paths relative to the repo root (e.g. docs/tutorial/foo.md),
    but Sphinx resolves links relative to the docs/ source directory, so the
    prefix needs to be removed for internal cross-references to work.
    """
    from sphinx.addnodes import pending_xref

    for node in doctree.traverse(pending_xref):
        target = node.get("reftarget", "")
        if target.startswith("docs/"):
            target = target[len("docs/") :]
            if target.endswith(".md"):
                target = target[: -len(".md")]
            node["reftarget"] = target


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("doctree-read", fix_readme_doc_links)
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
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- MyST parser configuration ------------------------------------------------
# Generate heading anchors up to depth 4 so that #anchor links in markdown work.
myst_heading_anchors = 4

# Suppress cross-reference warnings for links that are valid on GitHub but cannot
# resolve inside the Sphinx doc tree (e.g., README links to files at the repo root).
suppress_warnings = ["myst.xref_missing"]

# Links that are valid but not yet publicly available (e.g. unreleased paths on GitHub)
linkcheck_ignore = [
    # The repository is private; GitHub returns 404 for unauthenticated requests.
    r"https://github\.com/bdaiinstitute/exploy/",
]

# GitHub renders anchors client-side, so Sphinx linkcheck cannot verify them.
linkcheck_anchors_ignore_for_url = [
    r"https://github\.com/",
]

# Note: Eigen uses Doxygen, not Sphinx, so intersphinx won't work.
# For Eigen cross-references, use direct links or configure Doxygen tag files in breathe_doxygen_config_options

# -- GitHub Pages configuration ----------------------------------------------
# For hosting on GitHub Pages, we need to add a .nojekyll file
# This will be handled by adding it to the _build/html directory after build
