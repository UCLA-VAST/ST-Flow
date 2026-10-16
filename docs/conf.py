"""Sphinx configuration for the ST-Flow documentation (https://st-flow.readthedocs.io).

Build locally:
    pip install -r docs/requirements.txt
    sphinx-build -W -b html docs docs/_build/html
"""

import os
import re

project = "ST-Flow"
author = "Suhail Basalama, Stéphane Pouget, Jason Cong (UCLA VAST Lab)"
copyright = "2026, " + author  # noqa: A001

extensions = [
    "myst_parser",  # Markdown sources, including the repo-root README / AE guide
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "ae/**"]

# Generate anchors for #..#### so in-page links like (#pinned-ml-toolchain) resolve.
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline"]

# ---------------------------------------------------------------------------
# Links into the repository.
#
# README.md and ARTIFACT_EVALUATION.md are included verbatim from the repo root, so
# that GitHub, the Zenodo tarball and this site all show the same text. Their links
# to repository files (README.md, docs/ae/checklist.md, ...) are rewritten here to
# GitHub URLs at the version being built: a branch or tag on Read the Docs, `main`
# locally.
# ---------------------------------------------------------------------------
GITHUB = "https://github.com/UCLA-VAST/ST-Flow"
GIT_REF = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")

_REPO_LINK = re.compile(
    r"\]\((?!https?://|#|mailto:)"
    r"((?:README|ARTIFACT_EVALUATION|LICENSE|NOTICE|CITATION)[^)]*"
    r"|(?:docs|scripts|stflow|test|mlir|extern|PACT26_Artifacts)/[^)]*)\)"
)


def _to_github(match):
    target = match.group(1)
    path = target.split("#", 1)[0].rstrip("/")
    kind = "tree" if target.split("#", 1)[0].endswith("/") else "blob"
    return f"]({GITHUB}/{kind}/{GIT_REF}/{target if kind == 'blob' else path})"


# README anchors whose section is its own page on this site.
_README_SECTIONS = {
    "(#building-and-running)": "(installation.md)",
}


# Pages whose included README section sits one heading level deeper than the page
# title (the section's own "## ..." heading is replaced by the page title).
_PROMOTE_HEADINGS = {"installation"}


def _rewrite_repo_links(app, relative_path, parent_docname, content):
    text = _REPO_LINK.sub(_to_github, content[0])
    for anchor, page in _README_SECTIONS.items():
        text = text.replace(anchor, page)
    if parent_docname in _PROMOTE_HEADINGS:
        text = re.sub(r"(?m)^#(#{2,}) ", r"\1 ", text)
    content[0] = text


def setup(app):
    app.connect("include-read", _rewrite_repo_links)


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "ST-Flow"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": GITHUB,
    "source_branch": GIT_REF,
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#2774AE",  # UCLA blue
        "color-brand-content": "#2774AE",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8BB8E8",
        "color-brand-content": "#8BB8E8",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": GITHUB,
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                "8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01"
                ".37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01"
                " 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89"
                "-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64"
                "-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92"
                ".08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0"
                ' 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z">'
                "</path></svg>"
            ),
            "class": "",
        },
    ],
}
pygments_style = "friendly"
pygments_dark_style = "monokai"
