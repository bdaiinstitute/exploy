<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Documentation

This directory contains the Sphinx documentation for the Exporter
project.

## Building the Documentation

To build the documentation, run:

```bash
pixi run docs
```

The built HTML documentation will be in `_build/html/`. Open
`_build/html/index.html` in a browser to view it.

To clean the build directory:

```bash
pixi run docs-clean
```

## Hosting on GitHub Pages

To host the documentation on GitHub Pages:

1. Build the documentation:

   ```bash
   pixi run docs
   ```

2. Create a `.nojekyll` file in the output directory:

   ```bash
   touch docs/_build/html/.nojekyll
   ```

3. Configure GitHub Pages in your repository settings:
   - Go to Settings → Pages
   - Select "Deploy from a branch"
   - Choose the branch where you'll commit the built docs (e.g., `gh-pages`)

4. Option A: Manual deployment

   ```bash
   # Copy _build/html to a gh-pages branch
   git checkout --orphan gh-pages
   git rm -rf .
   cp -r docs/_build/html/* .
   git add .
   git commit -m "Update documentation"
   git push origin gh-pages
   ```

5. Option B: Automated with GitHub Actions (recommended)
   - Create `.github/workflows/docs.yml` to build and deploy automatically
   - The workflow will build docs on push and deploy to GitHub Pages

## Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `getting_started.rst` - Getting started guide
- `api/` - API reference documentation
- `_static/` - Static files (images, CSS, etc.)
- `_templates/` - Custom templates
- `_build/` - Generated documentation (gitignored)
