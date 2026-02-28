# Release Process

## Versioning

easylora follows [Semantic Versioning](https://semver.org/):

- **Patch** (0.1.x): bug fixes, documentation
- **Minor** (0.x.0): new features, backwards-compatible
- **Major** (x.0.0): breaking changes

The version is defined in `pyproject.toml` and read at runtime via
`importlib.metadata`.

## Release Steps

1. Update `CHANGELOG.md` with the new version number and date.
2. Bump `version` in `pyproject.toml`.
3. Commit: `git commit -am "chore: release v0.x.0"`
4. Tag: `git tag v0.x.0`
5. Push: `git push origin main --tags`
6. The `release.yml` workflow will:
    - Build the distribution
    - Create a GitHub Release with auto-generated notes
    - (When configured) Publish to PyPI via trusted publishing

## PyPI Trusted Publishing Setup

To enable automatic PyPI publishing without API tokens:

1. Create the package on [PyPI](https://pypi.org/).
2. Go to the package's "Publishing" settings.
3. Add a "trusted publisher":
    - Repository: `alexsuw/easylora`
    - Workflow: `release.yml`
    - Environment: `pypi`
4. Uncomment the `pypi-publish` job in `.github/workflows/release.yml`.
5. Create a `pypi` environment in GitHub repository settings.

See [PyPI trusted publishing docs](https://docs.pypi.org/trusted-publishers/)
for details.

## Manual Steps After GitHub Configuration

These require the GitHub web UI:

- [ ] Enable GitHub Pages (Settings > Pages > Source: GitHub Actions)
- [ ] Enable GitHub Discussions (Settings > General > Features)
- [ ] Configure trusted publishing on PyPI (see above)
- [ ] Add repository topics/description on GitHub
