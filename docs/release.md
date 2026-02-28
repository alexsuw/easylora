# Release Process

## Versioning

easylora follows [Semantic Versioning](https://semver.org/):

- **Patch** (0.1.x): bug fixes, documentation
- **Minor** (0.x.0): new features, backwards-compatible
- **Major** (x.0.0): breaking changes

The version is defined once in `pyproject.toml` and read at runtime via
`importlib.metadata`.

## Release Checklist

### 1. Prepare the release

- [ ] All CI checks are green on `main`.
- [ ] Update `CHANGELOG.md`: move items from "Unreleased" to the new version
      section with today's date.
- [ ] Bump `version` in `pyproject.toml` (e.g. `0.1.0` -> `0.1.1`).
- [ ] Commit:

```bash
git commit -am "chore: release v0.1.1"
```

### 2. Tag and push

```bash
git tag v0.1.1
git push origin main --tags
```

### 3. Automated workflow

The `release.yml` workflow triggers on the tag push and automatically:

1. Builds the sdist and wheel.
2. Validates with `twine check`.
3. Publishes to **PyPI** via OIDC trusted publishing (no API tokens).
4. Creates a **GitHub Release** with auto-generated notes and attached
   dist artifacts.

### 4. Verify

- Check the [Actions tab](https://github.com/alexsuw/easylora/actions/workflows/release.yml)
  for a successful run.
- Verify the package on [PyPI](https://pypi.org/project/easylora/).
- Test installation: `pip install easylora==0.1.1`

---

## TestPyPI (Pre-release Testing)

You can publish to TestPyPI first to verify the package before a real release.

### Publish to TestPyPI

1. Go to [Actions > Release > Run workflow](https://github.com/alexsuw/easylora/actions/workflows/release.yml).
2. Select `testpypi` as the target.
3. Click "Run workflow".

### Install from TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ easylora
```

The `--extra-index-url` flag is needed because easylora's dependencies
(torch, transformers, etc.) are only on the real PyPI.

---

## PyPI Trusted Publishing Setup (One-Time)

Trusted publishing uses OpenID Connect (OIDC) so the GitHub Actions workflow
can publish to PyPI without any API tokens or secrets stored in the repo.

### First-time setup for PyPI

1. Go to [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/).
2. Under **"Add a new pending publisher"** (if the project doesn't exist yet)
   or in the project's **Publishing** settings (if it does):
    - **PyPI project name**: `easylora`
    - **Owner**: `alexsuw`
    - **Repository**: `easylora`
    - **Workflow name**: `release.yml`
    - **Environment name**: `pypi`
3. Click "Add".

### First-time setup for TestPyPI

1. Go to [https://test.pypi.org/manage/account/publishing/](https://test.pypi.org/manage/account/publishing/).
2. Add a pending publisher with:
    - **PyPI project name**: `easylora`
    - **Owner**: `alexsuw`
    - **Repository**: `easylora`
    - **Workflow name**: `release.yml`
    - **Environment name**: `testpypi`
3. Click "Add".

### GitHub Environments

Create two environments in GitHub repository settings
(**Settings > Environments**):

1. **`pypi`** — optionally add required reviewers for production releases.
2. **`testpypi`** — no reviewers needed (for testing).

No secrets need to be added to either environment. OIDC handles authentication
automatically.

### How it works

```
┌──────────────┐    OIDC token    ┌────────────┐
│ GitHub Action │ ──────────────> │   PyPI     │
│ release.yml   │  (no secrets)   │            │
└──────────────┘                  └────────────┘
```

The `pypa/gh-action-pypi-publish` action requests a short-lived OIDC token
from GitHub, which PyPI validates against the trusted publisher configuration.
No long-lived API tokens are involved.

---

## Manual Steps After First Setup

These only need to be done once via the web UI:

- [ ] Create `pypi` environment in GitHub repo settings
- [ ] Create `testpypi` environment in GitHub repo settings
- [ ] Add pending publisher on [PyPI](https://pypi.org/manage/account/publishing/)
- [ ] Add pending publisher on [TestPyPI](https://test.pypi.org/manage/account/publishing/)
- [ ] Enable GitHub Pages (Settings > Pages > Source: GitHub Actions) — already done
- [ ] Enable GitHub Discussions (Settings > General > Features) — already done
