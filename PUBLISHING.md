# Publishing streamcutter to PyPI

This guide explains how to publish the `streamcutter` package to PyPI for the first time and for subsequent releases using [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (no API tokens required).

## Prerequisites

- A [PyPI account](https://pypi.org/account/register/)
- Maintainer access to the [salmanhiro/GC-tidal](https://github.com/salmanhiro/GC-tidal) repository on GitHub

---

## Step 1 — Create the package on PyPI (first release only)

PyPI trusted publishing can be configured *before* the package exists ("pending publisher"), so you only need to do this once.

1. Log in to <https://pypi.org>.
2. Go to **Your account → Publishing** (direct link: <https://pypi.org/manage/account/publishing/>).
3. Scroll down to **"Add a new pending publisher"** and fill in the form:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `streamcutter` |
   | Owner | `salmanhiro` |
   | Repository name | `GC-tidal` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

4. Click **Add**.

> **Why "pending"?** A pending publisher lets you configure trusted publishing before the project exists on PyPI. The first successful workflow run will both create the project and upload the files.

---

## Step 2 — Create the `pypi` environment in GitHub

The publish workflow (`publish.yml`) runs in a GitHub Actions environment named `pypi`. This environment acts as a gate — only workflows that reference it receive the short-lived OIDC token that PyPI accepts.

1. On GitHub, go to **Settings → Environments** in the `salmanhiro/GC-tidal` repository (direct link: <https://github.com/salmanhiro/GC-tidal/settings/environments>).
2. Click **New environment** and enter the name `pypi`.
3. (Optional but recommended) Under **Deployment protection rules**, enable **Required reviewers** and add yourself so every publish requires manual approval.
4. Click **Configure environment** to save.

---

## Step 3 — Tag a release and push

The publish workflow triggers on any tag that starts with `v`. To publish version `0.1.0`:

```bash
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions will:
1. Build the source distribution (`sdist`) and wheel.
2. Upload both files to PyPI using the trusted OIDC token — no stored secrets needed.

After the workflow completes successfully the package will be available at <https://pypi.org/project/streamcutter/>.

---

## Subsequent releases

To publish a new version:

1. Update `version` in `pyproject.toml` and `__version__` in `src/streamcutter/__init__.py`.
2. Commit and push the change.
3. Create and push a new tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The workflow runs automatically and uploads the new version to PyPI.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Workflow fails with `invalid-publisher` | Double-check the five fields in the PyPI trusted publisher form match the workflow exactly (case-sensitive). |
| Workflow fails with `environment not found` | Create the `pypi` environment in GitHub repository settings (Step 2). |
| Version already exists on PyPI | PyPI does not allow re-uploading the same version. Bump the version number, retag, and re-push. |
