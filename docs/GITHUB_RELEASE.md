# GitHub release checklist

## First publication

From the repository root:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```

## Do not commit

The `.gitignore` excludes common experiment artifacts, but verify the staged files before every push:

```bash
git status
git diff --cached --stat
```

Do not commit:

- raw IRMA or MURA images;
- local manifests containing private/local paths unless intentionally anonymized for release;
- large checkpoints unless the repository is specifically intended to host them;
- intermediate `.npz` indexes;
- local environment folders;
- temporary output directories.

## Recommended public reproducibility files

Commit or attach to a release:

- the source code at the exact paper version;
- manifest-generation scripts;
- class-mapping files that do not violate dataset terms;
- configuration files;
- final routing parameter CSVs;
- aggregate result tables;
- environment/version information;
- a tagged release matching the manuscript revision.

## Versioning

For the first paper-associated release:

```bash
git tag -a v1.0.0 -m "Paper experiment release"
git push origin v1.0.0
```

If the manuscript changes after peer review, create a new tag rather than rewriting an existing public tag.
