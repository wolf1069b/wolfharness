# Changelog Conventions

## Structure

```
changelog/
├── README.md             # This file
├── unreleased/           # Work-in-progress changes for the next version
├── 2.10.0/               # Released version folder
│   ├── README.md         # Summary: one line per change, linking detail files
│   ├── RELEASE.md        # GitHub Release body (written at release time)
│   ├── YYYY-MM-DD-short-slug.md  # Detail file per change entry
│   └── ...
└── ...
```

## Rules

### Work in progress goes in `unreleased/`

- Never in a numbered folder — the next version number is not knowable while work is being written.
- At release, rename `unreleased/` to the version actually shipped, swap its `# Unreleased` heading for `# Version X.Y.Z` plus a `Released on <date>.` line, write `RELEASE.md`, and create a fresh empty `unreleased/`.

### Released folders are frozen

Once a version ships, its folder is never modified.

### One detail file per logical change

- Entries are grouped by surface: Core, Protocol Servers, CLI, SDK, Docs, Tooling, etc.
- A related change extends the existing file instead of opening a new one.
- File naming: `YYYY-MM-DD-short-slug.md`

### RELEASE.md

- Written at release time, before the git tag is created.
- Contents: lead sentence, then `## Install`, `## Highlights`, `## Notable in this release`, `## Requirements`.

### Summary README.md per version

- One brief line per change, each linking its detail file:
  ```
  - [YYYY-MM-DD] Brief description. ([details](YYYY-MM-DD-short-slug.md))
  ```

### Backward compatibility

A batch that carries backward-compatibility handling collects it in one `YYYY-MM-DD-backward-compatibility.md`. The surface entries reference that file instead of re-explaining it.

### Language

Written in English. History starts at v2.9.5; earlier changes are not backfilled.