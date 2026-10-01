# 0001. Flat layout and package rename

- Status: Accepted
- Date: 2026-10-01

## Context

The package lives in `src/dvxa/` (src layout). The first proposed new name,
`pyxa`, is already taken on PyPI, as is `xalpha`. Names checked free on PyPI on
2026-10-01: `pydvxa`, `dvxalpha`, `pydvxalpha`, `pyxalpha`, `xalphamo`, `xamo`,
`pyxamo`, `slaterxa`, `xalphacluster`, `xaclust`, `discretexa`,
`discretexalpha`, `varxa`, `xalphakit`.

## Decision

- New name: **`pyxany`** ("pyxa" + the author's initials "ny"), free on
  PyPI as of 2026-10-01. Names containing "dv" were declined by the author.
- Move to the **flat layout**: `pyxany/` at the repository root, keeping the
  existing subpackages (`basis/`, `grid/`, `scf/`, `cli/`, ...). Subpackages are
  not merged into single modules.
- Distribution name, import name, CLI command and `python -m` entry all become
  the new name. No `dvxa` compatibility alias (0.1.0 alpha, no external users).
- The repository directory `dv-xa/` is not renamed.
- The move is done with `git mv` in a commit of its own, after the pending
  working-tree changes are committed separately, so rename detection keeps
  file history.
- Paths in `.code-check/history.md` are rewritten to the new layout.
- Stray optimisation output (`test_traj.xyz` at the root) is removed and
  `*_traj.xyz` is ignored, except deliberate files under `examples/`.
- Nothing is pushed to GitHub as part of this change.

## Consequences

Imports change from `dvxa.x` to `pyxany.x` everywhere, including tests, docs,
CI and `CITATION.cff`. Users of the old CLI must switch commands.
