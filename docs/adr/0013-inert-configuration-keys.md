# 0013. Configuration keys that are validated but never read

- Status: Accepted (not yet implemented; roadmap item F3, ADR 0012)
- Date: 2026-11-13
- Decisions: Q67, Q73

## Context

`config.py` promises that a key either does something or is an error. On
2026-11-13 seven keys were validated, documented in `docs/input_format.md`,
and read nowhere outside `config.py`:

| key | what happens today |
|---|---|
| `runtime.numba` (`auto`/`on`/`off`) | ignored; only `--no-numba` (which sets `PYXANY_DISABLE_NUMBA`) has an effect, and `--numba` does nothing |
| `runtime.mpi` | ignored; MPI is active whenever the process was started under `mpirun` |
| `runtime.verbosity` | overwritten by `cli/_shared.prepare` from `-v`/`-q` on every run |
| `passivation.scheme` | ignored; terminators are recognized from the `.xyz` labels and `passivation.species` |
| `passivation.basis` (`derived`) | ignored; pseudo-hydrogen exponents are always derived |
| `properties.bde.enabled` | ignored; no bond-dissociation code exists |
| `properties.esp.*` | ignored; no electrostatic-potential code exists |

The 2026-08-27 review rated the same pattern (`scf.checkpoint_every`) HIGH: an
input that silently does nothing is worse than one that is rejected.

## Decision (Q67: decide per key)

| key | decision |
|---|---|
| `runtime.numba` | **Wire it up.** `off` sets `PYXANY_DISABLE_NUMBA=1` before the first kernel call; `on` is an input error if numba cannot be imported; `auto` keeps today's behavior. `--numba` / `--no-numba` map to `on` / `off` and win over the file. |
| `runtime.verbosity` | **Wire it up** as the default when neither `-v` nor `-q` is given. |
| `runtime.mpi` | **Remove.** The launcher decides. The key becomes an error that says so ("run under mpirun/mpiexec for MPI; a serial run is a run without it"). |
| `passivation.scheme`, `passivation.basis` | **Remove** (only one behavior exists for each). `pyxany passivate --emit-yaml` stops writing them; the four passivated examples and `docs/input_format.md` drop them. Old files get an error naming the reason, as the renames of ADR 0009 and the American-spelling change did. `passivation.species` stays. |
| `properties.bde` | **Remove** (rejected in ADR 0012: separate runs or `pyxany scan` give it). Error names the alternative. |
| `properties.esp` | **Keep the key, implement it with the cube output** (ADR 0012, A2). Until then `enabled: true` is an input error ("not implemented yet, see ADR 0012 A2"); `enabled: false`, the default, is accepted. |

A test pins each case: every removed key is rejected with its message, and
`runtime.numba: off` / `runtime.verbosity: 0` change what a run does.

## Consequences

- Input files that carry `passivation.scheme`/`basis`, `runtime.mpi` or
  `properties.bde` stop loading, with a message that says what to delete. The
  shipped examples are updated in the same change.
- CHANGELOG entry under "Changed", listing the removed keys.
