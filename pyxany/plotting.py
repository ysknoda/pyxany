"""Summary figure for a completed calculation (matplotlib optional)."""

from __future__ import annotations

import numpy as np

from .constants import HARTREE_TO_EV
from .properties import (
    density_of_states,
    mulliken,
    mulliken_orbital_populations,
    wiberg_bond_orders,
)

_BG = "#0d1117"
_PANEL = "#161b22"
_FG = "white"
_DIM = "#b0b8c0"
_EDGE = "#30363d"


def summary_figure(result, config, path, forces=None) -> None:
    """Six-panel overview: levels, DOS, MO composition, charges, bonds, convergence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    mol, basis = result.molecule, result.basis
    eps_ev = result.eps * HARTREE_TO_EV
    finite = np.isfinite(eps_ev)
    occ = result.occupations
    homo, lumo = result.homo_lumo()
    homo_ev, lumo_ev = homo * HARTREE_TO_EV, lumo * HARTREE_TO_EV
    colours = plt.cm.Set2(np.linspace(0, 1, max(len(mol), 3)))

    fig = plt.figure(figsize=(19, 10), facecolor=_BG)
    fig.suptitle(
        f"pyxany  |  {mol.name}   basis {config.basis.set.upper()}   "
        f"alpha_xc {config.scf.alpha}   E = {result.total_energy:.6f} Ha",
        fontsize=13, color=_FG, fontweight="bold", y=0.975)
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.34,
                          left=0.05, right=0.97, top=0.90, bottom=0.07)
    ax_lev = fig.add_subplot(gs[:, 0])
    ax_dos = fig.add_subplot(gs[:, 1])
    ax_mo = fig.add_subplot(gs[0, 2])
    ax_q = fig.add_subplot(gs[1, 2])
    ax_bo = fig.add_subplot(gs[0, 3])
    ax_cv = fig.add_subplot(gs[1, 3])
    for ax in (ax_lev, ax_dos, ax_mo, ax_q, ax_bo, ax_cv):
        _style(ax)

    # -- orbital levels ----------------------------------------------------
    # Core levels sit hundreds of eV below the valence band; plotting the full
    # range compresses everything of interest into a few pixels, so the window
    # is set from the valence region and the number of hidden core levels is
    # stated instead.
    order = np.argsort(np.where(finite, eps_ev, np.inf))
    n_occ = int((occ > 1e-8).sum())
    lo_ev, hi_ev, n_core = _valence_window(eps_ev, finite, occ, homo_ev, lumo_ev)
    shown = 0
    for rank, i in enumerate(order[: min(n_occ + 8, int(finite.sum()))]):
        if not (lo_ev <= eps_ev[i] <= hi_ev):
            continue
        shown += 1
        col = "#4fc3f7" if occ[i] > 1e-8 else "#ef9a9a"
        lw = 2.6 if rank in (n_occ - 1, n_occ) else 1.4
        ax_lev.hlines(eps_ev[i], 0.15, 0.85, colors=col, linewidths=lw)
        ax_lev.text(0.87, eps_ev[i], f"{rank + 1}", color=_DIM, fontsize=6,
                    va="center")
    if np.isfinite(homo_ev) and np.isfinite(lumo_ev):
        ax_lev.axhspan(homo_ev, lumo_ev, alpha=0.15, color="#ffd54f")
        ax_lev.text(0.99, 0.5 * (homo_ev + lumo_ev),
                    f"gap\n{lumo_ev - homo_ev:.2f} eV",
                    color="#ffd54f", fontsize=8, va="center")
    ax_lev.set_ylim(lo_ev, hi_ev)
    ax_lev.set_xticks([])
    ax_lev.set_ylabel("energy (eV)")
    title = "orbital levels"
    if n_core:
        title += f"  ({n_core} core level{'s' if n_core > 1 else ''} below {lo_ev:.0f} eV)"
    ax_lev.set_title(title, fontsize=10)
    ax_lev.set_xlim(0, 1.35)
    ax_lev.legend(handles=[mpatches.Patch(color="#4fc3f7", label="occupied"),
                           mpatches.Patch(color="#ef9a9a", label="virtual")],
                  fontsize=8, facecolor=_BG, edgecolor=_EDGE, labelcolor=_DIM)

    # -- density of states -------------------------------------------------
    dos_cfg = config.properties.dos
    in_window = finite & (eps_ev >= lo_ev) & (eps_ev <= hi_ev)
    e, dos = density_of_states(eps_ev[in_window], sigma=dos_cfg.get("sigma", 0.3),
                               n_points=dos_cfg.get("n_points", 600),
                               e_min=lo_ev, e_max=hi_ev)
    ax_dos.fill_betweenx(e, dos, alpha=0.35, color="#64b5f6")
    ax_dos.plot(dos, e, color="#90caf9", lw=1.4)
    for value, col, lbl in ((homo_ev, "#4fc3f7", "HOMO"), (lumo_ev, "#ef9a9a", "LUMO")):
        if np.isfinite(value) and lo_ev <= value <= hi_ev:
            ax_dos.axhline(value, color=col, ls="--", lw=1, alpha=0.75)
            ax_dos.text(0, value, f" {lbl}", color=col, fontsize=8, va="bottom")
    ax_dos.set_ylim(lo_ev, hi_ev)
    ax_dos.set_xlabel("DOS (states/eV)")
    ax_dos.set_ylabel("energy (eV)")
    ax_dos.set_title("density of states (valence region)", fontsize=10)

    # -- MO composition ----------------------------------------------------
    pops = mulliken_orbital_populations(result.C, result.S, occ, basis, mol)
    if len(pops):
        x = np.arange(1, len(pops) + 1)
        bottom = np.zeros(len(pops))
        for ai, atom in enumerate(mol):
            ax_mo.bar(x, pops[:, ai], bottom=bottom, color=colours[ai],
                      width=0.75, label=f"{atom.symbol}{ai + 1}")
            bottom += pops[:, ai]
        ax_mo.set_xlabel("occupied MO (1-based, as in the text report)")
        ax_mo.set_ylabel("population")
        if len(mol) <= 8:
            ax_mo.legend(fontsize=7, facecolor=_BG, edgecolor=_EDGE, labelcolor=_DIM,
                         ncol=2)
    ax_mo.set_title("MO composition")

    # -- Mulliken charges --------------------------------------------------
    pop = mulliken(result.P, result.S, basis, mol)
    labels = [f"{a.symbol}{i + 1}" for i, a in enumerate(mol)]
    ax_q.bar(range(len(mol)), pop["charges"],
             color=["#ef9a9a" if q > 0 else "#81c784" for q in pop["charges"]])
    ax_q.axhline(0, color=_DIM, lw=0.8)
    ax_q.set_xticks(range(len(mol)))
    ax_q.set_xticklabels(labels, fontsize=max(5, min(8, 90 // max(len(mol), 1))),
                         rotation=90 if len(mol) > 10 else 0)
    ax_q.set_ylabel("net charge (e)")
    ax_q.set_title("Mulliken charges")

    # -- Wiberg bond orders ------------------------------------------------
    # The diagonal is an atom's self-term, not a bond order, and is one to two
    # orders of magnitude larger; leaving it in flattens every real bond to the
    # bottom of the colour scale.
    bo = wiberg_bond_orders(result.P, result.S, basis, mol)
    off_diagonal = bo.copy().astype(float)
    np.fill_diagonal(off_diagonal, np.nan)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(_PANEL)
    vmax = float(np.nanmax(off_diagonal)) if len(mol) > 1 else 1.0
    im = ax_bo.imshow(off_diagonal, cmap=cmap, vmin=0, vmax=max(vmax, 1e-6),
                      aspect="auto")
    fs = max(4, min(8, 80 // max(len(mol), 1)))
    ax_bo.set_xticks(range(len(mol)))
    ax_bo.set_xticklabels(labels, fontsize=fs, rotation=90)
    ax_bo.set_yticks(range(len(mol)))
    ax_bo.set_yticklabels(labels, fontsize=fs)
    if len(mol) <= 8:
        for i in range(len(mol)):
            for j in range(len(mol)):
                if i == j:
                    continue
                ax_bo.text(j, i, f"{bo[i, j]:.2f}", ha="center", va="center",
                           fontsize=fs, color="black")
    ax_bo.set_title("Wiberg bond orders (diagonal omitted)", fontsize=10)
    plt.colorbar(im, ax=ax_bo, fraction=0.046, pad=0.04)

    # -- SCF convergence ---------------------------------------------------
    hist = result.history
    if hist:
        it = [h["iter"] for h in hist]
        de = [max(h["dE"], 1e-16) if np.isfinite(h["dE"]) else 1e0 for h in hist]
        dp = [max(h["d_rms"], 1e-16) for h in hist]
        ax_cv.semilogy(it, de, "o-", ms=4, lw=1.2, color="#ffb74d", label="|dE|")
        ax_cv.semilogy(it, dp, "s-", ms=3, lw=1.0, color="#81c784", label="rms dP")
        ax_cv.axhline(config.scf.e_tol, color="#ef9a9a", ls="--", lw=1,
                      label=f"e_tol {config.scf.e_tol:.0e}")
        ax_cv.set_xlabel("SCF iteration")
        ax_cv.set_ylabel("convergence measure")
        ax_cv.legend(fontsize=7, facecolor=_BG, edgecolor=_EDGE, labelcolor=_DIM)
    ax_cv.set_title("SCF convergence")

    fig.savefig(path, dpi=140, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)


def _valence_window(eps_ev, finite, occ, homo_ev, lumo_ev, span_below=35.0,
                    span_above=15.0):
    """Energy window covering the valence region, plus the core-level count.

    Anchored on the HOMO so that the window follows the system rather than a
    fixed range, and clipped to the levels that actually exist.
    """
    values = eps_ev[finite]
    if len(values) == 0:
        return -1.0, 1.0, 0
    anchor = homo_ev if np.isfinite(homo_ev) else float(values.max())
    lo = anchor - span_below
    hi = (lumo_ev if np.isfinite(lumo_ev) else anchor) + span_above
    if values.min() >= lo:
        lo = values.min() - 0.05 * max(hi - values.min(), 1.0)
    n_core = int((values < lo).sum())
    return float(lo), float(hi), n_core


def _style(ax) -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_EDGE)
    ax.xaxis.label.set_color(_DIM)
    ax.yaxis.label.set_color(_DIM)
    ax.title.set_color(_FG)
