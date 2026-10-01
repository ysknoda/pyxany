"""Hierarchical wall-clock timing for every stage of a calculation.

Usage::

    from pyxany.timing import timer, timed, REGISTRY

    with timer("scf.iteration.vxc"):
        ...

    @timed("integrals.build_S")
    def build_S(...): ...

    REGISTRY.report()            # tree table on rank 0
    REGISTRY.to_dict()           # JSON-serialisable

Sections nest by call stack, so ``self`` time (a section's own time minus the
time spent in its children) is reported alongside cumulative time.  Under MPI
the report aggregates min/mean/max across ranks.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .parallel import PARALLEL, RANK, ROOT

try:  # optional, only used for the peak-RSS line
    import resource as _resource
except ImportError:  # pragma: no cover - Windows
    _resource = None


@dataclass
class Section:
    name: str
    calls: int = 0
    total: float = 0.0       # cumulative wall time including children
    children_time: float = 0.0  # time attributed to nested sections
    depth: int = 0
    order: int = 0

    @property
    def self_time(self) -> float:
        return max(self.total - self.children_time, 0.0)


@dataclass
class TimingRegistry:
    sections: dict[str, Section] = field(default_factory=dict)
    _stack: list[tuple[str, float]] = field(default_factory=list)
    _counter: int = 0
    enabled: bool = True

    def reset(self) -> None:
        self.sections.clear()
        self._stack.clear()
        self._counter = 0

    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        full = ".".join([s[0] for s in self._stack] + [name]) if self._stack else name
        sec = self.sections.get(full)
        if sec is None:
            sec = Section(name=full, depth=len(self._stack), order=self._counter)
            self._counter += 1
            self.sections[full] = sec
        t0 = time.perf_counter()
        self._stack.append((name, t0))
        try:
            yield sec
        finally:
            dt = time.perf_counter() - t0
            self._stack.pop()
            sec.calls += 1
            sec.total += dt
            if self._stack:
                parent_full = ".".join(s[0] for s in self._stack)
                parent = self.sections.get(parent_full)
                if parent is not None:
                    parent.children_time += dt

    # -- reporting ---------------------------------------------------------
    def to_dict(self) -> dict:
        out = {
            "sections": [
                {
                    "name": s.name,
                    "calls": s.calls,
                    "total_s": round(s.total, 6),
                    "self_s": round(s.self_time, 6),
                    "depth": s.depth,
                }
                for s in sorted(self.sections.values(), key=lambda x: x.order)
            ]
        }
        rss = peak_rss_mb()
        if rss is not None:
            out["peak_rss_mb"] = round(rss, 1)
        return out

    def report(self, printer=print, min_seconds: float = 0.0) -> None:
        """Print a nested table of timings, deepest-first within each parent."""
        if not self.sections:
            return
        ordered = sorted(self.sections.values(), key=lambda s: s.order)
        total_top = sum(s.total for s in ordered if s.depth == 0)

        printer("")
        printer("  TIMING REPORT")
        printer("  " + "-" * 74)
        printer(f"  {'section':<40} {'calls':>7} {'total (s)':>11} {'self (s)':>10} {'%':>5}")
        printer("  " + "-" * 74)
        for s in ordered:
            if s.total < min_seconds:
                continue
            label = "  " * s.depth + s.name.split(".")[-1]
            pct = 100.0 * s.total / total_top if total_top > 0 else 0.0
            printer(f"  {label:<40} {s.calls:>7} {s.total:>11.3f} {s.self_time:>10.3f} {pct:>5.1f}")
        printer("  " + "-" * 74)
        printer(f"  {'TOTAL':<40} {'':>7} {total_top:>11.3f}")
        rss = peak_rss_mb()
        if rss is not None:
            printer(f"  peak RSS: {rss:.1f} MB" + (f"  (rank {RANK})" if PARALLEL else ""))

    def report_mpi(self, printer=print) -> None:
        """Aggregate per-section times across ranks and report min/mean/max."""
        if not PARALLEL:
            self.report(printer=printer)
            return
        from .parallel import COMM

        local = {s.name: s.total for s in self.sections.values()}
        gathered = COMM.gather(local, root=ROOT)
        if RANK != ROOT:
            return
        names: list[str] = []
        for d in gathered:
            for k in d:
                if k not in names:
                    names.append(k)
        printer("")
        printer(f"  TIMING REPORT  ({len(gathered)} MPI ranks)")
        printer("  " + "-" * 74)
        printer(f"  {'section':<40} {'min (s)':>10} {'mean (s)':>10} {'max (s)':>10}")
        printer("  " + "-" * 74)
        for name in names:
            vals = [d.get(name, 0.0) for d in gathered]
            depth = name.count(".")
            label = "  " * depth + name.split(".")[-1]
            printer(f"  {label:<40} {min(vals):>10.3f} "
                    f"{sum(vals) / len(vals):>10.3f} {max(vals):>10.3f}")
        printer("  " + "-" * 74)


REGISTRY = TimingRegistry()


def timer(name: str):
    """Context manager timing a named section (nests automatically)."""
    return REGISTRY(name)


def timed(name: str | None = None):
    """Decorator form of :func:`timer`."""

    def deco(fn):
        label = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with REGISTRY(label):
                return fn(*args, **kwargs)

        return wrapper

    return deco


def peak_rss_mb() -> float | None:
    """Peak resident set size in MB, or None if unavailable."""
    if _resource is None:  # pragma: no cover
        return None
    usage = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    import sys

    return usage / (1024.0 * 1024.0) if sys.platform == "darwin" else usage / 1024.0
