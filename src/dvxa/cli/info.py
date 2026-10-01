"""``dvxa info`` -- runtime environment."""

from __future__ import annotations

import os
import platform
import sys

from .. import __version__
from ..kernels import backend_name
from ..parallel import PARALLEL, RANK, SIZE, numba_status
from . import EXIT_OK


def execute(args) -> int:
    import numpy
    import scipy

    print(f"  dvxa            {__version__}")
    print(f"  python          {sys.version.split()[0]}  ({platform.platform()})")
    print(f"  numpy           {numpy.__version__}")
    print(f"  scipy           {scipy.__version__}")
    for name, mod in (("matplotlib", "matplotlib"), ("PyYAML", "yaml"),
                      ("mpi4py", "mpi4py")):
        try:
            m = __import__(mod)
            print(f"  {name:<15} {getattr(m, '__version__', 'installed')}")
        except ImportError:
            print(f"  {name:<15} not installed")
    print(f"  numba           {numba_status()}")
    print(f"  active backend  {backend_name()}")
    print(f"  MPI             {SIZE} rank(s)" + (f"  (this is rank {RANK})" if PARALLEL else ""))
    threads = {k: os.environ.get(k) for k in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS")}
    # `"prefix " + joined or fallback` parses as `("prefix " + joined) or fallback`,
    # and the prefix alone is truthy, so the fallback could never fire and an
    # unset environment printed a bare "thread limits" with nothing after it.
    limits = ", ".join(f"{k}={v}" for k, v in threads.items() if v)
    print(f"  thread limits   {limits or '(unset, so the BLAS decides)'}")
    try:
        cfg = numpy.__config__.show(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
        print(f"  BLAS            {blas}")
    except Exception:
        pass
    return EXIT_OK
