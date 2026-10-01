"""Allow ``python -m pyxany`` as well as the ``pyxany`` console script.

On a cluster the console script often lands in a ``bin`` directory that is not
on ``PATH`` (``pip install --user``, a module-loaded interpreter, a compute node
with a different environment).  ``python -m pyxany`` only needs the interpreter
that already imports the package, which also makes
``mpirun -n 128 python -m pyxany run big.yaml`` reliable across nodes.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
