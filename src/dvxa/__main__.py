"""Allow ``python -m dvxa`` as well as the ``dvxa`` console script.

On a cluster the console script often lands in a ``bin`` directory that is not
on ``PATH`` (``pip install --user``, a module-loaded interpreter, a compute node
with a different environment).  ``python -m dvxa`` only needs the interpreter
that already imports the package, which also makes
``mpirun -n 128 python -m dvxa run big.yaml`` reliable across nodes.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
