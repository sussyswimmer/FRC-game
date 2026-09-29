"""FRC 2026 REBUILT match simulator and RL environments."""

import os

# The simulator works on small arrays, where multithreaded BLAS only adds overhead; with many
# worker processes it can also exhaust memory (OpenBLAS starts one thread per core per process).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0"
