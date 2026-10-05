"""
On Windows, the bare "python"/"python3" command can resolve to the
Microsoft Store's App Execution Alias stub instead of the active venv's
real interpreter -- even with the venv activated -- depending on PATH
ordering. PySpark uses that bare command to launch its worker
subprocesses unless told otherwise, so any Spark action that needs a
worker (collecting results back to Python, for example) hangs and then
fails with "Python worker failed to connect back".

Pin PySpark's worker interpreter explicitly to whatever interpreter is
currently running pytest. Harmless on macOS/Linux -- sys.executable is
just the normal interpreter there, so this is a no-op in practice.
setdefault() so it never overrides an explicit setting someone already
made.
"""

import os
import sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
