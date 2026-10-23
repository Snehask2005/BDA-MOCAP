import os
import sys

# Ensure PySpark workers use the current active Python interpreter
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
