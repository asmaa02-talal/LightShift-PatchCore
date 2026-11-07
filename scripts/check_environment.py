import platform
import sys

import numpy as np
import torch

print("Python:", sys.version)
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Mode CPU : OK pour les tests et petits essais.")

try:
    import anomalib
    print("Anomalib: OK", getattr(anomalib, "__version__", "version non exposée"))
except Exception as exc:
    print("Anomalib import failed:", repr(exc))
    raise
