import os
import sys
import warnings

# MLX closes a command buffer once its inputs exceed this many MB. Every decode layer reads the
# whole KV pool, so the default cuts one buffer per layer and async_eval blocks until the GPU
# catches up, which serializes scheduling with execution. MLX only reads it at import time.
if "mlx.core" in sys.modules and "MLX_MAX_MB_PER_BUFFER" not in os.environ:
    warnings.warn("Import macsgl before mlx (or set MLX_MAX_MB_PER_BUFFER) to overlap scheduling")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", str(1 << 20))
