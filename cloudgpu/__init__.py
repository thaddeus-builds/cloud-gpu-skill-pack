"""cloudgpu: run ComfyUI workflows (MiniMax-H3 first) on a GPU you rent from any vendor that gives SSH. See README.md and docs/cloud-gpu.md."""

__version__ = "0.3.0"

from .box import Box                  # noqa: F401
from .errors import CloudGpuError     # noqa: F401
from .pool import Pool, boxes_from  # noqa: F401
