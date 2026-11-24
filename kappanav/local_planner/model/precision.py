"""BF16 neural regions with FP32 metric/Flow arithmetic and master parameters."""

import torch

NEURAL_DTYPE = torch.bfloat16
GEOMETRY_DTYPE = torch.float32
