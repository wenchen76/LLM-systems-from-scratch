from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def sample_batch(
    tokens: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = torch.randint(len(tokens) - context_length, (batch_size,))
    inputs = torch.stack([
        torch.from_numpy(tokens[i : i + context_length].astype(np.int64))
        for i in start_indices
    ])
    targets = torch.stack([
        torch.from_numpy(tokens[i + 1 : i + 1 + context_length].astype(np.int64))
        for i in start_indices
    ])
    use_pinned = torch.device(device).type == "cuda"
    if use_pinned:
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs = inputs.to(device)
        targets = targets.to(device)
    return inputs, targets
