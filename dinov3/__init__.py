# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

__version__ = "0.0.1"

import torch

# PyTorch 2.6+ compatibility: default weights_only to False to allow custom checkpoints loading.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        try:
            return _orig_torch_load(*args, weights_only=False, **kwargs)
        except TypeError:
            pass
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

