"""
Minimal smoke tests — these do NOT test GPT-2 implementation.

As you implement each component, add your own shape and behavior tests.
"""

import torch as t

from model import get_device


def test_torch_tensor_shape():
    x = t.zeros((2, 3))
    assert x.shape == (2, 3)


def test_device_selection_returns_torch_device():
    assert isinstance(get_device(), t.device)
