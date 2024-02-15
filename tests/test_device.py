import torch

from thesis.device import get_device


def test_device():
    device = get_device()
    if torch.cuda.is_available():
        assert device.torch() == torch.device("cuda")
    else:
        assert device.torch() == torch.device("cpu")
