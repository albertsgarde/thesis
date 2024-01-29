from enum import Enum

import torch


class Device(Enum):
    CPU = 0
    GPU = 1

    def __str__(self) -> str:
        match self:
            case Device.CPU:
                return "cpu"
            case Device.GPU:
                return "cuda"

    def torch(self) -> torch.device:
        return torch.device(str(self))

    def lightning(self) -> str:
        match self:
            case Device.CPU:
                return "cpu"
            case Device.GPU:
                return "gpu"


def get_device() -> Device:
    """Return the current device."""
    return Device.GPU if torch.cuda.is_available() else Device.CPU
