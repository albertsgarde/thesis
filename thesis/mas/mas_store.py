import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..device import Device
from .sample_loader import Sample


class MASStore:
    _feature_samples: Int[Tensor, "num_features num_samples context"]
    _feature_activations: Float[Tensor, "num_features num_samples context"]
    _feature_max_activations: Float[Tensor, " num_features num_samples"]

    _num_samples_added: int

    def __init__(self, num_samples: int, num_features: int, context_size: int, device: Device) -> None:
        self._feature_samples = torch.zeros(
            size=(num_features, num_samples, context_size), dtype=torch.int64, device=device.torch()
        )
        self._feature_activations = torch.zeros(
            size=(num_features, num_samples, context_size), dtype=torch.float32, device=device.torch()
        )
        self._feature_activations.fill_(-float("inf"))
        self._feature_max_activations = torch.zeros(
            size=(num_features, num_samples), dtype=torch.float32, device=device.torch()
        )
        self._feature_max_activations.fill_(-float("inf"))

        self._num_samples_added = 0

    def num_features(self) -> int:
        return self._feature_samples.shape[0]

    def num_samples_per_feature(self) -> int:
        return self._feature_samples.shape[1]

    def num_samples_added(self) -> int:
        return self._num_samples_added

    def feature_samples(self) -> Int[Tensor, "num_features num_samples_added context"]:
        self._feature_max_activations, sorted_indices = self._feature_max_activations.sort(dim=1, descending=True)

        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, self._feature_samples.size(-1))
        self._feature_samples = torch.gather(self._feature_samples, 1, expanded_indices)
        self._feature_activations = torch.gather(self._feature_activations, 1, expanded_indices)

        return self._feature_samples[:, : self.num_samples_added(), :]

    def feature_activations(self) -> Float[Tensor, "num_features num_samples_added context"]:
        self._feature_max_activations, sorted_indices = self._feature_max_activations.sort(dim=1, descending=True)

        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, self._feature_samples.size(-1))
        self._feature_samples = torch.gather(self._feature_samples, 1, expanded_indices)
        self._feature_activations = torch.gather(self._feature_activations, 1, expanded_indices)

        return self._feature_activations[:, : self.num_samples_added(), :]

    def add_sample(
        self,
        sample: Sample,
        activations: Float[Tensor, "context num_features"],
    ) -> None:
        # Inputs must be padded to context length
        overlap = sample.overlap
        tokens = sample.tokens

        assert not torch.isinf(activations).any(), "Infinite activations found"
        assert not torch.isnan(activations).any(), "NaN activations found"
        assert tokens.shape[-1] == self._feature_samples.shape[-1], "Sample tokens must be padded to context length"

        max_activations: Float[Tensor, " num_features"]
        max_activations, _ = activations[overlap:, :].max(dim=-2)

        min_cur_activations: Float[Tensor, " num_features"]
        min_cur_activation_indices: Int[Tensor, " num_features"]
        min_cur_activations, min_cur_activation_indices = self._feature_max_activations.min(dim=-1)

        replace_mask: Bool[Tensor, " num_features"] = max_activations > min_cur_activations

        self._feature_samples[replace_mask, min_cur_activation_indices[replace_mask], :] = tokens
        self._feature_activations[replace_mask, min_cur_activation_indices[replace_mask], :] = activations.transpose(
            0, 1
        )[replace_mask, :]
        self._feature_max_activations[replace_mask, min_cur_activation_indices[replace_mask]] = max_activations[
            replace_mask
        ]

        self._num_samples_added += 1

        assert (self._feature_activations == -float("inf")).any(dim=0).any(dim=1).count_nonzero() == max(
            self.num_samples_per_feature() - self.num_samples_added(), 0
        )
