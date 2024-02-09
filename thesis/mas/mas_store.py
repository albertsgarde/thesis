import torch
from jaxtyping import Bool, Float, Int
from line_profiler import profile
from torch import Tensor

from ..device import Device
from .sample_loader import Sample


class MASStore:
    _feature_samples: Int[Tensor, "num_features num_samples sample_length"]
    _feature_activations: Float[Tensor, "num_features num_samples sample_length"]
    _feature_max_activations: Float[Tensor, " num_features num_samples"]

    _num_samples_added: int

    _sample_length_pre: int
    _sample_length_post: int

    _pad_token_id: int

    _device: Device

    def __init__(
        self,
        num_samples: int,
        num_features: int,
        context_size: int,
        sample_length_pre: int,
        sample_length_post: int,
        pad_token_id: int,
        device: Device,
    ) -> None:
        assert sample_length_pre <= context_size, "Pre sample length should always be less than context size."
        assert (
            sample_length_post > 0
        ), "Post sample length should always be greater than 0 in order to include the activating token."
        assert sample_length_post <= context_size, "Post sample length should always be less than context size."

        sample_length = min(sample_length_pre + sample_length_post, context_size)

        self._feature_samples = torch.zeros(
            size=(num_features, num_samples, sample_length),
            dtype=torch.int64,
            device=device.torch(),
        )
        self._feature_activations = torch.zeros(
            size=(num_features, num_samples, sample_length),
            dtype=torch.float32,
            device=device.torch(),
        )
        self._feature_activations.fill_(-float("inf"))
        self._feature_max_activations = torch.zeros(
            size=(num_features, num_samples), dtype=torch.float32, device=device.torch()
        )
        self._feature_max_activations.fill_(-float("inf"))

        self._num_samples_added = 0

        self._sample_length_pre = sample_length_pre
        self._sample_length_post = sample_length_post

        self._pad_token_id = pad_token_id

        self._device = device

    def num_features(self) -> int:
        return self._feature_samples.shape[0]

    def num_samples_per_feature(self) -> int:
        return self._feature_samples.shape[1]

    def sample_length(self) -> int:
        return self._feature_samples.shape[2]

    def num_samples_added(self) -> int:
        return self._num_samples_added

    def feature_samples(self) -> Int[Tensor, "num_features num_samples_added sample_length"]:
        self._feature_max_activations, sorted_indices = self._feature_max_activations.sort(dim=1, descending=True)

        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, self._feature_samples.size(-1))
        self._feature_samples = torch.gather(self._feature_samples, 1, expanded_indices)
        self._feature_activations = torch.gather(self._feature_activations, 1, expanded_indices)

        return self._feature_samples[:, : self.num_samples_added(), :]

    def feature_activations(self) -> Float[Tensor, "num_features num_samples_added sample_length"]:
        self._feature_max_activations, sorted_indices = self._feature_max_activations.sort(dim=1, descending=True)

        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, self._feature_samples.size(-1))
        self._feature_samples = torch.gather(self._feature_samples, 1, expanded_indices)
        self._feature_activations = torch.gather(self._feature_activations, 1, expanded_indices)

        return self._feature_activations[:, : self.num_samples_added(), :]

    @profile
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

        first_pad_tensor: Int[Tensor, " num_pad"] = torch.where(tokens == self._pad_token_id)[0]
        first_pad: int
        if first_pad_tensor.numel() == 0:
            first_pad = tokens.shape[0]
        else:
            first_pad = int(first_pad_tensor[0].item())

        assert (
            first_pad >= overlap
        ), "The overlap should not contain any pad tokens, as then there would be no point in this sample."
        assert (
            first_pad >= overlap + 1
        ), "The first token after overlap should not be a pad token, since that effectively means this sample is empty."

        max_activations: Float[Tensor, " num_features"]
        max_activating_indices: Int[Tensor, " num_features"]
        max_activations, max_activating_indices = activations[overlap:first_pad, :].max(dim=-2)
        max_activating_indices += overlap

        sample_starts: Int[Tensor, " num_features"] = torch.maximum(
            torch.tensor([0], device=self._device.torch()),
            torch.minimum(
                max_activating_indices - self._sample_length_pre,
                torch.tensor([tokens.shape[0] - self.sample_length()], device=self._device.torch()),
            ),
        )

        assert (
            sample_starts + self.sample_length() <= tokens.shape[0]
        ).all(), "Sample starts should be within the token length"

        sample_tokens: Int[Tensor, "num_features sample_length"] = tokens.unfold(0, self.sample_length(), 1)[
            sample_starts
        ]
        sample_activations: Float[Tensor, "num_features sample_length"] = activations.transpose(1, 0)[
            torch.arange(self.num_features(), device=self._device.torch())[:, None].expand(
                self.num_features(), self.sample_length()
            ),
            sample_starts[:, None] + torch.arange(self.sample_length(), device=self._device.torch())[None, :],
        ]

        min_cur_activations: Float[Tensor, " num_features"]
        min_cur_activation_indices: Int[Tensor, " num_features"]
        min_cur_activations, min_cur_activation_indices = self._feature_max_activations.min(dim=-1)

        replace_mask: Bool[Tensor, " num_features"] = max_activations > min_cur_activations

        self._feature_samples[replace_mask, min_cur_activation_indices[replace_mask], :] = sample_tokens[
            replace_mask, :
        ]
        self._feature_activations[replace_mask, min_cur_activation_indices[replace_mask], :] = sample_activations[
            replace_mask, :
        ]
        self._feature_max_activations[replace_mask, min_cur_activation_indices[replace_mask]] = max_activations[
            replace_mask
        ]

        self._num_samples_added += 1

        assert (self._feature_activations == -float("inf")).any(dim=0).any(dim=1).count_nonzero() == max(
            self.num_samples_per_feature() - self.num_samples_added(), 0
        )
