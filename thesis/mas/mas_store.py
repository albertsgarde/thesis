import pickle
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import torch
from jaxtyping import Bool, Float, Int
from line_profiler import profile
from torch import Tensor

from ..device import Device
from .sample_loader import Sample


class MASStore:
    _feature_samples: Int[Tensor, "num_features num_samples sample_length"]
    _feature_activations: Float[Tensor, "num_features num_samples sample_length"]
    _feature_max_activations: Float[Tensor, "num_features num_samples"]

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

    def save(self, file_name: Path) -> None:
        values_dict = {
            "num_samples_added": self._num_samples_added,
            "sample_length_pre": self._sample_length_pre,
            "sample_length_post": self._sample_length_post,
            "pad_token_id": self._pad_token_id,
        }
        ints_serialized = pickle.dumps(values_dict)

        tensors_dict: dict[str, Tensor] = {
            "feature_samples": self._feature_samples.int(),
            "feature_activations": self._feature_activations,
            "feature_max_activations": self._feature_max_activations,
        }

        tensors_serialized: dict[str, BytesIO] = {key: BytesIO() for key in tensors_dict.keys()}

        for key, value in tensors_dict.items():
            torch.save(value, tensors_serialized[key])

        with ZipFile(file_name, "w") as zip_file:
            zip_file.writestr("ints", ints_serialized)
            for key, value in tensors_serialized.items():  # type: ignore
                zip_file.writestr(key, value.getvalue())  # type: ignore

    @staticmethod
    def load(file_name: Path, device: Device) -> "MASStore":
        with ZipFile(file_name, "r") as zip_file:
            ints_serialized = zip_file.read("ints")
            values_dict = pickle.loads(ints_serialized)

            tensors_serialized = {name: zip_file.read(name) for name in zip_file.namelist() if name != "ints"}
            tensors_dict = {
                name: torch.load(BytesIO(tensors_serialized[name]), map_location=device.torch())
                for name in tensors_serialized
            }

        mas_store = MASStore.__new__(MASStore)
        mas_store._feature_samples = tensors_dict["feature_samples"].long()
        mas_store._feature_activations = tensors_dict["feature_activations"]
        mas_store._feature_max_activations = tensors_dict["feature_max_activations"]

        mas_store._num_samples_added = values_dict["num_samples_added"]
        mas_store._sample_length_pre = values_dict["sample_length_pre"]
        mas_store._sample_length_post = values_dict["sample_length_post"]
        mas_store._pad_token_id = values_dict["pad_token_id"]
        mas_store._device = device

        return mas_store

    def num_features(self) -> int:
        return self._feature_samples.shape[0]

    def num_samples_per_feature(self) -> int:
        return self._feature_samples.shape[1]

    def sample_length(self) -> int:
        return self._feature_samples.shape[2]

    def num_samples_added(self) -> int:
        return self._num_samples_added

    def _sort_samples(self) -> None:
        self._feature_max_activations, sorted_indices = self._feature_max_activations.sort(dim=1, descending=True)

        prev_samples_sum = self._feature_samples.sum(dim=1)

        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, self._feature_samples.size(-1))
        assert (expanded_indices == expanded_indices[:, :, 0][:, :, None]).all()
        self._feature_samples = torch.gather(self._feature_samples, 1, expanded_indices)
        self._feature_activations = torch.gather(self._feature_activations, 1, expanded_indices)

        assert (prev_samples_sum == self._feature_samples.sum(dim=1)).all(), "Activations should not change"

    def feature_samples(self) -> Int[Tensor, "num_features num_samples_added sample_length"]:
        """
        Sorts all data by max activations and returns the samples.
        """
        self._sort_samples()

        return self._feature_samples[:, : self.num_samples_added(), :]

    def feature_activations(self) -> Float[Tensor, "num_features num_samples_added sample_length"]:
        """
        Sorts all data by max activations and returns the activations.
        """
        self._sort_samples()

        return self._feature_activations[:, : self.num_samples_added(), :]

    def feature_max_activations(self) -> Float[Tensor, "num_features num_samples_added"]:
        """
        Sorts all data by max activations and returns the max activations.
        """
        self._sort_samples()

        return self._feature_max_activations[:, : self.num_samples_added()]

    @profile
    def add_sample(
        self,
        sample: Sample,
        activations: Float[Tensor, "context num_features"],
    ) -> None:
        # Inputs must be padded to context length
        overlap = sample.overlap
        tokens: Int[Tensor, " context"] = sample.tokens

        assert not torch.isinf(activations).any(), "Infinite activations found"
        assert not torch.isnan(activations).any(), "NaN activations found"

        # Find max activating sample for each feature, ignoring the overlap region,
        # since that will have been handled better in another sample.
        max_activations: Float[Tensor, " num_features"]
        max_activating_indices: Int[Tensor, " num_features"]
        max_activations, max_activating_indices = activations[overlap : sample.length, :].max(dim=-2)
        max_activating_indices += overlap

        # Find the starting index for the MAS sample for each feature.
        # The sample should start at the maximum activating index minus the pre sample length.
        # However, this is sometimes not possible, either because the index would be negative or
        # because there would not be room after the index for the an entire sample.
        # In that case, we clamp to the closest valid index.
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
        assert (sample_starts >= 0).all(), "Sample starts should be non-negative"

        # Unfold creates a view of the tokens tensor where each possible sample is a row.
        # The correct sample for each feature is selected by indexing with the sample starts.
        sample_tokens: Int[Tensor, "num_features sample_length"] = tokens.unfold(0, self.sample_length(), 1)[
            sample_starts
        ]
        assert sample_tokens.shape == (self.num_features(), self.sample_length())

        # Because activations are already a 2D tensor with a column for each feature, we cannot use the same approach.
        # Instead, we index with two tensors.
        sample_activations: Float[Tensor, "num_features sample_length"] = activations[
            sample_starts[:, None] + torch.arange(self.sample_length(), device=self._device.torch())[None, :],
            torch.arange(self.num_features(), device=self._device.torch())[:, None].expand(
                self.num_features(), self.sample_length()
            ),
        ]
        assert sample_activations.shape == (self.num_features(), self.sample_length())

        # Get minimum maximum activations of all currently stored samples for each feature.
        min_cur_max_activations: Float[Tensor, " num_features"]
        min_cur_max_activation_indices: Int[Tensor, " num_features"]
        min_cur_max_activations, min_cur_max_activation_indices = self._feature_max_activations.min(dim=-1)
        assert min_cur_max_activations.shape == (self.num_features(),)
        assert min_cur_max_activation_indices.shape == (self.num_features(),)

        # True for each feature where the new sample has a higher max activation
        # than the lowest currently stored sample.
        replace_mask: Bool[Tensor, " num_features"] = max_activations > min_cur_max_activations

        # Replace the lowest currently stored samples with the new sample.
        self._feature_samples[replace_mask, min_cur_max_activation_indices[replace_mask], :] = sample_tokens[
            replace_mask, :
        ]
        self._feature_activations[replace_mask, min_cur_max_activation_indices[replace_mask], :] = sample_activations[
            replace_mask, :
        ]
        self._feature_max_activations[replace_mask, min_cur_max_activation_indices[replace_mask]] = max_activations[
            replace_mask
        ]

        self._num_samples_added += 1

        assert (self._feature_activations == -float("inf")).any(dim=0).any(dim=1).count_nonzero() == max(
            self.num_samples_per_feature() - self.num_samples_added(), 0
        )
