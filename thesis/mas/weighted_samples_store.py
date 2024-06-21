import pickle
import random
from io import BytesIO
from pathlib import Path
from random import Random
from typing import Sequence
from zipfile import ZipFile

import torch
from jaxtyping import Bool, Float, Int
from line_profiler import profile
from torch import Tensor

from ..device import Device
from .sample_loader import Sample


class WeightedSamplesStore:
    _rng: Random

    _activation_bins: Float[Tensor, " num_activation_bins+1"]

    _feature_samples: Int[Tensor, "num_features num_samples sample_length"]
    _feature_activations: Float[Tensor, "num_features num_samples sample_length"]
    _feature_max_activations: Float[Tensor, "num_features num_samples"]
    _feature_sample_keys: Float[Tensor, "num_features num_samples"]
    _feature_activation_densities: Int[Tensor, "num_features num_activation_bins"]

    _high_activation_weighting: float
    _firing_threshold: float

    _num_samples_added: int

    _sample_length_pre: int
    _sample_length_post: int

    _pad_token_id: int

    _device: Device

    def __init__(
        self,
        activation_bins: Sequence[float],
        high_activation_weighting: float,
        firing_threshold: float,
        num_samples: int,
        num_features: int,
        context_size: int,
        sample_length_pre: int,
        sample_length_post: int,
        pad_token_id: int,
        rng: Random,
        device: Device,
    ) -> None:
        assert sample_length_pre <= context_size, "Pre sample length should always be less than context size."
        assert (
            sample_length_post > 0
        ), "Post sample length should always be greater than 0 in order to include the activating token."
        assert sample_length_post <= context_size, "Post sample length should always be less than context size."

        sample_length = min(sample_length_pre + sample_length_post, context_size)

        self._activation_bins = torch.concat(
            [
                torch.tensor([-float("inf")], device=device.torch()),
                torch.tensor(activation_bins, device=device.torch()),
                torch.tensor([float("inf")], device=device.torch()),
            ]
        )

        self._high_activation_weighting = high_activation_weighting
        self._firing_threshold = firing_threshold

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
        self._feature_sample_keys = torch.zeros(
            size=(num_features, num_samples), dtype=torch.float32, device=device.torch()
        )
        self._feature_sample_keys.fill_(-float("inf"))

        self._feature_activation_densities = torch.zeros(
            size=(num_features, self._activation_bins.shape[0] - 1), dtype=torch.int64, device=device.torch()
        )

        self._num_samples_added = 0

        self._sample_length_pre = sample_length_pre
        self._sample_length_post = sample_length_post

        self._pad_token_id = pad_token_id

        self._rng = rng

        self._device = device

    def save(self, file_name: Path) -> None:
        values_dict = {
            "high_activation_weighting": self._high_activation_weighting,
            "firing_threshold": self._firing_threshold,
            "num_samples_added": self._num_samples_added,
            "sample_length_pre": self._sample_length_pre,
            "sample_length_post": self._sample_length_post,
            "pad_token_id": self._pad_token_id,
            "rng_state": self._rng.getstate(),
        }
        ints_serialized = pickle.dumps(values_dict)

        tensors_dict: dict[str, Tensor] = {
            "activation_bins": self._activation_bins,
            "feature_samples": self._feature_samples.int(),
            "feature_activations": self._feature_activations,
            "feature_max_activations": self._feature_max_activations,
            "feature_sample_keys": self._feature_sample_keys,
            "feature_activation_densities": self._feature_activation_densities.int(),
        }

        tensors_serialized: dict[str, BytesIO] = {key: BytesIO() for key in tensors_dict.keys()}

        for key, value in tensors_dict.items():
            torch.save(value, tensors_serialized[key])

        with ZipFile(file_name, "w") as zip_file:
            zip_file.writestr("ints", ints_serialized)
            for key, value in tensors_serialized.items():  # type: ignore
                zip_file.writestr(key, value.getvalue())  # type: ignore

    @staticmethod
    def load(file_name: Path, device: Device) -> "WeightedSamplesStore":
        with ZipFile(file_name, "r") as zip_file:
            ints_serialized = zip_file.read("ints")
            values_dict = pickle.loads(ints_serialized)

            tensors_serialized = {name: zip_file.read(name) for name in zip_file.namelist() if name != "ints"}
            tensors_dict = {
                name: torch.load(BytesIO(tensors_serialized[name]), map_location=device.torch())
                for name in tensors_serialized
            }

        weighted_samples_store = WeightedSamplesStore.__new__(WeightedSamplesStore)
        weighted_samples_store._activation_bins = tensors_dict["activation_bins"]
        weighted_samples_store._feature_samples = tensors_dict["feature_samples"].long()
        weighted_samples_store._feature_activations = tensors_dict["feature_activations"]
        weighted_samples_store._feature_max_activations = tensors_dict["feature_max_activations"]
        weighted_samples_store._feature_sample_keys = tensors_dict["feature_sample_keys"]
        weighted_samples_store._feature_activation_densities = tensors_dict["feature_activation_densities"].long()

        weighted_samples_store._high_activation_weighting = values_dict["high_activation_weighting"]
        weighted_samples_store._firing_threshold = values_dict["firing_threshold"]
        weighted_samples_store._num_samples_added = values_dict["num_samples_added"]
        weighted_samples_store._sample_length_pre = values_dict["sample_length_pre"]
        weighted_samples_store._sample_length_post = values_dict["sample_length_post"]
        weighted_samples_store._pad_token_id = values_dict["pad_token_id"]
        weighted_samples_store._rng = random.Random()
        weighted_samples_store._rng.setstate(values_dict["rng_state"])
        weighted_samples_store._device = device

        return weighted_samples_store

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
        self._feature_sample_keys = torch.gather(self._feature_sample_keys, 1, expanded_indices[:, :, 0])

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

    @property
    def activation_bins(self) -> Float[Tensor, " num_activation_bins-1"]:
        return self._activation_bins

    @property
    def feature_densities(self) -> Int[Tensor, "num_features num_activation_bins"]:
        return self._feature_activation_densities

    @profile
    def add_sample(self, sample: Sample, activations: Float[Tensor, "context num_features"]) -> None:
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

        # A tensor of subsamples for each feature.
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

        assert sample_activations.isnan().count_nonzero() == 0, "NaN activations found"
        assert sample_activations.isinf().count_nonzero() == 0, "Infinite activations found"

        bin_mask = (self._activation_bins[None, None, :-1] <= sample_activations[:, :, None]) & (
            self._activation_bins[None, None, 1:] > sample_activations[:, :, None]
        )
        self._feature_activation_densities += bin_mask.sum(dim=1)
        assert (
            self._feature_activation_densities.sum()
            == (self.num_samples_added() + 1) * self.num_features() * self.sample_length()
        ), "All samples should be added"

        # Calculate the key for the sample for each feature.
        r: float = self._rng.uniform(0, 1)
        keys: Float[Tensor, " num_features"] = (
            torch.pow(r, torch.exp(-self._high_activation_weighting * max_activations))
            - (max_activations < self._firing_threshold).float()
        )

        assert keys.isnan().count_nonzero() == 0, "NaN keys found"
        assert keys.isinf().count_nonzero() == 0, "Infinite keys found"

        # Get minimum maximum activations of all currently stored samples for each feature.
        min_cur_keys: Float[Tensor, " num_features"]
        min_cur_key_indices: Int[Tensor, " num_features"]
        min_cur_keys, min_cur_key_indices = self._feature_sample_keys.min(dim=-1)
        assert min_cur_keys.shape == (self.num_features(),)
        assert min_cur_key_indices.shape == (self.num_features(),)

        # True for each feature where the new sample has a higher max activation
        # than the lowest currently stored sample.
        replace_mask: Bool[Tensor, " num_features"] = keys > min_cur_keys
        if self.num_samples_added() < self.num_samples_per_feature():
            assert replace_mask.count_nonzero() == self.num_features(), "All samples should be added"

        # Replace the lowest currently stored samples with the new sample.
        self._feature_samples[replace_mask, min_cur_key_indices[replace_mask], :] = sample_tokens[replace_mask, :]
        self._feature_activations[replace_mask, min_cur_key_indices[replace_mask], :] = sample_activations[
            replace_mask, :
        ]
        self._feature_max_activations[replace_mask, min_cur_key_indices[replace_mask]] = max_activations[replace_mask]
        self._feature_sample_keys[replace_mask, min_cur_key_indices[replace_mask]] = keys[replace_mask]

        self._num_samples_added += 1

        return

        num_non_inf = (
            self.num_samples_per_feature()
            - (self._feature_activations == -float("inf")).any(dim=0).any(dim=1).count_nonzero()
        )
        assert num_non_inf == min(self.num_samples_added(), self.num_samples_per_feature()), (
            f"Number of non-inf activations: {num_non_inf}\n" f"Number of samples added: {self.num_samples_added()}\n"
        )
