import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np
import torch
from datasets import IterableDataset  # type: ignore[import]
from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint  # type: ignore[import]
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]

from thesis.sae.sae import SparseAutoencoder  # type: ignore[import]

from ..device import Device, get_device
from .sample_loader import SampleDataset
from .weighted_samples_store import WeightedSamplesStore


@dataclass
class MASParams:
    """
    Parameters for the MAS algorithm.

    Args:
        high_activation_weighting: How much to prefer samples with high activation.
        sample_overlap: The number of tokens that overlap between samples.
        num_max_samples: The number of samples to store per feature.
        sample_length_pre: The number of tokens to store before the high activation token.
        sample_length_post: The number of tokens to store after the high activation token.
        samples_to_check: The number of samples to check.
        seed: The seed to use for sampling.
        activation_bins: The bins to use for the activation histogram.
    """

    high_activation_weighting: float
    sample_overlap: int
    num_max_samples: int
    sample_length_pre: int
    sample_length_post: int
    samples_to_check: int
    seed: int
    activation_bins: list[float]

    def __post_init__(self):
        if self.sample_overlap < 0:
            raise ValueError("Sample overlap must be at least 0.")
        if self.num_max_samples <= 0:
            raise ValueError("Number of max samples must be greater than 0.")
        if self.sample_length_pre < 0:
            raise ValueError("Sample length pre must be at least 0.")
        if self.sample_length_post <= 0:
            raise ValueError("Sample length post must be greater than 0.")
        if self.samples_to_check <= 0:
            raise ValueError("Samples to check must be greater than 0.")


@dataclass
class MASLayer:
    """
    A description of a layer to include in the MAS algorithm and how to handle it.

    Attributes:
        hook_id: The transformer_lens to get activations for this layer from.
        num_features: The number of features to store for this layer.
        activation_map: A function that takes the activations from the given hook and returns activations for the
                features of the layer.
                The last dimension of the output must always match the `num_features` attribute.
    """

    hook_id: str
    num_features: int
    activation_map: Callable[
        [Float[Tensor, "batch context neurons_in_hook"]], Float[Tensor, "batch context _num_features"]
    ]

    @staticmethod
    def from_hook_id(hook_id: str, num_features: int) -> "MASLayer":
        return MASLayer(hook_id, num_features, lambda x: x)

    @staticmethod
    def from_sae(sae: SparseAutoencoder) -> "MASLayer":
        return MASLayer(sae.hook_point, sae.num_features, sae.encode)


def run(
    model: HookedTransformer, dataset: IterableDataset, layers: list[MASLayer], params: MASParams, device: Device
) -> WeightedSamplesStore:
    with torch.no_grad():
        device = get_device()
        print(f"Using device: {device.torch()}")

        dataset = dataset.take(params.samples_to_check)  # type: ignore[reportUnknownMemberType]

        if model.tokenizer is None:
            raise ValueError("Model must have tokenizer.")
        if model.tokenizer.pad_token_id is None:
            raise ValueError("Model tokenizer must have pad token.")
        if model.cfg is None:
            raise ValueError("Model must have config.")

        context_size = model.cfg.n_ctx
        print(f"Model context size: {context_size}")

        sample_dataset = SampleDataset(context_size, params.sample_overlap, model, dataset)

        num_total_features = sum([layer.num_features for layer in layers])
        rng = random.Random(params.seed)
        activation_bins = np.arange(0, 3, 0.1)
        mas_store = WeightedSamplesStore(
            list(activation_bins),
            params.high_activation_weighting,
            params.num_max_samples,
            num_total_features,
            context_size,
            params.sample_length_pre,
            params.sample_length_post,
            model.tokenizer.pad_token_id,
            rng,
            device,
        )

        activation_scratch = torch.zeros((context_size, num_total_features), device=device.torch())

        def create_hook(
            layer: MASLayer, slice: slice
        ) -> Tuple[str, Callable[[Float[Tensor, "batch context neurons_per_layer"], Any], None]]:
            assert layer.num_features == slice.stop - slice.start

            def hook(activation: Float[Tensor, "batch context neurons_per_layer"], hook: HookPoint) -> None:
                activation_scratch[:, slice] = layer.activation_map(activation)[0, :, :]

            return (layer.hook_id, hook)

        indices = np.cumsum([0] + [layer.num_features for layer in layers])
        slices = [slice(start, end) for start, end in zip(indices[:-1], indices[1:], strict=True)]
        hooks = [create_hook(layer, slice) for layer, slice in zip(layers, slices, strict=True)]

        last_percentage = -1

        model_time = 0.0
        mas_time = 0.0
        start_time = time.time()
        for i, sample in itertools.islice(enumerate(sample_dataset), params.samples_to_check):
            model_start_time = time.time()
            model.run_with_hooks(sample.tokens, fwd_hooks=hooks)
            model_time += time.time() - model_start_time
            mas_start_time = time.time()
            mas_store.add_sample(sample, activation_scratch)
            mas_time += time.time() - mas_start_time
            assert mas_store.num_samples_added() == i + 1

            cur_percentage = int(math.floor(i / params.samples_to_check * 100))
            if cur_percentage > last_percentage:
                print(f"{cur_percentage}%")
                last_percentage = cur_percentage
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")
        print(f"Time taken per sample: {(end_time - start_time) / (params.samples_to_check)*1000:.2f}ms")

        print(f"Model time: {model_time:.2f}s ({model_time/(end_time - start_time)*100:.2f}%)")
        print(f"MAS time: {mas_time:.2f}s ({mas_time/(end_time - start_time)*100:.2f}%)")

        mas_store._sort_samples()
        return mas_store
