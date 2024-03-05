import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import torch
from datasets import IterableDataset  # type: ignore[import]
from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint  # type: ignore[import]
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]

from ..device import Device, get_device
from .mas_store import MASStore
from .sample_loader import SampleDataset


@dataclass
class MASParams:
    sample_overlap: int
    num_max_samples: int
    sample_length_pre: int
    sample_length_post: int
    samples_to_check: int

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


def run(model: HookedTransformer, dataset: IterableDataset, params: MASParams, device: Device) -> MASStore:
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

        num_layers = model.cfg.n_layers
        neurons_per_layer = model.cfg.d_mlp
        num_model_neurons = num_layers * neurons_per_layer
        mas_store = MASStore(
            params.num_max_samples,
            num_model_neurons,
            context_size,
            params.sample_length_pre,
            params.sample_length_post,
            model.tokenizer.pad_token_id,
            device,
        )

        def create_hooks(
            activation_scratch: Float[Tensor, "context num_features"],
        ) -> list[Tuple[str, Callable[[Float[Tensor, "batch context neurons_per_layer"], Any], None]]]:
            def hook(activation: Float[Tensor, "batch context neurons_per_layer"], hook: HookPoint) -> None:
                activation_scratch[
                    :, neurons_per_layer * hook.layer() : neurons_per_layer * (hook.layer() + 1)
                ] = activation

            return [(f"blocks.{layer}.mlp.hook_mid", hook) for layer in range(num_layers)]

        activation_scratch = torch.zeros((context_size, num_model_neurons), device=device.torch())

        model_time = 0.0
        mas_time = 0.0
        start_time = time.time()
        for i, sample in itertools.islice(enumerate(sample_dataset), params.samples_to_check):
            model_start_time = time.time()
            model.run_with_hooks(sample.tokens, fwd_hooks=create_hooks(activation_scratch))
            model_time += time.time() - model_start_time
            mas_start_time = time.time()
            mas_store.add_sample(sample, activation_scratch)
            mas_time += time.time() - mas_start_time
            assert mas_store.num_samples_added() == i + 1
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")
        print(f"Time taken per sample: {(end_time - start_time) / (params.samples_to_check)*1000:.2f}s")

        print(f"Model time: {model_time:.2f}s ({model_time/(end_time - start_time)*100:.2f}%)")
        print(f"MAS time: {mas_time:.2f}s ({mas_time/(end_time - start_time)*100:.2f}%)")

        return mas_store
