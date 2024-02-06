import itertools
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import datasets  # type: ignore[missingTypeStubs, import-untyped]
import torch
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformer import HookedTransformer

from ..device import get_device
from . import html
from .mas_store import MASStore
from .sample_loader import SampleDataset


@dataclass
class MASConfig:
    model_name: str
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


def run(config: MASConfig):
    with torch.no_grad():
        device = get_device()

        dataset: IterableDataset = datasets.load_dataset(  # type: ignore[reportUnknownMemberType]
            "monology/pile-uncopyrighted", streaming=True, split="train", trust_remote_code=True
        )
        dataset = dataset.take(config.samples_to_check)  # type: ignore[reportUnknownMemberType]

        model: HookedTransformer = HookedTransformer.from_pretrained(config.model_name, device=device.torch())  # type: ignore[reportUnknownVariableType]
        if model.tokenizer is None:
            raise ValueError("Model must have tokenizer.")
        if model.tokenizer.pad_token_id is None:
            raise ValueError("Model tokenizer must have pad token.")
        if model.cfg is None:
            raise ValueError("Model must have config.")

        neuron_index = 0

        context_size = model.cfg.n_ctx
        print(f"Model context size: {context_size}")

        sample_dataset = SampleDataset(context_size, config.sample_overlap, model, dataset)

        num_layers = model.cfg.n_layers
        neurons_per_layer = model.cfg.d_mlp
        num_model_neurons = num_layers * neurons_per_layer
        mas_store = MASStore(
            config.num_max_samples,
            num_model_neurons,
            context_size,
            config.sample_length_pre,
            config.sample_length_post,
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

        activation_scratch = torch.zeros((context_size, num_model_neurons))

        start_time = time.time()
        for i, sample in itertools.islice(enumerate(sample_dataset), config.samples_to_check):
            model.run_with_hooks(sample.tokens, fwd_hooks=create_hooks(activation_scratch))
            mas_store.add_sample(sample, activation_scratch)
            assert mas_store.num_samples_added() == i + 1
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")
        print(f"Time taken per sample: {(end_time - start_time) / (config.samples_to_check):.2f}s")

        os.makedirs("outputs/html", exist_ok=True)

        print("Generating and saving HTML")

        feature_samples = mas_store.feature_samples()
        feature_activations = mas_store.feature_activations()

        for neuron_index in range(40):
            with open(f"outputs/html/{neuron_index}.html", "w", encoding="utf-8") as f:
                html_str = html.generate_html(model, feature_samples[neuron_index], feature_activations[neuron_index])
                f.write(html_str)
