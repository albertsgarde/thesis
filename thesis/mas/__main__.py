import itertools
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import datasets  # type: ignore[missingTypeStubs, import-untyped]
import hydra
import torch
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from hydra.core.config_store import ConfigStore
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens.HookedTransformer import HookedTransformer

from ..device import get_device
from . import html
from .mas_store import MASStore
from .sample_loader import Sample, SampleDataset


@dataclass
class MASConfig:
    model_name: str = "solu-1l"


def main(config: MASConfig):
    with torch.no_grad():
        device = get_device()

        dataset: IterableDataset = datasets.load_dataset(  # type: ignore[reportUnknownMemberType]
            "monology/pile-uncopyrighted", streaming=True, split="train", trust_remote_code=True
        )
        dataset = dataset.take(1000)  # type: ignore[reportUnknownMemberType]

        model: HookedTransformer = HookedTransformer.from_pretrained(config.model_name, device=device.torch())  # type: ignore[reportUnknownVariableType]

        neuron_index = 0

        context_size = model.cfg.n_ctx
        print(f"Model context size: {context_size}")

        sample_dataset = SampleDataset(context_size, 256, model, dataset)

        mas_store = MASStore(20, 2048, context_size, device)

        def create_hook(
            sample: Sample, mas_store: MASStore
        ) -> Callable[[Float[Tensor, "batch context neurons_per_layer"], Any], None]:
            def hook(activation: Float[Tensor, "batch context neurons_per_layer"], hook: Any) -> None:
                mas_store.add_sample(sample, activation[0, :, :])

            return hook

        num_samples = 40
        start_time = time.time()
        for i, sample in itertools.islice(enumerate(sample_dataset), num_samples):
            model.run_with_hooks(sample.tokens, fwd_hooks=[("blocks.0.mlp.hook_mid", create_hook(sample, mas_store))])
            assert mas_store.num_samples_added() == i + 1
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")
        print(f"Time taken per sample: {(end_time - start_time) / (num_samples):.2f}s")

        os.makedirs("outputs/html", exist_ok=True)

        print("Generating and saving HTML")

        feature_samples = mas_store.feature_samples()
        feature_activations = mas_store.feature_activations()

        for neuron_index in range(40):
            with open(f"outputs/html/{neuron_index}.html", "w", encoding="utf-8") as f:
                html_str = html.generate_html(model, feature_samples[neuron_index], feature_activations[neuron_index])
                f.write(html_str)


cs = ConfigStore.instance()

cs.store(name="mas", node=MASConfig)


@hydra.main(config_path="../conf/mas", version_base="1.3")
def hydra_main(config: MASConfig):
    main(config)


if __name__ == "__main__":
    main(MASConfig())
