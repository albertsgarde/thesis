import json
from pathlib import Path
from typing import Callable, Tuple

import n2g
import torch
import transformer_lens  # type: ignore[import]
from jaxtyping import Float, Int
from n2g import NeuronModel, NeuronStats, Tokenizer
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint  # type: ignore[import]

from thesis.device import Device
from thesis.mas.mas_store import MASStore
from thesis.sae.sae import SparseAutoencoder

NUM_FEATURES = 2


def main() -> None:
    device = Device.get()

    mas_store = MASStore.load(Path("outputs/gelu-1l-sae_store.zip"), device)

    model = transformer_lens.HookedTransformer.from_pretrained("gelu-1l", device.torch())
    sae = SparseAutoencoder.from_hf("NeelNanda/sparse_autoencoder", "25.pt", "blocks.0.mlp.hook_post", device)

    tokenizer = Tokenizer(model)

    def feature_samples(feature_index: int) -> Tuple[list[str], float]:
        if feature_index < 0:
            raise ValueError(f"Feature index must be non-negative. {feature_index=}")
        elif feature_index < NUM_FEATURES:
            store_index = feature_index
        elif feature_index < NUM_FEATURES + NUM_FEATURES:
            store_index = feature_index - NUM_FEATURES + 2048
        else:
            raise ValueError(f"Feature index must be less than {NUM_FEATURES + NUM_FEATURES}. {feature_index=}")

        samples = mas_store.feature_samples()[store_index, :, :]
        max_activation = mas_store.feature_max_activations()[store_index, :].max().item()

        tokens = [
            "".join(model.tokenizer.batch_decode(sample, clean_up_tokenization_spaces=False)) for sample in samples
        ]

        return tokens, max_activation

    def model_feature_activation(
        model: HookedTransformer, layer_id: str, neuron_index: int
    ) -> Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]]:
        def result(samples: Int[Tensor, "num_samples sample_length"]) -> Float[Tensor, "num_samples sample_length"]:
            activations: Float[Tensor, "num_samples sample_length"] = torch.full(samples.shape, float("nan"))

            def hook(activation: Float[Tensor, "num_samples sample_length neurons_per_layer"], hook: HookPoint) -> None:
                activations[:] = activation[:, :, neuron_index]

            with torch.no_grad():
                model.run_with_hooks(samples, fwd_hooks=[(layer_id, hook)])
                assert not torch.isnan(activations).any(), "Activations should not contain NaNs"

            return activations

        return result

    def feature_activation(
        feature_index: int,
    ) -> Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]]:
        if feature_index < 0:
            raise ValueError(f"Feature index must be non-negative. {feature_index=}")
        elif feature_index < NUM_FEATURES:
            return model_feature_activation(model, "blocks.0.mlp.hook_post", feature_index)
        elif feature_index < NUM_FEATURES + NUM_FEATURES:
            return lambda samples: sae.feature_activations(model, samples, feature_index - NUM_FEATURES)
        else:
            raise ValueError(f"Feature index must be less than {NUM_FEATURES + NUM_FEATURES}. {feature_index=}")

    with (Path(__file__) / ".." / "word_to_casings.json").open("r", encoding="utf-8") as f:
        word_to_casings = json.load(f)

    train_config = n2g.TrainConfig()

    models: list[NeuronModel]
    stats: list[NeuronStats]
    models, stats = n2g.run_layer(
        NUM_FEATURES * 2, feature_activation, feature_samples, tokenizer, word_to_casings, device.torch(), train_config
    )

    neuron_stats = stats[:NUM_FEATURES]
    sae_stats = stats[NUM_FEATURES:]

    avg_neuron_precision = sum(neuron_stats.firing.precision for neuron_stats in neuron_stats) / NUM_FEATURES
    avg_neuron_recall = sum(neuron_stats.firing.recall for neuron_stats in neuron_stats) / NUM_FEATURES
    avg_neuron_f1 = sum(neuron_stats.firing.f1_score for neuron_stats in neuron_stats) / NUM_FEATURES

    avg_sae_precision = sum(sae_stats.firing.precision for sae_stats in sae_stats) / NUM_FEATURES
    avg_sae_recall = sum(sae_stats.firing.recall for sae_stats in sae_stats) / NUM_FEATURES
    avg_sae_f1 = sum(sae_stats.firing.f1_score for sae_stats in sae_stats) / NUM_FEATURES

    print(f"Neuron precision: {avg_neuron_precision:.2f}")
    print(f"Neuron recall: {avg_neuron_recall:.2f}")
    print(f"Neuron f1: {avg_neuron_f1:.2f}")

    print(f"Sae precision: {avg_sae_precision:.2f}")
    print(f"Sae recall: {avg_sae_recall:.2f}")
    print(f"Sae f1: {avg_sae_f1:.2f}")

    outputs_path = Path("outputs") / "n2g"
    outputs_path.mkdir(parents=True, exist_ok=True)

    stats_path = outputs_path / "stats.json"
    with stats_path.open("w") as f:
        json.dump(
            {
                "neurons": {index: stats.model_dump() for index, stats in enumerate(stats[:NUM_FEATURES])},
                "sae": {index: stats.model_dump() for index, stats in enumerate(stats[NUM_FEATURES:])},
            },
            f,
        )

    for i, model in enumerate(
        models,
    ):
        graph_path = outputs_path / (f"neuron_{i}" if i < NUM_FEATURES else f"sae_{i-NUM_FEATURES}")
        with graph_path.open("w") as f:
            f.write(model.graphviz().source)


if __name__ == "__main__":
    main()
