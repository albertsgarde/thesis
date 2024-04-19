import json
import pickle
from pathlib import Path
from typing import Callable, Tuple

import n2g
import torch
import transformer_lens  # type: ignore[import]
from jaxtyping import Float, Int
from n2g import NeuronStats, Tokenizer
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint  # type: ignore[import]

from thesis.device import Device
from thesis.mas.mas_store import MASStore
from thesis.sae.sae import SparseAutoencoder

NUM_NEURONS = 2048
NUM_SAE_FEATURES = 16384
NUM_FEATURES = NUM_NEURONS + NUM_SAE_FEATURES


def main() -> None:
    device = Device.get()

    mas_store = MASStore.load(Path("outputs/gelu-1l-sae_store.zip"), device)

    model = transformer_lens.HookedTransformer.from_pretrained("gelu-1l", device.torch())
    sae = SparseAutoencoder.from_hf("NeelNanda/sparse_autoencoder", "25.pt", "blocks.0.mlp.hook_post", device)

    tokenizer = Tokenizer(model)

    def feature_samples(feature_index: int) -> Tuple[list[str], float]:
        if feature_index < 0:
            raise ValueError(f"Feature index must be non-negative. {feature_index=}")
        elif feature_index < NUM_NEURONS:
            store_index = feature_index
        elif feature_index < NUM_FEATURES:
            store_index = feature_index
        else:
            raise ValueError(f"Feature index must be less than {NUM_FEATURES}. {feature_index=}")

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
        elif feature_index < NUM_NEURONS:
            return model_feature_activation(model, "blocks.0.mlp.hook_post", feature_index)
        elif feature_index < NUM_FEATURES:
            return lambda samples: sae.feature_activations(model, samples, feature_index - NUM_NEURONS)
        else:
            raise ValueError(f"Feature index must be less than {NUM_FEATURES}. {feature_index=}")

    with (Path(__file__).parent / "word_to_casings.json").open("r", encoding="utf-8") as f:
        word_to_casings = json.load(f)

    fit_config = n2g.FitConfig(
        prune_config=n2g.PruneConfig(prepend_bos=False),
        importance_config=n2g.ImportanceConfig(prepend_bos=False),
        augmentation_config=n2g.AugmentationConfig(prepend_bos=False),
    )
    train_config = n2g.TrainConfig(fit_config=fit_config)

    stats: list[NeuronStats]
    models, stats = n2g.run_layer(
        NUM_FEATURES, feature_activation, feature_samples, tokenizer, word_to_casings, device.torch(), train_config
    )

    stats_path = Path("outputs") / "stats.json"
    with stats_path.open("w") as f:
        json_object = [neuron_stats.model_dump() for neuron_stats in stats]
        json.dump(json_object, f)

    models_path = Path("outputs") / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    for i, model in enumerate(models):
        with (models_path / f"{i}.pkl").open("wb") as bin_file:
            pickle.dump(model, bin_file)
        with (models_path / f"{i}.dot").open("w", encoding="utf-8") as f:
            f.write(model.graphviz().source)


if __name__ == "__main__":
    main()
