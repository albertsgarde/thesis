import json
from pathlib import Path
from typing import Callable, Tuple

import n2g
import torch
import transformer_lens  # type: ignore[import]
from jaxtyping import Float, Int
from n2g import NeuronStats, Tokenizer  # type: ignore[import]
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint  # type: ignore[import]

from thesis.device import Device
from thesis.mas.mas_store import MASStore
from thesis.sae.sae import SparseAutoencoder

FEATURES: list[int] = [274]

# torch.cuda.memory._record_memory_history()

device = Device.get()

mas_store = MASStore.load(Path("outputs/gelu-1l-sae_store.zip"), device)

model = transformer_lens.HookedTransformer.from_pretrained("gelu-1l", device=device.torch())
sae = SparseAutoencoder.from_hf("NeelNanda/sparse_autoencoder", "25.pt", "blocks.0.mlp.hook_post", device)

tokenizer = Tokenizer(model)


def feature_samples(feature_index: int) -> Tuple[list[str], float]:
    store_index = FEATURES[feature_index] + 2048

    samples = mas_store.feature_samples()[store_index, :, :]
    max_activation = mas_store.feature_max_activations()[store_index, :].max().item()

    tokens = ["".join(model.tokenizer.batch_decode(sample, clean_up_tokenization_spaces=False)) for sample in samples]

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
    return lambda samples: sae.feature_activations(model, samples, FEATURES[feature_index])


with Path("thesis/n2g/word_to_casings.json").open("r", encoding="utf-8") as f:
    word_to_casings = json.load(f)

train_config = n2g.TrainConfig(
    n2g.FitConfig(n2g.PruneConfig(batch_size=1), n2g.ImportanceConfig(), n2g.AugmentationConfig())
)

stats: list[NeuronStats]
models, stats = n2g.run_layer(
    len(FEATURES), feature_activation, feature_samples, tokenizer, word_to_casings, device.torch(), train_config
)

# torch.cuda.memory._dump_snapshot("outputs/n2g_test_memory_snapshot.pt")
