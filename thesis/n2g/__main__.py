import json
import pickle
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import hydra
import n2g
import torch
import transformer_lens  # type: ignore[import]
from beartype import beartype
from hydra.core.config_store import ConfigStore
from jaxtyping import Float, Int
from n2g import FeatureModel, NeuronModel, NeuronStats, Tokenizer
from omegaconf import OmegaConf
from torch import Tensor
from transformer_lens.hook_points import HookPoint  # type: ignore[import]

from thesis.device import Device
from thesis.layer import Layer, LayerConfig
from thesis.mas.mas_store import MASStore


@dataclass
class N2GParams:
    stop_on_error: bool


@dataclass
class N2GLayerConfig:
    start_mas_index: int
    layer_config: LayerConfig


@dataclass
class N2GScriptConfig:
    mas_path: str
    model_name: str
    layers: list[N2GLayerConfig]
    out_path: str
    params: N2GParams


cs = ConfigStore.instance()

cs.store(name="n2g", node=N2GScriptConfig)


@beartype
def main(config: N2GScriptConfig) -> None:
    torch.set_grad_enabled(False)

    device = Device.get()

    mas_store = MASStore.load(Path(config.mas_path), device)

    model = transformer_lens.HookedTransformer.from_pretrained(config.model_name, device=device.torch())

    output_path = Path(config.out_path)
    output_path.mkdir(exist_ok=True, parents=True)

    tokenizer = Tokenizer(model)

    layers: list[tuple[int, Layer]] = [
        (layer_config.start_mas_index, layer_config.layer_config.to_layer(device)) for layer_config in config.layers
    ]
    num_features = sum(layer.num_features for _, layer in layers)

    def n2g_to_layer_index(n2g_index: int) -> Tuple[int, int]:
        if n2g_index < 0:
            raise ValueError(f"Feature index must be non-negative. {n2g_index=}")
        start_index = 0
        for i, (_, layer) in enumerate(layers):
            if start_index + layer.num_features > n2g_index:
                return i, n2g_index - start_index
            start_index += layer.num_features
        raise ValueError(f"Feature index must be less than {start_index}. {n2g_index=}")

    def feature_samples(n2g_index: int) -> Tuple[list[str], float]:
        layer_index, feature_index = n2g_to_layer_index(n2g_index)
        start_mas_index, _layer = layers[layer_index]

        mas_index = start_mas_index + feature_index
        samples = mas_store.feature_samples()[mas_index, :, :]
        max_activation = mas_store.feature_max_activations()[mas_index, :].max().item()

        if model.tokenizer is None:
            raise AttributeError("Model tokenizer must not be None.")

        tokens = [
            "".join(model.tokenizer.batch_decode(sample, clean_up_tokenization_spaces=False)) for sample in samples
        ]

        return tokens, max_activation

    def feature_activation(
        n2g_index: int,
    ) -> Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]]:
        layer_index, feature_index = n2g_to_layer_index(n2g_index)
        _start_mas_index, layer = layers[layer_index]

        def result(samples: Int[Tensor, "num_samples sample_length"]) -> Float[Tensor, "num_samples sample_length"]:
            squeeze = samples.ndim == 1
            if squeeze:
                samples = samples.unsqueeze(0)
            activations: Float[Tensor, "num_samples sample_length"] = torch.full(samples.shape, float("nan"))

            def hook(activation: Float[Tensor, "num_samples sample_length neurons_per_layer"], hook: HookPoint) -> None:
                activations[:, :] = layer.activation_map(activation)[:, :, feature_index]

            with torch.no_grad():
                model.run_with_hooks(samples, fwd_hooks=[(layer.hook_id, hook)])
                assert not torch.isnan(activations).any(), "Activations should not contain NaNs"

            if squeeze:
                activations = activations.squeeze(0)
            return activations

        return result

    with (Path(__file__).parent / "word_to_casings.json").open("r", encoding="utf-8") as f:
        word_to_casings = json.load(f)

    fit_config = n2g.FitConfig(
        prune_config=n2g.PruneConfig(prepend_bos=False),
        importance_config=n2g.ImportanceConfig(prepend_bos=False),
        augmentation_config=n2g.AugmentationConfig(prepend_bos=False),
    )
    train_config = n2g.TrainConfig(fit_config=fit_config, stop_on_error=config.params.stop_on_error)

    stats: list[NeuronStats]
    models: list[NeuronModel]
    models, stats = n2g.run_layer(
        range(num_features),
        feature_activation,
        feature_samples,
        tokenizer,
        word_to_casings,
        device.torch(),
        train_config,
    )

    none_models = sum(model is None for model in models)
    print(f"Errors: {none_models}/{len(models)}")

    stats_path = output_path / "stats.json"
    with stats_path.open("w") as f:
        json_object = [neuron_stats.model_dump() if neuron_stats is not None else {} for neuron_stats in stats]
        json.dump(json_object, f)

    models_path = output_path / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    for i, model in enumerate(models):
        if model is not None:
            with (models_path / f"{i}.pkl").open("wb") as bin_file:
                pickle.dump(model, bin_file)
            with (models_path / f"{i}.dot").open("w", encoding="utf-8") as f:
                f.write(model.graphviz().source)
    all_models_bytes = FeatureModel.list_to_bin(
        [FeatureModel.from_model(tokenizer, feature_model) for feature_model in models]
    )
    with (models_path / "all_models.bin").open("wb") as bf:
        bf.write(all_models_bytes)


@hydra.main(config_path="../../conf/n2g", version_base="1.3", config_name="n2g")
def hydra_main(omega_config: OmegaConf) -> None:
    dict_config = typing.cast(
        dict[typing.Any, typing.Any], OmegaConf.to_container(omega_config, resolve=True, enum_to_str=True)
    )
    dict_config["layers"] = [
        N2GLayerConfig(layer["start_mas_index"], LayerConfig(**layer["layer_config"]))
        for layer in dict_config["layers"]
    ]
    dict_config["params"] = N2GParams(**dict_config["params"])
    config = N2GScriptConfig(**dict_config)
    assert isinstance(config, N2GScriptConfig)
    main(config)


if __name__ == "__main__":
    hydra_main()
