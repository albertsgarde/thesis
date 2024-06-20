import json
import pickle
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import hydra
import n2g
import numpy as np
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
class N2GScriptConfig:
    mas_path: str
    model_name: str
    start_index: int
    end_index: int
    layers: list[LayerConfig]
    out_path: str
    create_dot: bool
    create_pkl: bool
    create_bin: bool
    save_activations: bool
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

    layers: list[Layer] = [layer_config.to_layer(device) for layer_config in config.layers]
    num_features = sum(layer.num_features for layer in layers)

    if num_features != mas_store.num_features():
        raise ValueError(
            f"Number of features in MAS store does not match number of features in layers. {num_features=}, {mas_store.num_features()=}"
        )

    def mas_to_layer_index(mas_index: int) -> Tuple[int, int]:
        if mas_index < 0:
            raise ValueError(f"Index must be non-negative. {mas_index=}")
        start_index = 0
        for i, layer in enumerate(layers):
            if start_index + layer.num_features > mas_index:
                return i, mas_index - start_index
            start_index += layer.num_features
        raise ValueError(f"Feature index must be less than {start_index}. {mas_index=}")

    def feature_samples(mas_index: int) -> Tuple[list[str], float]:
        samples = mas_store.feature_samples()[mas_index, :, :]
        max_activation = mas_store.feature_max_activations()[mas_index, :].max().item()

        if model.tokenizer is None:
            raise AttributeError("Model tokenizer must not be None.")

        tokens = [
            "".join(model.tokenizer.batch_decode(sample, clean_up_tokenization_spaces=False)) for sample in samples
        ]

        return tokens, max_activation

    def feature_activation(
        mas_index: int,
    ) -> Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]]:
        layer_index, feature_index = mas_to_layer_index(mas_index)
        layer = layers[layer_index]

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
    models, stats, activations, pred_activations = n2g.run_layer(
        range(config.start_index, config.end_index),
        feature_activation,
        feature_samples,
        tokenizer,
        word_to_casings,
        device.torch(),
        train_config,
    )

    num_none_models = sum(model is None for model in models)
    print(f"Errors: {num_none_models}/{len(models)}")

    stats_path = output_path / "stats.json"
    if stats_path.exists():
        with stats_path.open("r") as f:
            json_object = json.load(f)
            existing_stats = [
                NeuronStats.from_dict(neuron_stats) if neuron_stats else None for neuron_stats in json_object
            ]
    else:
        existing_stats = []
    if len(existing_stats) < config.end_index:
        existing_stats += [None] * (config.end_index - len(existing_stats))
    for i, neuron_stats in enumerate(stats):
        if neuron_stats is not None:
            existing_stats[i + config.start_index] = neuron_stats
    with stats_path.open("w") as f:
        json_object = [
            neuron_stats.model_dump() if neuron_stats is not None else None for neuron_stats in existing_stats
        ]
        json.dump(json_object, f)

    models_path = output_path / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    if config.create_dot or config.create_pkl:
        for i, model in enumerate(models):
            i = i + config.start_index
            if model is not None:
                if config.create_dot:
                    with (models_path / f"{i}.pkl").open("wb") as bin_file:
                        pickle.dump(model, bin_file)
                if config.create_pkl:
                    with (models_path / f"{i}.dot").open("w", encoding="utf-8") as f:
                        f.write(model.graphviz().source)
    if config.create_bin:
        bin_path = models_path / "all_models.bin"
        if bin_path.exists():
            with bin_path.open("rb") as read_file:
                feature_models = FeatureModel.list_from_bin(tokenizer, read_file.read())
        else:
            feature_models = []
        if len(feature_models) < config.end_index:
            feature_models += [None] * (config.end_index - len(feature_models))
        for i, model in enumerate(models):
            i = i + config.start_index
            if model is not None:
                feature_models[i] = FeatureModel.from_model(tokenizer, model)
        all_models_bytes = FeatureModel.list_to_bin(
            feature_models,
        )
        with bin_path.open("wb") as write_file:
            write_file.write(all_models_bytes)

    if config.save_activations:
        activations_path = output_path / "activations.npz"
        np.savez(activations_path, activations=activations, pred_activations=pred_activations)


@hydra.main(config_path="../../conf/n2g", version_base="1.3", config_name="n2g")
def hydra_main(omega_config: OmegaConf) -> None:
    dict_config = typing.cast(
        dict[typing.Any, typing.Any], OmegaConf.to_container(omega_config, resolve=True, enum_to_str=True)
    )
    dict_config["layers"] = [LayerConfig(**layer) for layer in dict_config["layers"]]
    dict_config["params"] = N2GParams(**dict_config["params"])
    config = N2GScriptConfig(**dict_config)
    assert isinstance(config, N2GScriptConfig)
    main(config)


if __name__ == "__main__":
    hydra_main()
