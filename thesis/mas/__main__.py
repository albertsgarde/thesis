import typing
from dataclasses import dataclass
from pathlib import Path

import datasets  # type: ignore[missingTypeStubs, import-untyped]
import hydra
from beartype import beartype
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]

from thesis.device import get_device
from thesis.layer import LayerConfig

from . import algorithm
from .algorithm import MASParams


@dataclass
class MASScriptConfig:
    params: MASParams
    dataset_name: str
    model_name: str
    layers: list[LayerConfig]
    out_path: str


cs = ConfigStore.instance()

cs.store(name="mas", node=MASScriptConfig)


@beartype
def main(config: MASScriptConfig) -> None:
    device = get_device()

    dataset: IterableDataset = datasets.load_dataset(  # type: ignore[reportUnknownMemberType]
        config.dataset_name, streaming=True, split="train", trust_remote_code=True
    )

    model: HookedTransformer = HookedTransformer.from_pretrained(config.model_name, device=device.torch())  # type: ignore[reportUnknownVariableType]

    mas_layers = [layer.to_layer(device) for layer in config.layers]

    mas_store = algorithm.run(model, dataset, mas_layers, config.params, device)

    if not config.out_path.endswith(".zip"):
        config.out_path += ".zip"
    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mas_store.save(out_path)


@hydra.main(config_path="../../conf/mas", version_base="1.3", config_name="mas")
def hydra_main(omega_config: OmegaConf) -> None:
    dict_config = typing.cast(
        dict[typing.Any, typing.Any], OmegaConf.to_container(omega_config, resolve=True, enum_to_str=True)
    )
    dict_config["layers"] = [LayerConfig(**layer) for layer in dict_config["layers"]]
    dict_config["params"] = MASParams(**dict_config["params"])
    config = MASScriptConfig(**dict_config)
    assert isinstance(config, MASScriptConfig)
    main(config)


if __name__ == "__main__":
    hydra_main()
