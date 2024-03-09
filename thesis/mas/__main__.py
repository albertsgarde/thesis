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
from thesis.sae.sae import SparseAutoencoder  # type: ignore[import]

from . import algorithm
from .algorithm import MASLayer, MASParams


@dataclass
class MASLayerConfig:
    hook_id: str
    num_features: int | None = None
    sae_repo: str | None = None
    sae_file: str | None = None

    def __post_init__(self):
        if self.num_features is None and (self.sae_repo is None or self.sae_file is None):
            raise ValueError("Either num_features or sae_repo and sae_file must be set.")
        if self.num_features is not None and (self.sae_repo is not None or self.sae_file is not None):
            raise ValueError("Only one of num_features or sae_repo and sae_file must be set.")

    def to_mas_layer(self) -> MASLayer:
        if self.num_features is not None:
            assert self.sae_repo is None
            assert self.sae_file is None
            return MASLayer.from_hook_id(self.hook_id, self.num_features)
        else:
            assert self.sae_repo is not None
            assert self.sae_file is not None
            sae = SparseAutoencoder.from_hf(self.sae_repo, self.sae_file, self.hook_id, get_device())
            return MASLayer.from_sae(sae)


@dataclass
class MASScriptConfig:
    params: MASParams
    dataset_name: str
    model_name: str
    layers: list[MASLayerConfig]
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

    mas_layers = [layer.to_mas_layer() for layer in config.layers]

    mas_store = algorithm.run(model, dataset, mas_layers, config.params, device)

    if not config.out_path.endswith(".zip"):
        config.out_path += ".zip"
    out_path = "outputs" / Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mas_store.save(out_path)


@hydra.main(config_path="../../conf/mas", version_base="1.3", config_name="mas")
def hydra_main(omega_config: OmegaConf) -> None:
    dict_config = typing.cast(
        dict[typing.Any, typing.Any], OmegaConf.to_container(omega_config, resolve=True, enum_to_str=True)
    )
    dict_config["layers"] = [MASLayerConfig(**layer) for layer in dict_config["layers"]]
    dict_config["params"] = MASParams(**dict_config["params"])
    config = MASScriptConfig(**dict_config)
    assert isinstance(config, MASScriptConfig)
    main(config)


if __name__ == "__main__":
    hydra_main()
