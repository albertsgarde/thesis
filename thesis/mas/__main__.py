import typing
from dataclasses import dataclass
from pathlib import Path

import blobfile as bf
import datasets  # type: ignore[missingTypeStubs, import-untyped]
import hydra
import torch
from beartype import beartype
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from sparse_autoencoder import Autoencoder as OAISparseAutoencoder
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]

from thesis.device import Device, get_device
from thesis.sae.sae import SparseAutoencoder  # type: ignore[import]

from . import algorithm
from .algorithm import MASLayer, MASParams


@dataclass
class MASLayerConfig:
    hook_id: str
    num_features: int | None = None
    sae_hf_repo: str | None = None
    sae_file: str | None = None
    sae_oai_layer: int | None = None

    def __post_init__(self):
        if self.sae_hf_repo is None != self.sae_file is None:
            raise ValueError("Either both or none of sae_hf_repo and sae_file must be set.")
        num_sources_set = sum(
            [self.num_features is not None, self.sae_hf_repo is not None, self.sae_oai_layer is not None]
        )
        if num_sources_set != 1:
            raise ValueError("Exactly one of num_features, sae_hf_repo and sae_open_ai must be set.")

    def to_mas_layer(self, device: Device) -> MASLayer:
        if self.num_features is not None:
            assert self.sae_hf_repo is None
            assert self.sae_file is None
            return MASLayer.from_hook_id(self.hook_id, self.num_features)
        elif self.sae_hf_repo is not None:
            assert self.sae_hf_repo is not None
            assert self.sae_file is not None
            sae = SparseAutoencoder.from_hf(self.sae_hf_repo, self.sae_file, self.hook_id, device)
            return MASLayer.from_sae(sae)
        else:
            assert self.sae_oai_layer is not None
            file_name = (
                f"az://openaipublic/sparse-autoencoder/gpt2-small/mlp_post_act/autoencoders/{self.sae_oai_layer}.pt"
            )
            with bf.BlobFile(file_name, "rb") as f:
                state_dict = torch.load(f, map_location=device.torch())
                sae = OAISparseAutoencoder.from_state_dict(state_dict)
                sae.to(device.torch())
            return MASLayer.from_oai_sae(self.hook_id, sae)


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

    mas_layers = [layer.to_mas_layer(device) for layer in config.layers]

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
