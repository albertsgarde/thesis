import os
from dataclasses import dataclass

import datasets  # type: ignore[missingTypeStubs, import-untyped]
import hydra
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from hydra.core.config_store import ConfigStore
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]

from thesis.device import get_device
from thesis.sae.sae import SparseAutoencoder  # type: ignore[import]

from . import algorithm
from .algorithm import MASLayer, MASParams


@dataclass
class MASScriptConfig:
    params: MASParams


cs = ConfigStore.instance()

cs.store(name="mas", node=MASScriptConfig)


@hydra.main(config_path="../../conf/mas", version_base="1.3")
def hydra_main(config: MASScriptConfig):
    device = get_device()

    model: HookedTransformer = HookedTransformer.from_pretrained("gelu-1l", device=device.torch())  # type: ignore[reportUnknownVariableType]

    dataset: IterableDataset = datasets.load_dataset(  # type: ignore[reportUnknownMemberType]
        "NeelNanda/c4-code-20k", streaming=True, split="train", trust_remote_code=True
    )

    mas_layers = [
        MASLayer.from_hook_id("blocks.0.mlp.hooks_post", 2048),
        MASLayer.from_sae(
            SparseAutoencoder.from_hf("NeelNanda/sparse_autoencoder", "25.pt", "blocks.0.mlp.hooks_post", device)
        ),
    ]

    mas_store = algorithm.run(model, dataset, mas_layers, config.params, device)

    os.makedirs("outputs", exist_ok=True)

    mas_store.save("outputs/mas_store.zip")


if __name__ == "__main__":
    hydra_main()
