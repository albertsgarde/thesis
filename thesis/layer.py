
from dataclasses import dataclass
from typing import Callable

import blobfile as bf
import torch
from jaxtyping import Float
from sparse_autoencoder import Autoencoder as OAISparseAutoencoder
from torch import Tensor

from thesis.device import Device
from thesis.sae.sae import SparseAutoencoder


@dataclass
class Layer:
    """
    A description of a layer to include in the MAS algorithm and how to handle it.

    Attributes:
        hook_id: The transformer_lens to get activations for this layer from.
        num_features: The number of features to store for this layer.
        activation_map: A function that takes the activations from the given hook and returns activations for the
                features of the layer.
                The last dimension of the output must always match the `num_features` attribute.
    """

    hook_id: str
    num_features: int
    activation_map: Callable[
        [Float[Tensor, "batch context neurons_in_hook"]], Float[Tensor, "batch context _num_features"]
    ]

    @staticmethod
    def from_hook_id(hook_id: str, num_features: int) -> "Layer":
        return Layer(hook_id, num_features, lambda x: x)

    @staticmethod
    def from_sae(sae: SparseAutoencoder) -> "Layer":
        return Layer(sae.hook_point, sae.num_features, sae.encode)

    @staticmethod
    def from_oai_sae(hook_id: str, sae: OAISparseAutoencoder) -> "Layer":
        return Layer(hook_id, sae.n_latents, sae.encode)

@dataclass
class LayerConfig:
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

    def to_mas_layer(self, device: Device) -> Layer:
        if self.num_features is not None:
            assert self.sae_hf_repo is None
            assert self.sae_file is None
            return Layer.from_hook_id(self.hook_id, self.num_features)
        elif self.sae_hf_repo is not None:
            assert self.sae_hf_repo is not None
            assert self.sae_file is not None
            sae = SparseAutoencoder.from_hf(self.sae_hf_repo, self.sae_file, self.hook_id, device)
            return Layer.from_sae(sae)
        else:
            assert self.sae_oai_layer is not None
            file_name = (
                f"az://openaipublic/sparse-autoencoder/gpt2-small/mlp_post_act/autoencoders/{self.sae_oai_layer}.pt"
            )
            with bf.BlobFile(file_name, "rb") as f:
                state_dict = torch.load(f, map_location=device.torch())
                sae = OAISparseAutoencoder.from_state_dict(state_dict)
                sae.to(device.torch())
            return Layer.from_oai_sae(self.hook_id, sae)
