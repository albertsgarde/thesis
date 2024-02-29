from typing import Callable, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer  # type: ignore[import]
from transformer_lens.hook_points import HookPoint  # type: ignore[import]


class SparseAutoencoder:
    _w_enc: Float[Tensor, "layer_dim num_features"]
    _b_enc: Float[Tensor, " num_features"]
    _w_dec: Float[Tensor, " num_features layer_dim"]
    _b_dec: Float[Tensor, " layer_dim"]
    _hook_point: str

    def __init__(
        self,
        w_enc: Float[Tensor, "layer_dim num_features"],
        b_enc: Float[Tensor, " num_features"],
        w_dec: Float[Tensor, " num_features layer_dim"],
        b_dec: Float[Tensor, " layer_dim"],
        hook_point: str,
        device: str,
    ) -> None:
        num_features = w_enc.shape[1]
        assert b_enc.shape[0] == num_features, (
            "The bias must have the same number of features as the weights. "
            f"w_enc features: {num_features}, b_enc features: {b_enc.shape[0]}"
        )
        assert w_dec.shape[0] == num_features, (
            "The number of features must match between the encoder and decoder. "
            f"w_enc features: {num_features}, w_dec features: {w_dec.shape[0]}"
        )
        layer_dim = w_enc.shape[0]
        assert b_dec.shape[0] == layer_dim, (
            "The bias must have the same layer_dim as the weights. "
            f"w_enc neurons: {layer_dim}, b_enc neurons: {b_dec.shape[0]}"
        )
        assert w_dec.shape[1] == layer_dim, (
            "The layer_dim must match between the encoder and decoder. "
            f"w_enc neurons: {layer_dim}, w_dec neurons: {w_dec.shape[1]}"
        )
        self._w_enc = w_enc.to(device)
        self._b_enc = b_enc.to(device)
        self._w_dec = w_dec.to(device)
        self._b_dec = b_dec.to(device)
        self._hook_point = hook_point

    @staticmethod
    def from_data(data: dict[str, torch.Tensor], hook_point: str, device: str) -> "SparseAutoencoder":
        return SparseAutoencoder(data["W_enc"], data["b_enc"], data["W_dec"], data["b_dec"], hook_point, device)

    @property
    def num_features(self) -> int:
        return self._w_enc.shape[1]

    @property
    def layer_dim(self) -> int:
        return self._w_enc.shape[0]

    def encode(self, x: Float[Tensor, "*batch layer_dim"]) -> Float[Tensor, "*batch num_sae_features"]:
        return torch.relu(x @ self._w_enc + self._b_enc)

    def hook(
        self, destination: Float[Tensor, "*batch sample_length num_sae_features"]
    ) -> Tuple[str, Callable[[Float[Tensor, "*batch sample_length layer_dim"], HookPoint], None]]:
        def hook(activation: Float[Tensor, "*batch sample_length layer_dim"], hook: HookPoint) -> None:
            destination[:] = self.encode(activation)

        return self._hook_point, hook

    def activations(
        self, model: HookedTransformer, sample_tokens: Float[Tensor, "*batch sample_length"]
    ) -> Float[Tensor, "*batch sample_length num_sae_features"]:
        result = torch.full(sample_tokens.shape + (self.num_features,), torch.nan)

        with torch.no_grad():
            model.run_with_hooks(sample_tokens, fwd_hooks=[self.hook(result)])
            assert not torch.isnan(result).any(), "Result should not contain NaNs"

        return result
