from typing import Callable, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer  # type: ignore[import]
from transformer_lens.hook_points import HookPoint  # type: ignore[import]

from thesis.device import Device  # type: ignore[import]


def hook(
    hook_point: str, destination: Float[Tensor, "*batch sample_length layer_dim"]
) -> Tuple[str, Callable[[Float[Tensor, "*batch sample_length layer_dim"], HookPoint], None]]:
    """
    A transformer_lens hook to store activations in a destination tensor.

    Args:
        hook_point: The hook point to get activations from.
        destination: The tensor to store the activations in.
    """

    def hook(activation: Float[Tensor, "*batch sample_length layer_dim"], hook: HookPoint) -> None:
        destination[:] = activation

    return hook_point, hook


def activations(
    model: HookedTransformer,
    hook_point: str,
    num_hook_point_features: int,
    sample_tokens: Float[Tensor, "*batch sample_length"],
    device: Device,
    check_outputs: bool = True,
) -> Float[Tensor, "*batch sample_length num_features"]:
    """
    Get activations for a given hook point from a model on a given sample.
    `model`, `sample_tokens`, and `device` must agree on the device.

    Args:
        model: The model to get activations from.
        hook_point: The hook point to get activations from.
        num_hook_point_features: The number of features at the hook point.
        sample_tokens: The sample to get activations for.
        device: The device to run the model on.
        check_outputs: Whether to check for NaNs in the output.
    """
    result = torch.full(sample_tokens.shape + (num_hook_point_features,), torch.nan, device=device.torch())

    with torch.no_grad():
        model.run_with_hooks(sample_tokens, fwd_hooks=[hook(hook_point, result)])
        if check_outputs:
            assert not torch.isnan(
                result
            ).any(), f"NaNs in result at index {tuple(torch.nonzero(torch.isnan(result))[0].tolist())}."

    return result


def neuron_hook(
    hook_point: str, destination: Float[Tensor, "*batch sample_length"], neuron_index: int
) -> Tuple[str, Callable[[Float[Tensor, "*batch sample_length hook_dim"], HookPoint], None]]:
    """
    A transformer_lens hook to store activations for a single neuron in a destination tensor.

    Args:
        hook_point: The hook point to get activations from.
        destination: The tensor to store the activations in.
        neuron_index: The index of the neuron to store activations for.
    """

    def hook(activation: Float[Tensor, "*batch sample_length hook_dim"], hook: HookPoint) -> None:
        destination[:] = activation[..., neuron_index]

    return hook_point, hook


def neuron_activations(
    model: HookedTransformer,
    hook_point: str,
    sample_tokens: Float[Tensor, "*batch sample_length"],
    neuron_index: int,
    device: Device,
    check_outputs: bool = True,
) -> Float[Tensor, "*batch sample_length"]:
    """
    Get activations for a single neuron at a given hook point from a model on a given sample.
    `model`, `sample_tokens`, and `device` must agree on the device.

    Args:
        model: The model to get activations from.
        hook_point: The hook point to get activations from.
        sample_tokens: The sample to get activations for.
        neuron_index: The index of the neuron to get activations for.
        device: The device to run the model on.
        check_outputs: Whether to check for NaNs in the output.
    """
    result = torch.full(sample_tokens.shape, torch.nan, device=device.torch())

    with torch.no_grad():
        model.run_with_hooks(sample_tokens, fwd_hooks=[neuron_hook(hook_point, result, neuron_index)])
        if check_outputs:
            assert not torch.isnan(
                result
            ).any(), f"NaNs in result at index {tuple(torch.nonzero(torch.isnan(result))[0].tolist())}."

    return result
