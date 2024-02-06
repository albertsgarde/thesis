import html
import json
from typing import Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens.HookedTransformer import HookedTransformer


def token_color(activation: float, max_abs_activation: float) -> str:
    activation = activation / max_abs_activation
    if activation >= 0:
        return f"rgba({int(255*(1-activation))}, {int(255*(1-activation))}, 255, {int(activation*256)/256})"
    else:
        return f"rgba(255, {int(255*(1+activation))}, {int(255*(1+activation))}, {int(-activation*256)/256})"


def token_html(token: str, activation: float, max_abs_activation: float) -> str:
    token = token.replace("\n", "\\n").replace("\t", "\\t")
    tooltip = f"Activation: {activation:.4f}"
    return f"""\
<span style='background-color: {token_color(activation, max_abs_activation)}' title='{tooltip}'>
    {html.escape(token)}
</span>\
"""


def sample_html(
    model: HookedTransformer,
    tokens: Int[Tensor, " context"],
    activations: Float[Tensor, " context"],
) -> str:
    assert tokens.shape == activations.shape, "Tokens and activations must have the same shape"
    first_pad: Int[Tensor, "0:1"] = (tokens == model.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
    if first_pad.numel() == 0:
        first_pad = tokens.shape[-1]
    else:
        first_pad = first_pad[0]
    token_strings: list[str] = model.to_str_tokens(tokens[:first_pad])
    activations = activations[:first_pad]
    max_abs_activation = activations.abs().max()
    text: str = "".join(
        token_html(token, activation, max_abs_activation)
        for token, activation in zip(token_strings, activations, strict=True)
    )
    return f"<p>{text}</p>"


def generate_html(
    model: HookedTransformer,
    samples: Int[Tensor, "num_samples context"],
    activations: Float[Tensor, "num_samples context"],
) -> str:
    assert not torch.isinf(activations).any(), "Infinite activations found"
    assert not torch.isnan(activations).any(), "NaN activations found"
    samples_html: str = "<hr>".join(
        sample_html(model, tokens, activations) for tokens, activations in zip(samples, activations, strict=True)
    )
    return f"""
<head>
	<meta charset="utf-8" />
    <title>Sample activations</title>
</head>
<body>
    {samples_html}
</body>
"""
