import html
import typing

import numpy as np
import torch
from beartype import beartype
from graphviz import Source  # type: ignore[import]
from jaxtyping import Float, Int
from numpy import ndarray
from torch import Tensor
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]


@beartype
def token_color(activation: float, max_abs_activation: float) -> str:
    activation = activation / max_abs_activation if max_abs_activation > 0 else 0
    if activation >= 0:
        return f"rgba({int(255*(1-activation))}, {int(255*(1-activation))}, 255, {int(activation*256)/256})"
    else:
        return f"rgba(255, {int(255*(1+activation))}, {int(255*(1+activation))}, {int(-activation*256)/256})"


@beartype
def token_html(token: str, activation: float, max_abs_activation: float) -> str:
    token = token.replace("\n", "\\n").replace("\t", "\\t")
    tooltip = f"Activation: {activation:.4f}"
    return f"""\
<span style='background-color: {token_color(activation, max_abs_activation)}' title='{tooltip}'>
    {html.escape(token)}
</span>\
"""


@beartype
def sample_html(
    model: HookedTransformer,
    tokens: list[str],
    activations: Float[ndarray, " sample_length"],
) -> str:
    assert len(tokens) == activations.shape[0], "Tokens and activations must have the same shape"
    if model.tokenizer is None:
        raise ValueError("Model must have tokenizer.")
    token_strings: list[str] = tokens
    activations = activations
    max_abs_activation = np.abs(activations).max()
    text: str = "".join(
        token_html(token, float(activation), float(max_abs_activation))
        for token, activation in zip(token_strings, activations, strict=True)
    )
    return f"<p>{text}</p>"


@beartype
def n2g_svg(n2g_source: Source) -> str:
    return n2g_source.pipe(format="svg", encoding="utf-8")


@beartype
def generate_html(
    model: HookedTransformer,
    samples: Int[Tensor, "num_samples sample_length"],
    activations: Float[Tensor, "num_samples sample_length"],
    n2g_source: Source | None = None,
) -> str:
    assert not torch.isinf(activations).any(), "Infinite activations found"
    assert not torch.isnan(activations).any(), "NaN activations found"
    samples_html: str = "<hr>".join(
        sample_html(model, tokens, activations)
        for tokens, activations in zip(
            (typing.cast(list[str], model.to_str_tokens(sample)) for sample in samples),
            activations.cpu().numpy(),
            strict=True,
        )
    )
    n2g_str = f"{n2g_svg(n2g_source)}<hr>" if n2g_source is not None else ""
    return f"""
<head>
	<meta charset="utf-8" />
    <title>Sample activations</title>
</head>
<body>
    {n2g_str}
    {samples_html}
</body>
"""
