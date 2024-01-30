import html
import json
from typing import Tuple

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
    return f"<span style='background-color: {token_color(activation, max_abs_activation)}'>{html.escape(token)}</span>"


def sample_html(
    model: HookedTransformer,
    tokens: Int[Tensor, " context"],
    activations: Float[Tensor, " context"],
) -> str:
    max_abs_activation = activations.abs().max()
    token_strings: list[str] = model.to_str_tokens(tokens)
    text: str = "".join(
        token_html(token, activation, max_abs_activation) for token, activation in zip(token_strings, activations)
    )
    return f"<p>{text}</p>"


def generate_html(
    model: HookedTransformer,
    samples: list[Tuple[Int[Tensor, " context"], Float[Tensor, " context"]]],
) -> str:
    samples = "<hr>".join(sample_html(model, tokens, activations) for tokens, activations in samples)
    return f"""
<head>
	<meta charset="utf-8" />
    <title>Sample activations</title>
</head>
<body>
    {samples}
</body>
"""
