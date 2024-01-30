import html
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import datasets  # type: ignore[missingTypeStubs, import-untyped]
import hydra
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from hydra.core.config_store import ConfigStore
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens.HookedTransformer import HookedTransformer

from .device import get_device


@dataclass
class MASConfig:
    model_name: str = "solu-1l"


def token_color(activation: float, max_abs_activation: float) -> str:
    activation = activation / max_abs_activation
    if activation >= 0:
        return f"rgba({255*(1-activation)}, {255*(1-activation)}, 255, {activation})"
    else:
        return f"rgba(255, {255*(1+activation)}, {255*(1+activation)}, {-activation})"


def token_html(token: str, activation: float, max_abs_activation: float) -> str:
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
    return "".join(sample_html(model, tokens, activations) for tokens, activations in samples)


def main(config: MASConfig):
    device = get_device()

    dataset: IterableDataset = datasets.load_dataset(  # type: ignore[reportUnknownMemberType]
        "monology/pile-uncopyrighted", streaming=True, split="train", trust_remote_code=True
    )
    dataset = dataset.take(1000)  # type: ignore[reportUnknownMemberType]

    model: HookedTransformer = HookedTransformer.from_pretrained(config.model_name, device=device.torch())  # type: ignore[reportUnknownVariableType]

    for i, datapoint in enumerate(dataset.iter(batch_size=1)):
        text = datapoint["text"][0]
        print(model.to_tokens(text, truncate=False).shape)
        if i == 10:
            break

    neuron_index = 0

    samples: list[Tuple[Int[Tensor, " context"], Float[Tensor, " context"]]] = []

    def create_hook(
        tokens: Int[Tensor, " context"],
    ) -> Callable[[Float[Tensor, "batch context neurons_per_layer"], Any], None]:
        def hook(activation: Float[Tensor, "batch context neurons_per_layer"], hook: Any) -> None:
            samples.append((tokens, activation[0, :, neuron_index]))

        return hook

    context_size = model.cfg.n_ctx
    print(f"Context size: {context_size}")

    for i, datapoint in enumerate(dataset.iter(batch_size=1)):
        text = datapoint["text"]
        tokens = model.to_tokens(datapoint["text"], truncate=False)
        print(tokens.shape)

        tokens = tokens[0, :context_size]
        model.run_with_hooks(tokens, fwd_hooks=[("blocks.0.mlp.hook_mid", create_hook(tokens))])
        if i == 0:
            break

    with open("outputs/output.html", "w") as f:
        f.write(generate_html(model, samples))


cs = ConfigStore.instance()

cs.store(name="mas", node=MASConfig)


@hydra.main(config_path="../conf/mas", version_base="1.3")
def hydra_main(config: MASConfig):
    main(config)


if __name__ == "__main__":
    main(MASConfig())
