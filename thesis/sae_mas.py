import itertools

import datasets  # type: ignore[missingTypeStubs, import-untyped]
import torch
import transformer_lens  # type: ignore
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from transformer_lens import HookedTransformer  # type: ignore[import]

from thesis.device import Device
from thesis.mas import html
from thesis.sae.sae import SparseAutoencoder

from .mas.mas_store import MASStore  # type: ignore[import]
from .mas.sample_loader import SampleDataset  # type: ignore[import]


def main() -> None:
    with torch.no_grad():
        device = Device.get()

        model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
        context_size = model.cfg.n_ctx
        print(f"Model context size: {context_size}")

        dataset: IterableDataset = datasets.load_dataset(  # type: ignore[reportUnknownMemberType]
            "monology/pile-uncopyrighted", streaming=True, split="train", trust_remote_code=True
        )
        sample_dataset = SampleDataset(context_size, 256, model, dataset)

        point = "resid_pre"
        layer_index = 5
        layer_id = f"blocks.{layer_index}.hook_resid_pre"

        sae_data = transformer_lens.utils.download_file_from_hf(
            "jacobcd52/gpt2-small-sparse-autoencoders",
            f"gpt2-small_6144_{point}_{layer_index}.pt",
            force_is_torch=True,
        )
        sae = SparseAutoencoder.from_data(sae_data, layer_id, device)

        num_sae_features = sae.num_features

        mas_store = MASStore(
            num_samples=32,
            num_features=num_sae_features,
            context_size=context_size,
            sample_length_pre=192,
            sample_length_post=64,
            pad_token_id=model.tokenizer.pad_token_id,
            device=device,
        )

        activation_scratch = torch.full((context_size, num_sae_features), torch.nan, device=device.torch())
        sae_hook = sae.hook(activation_scratch)

        for _i, sample in itertools.islice(enumerate(sample_dataset), 32):
            model.run_with_hooks(sample.tokens, fwd_hooks=[sae_hook])
            mas_store.add_sample(sample, activation_scratch)

        feature_samples = mas_store.feature_samples()
        feature_activations = mas_store.feature_activations()

        for neuron_index in range(40):
            with open(f"outputs/html/sae_mas/{neuron_index}.html", "w", encoding="utf-8") as f:
                html_str = html.generate_html(model, feature_samples[neuron_index], feature_activations[neuron_index])
                f.write(html_str)


if __name__ == "__main__":
    main()
