import itertools
import os
import time

import datasets  # type: ignore[missingTypeStubs, import-untyped]
import torch
import transformer_lens  # type: ignore
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from transformer_lens import HookedTransformer  # type: ignore[import]

from thesis.device import Device
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

        samples_to_check = 32

        point = "resid_pre"
        layer_index = 3
        layer_id = f"blocks.{layer_index}.hook_resid_pre"
        # layer_id = f"blocks.{layer_index}.mlp.hook_post"

        sae_data = transformer_lens.utils.download_file_from_hf(
            "jacobcd52/gpt2-small-sparse-autoencoders",
            f"gpt2-small_6144_{point}_{layer_index}.pt",
            force_is_torch=True,
        )
        sae = SparseAutoencoder.from_data(sae_data, layer_id, device)

        num_sae_features = sae.num_features

        mas_store = MASStore(
            num_samples=samples_to_check,
            num_features=num_sae_features,
            context_size=context_size,
            sample_length_pre=192,
            sample_length_post=64,
            pad_token_id=model.tokenizer.pad_token_id,
            device=device,
        )

        activation_scratch = torch.full((context_size, num_sae_features), torch.nan, device=device.torch())
        sae_hook = sae.hook(activation_scratch)

        model_time = 0.0
        mas_time = 0.0
        start_time = time.time()
        for _i, sample in itertools.islice(enumerate(sample_dataset), 32):
            model_start_time = time.time()
            model.run_with_hooks(sample.tokens, fwd_hooks=[sae_hook])
            model_time += time.time() - model_start_time
            mas_start_time = time.time()
            mas_store.add_sample(sample, activation_scratch)
            mas_time += time.time() - mas_start_time
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")
        print(f"Time taken per sample: {(end_time - start_time) / (samples_to_check)*1000:.2f}s")

        print(f"Model time: {model_time:.2f}s ({model_time/(end_time - start_time)*100:.2f}%)")
        print(f"MAS time: {mas_time:.2f}s ({mas_time/(end_time - start_time)*100:.2f}%)")

        os.makedirs("outputs", exist_ok=True)

        mas_store.save("outputs/sae_mas_store.zip")


if __name__ == "__main__":
    main()
