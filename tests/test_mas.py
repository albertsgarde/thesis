import datasets  # type: ignore[missingTypeStubs, import-untyped]
import torch
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]

from thesis import hooks
from thesis.device import get_device  # type: ignore[import]
from thesis.mas import algorithm
from thesis.mas.algorithm import MASLayer, MASParams


def test_mas() -> None:
    params = MASParams(
        high_activation_weighting=4.0,
        sample_overlap=128,
        num_max_samples=16,
        sample_length_pre=96,
        sample_length_post=32,
        samples_to_check=16,
        seed=0,
        activation_bins=[x / 10 for x in range(40)],
    )

    device = get_device()

    model: HookedTransformer = HookedTransformer.from_pretrained("gelu-1l", device=device.torch())  # type: ignore[reportUnknownVariableType]

    dataset: IterableDataset = datasets.load_dataset(  # type: ignore[reportUnknownMemberType]
        "monology/pile-uncopyrighted", streaming=True, split="train", trust_remote_code=True
    )

    hook_point = "blocks.0.mlp.hook_post"
    layers = [MASLayer.from_hook_id(hook_point, 2048)]

    mas_store = algorithm.run(model, dataset, layers, params, device)

    mas_samples = mas_store.feature_samples()
    mas_activations = mas_store.feature_activations()

    assert mas_samples.shape == (2048, 16, 128)
    assert mas_activations.shape == (2048, 16, 128)

    assert mas_samples.isfinite().all()
    assert mas_activations.isfinite().all()

    generator = torch.Generator().manual_seed(5231)
    test_indices = torch.randint(0, 2048, (16,), generator=generator)

    for i in test_indices:
        activations = hooks.neuron_activations(model, hook_point, mas_samples[i, :, :], i, device)
        assert (activations.argmax(dim=1) == mas_activations[i].argmax(dim=1)).all()
