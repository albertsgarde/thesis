import datasets  # type: ignore[missingTypeStubs, import-untyped]
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]

from thesis.device import get_device  # type: ignore[import]
from thesis.mas import algorithm
from thesis.mas.algorithm import MASParams


def test_mas() -> None:
    params = MASParams(
        sample_overlap=128, num_max_samples=8, sample_length_pre=96, sample_length_post=32, samples_to_check=16
    )

    device = get_device()

    model: HookedTransformer = HookedTransformer.from_pretrained("solu-1l", device=device.torch())  # type: ignore[reportUnknownVariableType]

    dataset: IterableDataset = datasets.load_dataset(  # type: ignore[reportUnknownMemberType]
        "monology/pile-uncopyrighted", streaming=True, split="train", trust_remote_code=True
    )

    mas_store = algorithm.run(model, dataset, params, device)

    samples = mas_store.feature_samples()
    activations = mas_store.feature_activations()

    assert samples.shape == (2048, 8, 128)
    assert activations.shape == (2048, 8, 128)
