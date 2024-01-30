from dataclasses import dataclass
from typing import Any, Generator, Optional

from datasets import IterableDataset
from jaxtyping import Int
from torch import Tensor
from transformer_lens.HookedTransformer import HookedTransformer


@dataclass
class SampleDataset:
    context_size: int
    overlap_size: int
    model: HookedTransformer
    dataset: IterableDataset

    def iter(self) -> "SampleIterator":
        return SampleIterator(self)

    def __iter__(self) -> "SampleIterator":
        return self.iter()


@dataclass
class Sample:
    tokens: Int[Tensor, " context"]
    start_of_context: bool


class SampleIterator:
    context_size: int
    overlap_size: int
    model: HookedTransformer
    iterator: Generator[dict[str, Any], Any, None]
    cur_sample: Optional[Int[Tensor, " sample_length"]]
    cur_sample_index: int

    def __init__(self, dataset: SampleDataset) -> None:
        self.context_size = dataset.context_size
        self.overlap_size = dataset.overlap_size
        if self.overlap_size > self.context_size:
            raise ValueError("Overlap size cannot be larger than context size")
        self.model = dataset.model
        self.iterator = dataset.dataset.iter(batch_size=1)
        self.cur_sample = None
        self.cur_sample_index = 0

    def __iter__(self) -> "SampleIterator":
        return self

    def __next__(self) -> Sample:
        if self.cur_sample is None or self.cur_sample_index + self.context_size >= self.cur_sample.shape[-1]:
            self.cur_sample = self.model.to_tokens(next(self.iterator)["text"], truncate=False)[0]
            self.cur_sample_index = 0
        else:
            self.cur_sample_index += self.context_size - self.overlap_size
        sample = self.cur_sample[self.cur_sample_index : self.cur_sample_index + self.context_size]
        return Sample(sample, self.cur_sample_index == 0)
