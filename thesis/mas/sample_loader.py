from dataclasses import dataclass
from typing import Any, Generator, Optional

import torch.nn.functional as f
from datasets import IterableDataset  # type: ignore[missingTypeStubs, import-untyped]
from jaxtyping import Int
from torch import Tensor
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]


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
    """
    A sample from the dataset with a fixed size and possible overlap.

    Attributes:
        tokens (Int[Tensor, " context"]): The tokens of the sample.
        overlap (int): The number of tokens that overlap with the previous sample.
        length (int): The number of tokens in the sample before padding.
    """

    tokens: Int[Tensor, " context"]
    overlap: int
    length: int

    def __post_init__(self) -> None:
        assert self.length <= self.tokens.shape[-1]
        assert self.overlap <= self.length


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
            text = next(self.iterator)["text"]
            self.cur_sample = self.model.to_tokens(text, truncate=False)[0]
            self.cur_sample_index = 0
        else:
            self.cur_sample_index += self.context_size - self.overlap_size
        assert self.cur_sample is not None, "Sample should have been loaded by the preceding code."
        sample = self.cur_sample[self.cur_sample_index : self.cur_sample_index + self.context_size]
        if self.model.tokenizer is None:
            raise ValueError("Model must have tokenizer.")
        length = sample.shape[-1]
        sample = f.pad(
            sample, (0, self.context_size - length), mode="constant", value=self.model.tokenizer.pad_token_id
        )
        assert sample.shape[-1] == self.context_size, f"""Sample must be padded to context length.
         Sample length: {sample.shape[-1]}, context size: {self.context_size}"""

        overlap = 0 if self.cur_sample_index == 0 else self.overlap_size
        return Sample(sample, overlap, length)
