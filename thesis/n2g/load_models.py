import asyncio
import pickle
from pathlib import Path
from typing import Sequence

import aiofiles
from n2g import FeatureModel, NeuronModel, Tokenizer


async def read_model(models_path: Path, index: int) -> NeuronModel:
    async with aiofiles.open(models_path / f"{index}.pkl", "rb") as f:
        data = await f.read()
    return pickle.loads(data)


def load_batch(models_path: Path, batch_indices: Sequence[int]) -> list[NeuronModel]:
    return asyncio.get_event_loop().run_until_complete(
        asyncio.gather(*[read_model(models_path, index) for index in batch_indices])
    )


def load_models(
    models_path: Path, indices: Sequence[int], print_progress: bool = False, batch_size: int = 2048
) -> list[NeuronModel]:
    num_indices = len(indices)
    num_batches = num_indices // batch_size
    if num_indices % batch_size != 0:
        num_batches += 1

    result = []
    for batch in range(num_batches):
        if print_progress:
            print(f"Loading batch {batch + 1} of {num_batches}...")
        batch_indices = indices[batch * batch_size : (batch + 1) * batch_size]
        result += load_batch(models_path, batch_indices)
    return result


class N2GModelIter:
    buffer: list[NeuronModel]
    cur_buffer_index: int
    models_path: Path
    indicies: Sequence[int]
    next_batch_start: int
    batch_size: int

    def __init__(self, models_path: Path, indices: Sequence[int], batch_size: int = 2048) -> None:
        self.buffer = []
        self.cur_buffer_index = 0
        self.models_path = models_path
        self.indices = indices
        self.next_buffer_start = 0
        self.batch_size = batch_size

    def __iter__(self) -> "N2GModelIter":
        return self

    def __next__(self) -> NeuronModel:
        if self.cur_buffer_index == len(self.buffer):
            if self.next_buffer_start >= len(self.indices):
                raise StopIteration
            self.buffer = load_batch(
                self.models_path,
                self.indices[self.next_buffer_start : min(self.next_buffer_start + self.batch_size, len(self.indices))],
            )
            self.next_buffer_start += self.batch_size
            self.cur_buffer_index = 0
        assert self.cur_buffer_index < len(self.buffer)
        result = self.buffer[self.cur_buffer_index]
        self.cur_buffer_index += 1
        return result


def iter_models(models_path: Path, indices: Sequence[int], batch_size: int = 2048) -> N2GModelIter:
    return N2GModelIter(models_path, indices, batch_size)


def load_all_models(tokenizer: Tokenizer, models_path: Path) -> list[FeatureModel | None]:
    with models_path.open("rb") as f:
        data = f.read()
        return FeatureModel.list_from_bin(tokenizer, data)
