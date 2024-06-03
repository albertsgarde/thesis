import asyncio
import pickle
from pathlib import Path
from typing import Sequence

import aiofiles
from n2g import NeuronModel


async def read_model(models_path: Path, index: int) -> NeuronModel:
    async with aiofiles.open(models_path / f"{index}.pkl", "rb") as f:
        data = await f.read()
    return pickle.loads(data)


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
        result += asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*[read_model(models_path, index) for index in batch_indices])
        )
    return result
