from typing import Optional
import nest_asyncio
import asyncio
import gc
import os
from transformers import T5TokenizerFast
import numpy as np
from torch.utils.data import TensorDataset


class DataCollator:

    def __init__(self, **kwargs):
        self.tokenizer: T5TokenizerFast = kwargs['tokenizer']
        self.input: np.ndarray = kwargs['input']
        self.targets: np.ndarray = kwargs['targets']
        self.datasets = []

    def __call__(self, n_batches: Optional[int]):
        nest_asyncio.apply()
        if n_batches:
            asyncio.run(self.main(n_batches))
        else:
            asyncio.run(self.main())

    async def main(self, n_batches: Optional[int]):

        tasks = []

        if n_batches:
            inps, tgts = await self.get_batches(n_batches)
        else:
            inps, tgts = await self.get_batches()

        for i in range(0, len(inps)):
            tasks.append(self.get_datasets(inps[i], tgts[i]))

        print(len(tasks))
        
        await asyncio.gather(*tasks)


    async def get_datasets(self, input, target):

        inputs = tokenizer.batch_encode_plus(
            input.tolist(),
            add_special_tokens = True,
            max_length=512, 
            padding=True,
            truncation=True,
            return_attention_mask = True,
            return_tensors="pt"
        )
        gc.collect()
        targets = tokenizer.batch_encode_plus(
            target.tolist(),
            add_special_tokens = True,
            max_length=128, 
            padding=True,
            truncation=True,
            return_attention_mask = False,
            return_tensors="pt"
        )
        gc.collect()

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_ids = targets['input_ids']

        self.datasets.extend(TensorDataset(input_ids, attention_mask, target_ids))
        print('dataset ok')

        gc.collect()


    async def get_batches(self, n_batches: Optional[int]=10): 
        inp_batches = []
        tgt_batches = []

        if len(self.input) % n_batches == 0:
            size = int(len(self.input) / n_batches)
            print(f'size = {size}')

            i=0
            for n in range(n_batches):
                inp_batches.append(self.input[i:i+size])
                tgt_batches.append(self.targets[i:i+size])
                i += size
        else:
            size = int(round(len(self.input) / n_batches))
            print(f'size = {size}')

            i=0
            for n in range(n_batches):
                try:
                    inp_batches.append(self.input[i:i+size])
                    tgt_batches.append(self.targets[i:i+size])
                    i += size
                except Exception:
                    inp_batches.append(self.input[i:])
                    tgt_batches.append(self.targets[i:])

        print(len(tgt_batches), len(inp_batches))
        return inp_batches, tgt_batches


