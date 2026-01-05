# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import logging

import datasets
from datasets import Dataset, NamedSplit


# add a row to each data item that represents a unique id
def make_map_fn(split: Dataset, data_source: str|NamedSplit):
    def process_fn(example, idx):
        gene_seq = example.pop('gene_seq')
        cre_seq = example.pop('cre_seq')

        data = {
            'data_source': data_source,
            'prompt': [{
                'role': 'user',
                'content': gene_seq
            }],
            'question': gene_seq,
            'answer': cre_seq,
            'extra_info': {
                'split': data_source,
                'index': idx,
                'cell_type': example.pop('cell_type'),
            },
        }
        return data

    return process_fn

def phase_dataset(dataset: datasets.Dataset):
    dataset = datasets.concatenate_datasets([dataset] * 36)
    def phasing(record: dict, idx: int):
        gene_phase = idx % 6
        cre_phase = (idx // 6) % 6
        record['question'] = record['question'][:-gene_phase]
        record['answer'] = record['answer'][:-cre_phase]
        return record
    dataset = dataset.map(phasing, with_indices=True)
    return dataset

if __name__ == "__main__":
    from os import getenv
    from pathlib import Path
    from random import sample
    from typing import Optional

    from tap import Tap

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class Args(Tap):
        dataset_dir: Path
        save_dir: Optional[Path] = None
        sample_ratio: float = 1.0
        phasing: bool = False
    args = Args().parse_args()
    if not args.save_dir:
        args.save_dir = Path(getenv('DATASETS', '')) / 'verl' / f'{args.dataset_dir.name}'
        print(f"No save_dir specified. Using default: {args.save_dir}")

    dataset = datasets.load_dataset(str(args.dataset_dir))
    assert isinstance(dataset, datasets.DatasetDict), f"Expected a DatasetDict but got {type(dataset)}:\n{dataset}"

    for name, split in dataset.items():
        logger.info(f"{name} dataset length: {len(split)}")

        if args.sample_ratio < 1.0:
            split_size = len(split)
            sampled_idxs = sample(range(split_size), k=int(split_size * args.sample_ratio))
            split = split.select(sampled_idxs)
            logger.info(f"After sampling with ratio {args.sample_ratio}, new length: {len(split)}")

        split = split.map(function=make_map_fn(split, name), with_indices=True)
        if args.phasing:
            split = phase_dataset(split)
        split.to_parquet(args.save_dir / f'{name}.parquet')
        logger.info(f"Saved processed {name} dataset to {args.save_dir / f'{name}.parquet'}")
