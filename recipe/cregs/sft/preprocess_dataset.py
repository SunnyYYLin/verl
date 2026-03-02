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
Preprocess the dataset to parquet format (no Tap dependency)
"""

import logging
import argparse
from pathlib import Path
from os import getenv
from random import sample
from typing import Optional

import datasets
from datasets import Dataset, NamedSplit


# add a row to each data item that represents a unique id
def make_map_fn(split: Dataset, data_source: str | NamedSplit):
    def process_fn(example, idx):
        distance = int(example.pop('cre_tss_distance'))
        gene_seq = example.pop('tss_seq_2kb')
        cre_seq = example.pop('cre_seq_500bp')
        cell_type = example.pop('CellType')

        # Add cell_type to the beginning of gene_seq
        gene_seq = f"{cell_type} {gene_seq}"

        # Add [SEP] every 10k distance
        sep_count = distance // 10000
        gene_seq += " [SEP]" * sep_count

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
                'cell_type': cell_type,
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

    dataset = dataset.map(
        phasing,
        with_indices=True,
        load_from_cache_file=False,
        keep_in_memory=True
    )
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset to verl-compatible parquet")

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to HuggingFace dataset directory"
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Directory to save parquet files"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Sampling ratio (0 < ratio <= 1.0)"
    )
    parser.add_argument(
        "--phasing",
        action="store_true",
        help="Enable gene/CRE phasing augmentation"
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()

    if args.save_dir is None:
        args.save_dir = Path(getenv('DATASETS', '')) / 'verl' / args.dataset_dir.name
        logger.info(f"No save_dir specified. Using default: {args.save_dir}")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(str(args.dataset_dir))
    assert isinstance(
        dataset, datasets.DatasetDict
    ), f"Expected DatasetDict but got {type(dataset)}:\n{dataset}"

    for name, split in dataset.items():
        logger.info(f"{name} dataset length: {len(split)}")

        if args.sample_ratio < 1.0:
            split_size = len(split)
            sampled_idxs = sample(
                range(split_size),
                k=int(split_size * args.sample_ratio)
            )
            split = split.select(sampled_idxs)
            logger.info(
                f"After sampling with ratio {args.sample_ratio}, new length: {len(split)}"
            )

        split = split.map(
            function=make_map_fn(split, name),
            with_indices=True,
            load_from_cache_file=False,
            keep_in_memory=True,
            desc=f"Processing {name}"
        )

        if args.phasing:
            split = phase_dataset(split)

        out_file = args.save_dir / f"{name}.parquet"
        split.to_parquet(out_file)
        logger.info(f"Saved processed {name} dataset to {out_file}")
