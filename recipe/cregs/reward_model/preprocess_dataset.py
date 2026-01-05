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

import datasets

from verl.utils import dataset


# add a row to each data item that represents a unique id
def make_map_fn(split: str, data_source: str):
    def process_fn(example: dict[str, str|float], idx: int):
        gene_seq = example.pop("gene_seq")
        cre_seq = example.pop("cre_seq")
        
        data = {
            "data_source": data_source,
            "prompt": [{'role': 'user', 'content': gene_seq}],
            "question": gene_seq,
            "answer": cre_seq,
            "abc_score": example.pop('abc_score'),
            "extra_info": {
                "split": split,
                "index": idx,
                "activity": example.pop('activity'),
            },
        }
        return data

    return process_fn

if __name__ == "__main__":
    from os import cpu_count, getenv
    from pathlib import Path
    from random import sample
    from typing import Optional

    from tap import Tap

    class Args(Tap):
        dataset_dir: Path
        save_dir: Optional[Path] = None
        sample_ratio: float = 1.0
    args = Args().parse_args()
    if not args.save_dir:
        args.save_dir = Path(getenv('DATASETS', '')) / 'verl' / f'{args.dataset_dir.name}'
        print(f"No save_dir specified. Using default: {args.save_dir}")

    dataset = datasets.load_dataset(str(args.dataset_dir))

    for name, split in dataset.items():
        assert isinstance(name, str), f"Expected split name to be a string, but got {type(name)}"
        print(f"Before processing: {split[0]}")
        split_size = len(split)
        split = split.select(sample(range(split_size), k=int(split_size * args.sample_ratio)))
        split = split.map(
            make_map_fn(
                split=name,
                data_source=args.dataset_dir.name,
            ),
            remove_columns=split.column_names,
            with_indices=True,
            num_proc=max(1, (cpu_count() or 1)//2),
        )
        print(f"After processing: {split[0]}")
        args.save_dir.mkdir(parents=True, exist_ok=True)
        split.to_parquet(args.save_dir / f'{name}.parquet')
