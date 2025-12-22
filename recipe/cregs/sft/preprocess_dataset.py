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
from verl.utils.hdfs_io import copy, makedirs

# add a row to each data item that represents a unique id
def make_map_fn(split):
    def process_fn(example, idx):
        gene_seq = example.pop("gene_seq")
        cre_seq = example.pop("cre_seq")
        
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": gene_seq,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": cre_seq},
            "extra_info": {
                "split": split,
                "index": idx,
                "gene_seq": gene_seq,
                "cre_seq": cre_seq,
                "cell_type": example.pop("cell_type"),
            },
        }
        return data

    return process_fn

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    from os import getenv

    parser = ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", type=Path, 
                        default=None, 
                        help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", type=Path, 
        default=Path(getenv("DATASETS", ""))/'verl'/'Gene-CRE', 
        help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "SunnyLin/Gene-CRE"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "default")
    else:
        # Force redownload / avoid using cached dataset files
        dataset = datasets.load_dataset(data_source, "default")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(local_save_dir / "train.parquet")
    val_dataset.to_parquet(local_save_dir / "val.parquet")
    test_dataset.to_parquet(local_save_dir / "test.parquet")
    

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
